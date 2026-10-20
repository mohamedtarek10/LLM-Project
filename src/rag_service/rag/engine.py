"""The RAG engine: one object holding the loaded index, the LLM and the chain.

Built once per process and reused, so the API does not reload FAISS per request.
"""

from dataclasses import dataclass

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory

from rag_service.config import Settings, get_settings
from rag_service.ingest.citations import format_citation, unique_citations
from rag_service.ingest.pipeline import count_documents, load_index
from rag_service.rag.prompts import DOCUMENT_PROMPT, answer_prompt, contextualize_prompt

MAX_HISTORY_MESSAGES = 4


@dataclass
class Answer:
    answer: str
    sources: list[str]


class RagEngine:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.vectorstore = load_index(self.settings)
        self.llm = self._build_llm()
        self._histories: dict[str, BaseChatMessageHistory] = {}
        self._chain = self._build_chain()

    # --- wiring ----------------------------------------------------------- #
    def _build_llm(self):
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=self.settings.llm_model,
            temperature=self.settings.llm_temperature,
            openai_api_key=self.settings.openai_api_key,
        )

    def _build_chain(self):
        retriever = self.vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": self.settings.retriever_k,
                "score_threshold": self.settings.score_threshold,
            },
        )

        def annotate(documents):
            """Attach the rendered citation so the prompt can quote it verbatim."""
            for doc in documents:
                doc.metadata = dict(doc.metadata or {})
                doc.metadata["citation"] = format_citation(doc.metadata)
            return documents

        history_aware = create_history_aware_retriever(
            self.llm, retriever, contextualize_prompt()
        ) | RunnableLambda(annotate)

        combine = create_stuff_documents_chain(
            llm=self.llm,
            prompt=answer_prompt(),
            document_prompt=DOCUMENT_PROMPT,
            document_variable_name="context",
        )

        return RunnableWithMessageHistory(
            create_retrieval_chain(history_aware, combine),
            self._get_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

    def _get_history(self, session_id: str) -> BaseChatMessageHistory:
        history = self._histories.setdefault(session_id, ChatMessageHistory())
        if len(history.messages) > MAX_HISTORY_MESSAGES:
            history.messages = history.messages[-MAX_HISTORY_MESSAGES:]
        return history

    # --- public API -------------------------------------------------------- #
    @property
    def documents_indexed(self) -> int:
        return count_documents(self.vectorstore)

    def ask(self, question: str, session_id: str = "default") -> Answer:
        result = self._chain.invoke(
            {"input": question},
            config={"configurable": {"session_id": session_id}},
        )
        return Answer(
            answer=result.get("answer", ""),
            sources=unique_citations(result.get("context")),
        )

    def reset_session(self, session_id: str) -> None:
        self._histories.pop(session_id, None)


_engine: RagEngine | None = None


def get_engine(settings: Settings | None = None) -> RagEngine:
    """Process-wide singleton."""
    global _engine
    if _engine is None:
        _engine = RagEngine(settings)
    return _engine
