"""Prompt templates for answering and for history-aware query rewriting."""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate

ANSWER_SYSTEM_PROMPT = (
    "You are a precise assistant that answers questions strictly from the "
    "provided context.\n\n"
    "Instructions:\n"
    "1. Answer using the retrieved context only. Do not use outside knowledge.\n"
    "2. If the context does not contain the answer, say you could not find it "
    "in the documents. Do not guess.\n"
    "3. Cite the source shown after each context passage, using the exact "
    "citation text given.\n"
    "4. If a cited source is marked repealed, say so explicitly.\n\n"
    "Context:\n{context}"
)

CONTEXTUALIZE_SYSTEM_PROMPT = (
    "Given a chat history and the latest user question, which might reference "
    "context in the chat history, formulate a standalone question that can be "
    "understood without the chat history. Do NOT answer the question; just "
    "reformulate it if needed, and otherwise return it as is."
)

# Rendered once per retrieved chunk. `citation` is attached to every document's
# metadata before it reaches the LLM, so this template never hits a missing key.
DOCUMENT_PROMPT = PromptTemplate(
    template="{page_content}\n(Source: {citation})",
    input_variables=["page_content", "citation"],
)


def answer_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            ("system", ANSWER_SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )


def contextualize_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            ("system", CONTEXTUALIZE_SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
