"""Streamlit front end. Thin UI over the same engine the API uses."""

import uuid

import streamlit as st

from rag_service.config import get_settings
from rag_service.ingest.pipeline import add_url
from rag_service.rag import RagEngine

st.set_page_config(page_title="RAG Knowledge Assistant", page_icon="📚", layout="wide")


@st.cache_resource
def get_engine() -> RagEngine:
    return RagEngine()


def main() -> None:
    st.session_state.setdefault("session_id", str(uuid.uuid4()))
    st.session_state.setdefault("messages", [])

    st.title("Knowledge Assistant")
    st.caption("Answers are drawn from the indexed corpus and cite their sources.")

    with st.spinner("Initializing system..."):
        engine = get_engine()

    with st.sidebar:
        st.metric("Documents indexed", engine.documents_indexed)

        st.header("Add knowledge")
        url_input = st.text_input("URL to scrape")
        if st.button("Add URL") and url_input:
            with st.spinner("Scraping and indexing..."):
                try:
                    added = add_url(url_input, engine.vectorstore, get_settings())
                    st.success(f"Added {added} chunks from {url_input}")
                    st.cache_resource.clear()
                    st.rerun()
                except Exception as exc:  # noqa: BLE001 - surfaced to the user
                    st.error(f"Failed: {exc}")

        if st.button("Start new session", use_container_width=True):
            engine.reset_session(st.session_state.session_id)
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.messages = []
            st.rerun()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander("Sources"):
                    for source in message["sources"]:
                        st.markdown(f"- {source}")

    question = st.chat_input("Ask a question...")
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"), st.spinner("Thinking..."):
        try:
            result = engine.ask(question, st.session_state.session_id)
            answer, sources = result.answer, result.sources
        except Exception as exc:  # noqa: BLE001 - surfaced to the user
            answer, sources = f"Something went wrong: {exc}", []

        st.markdown(answer)
        if sources:
            with st.expander("Sources"):
                for source in sources:
                    st.markdown(f"- {source}")

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})


if __name__ == "__main__":
    main()
