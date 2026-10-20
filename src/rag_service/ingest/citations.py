"""Turn chunk metadata into a human-readable citation.

The API contract says ``sources`` are *citations*, never chunk ids -- so this is
the only place allowed to render them.
"""

from typing import Any


def format_citation(metadata: dict[str, Any] | None) -> str:
    """Render one retrieved chunk's provenance.

    Prefers a structured article reference; falls back to file/page for corpora
    that have not been article-segmented yet.
    """
    meta = metadata or {}

    article = meta.get("article_number")
    if article is not None:
        label = f"Article {article}"
        if meta.get("repealed"):
            label += " (repealed)"
        section = meta.get("section") or meta.get("chapter")
        if section:
            label += f" — {section}"
        return label

    name = meta.get("file_name") or meta.get("source") or "unknown source"
    page = meta.get("page")
    return f"{name}, p. {page}" if page else str(name)


def unique_citations(documents) -> list[str]:
    """Citations for the retrieved documents, de-duplicated, order preserved."""
    seen: dict[str, None] = {}
    for doc in documents or []:
        seen.setdefault(format_citation(getattr(doc, "metadata", None)), None)
    return list(seen)
