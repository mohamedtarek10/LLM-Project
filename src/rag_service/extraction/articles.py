"""Segment a legal document into one record per article.

The PDF is a stream of pages; what the corpus needs is one record per article,
with the article number as an integer so records can be sorted, cited and
compared. This module does that segmentation and normalisation.

Supports Arabic ("مادة 123", "المادة ١٢٣") and English ("Article 123")
headings, and Arabic-Indic digits.
"""

import re
import unicodedata
from dataclasses import dataclass, field

# Arabic-Indic (٠-٩) and Extended Arabic-Indic (۰-۹) digits -> ASCII
_DIGIT_MAP = {
    **{ord("٠") + i: str(i) for i in range(10)},
    **{ord("۰") + i: str(i) for i in range(10)},
}

# "Article 12", "مادة 12", "المادة ١٢", optionally followed by bis/مكرر
ARTICLE_HEADING = re.compile(
    r"^[\s\(\[]*"
    r"(?:(?:ال)?مادة|Article|ARTICLE|Art\.)"
    r"[\s:\-–]*"
    r"(?P<number>[0-9\u0660-\u0669\u06F0-\u06F9]{1,4})"
    r"(?P<suffix>\s*(?:مكرر|bis|ter)\s*)?"
    r"[\s:\.\-–\)\]]*",
    re.MULTILINE,
)

# A repeal marker inside the article body.
REPEALED = re.compile(
    r"(ملغا(?:ة|ه)|ملغي(?:ة)?|أُلغيت|الغيت|\brepealed\b|\bdeleted\b|\bcancelled\b)",
    re.IGNORECASE,
)


def normalize_digits(text: str) -> str:
    """Map Arabic-Indic digits onto ASCII digits."""
    return text.translate(_DIGIT_MAP)


def parse_article_number(raw: str) -> int:
    """Normalise an article number to an integer.

    Raises ValueError if it is not a number, so a bad heading fails loudly
    rather than silently entering the corpus as a string.
    """
    value = normalize_digits(raw).strip()
    if not value.isdigit():
        raise ValueError(f"Article number is not numeric: {raw!r}")
    return int(value)


def normalize_text(text: str) -> str:
    """Tidy whitespace without touching the Arabic letters themselves."""
    # NFKC folds presentation forms onto canonical Arabic characters.
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


@dataclass
class Article:
    article_number: int
    text: str
    repealed: bool = False
    suffix: str | None = None
    section: str | None = None
    source_pages: list[int] = field(default_factory=list)

    def to_record(self) -> dict:
        record = {
            "article_number": self.article_number,
            "text": self.text,
            "repealed": self.repealed,
        }
        if self.suffix:
            record["suffix"] = self.suffix
        if self.section:
            record["section"] = self.section
        if self.source_pages:
            record["source_pages"] = self.source_pages
        return record


def segment_articles(pages: list[str]) -> list[Article]:
    """Split page texts into articles.

    Pages are joined with page markers so each article can report the pages it
    came from, then split on article headings.
    """
    marked: list[tuple[int, int]] = []  # (offset in joined text, page number)
    parts: list[str] = []
    offset = 0
    for page_number, page_text in enumerate(pages, start=1):
        marked.append((offset, page_number))
        parts.append(page_text)
        offset += len(page_text) + 1
    joined = normalize_text("\n".join(parts))

    def page_of(position: int) -> int:
        page = 1
        for start, number in marked:
            if start <= position:
                page = number
            else:
                break
        return page

    matches = list(ARTICLE_HEADING.finditer(joined))
    articles: list[Article] = []

    for index, match in enumerate(matches):
        try:
            number = parse_article_number(match.group("number"))
        except ValueError:
            continue

        body_start = match.end()
        body_end = matches[index + 1].start() if index + 1 < len(matches) else len(joined)
        body = joined[body_start:body_end].strip()
        if not body:
            continue

        suffix = (match.group("suffix") or "").strip() or None
        # body_end is the *next* heading's start, so step back inside this body.
        start_page = page_of(match.start())
        end_page = page_of(max(body_start, body_end - 1))
        pages_spanned = sorted({start_page, end_page})

        articles.append(
            Article(
                article_number=number,
                text=body,
                repealed=bool(REPEALED.search(body)),
                suffix=suffix,
                source_pages=pages_spanned,
            )
        )

    return articles


def articles_to_records(articles: list[Article]) -> list[dict]:
    """Sorted, de-duplicated records ready to serialise as the corpus JSON."""
    seen: dict[tuple[int, str | None], Article] = {}
    for article in articles:
        key = (article.article_number, article.suffix)
        # A repeated heading usually means a cross-reference; keep the longest body.
        if key not in seen or len(article.text) > len(seen[key].text):
            seen[key] = article
    ordered = sorted(seen.values(), key=lambda a: (a.article_number, a.suffix or ""))
    return [article.to_record() for article in ordered]
