"""PDF text extraction, with optional OCR for image-only pages.

Requires the ``ingest`` extra (``pip install -e ".[ingest]"``) for PyMuPDF;
Tesseract is only needed when ``extract_images`` is on.
"""

from pathlib import Path

import numpy as np
from langchain.schema import Document


class PyMuPDFImageReader:
    """Read a PDF into one document per page, optionally OCR-ing images."""

    def __init__(self, extract_images: bool = False, ocr_lang: str = "ara+eng"):
        self.extract_images = extract_images
        self.ocr_lang = ocr_lang

    def _page_images(self, doc, page) -> list[np.ndarray]:
        """Every embedded image on the page, normalised to 3-channel RGB."""
        import fitz

        images: list[np.ndarray] = []
        for img in page.get_images(full=True):
            try:
                pix = fitz.Pixmap(doc, img[0])
                if pix.n != 3 or pix.alpha:
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                images.append(
                    np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
                )
            except Exception as exc:  # noqa: BLE001 - a bad image must not kill ingestion
                print(f"Skipping image on page {page.number + 1}: {exc}")
        return images

    def _ocr(self, images: list[np.ndarray]) -> str:
        if not images:
            return ""
        import pytesseract

        texts = [pytesseract.image_to_string(img, lang=self.ocr_lang).strip() for img in images]
        return "\n".join(text for text in texts if text)

    def read_pages(self, file_path: Path | str) -> list[str]:
        """Plain page texts, in order -- what article segmentation consumes."""
        import fitz

        doc = fitz.open(str(file_path))
        pages: list[str] = []
        for page in doc:
            text = page.get_text()
            if self.extract_images:
                image_text = self._ocr(self._page_images(doc, page))
                if image_text:
                    text = f"{text}\n{image_text}"
            pages.append(text)
        return pages

    def load_data(self, file_path: Path | str, extra_info: dict | None = None) -> list[Document]:
        """One LangChain document per page, with provenance metadata."""
        path = Path(str(file_path))
        pages = self.read_pages(path)

        base = dict(extra_info or {})
        base.update({"file_path": str(path), "file_name": path.name, "total_pages": len(pages)})

        return [
            Document(
                page_content=text,
                metadata={**base, "page": number, "source": f"Page {number}"},
            )
            for number, text in enumerate(pages, start=1)
        ]
