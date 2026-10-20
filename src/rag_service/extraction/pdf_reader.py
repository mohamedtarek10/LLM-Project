"""PDF reader with optional OCR for image-only pages.

Requires the ``ingest`` extra (``pip install -e ".[ingest]"``).
"""

from pathlib import Path

import numpy as np
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document


class PyMuPDFImageReader(BaseReader):
    """Read PDF files using PyMuPDF, optionally OCR-ing embedded images."""

    def __init__(self, extract_images: bool = False, ocr_lang: str = "ara+eng"):
        self.extract_images = extract_images
        self.ocr_lang = ocr_lang

    def _page_images(self, doc, page) -> list[np.ndarray]:
        """Return every embedded image on the page as an RGB array."""
        import fitz

        images: list[np.ndarray] = []
        for img in page.get_images(full=True):
            xref = img[0]
            try:
                pix = fitz.Pixmap(doc, xref)
                # Normalise everything (grayscale, CMYK, alpha) to 3-channel RGB
                # so the reshape below is always valid.
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

        chunks = []
        for img in images:
            text = pytesseract.image_to_string(img, lang=self.ocr_lang).strip()
            if text:
                chunks.append(text)
        return "\n".join(chunks)

    def load_data(
        self,
        file_path: Path | str,
        metadata: bool = True,
        extra_info: dict | None = None,
        fs=None,
    ) -> list[Document]:
        import fitz

        if fs is not None and str(file_path).startswith("s3://"):
            with fs.open(file_path, "rb") as handle:
                doc = fitz.open("pdf", handle.read())
        else:
            doc = fitz.open(file_path)

        extra_info = dict(extra_info or {})
        if metadata:
            extra_info["total_pages"] = len(doc)
            extra_info["file_path"] = str(file_path)
            extra_info["file_name"] = Path(str(file_path)).name

        documents = []
        for page in doc:
            text = page.get_text()
            if self.extract_images:
                image_text = self._ocr(self._page_images(doc, page))
                if image_text:
                    text = f"{text}\n{image_text}"
            documents.append(
                Document(
                    text=text,
                    extra_info={
                        **extra_info,
                        "page": page.number + 1,
                        "source": f"Page {page.number + 1}",
                    },
                )
            )
        return documents

    # BaseReader in some llama-index versions calls .load()
    load = load_data
