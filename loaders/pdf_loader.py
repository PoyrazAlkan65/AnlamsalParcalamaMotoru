"""PDF loader: PyMuPDF ile sayfa bazlı metin çıkarımı."""
from __future__ import annotations

from pathlib import Path

from core.types import Document, Segment

from .base import Loader


class PDFLoader(Loader):
    source_type = "pdf"

    def load(self, source: str, whisper_language: str | None = None) -> Document:
        import fitz  # pymupdf

        path = Path(source)
        doc = fitz.open(path)

        segments: list[Segment] = []
        parts: list[str] = []

        for page_idx in range(len(doc)):
            page = doc.load_page(page_idx)
            page_text = page.get_text("text") or ""
            page_text = page_text.strip()
            if not page_text:
                continue
            segments.append(Segment(
                text=page_text,
                page=page_idx + 1,
            ))
            parts.append(page_text)

        full_text = "\n\n".join(parts)

        meta = doc.metadata or {}
        title = meta.get("title") or path.stem
        doc.close()

        return Document(
            text=full_text,
            source_type=self.source_type,
            source_uri=path.resolve().as_uri(),
            title=title,
            language=None,
            segments=segments,
            extra={
                "page_count": len(segments),
                "author": meta.get("author"),
                "byte_size": path.stat().st_size if path.exists() else 0,
            },
        )
