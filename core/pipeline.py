"""Kaynak → metin → chunk → embed → export orkestrasyonu."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

import loaders
from core.types import Chunk, Document, QdrantPoint
from export.qdrant_writer import build_points, write_outputs
from processing.chunker import chunk_text
from processing.embedder import get_embedder


Progress = Callable[[str, float | None], None]


@dataclass
class IngestResult:
    document: Document
    chunks: list[Chunk]
    embeddings: np.ndarray
    points: list[QdrantPoint]
    output_files: dict[str, Path] = field(default_factory=dict)

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)

    @property
    def avg_chunk_chars(self) -> float:
        if not self.chunks:
            return 0.0
        return sum(len(c.text) for c in self.chunks) / len(self.chunks)


def _noop(msg: str, pct: float | None = None) -> None:
    pass


def ingest(
    source: str,
    out_dir: Path | None = None,
    progress: Progress | None = None,
) -> IngestResult:
    """Tek bir kaynağı uçtan uca işle."""
    progress = progress or _noop

    progress(f"Kaynak okunuyor: {source}", 0.05)
    document = loaders.load(source)

    if not document.text or not document.text.strip():
        raise RuntimeError(f"Kaynaktan metin çıkarılamadı: {source}")

    progress(f"Anlamsal parçalama ({len(document.text):,} karakter)", 0.35)
    chunks = chunk_text(document.text)
    if not chunks:
        raise RuntimeError("Chunker boş döndü")

    progress(f"Embedding üretiliyor ({len(chunks)} parça)", 0.65)
    embedder = get_embedder()
    vectors = embedder.encode([c.text for c in chunks])

    progress("Qdrant point şemasına dönüştürülüyor", 0.85)
    points = build_points(document, chunks, vectors)

    progress("Dosyalar yazılıyor", 0.95)
    files = write_outputs(document, points, out_dir=out_dir)

    progress("Tamam", 1.0)
    return IngestResult(
        document=document,
        chunks=chunks,
        embeddings=vectors,
        points=points,
        output_files=files,
    )
