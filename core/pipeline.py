"""Kaynak → metin → chunk → embed → export orkestrasyonu."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

import loaders
from core.types import Chunk, Document, QdrantPoint
from export.qdrant_writer import build_points, upsert_to_qdrant, write_outputs
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
    qdrant_result: dict = field(default_factory=dict)

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
    whisper_language: str | None = None,
    system_language: str = "tr",
    similarity_threshold: float | None = None,
    min_tokens: int | None = None,
    max_tokens: int | None = None,
) -> IngestResult:
    """Tek bir kaynağı uçtan uca işle."""
    progress = progress or _noop

    # Yerelleştirilmiş mesajlar
    msgs = {
        "tr": {
            "read": f"Kaynak okunuyor: {source}",
            "chunk": "Anlamsal parçalama",
            "embed": "Embedding üretiliyor",
            "qdrant": "Qdrant point şemasına dönüştürülüyor",
            "write": "Dosyalar yazılıyor",
            "upload": "Qdrant'a yükleniyor",
            "done": "Tamam",
            "err_no_text": f"Kaynaktan metin çıkarılamadı: {source}",
            "err_empty_chunks": "Chunker boş döndü",
            "char_lbl": "karakter",
            "chunk_lbl": "parça"
        },
        "en": {
            "read": f"Reading source: {source}",
            "chunk": "Semantic chunking",
            "embed": "Generating embedding",
            "qdrant": "Converting to Qdrant point schema",
            "write": "Writing files",
            "upload": "Uploading to Qdrant",
            "done": "Completed",
            "err_no_text": f"Could not extract text from source: {source}",
            "err_empty_chunks": "Chunker returned empty results",
            "char_lbl": "characters",
            "chunk_lbl": "chunks"
        }
    }.get(system_language, "tr")

    progress(msgs["read"], 0.05)
    document = loaders.load(source, whisper_language=whisper_language)

    if not document.text or not document.text.strip():
        raise RuntimeError(msgs["err_no_text"])

    progress(f"{msgs['chunk']} ({len(document.text):,} {msgs['char_lbl']})", 0.35)
    chunks = chunk_text(
        document.text,
        similarity_threshold=similarity_threshold,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
    )
    if not chunks:
        raise RuntimeError(msgs["err_empty_chunks"])

    progress(f"{msgs['embed']} ({len(chunks)} {msgs['chunk_lbl']})", 0.65)
    embedder = get_embedder()
    vectors = embedder.encode([c.text for c in chunks])

    progress(msgs["qdrant"], 0.85)
    points = build_points(document, chunks, vectors)

    progress(msgs["write"], 0.90)
    files = write_outputs(document, points, out_dir=out_dir)

    # Qdrant'a otomatik yükle
    progress(msgs["upload"], 0.95)
    qdrant_res = upsert_to_qdrant(points)

    progress(msgs["done"], 1.0)
    return IngestResult(
        document=document,
        chunks=chunks,
        embeddings=vectors,
        points=points,
        output_files=files,
        qdrant_result=qdrant_res,
    )
