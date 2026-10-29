"""Kaynak → metin → chunk → embed → export orkestrasyonu.

İki aşama:
- process(): kaynağı okur ve parçalar; hiçbir yere yazmaz (önizleme).
- publish(): seçilen hedeflere (TXT / JSONL / Parquet / Qdrant) yazar; embedding yalnızca gerekiyorsa üretilir.
ingest() ikisini art arda çağırır (eski davranış).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

import config
import loaders
from core.types import Chunk, Document, OperationCancelled, QdrantPoint
from export.qdrant_writer import (
    build_points,
    upsert_to_qdrant,
    write_outputs,
    write_text_output,
)
from processing.chunker import chunk_text
from processing.embedder import get_embedder


Progress = Callable[[str, float | None], None]

OUTPUT_KINDS = ("txt", "jsonl", "parquet")
# Vektör içeren çıktılar: bunlar ya da Qdrant seçiliyse embedding gerekir
_VECTOR_KINDS = {"jsonl", "parquet"}


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

    @property
    def has_embeddings(self) -> bool:
        return len(self.embeddings) == len(self.chunks) > 0


def _noop(msg: str, pct: float | None = None) -> None:
    pass


# Yerelleştirilmiş mesajlar
_MSGS = {
    "tr": {
        "read": "Kaynak okunuyor: {source}",
        "chunk": "Anlamsal parçalama",
        "embed": "Embedding üretiliyor",
        "qdrant": "Qdrant point şemasına dönüştürülüyor",
        "write": "Dosyalar yazılıyor",
        "upload": "Qdrant'a yükleniyor",
        "done": "Tamam",
        "err_no_text": "Kaynaktan metin çıkarılamadı: {source}",
        "err_empty_chunks": "Chunker boş döndü",
        "err_needs_chunks": "Qdrant ve JSONL/Parquet çıktıları için parçalama gerekli",
        "char_lbl": "karakter",
        "chunk_lbl": "parça",
    },
    "en": {
        "read": "Reading source: {source}",
        "chunk": "Semantic chunking",
        "embed": "Generating embedding",
        "qdrant": "Converting to Qdrant point schema",
        "write": "Writing files",
        "upload": "Uploading to Qdrant",
        "done": "Completed",
        "err_no_text": "Could not extract text from source: {source}",
        "err_empty_chunks": "Chunker returned empty results",
        "err_needs_chunks": "Qdrant and JSONL/Parquet outputs require chunking",
        "char_lbl": "characters",
        "chunk_lbl": "chunks",
    },
}


def _msgs(system_language: str) -> dict[str, str]:
    return _MSGS.get(system_language) or _MSGS["tr"]


def _cancel_checker(check_cancelled: Callable[[], bool] | None) -> Callable[[], None]:
    def check_cancel() -> None:
        if check_cancelled and check_cancelled():
            raise OperationCancelled("İşlem kullanıcı tarafından iptal edildi.")
    return check_cancel


def _empty_result(document: Document, chunks: list[Chunk]) -> IngestResult:
    return IngestResult(
        document=document,
        chunks=chunks,
        embeddings=np.empty((0, config.EMBEDDING_DIM), dtype=np.float32),
        points=[],
    )


def _embed_texts(result: IngestResult) -> list[str]:
    """Embedding'e giden metin: QA ise soru+cevap; belge bir bağlam (ör. kayıt başlığı) taşıyorsa önüne eklenir."""
    context = result.document.extra.get("context")
    texts = []
    for c in result.chunks:
        text = f"Soru: {c.question}\nCevap: {c.text}" if c.question else c.text
        texts.append(f"{context}\n{text}" if context else text)
    return texts


def process(
    source: str,
    progress: Progress | None = None,
    whisper_language: str | None = None,
    system_language: str = "tr",
    similarity_threshold: float | None = None,
    min_tokens: int | None = None,
    max_tokens: int | None = None,
    qa_mode: bool = False,
    chunk: bool = True,
    whole: bool = False,
    check_cancelled: Callable[[], bool] | None = None,
) -> IngestResult:
    """Kaynağı okur ve (istenirse) parçalar. Hiçbir yere yazmaz, embedding üretmez.

    whole: parçalamadan metnin tamamı tek parça olur.
    """
    progress = progress or _noop
    msgs = _msgs(system_language)
    check_cancel = _cancel_checker(check_cancelled)

    check_cancel()
    progress(msgs["read"].format(source=source), 0.05)
    document = loaders.load(
        source,
        whisper_language=whisper_language,
        check_cancelled=check_cancelled,
    )

    check_cancel()
    if not document.text or not document.text.strip():
        raise RuntimeError(msgs["err_no_text"].format(source=source))

    chunks: list[Chunk] = []
    if chunk:
        progress(f"{msgs['chunk']} ({len(document.text):,} {msgs['char_lbl']})", 0.35)
        chunks = chunk_text(
            document.text,
            similarity_threshold=similarity_threshold,
            min_tokens=min_tokens,
            max_tokens=max_tokens,
            qa_mode=qa_mode,
            whole=whole,
        )
        if not chunks:
            raise RuntimeError(msgs["err_empty_chunks"])

    check_cancel()
    return _empty_result(document, chunks)


def publish(
    result: IngestResult,
    qdrant: bool = False,
    outputs: Iterable[str] = (),
    out_dir: Path | None = None,
    progress: Progress | None = None,
    system_language: str = "tr",
    check_cancelled: Callable[[], bool] | None = None,
) -> IngestResult:
    """`process()` sonucunu seçilen hedeflere yazar; sonucu yerinde günceller ve döndürür.

    Embedding yalnızca Qdrant ya da JSONL/Parquet istenirse ve daha önce üretilmemişse hesaplanır;
    böylece önizlemeden sonra kaydetmek kaynağı yeniden işlemez.
    """
    progress = progress or _noop
    msgs = _msgs(system_language)
    check_cancel = _cancel_checker(check_cancelled)

    outputs = [k for k in OUTPUT_KINDS if k in set(outputs)]
    needs_vectors = qdrant or any(k in _VECTOR_KINDS for k in outputs)

    if needs_vectors and not result.chunks:
        raise RuntimeError(msgs["err_needs_chunks"])

    if needs_vectors and not result.has_embeddings:
        check_cancel()
        progress(f"{msgs['embed']} ({len(result.chunks)} {msgs['chunk_lbl']})", 0.65)
        result.embeddings = get_embedder().encode(_embed_texts(result))

    if needs_vectors and not result.points:
        check_cancel()
        progress(msgs["qdrant"], 0.85)
        result.points = build_points(result.document, result.chunks, result.embeddings)

    if outputs:
        check_cancel()
        progress(msgs["write"], 0.90)
        if "txt" in outputs:
            result.output_files["txt"] = write_text_output(result.document, out_dir=out_dir)
        vector_kinds = [k for k in outputs if k in _VECTOR_KINDS]
        if vector_kinds:
            result.output_files.update(
                write_outputs(result.document, result.points, out_dir=out_dir, kinds=vector_kinds)
            )

    if qdrant:
        check_cancel()
        progress(msgs["upload"], 0.95)
        result.qdrant_result = upsert_to_qdrant(result.points)

    progress(msgs["done"], 1.0)
    return result


def ingest(
    source: str,
    out_dir: Path | None = None,
    progress: Progress | None = None,
    whisper_language: str | None = None,
    system_language: str = "tr",
    similarity_threshold: float | None = None,
    min_tokens: int | None = None,
    max_tokens: int | None = None,
    qa_mode: bool = False,
    text_only: bool = False,
    check_cancelled: Callable[[], bool] | None = None,
    qdrant: bool = True,
    outputs: Iterable[str] = ("jsonl", "parquet"),
) -> IngestResult:
    """Tek bir kaynağı uçtan uca işle: process() + publish().

    text_only: parçalama/embedding/Qdrant yapılmaz, yalnızca metin TXT olarak kaydedilir.
    """
    result = process(
        source,
        progress=progress,
        whisper_language=whisper_language,
        system_language=system_language,
        similarity_threshold=similarity_threshold,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
        qa_mode=qa_mode,
        chunk=not text_only,
        check_cancelled=check_cancelled,
    )
    if text_only:
        qdrant, outputs = False, ("txt",)
    return publish(
        result,
        qdrant=qdrant,
        outputs=outputs,
        out_dir=out_dir,
        progress=progress,
        system_language=system_language,
        check_cancelled=check_cancelled,
    )


# ---------------------------------------------------------------------------
# Toplu işleme: çok belgeli kaynaklar (ör. veritabanı tablosunun satırları)
# ---------------------------------------------------------------------------

@dataclass
class BatchResult:
    """Bir kaynaktan gelen çok sayıda belge; her belge ayrı parçalanır, parçalar asla belge sınırını aşmaz."""
    label: str
    source_uri: str
    results: list[IngestResult] = field(default_factory=list)
    skipped: list = field(default_factory=list)  # metni boş olduğu için atlanan kayıtlar
    output_files: dict[str, Path] = field(default_factory=dict)
    qdrant_result: dict = field(default_factory=dict)

    @property
    def chunk_count(self) -> int:
        return sum(r.chunk_count for r in self.results)

    @property
    def avg_chunk_chars(self) -> float:
        chunks = [c for r in self.results for c in r.chunks]
        return sum(len(c.text) for c in chunks) / len(chunks) if chunks else 0.0

    def as_document(self) -> Document:
        """Toplu dosya çıktıları için tek belge görünümü."""
        return Document(
            text="\n\n".join(f"### {r.document.title}\n\n{r.document.text}" for r in self.results),
            source_type="db",
            source_uri=self.source_uri,
            title=self.label,
        )


def process_documents(
    documents: Iterable[Document],
    label: str,
    source_uri: str,
    total: int | None = None,
    progress: Progress | None = None,
    system_language: str = "tr",
    similarity_threshold: float | None = None,
    min_tokens: int | None = None,
    max_tokens: int | None = None,
    qa_mode: bool = False,
    chunk: bool = True,
    whole: bool = False,
    check_cancelled: Callable[[], bool] | None = None,
) -> BatchResult:
    """Belgeleri tek tek parçalar. Hiçbir yere yazmaz, embedding üretmez."""
    progress = progress or _noop
    msgs = _msgs(system_language)
    check_cancel = _cancel_checker(check_cancelled)

    batch = BatchResult(label=label, source_uri=source_uri)
    for i, document in enumerate(documents, start=1):
        check_cancel()
        pct = 0.05 + 0.55 * i / total if total else None
        progress(f"{msgs['chunk']}: {i}{f'/{total}' if total else ''} · {document.title}", pct)
        chunks: list[Chunk] = []
        if chunk:
            chunks = chunk_text(
                document.text,
                similarity_threshold=similarity_threshold,
                min_tokens=min_tokens,
                max_tokens=max_tokens,
                qa_mode=qa_mode,
                whole=whole,
            )
        batch.results.append(_empty_result(document, chunks))
    return batch


def publish_batch(
    batch: BatchResult,
    qdrant: bool = False,
    outputs: Iterable[str] = (),
    out_dir: Path | None = None,
    progress: Progress | None = None,
    system_language: str = "tr",
    check_cancelled: Callable[[], bool] | None = None,
) -> BatchResult:
    """`publish()`'in toplu hali: bütün belgelerin parçaları tek seferde embed edilir; dosyalar tek dosyada toplanır."""
    progress = progress or _noop
    msgs = _msgs(system_language)
    check_cancel = _cancel_checker(check_cancelled)

    outputs = [k for k in OUTPUT_KINDS if k in set(outputs)]
    needs_vectors = qdrant or any(k in _VECTOR_KINDS for k in outputs)

    if needs_vectors and not batch.chunk_count:
        raise RuntimeError(msgs["err_needs_chunks"])

    if needs_vectors:
        pending = [r for r in batch.results if r.chunks and not r.has_embeddings]
        if pending:
            check_cancel()
            texts = [t for r in pending for t in _embed_texts(r)]
            progress(f"{msgs['embed']} ({len(texts)} {msgs['chunk_lbl']})", 0.65)
            vectors = get_embedder().encode(texts)
            offset = 0
            for r in pending:
                r.embeddings = vectors[offset:offset + r.chunk_count]
                offset += r.chunk_count

        check_cancel()
        progress(msgs["qdrant"], 0.85)
        for r in batch.results:
            if r.chunks and not r.points:
                r.points = build_points(r.document, r.chunks, r.embeddings)

    points = [p for r in batch.results for p in r.points]

    if outputs:
        check_cancel()
        progress(msgs["write"], 0.90)
        document = batch.as_document()
        if "txt" in outputs:
            batch.output_files["txt"] = write_text_output(document, out_dir=out_dir)
        vector_kinds = [k for k in outputs if k in _VECTOR_KINDS]
        if vector_kinds:
            batch.output_files.update(write_outputs(document, points, out_dir=out_dir, kinds=vector_kinds))

    if qdrant:
        check_cancel()
        progress(msgs["upload"], 0.95)
        batch.qdrant_result = upsert_to_qdrant(points)

    progress(msgs["done"], 1.0)
    return batch
