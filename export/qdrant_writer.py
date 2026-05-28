"""Qdrant point şemasında JSONL + Parquet yazıcı."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np

import config
from core.types import Chunk, Document, QdrantPoint
from core.utils import safe_filename, stable_chunk_id


def build_points(
    document: Document,
    chunks: list[Chunk],
    embeddings: np.ndarray,
) -> list[QdrantPoint]:
    if len(chunks) != len(embeddings):
        raise ValueError(
            f"chunk sayısı ({len(chunks)}) embedding sayısı ({len(embeddings)}) ile eşleşmiyor"
        )

    ingested = datetime.now(timezone.utc).isoformat()
    points: list[QdrantPoint] = []

    seg_lookup = _build_segment_lookup(document)

    for chunk, vec in zip(chunks, embeddings):
        parent = _find_parent_segment(chunk, seg_lookup, document)
        payload = {
            "text": chunk.text,
            "source_type": document.source_type,
            "source_uri": document.source_uri,
            "title": document.title,
            "language": document.language,
            "chunk_index": chunk.chunk_index,
            "char_start": chunk.char_start,
            "char_end": chunk.char_end,
            "token_count": chunk.token_count,
            "page": parent.get("page"),
            "timestamp_start": parent.get("start"),
            "timestamp_end": parent.get("end"),
            "ingested_at": ingested,
            "source_extra": document.extra,
        }
        points.append(QdrantPoint(
            id=stable_chunk_id(document.source_uri, chunk.chunk_index),
            vector=vec.astype(np.float32).tolist(),
            payload=payload,
        ))
    return points


def _build_segment_lookup(document: Document) -> list[tuple[int, int, dict]]:
    """Her segment için yaklaşık char_start/char_end üret."""
    lookup: list[tuple[int, int, dict]] = []
    if not document.segments:
        return lookup
    cursor = 0
    for seg in document.segments:
        if not seg.text:
            continue
        idx = document.text.find(seg.text, cursor)
        if idx < 0:
            idx = cursor
        end = idx + len(seg.text)
        meta = {"page": seg.page, "start": seg.start, "end": seg.end}
        lookup.append((idx, end, meta))
        cursor = end
    return lookup


def _find_parent_segment(chunk: Chunk, lookup, document: Document) -> dict:
    if not lookup:
        return {"page": None, "start": None, "end": None}
    mid = (chunk.char_start + chunk.char_end) // 2
    for start, end, meta in lookup:
        if start <= mid <= end:
            return meta
    # bulunamazsa en yakını
    closest = min(lookup, key=lambda t: abs((t[0] + t[1]) // 2 - mid))
    return closest[2]


def write_outputs(
    document: Document,
    points: Iterable[QdrantPoint],
    out_dir: Path | None = None,
) -> dict[str, Path]:
    out_dir = Path(out_dir or config.OUTPUTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    points_list = list(points)
    stem = _make_stem(document)

    jsonl_path = out_dir / f"{stem}.jsonl"
    parquet_path = out_dir / f"{stem}.parquet"

    _write_jsonl(points_list, jsonl_path)
    _write_parquet(points_list, parquet_path)

    return {"jsonl": jsonl_path, "parquet": parquet_path}


def _make_stem(document: Document) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    label = safe_filename(document.title or document.source_type)
    return f"{ts}_{document.source_type}_{label}"


def _write_jsonl(points: list[QdrantPoint], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for p in points:
            f.write(json.dumps({
                "id": p.id,
                "vector": p.vector,
                "payload": p.payload,
            }, ensure_ascii=False) + "\n")


def _write_parquet(points: list[QdrantPoint], path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    if not points:
        # boş şema yine de yaz
        schema = pa.schema([
            ("id", pa.string()),
            ("vector", pa.list_(pa.float32())),
            ("payload", pa.string()),
        ])
        table = pa.table({"id": [], "vector": [], "payload": []}, schema=schema)
        pq.write_table(table, path)
        return

    ids = [p.id for p in points]
    vectors = [p.vector for p in points]
    payloads = [json.dumps(p.payload, ensure_ascii=False) for p in points]

    table = pa.table({
        "id": ids,
        "vector": vectors,
        "payload": payloads,
    })
    pq.write_table(table, path, compression="zstd")
