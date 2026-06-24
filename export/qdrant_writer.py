"""Qdrant point şemasında JSONL + Parquet yazıcı ve Qdrant sunucusuna otomatik yükleme."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np

import config
from core.types import Chunk, Document, QdrantPoint

from core.utils import safe_filename, stable_chunk_id

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Qdrant client singleton
# ---------------------------------------------------------------------------
_qdrant_client = None


def get_qdrant_client():
    """Qdrant client singleton — ilk çağrıda bağlanır, sonrakilerde aynı nesneyi döner."""
    global _qdrant_client
    if _qdrant_client is not None:
        return _qdrant_client

    if not config.QDRANT_ENABLED:
        return None

    try:
        from qdrant_client import QdrantClient

        url = f"http://{config.QDRANT_HOST}:{config.QDRANT_PORT}"
        kwargs = {
            "url": url,
            "timeout": 30,
        }
        if config.QDRANT_API_KEY:
            kwargs["api_key"] = config.QDRANT_API_KEY

        _qdrant_client = QdrantClient(**kwargs)
        # Bağlantı testi
        _qdrant_client.get_collections()
        logger.info("Qdrant bağlantısı kuruldu: %s:%s", config.QDRANT_HOST, config.QDRANT_PORT)
        return _qdrant_client

    except Exception as exc:
        logger.error("Qdrant bağlantı hatası: %s", exc)
        _qdrant_client = None
        return None


def ensure_collection(collection_name: str | None = None, vector_size: int | None = None) -> bool:
    """Collection yoksa oluşturur. Başarılıysa True döner."""
    client = get_qdrant_client()
    if client is None:
        return False

    collection_name = collection_name or config.QDRANT_COLLECTION
    vector_size = vector_size or config.EMBEDDING_DIM

    try:
        from qdrant_client.models import Distance, VectorParams

        collections = [c.name for c in client.get_collections().collections]
        if collection_name in collections:
            logger.info("Collection zaten mevcut: %s", collection_name)
            return True

        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )
        logger.info("Collection oluşturuldu: %s (dim=%d)", collection_name, vector_size)
        return True

    except Exception as exc:
        logger.error("Collection oluşturma hatası: %s", exc)
        return False


def upsert_to_qdrant(
    points: list[QdrantPoint],
    collection_name: str | None = None,
    batch_size: int = 64,
) -> dict:
    """
    QdrantPoint listesini sunucuya yükler.

    Returns:
        {"success": bool, "upserted": int, "error": str | None}
    """
    collection_name = collection_name or config.QDRANT_COLLECTION
    result = {"success": False, "upserted": 0, "error": None}

    if not config.QDRANT_ENABLED:
        result["error"] = "Qdrant devre dışı (QDRANT_ENABLED=false)"
        return result

    client = get_qdrant_client()
    if client is None:
        result["error"] = "Qdrant bağlantısı kurulamadı"
        return result

    if not ensure_collection(collection_name):
        result["error"] = f"Collection oluşturulamadı: {collection_name}"
        return result

    if not points:
        result["success"] = True
        return result

    try:
        from qdrant_client.models import PointStruct

        # Batch olarak yükle
        total = 0
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            qdrant_points = [
                PointStruct(
                    id=pt.id,
                    vector=pt.vector,
                    payload=pt.payload,
                )
                for pt in batch
            ]
            client.upsert(
                collection_name=collection_name,
                points=qdrant_points,
            )
            total += len(batch)
            logger.info("Qdrant upsert: %d / %d", total, len(points))

        result["success"] = True
        result["upserted"] = total
        logger.info(
            "Qdrant yükleme tamamlandı: %d point → %s", total, collection_name
        )

    except Exception as exc:
        result["error"] = str(exc)
        logger.error("Qdrant upsert hatası: %s", exc)

    return result


# ---------------------------------------------------------------------------
# Point oluşturma (mevcut)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Dosya çıktıları (mevcut)
# ---------------------------------------------------------------------------

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
