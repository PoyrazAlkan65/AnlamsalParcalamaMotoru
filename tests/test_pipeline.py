from pathlib import Path
from unittest.mock import patch

from core.pipeline import ingest
from core.types import Document


def test_text_only_ingest_writes_transcript_without_embedding_or_qdrant(
    tmp_path: Path,
):
    document = Document(
        text="Transkript metni.",
        source_type="youtube",
        source_uri="https://www.youtube.com/watch?v=example",
        title="Örnek video",
    )

    with (
        patch("core.pipeline.loaders.load", return_value=document),
        patch("core.pipeline.chunk_text", side_effect=AssertionError("chunking called")),
        patch("core.pipeline.get_embedder", side_effect=AssertionError("embedding called")),
        patch("core.pipeline.upsert_to_qdrant", side_effect=AssertionError("Qdrant called")),
    ):
        result = ingest("https://www.youtube.com/watch?v=example", out_dir=tmp_path, text_only=True)

    text_path = result.output_files["txt"]
    assert text_path.read_text(encoding="utf-8") == document.text
    assert result.chunks == []
    assert result.points == []
    assert result.qdrant_result == {}


# ---- process() / publish() ----

from unittest.mock import MagicMock

import numpy as np
import pytest

from core.pipeline import process, publish
from core.types import Chunk


def _doc() -> Document:
    return Document(text="Birinci parça.\n\nİkinci parça.", source_type="text", source_uri="ornek.txt", title="Örnek")


def _chunks() -> list[Chunk]:
    return [Chunk(text="Birinci parça.", chunk_index=0), Chunk(text="İkinci parça.", chunk_index=1)]


def _embedder() -> MagicMock:
    embedder = MagicMock()
    embedder.encode.side_effect = lambda texts: np.ones((len(texts), 4), dtype=np.float32)
    return embedder


def test_process_previews_without_embedding_files_or_qdrant(tmp_path: Path):
    with (
        patch("core.pipeline.loaders.load", return_value=_doc()),
        patch("core.pipeline.chunk_text", return_value=_chunks()),
        patch("core.pipeline.get_embedder", side_effect=AssertionError("embedding called")),
        patch("core.pipeline.upsert_to_qdrant", side_effect=AssertionError("Qdrant called")),
    ):
        result = process("ornek.txt")

    assert result.chunk_count == 2
    assert result.points == []
    assert result.output_files == {}
    assert result.qdrant_result == {}
    assert not result.has_embeddings


def test_publish_after_preview_saves_to_qdrant_without_reloading(tmp_path: Path):
    embedder = _embedder()
    with (
        patch("core.pipeline.loaders.load", return_value=_doc()) as load,
        patch("core.pipeline.chunk_text", return_value=_chunks()),
        patch("core.pipeline.get_embedder", return_value=embedder),
        patch("core.pipeline.upsert_to_qdrant", return_value={"success": True, "upserted": 2, "error": None}) as upsert,
    ):
        result = process("ornek.txt")
        publish(result, qdrant=True, out_dir=tmp_path)

    load.assert_called_once()
    upsert.assert_called_once()
    assert len(upsert.call_args.args[0]) == 2
    assert result.qdrant_result["upserted"] == 2
    assert result.output_files == {}
    assert list(tmp_path.iterdir()) == []


def test_publish_reuses_embeddings_across_calls(tmp_path: Path):
    embedder = _embedder()
    with (
        patch("core.pipeline.loaders.load", return_value=_doc()),
        patch("core.pipeline.chunk_text", return_value=_chunks()),
        patch("core.pipeline.get_embedder", return_value=embedder),
        patch("core.pipeline.upsert_to_qdrant", return_value={"success": True, "upserted": 2, "error": None}),
    ):
        result = process("ornek.txt")
        publish(result, outputs=["jsonl"], out_dir=tmp_path)
        publish(result, qdrant=True, outputs=["parquet"], out_dir=tmp_path)

    embedder.encode.assert_called_once()
    assert set(result.output_files) == {"jsonl", "parquet"}
    assert all(p.exists() for p in result.output_files.values())


def test_publish_only_selected_files(tmp_path: Path):
    with (
        patch("core.pipeline.loaders.load", return_value=_doc()),
        patch("core.pipeline.chunk_text", return_value=_chunks()),
        patch("core.pipeline.get_embedder", return_value=_embedder()),
        patch("core.pipeline.upsert_to_qdrant", side_effect=AssertionError("Qdrant called")),
    ):
        result = publish(process("ornek.txt"), outputs=["txt", "parquet"], out_dir=tmp_path)

    assert set(result.output_files) == {"txt", "parquet"}
    assert sorted(p.suffix for p in tmp_path.iterdir()) == [".parquet", ".txt"]


def test_publish_vectors_without_chunks_is_rejected(tmp_path: Path):
    with patch("core.pipeline.loaders.load", return_value=_doc()):
        result = process("ornek.txt", chunk=False)
    with pytest.raises(RuntimeError, match="parçalama gerekli"):
        publish(result, qdrant=True, out_dir=tmp_path)


def test_ingest_default_still_writes_files_and_qdrant(tmp_path: Path):
    with (
        patch("core.pipeline.loaders.load", return_value=_doc()),
        patch("core.pipeline.chunk_text", return_value=_chunks()),
        patch("core.pipeline.get_embedder", return_value=_embedder()),
        patch("core.pipeline.upsert_to_qdrant", return_value={"success": True, "upserted": 2, "error": None}) as upsert,
    ):
        result = ingest("ornek.txt", out_dir=tmp_path)

    upsert.assert_called_once()
    assert set(result.output_files) == {"jsonl", "parquet"}
