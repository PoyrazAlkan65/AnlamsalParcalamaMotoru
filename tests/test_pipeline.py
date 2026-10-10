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
