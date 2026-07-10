"""MP3/WAV/M4A audio loader: faster-whisper ile transkripsiyon."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from core.types import Document

from ._whisper import get_whisper
from .base import Loader


class AudioLoader(Loader):
    source_type = "audio"

    def load(
        self,
        source: str,
        whisper_language: str | None = None,
        check_cancelled: Callable[[], bool] | None = None,
    ) -> Document:
        path = Path(source)
        engine = get_whisper()
        text, lang, segments = engine.transcribe(
            path,
            language=whisper_language,
            check_cancelled=check_cancelled,
        )

        duration = segments[-1].end if segments else None

        return Document(
            text=text,
            source_type=self.source_type,
            source_uri=path.resolve().as_uri(),
            title=path.stem,
            language=lang,
            segments=segments,
            extra={
                "duration_sec": duration,
                "byte_size": path.stat().st_size if path.exists() else 0,
                "whisper_model": engine.model.__class__.__name__,
            },
        )
