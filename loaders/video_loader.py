"""Video loader: ffmpeg ile sesi çıkar, Whisper ile transkripsiyon."""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Callable
from urllib.parse import unquote, urlparse

from core.types import Document

from ._whisper import get_whisper
from .base import Loader
from .ffmpeg_utils import find_ffmpeg


def _check_ffmpeg() -> str:
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        raise RuntimeError(
            "ffmpeg sistemde bulunamadı. README'deki kurulum adımlarını izle."
        )
    return ffmpeg


def extract_audio(video_path: str | Path, dst_wav: Path) -> None:
    """Videoyu 16kHz mono WAV'a indirger (Whisper'ın istediği format)."""
    ffmpeg = _check_ffmpeg()
    cmd = [
        ffmpeg, "-y", "-i", str(video_path),
        "-vn", "-acodec", "pcm_s16le",
        "-ar", "16000", "-ac", "1",
        str(dst_wav),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg başarısız: {proc.stderr[:500]}")


class VideoLoader(Loader):
    source_type = "video"

    def load(
        self,
        source: str,
        whisper_language: str | None = None,
        check_cancelled: Callable[[], bool] | None = None,
    ) -> Document:
        parsed = urlparse(source)
        is_url = parsed.scheme in {"http", "https"}
        path = Path(source) if not is_url else None
        title = (
            Path(unquote(parsed.path)).stem
            if is_url
            else path.stem
        )
        engine = get_whisper()

        with tempfile.TemporaryDirectory(prefix="apm_video_") as tmp:
            wav = Path(tmp) / "audio.wav"
            extract_audio(source if is_url else path, wav)
            
            if check_cancelled and check_cancelled():
                from core.types import OperationCancelled
                raise OperationCancelled("İşlem kullanıcı tarafından iptal edildi.")

            text, lang, segments = engine.transcribe(
                wav,
                language=whisper_language,
                check_cancelled=check_cancelled,
            )

        duration = segments[-1].end if segments else None
        return Document(
            text=text,
            source_type=self.source_type,
            source_uri=source if is_url else path.resolve().as_uri(),
            title=title,
            language=lang,
            segments=segments,
            extra={
                "duration_sec": duration,
                "byte_size": path.stat().st_size if path and path.exists() else None,
            },
        )
