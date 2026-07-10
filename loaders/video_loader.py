"""Video loader: ffmpeg ile sesi çıkar, Whisper ile transkripsiyon."""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Callable

from core.types import Document

from ._whisper import get_whisper
from .base import Loader


def _check_ffmpeg() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError(
            "ffmpeg sistemde bulunamadı. README'deki kurulum adımlarını izle."
        )
    return ffmpeg


def extract_audio(video_path: Path, dst_wav: Path) -> None:
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
        path = Path(source)
        engine = get_whisper()

        with tempfile.TemporaryDirectory(prefix="apm_video_") as tmp:
            wav = Path(tmp) / "audio.wav"
            extract_audio(path, wav)
            
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
            source_uri=path.resolve().as_uri(),
            title=path.stem,
            language=lang,
            segments=segments,
            extra={
                "duration_sec": duration,
                "byte_size": path.stat().st_size if path.exists() else 0,
            },
        )
