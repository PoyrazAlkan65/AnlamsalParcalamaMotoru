"""YouTube loader: yt-dlp ile sesi indir, Whisper'a yolla."""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Callable

from core.types import Document

from ._whisper import get_whisper
from .base import Loader


class YouTubeLoader(Loader):
    source_type = "youtube"

    def load(
        self,
        source: str,
        whisper_language: str | None = None,
        check_cancelled: Callable[[], bool] | None = None,
    ) -> Document:
        try:
            import yt_dlp
        except ImportError as e:
            raise RuntimeError("yt-dlp kurulu değil") from e

        engine = get_whisper()

        with tempfile.TemporaryDirectory(prefix="apm_yt_") as tmp:
            outtmpl = str(Path(tmp) / "%(id)s.%(ext)s")
            ydl_opts = {
                "format": "bestaudio/best",
                "outtmpl": outtmpl,
                "quiet": True,
                "noprogress": True,
                "postprocessors": [
                    {
                        "key": "FFmpegExtractAudio",
                        "preferredcodec": "wav",
                        "preferredquality": "0",
                    }
                ],
                "postprocessor_args": ["-ar", "16000", "-ac", "1"],
            }

            if check_cancelled and check_cancelled():
                from core.types import OperationCancelled
                raise OperationCancelled("İşlem kullanıcı tarafından iptal edildi.")

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(source, download=True)

            if check_cancelled and check_cancelled():
                from core.types import OperationCancelled
                raise OperationCancelled("İşlem kullanıcı tarafından iptal edildi.")

            video_id = info.get("id")
            title = info.get("title") or video_id
            duration = info.get("duration")
            uploader = info.get("uploader")

            wav = next(Path(tmp).glob(f"{video_id}.*"), None)
            if wav is None:
                raise RuntimeError("yt-dlp ses indirilemedi")

            text, lang, segments = engine.transcribe(
                wav,
                language=whisper_language,
                check_cancelled=check_cancelled,
            )

        return Document(
            text=text,
            source_type=self.source_type,
            source_uri=source,
            title=title,
            language=lang,
            segments=segments,
            extra={
                "video_id": video_id,
                "duration_sec": duration,
                "uploader": uploader,
                "webpage_url": info.get("webpage_url"),
            },
        )
