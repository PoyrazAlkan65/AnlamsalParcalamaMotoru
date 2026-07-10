"""Loader registry — kaynak tipini doğru loader'a yönlendirir."""
from __future__ import annotations

from typing import Callable

from core.types import Document
from core.utils import detect_source_type

from .base import Loader


def get_loader(source: str) -> Loader:
    """Kaynak string'ine (yol veya URL) uygun loader'ı döndür."""
    stype = detect_source_type(source)

    if stype == "pdf":
        from .pdf_loader import PDFLoader
        return PDFLoader()
    if stype == "image":
        from .image_loader import ImageLoader
        return ImageLoader()
    if stype == "audio":
        from .audio_loader import AudioLoader
        return AudioLoader()
    if stype == "video":
        from .video_loader import VideoLoader
        return VideoLoader()
    if stype == "youtube":
        from .youtube_loader import YouTubeLoader
        return YouTubeLoader()
    if stype == "text":
        from .text_loader import TextLoader
        return TextLoader()
    if stype == "web":
        from .web_loader import WebLoader
        return WebLoader()

    raise ValueError(f"Desteklenmeyen kaynak tipi: {source!r}")


def load(
    source: str,
    whisper_language: str | None = None,
    check_cancelled: Callable[[], bool] | None = None,
) -> Document:
    return get_loader(source).load(
        source,
        whisper_language=whisper_language,
        check_cancelled=check_cancelled,
    )


__all__ = ["Loader", "get_loader", "load"]
