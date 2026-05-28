"""Dosya tipi tespiti, encoding, hash yardımcıları."""
from __future__ import annotations

import hashlib
import re
import uuid
from pathlib import Path
from urllib.parse import urlparse

import chardet

_YOUTUBE_RE = re.compile(
    r"(?:youtube\.com/(?:watch\?v=|shorts/|embed/)|youtu\.be/)([A-Za-z0-9_-]{11})"
)

EXT_TO_TYPE = {
    ".pdf": "pdf",
    ".png": "image", ".jpg": "image", ".jpeg": "image",
    ".gif": "image", ".bmp": "image", ".tiff": "image", ".webp": "image",
    ".mp3": "audio", ".wav": "audio", ".m4a": "audio",
    ".ogg": "audio", ".flac": "audio", ".aac": "audio",
    ".mp4": "video", ".mkv": "video", ".webm": "video",
    ".mov": "video", ".avi": "video",
    ".txt": "text", ".md": "text", ".markdown": "text",
    ".csv": "text", ".json": "text", ".log": "text", ".rst": "text",
}


def detect_source_type(source: str) -> str:
    """`source` string'i için kaba kaynak tipini döndür."""
    if source.startswith(("http://", "https://")):
        if _YOUTUBE_RE.search(source):
            return "youtube"
        return "web"
    ext = Path(source).suffix.lower()
    return EXT_TO_TYPE.get(ext, "unknown")


def is_youtube_url(url: str) -> bool:
    return bool(_YOUTUBE_RE.search(url))


def read_text_file(path: str | Path) -> tuple[str, str]:
    """(metin, tespit_edilen_encoding) döndür."""
    raw = Path(path).read_bytes()
    if not raw:
        return "", "utf-8"
    detected = chardet.detect(raw)
    enc = detected.get("encoding") or "utf-8"
    try:
        return raw.decode(enc, errors="replace"), enc
    except LookupError:
        return raw.decode("utf-8", errors="replace"), "utf-8"


def stable_chunk_id(source_uri: str, chunk_index: int) -> str:
    """source_uri + chunk_index üzerinden deterministik UUID5."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_uri}#{chunk_index}"))


def safe_filename(name: str, max_len: int = 80) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    return cleaned[:max_len] or "untitled"


def hash_short(text: str, n: int = 8) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()[:n]


def domain_of(url: str) -> str:
    try:
        return urlparse(url).netloc
    except Exception:
        return ""
