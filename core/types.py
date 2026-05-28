"""Pipeline boyunca dolaşan veri sınıfları."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Segment:
    """Zaman damgalı ya da sayfa-numaralı bir parça (whisper/pdf gibi)."""
    text: str
    start: Optional[float] = None
    end: Optional[float] = None
    page: Optional[int] = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class Document:
    """Tek bir kaynağın metne dönüşmüş hali."""
    text: str
    source_type: str
    source_uri: str
    title: Optional[str] = None
    language: Optional[str] = None
    extra: dict[str, Any] = field(default_factory=dict)
    segments: list[Segment] = field(default_factory=list)


@dataclass
class Chunk:
    """Değişken uzunlukta semantik parça."""
    text: str
    chunk_index: int
    char_start: int = 0
    char_end: int = 0
    token_count: int = 0
    parent: dict[str, Any] = field(default_factory=dict)


@dataclass
class QdrantPoint:
    """Qdrant'a yüklenmek üzere hazır nokta."""
    id: str
    vector: list[float]
    payload: dict[str, Any]
