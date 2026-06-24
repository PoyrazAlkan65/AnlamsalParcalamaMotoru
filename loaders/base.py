"""Tüm loader'ların türediği soyut taban."""
from __future__ import annotations

from abc import ABC, abstractmethod

from core.types import Document


class Loader(ABC):
    """Tek bir kaynaktan `Document` üretir."""

    source_type: str = "unknown"

    @abstractmethod
    def load(self, source: str, whisper_language: str | None = None) -> Document:
        ...
