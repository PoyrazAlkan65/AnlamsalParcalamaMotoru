"""Tüm loader'ların türediği soyut taban."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

from core.types import Document


class Loader(ABC):
    """Tek bir kaynaktan `Document` üretir."""

    source_type: str = "unknown"

    @abstractmethod
    def load(
        self,
        source: str,
        whisper_language: str | None = None,
        check_cancelled: Callable[[], bool] | None = None,
    ) -> Document:
        ...
