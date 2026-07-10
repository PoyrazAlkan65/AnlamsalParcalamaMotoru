"""TXT/MD/CSV/JSON/log düz metin yükleyici."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from core.types import Document
from core.utils import read_text_file

from .base import Loader


class TextLoader(Loader):
    source_type = "text"

    def load(
        self,
        source: str,
        whisper_language: str | None = None,
        check_cancelled: Callable[[], bool] | None = None,
    ) -> Document:
        if check_cancelled and check_cancelled():
            from core.types import OperationCancelled
            raise OperationCancelled("İşlem kullanıcı tarafından iptal edildi.")

        path = Path(source)
        text, encoding = read_text_file(path)
        return Document(
            text=text,
            source_type=self.source_type,
            source_uri=path.resolve().as_uri(),
            title=path.name,
            language=None,
            extra={
                "encoding": encoding,
                "byte_size": path.stat().st_size if path.exists() else 0,
                "extension": path.suffix.lower(),
            },
        )
