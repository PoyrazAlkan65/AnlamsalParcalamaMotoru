"""BGE-M3 wrapper. Modeli bir kez yükler, batch encode sağlar."""
from __future__ import annotations

import threading
from typing import Iterable

import numpy as np

import config


class BGEEmbedder:
    """sentence-transformers ile BAAI/bge-m3, 1024d normalize edilmiş vektörler."""

    _instance: "BGEEmbedder | None" = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_name: str | None = None) -> None:
        if self._initialized:
            return
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name or config.EMBEDDING_MODEL
        self.dim = config.EMBEDDING_DIM
        self.model = SentenceTransformer(self.model_name)
        self._initialized = True

    def encode(self, texts: Iterable[str], batch_size: int | None = None) -> np.ndarray:
        texts = [t if t else " " for t in texts]
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vectors = self.model.encode(
            texts,
            batch_size=batch_size or config.EMBEDDING_BATCH,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return np.asarray(vectors, dtype=np.float32)


def get_embedder() -> BGEEmbedder:
    return BGEEmbedder()
