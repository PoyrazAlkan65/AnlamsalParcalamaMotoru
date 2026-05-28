"""Hibrit chunker: paragraf ön-bölme + embedding tabanlı semantic chunking.

Akış:
1. Metni yapısal sınırlardan (boş satır, başlık) kabaca böl.
2. Her büyük blok için chonkie SemanticChunker çağır.
3. Çok kısa parçaları bir sonrakine ekle, çok uzunları zorla böl.
"""
from __future__ import annotations

import re
from typing import Iterable

import config
from core.types import Chunk

_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n+")
_HEADING = re.compile(r"^(#{1,6}\s+|\d+\.\s+|[A-ZĞÜŞİÖÇ ]{8,}$)", re.MULTILINE)


class HybridChunker:
    """Paragraf ön-bölme + chonkie SemanticChunker."""

    def __init__(
        self,
        embedding_model: str | None = None,
        similarity_threshold: float | None = None,
        min_tokens: int | None = None,
        max_tokens: int | None = None,
    ) -> None:
        from chonkie import SemanticChunker

        self.embedding_model = embedding_model or config.EMBEDDING_MODEL
        self.similarity_threshold = (
            similarity_threshold if similarity_threshold is not None
            else config.CHUNK_SIMILARITY_THRESHOLD
        )
        self.min_tokens = min_tokens or config.CHUNK_MIN_TOKENS
        self.max_tokens = max_tokens or config.CHUNK_MAX_TOKENS

        self._semantic = SemanticChunker(
            embedding_model=self.embedding_model,
            threshold=self.similarity_threshold,
            chunk_size=self.max_tokens,
            min_sentences=1,
        )

    # ---- yapısal ön-bölme ----
    def _structural_blocks(self, text: str) -> list[tuple[int, str]]:
        """Paragraflara böl, her bloğun char offset'ini koru."""
        blocks: list[tuple[int, str]] = []
        cursor = 0
        for match in _PARAGRAPH_SPLIT.split(text):
            if not match.strip():
                cursor += len(match)
                continue
            idx = text.find(match, cursor)
            if idx < 0:
                idx = cursor
            blocks.append((idx, match))
            cursor = idx + len(match)
        if not blocks:
            blocks.append((0, text))
        return blocks

    # ---- public ----
    def chunk(self, text: str) -> list[Chunk]:
        text = (text or "").strip()
        if not text:
            return []

        results: list[Chunk] = []
        idx = 0
        for block_offset, block in self._structural_blocks(text):
            if len(block) < 40:
                # Çok kısaysa direkt küçük bir chunk
                results.append(Chunk(
                    text=block.strip(),
                    chunk_index=idx,
                    char_start=block_offset,
                    char_end=block_offset + len(block),
                    token_count=max(1, len(block) // 4),
                ))
                idx += 1
                continue

            try:
                sub_chunks = self._semantic.chunk(block)
            except Exception:
                # chonkie patlarsa düz olarak ekle
                sub_chunks = [type("X", (), {
                    "text": block,
                    "start_index": 0,
                    "end_index": len(block),
                    "token_count": max(1, len(block) // 4),
                })()]

            for sc in sub_chunks:
                start = getattr(sc, "start_index", 0)
                end = getattr(sc, "end_index", len(sc.text))
                results.append(Chunk(
                    text=sc.text.strip(),
                    chunk_index=idx,
                    char_start=block_offset + start,
                    char_end=block_offset + end,
                    token_count=getattr(sc, "token_count", max(1, len(sc.text) // 4)),
                ))
                idx += 1

        return self._merge_small(results)

    def _merge_small(self, chunks: list[Chunk]) -> list[Chunk]:
        """min_tokens altındaki ardışıkları komşusuyla birleştir."""
        if not chunks:
            return chunks
        merged: list[Chunk] = []
        for c in chunks:
            if merged and c.token_count < self.min_tokens \
                    and merged[-1].token_count + c.token_count <= self.max_tokens:
                prev = merged[-1]
                prev.text = (prev.text + "\n\n" + c.text).strip()
                prev.char_end = c.char_end
                prev.token_count += c.token_count
            else:
                merged.append(c)
        # yeniden indeksle
        for i, c in enumerate(merged):
            c.chunk_index = i
        return merged


_default: HybridChunker | None = None


def get_chunker() -> HybridChunker:
    global _default
    if _default is None:
        _default = HybridChunker()
    return _default


def chunk_text(text: str) -> list[Chunk]:
    return get_chunker().chunk(text)
