"""Transkripti anlamsal parçalara böler ve her parçaya video zamanı verir.

Zaman: Whisper segmentleri varsa parçanın karakter aralığına denk gelen segmentlerden; yoksa (düz metin içe
aktarma) karakter konumunun süreye oranıyla tahmin edilir.
"""
from __future__ import annotations

import bisect
import json
from dataclasses import dataclass

from . import settings


@dataclass
class TimedChunk:
    seq: int
    text: str
    char_start: int
    char_end: int
    tokens: int
    start: float | None = None
    end: float | None = None


def chunker_version() -> str:
    if settings.CHUNKER == "lines":
        return f"lines/max{settings.CHUNKER_MAX_CHARS}/merge{settings.CHUNK_MIN_TOKENS}"
    import config  # motor ayarları
    return (f"hybrid-{config.EMBEDDING_MODEL.split('/')[-1]}/sim{config.CHUNK_SIMILARITY_THRESHOLD}"
            f"/min{config.CHUNK_MIN_TOKENS}/max{config.CHUNK_MAX_TOKENS}/merge{settings.CHUNK_MIN_TOKENS}")


def _semantic(text: str) -> list[TimedChunk]:
    from processing.chunker import HybridChunker  # bge-m3 ilk çağrıda yüklenir
    return [TimedChunk(seq=i + 1, text=c.text, char_start=c.char_start, char_end=c.char_end, tokens=c.token_count)
            for i, c in enumerate(HybridChunker().chunk(text))]


def _lines(text: str, max_chars: int) -> list[TimedChunk]:
    """Model yüklemeden: satırları max_chars'a kadar gruplar (deneme ve yedek)."""
    out: list[TimedChunk] = []
    start = cursor = 0
    buf: list[str] = []
    for line in text.split("\n"):
        line_start = cursor
        cursor += len(line) + 1
        if not line.strip():
            continue
        if buf and sum(len(b) + 1 for b in buf) + len(line) > max_chars:
            chunk = "\n".join(buf)
            out.append(TimedChunk(len(out) + 1, chunk, start, start + len(chunk), max(1, len(chunk) // 4)))
            buf = []
        if not buf:
            start = line_start
        buf.append(line)
    if buf:
        chunk = "\n".join(buf)
        out.append(TimedChunk(len(out) + 1, chunk, start, start + len(chunk), max(1, len(chunk) // 4)))
    return out


def _merge_tiny(chunks: list[TimedChunk], text: str, min_tokens: int) -> list[TimedChunk]:
    """Motorun anlamca farklı diye bıraktığı çok kısa parçaları (ör. tek soru cümlesi) önceki parçaya ekler;
    ilk parça kısaysa sonrakiyle birleşir. Puanlanacak birim olarak bir iki cümlelik parça anlamsızdır."""
    out: list[TimedChunk] = []
    for c in chunks:
        if out and (c.tokens < min_tokens or out[-1].tokens < min_tokens):
            prev = out[-1]
            prev.char_end = c.char_end
            prev.text = text[prev.char_start:prev.char_end].strip()
            prev.tokens += c.tokens
        else:
            out.append(c)
    for i, c in enumerate(out, start=1):
        c.seq = i
    return out


def _attach_times(chunks: list[TimedChunk], text: str, segments_json: str | None, duration: float | None) -> None:
    segments = json.loads(segments_json) if segments_json else []
    if segments:
        # Metin segmentlerin "\n" ile birleşimidir: her segmentin karakter başlangıcı
        offsets, pos = [], 0
        for s in segments:
            offsets.append(pos)
            pos += len(str(s.get("metin", "")).strip()) + 1
        for c in chunks:
            i = max(0, bisect.bisect_right(offsets, c.char_start) - 1)
            j = max(0, bisect.bisect_right(offsets, max(c.char_start, c.char_end - 1)) - 1)
            c.start = float(segments[i]["baslangic"])
            c.end = float(segments[min(j, len(segments) - 1)]["bitis"])
    elif duration and text:
        for c in chunks:
            c.start = round(duration * c.char_start / len(text), 2)
            c.end = round(duration * c.char_end / len(text), 2)


def split(text: str, segments_json: str | None, duration: float | None) -> list[TimedChunk]:
    text = text.strip()
    chunks = _lines(text, settings.CHUNKER_MAX_CHARS) if settings.CHUNKER == "lines" else _semantic(text)
    chunks = _merge_tiny(chunks, text, settings.CHUNK_MIN_TOKENS)
    _attach_times(chunks, text, segments_json, duration)
    return chunks
