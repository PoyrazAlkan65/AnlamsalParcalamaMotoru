"""Komut satırından tek bir kaynağı ingest eder. UI'dan bağımsız doğrulama için."""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.pipeline import ingest


def main() -> int:
    if len(sys.argv) < 2:
        print("Kullanım: python scripts/ingest_cli.py <dosya_yolu_veya_URL>")
        return 2

    source = sys.argv[1]
    start = time.perf_counter()
    last_time = [start]

    def progress(msg: str, pct: float | None) -> None:
        now = time.perf_counter()
        delta = now - last_time[0]
        last_time[0] = now
        pct_str = f"{int(pct * 100):3d}%" if pct is not None else " ?? "
        print(f"[{int(now - start):4d}s] {pct_str}  +{delta:5.1f}s  {msg}", flush=True)

    result = ingest(source, progress=progress)

    print()
    print(f"Document: type={result.document.source_type} title={result.document.title!r}")
    print(f"Toplam metin: {len(result.document.text):,} karakter")
    print(f"Chunk sayısı: {result.chunk_count}")
    print(f"Ortalama chunk karakteri: {result.avg_chunk_chars:.1f}")
    print(f"Embedding shape: {result.embeddings.shape}, dtype={result.embeddings.dtype}")
    for kind, path in result.output_files.items():
        print(f"  {kind}: {path}")
    print(f"Toplam süre: {time.perf_counter() - start:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
