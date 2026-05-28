"""data/outputs altındaki en güncel parquet'i analiz eder."""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "outputs"


def main() -> int:
    parquets = sorted(OUT.glob("*.parquet"), key=lambda p: p.stat().st_mtime)
    if not parquets:
        print("data/outputs altında parquet yok.")
        return 1

    f = parquets[-1]
    print(f"Dosya: {f.name}")
    print(f"Boyut: {f.stat().st_size / 1024:.1f} KB")
    print()

    t = pq.read_table(f)
    print("=== ŞEMA ===")
    print(t.schema)
    print()

    n = t.num_rows
    print(f"=== ÖZET ===")
    print(f"Toplam parça : {n}")

    vectors = t["vector"].to_pylist()
    vec_dim = len(vectors[0])
    arr = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1)
    print(f"Vektör boyutu: {vec_dim}")
    print(f"Vektör normu : min={norms.min():.4f}  max={norms.max():.4f}  ort={norms.mean():.4f}  (1.0 ≈ normalize)")
    print(f"NaN/Inf      : {(~np.isfinite(arr).all(axis=1)).sum()} satırda")
    print()

    payloads = [json.loads(p) for p in t["payload"].to_pylist()]

    char_lens = [len(p["text"]) for p in payloads]
    tok_counts = [p.get("token_count", 0) for p in payloads]
    pages = [p.get("page") for p in payloads if p.get("page") is not None]

    print("=== CHUNK UZUNLUK DAĞILIMI (karakter) ===")
    print(f"min   : {min(char_lens)}")
    print(f"max   : {max(char_lens)}")
    print(f"medyan: {statistics.median(char_lens):.0f}")
    print(f"ort.  : {statistics.mean(char_lens):.0f}")
    print(f"stdev : {statistics.pstdev(char_lens):.0f}")
    print()
    print("=== CHUNK TOKEN DAĞILIMI ===")
    print(f"min/max/medyan/ort: {min(tok_counts)} / {max(tok_counts)} / {statistics.median(tok_counts):.0f} / {statistics.mean(tok_counts):.0f}")
    print()

    print("=== SAYFA KAPSAMI ===")
    if pages:
        page_set = sorted(set(pages))
        print(f"Sayfa kapsanan : {len(page_set)} / kullanılan en küçük {page_set[0]} - en büyük {page_set[-1]}")
        from collections import Counter
        cnt = Counter(pages)
        print("En yoğun ilk 5 sayfa:", cnt.most_common(5))
    else:
        print("Sayfa bilgisi yok.")
    print()

    print("=== HİSTOGRAM (karakter) ===")
    bins = [0, 100, 250, 500, 750, 1000, 1500, 2000, 3000, 5000, 99999]
    counts = [0] * (len(bins) - 1)
    for c in char_lens:
        for i in range(len(bins) - 1):
            if bins[i] <= c < bins[i + 1]:
                counts[i] += 1
                break
    for i, c in enumerate(counts):
        bar = "#" * int(c * 40 / max(counts))
        upper = bins[i + 1]
        upper_s = "+" if upper >= 99999 else f"<{upper}"
        print(f"  {bins[i]:>4}-{upper_s:<5}: {c:>4}  {bar}")
    print()

    print("=== İLK 3 PARÇA (metnin ilk 400 karakteri) ===")
    for i in range(min(3, n)):
        p = payloads[i]
        print(f"--- chunk #{p['chunk_index']}  sayfa={p.get('page')}  tokens={p.get('token_count')}  ({len(p['text'])} kar) ---")
        text = p["text"][:400].replace("\n", " ")
        print(text + ("…" if len(p["text"]) > 400 else ""))
        print()

    print("=== ÖRNEK PAYLOAD (chunk #0) ===")
    sample = dict(payloads[0])
    sample["text"] = sample["text"][:120] + "…"
    print(json.dumps(sample, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
