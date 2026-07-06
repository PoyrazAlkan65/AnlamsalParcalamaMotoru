"""Basit Whisper benchmark aracı.

Kullanım örneği:
python -m scripts.benchmark_whisper --audio path/to/file.wav --reference path/to/ref.txt

Script, konfigürasyondaki WHISPER_AVAILABLE_MODELS listesini kullanır ve her model
için yükleme süresi, transkripsiyon süresi ve (varsa) WER hesaplar.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
# import wave
# import contextlib

import config

try:
    from jiwer import wer
except Exception:
    wer = None

from faster_whisper import WhisperModel


# def get_audio_duration(path: str) -> float:
#     """WAV dosyasının süresini saniye cinsinden döndürür."""
#     with contextlib.closing(wave.open(path, 'r')) as f:
#         frames = f.getnframes()
#         rate = f.getframerate()
#         return frames / float(rate)


def load_and_transcribe(model_id: str, audio_path: str, device: str, compute: str) -> dict:
    out: dict = {"model_id": model_id}
    t0 = time.time()
    model = WhisperModel(model_id, device=device, compute_type=compute)
    out["load_time_s"] = time.time() - t0

    t1 = time.time()
    segments_iter, info = model.transcribe(str(audio_path), language=config.WHISPER_LANGUAGE or None)
    texts = [(s.text or "").strip() for s in segments_iter if s.text and s.text.strip()]
    trans_time = time.time() - t1
    text = "\n".join(texts)

    duration = info.duration  # <-- artık dosya formatından bağımsız, faster-whisper veriyor
    out.update({
        "trans_time_s": trans_time,
        "audio_duration_s": duration,
        "rtf": trans_time / duration if duration else None,
        "text": text,
        "detected_language": getattr(info, "language", None),
    })
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--audio", required=True)
    p.add_argument("--models", nargs="*", help="List of model simple names to benchmark")
    p.add_argument("--device", default=config.WHISPER_DEVICE)
    p.add_argument("--compute", default=config.WHISPER_COMPUTE_TYPE)
    p.add_argument("--reference", help="Path to reference transcript (plain text)")
    p.add_argument("--out", help="Output jsonl file path", default=None)
    args = p.parse_args()

    audio = Path(args.audio)
    if not audio.exists():
        raise SystemExit(f"Audio file not found: {audio}")

    models = args.models or config.WHISPER_AVAILABLE_MODELS
    results = []
    for m in models:
        model_id = config.WHISPER_MODEL_MAP.get(m, m)
        print(f"Benchmarking {m} -> {model_id}...")
        try:
            res = load_and_transcribe(model_id, audio, args.device, args.compute)
            res["simple_name"] = m
            results.append(res)
            print(f"  load {res['load_time_s']:.2f}s transcribe {res['trans_time_s']:.2f}s\n")
        except Exception as e:
            results.append({"simple_name": m, "model_id": model_id, "error": str(e)})
            print(f"  error: {e}\n")

    # If reference provided and jiwer available, compute WER
    if args.reference and wer is not None:
        ref_text = Path(args.reference).read_text(encoding="utf-8")
        for r in results:
            if "text" in r and r.get("text"):
                try:
                    r["wer"] = wer(ref_text, r["text"])
                except Exception:
                    r["wer"] = None
    elif args.reference:
        print("Reference provided but 'jiwer' not installed; install requirements to compute WER")

    # Output
    if args.out:
        out_path = Path(args.out)
        with out_path.open("w", encoding="utf-8") as fh:
            for r in results:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Results written to {out_path}")
    else:
        for r in results:
            print(json.dumps(r, ensure_ascii=False, indent=2))

    # Karşılaştırma özeti
    print("\n=== ÖZET ===")
    print(f"{'Model':<12}{'Load(s)':<10}{'Trans(s)':<10}{'RTF':<8}{'WER':<8}")
    for r in results:
        if "error" in r:
            print(f"{r['simple_name']:<12}HATA: {r['error']}")
            continue
        rtf_val = f"{r.get('rtf'):.3f}" if r.get("rtf") is not None else "-"
        wer_val = f"{r.get('wer'):.3f}" if r.get("wer") is not None else "-"
        print(f"{r['simple_name']:<12}{r['load_time_s']:<10.2f}{r['trans_time_s']:<10.2f}{rtf_val:<8}{wer_val:<8}")


if __name__ == "__main__":
    main()