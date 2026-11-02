#!/usr/bin/env python3
"""Kızagan-TTS'i orijinal PyTorch modeliyle çalıştırıp referans ses ve ölçüm üretir.

Yayımcının inference.py betiği CUDA'ya sabitlendiği için Apple Silicon (MPS) ve CPU'da
çalışmaz. Bu betik aynı VoxCPM API'sini cihaz seçilebilir şekilde kullanır. Üretilen
WAV dosyaları ve results.json, ONNX karşılaştırmasında referans olarak kullanılır.

Örnek:
    python scripts/kizagan/kizagan_reference.py --model models/kizagan \
        --reference models/kizagan/examples/quickstart.wav \
        --sentences-file data/tts_test/cumleler_tr.txt --device mps --output-dir outputs/ref_mps
"""

from __future__ import annotations

import argparse
import json
import platform
import resource
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from voxcpm import VoxCPM

DEFAULT_SAMPLE_RATE = 48000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Kızagan-TTS PyTorch referans çalıştırması ve ölçümü")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--text", help="Seslendirilecek tek cümle")
    src.add_argument("--sentences-file", help="UTF-8 dosya; her boş olmayan satır bir cümle (# ile başlayanlar atlanır)")
    parser.add_argument("--model", required=True, help="Yerel model klasörü")
    parser.add_argument("--reference", required=True, help="Referans ses (mono WAV)")
    parser.add_argument("--output-dir", required=True, help="Yeni çıktı klasörü; önceden var olmamalı")
    parser.add_argument("--device", default="auto", choices=["auto", "mps", "cpu", "cuda"])
    parser.add_argument("--cfg", type=float, default=2.0)
    parser.add_argument("--steps", type=int, default=10, help="Difüzyon adım sayısı (inference_timesteps)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-warmup", action="store_true", help="Ölçüm öncesi ısınma çalıştırmasını atla")
    return parser.parse_args()


def resolve_device(name: str) -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def synchronize(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()
    elif device.startswith("cuda"):
        torch.cuda.synchronize()


def peak_rss_mib() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS bayt, Linux kilobayt döndürür
    return peak / 1024**2 if sys.platform == "darwin" else peak / 1024


def device_memory_mib(device: str) -> float | None:
    if device == "mps":
        return torch.mps.driver_allocated_memory() / 1024**2
    if device.startswith("cuda"):
        return torch.cuda.max_memory_allocated() / 1024**2
    return None


def chip_name() -> str:
    if sys.platform == "darwin":
        try:
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                  capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            pass
    return platform.processor() or platform.machine()


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def load_sentences(args: argparse.Namespace) -> list[str]:
    if args.text:
        return [args.text.strip()]
    lines = Path(args.sentences_file).read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#")]


def main() -> int:
    args = parse_args()
    sentences = load_sentences(args)
    if not sentences:
        print("Seslendirilecek cümle bulunamadı.", file=sys.stderr)
        return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    device = resolve_device(args.device)
    gen_kwargs = dict(reference_wav_path=args.reference, cfg_value=args.cfg,
                      inference_timesteps=args.steps, retry_badcase=False)

    print(f"Cihaz: {device} | cümle sayısı: {len(sentences)}")
    t0 = time.perf_counter()
    model = VoxCPM.from_pretrained(args.model, device=device, load_denoiser=False, optimize=False)
    synchronize(device)
    load_s = time.perf_counter() - t0
    sample_rate = getattr(model.tts_model, "sample_rate", None) or DEFAULT_SAMPLE_RATE
    print(f"Model yüklendi: {load_s:.1f} sn | örnekleme hızı: {sample_rate} Hz")

    if not args.no_warmup:
        print("Isınma çalıştırması (ölçüme dahil değil)...")
        model.generate(text=sentences[0], seed=args.seed, **gen_kwargs)
        synchronize(device)

    rows = []
    for i, text in enumerate(sentences, start=1):
        synchronize(device)
        start = time.perf_counter()
        wav = model.generate(text=text, seed=args.seed, **gen_kwargs)
        synchronize(device)
        elapsed = time.perf_counter() - start

        wav = np.asarray(wav, dtype=np.float32).squeeze()
        audio_s = len(wav) / sample_rate
        wav_path = output_dir / f"{i:02d}.wav"
        sf.write(wav_path, wav, sample_rate)
        row = {"index": i, "text": text, "wav": wav_path.name, "elapsed_s": round(elapsed, 3),
               "audio_s": round(audio_s, 3), "rtf": round(elapsed / audio_s, 3) if audio_s else None,
               "device_memory_mib": device_memory_mib(device)}
        rows.append(row)
        print(f"[{i}/{len(sentences)}] {elapsed:.2f} sn işlem, {audio_s:.2f} sn ses, RTF {row['rtf']}")

    total_elapsed = sum(r["elapsed_s"] for r in rows)
    total_audio = sum(r["audio_s"] for r in rows)
    results = {
        "runtime": "pytorch",
        "device": device,
        "chip": chip_name(),
        "os": platform.platform(),
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "voxcpm": package_version("voxcpm"), "transformers": package_version("transformers")},
        "settings": {"cfg": args.cfg, "steps": args.steps, "seed": args.seed,
                     "retry_badcase": False, "warmup": not args.no_warmup,
                     "reference": str(args.reference), "model": str(args.model)},
        "sample_rate": sample_rate,
        "model_load_s": round(load_s, 2),
        "peak_rss_mib": round(peak_rss_mib(), 1),
        "aggregate_rtf": round(total_elapsed / total_audio, 3) if total_audio else None,
        "mean_elapsed_per_sentence_s": round(total_elapsed / len(rows), 3),
        "utterances": rows,
    }
    (output_dir / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n",
                                             encoding="utf-8")
    print(f"\nToplam RTF: {results['aggregate_rtf']} | cümle başına ort.: {results['mean_elapsed_per_sentence_s']} sn"
          f" | en yüksek bellek (RSS): {results['peak_rss_mib']} MiB")
    print(f"Sonuçlar: {output_dir / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
