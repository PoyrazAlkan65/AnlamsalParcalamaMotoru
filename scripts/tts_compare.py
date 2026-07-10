#!/usr/bin/env python3
"""Coqui TTS ve Tortoise TTS ile aynı metni seslendiren basit deneme betiği."""

from __future__ import annotations

import argparse
import sys
import wave
from pathlib import Path

DEFAULT_TEXT = (
    "Bu bir rüya yorumu denemesidir. Rüyada görülen semboller, duygularınızın bir yansıması olabilir."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Coqui TTS ve Tortoise TTS ile seslendirme denemesi")
    parser.add_argument("--text", default=DEFAULT_TEXT, help="Seslendirilecek metin")
    parser.add_argument("--output-dir", default="data/outputs/tts_demo", help="Çıktı klasörü")
    parser.add_argument("--coqui-model", default="tts_models/tr/vits", help="Coqui için model adı")
    parser.add_argument("--tortoise-voice", default="random", help="Tortoise için ses adı veya random")
    return parser.parse_args()


def ensure_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)


def run_coqui(text: str, output_path: Path, model_name: str) -> None:
    try:
        from TTS.api import TTS
    except ImportError as exc:
        raise RuntimeError("Coqui TTS kurulu değil.") from exc

    tts = TTS(model_name=model_name, gpu=False)
    tts.tts_to_file(text=text, file_path=str(output_path))


def run_tortoise(text: str, output_path: Path, voice_name: str) -> None:
    try:
        from tortoise.api import TextToSpeech
        from tortoise.utils.audio import load_voice
    except ImportError as exc:
        raise RuntimeError("Tortoise TTS kurulu değil.") from exc

    tts = TextToSpeech()

    if voice_name.lower() == "random":
        voice_samples, conditioning_latents = None, None
    else:
        voice_samples, conditioning_latents = load_voice(voice_name)

    try:
        wav = tts.tts_with_preset(
            text,
            voice_samples=voice_samples,
            conditioning_latents=conditioning_latents,
            preset="fast",
        )
    except TypeError:
        wav = tts.tts(
            text,
            voice_samples=voice_samples,
            conditioning_latents=conditioning_latents,
            preset="fast",
        )

    # Tensor verisini CPU'ya alıp numpy array'e ve ardından byte formatına çeviriyoruz
    wav_data = wav.squeeze().cpu().numpy()
    
    # Genlik sınırlarını (-1.0, 1.0) aralığından 16-bit integer aralığına normalize ediyoruz
    import numpy as np
    wav_data = (wav_data * 32767).astype(np.int16)

    # Hiçbir dış kütüphaneye (torchaudio/torchcodec) ihtiyaç duymadan standart wave ile kaydetme
    with wave.open(str(output_path), "wb") as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit (2 bytes)
        wav_file.setframerate(24000)  # 24kHz sample rate
        wav_file.writeframes(wav_data.tobytes())


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    ensure_output_dir(output_dir)

    coqui_path = output_dir / "coqui.wav"
    tortoise_path = output_dir / "tortoise.wav"

    print(f"Metin: {args.text}")
    print(f"Çıktı klasörü: {output_dir}")

    print("\n--- Coqui TTS Başlatılıyor ---")
    try:
        run_coqui(args.text, coqui_path, args.coqui_model)
        print(f"✓ Coqui TTS çıktı: {coqui_path}")
    except Exception as exc:
        print(f"✗ Coqui TTS çalışmadı: {exc}", file=sys.stderr)

    print("\n--- Tortoise TTS Başlatılıyor ---")
    try:
        run_tortoise(args.text, tortoise_path, args.tortoise_voice)
        print(f"✓ Tortoise TTS çıktı: {tortoise_path}")
    except Exception as exc:
        print(f"✗ Tortoise TTS çalışmadı: {exc}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())