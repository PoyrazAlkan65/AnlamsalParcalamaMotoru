"""Paylaşılan faster-whisper instance (audio + video + youtube için)."""
from __future__ import annotations

import threading
from pathlib import Path

import config
from core.types import Segment


class WhisperEngine:
    _instance: "WhisperEngine | None" = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        from faster_whisper import WhisperModel

        compute = config.WHISPER_COMPUTE_TYPE
        device = config.WHISPER_DEVICE
        if device == "auto":
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
        if compute == "auto":
            compute = "float16" if device == "cuda" else "int8"

        # Resolve the configured simple name to the actual model identifier
        model_name = getattr(config, "WHISPER_MODEL_MAP", {}).get(
            config.WHISPER_MODEL, config.WHISPER_MODEL
        )
        self.model = WhisperModel(
            model_name,
            device=device,
            compute_type=compute,
        )
        self.device = device
        self.compute = compute
        self._initialized = True

    def transcribe(self, audio_path: str | Path, language: str | None = None) -> tuple[str, str, list[Segment]]:
        """Return (full_text, detected_language, segments)."""
        lang = language or config.WHISPER_LANGUAGE or None
        segments_iter, info = self.model.transcribe(
            str(audio_path),
            language=lang,
            vad_filter=True,
            beam_size=5,
        )
        segments: list[Segment] = []
        texts: list[str] = []
        for seg in segments_iter:
            piece = (seg.text or "").strip()
            if not piece:
                continue
            segments.append(Segment(text=piece, start=float(seg.start), end=float(seg.end)))
            texts.append(piece)

        return "\n".join(texts), info.language, segments


def get_whisper() -> WhisperEngine:
    return WhisperEngine()
