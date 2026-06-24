"""Image loader: Tesseract OCR (tur+eng) + BLIP captioning birleşik."""
from __future__ import annotations

import threading
from pathlib import Path

import config
from core.types import Document

from .base import Loader


class _CaptionPipeline:
    _instance: "_CaptionPipeline | None" = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        from transformers import BlipForConditionalGeneration, BlipProcessor
        import torch

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = BlipProcessor.from_pretrained(config.CAPTION_MODEL)
        self.model = BlipForConditionalGeneration.from_pretrained(config.CAPTION_MODEL).to(self.device)
        self.model.eval()
        self._torch = torch
        self._initialized = True

    def caption(self, pil_image) -> str:
        with self._torch.inference_mode():
            inputs = self.processor(pil_image, return_tensors="pt").to(self.device)
            out = self.model.generate(**inputs, max_new_tokens=60)
            return self.processor.decode(out[0], skip_special_tokens=True).strip()


def _run_ocr(pil_image) -> tuple[str, float]:
    """Tesseract OCR → (text, avg_confidence). Tesseract yoksa boş döner."""
    try:
        import pytesseract
    except ImportError:
        return "", 0.0

    if config.TESSERACT_CMD:
        pytesseract.pytesseract.tesseract_cmd = config.TESSERACT_CMD

    try:
        data = pytesseract.image_to_data(
            pil_image,
            lang=config.OCR_LANGS,
            output_type=pytesseract.Output.DICT,
        )
    except pytesseract.TesseractNotFoundError:
        return "", 0.0
    except Exception:
        return "", 0.0

    words: list[str] = []
    confs: list[float] = []
    for word, conf in zip(data.get("text", []), data.get("conf", [])):
        if not word or not word.strip():
            continue
        try:
            c = float(conf)
        except (TypeError, ValueError):
            c = -1.0
        if c < 0:
            continue
        words.append(word.strip())
        confs.append(c)

    text = " ".join(words).strip()
    avg = (sum(confs) / len(confs)) if confs else 0.0
    return text, avg


class ImageLoader(Loader):
    source_type = "image"

    def load(self, source: str, whisper_language: str | None = None) -> Document:
        from PIL import Image

        path = Path(source)
        with Image.open(path) as im:
            im = im.convert("RGB")

            ocr_text, ocr_conf = _run_ocr(im)

            try:
                caption = _CaptionPipeline().caption(im)
            except Exception as e:
                caption = ""
                caption_error = str(e)
            else:
                caption_error = None

            width, height = im.size

        parts: list[str] = []
        if caption:
            parts.append(f"[Görsel açıklaması] {caption}")
        if ocr_text:
            parts.append(f"[OCR metni]\n{ocr_text}")
        full = "\n\n".join(parts).strip() or "[Boş görsel — metin/caption üretilemedi]"

        return Document(
            text=full,
            source_type=self.source_type,
            source_uri=path.resolve().as_uri(),
            title=path.name,
            language=None,
            extra={
                "width": width,
                "height": height,
                "ocr_confidence": ocr_conf,
                "has_caption": bool(caption),
                "has_ocr": bool(ocr_text),
                "caption_error": caption_error,
                "byte_size": path.stat().st_size if path.exists() else 0,
            },
        )
