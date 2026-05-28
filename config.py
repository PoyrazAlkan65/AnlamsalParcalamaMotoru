"""Merkezi yapılandırma."""
from __future__ import annotations

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
INPUTS_DIR = DATA_DIR / "inputs"
OUTPUTS_DIR = DATA_DIR / "outputs"

for _d in (INPUTS_DIR, OUTPUTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---- Embedding ----
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
EMBEDDING_DIM = 1024
EMBEDDING_BATCH = int(os.getenv("EMBEDDING_BATCH", "16"))

# ---- Whisper ----
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "large-v3")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "auto")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "auto")
WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "tr")

# ---- Image captioning ----
CAPTION_MODEL = os.getenv("CAPTION_MODEL", "Salesforce/blip-image-captioning-large")
OCR_LANGS = os.getenv("OCR_LANGS", "tur+eng")

# ---- Chunking ----
CHUNK_SIMILARITY_THRESHOLD = float(os.getenv("CHUNK_SIM_THRESHOLD", "0.5"))
CHUNK_MIN_TOKENS = int(os.getenv("CHUNK_MIN_TOKENS", "64"))
CHUNK_MAX_TOKENS = int(os.getenv("CHUNK_MAX_TOKENS", "512"))

# ---- Tesseract (Windows için yol) ----
TESSERACT_CMD = os.getenv("TESSERACT_CMD", "")

# ---- Web ----
WEB_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
PLAYWRIGHT_TIMEOUT_MS = int(os.getenv("PLAYWRIGHT_TIMEOUT_MS", "30000"))
