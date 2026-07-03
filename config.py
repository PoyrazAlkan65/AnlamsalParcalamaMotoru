"""Merkezi yapılandırma."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# .env dosyasını yükle (proje kökünde aranır)
load_dotenv(Path(__file__).resolve().parent / ".env")

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
# Supported simple names: 'turbo', 'large-v3', 'medium'
WHISPER_AVAILABLE_MODELS = ["turbo", "large-v3", "medium"]
# Map simple names to actual model identifiers (override via env if needed)
WHISPER_MODEL_MAP = {
    "turbo": os.getenv("WHISPER_MODEL_TURBO", "turbo"),
    "large-v3": os.getenv("WHISPER_MODEL_LARGE_V3", "large-v3"),
    "medium": os.getenv("WHISPER_MODEL_MEDIUM", "medium"),
}
# Choose one of the keys in WHISPER_AVAILABLE_MODELS. Default kept as large-v3 for
# quality/compatibility balance. A benchmark script can be used to change this.
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "large-v3")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "auto")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "auto")
WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "tr")
# If set, some automation or scripts may run benchmarking across models.
WHISPER_BENCHMARK = os.getenv("WHISPER_BENCHMARK", "false").lower() in ("1", "true", "yes")

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

# ---- Qdrant ----
QDRANT_HOST = os.getenv("QDRANT_HOST")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION")
QDRANT_ENABLED = os.getenv("QDRANT_ENABLED", "true").lower() in ("1", "true", "yes")
