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

# ---- YouTube (yt-dlp) ----
# YouTube imza çözümü için JS çalışma ortamı gerekir (yt-dlp-ejs). Sırayla denenir: deno, node (>=22), bun, quickjs
YTDLP_JS_RUNTIMES = [r.strip() for r in os.getenv("YTDLP_JS_RUNTIMES", "deno,node").split(",") if r.strip()]

# ---- Kaynak veritabanı (tablo kolonlarını parçalama) ----
# MSSQL: projelerdeki ortak MSSQL_* değişkenleri (salt okunur bir kullanıcı önerilir)
MSSQL_SERVER = os.getenv("MSSQL_SERVER", "")          # sunucu, sunucu,1433 · sunucu:1433 · sunucu\INSTANCE
MSSQL_DBNAME = os.getenv("MSSQL_DBNAME", "")
MSSQL_USER = os.getenv("MSSQL_USER", "")
MSSQL_PWD = os.getenv("MSSQL_PWD", "")
MSSQL_POOL_MAX = int(os.getenv("MSSQL_POOL_MAX", "10"))
MSSQL_POOL_MIN = int(os.getenv("MSSQL_POOL_MIN", "0"))  # SQLAlchemy'de karşılığı yok; bağlantılar gerektikçe açılır
MSSQL_POOL_TIMEOUT = int(os.getenv("MSSQL_POOL_TIMEOUT", "30000"))  # ms
MSSQL_ODBC_DRIVER = os.getenv("MSSQL_ODBC_DRIVER", "ODBC Driver 17 for SQL Server")
MSSQL_ENCRYPT = os.getenv("MSSQL_ENCRYPT", "")         # boş: sürücü varsayılanı · yes / no
MSSQL_TRUST_SERVER_CERTIFICATE = os.getenv("MSSQL_TRUST_SERVER_CERTIFICATE", "yes")

# Diğer veritabanları için SQLAlchemy URL'i; doluysa MSSQL_* yerine bu kullanılır. Örnekler:
#   postgresql+psycopg://kullanici:sifre@sunucu:5432/veritabani
#   sqlite:///data/inputs/ornek.db
SOURCE_DB_URL = os.getenv("SOURCE_DB_URL", "")
# source_uri'de görünen ad (db://<ad>/<tablo>/<id>); boşsa veritabanı adı
SOURCE_DB_LABEL = os.getenv("SOURCE_DB_LABEL", "")

# ---- Qdrant ----
QDRANT_HOST = os.getenv("QDRANT_HOST")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION")
QDRANT_ENABLED = os.getenv("QDRANT_ENABLED", "true").lower() in ("1", "true", "yes")
