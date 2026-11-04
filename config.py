"""Merkezi yapılandırma.

Tüm ayarlar proje kökündeki .env dosyasından okunur; burada varsayılan değer yoktur.
Örnek dosya: .env.example. Değeri boş bırakılabilen (isteğe bağlı) ayarlar da .env'de
anahtar olarak bulunmalıdır; eksik anahtar varsa içe aktarma sırasında hata verilir.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent
load_dotenv(ROOT_DIR / ".env")

DATA_DIR = ROOT_DIR / "data"
INPUTS_DIR = DATA_DIR / "inputs"
OUTPUTS_DIR = DATA_DIR / "outputs"

for _d in (INPUTS_DIR, OUTPUTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

_missing: list[str] = []
_invalid: list[str] = []


def _str(name: str, required: bool = True) -> str:
    """`required=False`: anahtar .env'de olmalı ama değeri boş olabilir."""
    value = os.getenv(name)
    if value is None:
        _missing.append(name)
        return ""
    value = value.strip()
    if required and not value:
        _missing.append(name)
    return value


def _num(name: str, cast):
    raw = _str(name)
    try:
        return cast(raw)
    except ValueError:
        if raw:
            _invalid.append(f"{name}={raw!r}")
        return cast(0)


def _int(name: str) -> int:
    return _num(name, int)


def _float(name: str) -> float:
    return _num(name, float)


def _bool(name: str) -> bool:
    return _str(name).lower() in ("1", "true", "yes")


# ---- Embedding ----
EMBEDDING_MODEL = _str("EMBEDDING_MODEL")
EMBEDDING_DIM = _int("EMBEDDING_DIM")  # modelin vektör boyutu (bge-m3: 1024)
EMBEDDING_BATCH = _int("EMBEDDING_BATCH")

# ---- Whisper ----
# Desteklenen kısa adlar; WHISPER_MODEL bunlardan biri olmalı
WHISPER_AVAILABLE_MODELS = ["turbo", "large-v3", "medium"]
# Kısa adların gerçek model kimlikleri
WHISPER_MODEL_MAP = {
    "turbo": _str("WHISPER_MODEL_TURBO"),
    "large-v3": _str("WHISPER_MODEL_LARGE_V3"),
    "medium": _str("WHISPER_MODEL_MEDIUM"),
}
WHISPER_MODEL = _str("WHISPER_MODEL")
WHISPER_COMPUTE_TYPE = _str("WHISPER_COMPUTE_TYPE")
WHISPER_DEVICE = _str("WHISPER_DEVICE")
WHISPER_LANGUAGE = _str("WHISPER_LANGUAGE")
# Bazı betikler modeller arası karşılaştırma yaparken kullanır
WHISPER_BENCHMARK = _bool("WHISPER_BENCHMARK")

# ---- Image captioning ----
CAPTION_MODEL = _str("CAPTION_MODEL")
OCR_LANGS = _str("OCR_LANGS")

# ---- Chunking ----
CHUNK_SIMILARITY_THRESHOLD = _float("CHUNK_SIM_THRESHOLD")
CHUNK_MIN_TOKENS = _int("CHUNK_MIN_TOKENS")
CHUNK_MAX_TOKENS = _int("CHUNK_MAX_TOKENS")

# ---- Tesseract (Windows için yol; boşsa PATH'teki tesseract) ----
TESSERACT_CMD = _str("TESSERACT_CMD", required=False)

# ---- Web ----
WEB_USER_AGENT = _str("WEB_USER_AGENT")
PLAYWRIGHT_TIMEOUT_MS = _int("PLAYWRIGHT_TIMEOUT_MS")

# ---- YouTube (yt-dlp) ----
# YouTube imza çözümü için JS çalışma ortamı gerekir (yt-dlp-ejs). Sırayla denenir: deno, node (>=22), bun, quickjs
YTDLP_JS_RUNTIMES = [r.strip() for r in _str("YTDLP_JS_RUNTIMES").split(",") if r.strip()]

# ---- Kaynak veritabanı (tablo kolonlarını parçalama) ----
# MSSQL: projelerdeki ortak MSSQL_* değişkenleri (salt okunur bir kullanıcı önerilir).
# SOURCE_DB_URL doluysa MSSQL_* bağlantı bilgileri boş bırakılabilir.
MSSQL_SERVER = _str("MSSQL_SERVER", required=False)   # sunucu, sunucu,1433 · sunucu:1433 · sunucu\INSTANCE
MSSQL_DBNAME = _str("MSSQL_DBNAME", required=False)
MSSQL_USER = _str("MSSQL_USER", required=False)
MSSQL_PWD = _str("MSSQL_PWD", required=False)
MSSQL_POOL_MAX = _int("MSSQL_POOL_MAX")
MSSQL_POOL_MIN = _int("MSSQL_POOL_MIN")  # SQLAlchemy'de karşılığı yok; bağlantılar gerektikçe açılır
MSSQL_POOL_TIMEOUT = _int("MSSQL_POOL_TIMEOUT")  # ms
MSSQL_ODBC_DRIVER = _str("MSSQL_ODBC_DRIVER")
MSSQL_ENCRYPT = _str("MSSQL_ENCRYPT", required=False)  # boş: sürücü varsayılanı · yes / no
MSSQL_TRUST_SERVER_CERTIFICATE = _str("MSSQL_TRUST_SERVER_CERTIFICATE", required=False)

# Diğer veritabanları için SQLAlchemy URL'i; doluysa MSSQL_* yerine bu kullanılır. Örnekler:
#   postgresql+psycopg://kullanici:sifre@sunucu:5432/veritabani
#   sqlite:///data/inputs/ornek.db
SOURCE_DB_URL = _str("SOURCE_DB_URL", required=False)
# source_uri'de görünen ad (db://<ad>/<tablo>/<id>); boşsa veritabanı adı
SOURCE_DB_LABEL = _str("SOURCE_DB_LABEL", required=False)

# ---- Qdrant ----
QDRANT_ENABLED = _bool("QDRANT_ENABLED")
QDRANT_HOST = _str("QDRANT_HOST", required=QDRANT_ENABLED)
QDRANT_PORT = _int("QDRANT_PORT")
QDRANT_API_KEY = _str("QDRANT_API_KEY", required=False) or None
QDRANT_COLLECTION = _str("QDRANT_COLLECTION", required=QDRANT_ENABLED)

# ---- Duygu analizi (NoemaVibeEmotionAI avatar modeli) ----
# local: model JSON'u bu süreçte çalışır (servis gerekmez)
# api:   NoemaVibeAPI nlp-service /classifyEmotion; servis yalnızca 127.0.0.1'i dinler,
#        bu yüzden uygulama NoemaVibeAPI ile aynı sunucuda çalışmalı
EMOTION_BACKENDS = ("local", "api")
EMOTION_ENABLED = _bool("EMOTION_ENABLED")
EMOTION_BACKEND = _str("EMOTION_BACKEND").lower()
if EMOTION_BACKEND and EMOTION_BACKEND not in EMOTION_BACKENDS:
    _invalid.append(f"EMOTION_BACKEND={EMOTION_BACKEND!r} (local · api)")
_emotion_model_path = _str("EMOTION_MODEL_PATH", required=EMOTION_BACKEND == "local")
# Göreli yol proje köküne göre çözülür
EMOTION_MODEL_PATH = (
    (Path(_emotion_model_path) if Path(_emotion_model_path).is_absolute() else ROOT_DIR / _emotion_model_path)
    if _emotion_model_path else None
)
EMOTION_API_URL = _str("EMOTION_API_URL", required=EMOTION_BACKEND == "api")
EMOTION_API_TIMEOUT = _float("EMOTION_API_TIMEOUT")
EMOTION_API_CONCURRENCY = _int("EMOTION_API_CONCURRENCY")

if _missing or _invalid:
    problems = []
    if _missing:
        problems.append("eksik/boş: " + ", ".join(dict.fromkeys(_missing)))
    if _invalid:
        problems.append("hatalı değer: " + ", ".join(_invalid))
    raise RuntimeError(
        f".env ayarları eksik ({ROOT_DIR / '.env'}) — " + " · ".join(problems)
        + ". Örnek için .env.example dosyasına bakın."
    )
