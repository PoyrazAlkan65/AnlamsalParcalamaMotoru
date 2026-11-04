"""Servis ayarları: services/lesson_analysis/.env (örnek: .env.example).

Burada varsayılan değer yoktur; eksik anahtar varsa içe aktarma sırasında hata verilir.
"""
from __future__ import annotations

import os
import socket
from pathlib import Path

from dotenv import load_dotenv

_ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(_ENV_PATH)

_missing: list[str] = []
_invalid: list[str] = []


def _env(name: str, required: bool = True) -> str:
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
    raw = _env(name)
    try:
        return cast(raw)
    except ValueError:
        if raw:
            _invalid.append(f"{name}={raw!r}")
        return cast(0)


# ---- LMS veritabanı ----
MSSQL_SERVER = _env("LA_MSSQL_SERVER")
MSSQL_DB = _env("LA_MSSQL_DB")
MSSQL_USER = _env("LA_MSSQL_USER")
MSSQL_PWD = _env("LA_MSSQL_PWD")
ODBC_DRIVER = _env("LA_ODBC_DRIVER")
MSSQL_ENCRYPT = _env("LA_MSSQL_ENCRYPT") == "1"

# ---- Yapay zekâ ----
# LLMModel.Id (sp_getLLMModelConfig); Admin'deki GROWTH_LLM_MODEL_ID ile aynı tutulmalı
LLM_MODEL_ID = _num("LA_LLM_MODEL_ID", int)
LLM_TIMEOUT_SEC = _num("LA_LLM_TIMEOUT_SEC", int)

# ---- Analiz ----
PROMPT_VERSION = "lesson-v1"  # istem metninin sürümü (kodla birlikte değişir)
MIN_RATIO = _num("LA_MIN_RATIO", float)               # ders dağılımında bu oranın altı atılır
MAX_CHUNK_GROUPS = _num("LA_MAX_CHUNK_GROUPS", int)   # parça başına en fazla köşe
AUTO_APPROVE = _env("LA_AUTO_APPROVE") == "1"         # 0: analiz admin onayı bekler (PENDING)

# ---- Parçalama ----
# semantic: motorun HybridChunker'ı (bge-m3); lines: satır gruplama (model yüklemeden, hızlı)
CHUNKER = _env("LA_CHUNKER")
CHUNKER_MAX_CHARS = _num("LA_CHUNKER_MAX_CHARS", int)  # lines modunda parça üst sınırı
CHUNK_MIN_TOKENS = _num("LA_CHUNK_MIN_TOKENS", int)    # bunun altındaki parçalar komşusuyla birleşir

# ---- Worker ----
# Boşsa makine adı ve süreç numarasından üretilir
WORKER_ID = _env("LA_WORKER_ID", required=False) or f"{socket.gethostname()}:{os.getpid()}"
POLL_SECONDS = _num("LA_POLL_SECONDS", int)
STALE_MINUTES = _num("LA_STALE_MINUTES", int)
HEARTBEAT_SECONDS = _num("LA_HEARTBEAT_SECONDS", int)

# ---- API ----
API_KEY = _env("LA_API_KEY", required=False)  # boşsa API istekleri 503 döner

if _missing or _invalid:
    problems = []
    if _missing:
        problems.append("eksik/boş: " + ", ".join(dict.fromkeys(_missing)))
    if _invalid:
        problems.append("hatalı değer: " + ", ".join(_invalid))
    raise RuntimeError(
        f".env ayarları eksik ({_ENV_PATH}) — " + " · ".join(problems)
        + ". Örnek için .env.example dosyasına bakın."
    )
