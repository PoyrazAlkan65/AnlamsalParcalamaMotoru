"""Servis ayarları: services/lesson_analysis/.env (yoksa ortam değişkenleri)."""
from __future__ import annotations

import os
import socket
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")


def _env(name: str, default: str = "") -> str:
    return (os.getenv(name) or default).strip()


# ---- LMS veritabanı ----
MSSQL_SERVER = _env("LA_MSSQL_SERVER")
MSSQL_DB = _env("LA_MSSQL_DB", "SedaUlgenTest")
MSSQL_USER = _env("LA_MSSQL_USER")
MSSQL_PWD = _env("LA_MSSQL_PWD")
ODBC_DRIVER = _env("LA_ODBC_DRIVER", "ODBC Driver 17 for SQL Server")
MSSQL_ENCRYPT = _env("LA_MSSQL_ENCRYPT", "0") == "1"

# ---- Yapay zekâ ----
# LLMModel.Id (sp_getLLMModelConfig); Admin'deki GROWTH_LLM_MODEL_ID ile aynı tutulmalı
LLM_MODEL_ID = int(_env("LA_LLM_MODEL_ID", "19"))
LLM_TIMEOUT_SEC = int(_env("LA_LLM_TIMEOUT_SEC", "300"))

# ---- Analiz ----
PROMPT_VERSION = "lesson-v1"
MIN_RATIO = float(_env("LA_MIN_RATIO", "0.05"))         # ders dağılımında bu oranın altı atılır
MAX_CHUNK_GROUPS = int(_env("LA_MAX_CHUNK_GROUPS", "3"))  # parça başına en fazla köşe
AUTO_APPROVE = _env("LA_AUTO_APPROVE", "1") == "1"      # 0: analiz admin onayı bekler (PENDING)

# ---- Parçalama ----
# semantic: motorun HybridChunker'ı (bge-m3); lines: satır gruplama (model yüklemeden, hızlı)
CHUNKER = _env("LA_CHUNKER", "semantic")
CHUNKER_MAX_CHARS = int(_env("LA_CHUNKER_MAX_CHARS", "1600"))  # lines modunda parça üst sınırı
CHUNK_MIN_TOKENS = int(_env("LA_CHUNK_MIN_TOKENS", "60"))       # bunun altındaki parçalar komşusuyla birleşir

# ---- Worker ----
WORKER_ID = _env("LA_WORKER_ID", f"{socket.gethostname()}:{os.getpid()}")
POLL_SECONDS = int(_env("LA_POLL_SECONDS", "15"))
STALE_MINUTES = int(_env("LA_STALE_MINUTES", "30"))
HEARTBEAT_SECONDS = int(_env("LA_HEARTBEAT_SECONDS", "60"))

# ---- API ----
API_KEY = _env("LA_API_KEY")
