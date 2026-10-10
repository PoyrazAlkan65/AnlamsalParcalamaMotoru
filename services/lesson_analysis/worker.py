"""Kuyruk işleyici: AnalysisJob'dan iş alır, işler, sonucu yazar.

Kullanım (proje kökünden):
  python -m services.lesson_analysis.worker          # sürekli çalışır
  python -m services.lesson_analysis.worker --once   # kuyruk boşalınca çıkar
Uzun aşamalarda (Whisper) ayrı bağlantıyla kalp atışı gönderilir; worker çökerse iş STALE_MINUTES sonra kuyruğa döner.
"""
from __future__ import annotations

import argparse
import logging
import sys
import threading
import time

from . import db, pipeline, settings

log = logging.getLogger("lesson_analysis")


class Heartbeat:
    """İşin aşamasını yazar ve LockedAt'i düzenli yeniler."""

    def __init__(self, job_id: int) -> None:
        self.job_id = job_id
        self.stage = "SOURCE"
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._conn = db.connect()  # pyodbc bağlantısı thread'ler arasında paylaşılmaz
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _send(self) -> None:
        with self._lock:
            db.call(self._conn, "sp_UpdateAnalysisJobStage", JobId=self.job_id, WorkerId=settings.WORKER_ID, Stage=self.stage)

    def set_stage(self, stage: str) -> None:
        self.stage = stage
        self._send()

    def _loop(self) -> None:
        while not self._stop.wait(settings.HEARTBEAT_SECONDS):
            try:
                self._send()
            except Exception as exc:  # kalp atışı hatası işi durdurmasın
                log.warning("kalp atışı gönderilemedi: %s", exc)

    def __enter__(self) -> "Heartbeat":
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join(timeout=5)
        self._conn.close()


def process_next(conn) -> bool:
    """Bir iş işler; kuyruk boşsa False."""
    job = db.one(conn, "sp_ClaimNextAnalysisJob", WorkerId=settings.WORKER_ID, StaleMinutes=settings.STALE_MINUTES)
    if not job:
        return False
    job_id = int(job["JobId"])
    log.info("iş %s alındı: ders %s %s (%s, deneme %s)", job_id, job["CourseDetailsId"], job["LessonsName"],
             job["JobType"], job["Attempts"])
    try:
        with Heartbeat(job_id) as hb:
            analysis_id = pipeline.run_job(conn, job, hb.set_stage)
        db.call(conn, "sp_FinishAnalysisJob", JobId=job_id, WorkerId=settings.WORKER_ID, Success=1,
                ErrorMessage=None, LessonAnalysisId=analysis_id)
        log.info("iş %s tamam", job_id)
    except Exception as exc:
        log.exception("iş %s başarısız", job_id)
        db.call(conn, "sp_FinishAnalysisJob", JobId=job_id, WorkerId=settings.WORKER_ID, Success=0,
                ErrorMessage=f"{type(exc).__name__}: {exc}"[:2000], LessonAnalysisId=None)
    return True


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Ders analizi kuyruk işleyici")
    p.add_argument("--once", action="store_true", help="Kuyruk boşalınca çık")
    args = p.parse_args()

    log.info("worker %s · DB %s · model %s · parçalama %s", settings.WORKER_ID, settings.MSSQL_DB,
             settings.LLM_MODEL_ID, settings.CHUNKER)
    conn = db.connect()
    try:
        while True:
            if process_next(conn):
                continue
            if args.once:
                return 0
            time.sleep(settings.POLL_SECONDS)
    except KeyboardInterrupt:
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
