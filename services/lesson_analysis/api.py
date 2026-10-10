"""HTTP API (Admin buradan iş açar ve durum okur). Ağır iş yapmaz; işleri worker yürütür.

Çalıştırma (proje kökünden): uvicorn services.lesson_analysis.api:app --host 127.0.0.1 --port 7600
Her istekte X-Api-Key başlığı LA_API_KEY ile aynı olmalı. Servis dışarıya açılmaz, iç ağdan çağrılır.
"""
from __future__ import annotations

import hmac
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from . import db, settings

app = FastAPI(title="SedaUlgen Ders Analizi", version="1.0")


def require_key(x_api_key: str = Header(default="")) -> None:
    if not settings.API_KEY:
        raise HTTPException(503, "LA_API_KEY tanımlı değil.")
    if not hmac.compare_digest(x_api_key, settings.API_KEY):
        raise HTTPException(401, "Geçersiz anahtar.")


def _call(fn, *args, **kwargs):
    with db.connection() as conn:
        try:
            return fn(conn, *args, **kwargs)
        except db.DbError as err:
            # SP'nin THROW mesajı kullanıcıya gösterilebilir
            raise HTTPException(400, str(err)) from err


class JobIn(BaseModel):
    courseDetailsId: int = Field(gt=0)
    jobType: Literal["FULL", "REANALYZE"] = "FULL"
    forceTranscribe: bool = False
    userId: int | None = None


class TranscriptIn(BaseModel):
    text: str = Field(min_length=50)
    segmentsJson: str | None = None
    source: Literal["IMPORT", "MANUAL"] = "MANUAL"
    userId: int | None = None


class ReviewIn(BaseModel):
    approve: bool
    userId: int | None = None


@app.get("/health")
def health():
    with db.connection() as conn:
        groups = db.rows(conn, "sp_getGrowthGroups")
    return {"ok": True, "db": settings.MSSQL_DB, "groups": len(groups)}


@app.post("/jobs", dependencies=[Depends(require_key)])
def enqueue(body: JobIn):
    return _call(db.one, "sp_EnqueueAnalysisJob", CourseDetailsId=body.courseDetailsId, JobType=body.jobType,
                 ForceTranscribe=1 if body.forceTranscribe else 0, CreateBy=body.userId)


@app.post("/courses/{course_id}/jobs", dependencies=[Depends(require_key)])
def enqueue_course(course_id: int, onlyMissing: bool = True, userId: int | None = None):
    return _call(db.one, "sp_EnqueueCourseAnalysisJobs", CourseId=course_id, OnlyMissing=1 if onlyMissing else 0,
                 CreateBy=userId)


@app.get("/courses/{course_id}/lessons", dependencies=[Depends(require_key)])
def course_overview(course_id: int):
    return _call(db.rows, "sp_GetCourseLessonAnalysisOverview", CourseId=course_id)


@app.get("/lessons/{lesson_id}/analysis", dependencies=[Depends(require_key)])
def lesson_analysis(lesson_id: int):
    sets = _call(db.call, "sp_GetLessonAnalysis", CourseDetailsId=lesson_id)
    info = sets[0][0] if sets and sets[0] else None
    return {"analysis": info, "groups": sets[1] if len(sets) > 1 else [], "chunks": sets[2] if len(sets) > 2 else []}


@app.post("/lessons/{lesson_id}/transcript", dependencies=[Depends(require_key)])
def save_transcript(lesson_id: int, body: TranscriptIn):
    return _call(db.one, "sp_SaveLessonTranscript", CourseDetailsId=lesson_id, TranscriptText=body.text.strip(),
                 SegmentsJson=body.segmentsJson, Source=body.source, Model=None, LanguageCode="tr",
                 DurationSec=None, CreateBy=body.userId)


@app.post("/analyses/{analysis_id}/review", dependencies=[Depends(require_key)])
def review(analysis_id: int, body: ReviewIn):
    return _call(db.one, "sp_ApproveLessonAnalysis", LessonAnalysisId=analysis_id,
                 Approve=1 if body.approve else 0, UserId=body.userId)
