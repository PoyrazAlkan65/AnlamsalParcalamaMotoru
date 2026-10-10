"""Tek bir kuyruk işini uçtan uca işler: SOURCE → TRANSCRIBE → CHUNK → SCORE → SAVE."""
from __future__ import annotations

import json
import logging
from typing import Callable

from . import chunking, db, llm, normalize, prompt, settings, transcribe

log = logging.getLogger("lesson_analysis")


def run_job(conn, job: dict, set_stage: Callable[[str], None]) -> int:
    """İşi işler, oluşan LessonAnalysisId'yi döndürür. Hata fırlatırsa çağıran işi başarısız kapatır."""
    lesson_id = int(job["CourseDetailsId"])
    transcript_id = job.get("TranscriptId")

    # 1) Transkript
    if job["JobType"] == "FULL" and (transcript_id is None or job.get("ForceTranscribe")):
        if not job.get("LessonsLink"):
            raise RuntimeError("Dersin video adresi (LessonsLink) yok.")
        set_stage("TRANSCRIBE")
        log.info("ders %s: transkript çıkarılıyor", lesson_id)
        t = transcribe.from_video(job["LessonsLink"])
        saved = db.one(conn, "sp_SaveLessonTranscript", CourseDetailsId=lesson_id, TranscriptText=t.text,
                       SegmentsJson=t.segments_json, Source="WHISPER", Model=t.model, LanguageCode=t.language,
                       DurationSec=t.duration, CreateBy=None)
        transcript_id = saved["TranscriptId"]
    if transcript_id is None:
        raise RuntimeError("Dersin transkripti yok.")

    tr = db.one(conn, "sp_GetLessonTranscript", TranscriptId=transcript_id)

    # 2) Parçalar (transkript başına bir kez)
    set_stage("CHUNK")
    chunks = db.rows(conn, "sp_GetLessonChunks", TranscriptId=transcript_id)
    if not chunks:
        duration = float(tr["DurationSec"]) if tr["DurationSec"] else (float(tr["LessonTime"]) or None)
        parts = chunking.split(tr["TranscriptText"], tr["SegmentsJson"], duration)
        payload = [{"seq": c.seq, "text": c.text, "start": c.start, "end": c.end, "tokens": c.tokens} for c in parts]
        db.call(conn, "sp_SaveLessonChunks", TranscriptId=transcript_id,
                ChunksJson=json.dumps(payload, ensure_ascii=False), ChunkerVersion=chunking.chunker_version())
        chunks = db.rows(conn, "sp_GetLessonChunks", TranscriptId=transcript_id)
    log.info("ders %s: %d parça", lesson_id, len(chunks))

    # 3) Köşe skoru
    set_stage("SCORE")
    groups = db.rows(conn, "sp_getGrowthGroups")
    if len(groups) != 9:
        raise RuntimeError(f"9 aktif köşe bekleniyordu, {len(groups)} bulundu.")
    known = {g["Code"] for g in groups}
    text, model = llm.ask(conn, prompt.build(job, groups, chunks))
    parsed = llm.parse_json(text)
    lesson_groups = normalize.lesson_groups(parsed, known, settings.MIN_RATIO)
    chunk_scores = normalize.chunk_scores(parsed, known, {int(c["Seq"]) for c in chunks}, settings.MAX_CHUNK_GROUPS)

    # 4) Kayıt (program dağılımı SP içinde derslerden yeniden üretilir)
    set_stage("SAVE")
    raw = {"summary": str(parsed.get("summary") or "")[:2000], "groups": lesson_groups,
           "chunks": chunk_scores, "minRatio": settings.MIN_RATIO, "llmRaw": text}
    saved = db.one(conn, "sp_SetLessonAnalysis", CourseDetailsId=lesson_id, TranscriptId=transcript_id,
                   RawResponse=json.dumps(raw, ensure_ascii=False), ModelVersion=model,
                   PromptVersion=settings.PROMPT_VERSION, AutoApprove=1 if settings.AUTO_APPROVE else 0, CreateBy=None)
    log.info("ders %s: analiz %s · %s", lesson_id, saved["LessonAnalysisId"],
             " ".join(f"{g['code']}={round(g['ratio'] * 100)}" for g in lesson_groups))
    return int(saved["LessonAnalysisId"])
