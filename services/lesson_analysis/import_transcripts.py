"""Elde hazır transkriptleri (txt, isteğe bağlı aynı adlı Whisper json'u) derslere içe aktarır.

Dosya adından sezon ve bölüm sırası okunur (ör. "...-Sezon1-Bolum2.txt", "sezon3-bölüm1.txt") ve programın
puanlanan derslerine sezon sırası + ders sırası ile eşlenir. Varsayılan kuru çalıştırmadır: eşlemeyi gösterir,
--yes ile yazar, --enqueue ile her ders için yeniden analiz işi açar.

  python -m services.lesson_analysis.import_transcripts --course 33 --dir "C:/Users/Tugce/Desktop/Eğitim-33"
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from . import db

_NAME = re.compile(r"sezon\s*(\d+)\D*?b[öo]l[üu]m\s*(\d+)", re.IGNORECASE)


def _lessons_by_position(conn, course_id: int) -> dict[tuple[int, int], dict]:
    """(sezon sırası, sezondaki ders sırası) → ders. Sıra sp_GetCourseLessonAnalysisOverview'dakidir."""
    out: dict[tuple[int, int], dict] = {}
    season_no: dict[str, int] = {}
    counter: dict[int, int] = {}
    for r in db.rows(conn, "sp_GetCourseLessonAnalysisOverview", CourseId=course_id):
        key = str(r["SeasonName"])
        s = season_no.setdefault(key, len(season_no) + 1)
        counter[s] = counter.get(s, 0) + 1
        out[(s, counter[s])] = r
    return out


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description="Hazır transkriptleri derslere içe aktarır")
    p.add_argument("--course", type=int, required=True, help="Courses.Id")
    p.add_argument("--dir", type=Path, required=True, help="Transkript klasörü")
    p.add_argument("--yes", action="store_true", help="Veritabanına yaz")
    p.add_argument("--enqueue", action="store_true", help="Yazılan dersler için analiz işi aç")
    args = p.parse_args()

    files: dict[tuple[int, int], Path] = {}
    for f in sorted(args.dir.glob("*.txt")):
        m = _NAME.search(f.stem)
        if not m:
            print(f"  ATLA (ad çözülemedi): {f.name}")
            continue
        files[(int(m.group(1)), int(m.group(2)))] = f

    with db.connection() as conn:
        lessons = _lessons_by_position(conn, args.course)
        plan = []
        print(f"Program {args.course}: {len(lessons)} puanlanan ders, {len(files)} dosya\n")
        for pos in sorted(set(lessons) | set(files)):
            lesson, f = lessons.get(pos), files.get(pos)
            text = f.read_text(encoding="utf-8-sig").strip() if f else ""
            label = f"S{pos[0]}B{pos[1]}"
            if not lesson:
                print(f"  {label}: dosya var ama ders yok → {f.name}")
            elif not f:
                print(f"  {label}: ders {lesson['CourseDetailsId']} {lesson['LessonsName']} → TRANSKRİPT YOK")
            elif len(text) < 50:
                print(f"  {label}: ders {lesson['CourseDetailsId']} {lesson['LessonsName']} → BOŞ DOSYA ({f.name})")
            else:
                print(f"  {label}: ders {lesson['CourseDetailsId']} {lesson['LessonsName']} ← {f.name} ({len(text)} kr)")
                plan.append((lesson, f, text))

        if not args.yes:
            print("\nKuru çalıştırma: yazmak için --yes ekleyin.")
            return 0

        print()
        for lesson, f, text in plan:
            seg_path = f.with_suffix(".json")
            segments, duration, source = None, None, "IMPORT"
            if seg_path.exists():  # SedaUlgen-Transkript çıktısı: zaman damgaları korunur
                meta = json.loads(seg_path.read_text(encoding="utf-8"))
                segments = json.dumps(meta.get("segmentler") or [], ensure_ascii=False)
                duration = meta.get("sure_sn")
            saved = db.one(conn, "sp_SaveLessonTranscript", CourseDetailsId=int(lesson["CourseDetailsId"]),
                           TranscriptText=text, SegmentsJson=segments, Source=source, Model=None, LanguageCode="tr",
                           DurationSec=duration, CreateBy=None)
            state = "aynı (değişmedi)" if saved["Unchanged"] else f"yeni sürüm {saved['TranscriptId']}"
            msg = f"  ders {lesson['CourseDetailsId']}: {state}"
            if args.enqueue:
                job = db.one(conn, "sp_EnqueueAnalysisJob", CourseDetailsId=int(lesson["CourseDetailsId"]),
                             JobType="REANALYZE", ForceTranscribe=0, CreateBy=None)
                msg += f" · iş {job['JobId']}{'' if job['Created'] else ' (zaten vardı)'}"
            print(msg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
