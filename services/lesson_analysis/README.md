# Ders Analizi Servisi (SedaUlgen LMS)

Ders videolarından 9 gelişim köşesine **parça ve ders bazlı** etki çıkarır, LMS veritabanına yazar.
Veritabanı tarafı: `SedaUlgenLMS-Admin/migrations/sprint22-lesson-transcript-analysis.sql`.

```
Admin ──► API (/jobs) ──► sp_EnqueueAnalysisJob ──► AnalysisJob
                                                       │
Worker ◄── sp_ClaimNextAnalysisJob ◄───────────────────┘
  ├─ TRANSCRIBE  LessonsLink (Bunny mp4) → ffmpeg → Whisper   (transkript yoksa; içe aktarılmışsa atlanır)
  ├─ CHUNK       HybridChunker (bge-m3) + kısa parçaları birleştirme + video zamanı
  ├─ SCORE       LLM (LLMModel tablosu): ders dağılımı + parça başına köşe skoru ve dayanak
  └─ SAVE        sp_SetLessonAnalysis → program dağılımı derslerden süre ağırlıklı yeniden üretilir
```

## Kurulum

Motorun `.venv`'i kullanılır; ek paketler:

```powershell
.venv\Scripts\pip install -r services\lesson_analysis\requirements.txt
copy services\lesson_analysis\.env.example services\lesson_analysis\.env   # değerleri doldurun
```

ODBC Driver 17 for SQL Server gerekir. Bütün komutlar **proje kökünden** çalıştırılır.

## Kullanım

Hazır transkriptleri içe aktarma (önce kuru çalıştırma, eşlemeyi kontrol edin):

```powershell
.venv\Scripts\python -m services.lesson_analysis.import_transcripts --course 33 --dir "C:\Users\Tugce\Desktop\Eğitim-33"
.venv\Scripts\python -m services.lesson_analysis.import_transcripts --course 33 --dir "..." --yes --enqueue
```

Dosya adındaki `SezonN` / `BölümN` sırası, programın sezon ve ders sırasına eşlenir. Aynı adlı `.json`
(SedaUlgen-Transkript çıktısı) varsa zaman damgaları da alınır.

Worker:

```powershell
.venv\Scripts\python -m services.lesson_analysis.worker --once   # kuyruk boşalınca çıkar
.venv\Scripts\python -m services.lesson_analysis.worker          # sürekli
```

API (iç ağ, `X-Api-Key` başlığı):

```powershell
.venv\Scripts\uvicorn services.lesson_analysis.api:app --host 127.0.0.1 --port 7600
```

| Uç | İş |
|---|---|
| `GET /health` | DB bağlantısı |
| `POST /jobs` `{courseDetailsId, jobType: FULL\|REANALYZE, forceTranscribe}` | Derse iş aç |
| `POST /courses/{id}/jobs?onlyMissing=true` | Programın analizi olmayan derslerine iş aç |
| `GET /courses/{id}/lessons` | Derslerin transkript / analiz / iş durumu |
| `GET /lessons/{id}/analysis` | Dersin köşeleri ve parçaları |
| `POST /lessons/{id}/transcript` `{text}` | Elle transkript |
| `POST /analyses/{id}/review` `{approve}` | Admin onayı (`LA_AUTO_APPROVE=0` iken) |

## Notlar

- İstem sürümü `lesson-v1` (`prompt.py`); köşe kuralları Admin `courseGrowth.js` / LMS `growthAnalyzeCourses.js` ile aynıdır.
- Ders dağılımında `LA_MIN_RATIO` (%5) altı köşeler atılır, kalanlar 100'e ölçeklenir.
- Parçalar transkript başına bir kez üretilir; analiz yinelenirse aynı parçalar yeniden puanlanır.
- CPU'da parçalama 15 dk'lık derste ~4 dk, Whisper turbo yaklaşık gerçek zamanlı sürer.
- Testler: `.venv\Scripts\python -m pytest -q tests/test_lesson_analysis.py`
