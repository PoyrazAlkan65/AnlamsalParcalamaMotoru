# Anlamsal Parçalama Motoru

Çok kaynaklı veriyi (resim · PDF · video · YouTube · metin dosyası · MP3 · web URL) tek bir Streamlit uygulamasında içeri alır, semantik metne çevirir, anlamsal chunk'lara böler ve Qdrant'a yüklenmek üzere `JSONL + Parquet` olarak dışa aktarır.

## Mimari özet

```
UI (Streamlit)
  └─► core/pipeline.ingest(source)
        ├─► loaders/<tip>.load()  → Document (düz metin + segmentler + meta)
        ├─► processing/chunker    → değişken uzunlukta semantik Chunk listesi
        ├─► processing/embedder   → BAAI/bge-m3 1024d, normalize
        └─► export/qdrant_writer  → data/outputs/<ts>_<tip>_<isim>.{jsonl,parquet}
```

- **Vektör boyutu sabit** (BGE-M3, 1024d). **Chunk uzunluğu değişken** — semantic chunker, ardışık cümleler arası benzerlik düştüğünde keser.
- Modellerin tamamı **local**: faster-whisper, sentence-transformers, BLIP, Tesseract.
- Qdrant'a şimdilik **yazılmaz**, sadece `qdrant-client.upsert` ile birebir uyumlu point şemasında dosyalara dökülür. Sonraki adımda `scripts/upload_to_qdrant.py` eklenecek.

## Sistem bağımlılıkları (Windows)

```powershell
# ffmpeg
winget install Gyan.FFmpeg

# Tesseract OCR (Türkçe dil paketi ile)
winget install UB-Mannheim.TesseractOCR
# Kurulumdan sonra "Additional language data" → Turkish seçilmeli
# Eğer yol bulunamazsa config.py içinde TESSERACT_CMD'i set et,
# veya env: setx TESSERACT_CMD "C:\Program Files\Tesseract-OCR\tesseract.exe"
```

`ffmpeg -version` ve `tesseract --version` PowerShell'de çalışmalı.

## Python kurulumu

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
playwright install chromium
```

GPU varsa CUDA için torch'u uygun sürümüyle (örn. `pip install torch --index-url https://download.pytorch.org/whl/cu121`) önceden yükle.

## Çalıştırma

```powershell
streamlit run app.py
```

Tarayıcıda açılan UI'da:
- **Dosya yükle**: çoklu yükleme — pdf, png/jpg, mp3/wav/m4a, mp4/mkv, txt/md/csv/json.
- **URL gir**: YouTube linki veya herhangi bir web sayfası. YouTube otomatik tespit edilir.
- **Sonuçlar**: her kaynak için chunk sayısı, ortalama uzunluk, ilk parça önizlemesi, JSONL/Parquet indirme.

Çıktı dosyaları `data/outputs/` altında birikir. Parquet şeması:

```
id      string
vector  list<float32>   # 1024
payload string          # JSON (text, source_type, source_uri, page, timestamp_start, …)
```

## Konfigürasyon

Tüm varsayılanlar `config.py` içinde, env değişkenleri ile override edilebilir:

| Env | Varsayılan | Not |
|---|---|---|
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | sentence-transformers ile yüklenir |
| `WHISPER_MODEL` | `large-v3` | Daha hızlı için `turbo` ya da `medium` |
| `WHISPER_LANGUAGE` | `tr` | `None` veya boşluk = otomatik tespit |
| `WHISPER_DEVICE` | `auto` | `cuda` / `cpu` |
| `CAPTION_MODEL` | `Salesforce/blip-image-captioning-large` | BLIP image captioning |
| `OCR_LANGS` | `tur+eng` | Tesseract dil dizesi |
| `CHUNK_SIM_THRESHOLD` | `0.5` | Düşürürsen daha uzun chunk |
| `CHUNK_MIN_TOKENS` | `64` | Bu altındakiler komşusuyla birleştirilir |
| `CHUNK_MAX_TOKENS` | `512` | Zorlama üst sınır |
| `TESSERACT_CMD` | (boş) | Tesseract exe'sinin tam yolu (Windows'ta gerekebilir) |

## Test

`tests/smoke_test.py` her loader için minimal sağlık kontrolü yapar:

```powershell
pytest -q
```

Şu an fixture dosyaları içermiyor — `tests/fixtures/` altına küçük örnekler koyup işaretleyebilirsin.

## Klasör yapısı

```
.
├── app.py                       # Streamlit UI
├── config.py                    # Ayarlar
├── core/
│   ├── pipeline.py              # ingest() orkestrasyon
│   ├── types.py                 # Document / Chunk / QdrantPoint
│   └── utils.py
├── loaders/
│   ├── base.py                  # Loader ABC
│   ├── _whisper.py              # Paylaşılan WhisperEngine
│   ├── text_loader.py
│   ├── web_loader.py            # Trafilatura → Playwright fallback
│   ├── pdf_loader.py            # PyMuPDF
│   ├── audio_loader.py          # Whisper
│   ├── video_loader.py          # ffmpeg → Whisper
│   ├── youtube_loader.py        # yt-dlp → Whisper
│   └── image_loader.py          # Tesseract OCR + BLIP caption
├── processing/
│   ├── chunker.py               # Paragraf ön-bölme + chonkie SemanticChunker
│   └── embedder.py              # BGE-M3 singleton
├── export/
│   └── qdrant_writer.py         # JSONL + Parquet
├── data/
│   ├── inputs/                  # geçici yüklemeler (gitignore)
│   └── outputs/                 # üretilen point dosyaları (gitignore)
├── tests/
│   └── smoke_test.py
└── requirements.txt
```

## Sonraki adımlar

- `scripts/upload_to_qdrant.py`: parquet → `qdrant-client.upsert`.
- Çoklu kaynak için batch işleme (kuyruk + paralel worker).
- Görsel için BLIP yerine daha güçlü VLM (Florence-2 / Qwen-VL) opsiyonu.
- Video frame örnekleme + caption (şimdilik sadece ses).
