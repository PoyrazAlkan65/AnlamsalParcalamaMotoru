import os
import glob
import time
import json
import argparse
from pathlib import Path

# Projenizin kendi config ayarlarını alıyoruz
import config

try:
    from jiwer import wer, cer
except ImportError:
    wer = None
    cer = None

# Orijinal scriptinizdeki faster_whisper altyapısını kullanıyoruz
from faster_whisper import WhisperModel

def load_and_transcribe(model_id: str, audio_path: str, device: str, compute: str) -> dict:
    """
    Orijinal benchmark_whisper.py'daki transkripsiyon fonksiyonu.
    faster_whisper ile kararlı ve format bağımsız çalışır.
    """
    out: dict = {"model_id": model_id}

    # 1. Model Yükleme Süresi
    t0 = time.time()
    model = WhisperModel(model_id, device=device, compute_type=compute)
    load_time = round(time.time() - t0, 2)
    out["load_time_s"] = load_time
    out["load_time_sec"] = load_time

    # 2. Transkripsiyon Süresi
    t1 = time.time()
    segments_iter, info = model.transcribe(str(audio_path), language=config.WHISPER_LANGUAGE or None)
    texts = [(s.text or "").strip() for s in segments_iter if s.text and s.text.strip()]
    trans_time = round(time.time() - t1, 2)
    text = "\n".join(texts)

    duration = info.duration
    out.update({
        "trans_time_s": trans_time,
        "transcribe_time_sec": trans_time,
        "audio_duration_s": round(duration, 2),
        "rtf": round(trans_time / duration if duration else 0, 3),
        "text": text,
        "detected_language": getattr(info, "language", None),
    })
    return out

def run_top_benchmark():
    p = argparse.ArgumentParser(description="Proje standartlarına uygun toplu faster_whisper testi.")
    p.add_argument("--audio_dir", type=str, default="data/inputs/test", help="Txt ve MP3 dosyalarının olduğu yer")
    p.add_argument("--ref_dir", type=str, default="data/inputs/test", help="Referans txt dosyalarının olduğu yer")
    p.add_argument("--output_dir", type=str, default="data/outputs/test", help="Jsonl çıktısının alınacağı yer")
    p.add_argument("--device", default=config.WHISPER_DEVICE)
    p.add_argument("--compute", default=config.WHISPER_COMPUTE_TYPE)
    args = p.parse_args()

    # Klasördeki tüm .mp3 dosyalarını bul
    audio_files = sorted(glob.glob(os.path.join(args.audio_dir, "*.mp3")))
    
    if not audio_files:
        print(f"❌ Hata: Girdi klasöründe .mp3 dosyası bulunamadı -> {args.audio_dir}")
        return

    # config.py içinde tanımlı olan modeller (turbo, medium, large-v3 vb.)
    models_to_test = config.WHISPER_AVAILABLE_MODELS or ["turbo", "medium", "large-v3"]

    print(f"🚀 Toplam {len(audio_files)} adet ses dosyası üzerinde toplu benchmark başlatılıyor...")
    print(f"🤖 Test edilecek modeller: {', '.join(models_to_test)}")
    print(f"💻 Donanım: {args.device} ({args.compute})\n")

    os.makedirs(args.output_dir, exist_ok=True)
    output_jsonl = os.path.join(args.output_dir, "toplu_30_ses_analizi.jsonl")

    # Sonuçları jsonl olarak biriktiriyoruz
    with open(output_jsonl, 'w', encoding='utf-8') as out_f:
        
        for idx, audio_path in enumerate(audio_files, start=1):
            base_name = os.path.splitext(os.path.basename(audio_path))[0]
            ref_path = os.path.join(args.ref_dir, f"{base_name}.txt")
            
            if not os.path.exists(ref_path):
                print(f"⚠️  [{idx}/{len(audio_files)}] Atlandı: {base_name}.mp3 için referans .txt yok")
                continue
                
            print(f"🎵 [{idx}/{len(audio_files)}] İşleniyor: {base_name}")
            ref_text = Path(ref_path).read_text(encoding="utf-8").strip()

            for m in models_to_test:
                # Config dosyasındaki model map eşleşmesini al (örn: turbo -> deepdml/faster-whisper-large-v3-turbo-ct2)
                model_id = config.WHISPER_MODEL_MAP.get(m, m)
                print(f"   ↳ ⏳ Benchmarking {m} -> {model_id}...")
                
                try:
                    # Orijinal faster_whisper transkripsiyon fonksiyonunu çağırıyoruz
                    res = load_and_transcribe(model_id, audio_path, args.device, args.compute)
                    
                    # WER/CER Hesaplama
                    calculated_wer = None
                    calculated_cer = None
                    if ref_text and res.get("text"):
                        try:
                            if wer is not None:
                                calculated_wer = wer(ref_text, res["text"])
                            if cer is not None:
                                calculated_cer = cer(ref_text, res["text"])
                        except Exception:
                            pass

                    detected_language = res.get("detected_language")
                    reference_language = config.WHISPER_LANGUAGE or None
                    language_match = None
                    if detected_language and reference_language:
                        language_match = str(detected_language).lower() == str(reference_language).lower()

                    benchmark_entry = {
                        "audio_file": f"{base_name}.mp3",
                        "model": m,
                        "model_id": model_id,
                        "load_time_sec": res["load_time_s"],
                        "transcribe_time_sec": res["trans_time_s"],
                        "rtf": res["rtf"],
                        "wer": round(calculated_wer, 4) if calculated_wer is not None else None,
                        "cer": round(calculated_cer, 4) if calculated_cer is not None else None,
                        "detected_language": detected_language,
                        "reference_language": reference_language,
                        "language_match": language_match,
                        "timestamp": time.time(),
                        "text": res.get("text", "")
                    }

                    print(f"   ✨ [{m}] -> WER: {benchmark_entry['wer'] if benchmark_entry['wer'] is not None else '-'} | RTF: {benchmark_entry['rtf']} | Süre: {benchmark_entry['transcribe_time_sec']}s")
                    
                    # JSONL dosyasına anlık yaz
                    out_f.write(json.dumps(benchmark_entry, ensure_ascii=False) + "\n")
                    out_f.flush()

                except Exception as e:
                    print(f"   ❌ {m} çalıştırılırken hata oluştu: {str(e)}")
                    continue

    print(f"\n🎯 Tüm süreç başarıyla tamamlandı!")
    print(f"📊 Rapor çıktısı şuraya kaydedildi -> {output_jsonl}")

if __name__ == "__main__":
    run_top_benchmark()

# python -m scripts.benchmark_all_whisper  