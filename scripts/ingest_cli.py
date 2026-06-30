"""Komut satırından tek bir kaynağı veya tüm bir klasörü ingest eder. UI'dan bağımsız doğrulama için."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.pipeline import ingest

# Desteklenen uzantılar
ACCEPT_EXTENSIONS = {
    "pdf",
    "png", "jpg", "jpeg", "bmp", "tiff", "webp", "gif",
    "mp3", "wav", "m4a", "ogg", "flac",
    "mp4", "mkv", "webm", "mov", "avi",
    "txt", "md", "csv", "json", "log", "rst",
}


def is_supported_file(path: Path) -> bool:
    """Dosyanın desteklenen uzantılardan birine sahip olup olmadığını kontrol eder."""
    if not path.is_file():
        return False
    ext = path.suffix.lower().lstrip(".")
    return ext in ACCEPT_EXTENSIONS


def process_single(source: str, show_logs: bool = True) -> tuple[bool, str]:
    """Tek bir kaynağı (dosya veya URL) işler. (Başarı, Mesaj) döner."""
    start = time.perf_counter()
    last_time = [start]

    def progress_cb(msg: str, pct: float | None) -> None:
        if not show_logs:
            return
        now = time.perf_counter()
        delta = now - last_time[0]
        last_time[0] = now
        pct_str = f"{int(pct * 100):3d}%" if pct is not None else " ?? "
        print(f"[{int(now - start):4d}s] {pct_str}  +{delta:5.1f}s  {msg}", flush=True)

    try:
        result = ingest(source, progress=progress_cb)
        duration = time.perf_counter() - start
        msg = f"Başarılı: type={result.document.source_type} chunks={result.chunk_count} ({duration:.1f}s)"
        return True, msg
    except Exception as e:
        duration = time.perf_counter() - start
        return False, f"Hata: {str(e)} ({duration:.1f}s)"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Komut satırından tek bir kaynağı veya bir klasördeki tüm dosyaları ingest eder."
    )
    parser.add_argument(
        "source",
        help="İşlenecek dosya yolu, klasör yolu veya internet adresi (URL) / YouTube linki."
    )
    parser.add_argument(
        "-r", "--recursive",
        action="store_true",
        help="Eğer kaynak bir klasörse, alt klasörleri de tarar."
    )
    parser.add_argument(
        "-w", "--workers",
        type=int,
        default=1,
        help="Paralel çalışacak iş parçacığı (worker) sayısı (varsayılan: 1)."
    )

    args = parser.parse_args()
    source_str = args.source
    source_path = Path(source_str)

    # Klasör tarama kontrolü
    if source_path.exists() and source_path.is_dir():
        print(f"\n[Klasör Algılandı] Tarama dizini: {source_path.resolve()}")
        pattern = "**/*" if args.recursive else "*"
        all_files = [p for p in source_path.glob(pattern) if is_supported_file(p)]

        if not all_files:
            print(f"[Hata] Klasörde desteklenen formatta dosya bulunamadı.", file=sys.stderr)
            return 1

        total_files = len(all_files)
        print(f"Bulunan dosya sayısı: {total_files}")
        print(f"Çalıştırılan paralel işçi (workers): {args.workers}")
        print("İşlem başlatılıyor...\n")

        start_all = time.perf_counter()
        success_count = 0
        fail_count = 0

        # tqdm entegrasyonu (varsa kullanır, yoksa konsola yazdırır)
        try:
            from tqdm import tqdm
            has_tqdm = True
        except ImportError:
            has_tqdm = False

        pbar = tqdm(total=total_files, desc="İlerleme", unit="dosya") if has_tqdm else None

        def update_progress(success: bool, filepath: Path, message: str):
            nonlocal success_count, fail_count
            if success:
                success_count += 1
            else:
                fail_count += 1

            status_str = f" [Başarılı: {success_count} | Hata: {fail_count}]"
            if pbar:
                pbar.set_postfix_str(status_str)
                pbar.set_description(f"İşleniyor: {filepath.name[:25]}")
                pbar.update(1)
            else:
                print(f"[{success_count + fail_count}/{total_files}] {filepath.name} -> {message}", flush=True)

        if args.workers <= 1:
            for f in all_files:
                success, msg = process_single(str(f), show_logs=False)
                update_progress(success, f, msg)
        else:
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                future_to_file = {
                    executor.submit(process_single, str(f), False): f
                    for f in all_files
                }
                for future in as_completed(future_to_file):
                    f = future_to_file[future]
                    try:
                        success, msg = future.result()
                    except Exception as e:
                        success, msg = False, f"Beklenmeyen Hata: {e}"
                    update_progress(success, f, msg)

        if pbar:
            pbar.close()

        duration_all = time.perf_counter() - start_all
        print("\n" + "=" * 60)
        print("TOPLU İŞLEM ÖZETİ")
        print(f"Toplam Süre      : {duration_all:.1f} saniye")
        print(f"Başarılı Dosya   : {success_count}")
        print(f"Hatalı Dosya     : {fail_count}")
        print("=" * 60)
        return 0 if fail_count == 0 else 1

    else:
        # Tek dosya veya URL
        start = time.perf_counter()
        last_time = [start]

        def progress(msg: str, pct: float | None) -> None:
            now = time.perf_counter()
            delta = now - last_time[0]
            last_time[0] = now
            pct_str = f"{int(pct * 100):3d}%" if pct is not None else " ?? "
            print(f"[{int(now - start):4d}s] {pct_str}  +{delta:5.1f}s  {msg}", flush=True)

        try:
            result = ingest(source_str, progress=progress)
            print()
            print(f"Document: type={result.document.source_type} title={result.document.title!r}")
            print(f"Toplam metin: {len(result.document.text):,} karakter")
            print(f"Chunk sayısı: {result.chunk_count}")
            print(f"Ortalama chunk karakteri: {result.avg_chunk_chars:.1f}")
            print(f"Embedding shape: {result.embeddings.shape}, dtype={result.embeddings.dtype}")
            for kind, path in result.output_files.items():
                print(f"  {kind}: {path}")
            print(f"Toplam süre: {time.perf_counter() - start:.1f}s")
            return 0
        except Exception as e:
            print(f"\n[Hata] Tek kaynak işleme başarısız oldu: {e}", file=sys.stderr)
            return 1


if __name__ == "__main__":
    raise SystemExit(main())
