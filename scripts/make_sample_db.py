"""Veritabanı parçalamayı denemek için örnek SQLite veritabanı üretir: data/inputs/ornek.db

  python scripts/make_sample_db.py
  .env → SOURCE_DB_URL=sqlite:///data/inputs/ornek.db
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "inputs" / "ornek.db"

ARTICLES = [
    (1, "İyi Bir Uykunun Sırları", "Ayşe Yılmaz", "2026-09-01", 1, """
<h2>Uyku neden önemli?</h2>
<p>Uyku, vücudun kendini onardığı en önemli süreçtir. Yetişkinlerin her gece <strong>yedi ile dokuz saat</strong>
arasında uyuması önerilir. Uyku sırasında beyin gün içinde öğrenilen bilgileri düzenler ve kalıcı hafızaya aktarır.</p>
<h2>Uyku düzeni nasıl kurulur?</h2>
<p>Her gün aynı saatte yatıp aynı saatte kalkmak gerekir. Yatmadan önceki bir saatte ekranlardan uzak durmak,
mavi ışığın melatonin salgısını baskılamasını önler.</p>
<ul><li>Yatak odası serin olmalı.</li><li>Oda karanlık ve sessiz olmalı.</li><li>Akşam kafeinden kaçınılmalı.</li></ul>
"""),
    (2, "Dengeli Beslenmenin Temelleri", "Mehmet Kaya", "2026-09-05", 1, """
<p>Beslenmede en temel kural tabağı renklendirmektir. Farklı renklerdeki sebze ve meyveler, vücudun ihtiyaç duyduğu
farklı vitamin ve mineralleri sağlar.</p>
<p>Günde ortalama iki litre su içmek sindirimi kolaylaştırır ve baş ağrılarını azaltır. Şekerli içecekler yerine su
ya da şekersiz bitki çayları tercih edilmelidir.</p>
"""),
    (3, "Pomodoro ile Odaklanma", "Ayşe Yılmaz", "2026-09-12", 1, """
<p>Pomodoro tekniğinde yirmi beş dakika kesintisiz çalışılır, ardından beş dakika mola verilir. Dört turdan sonra
daha uzun bir mola alınır.</p><p>Bu yöntem odaklanmayı artırır ve zihinsel yorgunluğu geciktirir. Çalışma sırasında
bildirimlerin kapatılması tekniğin verimini belirgin şekilde yükseltir.</p>
"""),
    (4, "Eski Kampanya Duyurusu", "Yönetici", "2025-01-10", 0, "<p>Bu duyuru yayından kaldırıldı.</p>"),
    (5, "Taslak Yazı", "Mehmet Kaya", "2026-10-01", 1, ""),
    (6, "Stresle Başa Çıkmak", "Zeynep Demir", "2026-10-15", 1,
     "Nefes egzersizleri stresle başa çıkmanın en hızlı yollarından biridir. Dört saniye burundan nefes alıp yedi "
     "saniye tutmak ve sekiz saniyede ağızdan vermek sinir sistemini sakinleştirir.\n\nDüzenli yapılan nefes "
     "çalışmaları kaygı düzeyini düşürür ve uykuya dalmayı kolaylaştırır."),
]

FAQ = [
    (1, "Üyeliğimi nasıl iptal ederim?", "Hesap ayarları sayfasındaki Üyelik bölümünden iptal edebilirsiniz."),
    (2, "Ödeme yöntemleri nelerdir?", "Kredi kartı, banka kartı ve havale ile ödeme yapabilirsiniz."),
    (3, "Videoları indirebilir miyim?", "Videolar yalnızca uygulama içinden çevrimdışı izlenebilir."),
]


def main() -> int:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    DB_PATH.unlink(missing_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.executescript("""
            CREATE TABLE Makaleler (
                Id INTEGER PRIMARY KEY, Baslik TEXT NOT NULL, Yazar TEXT, YayinTarihi TEXT,
                Aktif INTEGER NOT NULL DEFAULT 1, Icerik TEXT
            );
            CREATE TABLE SSS (Id INTEGER PRIMARY KEY, Soru TEXT NOT NULL, Cevap TEXT NOT NULL);
        """)
        conn.executemany("INSERT INTO Makaleler VALUES (?, ?, ?, ?, ?, ?)",
                         [(i, b, y, t, a, c.strip()) for i, b, y, t, a, c in ARTICLES])
        conn.executemany("INSERT INTO SSS VALUES (?, ?, ?)", FAQ)
    print(f"Oluşturuldu: {DB_PATH}")
    print("  .env → SOURCE_DB_URL=sqlite:///data/inputs/ornek.db")
    return 0


if __name__ == "__main__":
    sys.exit(main())
