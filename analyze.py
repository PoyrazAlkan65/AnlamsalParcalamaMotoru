"""
Whisper Benchmark - v2 (30 test) analiz ve grafik üretim scripti
==================================================================
Kullanım:
    python3 analyze.py

Girdi dosyaları (aynı klasörde olmalı):
    toplu_30_ses_analizi_2.jsonl   -> ikinci (transkriptli) çalıştırmanın çıktısı
    toplu_30_ses_analizi_1.jsonl   -> ilk çalıştırmanın çıktısı, run karşılaştırması için

Çıktılar:
    charts/*.png      -> tüm grafikler
    stats_summary.json -> tüm sayısal özet (rapora kopyalamak için)
    sentence_counts.csv -> her test/model için cümle sayısı tablosu
"""
import json
import re
import statistics as st
from collections import defaultdict
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# 0) AYARLAR
# ----------------------------------------------------------------------
RUN2_PATH = "data/outputs/test/toplu_30_ses_analizi_2.jsonl"
RUN1_PATH = "data/outputs/test/toplu_30_ses_analizi_1.jsonl"
OUT_DIR = "charts"
MODELS = ["turbo", "large-v3", "medium"]
COLORS = {"turbo": "#2E86AB", "large-v3": "#A23B72", "medium": "#F18F01"}
os.makedirs(OUT_DIR, exist_ok=True)

# ----------------------------------------------------------------------
# 1) VERİ YÜKLEME
# ----------------------------------------------------------------------
def load_jsonl(path):
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


run2 = load_jsonl(RUN2_PATH)
run1 = load_jsonl(RUN1_PATH)

assert run2, f"{RUN2_PATH} bulunamadı ya da boş!"

by_model_run2 = defaultdict(list)
for r in run2:
    by_model_run2[r["model"]].append(r)

tests = sorted(
    set(r["audio_file"] for r in run2),
    key=lambda x: int(re.search(r"Test (\d+)", x).group(1)),
)


def get(rows, test, model, field):
    for r in rows:
        if r["audio_file"] == test and r["model"] == model:
            return r.get(field)
    return None


# ----------------------------------------------------------------------
# 2) CÜMLE SAYISI HESABI
# ----------------------------------------------------------------------
def count_sentences(text):
    """Basit cümle sayaç: '.', '!', '?' ile biten parçaları sayar.
    Ondalık sayıları (2,5 / 2.5) ve kısaltmaları (Dr., vb.) yanlış
    bölmemek için birkaç basit kural uygular."""
    if not text:
        return 0
    t = text.replace("\\n", " ").replace("\n", " ")
    t = re.sub(r"(?<=\d)[.,](?=\d)", "§", t)
    for abbr in ["Dr.", "M.K.", "vb.", "sn.", "Bkz."]:
        t = t.replace(abbr, abbr.replace(".", "§"))
    parts = re.split(r"[.!?]+", t)
    parts = [p.strip() for p in parts if p.strip()]
    return len(parts)


sentence_rows = []
for t in tests:
    row = {"test": t}
    for m in MODELS:
        text = get(run2, t, m, "text") or ""
        row[m] = count_sentences(text)
    sentence_rows.append(row)

with open("sentence_counts.csv", "w", encoding="utf-8") as f:
    f.write("test," + ",".join(MODELS) + "\n")
    for row in sentence_rows:
        f.write(f'"{row["test"]}",' + ",".join(str(row[m]) for m in MODELS) + "\n")

print("sentence_counts.csv yazildi.")

# ----------------------------------------------------------------------
# 3) GENEL İSTATİSTİKLER (run2)
# ----------------------------------------------------------------------
summary = {"run2": {}, "run1_vs_run2": {}}

for m in MODELS:
    rows = by_model_run2[m]
    wer = [r["wer"] for r in rows]
    rtf = [r["rtf"] for r in rows]
    load = [r["load_time_sec"] for r in rows]
    trans = [r["transcribe_time_sec"] for r in rows]
    cer = [r.get("cer") for r in rows if r.get("cer") is not None]

    summary["run2"][m] = {
        "wer_mean": round(st.mean(wer), 4),
        "wer_median": round(st.median(wer), 4),
        "wer_std": round(st.pstdev(wer), 4),
        "wer_min": round(min(wer), 4),
        "wer_max": round(max(wer), 4),
        "rtf_mean": round(st.mean(rtf), 4),
        "rtf_median": round(st.median(rtf), 4),
        "rtf_min": round(min(rtf), 4),
        "rtf_max": round(max(rtf), 4),
        "load_mean": round(st.mean(load), 2),
        "trans_mean": round(st.mean(trans), 2),
        "trans_total_sec": round(sum(trans), 1),
        "cer_mean": round(st.mean(cer), 4) if cer else None,
    }

outlier_tests = {"Test 12", "Test 17"}
rtf_lv3_excl = [
    r["rtf"] for r in by_model_run2["large-v3"]
    if not any(r["audio_file"].startswith(ot) for ot in outlier_tests)
]
summary["run2"]["large-v3"]["rtf_mean_excl_outliers"] = round(st.mean(rtf_lv3_excl), 4)

wins_wer = defaultdict(int)
wins_rtf = defaultdict(int)
for t in tests:
    trs = [r for r in run2 if r["audio_file"] == t]
    best_wer = min(trs, key=lambda r: r["wer"])
    best_rtf = min(trs, key=lambda r: r["rtf"])
    wins_wer[best_wer["model"]] += 1
    wins_rtf[best_rtf["model"]] += 1

summary["run2"]["wer_wins"] = dict(wins_wer)
summary["run2"]["rtf_wins"] = dict(wins_rtf)

avg_sentences = {}
for m in MODELS:
    vals = [row[m] for row in sentence_rows]
    avg_sentences[m] = round(st.mean(vals), 2)
summary["run2"]["avg_sentence_count"] = avg_sentences

# ----------------------------------------------------------------------
# 4) RUN1 vs RUN2 KARŞILAŞTIRMASI (varsa)
# ----------------------------------------------------------------------
if run1:
    by_model_run1 = defaultdict(list)
    for r in run1:
        by_model_run1[r["model"]].append(r)

    deltas = []
    for t in tests:
        for m in MODELS:
            wer1 = get(run1, t, m, "wer")
            wer2 = get(run2, t, m, "wer")
            rtf1 = get(run1, t, m, "rtf")
            rtf2 = get(run2, t, m, "rtf")
            if wer1 is None or wer2 is None:
                continue
            deltas.append({
                "test": t, "model": m,
                "wer_run1": wer1, "wer_run2": wer2, "d_wer": round(wer2 - wer1, 4),
                "rtf_run1": rtf1, "rtf_run2": rtf2, "d_rtf": round(rtf2 - rtf1, 4) if (rtf1 and rtf2) else None,
            })

    WER_THRESHOLD = 0.05
    RTF_THRESHOLD = 0.5
    big_wer_changes = [d for d in deltas if abs(d["d_wer"]) >= WER_THRESHOLD]
    big_rtf_changes = [d for d in deltas if d["d_rtf"] is not None and abs(d["d_rtf"]) >= RTF_THRESHOLD]

    summary["run1_vs_run2"]["big_wer_changes"] = big_wer_changes
    summary["run1_vs_run2"]["big_rtf_changes"] = big_rtf_changes

    for m in MODELS:
        rows1 = by_model_run1[m]
        rows2 = by_model_run2[m]
        if not rows1:
            continue
        wer1_mean = st.mean([r["wer"] for r in rows1])
        wer2_mean = st.mean([r["wer"] for r in rows2])
        rtf1_mean = st.mean([r["rtf"] for r in rows1])
        rtf2_mean = st.mean([r["rtf"] for r in rows2])
        summary["run1_vs_run2"][m] = {
            "wer_run1_mean": round(wer1_mean, 4),
            "wer_run2_mean": round(wer2_mean, 4),
            "wer_diff": round(wer2_mean - wer1_mean, 4),
            "rtf_run1_mean": round(rtf1_mean, 4),
            "rtf_run2_mean": round(rtf2_mean, 4),
            "rtf_diff": round(rtf2_mean - rtf1_mean, 4),
        }

    print(f"Run1 vs Run2: {len(big_wer_changes)} testte WER farki >= {WER_THRESHOLD}, "
          f"{len(big_rtf_changes)} testte RTF farki >= {RTF_THRESHOLD}")
else:
    print(f"UYARI: {RUN1_PATH} bulunamadi, run1 vs run2 karsilastirmasi atlaniyor. "
          f"Ilk calistirmanin verisini ayni klasore '{RUN1_PATH}' adiyla koyarsan otomatik karsilastirir.")

with open("stats_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print("stats_summary.json yazildi.")

# ----------------------------------------------------------------------
# 5) GRAFİKLER
# ----------------------------------------------------------------------
def savefig(name):
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, name), dpi=150)
    plt.close()


fig, ax = plt.subplots(figsize=(7, 4.2))
means = [summary["run2"][m]["wer_mean"] for m in MODELS]
stds = [summary["run2"][m]["wer_std"] for m in MODELS]
bars = ax.bar(MODELS, means, yerr=stds, capsize=6, color=[COLORS[m] for m in MODELS], width=0.5)
for b, v in zip(bars, means):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center", fontweight="bold")
ax.set_ylabel("Ortalama WER")
ax.set_title("30 Test Ortalamasi — Model Basina WER (± std sapma)")
ax.spines[["top", "right"]].set_visible(False)
savefig("01_wer_ortalama.png")

fig, ax = plt.subplots(figsize=(7.5, 4.2))
rtf_all = [summary["run2"][m]["rtf_mean"] for m in MODELS]
rtf_lv3_excl_val = summary["run2"]["large-v3"]["rtf_mean_excl_outliers"]
labels = ["turbo", "large-v3\n(tum testler)", "large-v3\n(Test12,17 haric)", "medium"]
vals = [rtf_all[0], rtf_all[1], rtf_lv3_excl_val, rtf_all[2]]
cols = [COLORS["turbo"], COLORS["large-v3"], "#C77DA0", COLORS["medium"]]
bars = ax.bar(labels, vals, color=cols, width=0.55)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.03, f"{v:.2f}", ha="center", fontweight="bold")
ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, label="Gercek zamanli sinir (RTF=1)")
ax.set_ylabel("Ortalama RTF")
ax.set_title("30 Test Ortalamasi — Model Basina RTF")
ax.legend(fontsize=8)
ax.spines[["top", "right"]].set_visible(False)
savefig("02_rtf_ortalama.png")

fig, ax = plt.subplots(figsize=(7, 4.2))
data = [[r["wer"] for r in by_model_run2[m]] for m in MODELS]
bp = ax.boxplot(data, tick_labels=MODELS, patch_artist=True, widths=0.5)
for patch, m in zip(bp["boxes"], MODELS):
    patch.set_facecolor(COLORS[m])
    patch.set_alpha(0.7)
ax.set_ylabel("WER")
ax.set_title("30 Test Genelinde WER Dagilimi")
ax.spines[["top", "right"]].set_visible(False)
savefig("03_wer_dagilim_boxplot.png")

fig, ax = plt.subplots(figsize=(7, 4.2))
data = [[r["rtf"] for r in by_model_run2[m]] for m in MODELS]
bp = ax.boxplot(data, tick_labels=MODELS, patch_artist=True, widths=0.5)
for patch, m in zip(bp["boxes"], MODELS):
    patch.set_facecolor(COLORS[m])
    patch.set_alpha(0.7)
ax.set_ylabel("RTF")
ax.set_title("30 Test Genelinde RTF Dagilimi")
ax.spines[["top", "right"]].set_visible(False)
savefig("04_rtf_dagilim_boxplot.png")

fig, ax = plt.subplots(figsize=(6, 4))
w_vals = [summary["run2"]["wer_wins"].get(m, 0) for m in MODELS]
bars = ax.bar(MODELS, w_vals, color=[COLORS[m] for m in MODELS], width=0.5)
for b, v in zip(bars, w_vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.3, f"{v}/30", ha="center", fontweight="bold")
ax.set_ylabel("En dusuk WER verdigi test sayisi")
ax.set_title("30 Testten Kacinda Hangi Model En Dogru Sonucu Verdi?")
ax.set_ylim(0, 30)
ax.spines[["top", "right"]].set_visible(False)
savefig("05_wer_kazanan_sayaci.png")

fig, ax = plt.subplots(figsize=(6, 4))
r_vals = [summary["run2"]["rtf_wins"].get(m, 0) for m in MODELS]
bars = ax.bar(MODELS, r_vals, color=[COLORS[m] for m in MODELS], width=0.5)
for b, v in zip(bars, r_vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.3, f"{v}/30", ha="center", fontweight="bold")
ax.set_ylabel("En dusuk RTF verdigi test sayisi")
ax.set_title("30 Testten Kacinda Hangi Model En Hizli Oldu?")
ax.set_ylim(0, 30)
ax.spines[["top", "right"]].set_visible(False)
savefig("06_rtf_kazanan_sayaci.png")

fig, ax = plt.subplots(figsize=(12, 5))
x = list(range(1, len(tests) + 1))
for m in MODELS:
    y = [get(run2, t, m, "wer") for t in tests]
    ax.plot(x, y, marker="o", label=m, color=COLORS[m], linewidth=1.5, markersize=4)
ax.set_xticks(x)
ax.set_xticklabels([str(i) for i in x], fontsize=8)
ax.set_xlabel("Test No")
ax.set_ylabel("WER")
ax.set_title("Test Bazinda WER Karsilastirmasi (30 Test)")
ax.legend()
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", alpha=0.3)
savefig("07_test_bazinda_wer_cizgi.png")

fig, ax = plt.subplots(figsize=(12, 5))
for m in MODELS:
    y = [get(run2, t, m, "rtf") for t in tests]
    ax.plot(x, y, marker="o", label=m, color=COLORS[m], linewidth=1.5, markersize=4)
ax.set_xticks(x)
ax.set_xticklabels([str(i) for i in x], fontsize=8)
ax.set_xlabel("Test No")
ax.set_ylabel("RTF")
ax.set_yscale("log")
ax.set_title("Test Bazinda RTF Karsilastirmasi (30 Test, log olcek)")
ax.legend()
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", alpha=0.3, which="both")
savefig("08_test_bazinda_rtf_cizgi.png")

fig, ax = plt.subplots(figsize=(6, 4))
avg_vals = [avg_sentences[m] for m in MODELS]
bars = ax.bar(MODELS, avg_vals, color=[COLORS[m] for m in MODELS], width=0.5)
for b, v in zip(bars, avg_vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.3, f"{v:.1f}", ha="center", fontweight="bold")
ax.set_ylabel("Ortalama cumle sayisi (30 test)")
ax.set_title("Model Basina Ortalama Uretilen Cumle Sayisi")
ax.spines[["top", "right"]].set_visible(False)
savefig("09_ortalama_cumle_sayisi.png")

fig, ax = plt.subplots(figsize=(14, 5))
xpos = range(len(tests))
width = 0.25
for i, m in enumerate(MODELS):
    y = [row[m] for row in sentence_rows]
    ax.bar([p + i * width for p in xpos], y, width=width, label=m, color=COLORS[m])
ax.set_xticks([p + width for p in xpos])
ax.set_xticklabels([str(i + 1) for i in xpos], fontsize=8)
ax.set_xlabel("Test No")
ax.set_ylabel("Cumle sayisi")
ax.set_title("Test Bazinda Uretilen Cumle Sayisi (3 Model)")
ax.legend()
ax.spines[["top", "right"]].set_visible(False)
savefig("10_test_bazinda_cumle_sayisi.png")

fig, ax = plt.subplots(figsize=(6, 4))
load_vals = [summary["run2"][m]["load_mean"] for m in MODELS]
bars = ax.bar(MODELS, load_vals, color=[COLORS[m] for m in MODELS], width=0.5)
for b, v in zip(bars, load_vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.3, f"{v:.1f}s", ha="center", fontweight="bold")
ax.set_ylabel("Ortalama yukleme suresi (sn)")
ax.set_title("Model Basina Ortalama Yukleme (Load) Suresi")
ax.spines[["top", "right"]].set_visible(False)
savefig("11_ortalama_yukleme_suresi.png")

if run1:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = range(len(MODELS))
    w1 = [summary["run1_vs_run2"][m]["wer_run1_mean"] for m in MODELS if m in summary["run1_vs_run2"]]
    w2 = [summary["run1_vs_run2"][m]["wer_run2_mean"] for m in MODELS if m in summary["run1_vs_run2"]]
    width = 0.35
    ax.bar([p - width / 2 for p in x], w1, width=width, label="1. calistirma", color="#B0B0B0")
    ax.bar([p + width / 2 for p in x], w2, width=width, label="2. calistirma", color=[COLORS[m] for m in MODELS])
    ax.set_xticks(list(x))
    ax.set_xticklabels(MODELS)
    ax.set_ylabel("Ortalama WER")
    ax.set_title("1. ve 2. Calistirma Arasinda Ortalama WER Karsilastirmasi")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    savefig("12_run1_vs_run2_wer.png")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    r1 = [summary["run1_vs_run2"][m]["rtf_run1_mean"] for m in MODELS if m in summary["run1_vs_run2"]]
    r2 = [summary["run1_vs_run2"][m]["rtf_run2_mean"] for m in MODELS if m in summary["run1_vs_run2"]]
    ax.bar([p - width / 2 for p in x], r1, width=width, label="1. calistirma", color="#B0B0B0")
    ax.bar([p + width / 2 for p in x], r2, width=width, label="2. calistirma", color=[COLORS[m] for m in MODELS])
    ax.set_xticks(list(x))
    ax.set_xticklabels(MODELS)
    ax.set_ylabel("Ortalama RTF")
    ax.set_title("1. ve 2. Calistirma Arasinda Ortalama RTF Karsilastirmasi")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    savefig("13_run1_vs_run2_rtf.png")

print(f"\nTum grafikler '{OUT_DIR}/' klasorune yazildi.")
print("Uretilen dosyalar:")
for fn in sorted(os.listdir(OUT_DIR)):
    print("  -", fn)