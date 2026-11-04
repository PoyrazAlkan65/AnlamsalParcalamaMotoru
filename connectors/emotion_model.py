"""NoemaVibeEmotionAI avatar duygu modelinin yerel (servissiz) çalıştırıcısı.

Model, NoemaVibeEmotionAI/scripts/export_model.py'nin ürettiği tek JSON dosyasıdır
(format "noema-emotion-tfidf-lr": karakter n-gram TF-IDF + lojistik regresyon). Bağımlılık yoktur.

Referanslar (değişirlerse tests/test_emotion_model.py yeniden çalıştırılmalı):
    NoemaVibeEmotionAI/src/emotion/normalize.py        normalize()
    NoemaVibeEmotionAI/scripts/export_model.py         char_wb_ngrams(), predict_from_json()
    NoemaVibeEmotionAI/js/emotionModel.js              nlp-service'teki Node karşılığı

Yeni model: NoemaVibeEmotionAI/models/<deney>/ içindeki emotion-model.json ve golden.json
dosyalarını emotion_model/ klasörüne kopyalayın.
"""
from __future__ import annotations

import json
import math
import re
import threading
from pathlib import Path

FORMAT = "noema-emotion-tfidf-lr"

_DIACRITICS = str.maketrans({
    "ı": "i", "ğ": "g", "ü": "u", "ş": "s", "ö": "o", "ç": "c",
    "â": "a", "î": "i", "û": "u",
})
_URL = re.compile(r"https?://\S+|www\.\S+")
_APOSTROPHE = re.compile(r"[’'`´]")
_REPEAT = re.compile(r"(.)\1{2,}")  # "harikaaaa" -> "harikaa"
_SPACE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Eğitimdeki normalize() ile birebir aynı."""
    if not isinstance(text, str):
        return ""
    s = text.replace("İ", "i").replace("I", "ı").lower()  # Türkçe küçük harf
    s = _URL.sub(" ", s)
    s = _APOSTROPHE.sub("", s)
    s = s.translate(_DIACRITICS)
    s = _REPEAT.sub(r"\1\1", s)
    return _SPACE.sub(" ", s).strip()


def char_wb_ngrams(text: str, n_min: int, n_max: int) -> list[str]:
    """sklearn TfidfVectorizer(analyzer='char_wb') ile aynı n-gram üretimi."""
    grams = []
    for w in text.split():
        w = f" {w} "
        for n in range(n_min, n_max + 1):
            offset = 0
            grams.append(w[offset:offset + n])
            while offset + n < len(w):
                offset += 1
                grams.append(w[offset:offset + n])
            if offset == 0:  # n'den kısa kelime yalnızca bir kez sayılır
                break
    return grams


def load_model(path: str | Path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Duygu modeli bulunamadı: {path}")
    model = json.loads(path.read_text(encoding="utf-8"))
    if model.get("format") != FORMAT:
        raise ValueError(f"Beklenmeyen duygu modeli formatı: {model.get('format')!r} ({path})")
    return model


def classify(model: dict, text: str) -> dict:
    """nlp-service /classifyEmotion yanıtındaki `data` ile aynı biçim:
    {"emotion", "confidence", "scores": {etiket: olasılık}, "modelVersion"}"""
    n_min, n_max = model["features"]["ngramRange"]
    vocab, idf = model["vocab"], model["idf"]
    counts: dict[int, int] = {}
    for g in char_wb_ngrams(normalize(text), n_min, n_max):
        idx = vocab.get(g)
        if idx is not None:
            counts[idx] = counts.get(idx, 0) + 1
    vec = {i: (1 + math.log(c)) * idf[i] for i, c in counts.items()}  # sublinear tf * idf
    norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0

    labels = model["labels"]
    z = list(model["bias"])
    weights = model["weights"]
    for i, v in vec.items():
        w = weights[i]
        for k in range(len(labels)):
            z[k] += w[k] * v / norm
    m = max(z)
    e = [math.exp(x - m) for x in z]
    total = sum(e)
    scores = {lab: e[k] / total for k, lab in enumerate(labels)}

    best = labels[0]
    for lab in labels:
        if scores[lab] > scores[best]:
            best = lab
    confidence = scores[best]
    return {
        "emotion": "none" if confidence < model["noneThreshold"] else best,
        "confidence": confidence,
        "scores": scores,
        "modelVersion": model.get("modelVersion"),
    }


_cache: dict[Path, dict] = {}
_lock = threading.Lock()


def get_model(path: str | Path) -> dict:
    """Modeli bir kez yükler, sonraki çağrılarda bellekten döner."""
    key = Path(path).resolve()
    with _lock:
        if key not in _cache:
            _cache[key] = load_model(key)
        return _cache[key]
