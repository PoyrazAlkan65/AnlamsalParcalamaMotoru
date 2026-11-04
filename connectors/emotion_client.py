"""NoemaVibeAPI nlp-service'teki avatar duygu modeline parça gönderen istemci.

Sunucu: NoemaVibeAPI/nlp-service/index.js (varsayılan http://127.0.0.1:31531; yalnızca
        127.0.0.1'i dinler, bu yüzden bu uygulama NoemaVibeAPI ile aynı sunucuda çalışmalıdır)
İstek:  POST /classifyEmotion  {"Text": "..."}   (parça başına bir istek)
Yanıt:  {"success": true, "data": {"emotion", "confidence", "scores": {etiket: olasılık}, "modelVersion"}}

Sonuçlar arayüzün beklediği biçime çevrilir:
    {"index", "id", "text", "emotion", "emotionTr", "confidence", "scores": [{"label", "p"}], "modelVersion"}
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import config

# NoemaVibeEmotionAI/config/labels.json içindeki "tr" alanları
EMOTION_TR = {
    "wave": "El salla",
    "clap": "Alkışla",
    "cheer": "Sevin",
    "bounce": "Zıpla",
    "dance": "Dans et",
    "laugh": "Gül",
    "nod": "Onayla",
    "bow": "Selam dur",
    "think": "Düşün",
    "sad": "Üzül",
    "none": "Hareket yok",
}


class EmotionServiceError(RuntimeError):
    pass


def _classify(text: str) -> dict:
    if not config.EMOTION_API_URL:
        raise EmotionServiceError("EMOTION_API_URL tanımlı değil; .env dosyasına ekleyin.")
    req = urllib.request.Request(
        f"{config.EMOTION_API_URL.rstrip('/')}/classifyEmotion",
        data=json.dumps({"Text": text}, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=config.EMOTION_API_TIMEOUT) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")
        raise EmotionServiceError(f"Duygu servisi {e.code} döndü: {detail}") from e
    except (urllib.error.URLError, TimeoutError) as e:
        raise EmotionServiceError(
            f"Duygu servisine bağlanılamadı ({config.EMOTION_API_URL}). "
            "NoemaVibeAPI nlp-service ayakta mı ve bu uygulama aynı sunucuda mı çalışıyor?"
        ) from e
    if not body.get("success") or not isinstance(body.get("data"), dict):
        raise EmotionServiceError(f"Duygu servisi beklenmeyen yanıt döndü: {body.get('message') or body}")
    return body["data"]


def analyze_chunks(texts: list[str], source: str | None = None, ids: list[str] | None = None) -> list[dict]:
    """Parçaların duygusunu ölçer; sonuçlar `texts` ile aynı sırada döner.

    `source` eski toplu API ile uyum için tutulur; nlp-service kullanmaz.
    """
    if not texts:
        return []
    ids = ids or [None] * len(texts)
    # Boş parçalar servise gönderilmez (400 döner), "none" sayılır
    to_send = [(i, t) for i, t in enumerate(texts) if t and t.strip()]
    raw: dict[int, dict] = {}
    with ThreadPoolExecutor(max_workers=max(1, config.EMOTION_API_CONCURRENCY)) as pool:
        for i, data in zip((i for i, _ in to_send), pool.map(_classify, (t for _, t in to_send))):
            raw[i] = data

    results = []
    for i, (text, cid) in enumerate(zip(texts, ids)):
        data = raw.get(i) or {"emotion": "none", "confidence": 0.0, "scores": {}, "modelVersion": None}
        emotion = data.get("emotion", "none")
        ranked = sorted((data.get("scores") or {}).items(), key=lambda kv: -kv[1])
        results.append({
            "index": i,
            "id": cid,
            "text": text,
            "emotion": emotion,
            "emotionTr": EMOTION_TR.get(emotion, emotion),
            "confidence": round(float(data.get("confidence", 0.0)), 4),
            "scores": [{"label": label, "p": round(float(p), 4)} for label, p in ranked],
            "modelVersion": data.get("modelVersion"),
        })
    return results
