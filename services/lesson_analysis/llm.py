"""Yapay zekâ çağrısı. Model, adres ve anahtar LMS'teki LLMModel/LLMProvider tablosundan okunur.

Admin'deki surveyGrowth.js askModel() ile aynı sağlayıcı kuralları (openai, gemini, anthropic, OpenAI uyumlu Jarvis).
"""
from __future__ import annotations

import json
import re

import requests

from . import db, settings


def ask(conn, prompt: str) -> tuple[str, str]:
    """(cevap metni, 'sağlayıcı/model') döndürür."""
    row = db.one(conn, "sp_getLLMModelConfig", ModelId=settings.LLM_MODEL_ID)
    if not row or not row.get("ModelName"):
        raise RuntimeError(f"Yapay zekâ modeli bulunamadı (LLMModel {settings.LLM_MODEL_ID}).")
    provider = str(row["ProviderName"]).lower()
    model = row["ModelName"]
    base_url = str(row.get("BaseUrl") or "").rstrip("/")
    api_key = row.get("ApiKey") or row.get("AuthToken")
    if not base_url or not api_key:
        raise RuntimeError(f"{row['ProviderName']}: adres veya anahtar tanımlı değil.")

    timeout = settings.LLM_TIMEOUT_SEC
    if provider == "gemini":
        res = requests.post(
            f"{base_url}/v1beta/models/{model}:generateContent",
            json={"contents": [{"role": "user", "parts": [{"text": prompt}]}],
                  "generationConfig": {"temperature": 0.2, "responseMimeType": "application/json"}},
            headers={"x-goog-api-key": api_key}, timeout=timeout)
        res.raise_for_status()
        parts = (res.json().get("candidates") or [{}])[0].get("content", {}).get("parts", [])
        text = "".join(p.get("text", "") for p in parts)
    elif provider == "anthropic":
        res = requests.post(
            f"{base_url}/v1/messages",
            json={"model": model, "max_tokens": 8000, "temperature": 0.2,
                  "messages": [{"role": "user", "content": prompt}]},
            headers={"x-api-key": api_key, "anthropic-version": "2023-06-01"}, timeout=timeout)
        res.raise_for_status()
        text = "".join(c.get("text", "") for c in res.json().get("content", []) if c.get("type") == "text")
    else:
        # openai ve Jarvis (vLLM): OpenAI uyumlu chat completions
        body = {"model": model, "messages": [{"role": "user", "content": prompt}],
                "response_format": {"type": "json_object"}}
        if provider == "openai":
            body["max_completion_tokens"] = 16000  # gpt-5 ailesi: düşünme + parça bazlı cevap
        else:
            body["temperature"] = 0.2
            body["max_tokens"] = 6000
        res = requests.post(f"{base_url}/v1/chat/completions", json=body,
                            headers={"Authorization": f"Bearer {api_key}"}, timeout=timeout)
        res.raise_for_status()
        text = ((res.json().get("choices") or [{}])[0].get("message") or {}).get("content") or ""

    if not text:
        raise RuntimeError("Yapay zekâ boş cevap döndü.")
    return text, f"{provider}/{model}"


def parse_json(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass  # model JSON etrafına metin eklemiş olabilir
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        raise RuntimeError("Yapay zekâ cevabında JSON bulunamadı.")
    return json.loads(m.group(0))
