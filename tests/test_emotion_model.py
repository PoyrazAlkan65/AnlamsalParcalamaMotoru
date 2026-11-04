"""Yerel duygu modeli, NoemaVibeEmotionAI'nin altın tahminleriyle (golden.json) birebir aynı olmalı.

golden.json, eğitim projesinde orijinal sklearn modelinden üretilir; Node tarafı (nlp-service) da
aynı dosyayla doğrulanır. Model değiştiğinde emotion-model.json ile birlikte kopyalanmalıdır.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from connectors import emotion_model  # noqa: E402

MODEL_DIR = ROOT / "emotion_model"
TOL = 1e-6


@pytest.fixture(scope="module")
def model():
    return emotion_model.load_model(MODEL_DIR / "emotion-model.json")


@pytest.fixture(scope="module")
def golden():
    return json.loads((MODEL_DIR / "golden.json").read_text(encoding="utf-8"))


def test_golden_version_matches(model, golden):
    assert golden["modelVersion"] == model["modelVersion"]


def test_matches_golden_predictions(model, golden):
    fails = []
    for case in golden["cases"]:
        r = emotion_model.classify(model, case["text"])
        diff = max(abs(r["scores"][k] - p) for k, p in case["scores"].items())
        if r["emotion"] != case["emotion"] or diff > TOL:
            fails.append((case["text"], r["emotion"], case["emotion"], diff))
    assert not fails, f"{len(fails)} fark, ilkleri: {fails[:5]}"


def test_output_shape(model):
    r = emotion_model.classify(model, "Tebrikler, sınavı geçtin!")
    assert set(r) == {"emotion", "confidence", "scores", "modelVersion"}
    assert set(r["scores"]) == set(model["labels"])
    assert abs(sum(r["scores"].values()) - 1) < 1e-9
    assert r["confidence"] == max(r["scores"].values())


def test_normalize_turkish():
    assert emotion_model.normalize("İSTANBUL'DA IŞIK") == "istanbulda isik"
    assert emotion_model.normalize("HARİKAAAA!!!") == "harikaa!!"


def test_missing_model_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        emotion_model.load_model(tmp_path / "yok.json")


def test_client_local_backend(monkeypatch):
    import config
    from connectors import emotion_client

    monkeypatch.setattr(config, "EMOTION_BACKEND", "local")
    monkeypatch.setattr(config, "EMOTION_MODEL_PATH", MODEL_DIR / "emotion-model.json")
    out = emotion_client.analyze_chunks(["Merhaba!", "  ", "Bunu düşünelim."], ids=["a", "b", "c"])
    assert [o["id"] for o in out] == ["a", "b", "c"]
    assert out[1]["emotion"] == "none" and out[1]["confidence"] == 0.0
    assert out[0]["modelVersion"] == out[2]["modelVersion"] is not None
    assert out[0]["scores"][0]["p"] >= out[0]["scores"][-1]["p"]
