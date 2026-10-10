"""services/lesson_analysis: normalleştirme ve parça zamanlama (veritabanı ve model gerektirmez)."""
import json

import pytest

from services.lesson_analysis import chunking, normalize

CODES = {"OZFARKINDALIK", "BEDENSEL_FARKINDALIK", "ENERJI_DAYANIKLILIK", "STRES_YONETIMI", "ANLAM"}


def test_lesson_groups_threshold_and_sum():
    parsed = {"groups": [
        {"code": "bedensel_farkindalik", "percent": 50, "reason": "a"},
        {"code": "ENERJI_DAYANIKLILIK", "percent": 30, "reason": "b"},
        {"code": "ANLAM", "percent": 3, "reason": "c"},          # eşik altı
        {"code": "BILINMEYEN", "percent": 10, "reason": "d"},     # tanımsız
        {"code": "STRES_YONETIMI", "percent": 17, "reason": "e"},
    ]}
    out = normalize.lesson_groups(parsed, CODES, 0.05)
    assert [g["code"] for g in out] == ["BEDENSEL_FARKINDALIK", "ENERJI_DAYANIKLILIK", "STRES_YONETIMI"]
    assert round(sum(g["ratio"] for g in out), 4) == 1


def test_lesson_groups_empty_raises():
    with pytest.raises(normalize.AnalysisError):
        normalize.lesson_groups({"groups": [{"code": "X", "percent": 100}]}, CODES, 0.05)


def test_chunk_scores_filters():
    parsed = {"chunks": [
        {"seq": 1, "groups": [{"code": "ANLAM", "score": 1.4, "evidence": "e"}, {"code": "X", "score": 0.5}]},
        {"seq": 9, "groups": [{"code": "ANLAM", "score": 0.5}]},  # olmayan parça
        {"seq": 2, "groups": [{"code": "ANLAM", "score": 0.2}, {"code": "OZFARKINDALIK", "score": 0.9},
                              {"code": "STRES_YONETIMI", "score": 0.4}, {"code": "BEDENSEL_FARKINDALIK", "score": 0.1}]},
    ]}
    out = normalize.chunk_scores(parsed, CODES, {1, 2}, 3)
    assert out[0] == {"seq": 1, "groups": [{"code": "ANLAM", "score": 1.0, "evidence": "e"}]}
    assert [g["code"] for g in out[1]["groups"]] == ["OZFARKINDALIK", "STRES_YONETIMI", "ANLAM"]


def test_merge_tiny():
    text = "Kısa soru?\nUzun bir açıklama cümlesi burada yer alıyor.\nBu ne demek?\nSon uzun parça burada bitiyor."
    parts, pos = [], 0
    for i, line in enumerate(text.split("\n"), start=1):
        parts.append(chunking.TimedChunk(i, line, pos, pos + len(line), 2 if line.endswith("?") else 10))
        pos += len(line) + 1
    out = chunking._merge_tiny(parts, text, 5)
    assert [c.seq for c in out] == [1, 2]
    assert out[0].text == "Kısa soru?\nUzun bir açıklama cümlesi burada yer alıyor.\nBu ne demek?"
    assert all(text[c.char_start:c.char_end] == c.text for c in out)


def test_lines_chunker_and_proportional_times(monkeypatch):
    monkeypatch.setattr(chunking.settings, "CHUNK_MIN_TOKENS", 1)
    monkeypatch.setattr(chunking.settings, "CHUNKER", "lines")
    monkeypatch.setattr(chunking.settings, "CHUNKER_MAX_CHARS", 25)
    text = "birinci satır burada\nikinci satır burada\nüçüncü satır"
    chunks = chunking.split(text, None, 60.0)
    assert len(chunks) == 3
    assert chunks[0].start == 0 and chunks[-1].end == 60.0
    assert all(text[c.char_start:c.char_end] == c.text for c in chunks)


def test_segment_times(monkeypatch):
    monkeypatch.setattr(chunking.settings, "CHUNK_MIN_TOKENS", 1)
    monkeypatch.setattr(chunking.settings, "CHUNKER", "lines")
    monkeypatch.setattr(chunking.settings, "CHUNKER_MAX_CHARS", 20)
    segs = [{"baslangic": 0, "bitis": 4, "metin": "ilk cümle burada"},
            {"baslangic": 4, "bitis": 9, "metin": "ikinci cümle de"},
            {"baslangic": 9, "bitis": 15, "metin": "son cümle"}]
    text = "\n".join(s["metin"] for s in segs)
    chunks = chunking.split(text, json.dumps(segs), None)
    assert [(c.start, c.end) for c in chunks] == [(0, 4), (4, 9), (9, 15)]
