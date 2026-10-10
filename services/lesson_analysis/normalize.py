"""Yapay zekâ cevabını sp_SetLessonAnalysis biçimine getirir.

Ders dağılımı: bilinmeyen kod atılır, aynı kod birleşir, MIN_RATIO altı atılır, kalan toplam 1'e ölçeklenir
(4 hane, yuvarlama farkı en büyük köşeye). Parça skorları: bilinmeyen parça/kod atılır, 0-1 aralığına kırpılır.
"""
from __future__ import annotations


class AnalysisError(RuntimeError):
    pass


def lesson_groups(parsed: dict, known_codes: set[str], min_ratio: float) -> list[dict]:
    merged: dict[str, dict] = {}
    for g in parsed.get("groups") or []:
        code = str(g.get("code") or "").strip().upper()
        try:
            pct = float(g.get("percent") if g.get("percent") is not None else float(g.get("ratio")) * 100)
        except (TypeError, ValueError):
            continue
        if code not in known_codes or not pct > 0:
            continue
        prev = merged.get(code)
        merged[code] = {"code": code, "pct": (prev["pct"] if prev else 0) + pct,
                        "reason": (prev["reason"] if prev else str(g.get("reason") or ""))[:1000]}

    total = sum(g["pct"] for g in merged.values())
    if not total:
        raise AnalysisError("Yapay zekâ geçerli bir köşe dağılımı döndürmedi.")

    kept = [g for g in merged.values() if g["pct"] / total >= min_ratio]
    if not kept:  # eşik her şeyi eledi: en büyüğü tut
        kept = [max(merged.values(), key=lambda g: g["pct"])]
    kept.sort(key=lambda g: (-g["pct"], g["code"]))

    kept_total = sum(g["pct"] for g in kept)
    out = [{"code": g["code"], "ratio": round(g["pct"] / kept_total, 4), "reason": g["reason"]} for g in kept]
    out[0]["ratio"] = round(out[0]["ratio"] + round(1 - sum(g["ratio"] for g in out), 4), 4)
    return out


def chunk_scores(parsed: dict, known_codes: set[str], seqs: set[int], max_groups: int) -> list[dict]:
    out = []
    for c in parsed.get("chunks") or []:
        try:
            seq = int(c.get("seq"))
        except (TypeError, ValueError):
            continue
        if seq not in seqs:
            continue
        best: dict[str, dict] = {}
        for g in c.get("groups") or []:
            code = str(g.get("code") or "").strip().upper()
            try:
                score = min(1.0, float(g.get("score")))
            except (TypeError, ValueError):
                continue
            if code not in known_codes or score <= 0:
                continue
            if code not in best or score > best[code]["score"]:
                best[code] = {"code": code, "score": round(score, 4), "evidence": str(g.get("evidence") or "")[:500]}
        groups = sorted(best.values(), key=lambda g: -g["score"])[:max_groups]
        if groups:
            out.append({"seq": seq, "groups": groups})
    return out
