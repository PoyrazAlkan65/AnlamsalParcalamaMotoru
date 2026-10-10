"""Ders videosunu (Bunny mp4 adresi) motorun VideoLoader'ı ile transkripte çevirir."""
from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass
class Transcript:
    text: str
    segments_json: str
    language: str | None
    duration: float | None
    model: str


def from_video(url: str) -> Transcript:
    import config  # motor ayarları
    from loaders.video_loader import VideoLoader

    doc = VideoLoader().load(url)
    segments = [{"baslangic": round(s.start, 2), "bitis": round(s.end, 2), "metin": s.text} for s in doc.segments]
    return Transcript(
        text=doc.text.strip(),
        segments_json=json.dumps(segments, ensure_ascii=False),
        language=doc.language,
        duration=segments[-1]["bitis"] if segments else None,
        model=f"faster-whisper/{config.WHISPER_MODEL}",
    )
