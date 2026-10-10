"""FFmpeg discovery shared by audio/video loaders."""
from __future__ import annotations

import os
import shutil
from pathlib import Path


def find_ffmpeg() -> str | None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg

    local_app_data = os.environ.get("LOCALAPPDATA")
    if not local_app_data:
        return None

    packages = Path(local_app_data) / "Microsoft" / "WinGet" / "Packages"
    candidates = sorted(
        packages.glob("Gyan.FFmpeg_*/ffmpeg-*/bin/ffmpeg.exe"),
        reverse=True,
    )
    for candidate in candidates:
        if candidate.is_file() and candidate.with_name("ffprobe.exe").is_file():
            return str(candidate)
    return None
