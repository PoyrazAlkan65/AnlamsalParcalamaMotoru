"""Unit tests for loaders: text, pdf, audio, video, youtube, image, web."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.types import Document, Segment
from loaders.audio_loader import AudioLoader
from loaders.image_loader import ImageLoader
from loaders.pdf_loader import PDFLoader
from loaders.text_loader import TextLoader
from loaders.video_loader import VideoLoader
from loaders.web_loader import WebLoader
from loaders.youtube_loader import YouTubeLoader


# 1. TextLoader test
def test_text_loader(tmp_path: Path):
    p = tmp_path / "test.txt"
    p.write_text("Hello Text Loader!", encoding="utf-8")
    loader = TextLoader()
    doc = loader.load(str(p))
    assert isinstance(doc, Document)
    assert doc.text == "Hello Text Loader!"
    assert doc.source_type == "text"
    assert doc.extra["extension"] == ".txt"


# 2. PDFLoader test (mocking fitz/PyMuPDF)
@patch("fitz.open")
def test_pdf_loader(mock_fitz_open: MagicMock):
    # Mocking fitz page and document
    mock_page = MagicMock()
    mock_page.get_text.return_value = "Page 1 Content"

    mock_doc = MagicMock()
    mock_doc.__len__.return_value = 1
    mock_doc.load_page.return_value = mock_page
    mock_doc.metadata = {"title": "Test Title", "author": "Test Author"}
    mock_fitz_open.return_value = mock_doc

    loader = PDFLoader()
    doc = loader.load("dummy.pdf")

    assert isinstance(doc, Document)
    assert doc.text == "Page 1 Content"
    assert doc.title == "Test Title"
    assert doc.source_type == "pdf"
    assert len(doc.segments) == 1
    assert doc.segments[0].text == "Page 1 Content"
    assert doc.segments[0].page == 1
    assert doc.extra["author"] == "Test Author"
    mock_doc.close.assert_called_once()


# 3. AudioLoader test (mocking WhisperEngine / get_whisper)
@patch("loaders.audio_loader.get_whisper")
def test_audio_loader(mock_get_whisper: MagicMock, tmp_path: Path):
    mock_engine = MagicMock()
    mock_segments = [Segment(text="Hello Audio", start=0.0, end=2.5)]
    mock_engine.transcribe.return_value = ("Hello Audio", "en", mock_segments)
    mock_get_whisper.return_value = mock_engine

    # We need a dummy file that exists so Path(source).stat() works
    p = tmp_path / "dummy.mp3"
    p.write_text("audio-data")

    loader = AudioLoader()
    doc = loader.load(str(p), whisper_language="en")

    assert isinstance(doc, Document)
    assert doc.text == "Hello Audio"
    assert doc.source_type == "audio"
    assert doc.language == "en"
    assert len(doc.segments) == 1
    assert doc.extra["duration_sec"] == 2.5
    mock_engine.transcribe.assert_called_once_with(Path(p), language="en")


# 4. VideoLoader test (mocking ffmpeg call and get_whisper)
@patch("loaders.video_loader.get_whisper")
@patch("subprocess.run")
@patch("shutil.which")
def test_video_loader(
    mock_which: MagicMock, mock_run: MagicMock, mock_get_whisper: MagicMock, tmp_path: Path
):
    mock_which.return_value = "/usr/bin/ffmpeg"

    # Mock subprocess.run for ffmpeg success
    mock_proc = MagicMock()
    mock_proc.returncode = 0
    mock_run.return_value = mock_proc

    mock_engine = MagicMock()
    mock_segments = [Segment(text="Hello Video", start=0.0, end=5.0)]
    mock_engine.transcribe.return_value = ("Hello Video", "en", mock_segments)
    mock_get_whisper.return_value = mock_engine

    p = tmp_path / "dummy.mp4"
    p.write_text("video-data")

    loader = VideoLoader()
    doc = loader.load(str(p))

    assert isinstance(doc, Document)
    assert doc.text == "Hello Video"
    assert doc.source_type == "video"
    assert len(doc.segments) == 1
    assert doc.extra["duration_sec"] == 5.0

    mock_run.assert_called_once()
    mock_engine.transcribe.assert_called_once()


# 5. YouTubeLoader test (mocking yt_dlp and get_whisper)
@patch("loaders.youtube_loader.get_whisper")
@patch("yt_dlp.YoutubeDL")
@patch("pathlib.Path.glob")
def test_youtube_loader(mock_glob: MagicMock, mock_yt_dlp: MagicMock, mock_get_whisper: MagicMock):
    mock_instance = MagicMock()
    mock_instance.extract_info.return_value = {
        "id": "12345",
        "title": "Test Video",
        "duration": 120,
        "uploader": "Test Creator",
        "webpage_url": "https://youtube.com/watch?v=12345",
    }
    mock_yt_dlp.return_value.__enter__.return_value = mock_instance

    # Mock glob to find a dummy wav file
    mock_glob.return_value = iter([Path("dummy.wav")])

    mock_engine = MagicMock()
    mock_segments = [Segment(text="Hello YouTube", start=0.0, end=120.0)]
    mock_engine.transcribe.return_value = ("Hello YouTube", "tr", mock_segments)
    mock_get_whisper.return_value = mock_engine

    loader = YouTubeLoader()
    doc = loader.load("https://www.youtube.com/watch?v=12345")

    assert isinstance(doc, Document)
    assert doc.text == "Hello YouTube"
    assert doc.source_type == "youtube"
    assert doc.title == "Test Video"
    assert doc.extra["video_id"] == "12345"
    assert doc.extra["duration_sec"] == 120
    assert doc.extra["uploader"] == "Test Creator"
    mock_engine.transcribe.assert_called_once()


# 6. ImageLoader test (mocking PIL, pytesseract, BLIP captioning)
@patch("loaders.image_loader._run_ocr")
@patch("loaders.image_loader._CaptionPipeline")
@patch("PIL.Image.open")
def test_image_loader(
    mock_image_open: MagicMock, mock_caption_pipeline: MagicMock, mock_run_ocr: MagicMock, tmp_path: Path
):
    # Mock PIL Image
    mock_img = MagicMock()
    mock_img.convert.return_value = mock_img
    mock_img.size = (800, 600)
    mock_image_open.return_value.__enter__.return_value = mock_img

    # Mock OCR
    mock_run_ocr.return_value = ("OCR Text Output", 85.0)

    # Mock BLIP captioning
    mock_pipeline_instance = MagicMock()
    mock_pipeline_instance.caption.return_value = "A beautiful scene"
    mock_caption_pipeline.return_value = mock_pipeline_instance

    p = tmp_path / "dummy.png"
    p.write_text("image-data")

    loader = ImageLoader()
    doc = loader.load(str(p))

    assert isinstance(doc, Document)
    assert "[Görsel açıklaması] A beautiful scene" in doc.text
    assert "[OCR metni]\nOCR Text Output" in doc.text
    assert doc.source_type == "image"
    assert doc.extra["width"] == 800
    assert doc.extra["height"] == 600
    assert doc.extra["ocr_confidence"] == 85.0


# 7. WebLoader test (mocking trafilatura and playwright)
@patch("loaders.web_loader.WebLoader._try_trafilatura")
@patch("loaders.web_loader.WebLoader._try_playwright")
def test_web_loader(mock_try_playwright: MagicMock, mock_try_trafilatura: MagicMock):
    # Set default mock return value for playwright to prevent unpack errors if called
    mock_try_playwright.return_value = ("", None)

    # Case A: Trafilatura succeeds with a text length >= 80 characters
    trafilatura_text = "Web page content from trafilatura. " * 3  # 105 characters
    mock_try_trafilatura.return_value = (
        trafilatura_text,
        "Trafilatura Title",
        "trafilatura",
    )

    loader = WebLoader()
    doc = loader.load("https://example.com")

    assert isinstance(doc, Document)
    assert doc.text == trafilatura_text
    assert doc.title == "Trafilatura Title"
    assert doc.source_type == "web"
    assert doc.extra["extractor"] == "trafilatura"

    # Case B: Trafilatura fails, falls back to Playwright
    mock_try_trafilatura.return_value = ("", None, "trafilatura-empty")
    mock_try_playwright.return_value = ("Web page content from playwright", "Playwright Title")

    doc = loader.load("https://example.com")
    assert doc.text == "Web page content from playwright"
    assert doc.title == "Playwright Title"
    assert doc.extra["extractor"] == "playwright"