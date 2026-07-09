import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import benchmark_all_whisper as benchmark_all_whisper_module


class DummySegment:
    def __init__(self, text):
        self.text = text


class DummyInfo:
    def __init__(self, duration=1.0, language="tr"):
        self.duration = duration
        self.language = language


class DummyModel:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def transcribe(self, audio_path, language=None):
        return [DummySegment("merhaba")], DummyInfo(duration=1.23, language="tr")


class BenchmarkAllWhisperTests(unittest.TestCase):
    def test_load_and_transcribe_returns_compatible_keys(self):
        benchmark_all_whisper_module.WhisperModel = DummyModel

        result = benchmark_all_whisper_module.load_and_transcribe(
            "tiny",
            "dummy.mp3",
            "cpu",
            "int8",
        )

        self.assertIn("load_time_sec", result)
        self.assertIn("load_time_s", result)
        self.assertIn("transcribe_time_sec", result)
        self.assertIn("trans_time_s", result)

    def test_run_top_benchmark_writes_transcript_text_to_jsonl(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_dir = os.path.join(tmpdir, "audio")
            ref_dir = os.path.join(tmpdir, "refs")
            output_dir = os.path.join(tmpdir, "out")
            os.makedirs(audio_dir)
            os.makedirs(ref_dir)
            os.makedirs(output_dir)

            audio_path = os.path.join(audio_dir, "sample.mp3")
            ref_path = os.path.join(ref_dir, "sample.txt")
            Path(audio_path).write_bytes(b"dummy")
            Path(ref_path).write_text("referans metin", encoding="utf-8")

            with patch.object(benchmark_all_whisper_module, "load_and_transcribe", return_value={
                "load_time_s": 0.1,
                "load_time_sec": 0.1,
                "trans_time_s": 0.2,
                "transcribe_time_sec": 0.2,
                "rtf": 0.2,
                "text": "dönüşen metin",
                "detected_language": "tr",
            }), patch.object(benchmark_all_whisper_module.config, "WHISPER_AVAILABLE_MODELS", ["tiny"]), patch.object(benchmark_all_whisper_module.config, "WHISPER_MODEL_MAP", {}), patch.object(benchmark_all_whisper_module, "wer", lambda a, b: 0.25), patch.object(benchmark_all_whisper_module, "cer", lambda a, b: 0.1), patch.object(benchmark_all_whisper_module.config, "WHISPER_LANGUAGE", "tr"):
                with patch.object(sys, "argv", [
                    "benchmark_all_whisper.py",
                    "--audio_dir", audio_dir,
                    "--ref_dir", ref_dir,
                    "--output_dir", output_dir,
                    "--device", "cpu",
                    "--compute", "int8",
                ]):
                    benchmark_all_whisper_module.run_top_benchmark()

            output_file = os.path.join(output_dir, "toplu_30_ses_analizi.jsonl")
            with open(output_file, encoding="utf-8") as handle:
                lines = [json.loads(line) for line in handle if line.strip()]

            self.assertEqual(len(lines), 1)
            self.assertEqual(lines[0]["text"], "dönüşen metin")
            self.assertEqual(lines[0]["wer"], 0.25)
            self.assertEqual(lines[0]["cer"], 0.1)
            self.assertEqual(lines[0]["reference_language"], "tr")
            self.assertTrue(lines[0]["language_match"])


if __name__ == "__main__":
    unittest.main()
