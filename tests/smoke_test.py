"""Hafif sağlık kontrolü — ağır model yüklemeleri olmadan."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# repo root'u import yoluna ekle
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.types import Chunk, Document, QdrantPoint  # noqa: E402
from core.utils import (  # noqa: E402
    detect_source_type,
    is_youtube_url,
    safe_filename,
    stable_chunk_id,
)


def test_detect_source_type():
    assert detect_source_type("foo.pdf") == "pdf"
    assert detect_source_type("foo.PNG") == "image"
    assert detect_source_type("a.mp3") == "audio"
    assert detect_source_type("b.mp4") == "video"
    assert detect_source_type("c.txt") == "text"
    assert detect_source_type("https://example.com") == "web"
    assert detect_source_type("https://www.youtube.com/watch?v=dQw4w9WgXcQ") == "youtube"
    assert detect_source_type("https://youtu.be/dQw4w9WgXcQ") == "youtube"


def test_is_youtube_url():
    assert is_youtube_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    assert is_youtube_url("https://youtu.be/dQw4w9WgXcQ")
    assert is_youtube_url("https://www.youtube.com/shorts/aaaaaaaaaaa")
    assert not is_youtube_url("https://example.com")


def test_safe_filename():
    assert safe_filename("Hello World!.pdf") == "Hello_World_.pdf"
    assert safe_filename("../../etc/passwd") == "etc_passwd"
    assert safe_filename("") == "untitled"


def test_stable_chunk_id_deterministic():
    a = stable_chunk_id("file:///x.pdf", 3)
    b = stable_chunk_id("file:///x.pdf", 3)
    c = stable_chunk_id("file:///x.pdf", 4)
    assert a == b
    assert a != c


def test_dataclasses_construct():
    doc = Document(text="merhaba", source_type="text", source_uri="file:///x")
    assert doc.text == "merhaba"
    assert doc.segments == []
    ch = Chunk(text="abc", chunk_index=0)
    assert ch.token_count == 0
    pt = QdrantPoint(id="x", vector=[0.1, 0.2], payload={"k": "v"})
    assert pt.payload["k"] == "v"


def test_text_loader_reads_file(tmp_path: Path):
    from loaders.text_loader import TextLoader
    p = tmp_path / "ornek.txt"
    p.write_text("Merhaba dünya.\n\nİkinci paragraf.", encoding="utf-8")
    doc = TextLoader().load(str(p))
    assert "Merhaba" in doc.text
    assert doc.source_type == "text"
    assert doc.extra["extension"] == ".txt"


def test_qdrant_writer_jsonl_parquet(tmp_path: Path):
    import numpy as np
    from export.qdrant_writer import build_points, write_outputs

    doc = Document(text="abc def ghi", source_type="text", source_uri="file:///z.txt", title="z")
    chunks = [
        Chunk(text="abc", chunk_index=0, char_start=0, char_end=3, token_count=1),
        Chunk(text="def", chunk_index=1, char_start=4, char_end=7, token_count=1),
    ]
    vecs = np.zeros((2, 4), dtype=np.float32)
    points = build_points(doc, chunks, vecs)
    assert len(points) == 2
    files = write_outputs(doc, points, out_dir=tmp_path)
    assert files["jsonl"].exists()
    assert files["parquet"].exists()
    assert files["jsonl"].stat().st_size > 0


@pytest.mark.skipif(
    not (Path(__file__).parent / "fixtures" / "sample.pdf").exists(),
    reason="fixture yok",
)
def test_pdf_loader_smoke():
    from loaders.pdf_loader import PDFLoader
    fx = Path(__file__).parent / "fixtures" / "sample.pdf"
    doc = PDFLoader().load(str(fx))
    assert doc.source_type == "pdf"
    assert doc.extra["page_count"] >= 1


def test_bilingual_chunking():
    from processing.chunker import chunk_text
    
    # Türkçe test metni
    tr_text = (
        "Yapay zeka teknolojileri son yıllarda büyük bir hızla gelişmektedir. "
        "Bu gelişmeler, birçok sektörde devrim niteliğinde değişikliklere yol açmaktadır. "
        "Özellikle doğal dil işleme alanındaki yenilikler dikkat çekicidir.\n\n"
        "Diğer yandan, makine öğrenmesi algoritmaları da sürekli olarak optimize edilmektedir. "
        "Veri analitiği süreçleri bu sayede daha verimli hale gelmektedir."
    )
    tr_chunks = chunk_text(tr_text)
    assert len(tr_chunks) > 0
    assert any("Yapay zeka" in c.text for c in tr_chunks)
    
    # İngilizce test metni
    en_text = (
        "Artificial intelligence technologies have been developing rapidly in recent years. "
        "These developments lead to revolutionary changes in many industries. "
        "Particularly, innovations in natural language processing are remarkable.\n\n"
        "On the other hand, machine learning algorithms are also being continuously optimized. "
        "Data analytics processes become more efficient as a result."
    )
    en_chunks = chunk_text(en_text)
    assert len(en_chunks) > 0
    assert any("Artificial intelligence" in c.text for c in en_chunks)


def test_qa_chunking():
    from processing.chunker import chunk_text

    # Soru ve cevaplar içeren test metni
    qa_text = (
        "Bu ilk soru mudur?\n"
        "Evet, bu ilk sorunun cevabıdır. Oldukça açıklayıcı bir metin.\n\n"
        "Peki bu ikinci soru olabilir mi?\n"
        "Evet, bu da ikinci sorunun cevabıdır.\n"
        "İkinci cevabın ikinci paragrafıdır bu.\n\n"
        "Soru içermeyen son bir paragraf."
    )
    
    # 1. Normal semantik chunking ile (qa_mode=False)
    normal_chunks = chunk_text(qa_text, qa_mode=False)
    assert len(normal_chunks) > 0
    for c in normal_chunks:
        assert c.question is None
        
    # 2. QA Modunda (qa_mode=True)
    qa_chunks = chunk_text(qa_text, qa_mode=True)
    assert len(qa_chunks) == 3
    
    # İlk çift
    assert qa_chunks[0].question == "Bu ilk soru mudur?"
    assert qa_chunks[0].text == "Evet, bu ilk sorunun cevabıdır. Oldukça açıklayıcı bir metin."
    
    # İkinci çift (birden fazla paragraf birleştirilmeli)
    assert qa_chunks[1].question == "Peki bu ikinci soru olabilir mi?"
    assert "ikinci sorunun cevabıdır" in qa_chunks[1].text
    assert "İkinci cevabın ikinci paragrafıdır" in qa_chunks[1].text
    
    # Üçüncü kısım (sorusuz)
    assert qa_chunks[2].question is None
    assert qa_chunks[2].text == "Soru içermeyen son bir paragraf."

    # 3. Açıkça 'soru: cevap:' veya 'soru: ?' içeren metin testi
    explicit_text = (
        "soru:Bana rüyanı anlatır mısın? cevap:Dün gece eski ilkokuluma geri döndüğümü gördüm.\n\n"
        "soru:peki sonra ne oldu?  Daha sonra okulun bahçesine çıktığımda yağmur aniden durdu."
    )
    explicit_chunks = chunk_text(explicit_text, qa_mode=True)
    assert len(explicit_chunks) == 2
    
    assert explicit_chunks[0].question == "Bana rüyanı anlatır mısın?"
    assert explicit_chunks[0].text == "Dün gece eski ilkokuluma geri döndüğümü gördüm."
    
    assert explicit_chunks[1].question == "peki sonra ne oldu?"
    assert explicit_chunks[1].text == "Daha sonra okulun bahçesine çıktığımda yağmur aniden durdu."
