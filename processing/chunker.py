"""Hibrit chunker: paragraf ön-bölme + embedding tabanlı semantic chunking.

Akış:
1. Metni yapısal sınırlardan (boş satır, başlık) kabaca böl.
2. Her büyük blok için chonkie SemanticChunker çağır.
3. Çok kısa parçaları bir sonrakine ekle, çok uzunları zorla böl.
"""
from __future__ import annotations

import re
import threading
from typing import Iterable

import config
from core.types import Chunk

_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n+")
_HEADING = re.compile(r"^(#{1,6}\s+|\d+\.\s+|[A-ZĞÜŞİÖÇ ]{8,}$)", re.MULTILINE)


class HybridChunker:
    """Paragraf ön-bölme + chonkie SemanticChunker."""
    _shared_embeddings = None
    _lock = threading.Lock()

    def __init__(
        self,
        embedding_model: str | None = None,
        similarity_threshold: float | None = None,
        min_tokens: int | None = None,
        max_tokens: int | None = None,
    ) -> None:
        from chonkie import SemanticChunker
        from chonkie.embeddings import SentenceTransformerEmbeddings

        self.embedding_model_name = embedding_model or config.EMBEDDING_MODEL
        self.similarity_threshold = (
            similarity_threshold if similarity_threshold is not None
            else config.CHUNK_SIMILARITY_THRESHOLD
        )
        self.min_tokens = min_tokens or config.CHUNK_MIN_TOKENS
        self.max_tokens = max_tokens or config.CHUNK_MAX_TOKENS

        with self._lock:
            if HybridChunker._shared_embeddings is None or HybridChunker._shared_embeddings.model_name_or_path != self.embedding_model_name:
                HybridChunker._shared_embeddings = SentenceTransformerEmbeddings(self.embedding_model_name)

        self._semantic = SemanticChunker(
            embedding_model=HybridChunker._shared_embeddings,
            threshold=self.similarity_threshold,
            chunk_size=self.max_tokens,
            min_sentences_per_chunk=1,
            min_characters_per_sentence=12,
        )

    # ---- yapısal ön-bölme ----
    def _structural_blocks(self, text: str) -> list[tuple[int, str]]:
        """Paragraflara böl, her bloğun char offset'ini koru."""
        blocks: list[tuple[int, str]] = []
        cursor = 0
        for match in _PARAGRAPH_SPLIT.split(text):
            if not match.strip():
                cursor += len(match)
                continue
            idx = text.find(match, cursor)
            if idx < 0:
                idx = cursor
            blocks.append((idx, match))
            cursor = idx + len(match)
        if not blocks:
            blocks.append((0, text))
        return blocks

    # ---- public ----
    def chunk(self, text: str) -> list[Chunk]:
        text = (text or "").strip()
        if not text:
            return []

        results: list[Chunk] = []
        idx = 0
        for block_offset, block in self._structural_blocks(text):
            if len(block) < 40:
                # Çok kısaysa direkt küçük bir chunk
                results.append(Chunk(
                    text=block.strip(),
                    chunk_index=idx,
                    char_start=block_offset,
                    char_end=block_offset + len(block),
                    token_count=max(1, len(block) // 4),
                ))
                idx += 1
                continue

            try:
                sentences = self._semantic._prepare_sentences(block)
                if not sentences:
                    sub_chunks = []
                elif len(sentences) == 1:
                    sub_chunks = [self._semantic._create_chunks([sentences])[0]]
                else:
                    import numpy as np
                    embeddings = self._semantic.embedding_model.embed_batch([s.text for s in sentences])
                    similarities = []
                    for i in range(len(embeddings) - 1):
                        a = embeddings[i]
                        b = embeddings[i + 1]
                        norm_a = np.linalg.norm(a)
                        norm_b = np.linalg.norm(b)
                        sim = np.dot(a, b) / (norm_a * norm_b) if norm_a > 0 and norm_b > 0 else 0.0
                        similarities.append(float(sim))

                    chunks_data = []
                    current_group = [sentences[0]]
                    for i in range(len(similarities)):
                        sim = similarities[i]
                        next_sentence = sentences[i + 1]
                        if sim < self.similarity_threshold:
                            chunks_data.append(current_group)
                            current_group = [next_sentence]
                        else:
                            current_group.append(next_sentence)
                    if current_group:
                        chunks_data.append(current_group)

                    final_groups = self._semantic._split_groups(chunks_data)
                    sub_chunks = self._semantic._create_chunks(final_groups)
            except Exception:
                # Herhangi bir hata durumunda düz olarak ekle
                sub_chunks = [type("X", (), {
                    "text": block,
                    "start_index": 0,
                    "end_index": len(block),
                    "token_count": max(1, len(block) // 4),
                })()]

            for sc in sub_chunks:
                start = getattr(sc, "start_index", 0)
                end = getattr(sc, "end_index", len(sc.text))
                results.append(Chunk(
                    text=sc.text.strip(),
                    chunk_index=idx,
                    char_start=block_offset + start,
                    char_end=block_offset + end,
                    token_count=getattr(sc, "token_count", max(1, len(sc.text) // 4)),
                ))
                idx += 1

        return self._merge_small(results)

    def _merge_small(self, chunks: list[Chunk]) -> list[Chunk]:
        """min_tokens altındaki ardışıkları komşusuyla birleştir.

        Ancak anlamsal olarak farklı parçaları (benzerlik < threshold)
        birleştirmez, böylece semantik bölme sonuçları korunur.
        """
        if not chunks:
            return chunks

        import numpy as np

        # Embedding'leri bir kez hesapla
        texts = [c.text for c in chunks]
        try:
            embeddings = self._semantic.embedding_model.embed_batch(texts)
        except Exception:
            embeddings = None

        merged: list[Chunk] = []
        for i, c in enumerate(chunks):
            if not merged or c.token_count >= self.min_tokens:
                merged.append(c)
                continue

            # Boyut kontrolü
            if merged[-1].token_count + c.token_count > self.max_tokens:
                merged.append(c)
                continue

            # Benzerlik kontrolü: farklı anlamdaysa birleştirme
            if embeddings is not None:
                prev_idx = i - 1
                a = embeddings[prev_idx]
                b = embeddings[i]
                norm_a = np.linalg.norm(a)
                norm_b = np.linalg.norm(b)
                sim = float(np.dot(a, b) / (norm_a * norm_b)) if norm_a > 0 and norm_b > 0 else 0.0
                if sim < self.similarity_threshold:
                    # Anlamsal olarak farklı → birleştirme
                    merged.append(c)
                    continue

            # Benzer ve küçük → birleştir
            prev = merged[-1]
            prev.text = (prev.text + "\n\n" + c.text).strip()
            prev.char_end = c.char_end
            prev.token_count += c.token_count

        # yeniden indeksle
        for i, c in enumerate(merged):
            c.chunk_index = i
        return merged


def is_question(block: str) -> bool:
    """Bir bloğun soru olup olmadığını tespit eder (kısa ve soru işareti ile biten)."""
    s = block.strip()
    if not s:
        return False
    if len(s) > 250:
        return False
    return bool(re.search(r'\?\s*["\'”’]?$', s))


def parse_qa_text(text: str) -> list[Chunk]:
    """Metni soru ve cevap bloklarına göre parçalar."""
    text = (text or "").strip()
    if not text:
        return []

    # Paragraflara böl ve konumlarını belirle
    blocks = []
    for m in re.finditer(r'(?:[^\n]|\n(?!\s*\n))+', text):
        block_text = m.group(0).strip()
        if block_text:
            blocks.append({
                "text": block_text,
                "start": m.start(),
                "end": m.end()
            })

    results: list[Chunk] = []
    idx = 0
    i = 0
    while i < len(blocks):
        block = blocks[i]
        text_content = block["text"]
        
        # 1. Durum: Açıkça "soru:" ile başlıyor mu?
        if text_content.lower().startswith("soru:"):
            raw_qa = text_content[5:].strip()
            question_text = ""
            answer_text = ""
            
            # "cevap:" kelimesi var mı?
            cevap_idx = raw_qa.lower().find("cevap:")
            if cevap_idx >= 0:
                question_text = raw_qa[:cevap_idx].strip()
                answer_text = raw_qa[cevap_idx + 6:].strip()
            else:
                # "?" karakterine göre ayır
                q_mark_idx = raw_qa.find("?")
                if q_mark_idx >= 0:
                    question_text = raw_qa[:q_mark_idx + 1].strip()
                    answer_text = raw_qa[q_mark_idx + 1:].strip()
                else:
                    question_text = raw_qa
                    answer_text = ""
            
            # Eğer cevap kısmı bu blokta boş kaldıysa, sonraki blokları yut
            j = i + 1
            if not answer_text:
                answer_parts = []
                while j < len(blocks):
                    next_block = blocks[j]
                    next_text = next_block["text"]
                    # Eğer sonraki blok soru: ile başlıyorsa veya kısa soruysa yutmayı kes
                    if next_text.lower().startswith("soru:") or is_question(next_text.splitlines()[0]):
                        break
                    answer_parts.append(next_text)
                    j += 1
                answer_text = "\n\n".join(answer_parts)
                char_end = blocks[j-1]["end"] if j > i + 1 else block["end"]
            else:
                char_end = block["end"]
                j = i + 1
                
            results.append(Chunk(
                text=answer_text,
                chunk_index=idx,
                question=question_text,
                char_start=block["start"],
                char_end=char_end,
                token_count=max(1, len(answer_text) // 4),
            ))
            idx += 1
            i = j
            
        else:
            # 2. Durum: "soru:" ifadesi yok ama ilk satır "?" ile biten kısa bir soru
            lines = text_content.splitlines()
            first_line = lines[0].strip() if lines else ""
            
            if is_question(first_line):
                question_text = first_line
                
                # Cevabın bu bloktaki geri kalan kısmı
                remaining_lines = lines[1:]
                answer_parts = [l.strip() for l in remaining_lines if l.strip()]
                
                # Eğer bu blokta cevap yoksa (sadece soru varsa), sonraki blokları yut
                j = i + 1
                if not answer_parts:
                    while j < len(blocks):
                        next_block = blocks[j]
                        next_lines = next_block["text"].splitlines()
                        next_first_line = next_lines[0].strip() if next_lines else ""
                        
                        if next_first_line.lower().startswith("soru:") or is_question(next_first_line):
                            break
                        
                        answer_parts.append(next_block["text"])
                        j += 1
                        
                answer_text = "\n\n".join(answer_parts)
                char_start = block["start"]
                char_end = blocks[j-1]["end"] if j > i + 1 else block["end"]
                
                results.append(Chunk(
                    text=answer_text,
                    chunk_index=idx,
                    question=question_text,
                    char_start=char_start,
                    char_end=char_end,
                    token_count=max(1, len(answer_text) // 4),
                ))
                idx += 1
                i = j
            else:
                # 3. Durum: Soru içermeyen normal paragraf
                results.append(Chunk(
                    text=block["text"],
                    chunk_index=idx,
                    question=None,
                    char_start=block["start"],
                    char_end=block["end"],
                    token_count=max(1, len(block["text"]) // 4),
                ))
                idx += 1
                i += 1

    return results


_default: HybridChunker | None = None


def get_chunker(
    similarity_threshold: float | None = None,
    min_tokens: int | None = None,
    max_tokens: int | None = None,
) -> HybridChunker:
    """Verilen parametrelerle yeni bir HybridChunker döndür.

    Parametreler verilmezse config'den varsayılanlar kullanılır.
    """
    global _default
    # Parametreler açıkça belirtildiyse her zaman yeni chunker oluştur
    if similarity_threshold is not None or min_tokens is not None or max_tokens is not None:
        return HybridChunker(
            similarity_threshold=similarity_threshold,
            min_tokens=min_tokens,
            max_tokens=max_tokens,
        )
    # Parametresiz çağrılmışsa varsayılan singleton kullan
    if _default is None:
        _default = HybridChunker()
    return _default


def chunk_text(
    text: str,
    similarity_threshold: float | None = None,
    min_tokens: int | None = None,
    max_tokens: int | None = None,
    qa_mode: bool = False,
) -> list[Chunk]:
    if qa_mode:
        return parse_qa_text(text)
    return get_chunker(similarity_threshold, min_tokens, max_tokens).chunk(text)

