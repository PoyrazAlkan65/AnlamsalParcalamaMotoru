"""Streamlit UI — çok kaynaklı veri yükle, anlamsal chunk + embed + export."""
from __future__ import annotations

import traceback
from pathlib import Path

import streamlit as st

import config
from core.pipeline import IngestResult, ingest
from core.utils import detect_source_type, is_youtube_url, safe_filename

st.set_page_config(
    page_title="Anlamsal Parçalama Motoru",
    page_icon=":material/graph_3:",
    layout="wide",
)

ACCEPT_EXTENSIONS = [
    "pdf",
    "png", "jpg", "jpeg", "bmp", "tiff", "webp", "gif",
    "mp3", "wav", "m4a", "ogg", "flac",
    "mp4", "mkv", "webm", "mov", "avi",
    "txt", "md", "csv", "json", "log", "rst",
]


# ---- Session state ----
if "results" not in st.session_state:
    st.session_state.results = []  # list[dict]
if "errors" not in st.session_state:
    st.session_state.errors = []


# ---- Sidebar ----
with st.sidebar:
    st.header("Ayarlar")
    st.text_input("Çıktı dizini", value=str(config.OUTPUTS_DIR), disabled=True)
    st.caption(f"Embedding: `{config.EMBEDDING_MODEL}` ({config.EMBEDDING_DIM}d)")
    st.caption(f"Whisper: `{config.WHISPER_MODEL}` ({config.WHISPER_LANGUAGE})")
    st.caption(f"Caption: `{config.CAPTION_MODEL}`")

    st.divider()
    st.subheader("Chunking")
    st.slider(
        "Benzerlik eşiği",
        min_value=0.1, max_value=0.95, step=0.05,
        value=config.CHUNK_SIMILARITY_THRESHOLD,
        key="sim_threshold",
        help="Düşük eşik = daha az parça. Yüksek eşik =daha çok parça (agresif bölme).",
    )
    st.slider(
        "Min token",
        min_value=16, max_value=256, step=16,
        value=config.CHUNK_MIN_TOKENS,
        key="min_tokens",
    )
    st.slider(
        "Max token",
        min_value=128, max_value=2048, step=64,
        value=config.CHUNK_MAX_TOKENS,
        key="max_tokens",
    )

    st.divider()
    if st.button("Sonuçları temizle", use_container_width=True):
        st.session_state.results = []
        st.session_state.errors = []
        st.rerun()


# ---- Main ----
st.title("Anlamsal Parçalama Motoru")
st.caption(
    "Resim · PDF · Video · YouTube · Metin · MP3 · Web — semantik metne çevir, "
    "anlamsal parçalara böl, Qdrant'a yüklenmek üzere dışa aktar."
)

tab_files, tab_url, tab_results = st.tabs(["Dosya yükle", "URL gir", f"Sonuçlar ({len(st.session_state.results)})"])


def _save_upload(uploaded_file) -> Path:
    dst_dir = Path(config.INPUTS_DIR)
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / safe_filename(uploaded_file.name, max_len=120)
    with dst.open("wb") as f:
        f.write(uploaded_file.getbuffer())
    return dst


def _run_one(source: str, status, label: str) -> IngestResult | None:
    def progress_cb(msg: str, pct: float | None) -> None:
        if pct is not None:
            status.update(label=f"{label} — {msg}", state="running")

    try:
        result = ingest(
            source,
            progress=progress_cb,
            similarity_threshold=st.session_state.get("sim_threshold"),
            min_tokens=st.session_state.get("min_tokens"),
            max_tokens=st.session_state.get("max_tokens"),
        )
    except Exception as e:
        tb = traceback.format_exc(limit=4)
        st.session_state.errors.append({"source": source, "error": str(e), "trace": tb})
        status.update(label=f"{label} — HATA: {e}", state="error")
        return None

    qdrant = result.qdrant_result or {}
    st.session_state.results.append({
        "source": source,
        "source_type": result.document.source_type,
        "title": result.document.title,
        "language": result.document.language,
        "chunk_count": result.chunk_count,
        "text_chars": len(result.document.text),
        "avg_chunk_chars": round(result.avg_chunk_chars, 1),
        "files": {k: str(v) for k, v in result.output_files.items()},
        "first_chunks": [c.text[:300] for c in result.chunks[:3]],
        "extra": result.document.extra,
        "qdrant": qdrant,
    })

    qdrant_label = ""
    if qdrant.get("success"):
        qdrant_label = f" · Qdrant ✓ ({qdrant.get('upserted', 0)} point)"
    elif qdrant.get("error"):
        qdrant_label = f" · Qdrant ✗"

    status.update(label=f"{label} — Tamam ({result.chunk_count} parça{qdrant_label})", state="complete")
    return result


# ===== Dosya yükle tab =====
with tab_files:
    uploaded = st.file_uploader(
        "Bir veya daha fazla dosya seç",
        type=ACCEPT_EXTENSIONS,
        accept_multiple_files=True,
    )
    process_files = st.button(
        "Dosyaları işle",
        type="primary",
        disabled=not uploaded,
        key="btn_files",
    )

    if process_files and uploaded:
        for uf in uploaded:
            saved = _save_upload(uf)
            label = f"{uf.name}"
            with st.status(f"İşleniyor: {label}", expanded=False) as status:
                _run_one(str(saved), status, label)
        st.rerun()


# ===== URL tab =====
with tab_url:
    url = st.text_input(
        "URL (YouTube veya herhangi bir web sayfası)",
        placeholder="https://… veya https://www.youtube.com/watch?v=…",
    )
    detected = detect_source_type(url) if url else None
    if url:
        if detected == "youtube":
            st.info("YouTube linki algılandı → sesi çekilip transkribe edilecek.")
        elif detected == "web":
            st.info("Web sayfası algılandı → Trafilatura, gerekirse Playwright.")
        else:
            st.warning("URL geçersiz görünüyor.")

    process_url = st.button(
        "URL'yi işle",
        type="primary",
        disabled=not url or detected not in {"web", "youtube"},
        key="btn_url",
    )

    if process_url and url:
        label = url[:80]
        with st.status(f"İşleniyor: {label}", expanded=False) as status:
            _run_one(url, status, label)
        st.rerun()


# ===== Sonuçlar tab =====
with tab_results:
    if st.session_state.errors:
        with st.expander(f"Hatalar ({len(st.session_state.errors)})", expanded=False):
            for err in st.session_state.errors:
                st.error(f"**{err['source']}**\n\n{err['error']}")
                st.code(err["trace"], language="text")

    if not st.session_state.results:
        st.info("Henüz işlenmiş kaynak yok. Dosya yükle veya URL gir.")
    else:
        for i, r in enumerate(reversed(st.session_state.results)):
            with st.container(border=True):
                top = st.columns([3, 1, 1, 1])
                top[0].markdown(f"**{r['title'] or r['source']}**")
                top[0].caption(r["source"])
                top[1].metric("Tip", r["source_type"])
                top[2].metric("Parça", r["chunk_count"])
                top[3].metric("Ort. karakter", r["avg_chunk_chars"])

                meta_cols = st.columns(3)
                meta_cols[0].caption(f"Dil: `{r['language'] or 'auto'}`")
                meta_cols[1].caption(f"Toplam metin: {r['text_chars']:,} karakter")
                meta_cols[2].caption(f"Çıktı: `{Path(r['files'].get('parquet', '')).name}`")

                # Qdrant durumu
                qdrant_info = r.get("qdrant", {})
                if qdrant_info.get("success"):
                    st.success(f"Qdrant ✓ — {qdrant_info.get('upserted', 0)} point yüklendi → `{config.QDRANT_COLLECTION}`")
                elif qdrant_info.get("error"):
                    st.warning(f"Qdrant ✗ — {qdrant_info['error']}")

                with st.expander("İlk parçalar"):
                    for j, txt in enumerate(r["first_chunks"]):
                        st.markdown(f"**#{j}**")
                        st.write(txt + ("…" if len(txt) >= 300 else ""))

                with st.expander("Kaynak meta"):
                    st.json(r["extra"])

                dl_cols = st.columns(2)
                for col, kind in zip(dl_cols, ["jsonl", "parquet"]):
                    p = r["files"].get(kind)
                    if p and Path(p).exists():
                        with open(p, "rb") as f:
                            col.download_button(
                                f"{kind.upper()} indir",
                                f.read(),
                                file_name=Path(p).name,
                                mime="application/octet-stream",
                                key=f"dl_{kind}_{i}",
                                use_container_width=True,
                            )
