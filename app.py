"""Streamlit UI — çok kaynaklı veri yükle, anlamsal chunk; seçilen hedeflere (Qdrant / dosya) kaydet ya da önizle."""
from __future__ import annotations

import traceback
import uuid
from pathlib import Path
from typing import Callable

import streamlit as st

import config
from connectors import sql_source
from core.pipeline import (
    OUTPUT_KINDS,
    BatchResult,
    IngestResult,
    process,
    process_documents,
    publish,
    publish_batch,
)
from core.types import OperationCancelled
from core.utils import detect_source_type, is_youtube_url, safe_filename

st.set_page_config(
    page_title="Anlamsal Parçalama Motoru",
    page_icon=":material/graph_3:",
    layout="wide",
)

# ---- Keycloak girişi ----
if not st.user.is_logged_in:
    st.title("Anlamsal Parçalama Motoru")
    st.write("Devam etmek için giriş yapın.")
    st.button("Giriş yap", on_click=st.login, type="primary")
    st.stop()

with st.sidebar:
    _kullanici = st.user.get("name") or st.user.get("preferred_username") or ""
    st.caption(f"Giriş yapan: {_kullanici}")
    st.button("Çıkış yap", on_click=st.logout)

ACCEPT_EXTENSIONS = [
    "pdf",
    "png", "jpg", "jpeg", "bmp", "tiff", "webp", "gif",
    "mp3", "wav", "m4a", "ogg", "flac",
    "mp4", "mkv", "webm", "mov", "avi",
    "txt", "md", "csv", "json", "log", "rst",
]

# Qdrant hedefi .env'den gelir (kurulum = müşteri); arayüzde yalnızca gösterilir
QDRANT_READY = bool(config.QDRANT_ENABLED and config.QDRANT_HOST and config.QDRANT_COLLECTION)
_KIND_LABELS = {"txt": "TXT", "jsonl": "JSONL", "parquet": "Parquet"}


# ---- Session state ----
if "results" not in st.session_state:
    st.session_state.results = []  # list[dict]: {id, source, result: IngestResult}
if "errors" not in st.session_state:
    st.session_state.errors = []
if "cancel_requested" not in st.session_state:
    st.session_state.cancel_requested = False


def request_cancel():
    st.session_state.cancel_requested = True


if st.session_state.cancel_requested:
    st.toast("İşlem kullanıcı tarafından iptal edildi.", icon="ℹ️")
    st.session_state.cancel_requested = False


# ---- Sidebar ----
with st.sidebar:
    st.header("Ayarlar")
    st.text_input("Çıktı dizini", value=str(config.OUTPUTS_DIR), disabled=True)
    st.caption(f"Embedding: `{config.EMBEDDING_MODEL}` ({config.EMBEDDING_DIM}d)")
    st.caption(f"Whisper: `{config.WHISPER_MODEL}` ({config.WHISPER_LANGUAGE})")
    st.caption(f"Caption: `{config.CAPTION_MODEL}`")

    st.divider()
    st.subheader("Parçalama")
    chunk_enabled = st.checkbox(
        "Parçala",
        value=True,
        key="chunk_enabled",
        help="Kapalıysa yalnızca metin çıkarılır (ör. sadece transkript).",
    )
    strategy = st.radio(
        "Strateji",
        ["Anlamsal", "Soru-Cevap", "Tek parça"],
        key="chunk_strategy",
        horizontal=True,
        disabled=not chunk_enabled,
        help=(
            "Soru-Cevap: ardışık soru ve cevapları eşleştirip tek parça yapar. "
            "Tek parça: metin bölünmez (ör. kısa veritabanı kayıtları; her satır bir parça)."
        ),
    )
    semantic_off = not chunk_enabled or strategy != "Anlamsal"
    st.slider(
        "Benzerlik eşiği",
        min_value=0.1, max_value=0.95, step=0.05,
        value=config.CHUNK_SIMILARITY_THRESHOLD,
        key="sim_threshold",
        disabled=semantic_off,
        help="Düşük eşik = daha az parça. Yüksek eşik =daha çok parça (agresif bölme).",
    )
    st.slider(
        "Min token",
        min_value=16, max_value=256, step=16,
        value=config.CHUNK_MIN_TOKENS,
        key="min_tokens",
        disabled=semantic_off,
    )
    st.slider(
        "Max token",
        min_value=128, max_value=2048, step=64,
        value=config.CHUNK_MAX_TOKENS,
        key="max_tokens",
        disabled=semantic_off,
    )

    st.divider()
    st.subheader("Kayıt")
    st.checkbox(
        "Qdrant'a otomatik kaydet",
        value=False,
        key="auto_qdrant",
        disabled=not (chunk_enabled and QDRANT_READY),
        help="İşaretsizse sonuçlar önizlenir; Sonuçlar sekmesinden sonradan kaydedilebilir.",
    )
    if QDRANT_READY:
        st.caption(f"Koleksiyon: `{config.QDRANT_COLLECTION}`")
    else:
        st.caption("Qdrant yapılandırılmamış (`.env`: QDRANT_HOST / QDRANT_COLLECTION / QDRANT_ENABLED).")
    st.caption("Dosya olarak kaydet")
    file_cols = st.columns(3)
    for col, kind in zip(file_cols, OUTPUT_KINDS):
        col.checkbox(
            _KIND_LABELS[kind],
            value=False,
            key=f"out_{kind}",
            # JSONL/Parquet parça vektörlerini taşır
            disabled=kind != "txt" and not chunk_enabled,
        )

    st.divider()
    if st.button("Sonuçları temizle", width="stretch"):
        st.session_state.results = []
        st.session_state.errors = []
        st.rerun()


def _options() -> dict:
    """Kenar çubuğundaki seçimler; devre dışı bırakılmış seçenekler sayılmaz."""
    chunk = st.session_state.get("chunk_enabled", True)
    return {
        "chunk": chunk,
        "qa_mode": chunk and st.session_state.get("chunk_strategy") == "Soru-Cevap",
        "whole": chunk and st.session_state.get("chunk_strategy") == "Tek parça",
        "qdrant": chunk and QDRANT_READY and st.session_state.get("auto_qdrant", False),
        "outputs": [
            k for k in OUTPUT_KINDS
            if st.session_state.get(f"out_{k}", False) and (chunk or k == "txt")
        ],
    }


# ---- Main ----
st.title("Anlamsal Parçalama Motoru")
st.caption(
    "Resim · PDF · Video · YouTube · Metin · MP3 · Web · Veritabanı — semantik metne çevir, "
    "anlamsal parçalara böl, önizle ya da Qdrant'a / dosyaya kaydet."
)

_opts = _options()
_targets = ([f"Qdrant (`{config.QDRANT_COLLECTION}`)"] if _opts["qdrant"] else []) + [
    _KIND_LABELS[k] for k in _opts["outputs"]
]
if _targets:
    st.info(f"Otomatik kayıt: {' · '.join(_targets)}", icon=":material/save:")
else:
    st.info(
        "Önizleme modu — hiçbir yere kaydedilmez; sonuçları inceledikten sonra Sonuçlar sekmesinden kaydedebilirsiniz.",
        icon=":material/visibility:",
    )

tab_files, tab_url, tab_db, tab_results = st.tabs(
    ["Dosya yükle", "URL gir", "Veritabanı", f"Sonuçlar ({len(st.session_state.results)})"]
)


def _save_upload(uploaded_file) -> Path:
    dst_dir = Path(config.INPUTS_DIR)
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / safe_filename(uploaded_file.name, max_len=120)
    with dst.open("wb") as f:
        f.write(uploaded_file.getbuffer())
    return dst


def _record_error(source: str, e: Exception) -> None:
    tb = traceback.format_exc(limit=4)
    st.session_state.errors.append({"source": source, "error": str(e), "trace": tb})
    st.toast(f"Hata: {e}", icon="🚨")


def _run_one(
    source: str,
    status,
    label: str,
    check_cancelled: Callable[[], bool] | None = None,
) -> IngestResult | None:
    def progress_cb(msg: str, pct: float | None) -> None:
        if pct is not None:
            status.update(label=f"{label} — {msg}", state="running")

    opts = _options()
    try:
        result = process(
            source,
            progress=progress_cb,
            similarity_threshold=st.session_state.get("sim_threshold"),
            min_tokens=st.session_state.get("min_tokens"),
            max_tokens=st.session_state.get("max_tokens"),
            qa_mode=opts["qa_mode"],
            chunk=opts["chunk"],
            whole=opts["whole"],
            check_cancelled=check_cancelled,
        )
    except OperationCancelled:
        status.update(label=f"{label} — İptal Edildi", state="error")
        return None
    except Exception as e:
        _record_error(source, e)
        status.update(label=f"{label} — HATA: {e}", state="error")
        return None

    # İşlenen sonuç kayıt başarısız olsa da saklanır: Sonuçlar'dan yeniden denenebilir
    st.session_state.results.append({"id": uuid.uuid4().hex, "source": source, "result": result})

    if opts["qdrant"] or opts["outputs"]:
        try:
            publish(
                result,
                qdrant=opts["qdrant"],
                outputs=opts["outputs"],
                progress=progress_cb,
                check_cancelled=check_cancelled,
            )
        except OperationCancelled:
            status.update(label=f"{label} — Kayıt iptal edildi (önizleme saklandı)", state="error")
            return result
        except Exception as e:
            _record_error(source, e)
            status.update(label=f"{label} — Kayıt HATASI: {e} (önizleme saklandı)", state="error")
            return result

    qdrant = result.qdrant_result or {}
    parts = [f"{result.chunk_count} parça" if result.chunks else "metin çıkarıldı"]
    if qdrant.get("success"):
        parts.append(f"Qdrant ✓ ({qdrant.get('upserted', 0)} point)")
    elif qdrant.get("error"):
        parts.append("Qdrant ✗")
    if result.output_files:
        parts.append(" · ".join(_KIND_LABELS[k] for k in result.output_files))
    if not qdrant and not result.output_files:
        parts.append("önizleme")
    status.update(label=f"{label} — Tamam ({' · '.join(parts)})", state="complete")
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
        st.session_state.cancel_requested = False
        st.button("İşlemi İptal Et", key="cancel_files", on_click=request_cancel)

        for uf in uploaded:
            if st.session_state.get("cancel_requested", False):
                break
            saved = _save_upload(uf)
            label = f"{uf.name}"
            with st.status(f"İşleniyor: {label}", expanded=False) as status:
                res = _run_one(
                    str(saved),
                    status,
                    label,
                    check_cancelled=lambda: st.session_state.get("cancel_requested", False),
                )
                if res is None and st.session_state.get("cancel_requested", False):
                    break
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
        elif detected == "video":
            st.info("Doğrudan video bağlantısı algılandı → sesi çıkarılıp transkribe edilecek.")
        elif detected == "web":
            st.info("Web sayfası algılandı → Trafilatura, gerekirse Playwright.")
        else:
            st.warning("URL geçersiz görünüyor.")

    process_url = st.button(
        "URL'yi işle",
        type="primary",
        disabled=not url or detected not in {"web", "youtube", "video"},
        key="btn_url",
    )

    if process_url and url:
        st.session_state.cancel_requested = False
        st.button("İşlemi İptal Et", key="cancel_url", on_click=request_cancel)
        label = url[:80]
        with st.status(f"İşleniyor: {label}", expanded=False) as status:
            _run_one(
                url,
                status,
                label,
                check_cancelled=lambda: st.session_state.get("cancel_requested", False),
            )
        st.rerun()


# ===== Veritabanı tab =====
_NONE = "—"
_FORMAT_LABELS = {"auto": "Otomatik (HTML ise temizlenir)", "plain": "Düz metin", "html": "HTML"}
# Parçalanacak kolon için varsayılan tahmini: adı içerik taşıdığını düşündüren ilk metin kolonu
_TEXT_HINTS = ("content", "icerik", "içerik", "body", "text", "metin", "desc", "aciklama", "açıklama", "cevap", "answer")


def _default_text_column(text_cols: list[str]) -> int:
    for i, name in enumerate(text_cols):
        if any(h in name.lower() for h in _TEXT_HINTS):
            return i
    return 0


def _run_table(engine, sel: sql_source.TableSelection, status, total: int, check_cancelled) -> None:
    label = sel.table

    def progress_cb(msg: str, pct: float | None) -> None:
        status.update(label=f"{label} — {msg}", state="running")

    opts = _options()
    skipped: list = []
    try:
        batch = process_documents(
            sql_source.iter_documents(engine, sel, check_cancelled=check_cancelled, on_skip=skipped.append),
            label=label,
            source_uri=f"db://{sql_source.source_label(engine)}/{sel.table}",
            total=total,
            progress=progress_cb,
            similarity_threshold=st.session_state.get("sim_threshold"),
            min_tokens=st.session_state.get("min_tokens"),
            max_tokens=st.session_state.get("max_tokens"),
            qa_mode=opts["qa_mode"],
            chunk=opts["chunk"],
            whole=opts["whole"],
            check_cancelled=check_cancelled,
        )
    except OperationCancelled:
        status.update(label=f"{label} — İptal Edildi", state="error")
        return
    except Exception as e:
        _record_error(label, e)
        status.update(label=f"{label} — HATA: {e}", state="error")
        return
    batch.skipped = skipped

    if not batch.results:
        status.update(label=f"{label} — İşlenecek metin yok ({len(skipped)} boş satır)", state="error")
        return
    st.session_state.results.append({"id": uuid.uuid4().hex, "source": batch.source_uri, "batch": batch})

    if opts["qdrant"] or opts["outputs"]:
        try:
            publish_batch(
                batch,
                qdrant=opts["qdrant"],
                outputs=opts["outputs"],
                progress=progress_cb,
                check_cancelled=check_cancelled,
            )
        except OperationCancelled:
            status.update(label=f"{label} — Kayıt iptal edildi (önizleme saklandı)", state="error")
            return
        except Exception as e:
            _record_error(label, e)
            status.update(label=f"{label} — Kayıt HATASI: {e} (önizleme saklandı)", state="error")
            return

    parts = [f"{len(batch.results)} satır", f"{batch.chunk_count} parça"]
    if batch.qdrant_result.get("success"):
        parts.append(f"Qdrant ✓ ({batch.qdrant_result.get('upserted', 0)} point)")
    elif batch.qdrant_result.get("error"):
        parts.append("Qdrant ✗")
    if batch.output_files:
        parts.append(" · ".join(_KIND_LABELS[k] for k in batch.output_files))
    if not batch.qdrant_result and not batch.output_files:
        parts.append("önizleme")
    status.update(label=f"{label} — Tamam ({' · '.join(parts)})", state="complete")


with tab_db:
    if not sql_source.is_configured():
        st.info("Kaynak veritabanı tanımlı değil. Bağlantıyı `.env` dosyasına ekleyip uygulamayı yeniden başlatın:")
        st.code(
            "# MSSQL\n"
            "MSSQL_USER=...\n"
            "MSSQL_PWD=...\n"
            "MSSQL_SERVER=...\n"
            "MSSQL_DBNAME=...\n"
            "MSSQL_POOL_MAX=10\n"
            "MSSQL_POOL_TIMEOUT=30000\n\n"
            "# Diğer veritabanları (doluysa MSSQL_* yerine kullanılır)\n"
            "# SOURCE_DB_URL=postgresql+psycopg://kullanici:sifre@sunucu:5432/veritabani\n"
            "# Denemek için örnek veritabanı: python scripts/make_sample_db.py\n"
            "# SOURCE_DB_URL=sqlite:///data/inputs/ornek.db",
            language="bash",
        )
    else:
        try:
            engine = sql_source.get_engine()
            tables = sql_source.list_tables(engine)
        except Exception as e:
            st.error(f"Veritabanına bağlanılamadı: {e}")
            tables = None

        if tables is not None:
            st.caption(f"Bağlantı: `{sql_source.describe(engine)}`")
            if not tables:
                st.warning("Veritabanında tablo bulunamadı.")
            else:
                table = st.selectbox("Tablo", tables, key="db_table")
                columns = sql_source.list_columns(engine, table)
                names = [c.name for c in columns]
                text_cols = [c.name for c in columns if c.is_text] or names
                pk = next((c.name for c in columns if c.primary), names[0])

                # Anahtarlar tabloya bağlı: tablo değişince seçimler o tablonun varsayılanlarına döner
                c1, c2 = st.columns(2)
                id_col = c1.selectbox(
                    "Kimlik kolonu", names, index=names.index(pk), key=f"db_id_{table}",
                    help="Her satırın parçaları bu değere bağlanır (db://…/tablo/<kimlik>).",
                )
                text_col = c2.selectbox(
                    "Parçalanacak kolon", text_cols, index=_default_text_column(text_cols), key=f"db_text_{table}",
                )
                c3, c4 = st.columns(2)
                title_col = c3.selectbox(
                    "Başlık kolonu (opsiyonel)", [_NONE] + names, key=f"db_title_{table}",
                    help="Her parçanın embedding'ine bağlam olarak eklenir; aramada parçanın neye ait olduğunu belli eder.",
                )
                fmt = c4.selectbox(
                    "İçerik biçimi", list(_FORMAT_LABELS), format_func=_FORMAT_LABELS.get, key=f"db_fmt_{table}",
                )
                meta_cols = st.multiselect(
                    "Ek bilgi kolonları (parçanın yanında saklanır, metne girmez)",
                    [n for n in names if n not in (id_col, text_col)],
                    key=f"db_meta_{table}",
                )
                f1, f2 = st.columns(2)
                filter_col = f1.selectbox("Filtre kolonu (opsiyonel)", [_NONE] + names, key=f"db_fcol_{table}")
                filter_val = f2.text_input(
                    "Filtre değeri", placeholder="ör. 1 · 89-92 · 89,90,95 · NULL",
                    key=f"db_fval_{table}", disabled=filter_col == _NONE,
                    help=sql_source.FILTER_HELP,
                )
                l1, l2 = st.columns(2, vertical_alignment="bottom")
                all_rows = l2.checkbox("Tüm satırlar", value=False, key="db_all_rows")
                limit = l1.number_input(
                    "Satır sayısı", min_value=1, max_value=1_000_000, value=20, step=10,
                    key="db_limit", disabled=all_rows,
                    help="Önizleme için küçük başlayın; beğenince 'Tüm satırlar' ile tamamını işleyin.",
                )

                sel = sql_source.TableSelection(
                    table=table,
                    id_column=id_col,
                    text_column=text_col,
                    title_column=None if title_col == _NONE else title_col,
                    meta_columns=meta_cols,
                    filters={} if filter_col == _NONE else {filter_col: filter_val},
                    limit=None if all_rows else int(limit),
                    content_format=fmt,
                )
                try:
                    matched = sql_source.count_rows(engine, sel)
                except Exception as e:
                    matched = None
                    st.warning(f"Satırlar sayılamadı: {e}")
                if matched is not None:
                    to_process = matched if all_rows else min(matched, int(limit))
                    st.caption(f"Eşleşen satır: **{matched:,}** · işlenecek: **{to_process:,}**")

                run_table = st.button(
                    "Tabloyu işle", type="primary", key="btn_db", disabled=not matched,
                )
                if run_table:
                    st.session_state.cancel_requested = False
                    st.button("İşlemi İptal Et", key="cancel_db", on_click=request_cancel)
                    with st.status(f"İşleniyor: {table}", expanded=False) as status:
                        _run_table(
                            engine,
                            sel,
                            status,
                            total=to_process,
                            check_cancelled=lambda: st.session_state.get("cancel_requested", False),
                        )
                    st.rerun()


# ===== Sonuçlar tab =====
_MD_SPECIAL = str.maketrans({ch: "\\" + ch for ch in "\\`*_{}[]<>()#+-.!|~$"})
# Toplu sonuçlarda satır kartı sayısı sınırı (tarayıcıyı yormamak için)
_MAX_ROWS_SHOWN = 100


def _md_escape(text: str) -> str:
    """Parça metni olduğu gibi görünsün: markdown/LaTeX olarak yorumlanmasın."""
    return text.translate(_MD_SPECIAL).replace("\n", "  \n")


def _publish_later(entry: dict, **targets) -> None:
    """Önizlenmiş sonucu kaynağı yeniden işlemeden kaydeder."""
    try:
        with st.spinner("Kaydediliyor…"):
            if "batch" in entry:
                publish_batch(entry["batch"], **targets)
            else:
                publish(entry["result"], **targets)
    except Exception as e:
        _record_error(entry["source"], e)
    st.rerun()


def _qdrant_section(entry: dict, qdrant_info: dict, chunk_count: int) -> None:
    """Qdrant durumu + "kaydetmek ister misiniz?"."""
    if qdrant_info.get("success"):
        st.success(f"Qdrant ✓ — {qdrant_info.get('upserted', 0)} point yüklendi → `{config.QDRANT_COLLECTION}`")
    elif chunk_count and QDRANT_READY:
        q_cols = st.columns([4, 1], vertical_alignment="center")
        if qdrant_info.get("error"):
            q_cols[0].warning(f"Qdrant ✗ — {qdrant_info['error']}")
            q_label = "Tekrar dene"
        else:
            q_cols[0].info(
                f"{chunk_count} parça henüz Qdrant'a kaydedilmedi. "
                f"Kaydetmek ister misiniz? → `{config.QDRANT_COLLECTION}`"
            )
            q_label = "Qdrant'a kaydet"
        if q_cols[1].button(q_label, key=f"q_{entry['id']}", type="primary", width="stretch"):
            _publish_later(entry, qdrant=True)


def _chunk_cards(chunks) -> None:
    # st.dataframe kullanılmaz: Arrow serileştirmesi bu ortamda arrow.dll içinde süreci çökertiyor
    for c in chunks:
        with st.container(border=True):
            st.caption(f"#{c.chunk_index} · {c.token_count} token · {len(c.text):,} karakter")
            if c.question:
                st.markdown(f"**Soru:** {_md_escape(c.question)}")
            st.markdown(_md_escape(c.text))


def _file_buttons(entry: dict, output_files: dict, has_chunks: bool) -> None:
    """Dosyalar: kaydedilmişse indir, değilse oluştur."""
    eid = entry["id"]
    file_cols = st.columns(len(OUTPUT_KINDS))
    for col, kind in zip(file_cols, OUTPUT_KINDS):
        path = output_files.get(kind)
        if path and Path(path).exists():
            with open(path, "rb") as f:
                col.download_button(
                    f"{_KIND_LABELS[kind]} indir",
                    f.read(),
                    file_name=Path(path).name,
                    mime="application/octet-stream",
                    key=f"dl_{kind}_{eid}",
                    width="stretch",
                )
        elif col.button(
            f"{_KIND_LABELS[kind]} oluştur",
            key=f"mk_{kind}_{eid}",
            disabled=kind != "txt" and not has_chunks,
            width="stretch",
        ):
            _publish_later(entry, outputs=[kind])


def _render_result(entry: dict) -> None:
    res: IngestResult = entry["result"]
    doc = res.document

    with st.container(border=True):
        top = st.columns([3, 1, 1, 1])
        top[0].markdown(f"**{doc.title or entry['source']}**")
        top[0].caption(entry["source"])
        top[1].metric("Tip", doc.source_type)
        top[2].metric("Parça", res.chunk_count)
        top[3].metric("Ort. karakter", round(res.avg_chunk_chars, 1))

        meta_cols = st.columns(2)
        meta_cols[0].caption(f"Dil: `{doc.language or 'auto'}`")
        meta_cols[1].caption(f"Toplam metin: {len(doc.text):,} karakter")

        _qdrant_section(entry, res.qdrant_result or {}, res.chunk_count)

        if res.chunks:
            with st.expander(f"Parçalar ({res.chunk_count})", expanded=True):
                with st.container(height=520):
                    _chunk_cards(res.chunks)

        with st.expander("Tam metin"):
            st.text(doc.text)

        with st.expander("Kaynak meta"):
            st.json(doc.extra)

        _file_buttons(entry, res.output_files, bool(res.chunks))


def _render_batch(entry: dict) -> None:
    batch: BatchResult = entry["batch"]

    with st.container(border=True):
        top = st.columns([3, 1, 1, 1])
        top[0].markdown(f"**{batch.label}**")
        top[0].caption(batch.source_uri)
        top[1].metric("Satır", len(batch.results))
        top[2].metric("Parça", batch.chunk_count)
        top[3].metric("Ort. karakter", round(batch.avg_chunk_chars, 1))

        if batch.skipped:
            shown = ", ".join(map(str, batch.skipped[:20])) + ("…" if len(batch.skipped) > 20 else "")
            st.caption(f"Metni boş olduğu için atlanan {len(batch.skipped)} satır (kimlik: {shown})")

        _qdrant_section(entry, batch.qdrant_result, batch.chunk_count)

        with st.expander(f"Satırlar ({len(batch.results)})", expanded=True):
            with st.container(height=560):
                for r in batch.results[:_MAX_ROWS_SHOWN]:
                    with st.container(border=True):
                        st.markdown(f"**{_md_escape(r.document.title or '')}**")
                        meta = r.document.extra.get("meta") or {}
                        info = [f"kimlik {r.document.extra.get('pk')}", f"{r.chunk_count} parça"]
                        info += [f"{k}: {v}" for k, v in meta.items()]
                        st.caption(" · ".join(info))
                        if r.chunks:
                            _chunk_cards(r.chunks)
                        else:
                            st.markdown(_md_escape(r.document.text[:1000]))
            if len(batch.results) > _MAX_ROWS_SHOWN:
                st.caption(
                    f"İlk {_MAX_ROWS_SHOWN} satır gösteriliyor; kayıt ve dosyalar {len(batch.results):,} satırın tamamını içerir."
                )

        _file_buttons(entry, batch.output_files, bool(batch.chunk_count))


with tab_results:
    if st.session_state.errors:
        with st.expander(f"Hatalar ({len(st.session_state.errors)})", expanded=False):
            for err in st.session_state.errors:
                st.error(f"**{err['source']}**\n\n{err['error']}")
                st.code(err["trace"], language="text")

    if not st.session_state.results:
        st.info("Henüz işlenmiş kaynak yok. Dosya yükle, URL gir ya da bir veritabanı tablosu işle.")
    else:
        for entry in reversed(st.session_state.results):
            if "batch" in entry:
                _render_batch(entry)
            else:
                _render_result(entry)
