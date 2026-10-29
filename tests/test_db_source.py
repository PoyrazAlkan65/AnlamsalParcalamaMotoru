import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from connectors import sql_source
from connectors.sql_source import TableSelection
from core.pipeline import process_documents, publish_batch
from core.types import QdrantPoint
from processing.html_clean import clean_text, html_to_text


@pytest.fixture
def engine(tmp_path: Path):
    db = tmp_path / "t.db"
    with sqlite3.connect(db) as conn:
        conn.executescript("""
            CREATE TABLE Yazilar (Id INTEGER PRIMARY KEY, Baslik TEXT, Icerik TEXT, Aktif INTEGER, Silindi TEXT);
            INSERT INTO Yazilar VALUES
              (1, 'Uyku', '<p>Uyku önemlidir.</p><p>Düzenli uyuyun.</p>', 1, NULL),
              (2, 'Su', 'Günde iki litre su için.', 1, NULL),
              (3, 'Eski', 'Kaldırıldı.', 0, NULL),
              (4, 'Boş', '', 1, NULL),
              (5, 'Silinmiş', 'Silindi.', 1, '2026-01-01');
            CREATE VIEW AktifYazilar AS SELECT * FROM Yazilar WHERE Aktif = 1;
        """)
    return sql_source.get_engine(f"sqlite:///{db.as_posix()}")


def _sel(**kw) -> TableSelection:
    return TableSelection(table="Yazilar", id_column="Id", text_column="Icerik", title_column="Baslik", **kw)


# ---- şema ----

def test_lists_tables_views_and_columns(engine):
    assert sql_source.list_tables(engine) == ["AktifYazilar", "Yazilar"]
    cols = {c.name: c for c in sql_source.list_columns(engine, "Yazilar")}
    assert cols["Id"].primary and not cols["Id"].is_text
    assert cols["Icerik"].is_text


def test_unknown_table_or_column_is_rejected(engine):
    with pytest.raises(ValueError, match="Tablo bulunamadı"):
        sql_source.count_rows(engine, TableSelection(table="Yazilar; DROP TABLE Yazilar", id_column="Id", text_column="Icerik"))
    with pytest.raises(ValueError, match="Kolon bulunamadı"):
        list(sql_source.iter_documents(engine, TableSelection(table="Yazilar", id_column="Id", text_column="Yok")))


# ---- okuma ----

def test_filters_are_typed_and_null_aware(engine):
    assert sql_source.count_rows(engine, _sel()) == 5
    assert sql_source.count_rows(engine, _sel(filters={"Aktif": "1"})) == 4
    assert sql_source.count_rows(engine, _sel(filters={"Aktif": "1", "Silindi": "NULL"})) == 3


def test_filter_ranges_lists_and_errors(engine):
    def ids(value):
        return [d.extra["pk"] for d in sql_source.iter_documents(engine, _sel(filters={"Id": value}))]

    assert ids("2-4") == [2, 3]          # 4'ün metni boş → atlanır
    assert ids("5 - 1") == [1, 2, 3, 5]  # ters yazılan aralık da olur
    assert ids("1,3,5") == [1, 3, 5]
    assert ids("3") == [3]
    with pytest.raises(ValueError, match="aralık"):
        sql_source.count_rows(engine, _sel(filters={"Id": "abc"}))
    # metin kolonunda tire aralık sayılmaz (ör. tarih 2026-01-01)
    assert sql_source.count_rows(engine, _sel(filters={"Silindi": "2026-01-01"})) == 1


def test_rows_become_documents_and_empty_rows_are_skipped(engine):
    skipped = []
    docs = list(sql_source.iter_documents(engine, _sel(filters={"Aktif": "1"}, meta_columns=["Aktif"]), on_skip=skipped.append))

    assert [d.extra["pk"] for d in docs] == [1, 2, 5]
    assert skipped == [4]
    first = docs[0]
    assert first.text == "Uyku önemlidir.\n\nDüzenli uyuyun."  # HTML temizlendi, paragraflar korundu
    assert first.title == "Uyku"
    assert first.source_type == "db"
    assert first.source_uri == "db://t.db/Yazilar/1"
    assert first.extra["context"] == "Uyku"
    assert first.extra["meta"] == {"Aktif": 1}


def test_limit_and_views(engine):
    docs = list(sql_source.iter_documents(engine, TableSelection(table="AktifYazilar", id_column="Id", text_column="Icerik", limit=2)))
    assert [d.extra["pk"] for d in docs] == [1, 2]
    assert docs[0].title == "AktifYazilar #1"  # başlık kolonu seçilmediyse


def test_relative_sqlite_path_resolves_to_project_root():
    import config
    engine = sql_source.get_engine("sqlite:///data/inputs/__olmayan__.db")
    assert Path(engine.url.database) == config.ROOT_DIR / "data" / "inputs" / "__olmayan__.db"
    assert sql_source.describe(engine) == f"sqlite: {config.ROOT_DIR / 'data' / 'inputs' / '__olmayan__.db'}"


def test_mssql_env_vars_build_engine(monkeypatch):
    import config
    for name, value in {
        "SOURCE_DB_URL": "", "MSSQL_SERVER": "db.ornek.local,1444", "MSSQL_DBNAME": "Musteri",
        "MSSQL_USER": "okuyucu", "MSSQL_PWD": "p@ss:w/rd;1", "MSSQL_POOL_MAX": 7, "MSSQL_POOL_TIMEOUT": 15000,
        "MSSQL_ENCRYPT": "", "MSSQL_TRUST_SERVER_CERTIFICATE": "yes",
    }.items():
        monkeypatch.setattr(config, name, value)
    monkeypatch.setattr(sql_source, "_engines", {})

    assert sql_source.is_configured()
    url = sql_source.mssql_url()
    assert (url.host, url.port, url.database, url.username) == ("db.ornek.local", 1444, "Musteri", "okuyucu")
    assert url.password == "p@ss:w/rd;1"  # özel karakterler bozulmaz
    assert url.query["driver"] == "ODBC Driver 17 for SQL Server"
    assert "Encrypt" not in url.query

    engine = sql_source.get_engine()
    assert engine.url.get_backend_name() == "mssql"
    assert engine.pool.size() == 7 and engine.pool._timeout == 15
    assert "p@ss" not in sql_source.describe(engine)
    assert sql_source.source_label(engine) == "Musteri"


def test_server_formats():
    assert sql_source._split_server("sunucu") == ("sunucu", None)
    assert sql_source._split_server("sunucu:1433") == ("sunucu", 1433)
    assert sql_source._split_server(r"sunucu\SQLEXPRESS") == (r"sunucu\SQLEXPRESS", None)
    assert sql_source._split_server("10.0.0.5, 1500") == ("10.0.0.5", 1500)


def test_source_db_url_takes_precedence(monkeypatch, tmp_path: Path):
    import config
    monkeypatch.setattr(config, "MSSQL_SERVER", "sunucu")
    monkeypatch.setattr(config, "SOURCE_DB_URL", f"sqlite:///{(tmp_path / 'x.db').as_posix()}")
    monkeypatch.setattr(sql_source, "_engines", {})
    assert sql_source.get_engine().url.get_backend_name() == "sqlite"


def test_describe_hides_password():
    engine = sql_source.get_engine("mssql+pyodbc://kullanici:gizli123@sunucu/vt?driver=ODBC+Driver+17+for+SQL+Server")
    assert "gizli123" not in sql_source.describe(engine)
    assert sql_source.source_label(engine) == "vt"


# ---- HTML ----

def test_html_to_text_keeps_structure():
    html = "<h2>Başlık</h2><p>Bir <b>kalın</b> cümle.<br>Yeni satır.</p><ul><li>A</li><li>B</li></ul><script>x()</script>"
    assert html_to_text(html) == "Başlık\n\nBir kalın cümle.\nYeni satır.\n\n- A\n- B"
    # kaynak HTML'deki satır sonları paragrafı bölmez; <pre> korunur
    assert html_to_text("<p>yedi ile\n   dokuz saat</p><pre>a\n  b</pre>") == "yedi ile dokuz saat\n\na\nb"


def test_clean_text_formats():
    assert clean_text("Düz <metin>", "auto") == "Düz <metin>"
    assert clean_text("<p>a</p>", "plain") == "<p>a</p>"
    assert clean_text("<p>a</p>", "auto") == "a"


# ---- toplu işleme ----

def _embedder():
    e = MagicMock()
    e.encode.side_effect = lambda texts: np.arange(len(texts) * 4, dtype=np.float32).reshape(len(texts), 4)
    return e


def test_batch_embeds_once_and_keeps_rows_separate(engine, tmp_path: Path):
    docs = sql_source.iter_documents(engine, _sel(filters={"Aktif": "1"}))
    embedder = _embedder()
    with (
        patch("core.pipeline.get_embedder", return_value=embedder),
        patch("core.pipeline.upsert_to_qdrant", return_value={"success": True, "upserted": 3, "error": None}) as upsert,
    ):
        batch = process_documents(docs, label="Yazilar", source_uri="db://t.db/Yazilar", whole=True)
        assert batch.chunk_count == 3 and not any(r.points for r in batch.results)
        publish_batch(batch, qdrant=True, outputs=["jsonl"], out_dir=tmp_path)

    embedder.encode.assert_called_once()
    texts = embedder.encode.call_args.args[0]
    assert texts[0] == "Uyku\nUyku önemlidir.\n\nDüzenli uyuyun."  # başlık bağlam olarak eklendi
    points = upsert.call_args.args[0]
    assert [p.payload["source_uri"] for p in points] == [f"db://t.db/Yazilar/{i}" for i in (1, 2, 5)]
    assert len({p.id for p in points}) == 3
    assert batch.output_files["jsonl"].exists()


def test_batch_rejects_vectors_without_chunks(engine, tmp_path: Path):
    batch = process_documents(sql_source.iter_documents(engine, _sel()), label="Y", source_uri="db://t", chunk=False)
    with pytest.raises(RuntimeError, match="parçalama gerekli"):
        publish_batch(batch, qdrant=True, out_dir=tmp_path)


# ---- Qdrant: her kaynağın eski parçaları silinir ----

def test_upsert_deletes_old_points_of_every_source():
    from export import qdrant_writer

    client = MagicMock()
    points = [
        QdrantPoint(id=f"p{i}", vector=[0.0], payload={"source_uri": uri})
        for i, uri in enumerate(["db://x/T/1", "db://x/T/1", "db://x/T/2", "db://x/T/3"])
    ]
    with (
        patch.object(qdrant_writer.config, "QDRANT_ENABLED", True),
        patch.object(qdrant_writer, "get_qdrant_client", return_value=client),
        patch.object(qdrant_writer, "ensure_collection", return_value=True),
    ):
        res = qdrant_writer.upsert_to_qdrant(points, collection_name="c")

    assert res["success"] and res["upserted"] == 4
    selector = client.delete.call_args.kwargs["points_selector"]
    assert selector.must[0].match.any == ["db://x/T/1", "db://x/T/2", "db://x/T/3"]
