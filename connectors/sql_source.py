"""SQL veritabanından tablo satırlarını okur; her satır bir `Document` olur.

Bağlantı kurulumun `.env`'inden gelir: MSSQL için projelerdeki ortak MSSQL_* değişkenleri, diğer veritabanları
(PostgreSQL, MySQL, SQLite…) için SOURCE_DB_URL (SQLAlchemy URL'i).
Yalnızca SELECT çalıştırılır. Tablo ve kolon adları şemadan yansıtılır (reflection) ve doğrulanır; kullanıcı
girdisi SQL'e metin olarak eklenmez, filtre değerleri parametre olarak bağlanır.
"""
from __future__ import annotations

import datetime as _dt
import decimal
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

import config
from core.types import Document, OperationCancelled
from processing.html_clean import clean_text

_engines: dict[str, Any] = {}
_lock = threading.Lock()

CONTENT_FORMATS = ("auto", "plain", "html")


@dataclass
class ColumnInfo:
    name: str
    type: str
    is_text: bool
    primary: bool = False


@dataclass
class TableSelection:
    """Hangi tablonun hangi kolonunun parçalanacağı."""
    table: str
    id_column: str
    text_column: str
    title_column: str | None = None
    meta_columns: list[str] = field(default_factory=list)
    filters: dict[str, str] = field(default_factory=dict)  # kolon → değer; biçimler: FILTER_HELP
    limit: int | None = None
    content_format: str = "auto"


def is_configured() -> bool:
    return bool(config.SOURCE_DB_URL or config.MSSQL_SERVER)


def _split_server(server: str) -> tuple[str, int | None]:
    """"sunucu,1433" / "sunucu:1433" → (sunucu, 1433); "sunucu\\INSTANCE" olduğu gibi kalır."""
    server = server.strip()
    for sep in (",", ":"):
        host, found, port = server.rpartition(sep)
        if found and port.strip().isdigit():
            return host.strip(), int(port)
    return server, None


def mssql_url():
    """MSSQL_* değişkenlerinden SQLAlchemy URL'i; parola URL.create ile kaçışlanır (özel karakterler güvenli)."""
    from sqlalchemy.engine import URL

    host, port = _split_server(config.MSSQL_SERVER)
    query = {"driver": config.MSSQL_ODBC_DRIVER}
    if config.MSSQL_TRUST_SERVER_CERTIFICATE:
        query["TrustServerCertificate"] = config.MSSQL_TRUST_SERVER_CERTIFICATE
    if config.MSSQL_ENCRYPT:
        query["Encrypt"] = config.MSSQL_ENCRYPT
    return URL.create(
        "mssql+pyodbc",
        username=config.MSSQL_USER or None,
        password=config.MSSQL_PWD or None,
        host=host,
        port=port,
        database=config.MSSQL_DBNAME or None,
        query=query,
    )


def get_engine(url: str | None = None):
    """Bağlantı başına tek engine (bağlantı havuzu).

    Öncelik: verilen url → SOURCE_DB_URL → MSSQL_* değişkenleri.
    """
    from sqlalchemy import create_engine
    from sqlalchemy.engine import make_url

    url = url or config.SOURCE_DB_URL
    if not url and not config.MSSQL_SERVER:
        raise RuntimeError("Kaynak veritabanı tanımlı değil (.env: MSSQL_SERVER/MSSQL_DBNAME/… ya da SOURCE_DB_URL).")
    key = url or "MSSQL_*"
    with _lock:
        if key not in _engines:
            if url:
                parsed = make_url(url)
                # sqlite göreli yolu çalışma dizinine değil proje köküne göre çözülsün
                db = parsed.database
                if parsed.get_backend_name() == "sqlite" and db and db != ":memory:" and not Path(db).is_absolute():
                    parsed = parsed.set(database=str(config.ROOT_DIR / db))
                _engines[key] = create_engine(parsed, pool_pre_ping=True)
            else:
                _engines[key] = create_engine(
                    mssql_url(),
                    pool_pre_ping=True,
                    pool_size=max(1, config.MSSQL_POOL_MAX),
                    max_overflow=0,
                    pool_timeout=config.MSSQL_POOL_TIMEOUT / 1000,
                )
        return _engines[key]


def describe(engine) -> str:
    """Parolası gizlenmiş bağlantı adresi (arayüzde göstermek için)."""
    if engine.url.get_backend_name() == "sqlite":
        return f"sqlite: {engine.url.database}"
    return engine.url.render_as_string(hide_password=True)


def source_label(engine) -> str:
    """source_uri'de kullanılan veritabanı adı."""
    if config.SOURCE_DB_LABEL:
        return config.SOURCE_DB_LABEL
    database = engine.url.database or engine.url.get_backend_name()
    # sqlite'ta database bir dosya yoludur
    return database.replace("\\", "/").rsplit("/", 1)[-1]


def list_tables(engine) -> list[str]:
    from sqlalchemy import inspect

    insp = inspect(engine)
    return sorted(set(insp.get_table_names()) | set(insp.get_view_names()), key=str.lower)


def _reflect(engine, table: str):
    from sqlalchemy import MetaData, Table

    if table not in list_tables(engine):
        raise ValueError(f"Tablo bulunamadı: {table}")
    return Table(table, MetaData(), autoload_with=engine)


def list_columns(engine, table: str) -> list[ColumnInfo]:
    from sqlalchemy import String, Text

    tbl = _reflect(engine, table)
    return [
        ColumnInfo(
            name=c.name,
            type=str(c.type),
            is_text=isinstance(c.type, (String, Text)),
            primary=c.primary_key,
        )
        for c in tbl.columns
    ]


def _column(tbl, name: str):
    if name not in tbl.c:
        raise ValueError(f"Kolon bulunamadı: {tbl.name}.{name}")
    return tbl.c[name]


FILTER_HELP = "Tek değer (89), aralık (89-92, yalnızca sayısal kolonlar), liste (89,90,95) ya da boşlar için NULL."
_RANGE = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*-\s*(-?\d+(?:\.\d+)?)\s*$")


def _python_type(col):
    try:
        return col.type.python_type
    except NotImplementedError:
        return None


def _coerce(col, value: str):
    """Arayüzden gelen metni kolonun tipine çevirir (ör. IsActive = "1" → True)."""
    py_type = _python_type(col)
    value = value.strip()
    if py_type is bool:
        return value.lower() in ("1", "true", "evet", "yes")
    if py_type in (int, float, decimal.Decimal):
        try:
            return py_type(value)
        except (ValueError, decimal.InvalidOperation):
            raise ValueError(
                f"'{value}' değeri {col.name} kolonunun tipine ({col.type}) uymuyor. {FILTER_HELP}"
            ) from None
    return value


def _condition(col, value: str):
    """Filtre değerini koşula çevirir: NULL · liste (a,b,c) · sayısal aralık (a-b) · tek değer."""
    raw = str(value or "").strip()
    if raw.upper() in ("", "NULL"):
        return col.is_(None)
    if "," in raw:
        return col.in_([_coerce(col, v) for v in raw.split(",") if v.strip()])
    match = _RANGE.match(raw)
    if match and _python_type(col) in (int, float, decimal.Decimal):
        low, high = sorted((_coerce(col, match.group(1)), _coerce(col, match.group(2))))
        return col.between(low, high)
    return col == _coerce(col, raw)


def _where(tbl, sel: TableSelection) -> list:
    return [_condition(_column(tbl, name), value) for name, value in sel.filters.items()]


def count_rows(engine, sel: TableSelection) -> int:
    from sqlalchemy import func, select

    tbl = _reflect(engine, sel.table)
    stmt = select(func.count()).select_from(tbl).where(*_where(tbl, sel))
    with engine.connect() as conn:
        return int(conn.execute(stmt).scalar_one())


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, decimal.Decimal):
        return float(value)
    if isinstance(value, (_dt.date, _dt.datetime, _dt.time)):
        return value.isoformat()
    if isinstance(value, (bytes, bytearray, memoryview)):
        return None
    return str(value)


def iter_documents(
    engine,
    sel: TableSelection,
    check_cancelled: Callable[[], bool] | None = None,
    on_skip: Callable[[Any], None] | None = None,
) -> Iterator[Document]:
    """Seçili satırları id sırasıyla okur; metni boş olan satırları atlar (on_skip ile bildirir)."""
    from sqlalchemy import select

    tbl = _reflect(engine, sel.table)
    id_col = _column(tbl, sel.id_column)
    text_col = _column(tbl, sel.text_column)
    cols = [id_col, text_col]
    if sel.title_column:
        cols.append(_column(tbl, sel.title_column))
    cols += [_column(tbl, m) for m in sel.meta_columns if m not in {c.name for c in cols}]

    stmt = select(*cols).where(*_where(tbl, sel)).order_by(id_col)
    if sel.limit:
        stmt = stmt.limit(sel.limit)

    label = source_label(engine)
    with engine.connect() as conn:
        for row in conn.execution_options(yield_per=500).execute(stmt).mappings():
            if check_cancelled and check_cancelled():
                raise OperationCancelled("İşlem kullanıcı tarafından iptal edildi.")
            pk = row[sel.id_column]
            text = clean_text(str(row[sel.text_column] or ""), sel.content_format)
            if not text:
                if on_skip:
                    on_skip(pk)
                continue
            title = row[sel.title_column] if sel.title_column else None
            title = str(title).strip() if title is not None and str(title).strip() else None
            yield Document(
                text=text,
                source_type="db",
                source_uri=f"db://{label}/{sel.table}/{pk}",
                title=title or f"{sel.table} #{pk}",
                extra={
                    "table": sel.table,
                    "pk": _json_safe(pk),
                    "text_column": sel.text_column,
                    # Başlık her parçanın embedding'ine bağlam olarak eklenir
                    "context": title,
                    "meta": {m: _json_safe(row[m]) for m in sel.meta_columns},
                },
            )
