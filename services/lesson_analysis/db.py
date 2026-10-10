"""LMS veritabanı erişimi: yalnızca stored procedure çağrılır, değerler parametre olarak gider."""
from __future__ import annotations

import re
from contextlib import contextmanager
from typing import Any, Iterator

import pyodbc

from . import settings

_PROC_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class DbError(RuntimeError):
    """SP'nin THROW ettiği hata; number = 50xxx kodu (bulunabildiyse)."""

    def __init__(self, message: str, number: int | None = None) -> None:
        super().__init__(message)
        self.number = number


def connect() -> pyodbc.Connection:
    if not settings.MSSQL_SERVER:
        raise RuntimeError("LA_MSSQL_SERVER tanımlı değil (services/lesson_analysis/.env).")
    conn_str = (
        f"DRIVER={{{settings.ODBC_DRIVER}}};SERVER={settings.MSSQL_SERVER};DATABASE={settings.MSSQL_DB};"
        f"UID={settings.MSSQL_USER};PWD={settings.MSSQL_PWD};"
        f"Encrypt={'yes' if settings.MSSQL_ENCRYPT else 'no'};TrustServerCertificate=yes"
    )
    # SP'ler kendi transaction'larını yönetir
    return pyodbc.connect(conn_str, autocommit=True)


@contextmanager
def connection() -> Iterator[pyodbc.Connection]:
    conn = connect()
    try:
        yield conn
    finally:
        conn.close()


def _clean_message(err: pyodbc.Error) -> tuple[str, int | None]:
    raw = str(err.args[1] if len(err.args) > 1 else err)
    # "[42000] [Microsoft][ODBC Driver 17 for SQL Server][SQL Server]Mesaj (50017) (SQLExecDirectW)"
    msg = re.sub(r"\[[^\]]*\]", "", raw)
    msg = re.sub(r"\(SQLExecDirectW\)|\(SQLMoreResults\)", "", msg).strip()
    m = re.search(r"\((5\d{4})\)", raw)
    return msg, int(m.group(1)) if m else None


def call(conn: pyodbc.Connection, proc: str, **params: Any) -> list[list[dict[str, Any]]]:
    """EXEC dbo.<proc> @Ad=? ... ; bütün sonuç kümelerini satır sözlükleri olarak döndürür."""
    if not _PROC_NAME.match(proc):
        raise ValueError(f"Geçersiz prosedür adı: {proc}")
    args = ", ".join(f"@{name}=?" for name in params)
    sql = f"SET NOCOUNT ON; EXEC dbo.{proc} {args}"
    cur = conn.cursor()
    try:
        cur.execute(sql, *params.values())
        sets: list[list[dict[str, Any]]] = []
        while True:
            if cur.description:
                cols = [c[0] for c in cur.description]
                sets.append([dict(zip(cols, row)) for row in cur.fetchall()])
            if not cur.nextset():
                break
        return sets
    except pyodbc.Error as err:
        msg, number = _clean_message(err)
        raise DbError(msg, number) from err
    finally:
        cur.close()


def one(conn: pyodbc.Connection, proc: str, **params: Any) -> dict[str, Any] | None:
    """İlk sonuç kümesinin ilk satırı."""
    sets = call(conn, proc, **params)
    return sets[0][0] if sets and sets[0] else None


def rows(conn: pyodbc.Connection, proc: str, **params: Any) -> list[dict[str, Any]]:
    """İlk sonuç kümesi."""
    sets = call(conn, proc, **params)
    return sets[0] if sets else []
