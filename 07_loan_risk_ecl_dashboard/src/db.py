"""
db.py
Thin data-access layer over the SQLite loan book.

All analytic SQL lives in standalone ``.sql`` files under ``sql/`` so it is
reviewable independently of the Python code. This module is responsible for:

  1. Locating the project's ``sql/`` directory and ``data/loan_book.db``.
  2. Loading a named ``.sql`` file's text.
  3. Executing it via ``pandas.read_sql_query`` (or ``sqlite3`` directly for
     DDL scripts) with *bound* parameters -- never with Python string
     formatting/concatenation of user- or UI-supplied values.

SQLite's parameter style is "named" (``:param``), which is also valid
syntax on most other engines (Oracle, some ORMs), and maps cleanly onto
SQL Server's ``@param`` / PostgreSQL's ``%(param)s`` styles if this project
were ever migrated off SQLite.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Mapping, Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SQL_DIR = PROJECT_ROOT / "sql"
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "loan_book.db"


def get_connection(db_path: Path | str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Open a SQLite connection with foreign keys enforced."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def load_sql(filename: str) -> str:
    """Read a ``.sql`` file from the ``sql/`` directory and return its text."""
    path = SQL_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"SQL file not found: {path}")
    return path.read_text(encoding="utf-8")


def run_query(
    filename: str,
    params: Optional[Mapping[str, Any]] = None,
    conn: Optional[sqlite3.Connection] = None,
    db_path: Path | str = DEFAULT_DB_PATH,
) -> pd.DataFrame:
    """
    Execute a named ``.sql`` file with bound parameters and return a
    DataFrame. Parameters are always passed through the DB-API's own
    parameter-binding mechanism (``pandas.read_sql_query(..., params=...)``),
    never interpolated into the SQL text.
    """
    sql_text = load_sql(filename)
    owns_conn = conn is None
    if conn is None:
        conn = get_connection(db_path)
    try:
        return pd.read_sql_query(sql_text, conn, params=dict(params or {}))
    finally:
        if owns_conn:
            conn.close()


def run_script(filename: str, conn: sqlite3.Connection) -> None:
    """Execute a multi-statement DDL script (e.g. schema.sql) via executescript.

    DDL scripts contain no user-supplied values, so ``executescript`` (which
    does not support parameter binding) is appropriate here -- this is
    distinct from ``run_query``, which is used for every parameterised,
    data-dependent query in the app.
    """
    sql_text = load_sql(filename)
    conn.executescript(sql_text)
    conn.commit()
