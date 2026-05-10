from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
from sqlalchemy import text

from app.settings import build_engine


def get_engine():
    return build_engine()


def apply_sql_file(engine, path: str | Path) -> None:
    sql = Path(path).read_text(encoding="utf-8")
    statements = [part.strip() for part in sql.split(";") if part.strip()]
    with engine.begin() as conn:
        for stmt in statements:
            conn.execute(text(stmt))


def fetch_df(engine, sql: str, params: dict | None = None) -> pd.DataFrame:
    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn, params=params or {})


def chunked_rows(rows: list[dict], size: int) -> Iterable[list[dict]]:
    for index in range(0, len(rows), size):
        yield rows[index : index + size]
