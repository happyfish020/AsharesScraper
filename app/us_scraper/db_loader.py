from __future__ import annotations

import re
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from sqlalchemy import text


def _sanitize_identifier(name: str) -> str:
    out = re.sub(r"[^a-zA-Z0-9_]", "_", str(name or "").strip().lower()).strip("_")
    return out or "unnamed"


def _sql_col(name: str) -> str:
    return f"`{_sanitize_identifier(name)}`"


def _read_csv_normalized(csv_path: str | Path) -> pd.DataFrame:
    path = Path(csv_path)
    raw = pd.read_csv(path, low_memory=False)
    if raw.empty:
        return raw
    date_col = "date" if "date" in raw.columns else raw.columns[0]
    raw[date_col] = pd.to_datetime(raw[date_col], errors="coerce")
    raw = raw.dropna(subset=[date_col]).set_index(date_col)
    raw.index.name = "date"
    raw.columns = [_sanitize_identifier(c) for c in raw.columns]
    for col in raw.columns:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    return raw.dropna(how="all", axis=1).sort_index()


def upsert_csv_to_mysql(
    engine,
    csv_path: str | Path,
    table_name: str,
    *,
    start_date: Optional[date | datetime | str] = None,
    end_date: Optional[date | datetime | str] = None,
) -> int:
    df = _read_csv_normalized(csv_path)
    if df.empty:
        return 0
    df = df.reset_index()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    if start_date is not None:
        df = df.loc[df["date"] >= pd.to_datetime(start_date).date()]
    if end_date is not None:
        df = df.loc[df["date"] <= pd.to_datetime(end_date).date()]
    if df.empty:
        return 0

    table = _sanitize_identifier(table_name)
    value_cols = [c for c in df.columns if c != "date"]
    if not value_cols:
        return 0

    col_defs = ["`date` DATE NOT NULL PRIMARY KEY"] + [f"{_sql_col(c)} DOUBLE NULL" for c in value_cols]
    create_sql = f"CREATE TABLE IF NOT EXISTS `{table}` (" + ", ".join(col_defs) + ") ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci"
    cols = ["date"] + value_cols
    col_list = ", ".join("`date`" if c == "date" else _sql_col(c) for c in cols)
    placeholders = ", ".join(f":{_sanitize_identifier(c)}" for c in cols)
    update_clause = ", ".join(f"{_sql_col(c)} = VALUES({_sql_col(c)})" for c in value_cols)
    insert_sql = f"INSERT INTO `{table}` ({col_list}) VALUES ({placeholders}) ON DUPLICATE KEY UPDATE {update_clause}"

    records = []
    clean = df[cols].astype(object).where(pd.notna(df[cols]), None)
    for row in clean.to_dict(orient="records"):
        records.append({_sanitize_identifier(k): v for k, v in row.items()})

    with engine.begin() as conn:
        conn.execute(text(create_sql))
        conn.execute(text(insert_sql), records)
    return len(records)
