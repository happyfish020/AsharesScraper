from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys
import time
from typing import Iterable, List

import pandas as pd
import tushare as ts
from sqlalchemy import text

# Ensure project root is on sys.path when run directly.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.defaults import DEFAULT_INDEX_SYMBOLS
from app.settings import build_engine
from app.tools.sync_cn_stock_daily_price_from_tushare import (
    resolve_tushare_token,
    patch_pandas_fillna_method_compat,
)


def _to_ymd(s: str) -> str:
    s = (s or "").strip()
    if len(s) == 8 and s.isdigit():
        return s
    return datetime.strptime(s, "%Y-%m-%d").strftime("%Y%m%d")


def _ts_code_from_index(index_code: str) -> str:
    code_clean = str(index_code).strip().lower()
    if code_clean.startswith(("sh", "sz")) and len(code_clean) >= 8:
        code6 = code_clean[2:]
        sfx = code_clean[:2].upper()
    else:
        code6 = code_clean
        sfx = "SZ" if code6.startswith(("0", "3")) else "SH"
    return f"{code6}.{sfx}"


def _fetch_index_daily(pro, ts_code: str, start: str, end: str) -> pd.DataFrame:
    df = pro.index_daily(ts_code=ts_code, start_date=start, end_date=end)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.rename(
        columns={
            "trade_date": "trade_date",
            "pre_close": "pre_close",
            "pct_chg": "chg_pct",
            "vol": "volume",
        }
    )
    df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d", errors="coerce")
    for col in ["open", "close", "high", "low", "pre_close", "volume", "amount", "chg_pct"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.sort_values("trade_date").reset_index(drop=True)
    df["source"] = "tushare"
    keep = ["trade_date", "open", "close", "high", "low", "volume", "amount", "source", "pre_close", "chg_pct"]
    for c in keep:
        if c not in df.columns:
            df[c] = None
    out = df[keep].dropna(subset=["trade_date"]).copy()
    quote_cols = ["open", "close", "high", "low", "pre_close", "volume", "amount", "chg_pct"]
    out = out.loc[out[quote_cols].notna().any(axis=1)].copy()
    return out


def _overwrite_index_range(engine, index_code: str, df: pd.DataFrame, start: str, end: str) -> int:
    if df is None or df.empty:
        return 0
    df = df.copy()
    df["index_code"] = index_code
    table_cols = [
        "index_code",
        "trade_date",
        "open",
        "close",
        "high",
        "low",
        "volume",
        "amount",
        "source",
        "pre_close",
        "chg_pct",
    ]
    for c in table_cols:
        if c not in df.columns:
            df[c] = None
    df = df[table_cols].copy()
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                DELETE FROM cn_index_daily_price
                WHERE index_code = :idx
                  AND trade_date BETWEEN :s AND :e
                """
            ),
            {"idx": index_code, "s": start, "e": end},
        )
        df.to_sql("cn_index_daily_price", conn, if_exists="append", index=False, chunksize=200)
    return len(df)


def main() -> int:
    p = argparse.ArgumentParser(description="Backfill index daily prices from Tushare (overwrite range).")
    p.add_argument("--start", required=True, help="YYYY-MM-DD or YYYYMMDD")
    p.add_argument("--end", required=True, help="YYYY-MM-DD or YYYYMMDD")
    p.add_argument("--indices", default="", help="comma-separated index codes, default uses DEFAULT_INDEX_SYMBOLS")
    p.add_argument("--sleep", type=float, default=0.2, help="sleep seconds between indices")
    args = p.parse_args()

    start = _to_ymd(args.start)
    end = _to_ymd(args.end)
    if start > end:
        raise SystemExit("start must be <= end")

    patch_pandas_fillna_method_compat()
    token, _ = resolve_tushare_token("", "")
    if not token:
        raise SystemExit("Tushare token required. Set TUSHARE_TOKEN/TS_TOKEN or config file.")
    pro = ts.pro_api(token)

    if args.indices.strip():
        indices = [s.strip() for s in args.indices.replace(";", ",").split(",") if s.strip()]
    else:
        indices = list(DEFAULT_INDEX_SYMBOLS)

    engine = build_engine()

    for i, idx in enumerate(indices, start=1):
        ts_code = _ts_code_from_index(idx)
        print(f"[TUSHARE][INDEX] {i}/{len(indices)} {idx} ({ts_code}) {start}..{end}")
        df = _fetch_index_daily(pro, ts_code, start, end)
        if df is None or df.empty:
            print(f"[TUSHARE][INDEX] {idx} empty")
        else:
            n = _overwrite_index_range(engine, idx, df, start, end)
            print(f"[TUSHARE][INDEX] {idx} inserted {n}")
        if args.sleep and args.sleep > 0:
            time.sleep(args.sleep)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
