from __future__ import annotations

import argparse
import builtins
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, List, Tuple

import pandas as pd
import tushare as ts
from sqlalchemy import text

from app.settings import build_engine, load_sql_for_current_db
from app.utils.progress import ProgressLogger


def print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    return builtins.print(*args, **kwargs)
from app.tools.sync_cn_stock_daily_price_from_tushare import (
    _parse_ymd,
    patch_pandas_fillna_method_compat,
    resolve_tushare_token,
)


TABLE_NAME = "cn_inst_fund_hold_summary"
SOURCE_LABEL = "tushare_fund_portfolio"
INSTITUTION_TYPE = "FUND"
FETCH_FIELDS = "ts_code,ann_date,end_date,symbol,mkv,amount,stk_mkv_ratio,stk_float_ratio"
UPSERT_COLS = [
    "INSTITUTION_TYPE",
    "HOLD_DATE",
    "STOCK_CODE",
    "STOCK_NAME",
    "FUND_HOUSE_NUM",
    "TOTAL_SHARES",
    "TOTAL_MARKET_VALUE",
    "CHANGE_TYPE",
    "CHANGE_SHARES",
    "CHANGE_RATIO",
    "CREATION_DATE",
]


def apply_ddl(engine, ddl_path: str) -> None:
    sql = load_sql_for_current_db(ddl_path)
    with engine.begin() as conn:
        statements = [part.strip() for part in sql.split(";") if part.strip()]
        for stmt in statements:
            conn.execute(text(stmt))


def ensure_tables(engine) -> None:
    apply_ddl(engine, "docs/DDL/cn_market.cn_inst_fund_hold_summary.sql")
    with engine.begin() as conn:
        stmts = [
            "ALTER TABLE cn_inst_fund_hold_summary MODIFY COLUMN TOTAL_SHARES DECIMAL(24,2) NULL",
            "ALTER TABLE cn_inst_fund_hold_summary MODIFY COLUMN TOTAL_MARKET_VALUE DECIMAL(20,2) NULL",
            "ALTER TABLE cn_inst_fund_hold_summary MODIFY COLUMN CHANGE_SHARES DECIMAL(24,2) NULL",
            "ALTER TABLE cn_inst_fund_hold_summary MODIFY COLUMN CHANGE_RATIO DECIMAL(18,6) NULL",
        ]
        for stmt in stmts:
            try:
                conn.execute(text(stmt))
            except Exception:
                pass


def _quarter_end(dt: date) -> date:
    month = ((dt.month - 1) // 3 + 1) * 3
    if month == 3:
        return date(dt.year, 3, 31)
    if month == 6:
        return date(dt.year, 6, 30)
    if month == 9:
        return date(dt.year, 9, 30)
    return date(dt.year, 12, 31)


def _quarter_start(dt: date) -> date:
    month = ((dt.month - 1) // 3) * 3 + 1
    return date(dt.year, month, 1)


def _prev_quarter_end(dt: date) -> date:
    return _quarter_start(dt) - timedelta(days=1)


def _iter_quarter_ends(start_date: date, end_date: date) -> Iterable[date]:
    cursor = _quarter_end(start_date)
    if cursor < start_date:
        cursor = _quarter_end(date(cursor.year, cursor.month, 1) + timedelta(days=92))
    while cursor <= end_date:
        yield cursor
        next_day = cursor + timedelta(days=1)
        cursor = _quarter_end(next_day)


def _coerce_date_series(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, format="%Y%m%d", errors="coerce").dt.date


def _normalize_symbol_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.split(".").str[0].str.strip()


def _fetch_fund_portfolio_period(
    pro,
    period: date,
    *,
    page_size: int,
    sleep_seconds: float,
    max_retries: int = 3,
    rate_limit_sleep_seconds: float = 65.0,
) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    offset = 0
    period_str = period.strftime("%Y%m%d")
    while True:
        raw = None
        for attempt in range(1, max_retries + 1):
            try:
                raw = pro.fund_portfolio(
                    period=period_str,
                    offset=offset,
                    limit=page_size,
                    fields=FETCH_FIELDS,
                )
                break
            except Exception as exc:
                message = str(exc)
                if "每分钟最多访问该接口60次" in message:
                    time.sleep(max(1.0, rate_limit_sleep_seconds))
                    continue
                if attempt >= max_retries:
                    raise
                time.sleep(max(0.5, sleep_seconds) * attempt)

        if raw is None or raw.empty:
            break
        frames.append(raw)
        row_count = len(raw)
        if row_count < page_size:
            break
        offset += row_count
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    if not frames:
        return pd.DataFrame(columns=FETCH_FIELDS.split(","))
    return pd.concat(frames, ignore_index=True)


def _normalize_fund_portfolio(raw: pd.DataFrame) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame(columns=FETCH_FIELDS.split(","))

    out = raw.copy()
    out["symbol"] = _normalize_symbol_series(out["symbol"])
    out["ann_date"] = _coerce_date_series(out["ann_date"])
    out["end_date"] = _coerce_date_series(out["end_date"])

    for col in ["mkv", "amount", "stk_mkv_ratio", "stk_float_ratio"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out["ts_code"] = out["ts_code"].astype(str).str.strip()
    out = out.dropna(subset=["symbol", "ann_date", "end_date"])
    out = out[out["symbol"] != ""].copy()
    return out


def _summarize_fund_portfolio(raw: pd.DataFrame, source_label: str) -> pd.DataFrame:
    norm = _normalize_fund_portfolio(raw)
    if norm.empty:
        return pd.DataFrame(columns=UPSERT_COLS)

    grouped = (
        norm.groupby(["symbol", "end_date"], as_index=False)
        .agg(
            FUND_HOUSE_NUM=("ts_code", "nunique"),
            TOTAL_MARKET_VALUE=("mkv", "sum"),
            TOTAL_SHARES=("amount", "sum"),
        )
        .sort_values(["end_date", "symbol"])
        .reset_index(drop=True)
    )
    grouped["INSTITUTION_TYPE"] = INSTITUTION_TYPE
    grouped["HOLD_DATE"] = pd.to_datetime(grouped["end_date"]).dt.strftime("%Y-%m-%d")
    grouped["STOCK_CODE"] = grouped["symbol"]
    grouped["STOCK_NAME"] = None
    grouped["CHANGE_TYPE"] = None
    grouped["CHANGE_SHARES"] = None
    grouped["CHANGE_RATIO"] = None
    grouped["CREATION_DATE"] = date.today()
    return grouped[UPSERT_COLS]


def _chunked(records: List[dict], chunk_size: int) -> Iterable[List[dict]]:
    for i in range(0, len(records), chunk_size):
        yield records[i : i + chunk_size]


def _load_stock_name_map(engine, symbols: List[str]) -> dict[str, str]:
    if not symbols:
        return {}
    mapping: dict[str, str] = {}
    with engine.connect() as conn:
        for chunk in _chunked(symbols, 500):
            bind_names = []
            params = {}
            for i, symbol in enumerate(chunk):
                key = f"s{i}"
                bind_names.append(f":{key}")
                params[key] = symbol
            sql = text(
                f"""
                SELECT symbol, MAX(name) AS stock_name
                FROM cn_stock_daily_price
                WHERE symbol IN ({", ".join(bind_names)})
                  AND name IS NOT NULL
                  AND name <> ''
                GROUP BY symbol
                """
            )
            for row in conn.execute(sql, params).fetchall():
                if row[0] and row[1]:
                    mapping[str(row[0])] = str(row[1])
    return mapping


def _load_previous_period_summary(engine, period: date) -> pd.DataFrame:
    prev_period = _prev_quarter_end(period)
    sql = text(
        f"""
        SELECT STOCK_CODE, TOTAL_SHARES
        FROM {TABLE_NAME}
        WHERE INSTITUTION_TYPE = :institution_type
          AND HOLD_DATE = :hold_date
        """
    )
    with engine.connect() as conn:
        rows = conn.execute(
            sql,
            {
                "institution_type": INSTITUTION_TYPE,
                "hold_date": prev_period.strftime("%Y-%m-%d"),
            },
        ).fetchall()
    if not rows:
        return pd.DataFrame(columns=["STOCK_CODE", "PREV_TOTAL_SHARES"])
    return pd.DataFrame(rows, columns=["STOCK_CODE", "PREV_TOTAL_SHARES"])


def _apply_stock_names_and_change(engine, period: date, summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary

    out = summary.copy()
    name_map = _load_stock_name_map(engine, out["STOCK_CODE"].astype(str).tolist())
    out["STOCK_NAME"] = out["STOCK_CODE"].map(name_map)

    prev = _load_previous_period_summary(engine, period)
    out = out.merge(prev, on="STOCK_CODE", how="left")
    out["TOTAL_SHARES"] = pd.to_numeric(out["TOTAL_SHARES"], errors="coerce")
    out["PREV_TOTAL_SHARES"] = pd.to_numeric(out["PREV_TOTAL_SHARES"], errors="coerce")
    out["CHANGE_SHARES"] = out["TOTAL_SHARES"] - out["PREV_TOTAL_SHARES"]

    def _classify(row) -> str:
        prev_total = row["PREV_TOTAL_SHARES"]
        delta = row["CHANGE_SHARES"]
        if pd.isna(prev_total) or float(prev_total) == 0.0:
            return "\u65b0\u8fdb"
        if pd.isna(delta):
            return "\u65b0\u8fdb"
        if delta > 0:
            return "\u589e\u4ed3"
        if delta < 0:
            return "\u51cf\u4ed3"
        return "\u4e0d\u53d8"

    out["CHANGE_TYPE"] = out.apply(_classify, axis=1)
    out["CHANGE_RATIO"] = None
    ratio_mask = out["PREV_TOTAL_SHARES"].notna() & (out["PREV_TOTAL_SHARES"] != 0)
    out.loc[ratio_mask, "CHANGE_RATIO"] = (
        out.loc[ratio_mask, "CHANGE_SHARES"] / out.loc[ratio_mask, "PREV_TOTAL_SHARES"] * 100.0
    )
    out["CHANGE_SHARES"] = out["CHANGE_SHARES"].where(out["CHANGE_SHARES"].notna(), out["TOTAL_SHARES"])
    out = out.drop(columns=["PREV_TOTAL_SHARES"])
    return out[UPSERT_COLS]


def _replace_period_summary(engine, period: date, summary: pd.DataFrame) -> int:
    with engine.begin() as conn:
        conn.execute(
            text(
                f"DELETE FROM {TABLE_NAME} WHERE INSTITUTION_TYPE = :institution_type AND HOLD_DATE = :hold_date"
            ),
            {
                "institution_type": INSTITUTION_TYPE,
                "hold_date": period.strftime("%Y-%m-%d"),
            },
        )
        if summary.empty:
            return 0

        insert_cols = ", ".join(UPSERT_COLS)
        params = ", ".join(f":{col}" for col in UPSERT_COLS)
        sql = text(f"INSERT INTO {TABLE_NAME} ({insert_cols}) VALUES ({params})")
        payload = summary.where(pd.notna(summary), None).to_dict(orient="records")
        for chunk in _chunked(payload, 1000):
            conn.execute(sql, chunk)
    return int(len(summary))


def load_inst_fund_hold_summary_tushare(
    *,
    engine,
    start_date: date,
    end_date: date,
    token: str,
    source_label: str = SOURCE_LABEL,
    page_size: int = 5000,
    sleep_seconds: float = 0.2,
    rate_limit_sleep_seconds: float = 65.0,
    log=None,
) -> Tuple[int, int, List[date]]:
    if end_date < start_date:
        return 0, 0, []

    ensure_tables(engine)
    pro = ts.pro_api(token)
    total_raw_rows = 0
    total_summary_rows = 0
    periods_done: List[date] = []
    periods = list(_iter_quarter_ends(start_date, end_date))
    progress = ProgressLogger(name="inst_fund_hold_summary", total=len(periods), unit="quarters", log=log, every=1, min_interval_seconds=10.0)

    for period in periods:
        progress.note(f"[inst_fund_hold_summary] fetching period={period}")
        raw = _fetch_fund_portfolio_period(
            pro,
            period,
            page_size=page_size,
            sleep_seconds=sleep_seconds,
            rate_limit_sleep_seconds=rate_limit_sleep_seconds,
        )
        summary = _summarize_fund_portfolio(raw, source_label)
        summary = _apply_stock_names_and_change(engine, period, summary)
        affected = _replace_period_summary(engine, period, summary)
        total_raw_rows += int(len(raw))
        total_summary_rows += affected
        periods_done.append(period)
        progress.update(current_item=str(period), rows=int(len(raw)), affected=affected)
        if log is not None:
            log.info(
                "[inst_fund_hold_summary] period=%s raw_rows=%s summary_rows=%s",
                period,
                len(raw),
                affected,
            )
    progress.finish()

    return total_raw_rows, total_summary_rows, periods_done


def get_existing_max_end_date(engine) -> date | None:
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                f"""
                SELECT HOLD_DATE
                FROM {TABLE_NAME}
                WHERE INSTITUTION_TYPE = :institution_type
                  AND HOLD_DATE IS NOT NULL
                  AND HOLD_DATE <> ''
                """
            ),
            {"institution_type": INSTITUTION_TYPE},
        ).fetchall()
    max_dt = None
    for row in rows:
        dt = pd.to_datetime(row[0], errors="coerce")
        if pd.isna(dt):
            continue
        cur = dt.date()
        if max_dt is None or cur > max_dt:
            max_dt = cur
    return max_dt


def compute_incremental_start(
    engine,
    *,
    requested_start: date,
    history_start: date,
    lookback_quarters: int,
) -> date:
    existing_max = get_existing_max_end_date(engine)
    if existing_max is None:
        return max(history_start, requested_start)

    start = existing_max
    for _ in range(max(1, lookback_quarters) - 1):
        start = _prev_quarter_end(start)
    return max(history_start, requested_start, start)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sync cn_inst_fund_hold_summary from Tushare fund_portfolio")
    parser.add_argument("--start-date", default="20100101", help="YYYYMMDD")
    parser.add_argument("--end-date", default="latest", help="YYYYMMDD or latest")
    parser.add_argument("--token", default="", help="Tushare token")
    parser.add_argument("--config", default="", help="Optional config file path to resolve Tushare token")
    parser.add_argument("--page-size", type=int, default=5000, help="Tushare page size")
    parser.add_argument("--sleep-seconds", type=float, default=0.2, help="Sleep between paged requests")
    parser.add_argument("--rate-limit-sleep-seconds", type=float, default=65.0, help="Sleep when Tushare rate limit is hit")
    args = parser.parse_args(argv)

    patch_pandas_fillna_method_compat()
    token, tried_files = resolve_tushare_token(args.token, args.config)
    if not token:
        msg = "Tushare token is required"
        if tried_files:
            msg += f"; tried_files={', '.join(str(p) for p in tried_files)}"
        raise RuntimeError(msg)

    start_date = _parse_ymd(args.start_date)
    end_date = date.today() if str(args.end_date).strip().lower() == "latest" else _parse_ymd(args.end_date)
    engine = build_engine()
    total_raw, total_summary, periods_done = load_inst_fund_hold_summary_tushare(
        engine=engine,
        start_date=start_date,
        end_date=end_date,
        token=token,
        page_size=max(1, int(args.page_size)),
        sleep_seconds=max(0.0, float(args.sleep_seconds)),
        rate_limit_sleep_seconds=max(1.0, float(args.rate_limit_sleep_seconds)),
    )
    print(
        f"inst_fund_hold_summary synced periods={len(periods_done)} raw_rows={total_raw} summary_rows={total_summary} "
        f"start={start_date} end={end_date}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
