"""
scripts/backfill_sw_industry_daily.py
========================================
GrowthAlpha V8 — Backfill cn_sw_industry_daily from Tushare Pro sw_daily API.

Covers: 2010-01-01 → 2026-03-31 (or custom --end)

Features:
  - Incremental by default (auto-detects max(trade_date) per ts_code)
  - --force to overwrite specified date range
  - --resume to skip completed chunks via cn_mainline_backfill_job_state
  - --sleep configurable throttle (default 8.5s between codes)
  - --limit-codes for smoke test
  - Auto-retry with exponential backoff on rate-limit / network errors
  - Auto-reconnect via SQLAlchemy pool_pre_ping
  - Logs to logs/mainline_data/

Usage:
  # Full backfill
  python scripts/backfill_sw_industry_daily.py --start 2010-01-01 --end 2026-03-31 --resume

  # Incremental (default: detect missing dates)
  python scripts/backfill_sw_industry_daily.py

  # Force re-fetch a specific range
  python scripts/backfill_sw_industry_daily.py --start 2026-01-01 --end 2026-03-31 --force

  # Smoke test with 3 codes
  python scripts/backfill_sw_industry_daily.py --limit-codes 3
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import text

# Bootstrap project root to sys.path so app/ and data_pipeline/ are importable
_project_root = str(Path(__file__).resolve().parents[1])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from app.settings import build_engine
from data_pipeline.common.cli import add_shared_args, resolve_date_range
from data_pipeline.common.logging_utils import build_logger
from data_pipeline.common.state import BackfillState
from data_pipeline.tushare.client import TushareClient, resolve_tushare_token

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

UPSERT_COLS = [
    "ts_code",
    "trade_date",
    "name",
    "open",
    "high",
    "low",
    "close",
    "change",
    "pct_change",
    "vol",
    "amount",
    "pe",
    "pb",
    "float_mv",
    "source",
]

SW_API_FIELDS = "ts_code,trade_date,name,open,high,low,close,change,pct_chg,vol,amount,pe,pb,float_mv"

DEFAULT_SLEEP_SECONDS = 8.5
MAX_RETRIES = 6
RETRY_BASE_SLEEP = 12.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_yyyymmdd(d: date) -> str:
    return d.strftime("%Y%m%d")


def _parse_date(value: Any) -> date | None:
    if value is None:
        return None
    if isinstance(value, date):
        return value
    if isinstance(value, datetime):
        return value.date()
    try:
        return datetime.strptime(str(value)[:10], "%Y-%m-%d").date()
    except (ValueError, TypeError):
        pass
    try:
        return datetime.strptime(str(value).strip(), "%Y%m%d").date()
    except (ValueError, TypeError):
        pass
    return None


def _normalize_sw_daily(raw: pd.DataFrame, source_label: str) -> pd.DataFrame:
    """Normalize raw Tushare sw_daily response to UPSERT_COLS schema."""
    if raw is None or raw.empty:
        return pd.DataFrame(columns=UPSERT_COLS)

    out = raw.copy()
    out["ts_code"] = out["ts_code"].astype(str).str.strip()
    out["trade_date"] = pd.to_datetime(out["trade_date"], format="%Y%m%d", errors="coerce").dt.date
    out = out.rename(columns={"pct_chg": "pct_change"})

    numeric_cols = ["open", "high", "low", "close", "change", "pct_change", "vol", "amount", "pe", "pb", "float_mv"]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
        else:
            out[col] = None

    if "name" not in out.columns:
        out["name"] = None
    out["source"] = source_label
    out = out.dropna(subset=["ts_code", "trade_date"])
    return out[UPSERT_COLS].copy()


def _chunked(records: list[dict], size: int):
    for i in range(0, len(records), size):
        yield records[i : i + size]


# ---------------------------------------------------------------------------
# SW Industry code resolution
# ---------------------------------------------------------------------------


def resolve_sw_codes(engine, src: str = "SW2021", client: TushareClient | None = None) -> list[str]:
    """Resolve SW L1 industry codes from DB or Tushare."""
    # Try DB first
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT DISTINCT ts_code
                FROM cn_sw_industry_daily
                WHERE ts_code IS NOT NULL AND ts_code != ''
                ORDER BY ts_code
                """
            )
        ).fetchall()
    codes = sorted({str(r[0]).strip() for r in rows if str(r[0]).strip()})
    if codes:
        return codes

    # Try cn_local_industry_master
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT industry_id
                FROM cn_local_industry_master
                WHERE src = :src AND industry_level = 'L1'
                ORDER BY industry_id
                """
            ),
            {"src": src},
        ).fetchall()
    codes = sorted({str(r[0]).strip() for r in rows if str(r[0]).strip()})
    if codes:
        return codes

    # Fallback: fetch from Tushare
    if client is not None:
        logger = build_logger("backfill_sw_industry_daily")
        logger.info("Fetching SW L1 codes from Tushare index_classify")
        frame = client.call(
            "index_classify",
            {"src": src, "level": "L1"},
            "index_code,industry_name",
            cache_key=f"index_classify|src={src}|level=L1",
        )
        codes = sorted({str(v).strip() for v in frame["index_code"].tolist() if str(v).strip()})
        if codes:
            return codes

    raise SystemExit(f"No SW L1 codes resolved from DB or Tushare (src={src}).")


def resolve_start_dates(engine, ts_codes: list[str], default_start: date) -> dict[str, date]:
    """Return {ts_code: max_trade_date} for incremental mode."""
    if not ts_codes:
        return {}
    placeholders = ", ".join(f":p{i}" for i in range(len(ts_codes)))
    params = {f"p{i}": code for i, code in enumerate(ts_codes)}
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                f"""
                SELECT ts_code, MAX(trade_date) AS max_trade_date
                FROM cn_sw_industry_daily
                WHERE ts_code IN ({placeholders})
                GROUP BY ts_code
                """
            ),
            params,
        ).fetchall()
    out = {code: default_start for code in ts_codes}
    for ts_code, max_trade_date in rows:
        if max_trade_date is None:
            continue
        parsed = _parse_date(max_trade_date)
        if parsed:
            out[str(ts_code)] = parsed
    return out


# ---------------------------------------------------------------------------
# Tushare sw_daily fetch with retry
# ---------------------------------------------------------------------------


def fetch_sw_daily_with_retry(
    client: TushareClient,
    ts_code: str,
    start_date: date,
    end_date: date,
    logger: Any,
    sleep_seconds: float = DEFAULT_SLEEP_SECONDS,
) -> pd.DataFrame:
    """Fetch sw_daily for one code with retry logic."""
    params = {
        "ts_code": ts_code,
        "start_date": _to_yyyymmdd(start_date),
        "end_date": _to_yyyymmdd(end_date),
    }
    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            frame = client.call(
                "sw_daily",
                params,
                SW_API_FIELDS,
                cache_key=f"sw_daily|{ts_code}|{start_date}|{end_date}",
                use_cache=False,
            )
            return frame
        except Exception as exc:
            last_error = exc
            msg = str(exc).lower()
            is_rate_limit = "rate" in msg or "limit" in msg or "frequency" in msg or "瓒呴檺" in msg
            if not is_rate_limit or attempt == MAX_RETRIES:
                logger.warning("sw_daily fetch failed ts_code=%s attempt=%s/%s err=%s", ts_code, attempt, MAX_RETRIES, exc)
                if attempt == MAX_RETRIES:
                    raise
            wait = min(RETRY_BASE_SLEEP * (1.5 ** (attempt - 1)), 60.0)
            logger.warning("sw_daily rate_limit ts_code=%s attempt=%s/%s sleep=%.1fs", ts_code, attempt, MAX_RETRIES, wait)
            time.sleep(wait)
    raise RuntimeError(f"sw_daily fetch failed after {MAX_RETRIES} retries: {last_error}")


# ---------------------------------------------------------------------------
# Upsert
# ---------------------------------------------------------------------------


def upsert_sw_daily(engine, df: pd.DataFrame, chunk_size: int = 2000) -> int:
    """Upsert normalized DataFrame into cn_sw_industry_daily. Returns affected rows."""
    if df is None or df.empty:
        return 0

    work = df.copy().astype(object)
    work = work.where(pd.notna(work), None)
    insert_sql = """
        INSERT INTO cn_sw_industry_daily (
            ts_code, trade_date, name, open, high, low, close,
            `change`, pct_change, vol, amount, pe, pb, float_mv, source
        ) VALUES (
            :ts_code, :trade_date, :name, :open, :high, :low, :close,
            :change, :pct_change, :vol, :amount, :pe, :pb, :float_mv, :source
        )
        ON DUPLICATE KEY UPDATE
            name = VALUES(name),
            open = VALUES(open),
            high = VALUES(high),
            low = VALUES(low),
            close = VALUES(close),
            `change` = VALUES(`change`),
            pct_change = VALUES(pct_change),
            vol = VALUES(vol),
            amount = VALUES(amount),
            pe = VALUES(pe),
            pb = VALUES(pb),
            float_mv = VALUES(float_mv),
            source = VALUES(source)
    """
    affected = 0
    with engine.begin() as conn:
        for batch in _chunked(work[UPSERT_COLS].to_dict(orient="records"), chunk_size):
            ret = conn.execute(text(insert_sql), batch)
            affected += int(ret.rowcount or 0)
    return affected


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="GrowthAlpha V8 — Backfill cn_sw_industry_daily from Tushare sw_daily"
    )
    parser.add_argument("--start", default="2010-01-01", help="Start date YYYY-MM-DD or YYYYMMDD")
    parser.add_argument("--end", default="", help="End date YYYY-MM-DD or YYYYMMDD (default: today)")
    parser.add_argument("--resume", action="store_true", help="Skip completed chunks in cn_mainline_backfill_job_state")
    parser.add_argument("--force", action="store_true", help="Force re-fetch for the specified range (ignore incremental)")
    parser.add_argument("--sleep", type=float, default=DEFAULT_SLEEP_SECONDS, help=f"Throttle seconds between codes (default: {DEFAULT_SLEEP_SECONDS})")
    parser.add_argument("--limit-codes", type=int, default=0, help="Limit number of SW codes for smoke test (0=all)")
    parser.add_argument("--token", default="", help="Tushare token")
    parser.add_argument("--config", default="", help="Optional config path for Tushare token")
    parser.add_argument("--src", default="SW2021", help="Tushare Shenwan classification source")
    parser.add_argument("--source-label", default="tushare_sw_daily", help="Source label for upsert")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    date_range = resolve_date_range(args.start, args.end)
    logger = build_logger("backfill_sw_industry_daily")
    engine = build_engine()

    # Ensure DDL
    logger.info("Ensuring cn_sw_industry_daily table exists")
    from app.settings import load_sql_for_current_db
    ddl = load_sql_for_current_db("docs/DDL/cn_market.cn_sw_industry_daily.sql")
    with engine.begin() as conn:
        conn.execute(text(ddl))

    # Ensure backfill state table
    state_ddl = """
    CREATE TABLE IF NOT EXISTS `cn_mainline_backfill_job_state` (
        `job_name` VARCHAR(64) NOT NULL,
        `chunk_key` VARCHAR(128) NOT NULL,
        `range_start` DATE DEFAULT NULL,
        `range_end` DATE DEFAULT NULL,
        `status` VARCHAR(16) NOT NULL,
        `attempts` INT NOT NULL DEFAULT 0,
        `last_rows` BIGINT NOT NULL DEFAULT 0,
        `last_error` TEXT DEFAULT NULL,
        `last_run_id` VARCHAR(64) DEFAULT NULL,
        `updated_at` TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        `created_at` TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (`job_name`, `chunk_key`),
        KEY `idx_cn_mainline_backfill_job_state_status` (`status`, `updated_at`)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """
    with engine.begin() as conn:
        conn.execute(text(state_ddl))

    state = BackfillState(engine=engine, job_name="backfill_sw_industry_daily")
    token = resolve_tushare_token(args.token, args.config)
    client = TushareClient(token=token, logger=logger)

    # Resolve SW codes
    codes = resolve_sw_codes(engine, src=args.src, client=client)
    if args.limit_codes > 0:
        codes = codes[: args.limit_codes]
        logger.info("Limited to %s codes (smoke test)", len(codes))

    logger.info(
        "Starting backfill codes=%s start=%s end=%s force=%s resume=%s sleep=%.1fs",
        len(codes),
        date_range.start,
        date_range.end,
        args.force,
        args.resume,
        args.sleep,
    )

    # Resolve start dates per code
    if args.force:
        start_dates = {code: date_range.start for code in codes}
    else:
        existing = resolve_start_dates(engine, codes, date_range.start)
        start_dates = {}
        for code in codes:
            existing_start = existing.get(code, date_range.start)
            if existing_start and existing_start >= date_range.end:
                start_dates[code] = existing_start  # skip, already up to date
            else:
                start_dates[code] = existing_start if existing_start > date_range.start else date_range.start

    total_rows = 0
    total_affected = 0
    touched = 0
    skipped = 0

    for i, code in enumerate(codes, start=1):
        code_start = start_dates.get(code, date_range.start)
        if code_start is None:
            code_start = date_range.start

        # If code_start >= end_date, skip (already up to date)
        if code_start >= date_range.end:
            skipped += 1
            logger.info("[%s/%s] %s skip up-to-date (max=%s)", i, len(codes), code, code_start)
            continue

        chunk_key = f"{code}:{_to_yyyymmdd(code_start)}:{_to_yyyymmdd(date_range.end)}"
        if args.resume and state.is_completed(chunk_key):
            logger.info("[%s/%s] %s resume-skip chunk=%s", i, len(codes), code, chunk_key)
            skipped += 1
            continue

        state.start(chunk_key, code_start, date_range.end)
        logger.info("[%s/%s] %s fetching %s..%s", i, len(codes), code, code_start, date_range.end)

        try:
            raw = fetch_sw_daily_with_retry(client, code, code_start, date_range.end, logger, sleep_seconds=args.sleep)
            df = _normalize_sw_daily(raw, args.source_label)
            if df.empty:
                logger.info("[%s/%s] %s rows=0 (no data in range)", i, len(codes), code)
                state.complete(chunk_key, 0)
                skipped += 1
                continue

            rows = len(df)
            affected = upsert_sw_daily(engine, df)
            total_rows += rows
            total_affected += affected
            touched += 1
            state.complete(chunk_key, rows)
            logger.info("[%s/%s] %s rows=%s affected=%s", i, len(codes), code, rows, affected)

            # Throttle between codes
            if i < len(codes):
                logger.debug("Throttle %.1fs before next code", args.sleep)
                time.sleep(args.sleep)

        except Exception as exc:
            state.fail(chunk_key, exc)
            logger.error("[%s/%s] %s failed: %s", i, len(codes), code, exc)
            # Continue with next code instead of aborting
            continue

    # Summary
    with engine.connect() as conn:
        summary = conn.execute(
            text(
                """
                SELECT COUNT(*) AS row_cnt,
                       COUNT(DISTINCT ts_code) AS code_cnt,
                       MIN(trade_date) AS min_trade_date,
                       MAX(trade_date) AS max_trade_date
                FROM cn_sw_industry_daily
                """
            )
        ).one()

    logger.info(
        "backfill_sw_industry_daily done "
        "codes=%s touched=%s skipped=%s fetched_rows=%s affected=%s "
        "table_rows=%s code_cnt=%s min_trade_date=%s max_trade_date=%s",
        len(codes),
        touched,
        skipped,
        total_rows,
        total_affected,
        summary[0],
        summary[1],
        summary[2],
        summary[3],
    )
    print(
        f"\n=== backfill_sw_industry_daily complete ==="
        f"\n  Codes:     {len(codes)} total, {touched} touched, {skipped} skipped"
        f"\n  Fetched:   {total_rows} rows, {total_affected} affected"
        f"\n  Table:     {summary[0]:,} rows, {summary[1]} codes"
        f"\n  Date:      {summary[2]} -> {summary[3]}"
    )


if __name__ == "__main__":
    main()
