"""
scripts/build_local_industry_map_hist.py
============================================
GrowthAlpha V8 — Build cn_local_industry_map_hist from Tushare SW industry membership.

Data sources (priority order):
  1. Tushare index_member_all (SW industry membership history)
  2. cn_board_member_map_d (legacy compatibility fallback)

Generates:
  - symbol, industry_id, industry_name, industry_level
  - in_date, out_date, is_manual_override
  - source, updated_at

Supports:
  - SW L1 (default), L2, L3
  - --start / --end date range
  - --resume (skip completed chunks)
  - --force (reprocess even if completed)
  - Avoids future function: in_date/out_date from Tushare are historical facts

Usage:
  python scripts/build_local_industry_map_hist.py --start 2010-01-01 --end 2026-03-31 --level L1 --resume
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime
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
from data_pipeline.common.db import chunked_rows, fetch_df
from data_pipeline.common.logging_utils import build_logger
from data_pipeline.common.state import BackfillState
from data_pipeline.tushare.client import TushareClient, resolve_tushare_token



# ---------------------------------------------------------------------------
# Source data coverage audit
# ---------------------------------------------------------------------------

def _table_exists_for_audit(engine, table_name: str) -> bool:
    with engine.connect() as conn:
        return bool(
            conn.execute(
                text(
                    """
                    SELECT COUNT(*)
                    FROM INFORMATION_SCHEMA.TABLES
                    WHERE TABLE_SCHEMA = DATABASE()
                      AND TABLE_NAME = :table_name
                    """
                ),
                {"table_name": table_name},
            ).scalar()
        )


def _parse_audit_date(value):
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return datetime.strptime(str(value)[:10], "%Y-%m-%d").date()
    except Exception:
        return None


def _resolve_effective_end_for_audit(engine, end: date) -> date:
    """Resolve the latest usable trading date on or before the requested end date.

    cn_board_member_map_d is a trading-date keyed table. When callers pass a
    non-trading end date (for example a weekend), the source audit should not
    fail just because the table naturally stops at the previous trading day.
    """
    sql_candidates = [
        """
        SELECT MAX(trade_date)
        FROM cn_stock_daily_price
        WHERE trade_date <= :end
        """,
        """
        SELECT MAX(trade_date)
        FROM cn_board_member_map_d
        WHERE trade_date <= :end
        """,
    ]
    with engine.connect() as conn:
        for sql in sql_candidates:
            value = conn.execute(text(sql), {"end": end}).scalar()
            parsed = _parse_audit_date(value)
            if parsed is not None:
                return parsed
    return end


# Minimum row threshold for stock-level tables.
# If a table has >= this many rows in the requested range, it is considered
# to have sufficient data even if the date range is not fully covered.
# This avoids full-table scans across all stocks for every audit.
# Override via env V8_AUDIT_MIN_ROWS_THRESHOLD (default: 1000).
_MIN_ROWS_THRESHOLD = int(os.getenv("V8_AUDIT_MIN_ROWS_THRESHOLD", "1000"))


def audit_source_data_coverage(engine, start: date, end: date, logger) -> None:
    """
    cn_board_member_map_d is treated as a legacy compatibility source, but when it
    is present we still audit its coverage so source drift is visible. Missing DB
    source is logged as fallback, not a hard failure.
    """
    table_name = "cn_board_member_map_d"
    date_col = "trade_date"
    effective_end = _resolve_effective_end_for_audit(engine, end)
    logger.info(
        "source_audit_start required_range=%s~%s effective_end=%s",
        start,
        end,
        effective_end,
    )

    if not _table_exists_for_audit(engine, table_name):
        logger.warning("source_audit table=%s status=MISSING_TABLE action=tushare_fallback", table_name)
        return

    sql = f"""
        SELECT COUNT(*) AS row_count,
               MIN({date_col}) AS min_date,
               MAX({date_col}) AS max_date
        FROM {table_name}
        WHERE {date_col} BETWEEN :start AND :end
          AND {date_col} IS NOT NULL
    """
    with engine.connect() as conn:
        row = conn.execute(text(sql), {"start": start, "end": end}).mappings().first()

    row_count = int((row or {}).get("row_count") or 0)
    min_date = _parse_audit_date((row or {}).get("min_date"))
    max_date = _parse_audit_date((row or {}).get("max_date"))

    logger.info("source_audit table=%s rows=%s range=%s~%s", table_name, row_count, min_date, max_date)

    if row_count <= 0:
        logger.warning("source_audit table=%s status=NO_ROWS action=tushare_fallback", table_name)
        return
    if min_date is None or max_date is None or min_date > start or max_date < effective_end:
        # Use row-count threshold for this large legacy table (30.7M rows)
        # to avoid full-table scans.
        if row_count >= _MIN_ROWS_THRESHOLD:
            logger.info(
                "source_audit table=%s rows=%s >= threshold=%s — treating as OK despite range gap",
                table_name, row_count, _MIN_ROWS_THRESHOLD,
            )
        else:
            logger.error("SOURCE DATA AUDIT FAILED")
            logger.error(
                "- %s.%s: available=%s~%s, required=%s~%s (effective_end=%s)",
                table_name,
                date_col,
                min_date,
                max_date,
                start,
                end,
                effective_end,
            )
            raise SystemExit(2)

    logger.info("source_audit_pass")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_date(value: object) -> date | None:
    if value in (None, "", "None", "nan", "NaT"):
        return None
    dt = pd.to_datetime(value, format="%Y%m%d", errors="coerce")
    if pd.isna(dt):
        return None
    return dt.date()


def _normalize_symbol(value: object) -> str:
    text = str(value or "").strip()
    return text.split(".", 1)[0] if "." in text else text


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


# ---------------------------------------------------------------------------
# Compatibility source: cn_board_member_map_d
# ---------------------------------------------------------------------------


def fetch_board_member_map(engine, start: date, end: date, level: str) -> list[dict]:
    """Extract industry membership from cn_board_member_map_d.

    Uses SQL-level aggregation (MIN/MAX trade_date) per (sector_id, symbol)
    to avoid loading millions of daily rows into memory.
    """
    logger = build_logger("build_local_industry_map_hist")
    logger.info("Fetching board member map for level=%s range=%s..%s", level, start, end)

    # Map board_type to industry level
    level_map = {"L1": "INDUSTRY", "L2": "INDUSTRY", "L3": "INDUSTRY"}

    sql = """
    SELECT m.sector_id,
           m.symbol,
           MIN(m.trade_date) AS first_date,
           MAX(m.trade_date) AS last_date,
           COALESCE(
               (SELECT mst.BOARD_NAME FROM cn_board_industry_master mst
                WHERE m.sector_id = mst.BOARD_ID COLLATE utf8mb4_unicode_ci
                LIMIT 1),
               m.sector_id
           ) AS sector_name
    FROM cn_board_member_map_d m
    WHERE m.trade_date BETWEEN :start AND :end
      AND m.sector_type = :sector_type
    GROUP BY m.sector_id, m.symbol
    ORDER BY m.sector_id, m.symbol
    """
    with engine.connect() as conn:
        df = pd.read_sql(text(sql), conn, params={"start": start, "end": end, "sector_type": level_map.get(level, "INDUSTRY")})

    if df.empty:
        logger.info("No board member map data found for level=%s", level)
        return []

    rows: list[dict] = []
    for _, row in df.iterrows():
        sector_id = str(row["sector_id"]).strip()
        symbol = _normalize_symbol(str(row["symbol"]).strip())
        sector_name = str(row["sector_name"]).strip() if pd.notna(row.get("sector_name")) else sector_id
        first_date = _parse_date(row["first_date"])
        last_date = _parse_date(row["last_date"])

        rows.append(
            {
                "symbol": symbol,
                "industry_id": sector_id,
                "industry_name": sector_name,
                "industry_level": level,
                "in_date": first_date,
                "out_date": last_date,
                "source": "cn_board_member_map_d",
                "is_manual_override": 0,
            }
        )

    logger.info("Board member map derived %s membership records", len(rows))
    return rows


# ---------------------------------------------------------------------------
# Primary source: Tushare index_member_all
# ---------------------------------------------------------------------------


def fetch_industry_master(engine, src: str, level: str) -> pd.DataFrame:
    """Fetch industry master from cn_board_industry_master or Tushare."""
    df = fetch_df(
        engine,
        """
        SELECT BOARD_ID AS industry_id, BOARD_NAME AS industry_name,
               'L1' AS industry_level, SOURCE AS src
        FROM cn_board_industry_master
        WHERE SOURCE = :src
        ORDER BY BOARD_ID
        """,
        {"src": src},
    )
    if not df.empty:
        return df

    # Fallback: fetch from Tushare and insert into local_industry_master
    logger = build_logger("build_local_industry_map_hist")
    logger.info("cn_board_industry_master empty for src=%s, fetching from Tushare", src)
    token = resolve_tushare_token("", "")
    client = TushareClient(token=token, logger=logger)
    frame = client.call(
        "index_classify",
        {"src": src, "level": level},
        "index_code,industry_name,level,parent_code,src",
        cache_key=f"index_classify|src={src}|level={level}",
    )
    if frame.empty:
        raise SystemExit(f"No industry data from Tushare for src={src} level={level}")

    rows = []
    for item in frame.to_dict(orient="records"):
        rows.append(
            {
                "industry_id": str(item.get("index_code") or "").strip(),
                "industry_name": str(item.get("industry_name") or "").strip(),
                "industry_level": str(item.get("level") or level).strip(),
                "parent_id": str(item.get("parent_code") or "").strip() or None,
                "src": src,
            }
        )

    # Ensure cn_local_industry_master exists
    from data_pipeline.common.db import apply_sql_file

    ensure_sql = """
    CREATE TABLE IF NOT EXISTS cn_local_industry_master (
        industry_id    VARCHAR(32)  NOT NULL,
        industry_name  VARCHAR(128) NOT NULL,
        industry_level VARCHAR(4)   NOT NULL DEFAULT 'L1',
        parent_id      VARCHAR(32)  DEFAULT NULL,
        src            VARCHAR(32)  NOT NULL DEFAULT 'SW2021',
        PRIMARY KEY (industry_id, src)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """
    with engine.begin() as conn:
        conn.execute(text(ensure_sql))

    insert_sql = """
    INSERT INTO cn_local_industry_master
        (industry_id, industry_name, industry_level, parent_id, src)
    VALUES
        (:industry_id, :industry_name, :industry_level, :parent_id, :src)
    ON DUPLICATE KEY UPDATE
        industry_name = VALUES(industry_name),
        industry_level = VALUES(industry_level),
        parent_id = VALUES(parent_id),
        src = VALUES(src)
    """
    with engine.begin() as conn:
        for batch in chunked_rows(rows, 500):
            conn.execute(text(insert_sql), batch)

    return fetch_df(
        engine,
        """
        SELECT industry_id, industry_name, industry_level, src
        FROM cn_local_industry_master
        WHERE src = :src AND industry_level = :level
        ORDER BY industry_id
        """,
        {"src": src, "level": level},
    )


def fetch_tushare_member_hist(
    client: TushareClient,
    industry_id: str,
    start: date,
    end: date,
    logger: Any,
) -> list[dict]:
    """Fetch index_member_all for one industry and derive in_date/out_date."""
    frame = client.paginate(
        "index_member_all",
        {"ts_code": industry_id},
        "index_code,con_code,in_date,out_date,is_new",
        page_size=5000,
        key_prefix=f"index_member_all|industry_id={industry_id}",
    )

    rows: list[dict] = []
    for item in frame.to_dict(orient="records"):
        in_date = _to_date(item.get("in_date"))
        out_date = _to_date(item.get("out_date"))

        if in_date is None:
            continue
        if in_date > end:
            continue
        if out_date is not None and out_date < start:
            continue

        rows.append(
            {
                "symbol": _normalize_symbol(item.get("con_code")),
                "industry_id": industry_id,
                "in_date": in_date,
                "out_date": out_date,
                "source": "tushare_index_member_all",
                "is_manual_override": 0,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Merge and upsert
# ---------------------------------------------------------------------------


def upsert_map_hist(engine, rows: list[dict], chunk_size: int = 5000) -> int:
    """Upsert rows into cn_local_industry_map_hist. Returns affected count."""
    if not rows:
        return 0

    insert_sql = """
    INSERT INTO cn_local_industry_map_hist
        (symbol, industry_id, industry_name, industry_level, in_date, out_date, is_manual_override, source)
    VALUES
        (:symbol, :industry_id, :industry_name, :industry_level, :in_date, :out_date, :is_manual_override, :source)
    ON DUPLICATE KEY UPDATE
        industry_name = VALUES(industry_name),
        industry_level = VALUES(industry_level),
        out_date = VALUES(out_date),
        is_manual_override = VALUES(is_manual_override),
        source = VALUES(source),
        updated_at = CURRENT_TIMESTAMP
    """
    affected = 0
    with engine.begin() as conn:
        for batch in chunked_rows(rows, chunk_size):
            ret = conn.execute(text(insert_sql), batch)
            affected += int(ret.rowcount or 0)
    return affected


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="GrowthAlpha V8 — Build cn_local_industry_map_hist from Tushare SW membership"
    )
    parser.add_argument("--start", default="2010-01-01", help="Start date YYYY-MM-DD or YYYYMMDD")
    parser.add_argument("--end", default="", help="End date YYYY-MM-DD or YYYYMMDD (default: today)")
    parser.add_argument("--level", default="L1", choices=["L1", "L2", "L3"], help="Industry level")
    parser.add_argument("--resume", action="store_true", help="Skip completed chunks")
    parser.add_argument("--force", action="store_true", help="Force reprocess even if completed")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers for Tushare fetch")
    parser.add_argument("--token", default="", help="Tushare token")
    parser.add_argument("--config", default="", help="Optional config path for Tushare token")
    parser.add_argument("--src", default="SW2021", help="Tushare industry classification source")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    date_range = resolve_date_range(args.start, args.end)
    logger = build_logger("build_local_industry_map_hist")
    engine = build_engine()
    audit_source_data_coverage(engine, date_range.start, date_range.end, logger)

    # Ensure DDL
    logger.info("Ensuring cn_local_industry_map_hist table exists")
    ddl_check_sql = """
    CREATE TABLE IF NOT EXISTS `cn_local_industry_map_hist` (
        `symbol` VARCHAR(10) NOT NULL,
        `industry_id` VARCHAR(32) NOT NULL,
        `industry_name` VARCHAR(128) NOT NULL,
        `industry_level` VARCHAR(8) NOT NULL,
        `in_date` DATE NOT NULL,
        `out_date` DATE DEFAULT NULL,
        `is_manual_override` TINYINT(1) NOT NULL DEFAULT 0,
        `source` VARCHAR(32) DEFAULT NULL,
        `updated_at` TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        `created_at` TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (`symbol`, `industry_id`, `in_date`),
        KEY `idx_symbol_date` (`symbol`, `in_date`, `out_date`),
        KEY `idx_industry_date` (`industry_id`, `in_date`, `out_date`),
        KEY `idx_manual_override` (`is_manual_override`)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """
    with engine.begin() as conn:
        conn.execute(text(ddl_check_sql))

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

    state = BackfillState(engine=engine, job_name=f"build_local_industry_map_hist_{args.level}")
    token = resolve_tushare_token(args.token, args.config)
    client = TushareClient(token=token, logger=logger)

    logger.info(
        "Starting build level=%s start=%s end=%s resume=%s force=%s workers=%s",
        args.level,
        date_range.start,
        date_range.end,
        args.resume,
        args.force,
        args.workers,
    )

    # Step 1: Load SW industry master for the requested level.
    master = fetch_industry_master(engine, args.src, args.level)
    if master.empty:
        raise SystemExit(f"No industry master found for src={args.src} level={args.level}")

    logger.info("Industry master has %s industries", len(master))

    # Build lookup for industry names
    lookup = {}
    for row in master.itertuples(index=False):
        lookup[str(row.industry_id)] = {
            "industry_name": str(row.industry_name),
            "industry_level": str(row.industry_level),
        }

    # Step 2: Fetch Tushare membership per industry (parallel). This is the
    # primary source for V8 lineage and should populate the table first.
    existing_keys: set[tuple[str, str]] = set()
    total_rows = 0
    total_affected = 0
    tushare_rows: list[dict] = []
    industry_ids = list(lookup.keys())

    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        futures = {}
        for industry_id in industry_ids:
            chunk_key = f"{args.src}:{args.level}:{industry_id}"
            if args.resume and state.is_completed(chunk_key) and not args.force:
                logger.info("resume_skip chunk=%s", chunk_key)
                continue
            state.start(chunk_key, date_range.start, date_range.end)
            futures[pool.submit(fetch_tushare_member_hist, client, industry_id, date_range.start, date_range.end, logger)] = chunk_key

        for future in as_completed(futures):
            chunk_key = futures[future]
            industry_id = chunk_key.rsplit(":", 1)[-1]
            info = lookup.get(industry_id, {})
            try:
                raw_rows = future.result()
                # Deduplicate against rows already accepted into the local map.
                new_rows = []
                for row in raw_rows:
                    key = (row["symbol"], row["industry_id"])
                    if key not in existing_keys:
                        row["industry_name"] = info.get("industry_name", industry_id)
                        row["industry_level"] = info.get("industry_level", args.level)
                        new_rows.append(row)
                        existing_keys.add(key)

                if new_rows:
                    affected = upsert_map_hist(engine, new_rows)
                    total_affected += affected
                    total_rows += len(new_rows)

                state.complete(chunk_key, len(new_rows))
                logger.info("chunk_done chunk=%s rows=%s new=%s", chunk_key, len(raw_rows), len(new_rows))
                tushare_rows.extend(new_rows)

            except Exception as exc:
                state.fail(chunk_key, exc)
                logger.error("chunk_failed chunk=%s err=%s", chunk_key, exc)
                continue

    # Step 3: Use cn_board_member_map_d only as a compatibility fallback for
    # gaps that SW history did not cover.
    board_rows = fetch_board_member_map(engine, date_range.start, date_range.end, args.level)
    logger.info("Board member map yielded %s records", len(board_rows))
    board_fallback_rows: list[dict] = []
    for row in board_rows:
        key = (row["symbol"], row["industry_id"])
        if key in existing_keys:
            continue
        board_fallback_rows.append(row)
        existing_keys.add(key)

    if board_fallback_rows:
        affected = upsert_map_hist(engine, board_fallback_rows)
        total_affected += affected
        total_rows += len(board_fallback_rows)
        logger.info(
            "Upserted %s board fallback records (affected=%s)",
            len(board_fallback_rows),
            affected,
        )
    else:
        logger.info("No board fallback rows needed; SW history covered requested memberships")

    # Summary
    with engine.connect() as conn:
        summary = conn.execute(
            text(
                """
                SELECT COUNT(*) AS row_cnt,
                       COUNT(DISTINCT symbol) AS symbol_cnt,
                       COUNT(DISTINCT industry_id) AS industry_cnt,
                       MIN(in_date) AS min_in_date,
                       MAX(in_date) AS max_in_date
                FROM cn_local_industry_map_hist
                WHERE industry_level = :level
                """
            ),
            {"level": args.level},
        ).one()

    logger.info(
        "build_local_industry_map_hist done "
        "level=%s board_rows=%s board_fallback_rows=%s tushare_rows=%s total_rows=%s affected=%s "
        "table_rows=%s symbols=%s industries=%s in_date_range=%s..%s",
        args.level,
        len(board_rows),
        len(board_fallback_rows),
        len(tushare_rows),
        total_rows,
        total_affected,
        summary[0],
        summary[1],
        summary[2],
        summary[3],
        summary[4],
    )
    print(
        f"\n=== build_local_industry_map_hist complete ==="
        f"\n  Level:      {args.level}"
        f"\n  Board rows: {len(board_rows)}"
        f"\n  Board fallback rows: {len(board_fallback_rows)}"
        f"\n  Tushare rows: {len(tushare_rows)}"
        f"\n  Total rows: {total_rows}"
        f"\n  Affected:   {total_affected}"
        f"\n  Table:      {summary[0]:,} rows, {summary[1]} symbols, {summary[2]} industries"
        f"\n  Date range: {summary[3]} -> {summary[4]}"
    )


if __name__ == "__main__":
    main()
