"""
scripts/build_local_industry_proxy_daily.py
================================================
GrowthAlpha V8 — Build cn_local_industry_proxy_daily from stock prices and industry mapping.

Input tables:
  - cn_local_industry_map_hist (industry membership with in_date/out_date)
  - cn_stock_daily_price (daily OHLCV)
  - cn_stock_daily_basic (market cap, turnover)
  - cn_stock_leader_score_daily (optional, for leader_return)

Calculated metrics per (industry_id, trade_date):
  - member_count: number of constituent stocks
  - ret_eqw: equal-weighted return of constituents
  - amount_total: total turnover amount
  - turnover_avg: average turnover rate
  - market_cap_total: total market cap
  - leader_return: from leader_score_daily or top-5 return avg
  - top5_concentration: top-5 market cap / total market cap

Supports:
  - --start / --end date range
  - --resume (skip completed chunks)
  - --workers (parallel chunk processing)
  - --chunk-months (month count per chunk)

Usage:
  python scripts/build_local_industry_proxy_daily.py --start 2010-01-01 --end 2026-03-31 --resume --workers 4
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import text

# Bootstrap project root to sys.path so app/ and data_pipeline/ are importable
_project_root = str(Path(__file__).resolve().parents[1])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from app.settings import build_engine
from data_pipeline.common.cli import add_shared_args, month_chunks, resolve_date_range
from data_pipeline.common.db import apply_sql_file
from data_pipeline.common.logging_utils import build_logger
from data_pipeline.common.state import BackfillState


# ---------------------------------------------------------------------------
# SQL templates
# ---------------------------------------------------------------------------

# Main aggregation SQL: computes all proxy metrics from map_hist + daily_price + daily_basic
# NOTE: COLLATE is used on ALL symbol JOINs because tables have mixed collations:
#   cn_local_industry_map_hist  → utf8mb4_0900_ai_ci
#   cn_stock_daily_price        → utf8mb4_unicode_ci
#   cn_stock_daily_basic        → utf8mb4_0900_ai_ci
#   cn_stock_leader_score_daily → utf8mb4_unicode_ci
PROXY_AGG_SQL = """
INSERT INTO cn_local_industry_proxy_daily (
    industry_id, industry_name, trade_date, member_count, ret_eqw,
    amount_total, turnover_avg, market_cap_total, leader_return, top5_concentration,
    industry_level, source
)
WITH base AS (
    SELECT
        m.industry_id,
        m.industry_name,
        p.TRADE_DATE,
        p.SYMBOL,
        COALESCE(b.total_mv, b.circ_mv, 0) AS market_cap,
        COALESCE(p.AMOUNT, 0) AS amount,
        p.TURNOVER_RATE,
        CASE
            WHEN p.PRE_CLOSE IS NOT NULL AND p.PRE_CLOSE <> 0
                THEN (p.CLOSE / p.PRE_CLOSE) - 1
            WHEN p.CHG_PCT IS NOT NULL AND ABS(p.CHG_PCT) > 1
                THEN p.CHG_PCT / 100
            WHEN p.CHG_PCT IS NOT NULL
                THEN p.CHG_PCT
            ELSE NULL
        END AS stock_ret,
        :industry_level AS industry_level
    FROM cn_local_industry_map_hist m
    JOIN cn_stock_daily_price p
        ON CONVERT(p.SYMBOL USING utf8mb4) COLLATE utf8mb4_unicode_ci = CONVERT(m.symbol USING utf8mb4) COLLATE utf8mb4_unicode_ci
        AND p.TRADE_DATE BETWEEN m.in_date AND COALESCE(m.out_date, DATE('2099-12-31'))
    LEFT JOIN cn_stock_daily_basic b
        ON CONVERT(b.symbol USING utf8mb4) COLLATE utf8mb4_unicode_ci = CONVERT(p.SYMBOL USING utf8mb4) COLLATE utf8mb4_unicode_ci
        AND b.trade_date = p.TRADE_DATE
    WHERE p.TRADE_DATE BETWEEN :chunk_start AND :chunk_end
      AND m.industry_level = :industry_level
),
ranked AS (
    SELECT
        base.*,
        ROW_NUMBER() OVER (
            PARTITION BY base.industry_id, base.trade_date
            ORDER BY base.market_cap DESC, base.symbol
        ) AS mc_rank
    FROM base
),
leader_agg AS (
    SELECT
        industry_id,
        trade_date,
        AVG(stock_ret) AS leader_return_avg
    FROM (
        SELECT
            industry_id,
            trade_date,
            stock_ret,
            ROW_NUMBER() OVER (
                PARTITION BY industry_id, trade_date
                ORDER BY COALESCE(stock_ret, -999) DESC
            ) AS ret_rank
        FROM base
        WHERE stock_ret IS NOT NULL
    ) t
    WHERE ret_rank <= 5
    GROUP BY industry_id, trade_date
)
SELECT
    r.industry_id,
    MAX(r.industry_name) AS industry_name,
    r.trade_date,
    COUNT(*) AS member_count,
    AVG(r.stock_ret) AS ret_eqw,
    SUM(r.amount) AS amount_total,
    AVG(r.turnover_rate) AS turnover_avg,
    SUM(r.market_cap) AS market_cap_total,
    COALESCE(
        (SELECT MAX(leader_return_avg) FROM leader_agg la
         WHERE la.industry_id = r.industry_id AND la.trade_date = r.trade_date),
        MAX(CASE WHEN r.mc_rank <= 5 THEN r.stock_ret ELSE NULL END)
    ) AS leader_return,
    CASE
        WHEN SUM(r.market_cap) = 0 THEN NULL
        ELSE SUM(CASE WHEN r.mc_rank <= 5 THEN r.market_cap ELSE 0 END) / SUM(r.market_cap)
    END AS top5_concentration,
    MAX(r.industry_level) AS industry_level,
    'local_proxy_from_stock_daily' AS source
FROM ranked r
GROUP BY r.industry_id, r.trade_date
ON DUPLICATE KEY UPDATE
    industry_name = VALUES(industry_name),
    member_count = VALUES(member_count),
    ret_eqw = VALUES(ret_eqw),
    amount_total = VALUES(amount_total),
    turnover_avg = VALUES(turnover_avg),
    market_cap_total = VALUES(market_cap_total),
    leader_return = VALUES(leader_return),
    top5_concentration = VALUES(top5_concentration),
    industry_level = VALUES(industry_level),
    source = VALUES(source),
    updated_at = CURRENT_TIMESTAMP
"""

# Alternative: use leader_score_daily if available
PROXY_AGG_WITH_LEADER_SCORE_SQL = """
INSERT INTO cn_local_industry_proxy_daily (
    industry_id, industry_name, trade_date, member_count, ret_eqw,
    amount_total, turnover_avg, market_cap_total, leader_return, top5_concentration,
    industry_level, source
)
WITH base AS (
    SELECT
        m.industry_id,
        m.industry_name,
        p.TRADE_DATE,
        p.SYMBOL,
        COALESCE(b.total_mv, b.circ_mv, 0) AS market_cap,
        COALESCE(p.AMOUNT, 0) AS amount,
        p.TURNOVER_RATE,
        CASE
            WHEN p.PRE_CLOSE IS NOT NULL AND p.PRE_CLOSE <> 0
                THEN (p.CLOSE / p.PRE_CLOSE) - 1
            WHEN p.CHG_PCT IS NOT NULL AND ABS(p.CHG_PCT) > 1
                THEN p.CHG_PCT / 100
            WHEN p.CHG_PCT IS NOT NULL
                THEN p.CHG_PCT
            ELSE NULL
        END AS stock_ret,
        ls.leader_score,
        :industry_level AS industry_level
    FROM cn_local_industry_map_hist m
    JOIN cn_stock_daily_price p
        ON CONVERT(p.SYMBOL USING utf8mb4) COLLATE utf8mb4_unicode_ci = CONVERT(m.symbol USING utf8mb4) COLLATE utf8mb4_unicode_ci
        AND p.TRADE_DATE BETWEEN m.in_date AND COALESCE(m.out_date, DATE('2099-12-31'))
    LEFT JOIN cn_stock_daily_basic b
        ON CONVERT(b.symbol USING utf8mb4) COLLATE utf8mb4_unicode_ci = CONVERT(p.SYMBOL USING utf8mb4) COLLATE utf8mb4_unicode_ci
        AND b.trade_date = p.TRADE_DATE
    LEFT JOIN cn_stock_leader_score_daily ls
        ON CONVERT(ls.symbol USING utf8mb4) COLLATE utf8mb4_unicode_ci = CONVERT(p.SYMBOL USING utf8mb4) COLLATE utf8mb4_unicode_ci
        AND ls.trade_date = p.TRADE_DATE
    WHERE p.TRADE_DATE BETWEEN :chunk_start AND :chunk_end
      AND m.industry_level = :industry_level
),
ranked AS (
    SELECT
        base.*,
        ROW_NUMBER() OVER (
            PARTITION BY base.industry_id, base.trade_date
            ORDER BY COALESCE(base.leader_score, -999) DESC, base.market_cap DESC
        ) AS leader_rank,
        ROW_NUMBER() OVER (
            PARTITION BY base.industry_id, base.trade_date
            ORDER BY base.market_cap DESC, base.symbol
        ) AS mc_rank
    FROM base
)
SELECT
    r.industry_id,
    MAX(r.industry_name) AS industry_name,
    r.trade_date,
    COUNT(*) AS member_count,
    AVG(r.stock_ret) AS ret_eqw,
    SUM(r.amount) AS amount_total,
    AVG(r.turnover_rate) AS turnover_avg,
    SUM(r.market_cap) AS market_cap_total,
    MAX(CASE WHEN r.leader_rank = 1 THEN r.stock_ret ELSE NULL END) AS leader_return,
    CASE
        WHEN SUM(r.market_cap) = 0 THEN NULL
        ELSE SUM(CASE WHEN r.mc_rank <= 5 THEN r.market_cap ELSE 0 END) / SUM(r.market_cap)
    END AS top5_concentration,
    MAX(r.industry_level) AS industry_level,
    'local_proxy_from_stock_daily' AS source
FROM ranked r
GROUP BY r.industry_id, r.trade_date
ON DUPLICATE KEY UPDATE
    industry_name = VALUES(industry_name),
    member_count = VALUES(member_count),
    ret_eqw = VALUES(ret_eqw),
    amount_total = VALUES(amount_total),
    turnover_avg = VALUES(turnover_avg),
    market_cap_total = VALUES(market_cap_total),
    leader_return = VALUES(leader_return),
    top5_concentration = VALUES(top5_concentration),
    industry_level = VALUES(industry_level),
    source = VALUES(source),
    updated_at = CURRENT_TIMESTAMP
"""



# ---------------------------------------------------------------------------
# Source data coverage audit
# ---------------------------------------------------------------------------

REQUIRED_SOURCE_TABLES = [
    ("cn_local_industry_map_hist", "in_date"),
    ("cn_stock_daily_price", "TRADE_DATE"),
    ("cn_stock_daily_basic", "trade_date"),
]

OPTIONAL_SOURCE_TABLES = [
    ("cn_stock_leader_score_daily", "trade_date"),
]


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


def _audit_table_range(engine, table_name: str, date_col: str, start: date, end: date, required: bool, logger) -> list[str]:
    failures: list[str] = []
    if not _table_exists_for_audit(engine, table_name):
        status = "MISSING_TABLE"
        logger.error("source_audit table=%s status=%s required=%s", table_name, status, required)
        if required:
            failures.append(f"- {table_name}: {status}")
        return failures

    sql = f"""
        SELECT COUNT(*) AS row_count,
               MIN({date_col}) AS min_date,
               MAX({date_col}) AS max_date
        FROM {table_name}
        WHERE {date_col} IS NOT NULL
    """
    with engine.connect() as conn:
        row = conn.execute(text(sql)).mappings().first()

    row_count = int((row or {}).get("row_count") or 0)
    min_date = _parse_audit_date((row or {}).get("min_date"))
    max_date = _parse_audit_date((row or {}).get("max_date"))

    status = "OK"
    reason = ""
    if row_count <= 0 or min_date is None or max_date is None:
        status = "NO_DATA"
        reason = "empty or invalid date range"
    elif max_date < start:
        status = "RANGE_NOT_COVERED"
        reason = f"available={min_date}~{max_date}, required_through={end}"
    elif table_name != "cn_local_industry_map_hist" and (min_date > start or max_date < end):
        status = "RANGE_NOT_COVERED"
        reason = f"available={min_date}~{max_date}, required={start}~{end}"

    logger.info(
        "source_audit table=%s date_col=%s rows=%s range=%s~%s status=%s required=%s",
        table_name, date_col, row_count, min_date, max_date, status, required,
    )
    if required and status != "OK":
        failures.append(f"- {table_name}.{date_col}: {status} | {reason}")
    return failures


def audit_source_data_coverage(engine, start: date, end: date, logger, use_leader_score: bool = False) -> None:
    logger.info("source_audit_start required_range=%s~%s", start, end)
    failures: list[str] = []
    for table_name, date_col in REQUIRED_SOURCE_TABLES:
        failures.extend(_audit_table_range(engine, table_name, date_col, start, end, True, logger))

    if use_leader_score:
        for table_name, date_col in OPTIONAL_SOURCE_TABLES:
            failures.extend(_audit_table_range(engine, table_name, date_col, start, end, True, logger))
    else:
        for table_name, date_col in OPTIONAL_SOURCE_TABLES:
            _audit_table_range(engine, table_name, date_col, start, end, False, logger)

    if failures:
        logger.error("SOURCE DATA AUDIT FAILED")
        for item in failures:
            logger.error(item)
        raise SystemExit(2)
    logger.info("source_audit_pass")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def check_leader_score_table(engine) -> bool:
    """Check if cn_stock_leader_score_daily exists and has data."""
    with engine.connect() as conn:
        exists = conn.execute(
            text(
                """
                SELECT COUNT(*) FROM INFORMATION_SCHEMA.TABLES
                WHERE TABLE_SCHEMA = DATABASE()
                  AND TABLE_NAME = 'cn_stock_leader_score_daily'
                """
            )
        ).scalar()
        if not exists:
            return False
        has_data = conn.execute(
            text("SELECT COUNT(*) FROM cn_stock_leader_score_daily LIMIT 1")
        ).scalar()
        return has_data > 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="GrowthAlpha V8 — Build cn_local_industry_proxy_daily from stock prices and industry mapping"
    )
    parser.add_argument("--start", default="2010-01-01", help="Start date YYYY-MM-DD or YYYYMMDD")
    parser.add_argument("--end", default="", help="End date YYYY-MM-DD or YYYYMMDD (default: today)")
    parser.add_argument("--resume", action="store_true", help="Skip completed chunks in cn_mainline_backfill_job_state")
    parser.add_argument("--force", action="store_true", help="Force rebuild even if completed")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers (for future use)")
    parser.add_argument("--chunk-months", type=int, default=1, help="Month count per chunk")
    parser.add_argument("--industry-level", default="L1", choices=["L1", "L2", "L3"], help="Industry level")
    parser.add_argument("--use-leader-score", action="store_true", help="Use cn_stock_leader_score_daily for leader_return (auto-detected if available)")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    date_range = resolve_date_range(args.start, args.end)
    logger = build_logger("build_local_industry_proxy_daily")
    engine = build_engine()
    audit_source_data_coverage(engine, date_range.start, date_range.end, logger, args.use_leader_score)

    # Ensure DDL — use CREATE TABLE IF NOT EXISTS for initial creation,
    # then ALTER ADD COLUMN for any columns that may be missing from an
    # earlier schema version of this table.
    logger.info("Ensuring cn_local_industry_proxy_daily table exists")
    ddl_check_sql = """
    CREATE TABLE IF NOT EXISTS `cn_local_industry_proxy_daily` (
        `industry_id` VARCHAR(32) NOT NULL,
        `industry_name` VARCHAR(128) DEFAULT NULL,
        `trade_date` DATE NOT NULL,
        `member_count` INT NOT NULL DEFAULT 0,
        `ret_eqw` DECIMAL(18,8) DEFAULT NULL,
        `amount_total` DECIMAL(24,4) DEFAULT NULL,
        `turnover_avg` DECIMAL(18,6) DEFAULT NULL,
        `market_cap_total` DECIMAL(24,4) DEFAULT NULL,
        `leader_return` DECIMAL(18,8) DEFAULT NULL,
        `top5_concentration` DECIMAL(18,8) DEFAULT NULL,
        `created_at` TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP,
        `updated_at` TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        UNIQUE KEY `uk_industry_trade_date` (`industry_id`, `trade_date`),
        KEY `idx_trade_date` (`trade_date`),
        KEY `idx_industry_id` (`industry_id`)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """
    with engine.begin() as conn:
        conn.execute(text(ddl_check_sql))

    # Ensure all required columns exist (idempotent ALTER ADD COLUMN)
    _ensure_columns_sql = """
    SET @db_name = DATABASE();

    SET @sql = NULL;
    SELECT COUNT(*) INTO @cnt
    FROM INFORMATION_SCHEMA.COLUMNS
    WHERE TABLE_SCHEMA = @db_name
      AND TABLE_NAME = 'cn_local_industry_proxy_daily'
      AND COLUMN_NAME = 'amount_total';
    SET @sql = IF(@cnt = 0,
        'ALTER TABLE `cn_local_industry_proxy_daily` ADD COLUMN `amount_total` DECIMAL(24,4) DEFAULT NULL AFTER `ret_eqw`',
        NULL);
    PREPARE stmt FROM @sql;
    EXECUTE stmt;
    DEALLOCATE PREPARE stmt;

    SET @sql = NULL;
    SELECT COUNT(*) INTO @cnt
    FROM INFORMATION_SCHEMA.COLUMNS
    WHERE TABLE_SCHEMA = @db_name
      AND TABLE_NAME = 'cn_local_industry_proxy_daily'
      AND COLUMN_NAME = 'turnover_avg';
    SET @sql = IF(@cnt = 0,
        'ALTER TABLE `cn_local_industry_proxy_daily` ADD COLUMN `turnover_avg` DECIMAL(18,6) DEFAULT NULL AFTER `amount_total`',
        NULL);
    PREPARE stmt FROM @sql;
    EXECUTE stmt;
    DEALLOCATE PREPARE stmt;

    SET @sql = NULL;
    SELECT COUNT(*) INTO @cnt
    FROM INFORMATION_SCHEMA.COLUMNS
    WHERE TABLE_SCHEMA = @db_name
      AND TABLE_NAME = 'cn_local_industry_proxy_daily'
      AND COLUMN_NAME = 'market_cap_total';
    SET @sql = IF(@cnt = 0,
        'ALTER TABLE `cn_local_industry_proxy_daily` ADD COLUMN `market_cap_total` DECIMAL(24,4) DEFAULT NULL AFTER `turnover_avg`',
        NULL);
    PREPARE stmt FROM @sql;
    EXECUTE stmt;
    DEALLOCATE PREPARE stmt;

    SET @sql = NULL;
    SELECT COUNT(*) INTO @cnt
    FROM INFORMATION_SCHEMA.COLUMNS
    WHERE TABLE_SCHEMA = @db_name
      AND TABLE_NAME = 'cn_local_industry_proxy_daily'
      AND COLUMN_NAME = 'leader_return';
    SET @sql = IF(@cnt = 0,
        'ALTER TABLE `cn_local_industry_proxy_daily` ADD COLUMN `leader_return` DECIMAL(18,8) DEFAULT NULL AFTER `market_cap_total`',
        NULL);
    PREPARE stmt FROM @sql;
    EXECUTE stmt;
    DEALLOCATE PREPARE stmt;

    SET @sql = NULL;
    SELECT COUNT(*) INTO @cnt
    FROM INFORMATION_SCHEMA.COLUMNS
    WHERE TABLE_SCHEMA = @db_name
      AND TABLE_NAME = 'cn_local_industry_proxy_daily'
      AND COLUMN_NAME = 'top5_concentration';
    SET @sql = IF(@cnt = 0,
        'ALTER TABLE `cn_local_industry_proxy_daily` ADD COLUMN `top5_concentration` DECIMAL(18,8) DEFAULT NULL AFTER `leader_return`',
        NULL);
    PREPARE stmt FROM @sql;
    EXECUTE stmt;
    DEALLOCATE PREPARE stmt;
    """
    with engine.begin() as conn:
        for statement in _ensure_columns_sql.strip().split(";"):
            stmt = statement.strip()
            if stmt:
                try:
                    conn.execute(text(stmt + ";"))
                except Exception:
                    pass  # Some statements may be no-op (NULL SET @sql)

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
        KEY `idx_mainline_backfill_job_state_status` (`status`, `updated_at`)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """
    with engine.begin() as conn:
        conn.execute(text(state_ddl))

    state = BackfillState(engine=engine, job_name=f"build_local_industry_proxy_daily_{args.industry_level}")

    # Auto-detect leader score availability
    use_leader_score = args.use_leader_score or check_leader_score_table(engine)
    agg_sql = PROXY_AGG_WITH_LEADER_SCORE_SQL if use_leader_score else PROXY_AGG_SQL
    logger.info(
        "Starting build level=%s start=%s end=%s resume=%s force=%s chunk_months=%s use_leader_score=%s",
        args.industry_level,
        date_range.start,
        date_range.end,
        args.resume,
        args.force,
        args.chunk_months,
        use_leader_score,
    )

    total_affected = 0
    chunks = month_chunks(date_range.start, date_range.end, max(1, int(args.chunk_months)))
    logger.info("Processing %s chunks", len(chunks))

    for chunk_idx, (chunk_start, chunk_end) in enumerate(chunks, start=1):
        chunk_key = f"{args.industry_level}:{chunk_start}:{chunk_end}"
        if args.resume and state.is_completed(chunk_key) and not args.force:
            logger.info("[%s/%s] resume_skip chunk=%s", chunk_idx, len(chunks), chunk_key)
            continue

        state.start(chunk_key, chunk_start, chunk_end)
        logger.info("[%s/%s] processing %s..%s", chunk_idx, len(chunks), chunk_start, chunk_end)

        try:
            with engine.begin() as conn:
                result = conn.execute(
                    text(agg_sql),
                    {
                        "industry_level": args.industry_level,
                        "chunk_start": chunk_start,
                        "chunk_end": chunk_end,
                    },
                )
            affected = int(result.rowcount or 0)
            total_affected += affected
            state.complete(chunk_key, affected)
            logger.info("[%s/%s] chunk_done chunk=%s affected=%s", chunk_idx, len(chunks), chunk_key, affected)
        except Exception as exc:
            state.fail(chunk_key, exc)
            logger.error("[%s/%s] chunk_failed chunk=%s err=%s", chunk_idx, len(chunks), chunk_key, exc)
            raise SystemExit(f"Fatal error processing chunk={chunk_key}: {exc}")

    # Summary
    with engine.connect() as conn:
        summary = conn.execute(
            text(
                """
                SELECT COUNT(*) AS row_cnt,
                       COUNT(DISTINCT industry_id) AS industry_cnt,
                       MIN(trade_date) AS min_trade_date,
                       MAX(trade_date) AS max_trade_date
                FROM cn_local_industry_proxy_daily
                """
            )
        ).one()

    logger.info(
        "build_local_industry_proxy_daily done "
        "level=%s affected=%s chunks=%s "
        "table_rows=%s industries=%s date_range=%s..%s",
        args.industry_level,
        total_affected,
        len(chunks),
        summary[0],
        summary[1],
        summary[2],
        summary[3],
    )
    print(
        f"\n=== build_local_industry_proxy_daily complete ==="
        f"\n  Level:      {args.industry_level}"
        f"\n  Affected:   {total_affected:,}"
        f"\n  Chunks:     {len(chunks)}"
        f"\n  Table:      {summary[0]:,} rows, {summary[1]} industries"
        f"\n  Date range: {summary[2]} -> {summary[3]}"
    )


if __name__ == "__main__":
    main()
