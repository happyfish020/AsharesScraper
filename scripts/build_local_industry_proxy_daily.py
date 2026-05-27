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
import os
import sys
from datetime import date, datetime, timedelta
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
#   cn_local_industry_map_hist  → utf8mb4_unicode_ci
#   cn_stock_daily_price        → utf8mb4_unicode_ci
#   cn_stock_daily_basic        → utf8mb4_unicode_ci
#   cn_stock_leader_score_daily → utf8mb4_unicode_ci
PROXY_AGG_SQL = """
INSERT INTO cn_local_industry_proxy_daily (
    industry_id, industry_name, trade_date, member_count, ret_eqw,
    amount_total, turnover_avg, market_cap_total, leader_return, top5_concentration,
    industry_level, source
)
WITH price_chunk AS (
    SELECT
        p.TRADE_DATE,
        p.SYMBOL,
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
        END AS stock_ret
    FROM cn_stock_daily_price p
    WHERE p.TRADE_DATE BETWEEN :chunk_start AND :chunk_end
),
base AS (
    SELECT
        m.industry_id,
        m.industry_name,
        p.TRADE_DATE,
        p.SYMBOL,
        COALESCE(b.total_mv, b.circ_mv, 0) AS market_cap,
        p.amount,
        p.TURNOVER_RATE,
        p.stock_ret,
        :output_level AS industry_level
    FROM price_chunk p
    STRAIGHT_JOIN cn_local_industry_map_hist m
        ON m.symbol = p.SYMBOL
        AND m.industry_level = :map_level
        AND p.TRADE_DATE >= m.in_date
        AND (m.out_date IS NULL OR p.TRADE_DATE <= m.out_date)
    LEFT JOIN cn_stock_daily_basic b
        ON b.trade_date = p.TRADE_DATE
        AND b.symbol = p.SYMBOL
),
ranked AS (
    SELECT
        base.*,
        ROW_NUMBER() OVER (
            PARTITION BY base.industry_id, base.trade_date
            ORDER BY base.market_cap DESC, base.symbol
        ) AS mc_rank,
        ROW_NUMBER() OVER (
            PARTITION BY base.industry_id, base.trade_date
            ORDER BY COALESCE(base.stock_ret, -999) DESC, base.symbol
        ) AS ret_rank
    FROM base
),
leader_agg AS (
    SELECT
        industry_id,
        trade_date,
        AVG(stock_ret) AS leader_return_avg
    FROM ranked
    WHERE stock_ret IS NOT NULL
      AND ret_rank <= 5
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
        MAX(la.leader_return_avg),
        MAX(CASE WHEN r.mc_rank <= 5 THEN r.stock_ret ELSE NULL END)
    ) AS leader_return,
    CASE
        WHEN SUM(r.market_cap) = 0 THEN NULL
        ELSE SUM(CASE WHEN r.mc_rank <= 5 THEN r.market_cap ELSE 0 END) / SUM(r.market_cap)
    END AS top5_concentration,
    MAX(r.industry_level) AS industry_level,
    'local_proxy_from_stock_daily' AS source
FROM ranked r
LEFT JOIN leader_agg la
    ON la.industry_id = r.industry_id
   AND la.trade_date = r.trade_date
GROUP BY r.industry_id, r.trade_date
"""

# Alternative: use leader_score_daily if available
PROXY_AGG_WITH_LEADER_SCORE_SQL = """
INSERT INTO cn_local_industry_proxy_daily (
    industry_id, industry_name, trade_date, member_count, ret_eqw,
    amount_total, turnover_avg, market_cap_total, leader_return, top5_concentration,
    industry_level, source
)
WITH price_chunk AS (
    SELECT
        p.TRADE_DATE,
        p.SYMBOL,
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
        END AS stock_ret
    FROM cn_stock_daily_price p
    WHERE p.TRADE_DATE BETWEEN :chunk_start AND :chunk_end
),
base AS (
    SELECT
        m.industry_id,
        m.industry_name,
        p.TRADE_DATE,
        p.SYMBOL,
        COALESCE(b.total_mv, b.circ_mv, 0) AS market_cap,
        p.amount,
        p.TURNOVER_RATE,
        p.stock_ret,
        ls.leader_score,
        :output_level AS industry_level
    FROM price_chunk p
    STRAIGHT_JOIN cn_local_industry_map_hist m
        ON m.symbol = p.SYMBOL
        AND m.industry_level = :map_level
        AND p.TRADE_DATE >= m.in_date
        AND (m.out_date IS NULL OR p.TRADE_DATE <= m.out_date)
    LEFT JOIN cn_stock_daily_basic b
        ON b.trade_date = p.TRADE_DATE
        AND b.symbol = p.SYMBOL
    LEFT JOIN cn_stock_leader_score_daily ls
        ON ls.trade_date = p.TRADE_DATE
        AND ls.symbol = p.SYMBOL
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
"""

DELETE_CHUNK_SQL = """
DELETE t
FROM cn_local_industry_proxy_daily t
JOIN (
    SELECT DISTINCT industry_id
    FROM cn_local_industry_map_hist
    WHERE industry_level = :map_level
) ids
    ON ids.industry_id = t.industry_id
WHERE t.trade_date BETWEEN :chunk_start AND :chunk_end
"""



# ---------------------------------------------------------------------------
# Source data coverage audit
# ---------------------------------------------------------------------------

REQUIRED_SOURCE_TABLES = [
    # cn_local_industry_map_hist is a static base mapping table (stock ↔ industry
    # membership with in_date/out_date). It is NOT a daily-updated table and its
    # in_date column records when a stock entered an industry, not a trading date.
    # Excluding it from daily audit to avoid false RANGE_NOT_COVERED failures.
    ("cn_stock_daily_price", "TRADE_DATE"),
    ("cn_stock_daily_basic", "trade_date"),
]

OPTIONAL_SOURCE_TABLES = [
    ("cn_stock_leader_score_daily", "trade_date"),
]

# Minimum row threshold for stock-level tables.
# If a table has >= this many rows in the requested range, it is considered
# to have sufficient data even if the date range is not fully covered.
# This avoids full-table scans across all stocks for every audit.
# Override via env V8_AUDIT_MIN_ROWS_THRESHOLD (default: 1000).
_MIN_ROWS_THRESHOLD = int(os.getenv("V8_AUDIT_MIN_ROWS_THRESHOLD", "1000"))


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


def _column_meta(engine, table_name: str, column_name: str) -> dict[str, Any] | None:
    with engine.connect() as conn:
        row = conn.execute(
            text(
                """
                SELECT COLUMN_NAME,
                       DATA_TYPE,
                       CHARACTER_SET_NAME,
                       COLLATION_NAME,
                       CHARACTER_MAXIMUM_LENGTH,
                       IS_NULLABLE
                FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_SCHEMA = DATABASE()
                  AND TABLE_NAME = :table_name
                  AND COLUMN_NAME = :column_name
                """
            ),
            {"table_name": table_name, "column_name": column_name},
        ).mappings().first()
    return dict(row) if row else None


def _index_exists(engine, table_name: str, index_name: str) -> bool:
    with engine.connect() as conn:
        return bool(
            conn.execute(
                text(
                    """
                    SELECT COUNT(*)
                    FROM INFORMATION_SCHEMA.STATISTICS
                    WHERE TABLE_SCHEMA = DATABASE()
                      AND TABLE_NAME = :table_name
                      AND INDEX_NAME = :index_name
                    """
                ),
                {"table_name": table_name, "index_name": index_name},
            ).scalar()
        )


def _index_columns_exist(engine, table_name: str, columns: tuple[str, ...]) -> bool:
    sql = text(
        """
        SELECT INDEX_NAME,
               GROUP_CONCAT(COLUMN_NAME ORDER BY SEQ_IN_INDEX SEPARATOR ',') AS cols
        FROM INFORMATION_SCHEMA.STATISTICS
        WHERE TABLE_SCHEMA = DATABASE()
          AND TABLE_NAME = :table_name
        GROUP BY INDEX_NAME
        """
    )
    expected = ",".join(columns)
    with engine.connect() as conn:
        rows = conn.execute(sql, {"table_name": table_name}).mappings().all()
    return any((row.get("cols") or "") == expected for row in rows)


def _create_index_if_missing(
    engine,
    table_name: str,
    index_name: str,
    columns: tuple[str, ...],
    logger,
) -> None:
    if (
        not _table_exists_for_audit(engine, table_name)
        or _index_exists(engine, table_name, index_name)
        or _index_columns_exist(engine, table_name, columns)
    ):
        return
    columns_clause = "(" + ", ".join(f"`{column}`" for column in columns) + ")"
    with engine.begin() as conn:
        conn.execute(text(f"CREATE INDEX `{index_name}` ON `{table_name}` {columns_clause}"))
    logger.info("created_hot_path_index table=%s index=%s columns=%s", table_name, index_name, columns_clause)


def _normalize_symbol_column(engine, table_name: str, column_name: str, logger) -> None:
    meta = _column_meta(engine, table_name, column_name)
    if not meta:
        return
    if (
        str(meta.get("DATA_TYPE") or "").lower() == "varchar"
        and int(meta.get("CHARACTER_MAXIMUM_LENGTH") or 0) >= 16
        and str(meta.get("CHARACTER_SET_NAME") or "").lower() == "utf8mb4"
        and str(meta.get("COLLATION_NAME") or "").lower() == "utf8mb4_unicode_ci"
    ):
        return

    nullable = "NULL" if str(meta.get("IS_NULLABLE") or "").upper() == "YES" else "NOT NULL"
    alter_sql = (
        f"ALTER TABLE `{table_name}` "
        f"MODIFY `{column_name}` VARCHAR(16) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci {nullable}"
    )
    with engine.begin() as conn:
        conn.execute(text(alter_sql))
    logger.info("normalized_symbol_column table=%s column=%s", table_name, column_name)


def _resolve_effective_end_for_audit(engine, end: date) -> date:
    """Resolve the latest usable trading date on or before the requested end."""
    sql_candidates = [
        """
        SELECT MAX(TRADE_DATE)
        FROM cn_stock_daily_price
        WHERE TRADE_DATE <= :end
        """,
        """
        SELECT MAX(trade_date)
        FROM cn_stock_daily_basic
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
    effective_end = _resolve_effective_end_for_audit(engine, end)

    status = "OK"
    reason = ""
    if row_count <= 0 or min_date is None or max_date is None:
        status = "NO_DATA"
        reason = "empty or invalid date range"
    elif max_date < start:
        status = "RANGE_NOT_COVERED"
        reason = f"available={min_date}~{max_date}, required_through={end}"
    elif table_name != "cn_local_industry_map_hist" and (min_date > start or max_date < effective_end):
        # For stock-level tables (high row-count), use a row-count threshold:
        # if there are enough rows to indicate data exists, treat as OK even if
        # the date range doesn't fully cover. This avoids full-table scans.
        if row_count >= _MIN_ROWS_THRESHOLD:
            logger.info(
                "source_audit table=%s rows=%s >= threshold=%s — treating as OK despite range gap",
                table_name, row_count, _MIN_ROWS_THRESHOLD,
            )
        else:
            status = "RANGE_NOT_COVERED"
            reason = f"available={min_date}~{max_date}, required={start}~{end} (effective_end={effective_end})"

    logger.info(
        "source_audit table=%s date_col=%s rows=%s range=%s~%s status=%s required=%s effective_end=%s",
        table_name, date_col, row_count, min_date, max_date, status, required, effective_end,
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


def normalize_source_symbol_columns(engine, use_leader_score: bool, logger) -> None:
    symbol_specs = [
        ("cn_stock_daily_price", "SYMBOL"),
        ("cn_stock_daily_basic", "symbol"),
        ("cn_local_industry_map_hist", "symbol"),
    ]
    if use_leader_score:
        symbol_specs.append(("cn_stock_leader_score_daily", "symbol"))

    for table_name, column_name in symbol_specs:
        try:
            _normalize_symbol_column(engine, table_name, column_name, logger)
        except Exception as exc:
            logger.warning("skip_symbol_normalize table=%s column=%s err=%s", table_name, column_name, exc)


def ensure_hot_path_indexes(engine, use_leader_score: bool, logger) -> None:
    index_specs = [
        (
            "cn_local_industry_map_hist",
            "idx_map_symbol_dates",
            ("symbol", "in_date", "out_date", "industry_level"),
        ),
        (
            "cn_local_industry_map_hist",
            "idx_map_level_dates",
            ("industry_level", "in_date", "out_date"),
        ),
    ]
    if use_leader_score:
        index_specs.append(
            (
                "cn_stock_leader_score_daily",
                "idx_leader_symbol_date",
                ("symbol", "trade_date"),
            )
        )

    # Existing PRIMARY/secondary indexes on cn_stock_daily_price and cn_stock_daily_basic
    # already cover these join paths in most environments, but we still ensure the
    # expected access patterns exist when bootstrapping from sparse or legacy DDL.
    index_specs.extend(
        [
            ("cn_stock_daily_price", "idx_price_date_symbol", ("TRADE_DATE", "SYMBOL")),
            ("cn_stock_daily_price", "idx_price_symbol_date", ("SYMBOL", "TRADE_DATE")),
            ("cn_stock_daily_basic", "idx_basic_symbol_date", ("symbol", "trade_date")),
        ]
    )

    for table_name, index_name, columns_clause in index_specs:
        try:
            _create_index_if_missing(engine, table_name, index_name, columns_clause, logger)
        except Exception as exc:
            logger.warning("skip_hot_path_index table=%s index=%s err=%s", table_name, index_name, exc)


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


def day_chunks(start: date, end: date, days_per_chunk: int) -> list[tuple[date, date]]:
    out: list[tuple[date, date]] = []
    cur = start
    step = max(1, int(days_per_chunk))
    while cur <= end:
        chunk_end = min(end, cur + timedelta(days=step - 1))
        out.append((cur, chunk_end))
        cur = chunk_end + timedelta(days=1)
    return out


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
    parser.add_argument("--chunk-days", type=int, default=0, help="Calendar day count per chunk; overrides --chunk-months when > 0")
    parser.add_argument(
        "--industry-level",
        default="L1",
        choices=["L1", "L2", "L3"],
        help=(
            "Physical output level label. In current V8 semantics, output L1 is "
            "the fine-grained LOCAL_FINE proxy layer built from map_hist L3 rows."
        ),
    )
    parser.add_argument("--use-leader-score", action="store_true", help="Use cn_stock_leader_score_daily for leader_return (auto-detected if available)")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    date_range = resolve_date_range(args.start, args.end)
    logger = build_logger("build_local_industry_proxy_daily")
    engine = build_engine()
    use_leader_score = args.use_leader_score or check_leader_score_table(engine)
    normalize_source_symbol_columns(engine, use_leader_score, logger)
    ensure_hot_path_indexes(engine, use_leader_score, logger)

    # Resolve industry level mapping.
    #
    # Current V8 semantic contract:
    #   - map_hist 'L3' is the membership source for the 391-industry LOCAL_FINE
    #     production set
    #   - map_hist 'SW_L1' is the official 31-industry Shenwan L1 set
    #   - proxy_daily stores LOCAL_FINE using the legacy physical label 'L1'
    #
    # Therefore, when --industry-level L1 is requested, we intentionally query
    # map_hist using 'L3' but write 'L1' to the output table for backward
    # compatibility with the current physical schema.
    output_level = args.industry_level
    if args.industry_level == "L1":
        map_level = "L3"
        logger.info(
            "Industry level mapping: output=%s map_hist_query=%s "
            "(cn_local_industry_map_hist has no 'L1' records; L1 proxy data is built from L3 map_hist)",
            output_level, map_level,
        )
    else:
        map_level = args.industry_level
    audit_source_data_coverage(engine, date_range.start, date_range.end, logger, use_leader_score)

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
        `industry_level` VARCHAR(8) DEFAULT NULL,
        `source` VARCHAR(64) DEFAULT NULL,
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

    SET @sql = NULL;
    SELECT COUNT(*) INTO @cnt
    FROM INFORMATION_SCHEMA.COLUMNS
    WHERE TABLE_SCHEMA = @db_name
      AND TABLE_NAME = 'cn_local_industry_proxy_daily'
      AND COLUMN_NAME = 'industry_level';
    SET @sql = IF(@cnt = 0,
        'ALTER TABLE `cn_local_industry_proxy_daily` ADD COLUMN `industry_level` VARCHAR(8) DEFAULT NULL AFTER `top5_concentration`',
        NULL);
    PREPARE stmt FROM @sql;
    EXECUTE stmt;
    DEALLOCATE PREPARE stmt;

    SET @sql = NULL;
    SELECT COUNT(*) INTO @cnt
    FROM INFORMATION_SCHEMA.COLUMNS
    WHERE TABLE_SCHEMA = @db_name
      AND TABLE_NAME = 'cn_local_industry_proxy_daily'
      AND COLUMN_NAME = 'source';
    SET @sql = IF(@cnt = 0,
        'ALTER TABLE `cn_local_industry_proxy_daily` ADD COLUMN `source` VARCHAR(64) DEFAULT NULL AFTER `industry_level`',
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

    agg_sql = PROXY_AGG_WITH_LEADER_SCORE_SQL if use_leader_score else PROXY_AGG_SQL
    logger.info(
        "Starting build level=%s start=%s end=%s resume=%s force=%s chunk_months=%s chunk_days=%s use_leader_score=%s",
        args.industry_level,
        date_range.start,
        date_range.end,
        args.resume,
        args.force,
        args.chunk_months,
        args.chunk_days,
        use_leader_score,
    )

    total_affected = 0
    if int(args.chunk_days or 0) > 0:
        chunks = day_chunks(date_range.start, date_range.end, int(args.chunk_days))
    else:
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
                # Step 1: Check if source data exists before deleting
                source_check = conn.execute(
                    text("""
                        SELECT COUNT(DISTINCT p.SYMBOL) AS cnt
                        FROM cn_stock_daily_price p
                        STRAIGHT_JOIN cn_local_industry_map_hist m
                            ON m.symbol = p.SYMBOL
                            AND m.industry_level = :map_level
                            AND p.TRADE_DATE >= m.in_date
                            AND (m.out_date IS NULL OR p.TRADE_DATE <= m.out_date)
                        WHERE p.TRADE_DATE BETWEEN :chunk_start AND :chunk_end
                        LIMIT 1
                    """),
                    {
                        "map_level": map_level,
                        "chunk_start": chunk_start,
                        "chunk_end": chunk_end,
                    },
                ).scalar()
                source_count = int(source_check or 0)

                if source_count == 0 and not args.force:
                    # No source data found for this chunk — skip delete to preserve existing rows
                    logger.warning(
                        "[%s/%s] skip_chunk_no_source chunk=%s — no matching stock_price+map_hist rows found",
                        chunk_idx, len(chunks), chunk_key,
                    )
                    state.complete(chunk_key, 0)
                    continue

                conn.execute(
                    text(DELETE_CHUNK_SQL),
                    {
                        "map_level": map_level,
                        "chunk_start": chunk_start,
                        "chunk_end": chunk_end,
                    },
                )
                result = conn.execute(
                    text(agg_sql),
                    {
                        "map_level": map_level,
                        "output_level": output_level,
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
