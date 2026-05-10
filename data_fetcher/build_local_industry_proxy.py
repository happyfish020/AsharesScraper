from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

from sqlalchemy import text

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    sys.path.append(str(Path(__file__).resolve().parent))
    from app.settings import build_engine
    from config import (
        LOCAL_INDUSTRY_LEVEL,
        LOCAL_INDUSTRY_MAP_TABLE,
        LOCAL_MAP_MEMBER_SOURCE,
        LOCAL_INDUSTRY_PROXY_LOG_FILE,
        LOCAL_INDUSTRY_PROXY_TABLE,
        LOCAL_MAP_BUILD_SOURCE,
        LOCAL_MAP_MASTER_SOURCE,
        LOCAL_PROXY_SOURCE,
        LOCAL_TASK_STATUS_TABLE,
    )
else:
    from app.settings import build_engine
    from .config import (
        LOCAL_INDUSTRY_LEVEL,
        LOCAL_INDUSTRY_MAP_TABLE,
        LOCAL_MAP_MEMBER_SOURCE,
        LOCAL_INDUSTRY_PROXY_LOG_FILE,
        LOCAL_INDUSTRY_PROXY_TABLE,
        LOCAL_MAP_BUILD_SOURCE,
        LOCAL_MAP_MASTER_SOURCE,
        LOCAL_PROXY_SOURCE,
        LOCAL_TASK_STATUS_TABLE,
    )


@dataclass(frozen=True)
class RunOptions:
    mode: str
    start_date: date | None
    end_date: date | None
    weekly_lookback_days: int


def setup_logger() -> logging.Logger:
    LOCAL_INDUSTRY_PROXY_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("local_industry_proxy")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    file_handler = logging.FileHandler(LOCAL_INDUSTRY_PROXY_LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger


def parse_date(text_value: str | None) -> date | None:
    value = (text_value or "").strip()
    if not value:
        return None
    if len(value) == 8:
        return datetime.strptime(value, "%Y%m%d").date()
    return datetime.strptime(value, "%Y-%m-%d").date()


def parse_args() -> RunOptions:
    parser = argparse.ArgumentParser(description="Build local SW L1 mapping and proxy index from local DB tables.")
    parser.add_argument(
        "--mode",
        choices=["refresh-map", "validate-map", "daily", "weekly", "backfill", "audit", "all"],
        default="daily",
    )
    parser.add_argument("--start", default="", help="start date in YYYYMMDD or YYYY-MM-DD")
    parser.add_argument("--end", default="", help="end date in YYYYMMDD or YYYY-MM-DD")
    parser.add_argument("--weekly-lookback-days", type=int, default=35, help="weekly rebuild lookback days")
    args = parser.parse_args()
    start_date = parse_date(args.start)
    end_date = parse_date(args.end)
    if start_date and end_date and start_date > end_date:
        raise ValueError("start date cannot be greater than end date")
    if args.weekly_lookback_days < 1:
        raise ValueError("weekly-lookback-days must be >= 1")
    return RunOptions(args.mode, start_date, end_date, args.weekly_lookback_days)


def ensure_column(engine, table_name: str, column_name: str, ddl_fragment: str) -> None:
    sql = text(
        """
        SELECT COUNT(*)
        FROM information_schema.columns
        WHERE table_schema = 'cn_market'
          AND table_name = :table_name
          AND column_name = :column_name
        """
    )
    with engine.begin() as conn:
        exists = conn.execute(sql, {"table_name": table_name, "column_name": column_name}).scalar() or 0
        if not exists:
            conn.execute(text(f"ALTER TABLE {table_name} ADD COLUMN {ddl_fragment}"))


def ensure_schema(engine, logger: logging.Logger) -> None:
    logger.info("ensuring local proxy schema")
    ddl_statements = [
        f"""
        CREATE TABLE IF NOT EXISTS {LOCAL_INDUSTRY_MAP_TABLE} (
            symbol VARCHAR(10) NOT NULL,
            industry_id VARCHAR(32) NOT NULL,
            industry_name VARCHAR(80) DEFAULT NULL,
            industry_level VARCHAR(16) NOT NULL,
            valid_from DATE NOT NULL,
            valid_to DATE NOT NULL,
            source VARCHAR(64) NOT NULL,
            is_manual_override TINYINT(1) NOT NULL DEFAULT 0,
            updated_at TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            PRIMARY KEY (symbol, industry_id, valid_from),
            KEY idx_{LOCAL_INDUSTRY_MAP_TABLE}_industry (industry_id, valid_from, valid_to),
            KEY idx_{LOCAL_INDUSTRY_MAP_TABLE}_symbol (symbol, valid_from, valid_to)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci
        """,
        f"""
        CREATE TABLE IF NOT EXISTS {LOCAL_INDUSTRY_PROXY_TABLE} (
            trade_date DATE NOT NULL,
            industry_id VARCHAR(32) NOT NULL,
            industry_name VARCHAR(80) DEFAULT NULL,
            industry_level VARCHAR(16) NOT NULL,
            member_count INT NOT NULL,
            total_member_count INT NOT NULL,
            coverage_ratio DECIMAL(12,6) DEFAULT NULL,
            ret_eqw DECIMAL(18,8) DEFAULT NULL,
            ret_amt_w DECIMAL(18,8) DEFAULT NULL,
            amount_sum DECIMAL(24,4) DEFAULT NULL,
            up_count INT NOT NULL DEFAULT 0,
            down_count INT NOT NULL DEFAULT 0,
            up_ratio DECIMAL(12,6) DEFAULT NULL,
            rs20_eqw DECIMAL(18,8) DEFAULT NULL,
            rs20_amt_w DECIMAL(18,8) DEFAULT NULL,
            rs60_eqw DECIMAL(18,8) DEFAULT NULL,
            rs60_amt_w DECIMAL(18,8) DEFAULT NULL,
            proxy_close_eqw DECIMAL(18,4) DEFAULT NULL,
            proxy_close_amt_w DECIMAL(18,4) DEFAULT NULL,
            source VARCHAR(64) NOT NULL,
            updated_at TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            PRIMARY KEY (trade_date, industry_id),
            KEY idx_{LOCAL_INDUSTRY_PROXY_TABLE}_industry (industry_id, trade_date)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci
        """,
        f"""
        CREATE TABLE IF NOT EXISTS {LOCAL_TASK_STATUS_TABLE} (
            task_name VARCHAR(64) NOT NULL,
            last_success_date DATE DEFAULT NULL,
            last_checked_date DATE DEFAULT NULL,
            last_source_max_date DATE DEFAULT NULL,
            status VARCHAR(16) NOT NULL,
            message VARCHAR(255) DEFAULT NULL,
            updated_at TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            PRIMARY KEY (task_name)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci
        """,
    ]
    with engine.begin() as conn:
        for ddl in ddl_statements:
            conn.execute(text(ddl))

    ensure_column(engine, LOCAL_INDUSTRY_MAP_TABLE, "valid_to", "`valid_to` DATE NOT NULL")
    ensure_column(engine, LOCAL_INDUSTRY_PROXY_TABLE, "total_member_count", "`total_member_count` INT NOT NULL DEFAULT 0 AFTER `member_count`")
    ensure_column(engine, LOCAL_INDUSTRY_PROXY_TABLE, "coverage_ratio", "`coverage_ratio` DECIMAL(12,6) DEFAULT NULL AFTER `total_member_count`")
    ensure_column(engine, LOCAL_INDUSTRY_PROXY_TABLE, "amount_sum", "`amount_sum` DECIMAL(24,4) DEFAULT NULL AFTER `ret_amt_w`")
    ensure_column(engine, LOCAL_INDUSTRY_PROXY_TABLE, "up_count", "`up_count` INT NOT NULL DEFAULT 0 AFTER `amount_sum`")
    ensure_column(engine, LOCAL_INDUSTRY_PROXY_TABLE, "down_count", "`down_count` INT NOT NULL DEFAULT 0 AFTER `up_count`")
    ensure_column(engine, LOCAL_INDUSTRY_PROXY_TABLE, "up_ratio", "`up_ratio` DECIMAL(12,6) DEFAULT NULL AFTER `down_count`")
    ensure_column(engine, LOCAL_INDUSTRY_PROXY_TABLE, "rs20_eqw", "`rs20_eqw` DECIMAL(18,8) DEFAULT NULL AFTER `up_ratio`")
    ensure_column(engine, LOCAL_INDUSTRY_PROXY_TABLE, "rs20_amt_w", "`rs20_amt_w` DECIMAL(18,8) DEFAULT NULL AFTER `rs20_eqw`")
    ensure_column(engine, LOCAL_INDUSTRY_PROXY_TABLE, "rs60_eqw", "`rs60_eqw` DECIMAL(18,8) DEFAULT NULL AFTER `rs20_amt_w`")
    ensure_column(engine, LOCAL_INDUSTRY_PROXY_TABLE, "rs60_amt_w", "`rs60_amt_w` DECIMAL(18,8) DEFAULT NULL AFTER `rs60_eqw`")


def update_task_status(engine, task_name: str, status: str, message: str, success_date: date | None, source_max_date: date | None) -> None:
    sql = text(
        f"""
        INSERT INTO {LOCAL_TASK_STATUS_TABLE} (
            task_name, last_success_date, last_checked_date, last_source_max_date, status, message
        )
        VALUES (:task_name, :last_success_date, CURRENT_DATE(), :last_source_max_date, :status, :message)
        ON DUPLICATE KEY UPDATE
            last_success_date = VALUES(last_success_date),
            last_checked_date = VALUES(last_checked_date),
            last_source_max_date = VALUES(last_source_max_date),
            status = VALUES(status),
            message = VALUES(message)
        """
    )
    with engine.begin() as conn:
        conn.execute(
            sql,
            {
                "task_name": task_name,
                "last_success_date": success_date,
                "last_source_max_date": source_max_date,
                "status": status,
                "message": message[:255],
            },
        )


def scalar_date(engine, sql_text: str, params: dict | None = None) -> date | None:
    with engine.connect() as conn:
        return conn.execute(text(sql_text), params or {}).scalar()


def resolve_sw_l1_source_max_date(engine) -> date | None:
    return scalar_date(
        engine,
        f"""
        WITH sw_l1 AS (
            SELECT DISTINCT board_id
            FROM cn_board_industry_master
            WHERE source = :master_source
        )
        SELECT MAX(COALESCE(h.valid_to, DATE('9999-12-31')))
        FROM cn_board_industry_member_hist h
        JOIN sw_l1 s
          ON s.board_id COLLATE utf8mb4_general_ci = h.board_id COLLATE utf8mb4_general_ci
        WHERE h.source = :member_source
        """,
        {"master_source": LOCAL_MAP_MASTER_SOURCE, "member_source": LOCAL_MAP_MEMBER_SOURCE},
    )


def resolve_sw_l1_source_min_date(engine) -> date | None:
    return scalar_date(
        engine,
        f"""
        WITH sw_l1 AS (
            SELECT DISTINCT board_id
            FROM cn_board_industry_master
            WHERE source = :master_source
        )
        SELECT MIN(h.valid_from)
        FROM cn_board_industry_member_hist h
        JOIN sw_l1 s
          ON s.board_id COLLATE utf8mb4_general_ci = h.board_id COLLATE utf8mb4_general_ci
        WHERE h.source = :member_source
        """,
        {"master_source": LOCAL_MAP_MASTER_SOURCE, "member_source": LOCAL_MAP_MEMBER_SOURCE},
    )


def validate_source_daily_uniqueness(engine) -> tuple[int, int, int]:
    with engine.connect() as conn:
        duplicate_cnt = conn.execute(
            text(
                f"""
                WITH sw_l1 AS (
                    SELECT DISTINCT board_id
                    FROM cn_board_industry_master
                    WHERE source = :master_source
                )
                SELECT COUNT(*)
                FROM (
                    SELECT h.symbol, h.valid_from, COALESCE(h.valid_to, DATE('9999-12-31'))
                    FROM cn_board_industry_member_hist h
                    JOIN sw_l1 s
                      ON s.board_id COLLATE utf8mb4_general_ci = h.board_id COLLATE utf8mb4_general_ci
                    WHERE h.source = :member_source
                    GROUP BY h.symbol, h.valid_from, COALESCE(h.valid_to, DATE('9999-12-31'))
                    HAVING COUNT(*) > 1
                ) x
                """
            ),
            {"master_source": LOCAL_MAP_MASTER_SOURCE, "member_source": LOCAL_MAP_MEMBER_SOURCE},
        ).scalar() or 0
        invalid_industry_cnt = conn.execute(
            text(
                f"""
                SELECT COUNT(*)
                FROM cn_board_industry_member_hist h
                WHERE h.source = :member_source
                  AND (h.board_id IS NULL OR TRIM(h.board_id) = '')
                """
            ),
            {"member_source": LOCAL_MAP_MEMBER_SOURCE},
        ).scalar() or 0
        max_date = resolve_sw_l1_source_max_date(engine)
        coverage_cnt = conn.execute(
            text(
                f"""
                WITH sw_l1 AS (
                    SELECT DISTINCT board_id
                    FROM cn_board_industry_master
                    WHERE source = :master_source
                )
                SELECT COUNT(DISTINCT h.symbol)
                FROM cn_board_industry_member_hist h
                JOIN sw_l1 s
                  ON s.board_id COLLATE utf8mb4_general_ci = h.board_id COLLATE utf8mb4_general_ci
                WHERE h.source = :member_source
                  AND :max_date BETWEEN h.valid_from AND COALESCE(h.valid_to, DATE('9999-12-31'))
                """
            ),
            {"master_source": LOCAL_MAP_MASTER_SOURCE, "member_source": LOCAL_MAP_MEMBER_SOURCE, "max_date": max_date},
        ).scalar() or 0
    return int(duplicate_cnt), int(invalid_industry_cnt), int(coverage_cnt)


def validate_local_map(engine, logger: logging.Logger) -> None:
    with engine.connect() as conn:
        empty_industry_cnt = conn.execute(
            text(
                f"""
                SELECT COUNT(*)
                FROM {LOCAL_INDUSTRY_MAP_TABLE}
                WHERE industry_id IS NULL OR TRIM(industry_id) = ''
                """
            )
        ).scalar() or 0
        invalid_date_cnt = conn.execute(
            text(
                f"""
                SELECT COUNT(*)
                FROM {LOCAL_INDUSTRY_MAP_TABLE}
                WHERE valid_from > valid_to
                """
            )
        ).scalar() or 0
        overlap_cnt = conn.execute(
            text(
                f"""
                SELECT COUNT(*)
                FROM {LOCAL_INDUSTRY_MAP_TABLE} a
                JOIN {LOCAL_INDUSTRY_MAP_TABLE} b
                  ON a.symbol COLLATE utf8mb4_general_ci = b.symbol COLLATE utf8mb4_general_ci
                 AND (
                     a.valid_from < b.valid_from
                     OR (a.valid_from = b.valid_from AND a.industry_id COLLATE utf8mb4_general_ci < b.industry_id COLLATE utf8mb4_general_ci)
                 )
                 AND a.valid_from <= b.valid_to
                 AND b.valid_from <= a.valid_to
                """
            )
        ).scalar() or 0
    logger.info(
        "map_validation empty_industry_id=%s invalid_date_range=%s overlapping_windows=%s",
        empty_industry_cnt,
        invalid_date_cnt,
        overlap_cnt,
    )
    if empty_industry_cnt or invalid_date_cnt or overlap_cnt:
        raise RuntimeError(
            f"local map validation failed: empty_industry_id={empty_industry_cnt}, "
            f"invalid_date_range={invalid_date_cnt}, overlapping_windows={overlap_cnt}"
        )


def refresh_local_map(engine, logger: logging.Logger) -> tuple[int, date | None]:
    duplicate_cnt, invalid_industry_cnt, coverage_cnt = validate_source_daily_uniqueness(engine)
    logger.info(
        "source_validation duplicate_windows=%s invalid_industry_rows=%s latest_symbol_coverage=%s source=%s",
        duplicate_cnt,
        invalid_industry_cnt,
        coverage_cnt,
        LOCAL_MAP_MEMBER_SOURCE,
    )
    if duplicate_cnt:
        raise RuntimeError(f"source map has duplicate symbol windows: {duplicate_cnt}")

    source_min_date = resolve_sw_l1_source_min_date(engine)
    latest_source_date = resolve_sw_l1_source_max_date(engine)
    if source_min_date is None or latest_source_date is None:
        raise RuntimeError("no SW L1 history coverage found in cn_board_industry_member_hist")
    with engine.begin() as conn:
        deleted = conn.execute(
            text(
                f"""
                DELETE FROM {LOCAL_INDUSTRY_MAP_TABLE}
                WHERE is_manual_override = 0
                """
            ),
        ).rowcount or 0
        logger.info("local map cleared rows=%s", deleted)
        inserted = conn.execute(
            text(
                f"""
                INSERT INTO {LOCAL_INDUSTRY_MAP_TABLE} (
                    symbol, industry_id, industry_name, industry_level, valid_from, valid_to, source, is_manual_override
                )
                WITH sw_l1 AS (
                    SELECT board_id, board_name
                    FROM (
                        SELECT
                            m.board_id,
                            m.board_name,
                            ROW_NUMBER() OVER (PARTITION BY m.board_id ORDER BY m.asof_date DESC) AS rn
                        FROM cn_board_industry_master m
                        WHERE m.source = :master_source
                    ) x
                    WHERE x.rn = 1
                ),
                hist_base AS (
                    SELECT
                        h.symbol,
                        h.board_id AS industry_id,
                        COALESCE(s.board_name, h.board_id) AS industry_name,
                        h.valid_from
                    FROM cn_board_industry_member_hist h
                    JOIN sw_l1 s
                      ON s.board_id COLLATE utf8mb4_general_ci = h.board_id COLLATE utf8mb4_general_ci
                    WHERE h.source = :member_source
                      AND h.valid_from BETWEEN :source_min_date AND :source_max_date
                ),
                hist_final AS (
                    SELECT
                        symbol,
                        industry_id,
                        industry_name,
                        valid_from,
                        COALESCE(
                            DATE_SUB(
                                LEAD(valid_from) OVER (PARTITION BY symbol ORDER BY valid_from),
                                INTERVAL 1 DAY
                            ),
                            DATE('9999-12-31')
                        ) AS valid_to
                    FROM hist_base
                )
                SELECT
                    symbol,
                    industry_id,
                    industry_name,
                    :industry_level AS industry_level,
                    valid_from,
                    valid_to,
                    :build_source AS source,
                    0 AS is_manual_override
                FROM hist_final
                """
            ),
            {
                "master_source": LOCAL_MAP_MASTER_SOURCE,
                "member_source": LOCAL_MAP_MEMBER_SOURCE,
                "industry_level": LOCAL_INDUSTRY_LEVEL,
                "build_source": LOCAL_MAP_BUILD_SOURCE,
                "source_min_date": source_min_date,
                "source_max_date": latest_source_date,
            },
        ).rowcount or 0
    validate_local_map(engine, logger)
    logger.info(
        "local map refreshed rows=%s source_min_date=%s source_max_date=%s",
        inserted,
        source_min_date,
        latest_source_date,
    )
    update_task_status(
        engine,
        task_name="local_industry_map_refresh",
        status="OK",
        message=f"refreshed rows={inserted} coverage={coverage_cnt}",
        success_date=latest_source_date,
        source_max_date=latest_source_date,
    )
    return inserted, latest_source_date


def resolve_proxy_safe_end_date(engine) -> date | None:
    return scalar_date(
        engine,
        f"""
        SELECT MAX(p.trade_date)
        FROM cn_stock_daily_price p
        WHERE EXISTS (
            SELECT 1
            FROM {LOCAL_INDUSTRY_MAP_TABLE} m
            WHERE m.symbol COLLATE utf8mb4_general_ci = p.symbol COLLATE utf8mb4_general_ci
              AND p.trade_date BETWEEN m.valid_from AND m.valid_to
        )
        """,
    )


def resolve_proxy_min_date(engine) -> date | None:
    return scalar_date(
        engine,
        f"""
        SELECT MIN(p.trade_date)
        FROM cn_stock_daily_price p
        WHERE EXISTS (
            SELECT 1
            FROM {LOCAL_INDUSTRY_MAP_TABLE} m
            WHERE m.symbol COLLATE utf8mb4_general_ci = p.symbol COLLATE utf8mb4_general_ci
              AND p.trade_date BETWEEN m.valid_from AND m.valid_to
        )
        """,
    )


def resolve_proxy_max_date(engine) -> date | None:
    return scalar_date(engine, f"SELECT MAX(trade_date) FROM {LOCAL_INDUSTRY_PROXY_TABLE}")


def resolve_missing_proxy_dates(engine, start_date: date, end_date: date) -> list[date]:
    sql = text(
        f"""
        WITH expected_dates AS (
            SELECT DISTINCT p.trade_date
            FROM cn_stock_daily_price p
            WHERE p.trade_date BETWEEN :start_date AND :end_date
              AND EXISTS (
                  SELECT 1
                  FROM {LOCAL_INDUSTRY_MAP_TABLE} m
                  WHERE m.symbol COLLATE utf8mb4_general_ci = p.symbol COLLATE utf8mb4_general_ci
                    AND p.trade_date BETWEEN m.valid_from AND m.valid_to
              )
        )
        SELECT e.trade_date
        FROM expected_dates e
        LEFT JOIN (
            SELECT DISTINCT trade_date
            FROM {LOCAL_INDUSTRY_PROXY_TABLE}
            WHERE trade_date BETWEEN :start_date AND :end_date
        ) p
          ON p.trade_date = e.trade_date
        WHERE p.trade_date IS NULL
        ORDER BY e.trade_date
        """
    )
    with engine.connect() as conn:
        rows = conn.execute(sql, {"start_date": start_date, "end_date": end_date}).fetchall()
    return [row[0] for row in rows]


def resolve_target_dates(engine, start_date: date, end_date: date) -> list[date]:
    sql = text(
        f"""
        SELECT DISTINCT p.trade_date
        FROM cn_stock_daily_price p
        WHERE p.trade_date BETWEEN :start_date AND :end_date
          AND EXISTS (
              SELECT 1
              FROM {LOCAL_INDUSTRY_MAP_TABLE} m
              WHERE m.symbol COLLATE utf8mb4_general_ci = p.symbol COLLATE utf8mb4_general_ci
                AND p.trade_date BETWEEN m.valid_from AND m.valid_to
          )
        ORDER BY p.trade_date
        """
    )
    with engine.connect() as conn:
        rows = conn.execute(sql, {"start_date": start_date, "end_date": end_date}).fetchall()
    return [row[0] for row in rows]


def rebuild_proxy_range(engine, logger: logging.Logger, start_date: date, end_date: date) -> int:
    logger.info("rebuilding proxy range start=%s end=%s", start_date, end_date)
    target_dates = resolve_target_dates(engine, start_date, end_date)
    if not target_dates:
        logger.warning("rebuild skipped: no target dates in range")
        return 0

    with engine.begin() as conn:
        deleted = conn.execute(
            text(
                f"""
                DELETE FROM {LOCAL_INDUSTRY_PROXY_TABLE}
                WHERE trade_date BETWEEN :start_date AND :end_date
                """
            ),
            {"start_date": start_date, "end_date": end_date},
        ).rowcount or 0
    logger.info("proxy rows deleted=%s", deleted)

    inserted = 0
    insert_sql = text(
        f"""
        INSERT INTO {LOCAL_INDUSTRY_PROXY_TABLE} (
            trade_date,
            industry_id,
            industry_name,
            industry_level,
            member_count,
            total_member_count,
            coverage_ratio,
            ret_eqw,
            ret_amt_w,
            amount_sum,
            up_count,
            down_count,
            up_ratio,
            source
        )
        WITH members AS (
            SELECT
                :trade_date AS trade_date,
                m.symbol,
                m.industry_id,
                COALESCE(m.industry_name, m.industry_id) AS industry_name
            FROM {LOCAL_INDUSTRY_MAP_TABLE} m
            WHERE :trade_date BETWEEN m.valid_from AND m.valid_to
        ),
        priced AS (
            SELECT
                p.symbol,
                p.amount,
                CASE
                    WHEN p.close IS NOT NULL
                     AND p.amount IS NOT NULL
                     AND (
                         (p.pre_close IS NOT NULL AND p.pre_close <> 0)
                         OR p.chg_pct IS NOT NULL
                     ) THEN 1
                    ELSE 0
                END AS is_effective,
                CASE
                    WHEN p.close IS NOT NULL AND p.pre_close IS NOT NULL AND p.pre_close <> 0 THEN p.close / p.pre_close - 1
                    WHEN p.close IS NOT NULL AND p.chg_pct IS NOT NULL THEN CASE WHEN ABS(p.chg_pct) > 1 THEN p.chg_pct / 100 ELSE p.chg_pct END
                    ELSE NULL
                END AS ret_value
            FROM cn_stock_daily_price p
            WHERE p.trade_date = :trade_date
        ),
        merged AS (
            SELECT
                m.trade_date,
                m.industry_id,
                m.industry_name,
                m.symbol,
                p.amount,
                p.ret_value,
                p.is_effective
            FROM members m
            LEFT JOIN priced p
              ON p.symbol COLLATE utf8mb4_general_ci = m.symbol COLLATE utf8mb4_general_ci
        ),
        total_stats AS (
            SELECT
                trade_date,
                industry_id,
                MAX(industry_name) AS industry_name,
                COUNT(DISTINCT symbol) AS total_member_count
            FROM merged
            GROUP BY trade_date, industry_id
        ),
        active_stats AS (
            SELECT
                trade_date,
                industry_id,
                COUNT(DISTINCT CASE WHEN is_effective = 1 THEN symbol END) AS member_count,
                SUM(CASE WHEN is_effective = 1 THEN COALESCE(amount, 0) ELSE 0 END) AS amount_sum,
                AVG(CASE WHEN is_effective = 1 THEN ret_value END) AS ret_eqw,
                CASE
                    WHEN SUM(CASE WHEN is_effective = 1 AND amount IS NOT NULL THEN amount ELSE 0 END) = 0 THEN NULL
                    ELSE SUM(CASE WHEN is_effective = 1 THEN ret_value * amount ELSE 0 END)
                       / SUM(CASE WHEN is_effective = 1 AND amount IS NOT NULL THEN amount ELSE 0 END)
                END AS ret_amt_w,
                SUM(CASE WHEN is_effective = 1 AND ret_value > 0 THEN 1 ELSE 0 END) AS up_count,
                SUM(CASE WHEN is_effective = 1 AND ret_value < 0 THEN 1 ELSE 0 END) AS down_count
            FROM merged
            GROUP BY trade_date, industry_id
        )
        SELECT
            t.trade_date,
            t.industry_id,
            t.industry_name,
            :industry_level AS industry_level,
            COALESCE(a.member_count, 0) AS member_count,
            t.total_member_count,
            CASE
                WHEN t.total_member_count = 0 THEN NULL
                ELSE COALESCE(a.member_count, 0) / t.total_member_count
            END AS coverage_ratio,
            a.ret_eqw,
            a.ret_amt_w,
            a.amount_sum,
            COALESCE(a.up_count, 0) AS up_count,
            COALESCE(a.down_count, 0) AS down_count,
            CASE
                WHEN COALESCE(a.member_count, 0) = 0 THEN NULL
                ELSE COALESCE(a.up_count, 0) / a.member_count
            END AS up_ratio,
            :proxy_source AS source
        FROM total_stats t
        LEFT JOIN active_stats a
          ON a.trade_date = t.trade_date
         AND a.industry_id COLLATE utf8mb4_general_ci = t.industry_id COLLATE utf8mb4_general_ci
        """
    )
    for trade_date in target_dates:
        with engine.begin() as conn:
            day_rows = conn.execute(
                insert_sql,
                {
                    "trade_date": trade_date,
                    "industry_level": LOCAL_INDUSTRY_LEVEL,
                    "proxy_source": LOCAL_PROXY_SOURCE,
                },
            ).rowcount or 0
        inserted += int(day_rows)
        logger.info("proxy day built trade_date=%s rows=%s", trade_date, day_rows)
    logger.info("proxy rows inserted=%s", inserted)
    recalculate_proxy_metrics(engine)
    logger.info("proxy close/rs fields recalculated")
    return inserted


def recalculate_proxy_metrics(engine) -> None:
    with engine.begin() as conn:
        conn.execute(text("DROP TEMPORARY TABLE IF EXISTS tmp_local_proxy_metrics"))
        conn.execute(
            text(
                f"""
                CREATE TEMPORARY TABLE tmp_local_proxy_close AS
                SELECT
                    trade_date,
                    industry_id,
                    CAST(
                        1000 * EXP(
                            SUM(LN(GREATEST(1 + COALESCE(ret_eqw, 0), 0.000001)))
                            OVER (PARTITION BY industry_id ORDER BY trade_date ROWS UNBOUNDED PRECEDING)
                        ) AS DECIMAL(18, 4)
                    ) AS proxy_close_eqw,
                    CAST(
                        1000 * EXP(
                            SUM(LN(GREATEST(1 + COALESCE(ret_amt_w, 0), 0.000001)))
                            OVER (PARTITION BY industry_id ORDER BY trade_date ROWS UNBOUNDED PRECEDING)
                        ) AS DECIMAL(18, 4)
                    ) AS proxy_close_amt_w
                FROM {LOCAL_INDUSTRY_PROXY_TABLE}
                """
            )
        )
        conn.execute(
            text(
                f"""
                CREATE TEMPORARY TABLE tmp_local_proxy_metrics AS
                SELECT
                    trade_date,
                    industry_id,
                    proxy_close_eqw,
                    proxy_close_amt_w,
                    CAST(
                        CASE
                            WHEN LAG(proxy_close_eqw, 20) OVER (PARTITION BY industry_id ORDER BY trade_date) IS NULL THEN NULL
                            ELSE proxy_close_eqw / LAG(proxy_close_eqw, 20) OVER (PARTITION BY industry_id ORDER BY trade_date) - 1
                        END AS DECIMAL(18, 8)
                    ) AS rs20_eqw,
                    CAST(
                        CASE
                            WHEN LAG(proxy_close_amt_w, 20) OVER (PARTITION BY industry_id ORDER BY trade_date) IS NULL THEN NULL
                            ELSE proxy_close_amt_w / LAG(proxy_close_amt_w, 20) OVER (PARTITION BY industry_id ORDER BY trade_date) - 1
                        END AS DECIMAL(18, 8)
                    ) AS rs20_amt_w,
                    CAST(
                        CASE
                            WHEN LAG(proxy_close_eqw, 60) OVER (PARTITION BY industry_id ORDER BY trade_date) IS NULL THEN NULL
                            ELSE proxy_close_eqw / LAG(proxy_close_eqw, 60) OVER (PARTITION BY industry_id ORDER BY trade_date) - 1
                        END AS DECIMAL(18, 8)
                    ) AS rs60_eqw,
                    CAST(
                        CASE
                            WHEN LAG(proxy_close_amt_w, 60) OVER (PARTITION BY industry_id ORDER BY trade_date) IS NULL THEN NULL
                            ELSE proxy_close_amt_w / LAG(proxy_close_amt_w, 60) OVER (PARTITION BY industry_id ORDER BY trade_date) - 1
                        END AS DECIMAL(18, 8)
                    ) AS rs60_amt_w
                FROM tmp_local_proxy_close
                """
            )
        )
        conn.execute(
            text(
                f"""
                UPDATE {LOCAL_INDUSTRY_PROXY_TABLE} t
                JOIN tmp_local_proxy_metrics x
                  ON x.trade_date = t.trade_date
                 AND x.industry_id COLLATE utf8mb4_general_ci = t.industry_id COLLATE utf8mb4_general_ci
                SET
                    t.proxy_close_eqw = x.proxy_close_eqw,
                    t.proxy_close_amt_w = x.proxy_close_amt_w,
                    t.rs20_eqw = x.rs20_eqw,
                    t.rs20_amt_w = x.rs20_amt_w,
                    t.rs60_eqw = x.rs60_eqw,
                    t.rs60_amt_w = x.rs60_amt_w
                """
            )
        )
        conn.execute(text("DROP TEMPORARY TABLE IF EXISTS tmp_local_proxy_close"))
        conn.execute(text("DROP TEMPORARY TABLE IF EXISTS tmp_local_proxy_metrics"))


def run_validate_map(engine, logger: logging.Logger) -> None:
    validate_local_map(engine, logger)
    update_task_status(engine, "local_industry_map_validate", "OK", "validation passed", None, resolve_sw_l1_source_max_date(engine))


def run_daily(engine, logger: logging.Logger, options: RunOptions) -> None:
    safe_end = options.end_date or resolve_proxy_safe_end_date(engine)
    proxy_max = resolve_proxy_max_date(engine)
    safe_min = resolve_proxy_min_date(engine)
    if safe_end is None or safe_min is None:
        logger.warning("daily refresh skipped: no safe source dates")
        update_task_status(engine, "cn_local_industry_proxy_daily", "WARN", "no safe source dates", None, None)
        return

    start_date = options.start_date or (proxy_max + timedelta(days=1) if proxy_max else safe_min)
    if start_date > safe_end:
        logger.info("daily refresh skipped: already up to date start=%s end=%s", start_date, safe_end)
        update_task_status(engine, "cn_local_industry_proxy_daily", "OK", "already up to date", proxy_max, safe_end)
        return

    inserted = rebuild_proxy_range(engine, logger, start_date, safe_end)
    update_task_status(engine, "cn_local_industry_proxy_daily", "OK", f"rebuilt rows={inserted}", safe_end, safe_end)


def run_weekly(engine, logger: logging.Logger, options: RunOptions) -> None:
    _, latest_map_date = refresh_local_map(engine, logger)
    safe_end = options.end_date or resolve_proxy_safe_end_date(engine)
    safe_min = resolve_proxy_min_date(engine)
    if safe_end is None or safe_min is None:
        logger.warning("weekly refresh skipped: no safe source dates")
        update_task_status(engine, "local_industry_proxy_weekly", "WARN", "no safe source dates", latest_map_date, latest_map_date)
        return

    start_candidate = safe_end - timedelta(days=options.weekly_lookback_days)
    start_date = options.start_date or max(start_candidate, safe_min)
    inserted = rebuild_proxy_range(engine, logger, start_date, safe_end)
    missing_dates = resolve_missing_proxy_dates(engine, safe_min, safe_end)
    if missing_dates:
        logger.info("weekly missing dates detected count=%s first=%s last=%s", len(missing_dates), missing_dates[0], missing_dates[-1])
        rebuild_proxy_range(engine, logger, missing_dates[0], missing_dates[-1])
    update_task_status(
        engine,
        "local_industry_proxy_weekly",
        "OK",
        f"weekly rows={inserted} missing_dates={len(missing_dates)}",
        safe_end,
        latest_map_date or safe_end,
    )


def run_backfill(engine, logger: logging.Logger, options: RunOptions) -> None:
    safe_min = resolve_proxy_min_date(engine)
    safe_end = options.end_date or resolve_proxy_safe_end_date(engine)
    start_date = options.start_date or safe_min
    if start_date is None or safe_end is None:
        raise RuntimeError("backfill requires valid source range")
    if start_date > safe_end:
        raise RuntimeError(f"backfill start {start_date} is greater than end {safe_end}")
    inserted = rebuild_proxy_range(engine, logger, start_date, safe_end)
    update_task_status(engine, "local_industry_proxy_backfill", "OK", f"backfill rows={inserted}", safe_end, safe_end)


def run_audit(engine, logger: logging.Logger, options: RunOptions) -> None:
    safe_min = options.start_date or resolve_proxy_min_date(engine)
    safe_end = options.end_date or resolve_proxy_safe_end_date(engine)
    if safe_min is None or safe_end is None:
        logger.warning("audit skipped: no comparable range")
        update_task_status(engine, "local_industry_proxy_audit", "WARN", "no comparable range", None, None)
        return

    missing_dates = resolve_missing_proxy_dates(engine, safe_min, safe_end)
    with engine.connect() as conn:
        latest_stock = conn.execute(text("SELECT MAX(trade_date) FROM cn_stock_daily_price")).scalar()
        latest_proxy = conn.execute(text(f"SELECT MAX(trade_date) FROM {LOCAL_INDUSTRY_PROXY_TABLE}")).scalar()
        low_coverage = conn.execute(
            text(
                f"""
                SELECT COUNT(*)
                FROM {LOCAL_INDUSTRY_PROXY_TABLE}
                WHERE trade_date BETWEEN :start_date AND :end_date
                  AND coverage_ratio IS NOT NULL
                  AND coverage_ratio < 0.6
                """
            ),
            {"start_date": safe_min, "end_date": safe_end},
        ).scalar() or 0
        extreme_ret = conn.execute(
            text(
                f"""
                SELECT COUNT(*)
                FROM {LOCAL_INDUSTRY_PROXY_TABLE}
                WHERE trade_date BETWEEN :start_date AND :end_date
                  AND (
                      ABS(COALESCE(ret_eqw, 0)) > 0.2
                      OR ABS(COALESCE(ret_amt_w, 0)) > 0.2
                  )
                """
            ),
            {"start_date": safe_min, "end_date": safe_end},
        ).scalar() or 0
        low_member = conn.execute(
            text(
                f"""
                SELECT COUNT(*)
                FROM {LOCAL_INDUSTRY_PROXY_TABLE}
                WHERE trade_date BETWEEN :start_date AND :end_date
                  AND member_count < 5
                """
            ),
            {"start_date": safe_min, "end_date": safe_end},
        ).scalar() or 0
        partial_rows = conn.execute(
            text(
                f"""
                WITH expected AS (
                    SELECT
                        p.trade_date,
                        COUNT(DISTINCT m.industry_id) AS expected_cnt
                    FROM cn_stock_daily_price p
                    JOIN {LOCAL_INDUSTRY_MAP_TABLE} m
                      ON m.symbol COLLATE utf8mb4_general_ci = p.symbol COLLATE utf8mb4_general_ci
                     AND p.trade_date BETWEEN m.valid_from AND m.valid_to
                    WHERE p.trade_date BETWEEN :start_date AND :end_date
                    GROUP BY p.trade_date
                ),
                actual AS (
                    SELECT trade_date, COUNT(*) AS actual_cnt
                    FROM {LOCAL_INDUSTRY_PROXY_TABLE}
                    WHERE trade_date BETWEEN :start_date AND :end_date
                    GROUP BY trade_date
                )
                SELECT e.trade_date, e.expected_cnt, COALESCE(a.actual_cnt, 0) AS actual_cnt
                FROM expected e
                LEFT JOIN actual a
                  ON a.trade_date = e.trade_date
                WHERE COALESCE(a.actual_cnt, 0) <> e.expected_cnt
                ORDER BY e.trade_date
                LIMIT 20
                """
            ),
            {"start_date": safe_min, "end_date": safe_end},
        ).fetchall()

    logger.info(
        "audit summary stock_max=%s proxy_max=%s missing_dates=%s partial_dates=%s low_coverage_rows=%s low_member_rows=%s extreme_ret_rows=%s",
        latest_stock,
        latest_proxy,
        len(missing_dates),
        len(partial_rows),
        low_coverage,
        low_member,
        extreme_ret,
    )
    for row in partial_rows:
        logger.warning("audit partial trade_date=%s expected=%s actual=%s", row[0], row[1], row[2])
    if missing_dates:
        logger.warning("audit missing dates first=%s last=%s count=%s", missing_dates[0], missing_dates[-1], len(missing_dates))
    update_task_status(
        engine,
        "local_industry_proxy_audit",
        "OK" if not missing_dates and not partial_rows else "WARN",
        f"missing_dates={len(missing_dates)} partial_dates={len(partial_rows)} low_coverage={low_coverage} low_member={low_member} extreme_ret={extreme_ret}",
        latest_proxy,
        latest_stock,
    )


def main() -> int:
    logger = setup_logger()
    try:
        options = parse_args()
    except ValueError as exc:
        logger.error("invalid arguments: %s", exc)
        return 1

    engine = build_engine()
    ensure_schema(engine, logger)
    logger.info("job_started mode=%s start=%s end=%s weekly_lookback_days=%s", options.mode, options.start_date, options.end_date, options.weekly_lookback_days)
    try:
        if options.mode == "refresh-map":
            refresh_local_map(engine, logger)
        elif options.mode == "validate-map":
            run_validate_map(engine, logger)
        elif options.mode == "daily":
            run_daily(engine, logger, options)
        elif options.mode == "weekly":
            run_weekly(engine, logger, options)
        elif options.mode == "backfill":
            run_backfill(engine, logger, options)
        elif options.mode == "audit":
            run_audit(engine, logger, options)
        elif options.mode == "all":
            refresh_local_map(engine, logger)
            run_validate_map(engine, logger)
            run_daily(engine, logger, options)
            run_audit(engine, logger, options)
    except Exception as exc:  # noqa: BLE001
        logger.exception("job_failed mode=%s error=%s", options.mode, exc)
        update_task_status(engine, f"local_industry_proxy_{options.mode}", "ERROR", str(exc), None, resolve_sw_l1_source_max_date(engine))
        return 1
    logger.info("job_finished mode=%s", options.mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
