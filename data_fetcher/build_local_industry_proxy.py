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
        LOCAL_INDUSTRY_PROXY_LOG_FILE,
        LOCAL_INDUSTRY_PROXY_TABLE,
        LOCAL_MAP_MASTER_SOURCE,
        LOCAL_MAP_MEMBER_SOURCE,
        LOCAL_PROXY_SOURCE,
        LOCAL_TASK_STATUS_TABLE,
    )
else:
    from app.settings import build_engine
    from .config import (
        LOCAL_INDUSTRY_LEVEL,
        LOCAL_INDUSTRY_MAP_TABLE,
        LOCAL_INDUSTRY_PROXY_LOG_FILE,
        LOCAL_INDUSTRY_PROXY_TABLE,
        LOCAL_MAP_MASTER_SOURCE,
        LOCAL_MAP_MEMBER_SOURCE,
        LOCAL_PROXY_SOURCE,
        LOCAL_TASK_STATUS_TABLE,
    )


DATE_FMT = "%Y-%m-%d"


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
    parser = argparse.ArgumentParser(description="Build local industry mapping and proxy index from local DB tables.")
    parser.add_argument(
        "--mode",
        choices=["refresh-map", "daily", "weekly", "backfill", "audit", "all"],
        default="daily",
        help="refresh-map=only rebuild local mapping; daily=incremental proxy refresh; weekly=refresh map + recent rebuild + missing repair; backfill=rebuild explicit range; audit=coverage checks; all=refresh-map + daily + audit",
    )
    parser.add_argument("--start", default="", help="start date in YYYYMMDD or YYYY-MM-DD")
    parser.add_argument("--end", default="", help="end date in YYYYMMDD or YYYY-MM-DD")
    parser.add_argument("--weekly-lookback-days", type=int, default=35, help="weekly rebuild lookback days after map refresh")
    args = parser.parse_args()

    start_date = parse_date(args.start)
    end_date = parse_date(args.end)
    if args.weekly_lookback_days < 1:
        raise ValueError("weekly-lookback-days must be >= 1")
    if start_date and end_date and start_date > end_date:
        raise ValueError("start date cannot be greater than end date")
    return RunOptions(
        mode=args.mode,
        start_date=start_date,
        end_date=end_date,
        weekly_lookback_days=args.weekly_lookback_days,
    )


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
            valid_to DATE DEFAULT NULL,
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
            amount_sum DECIMAL(24,4) DEFAULT NULL,
            ret_eqw DECIMAL(18,8) DEFAULT NULL,
            ret_amt_w DECIMAL(18,8) DEFAULT NULL,
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


def refresh_local_map(engine, logger: logging.Logger) -> tuple[int, date | None]:
    logger.info("refreshing local industry map from cn_board_industry_member_hist source=%s", LOCAL_MAP_MEMBER_SOURCE)
    latest_source_date = scalar_date(
        engine,
        f"SELECT MAX(asof_date) FROM cn_board_industry_master WHERE source = :src",
        {"src": LOCAL_MAP_MASTER_SOURCE},
    )
    with engine.begin() as conn:
        deleted = conn.execute(
            text(
                f"""
                DELETE FROM {LOCAL_INDUSTRY_MAP_TABLE}
                WHERE is_manual_override = 0
                  AND source = :member_source
                """
            ),
            {"member_source": LOCAL_MAP_MEMBER_SOURCE},
        ).rowcount or 0
        logger.info("local map cleared rows=%s", deleted)
        inserted = conn.execute(
            text(
                f"""
                INSERT INTO {LOCAL_INDUSTRY_MAP_TABLE} (
                    symbol, industry_id, industry_name, industry_level, valid_from, valid_to, source, is_manual_override
                )
                WITH latest_name AS (
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
                )
                SELECT
                    h.symbol,
                    h.board_id AS industry_id,
                    n.board_name AS industry_name,
                    :industry_level AS industry_level,
                    h.valid_from,
                    h.valid_to,
                    h.source,
                    0 AS is_manual_override
                FROM cn_board_industry_member_hist h
                LEFT JOIN latest_name n
                  ON n.board_id COLLATE utf8mb4_general_ci = h.board_id COLLATE utf8mb4_general_ci
                WHERE h.source = :member_source
                """
            ),
            {
                "member_source": LOCAL_MAP_MEMBER_SOURCE,
                "master_source": LOCAL_MAP_MASTER_SOURCE,
                "industry_level": LOCAL_INDUSTRY_LEVEL,
            },
        ).rowcount or 0
    logger.info("local map refreshed rows=%s latest_master_asof=%s", inserted, latest_source_date)
    update_task_status(
        engine,
        task_name="local_industry_map_refresh",
        status="OK",
        message=f"refreshed rows={inserted}",
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
              AND p.trade_date >= m.valid_from
              AND p.trade_date <= COALESCE(m.valid_to, DATE('9999-12-31'))
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
              AND p.trade_date >= m.valid_from
              AND p.trade_date <= COALESCE(m.valid_to, DATE('9999-12-31'))
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
                    AND p.trade_date >= m.valid_from
                    AND p.trade_date <= COALESCE(m.valid_to, DATE('9999-12-31'))
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


def rebuild_proxy_range(engine, logger: logging.Logger, start_date: date, end_date: date) -> int:
    logger.info("rebuilding proxy range start=%s end=%s", start_date, end_date)
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
        inserted = conn.execute(
            text(
                f"""
                INSERT INTO {LOCAL_INDUSTRY_PROXY_TABLE} (
                    trade_date,
                    industry_id,
                    industry_name,
                    industry_level,
                    member_count,
                    amount_sum,
                    ret_eqw,
                    ret_amt_w,
                    source
                )
                WITH priced AS (
                    SELECT
                        p.trade_date,
                        p.symbol,
                        p.amount,
                        CASE
                            WHEN p.pre_close IS NOT NULL AND p.pre_close <> 0 THEN p.close / p.pre_close - 1
                            WHEN p.chg_pct IS NOT NULL THEN CASE WHEN ABS(p.chg_pct) > 1 THEN p.chg_pct / 100 ELSE p.chg_pct END
                            ELSE NULL
                        END AS ret_value
                    FROM cn_stock_daily_price p
                    WHERE p.trade_date BETWEEN :start_date AND :end_date
                )
                SELECT
                    pr.trade_date,
                    m.industry_id,
                    COALESCE(MAX(m.industry_name), m.industry_id) AS industry_name,
                    :industry_level AS industry_level,
                    COUNT(DISTINCT pr.symbol) AS member_count,
                    SUM(COALESCE(pr.amount, 0)) AS amount_sum,
                    AVG(pr.ret_value) AS ret_eqw,
                    CASE
                        WHEN SUM(CASE WHEN pr.ret_value IS NOT NULL AND pr.amount IS NOT NULL THEN pr.amount ELSE 0 END) = 0 THEN NULL
                        ELSE SUM(pr.ret_value * pr.amount) / SUM(CASE WHEN pr.ret_value IS NOT NULL AND pr.amount IS NOT NULL THEN pr.amount ELSE 0 END)
                    END AS ret_amt_w,
                    :source AS source
                FROM priced pr
                JOIN {LOCAL_INDUSTRY_MAP_TABLE} m
                  ON m.symbol COLLATE utf8mb4_general_ci = pr.symbol COLLATE utf8mb4_general_ci
                 AND pr.trade_date >= m.valid_from
                 AND pr.trade_date <= COALESCE(m.valid_to, DATE('9999-12-31'))
                GROUP BY pr.trade_date, m.industry_id
                """
            ),
            {
                "start_date": start_date,
                "end_date": end_date,
                "industry_level": LOCAL_INDUSTRY_LEVEL,
                "source": LOCAL_PROXY_SOURCE,
            },
        ).rowcount or 0
        logger.info("proxy rows inserted=%s", inserted)
        conn.execute(text("DROP TEMPORARY TABLE IF EXISTS tmp_local_proxy_close"))
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
                UPDATE {LOCAL_INDUSTRY_PROXY_TABLE} t
                JOIN tmp_local_proxy_close x
                  ON x.trade_date = t.trade_date
                 AND x.industry_id COLLATE utf8mb4_general_ci = t.industry_id COLLATE utf8mb4_general_ci
                SET
                    t.proxy_close_eqw = x.proxy_close_eqw,
                    t.proxy_close_amt_w = x.proxy_close_amt_w
                """
            )
        )
        conn.execute(text("DROP TEMPORARY TABLE IF EXISTS tmp_local_proxy_close"))
    logger.info("proxy close series recalculated")
    return inserted


def run_daily(engine, logger: logging.Logger, options: RunOptions) -> None:
    safe_end = options.end_date or resolve_proxy_safe_end_date(engine)
    proxy_max = resolve_proxy_max_date(engine)
    safe_min = resolve_proxy_min_date(engine)
    if safe_end is None or safe_min is None:
        logger.warning("daily refresh skipped: no safe source dates found")
        update_task_status(engine, "local_industry_proxy_daily", "WARN", "no safe source dates", None, None)
        return

    start_date = options.start_date or (proxy_max + timedelta(days=1) if proxy_max else safe_min)
    if start_date > safe_end:
        logger.info("daily refresh skipped: proxy already up to date start=%s end=%s", start_date, safe_end)
        update_task_status(engine, "local_industry_proxy_daily", "OK", "already up to date", proxy_max, safe_end)
        return

    inserted = rebuild_proxy_range(engine, logger, start_date, safe_end)
    update_task_status(
        engine,
        "local_industry_proxy_daily",
        "OK",
        f"rebuilt range {start_date} to {safe_end}, rows={inserted}",
        safe_end,
        safe_end,
    )


def run_weekly(engine, logger: logging.Logger, options: RunOptions) -> None:
    _, latest_map_date = refresh_local_map(engine, logger)
    safe_end = options.end_date or resolve_proxy_safe_end_date(engine)
    if safe_end is None:
        logger.warning("weekly refresh skipped: no safe end date after map refresh")
        update_task_status(engine, "local_industry_proxy_weekly", "WARN", "no safe end date", latest_map_date, latest_map_date)
        return

    start_candidate = safe_end - timedelta(days=options.weekly_lookback_days)
    start_date = options.start_date or max(start_candidate, resolve_proxy_min_date(engine) or start_candidate)
    inserted = rebuild_proxy_range(engine, logger, start_date, safe_end)
    missing_dates = resolve_missing_proxy_dates(engine, resolve_proxy_min_date(engine) or start_date, safe_end)
    if missing_dates:
        logger.info("weekly detected missing proxy dates count=%s first=%s last=%s", len(missing_dates), missing_dates[0], missing_dates[-1])
        rebuild_proxy_range(engine, logger, missing_dates[0], missing_dates[-1])
    update_task_status(
        engine,
        "local_industry_proxy_weekly",
        "OK",
        f"weekly refresh done rows={inserted} missing_dates={len(missing_dates)}",
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
    update_task_status(
        engine,
        "local_industry_proxy_backfill",
        "OK",
        f"backfill rebuilt rows={inserted}",
        safe_end,
        safe_end,
    )


def run_audit(engine, logger: logging.Logger, options: RunOptions) -> None:
    safe_min = options.start_date or resolve_proxy_min_date(engine)
    safe_end = options.end_date or resolve_proxy_safe_end_date(engine)
    if safe_min is None or safe_end is None:
        logger.warning("audit skipped: no comparable source range")
        update_task_status(engine, "local_industry_proxy_audit", "WARN", "no comparable source range", None, None)
        return

    missing_dates = resolve_missing_proxy_dates(engine, safe_min, safe_end)
    with engine.connect() as conn:
        latest_stock = conn.execute(text("SELECT MAX(trade_date) FROM cn_stock_daily_price")).scalar()
        latest_proxy = conn.execute(text(f"SELECT MAX(trade_date) FROM {LOCAL_INDUSTRY_PROXY_TABLE}")).scalar()
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
                     AND p.trade_date >= m.valid_from
                     AND p.trade_date <= COALESCE(m.valid_to, DATE('9999-12-31'))
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
        "audit summary stock_max=%s proxy_max=%s missing_date_count=%s partial_date_count=%s",
        latest_stock,
        latest_proxy,
        len(missing_dates),
        len(partial_rows),
    )
    if missing_dates:
        logger.warning("audit missing dates first=%s last=%s count=%s", missing_dates[0], missing_dates[-1], len(missing_dates))
    for row in partial_rows:
        logger.warning("audit partial trade_date=%s expected=%s actual=%s", row[0], row[1], row[2])
    update_task_status(
        engine,
        "local_industry_proxy_audit",
        "OK" if not missing_dates and not partial_rows else "WARN",
        f"missing_dates={len(missing_dates)} partial_dates={len(partial_rows)}",
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
            run_daily(engine, logger, options)
            run_audit(engine, logger, options)
    except Exception as exc:  # noqa: BLE001
        logger.exception("job_failed mode=%s error=%s", options.mode, exc)
        update_task_status(engine, f"local_industry_proxy_{options.mode}", "ERROR", str(exc), None, None)
        return 1

    logger.info("job_finished mode=%s", options.mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
