from __future__ import annotations

import argparse
import time
from datetime import date

from sqlalchemy import text

from data_pipeline.common.cli import add_shared_args, month_chunks, quarter_periods, resolve_date_range
from data_pipeline.common.db import apply_sql_file, get_engine
from data_pipeline.common.logging_utils import build_logger
from data_pipeline.common.state import BackfillState


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Populate cn_stock_fundamental_daily from existing cn_stock_* quarterly tables."
    )
    add_shared_args(parser)
    parser.add_argument("--replace", action="store_true", help="Delete existing daily rows before inserting")
    return parser


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _delete_daily_chunk_if_replace(engine, chunk_start: date, chunk_end: date, replace: bool, logger) -> int:
    if not replace:
        return 0
    sql = """
        DELETE FROM cn_stock_fundamental_daily
        WHERE trade_date BETWEEN :chunk_start AND :chunk_end
    """
    with engine.begin() as conn:
        result = conn.execute(text(sql), {"chunk_start": chunk_start, "chunk_end": chunk_end})
    deleted = int(result.rowcount or 0)
    logger.info("daily_chunk_replace_delete start=%s end=%s deleted=%s", chunk_start, chunk_end, deleted)
    return deleted


# ---------------------------------------------------------------------------
# SQL: populate cn_local_stock_*_q from existing cn_stock_* tables
# ---------------------------------------------------------------------------

POPULATE_INCOME_SQL = """
    INSERT INTO cn_local_stock_income_q
        (symbol, end_date, ann_date, f_ann_date, report_type,
         total_revenue, revenue, n_income_attr_p, source)
    SELECT
        i.symbol,
        i.end_date,
        i.ann_date,
        i.f_ann_date,
        i.report_type,
        i.total_revenue,
        i.revenue,
        i.n_income_attr_p,
        'cn_stock_income' AS source
    FROM cn_stock_income i
    WHERE i.end_date BETWEEN :lookback_start AND :chunk_end
      AND i.ann_date IS NOT NULL
    ON DUPLICATE KEY UPDATE
        ann_date = VALUES(ann_date),
        f_ann_date = VALUES(f_ann_date),
        report_type = VALUES(report_type),
        total_revenue = VALUES(total_revenue),
        revenue = VALUES(revenue),
        n_income_attr_p = VALUES(n_income_attr_p),
        source = VALUES(source),
        updated_at = CURRENT_TIMESTAMP
"""

POPULATE_BALANCE_SQL = """
    INSERT INTO cn_local_stock_balancesheet_q
        (symbol, end_date, ann_date, f_ann_date, report_type,
         inventory, contract_liability, fixed_assets, total_assets, total_liab, source)
    SELECT
        b.symbol,
        b.end_date,
        b.ann_date,
        b.f_ann_date,
        b.report_type,
        b.inventories       AS inventory,
        NULL                AS contract_liability,  -- not available in cn_stock_balancesheet
        b.fix_assets,
        b.total_assets,
        b.total_liab,
        'cn_stock_balancesheet' AS source
    FROM cn_stock_balancesheet b
    WHERE b.end_date BETWEEN :lookback_start AND :chunk_end
      AND b.ann_date IS NOT NULL
    ON DUPLICATE KEY UPDATE
        ann_date = VALUES(ann_date),
        f_ann_date = VALUES(f_ann_date),
        report_type = VALUES(report_type),
        inventory = VALUES(inventory),
        fixed_assets = VALUES(fixed_assets),
        total_assets = VALUES(total_assets),
        total_liab = VALUES(total_liab),
        source = VALUES(source),
        updated_at = CURRENT_TIMESTAMP
"""

POPULATE_FINA_SQL = """
    INSERT INTO cn_local_stock_fina_indicator_q
        (symbol, end_date, ann_date, report_type,
         revenue_yoy, profit_yoy, roe, gross_margin, debt_to_assets, ocfps, source)
    SELECT
        f.symbol,
        f.end_date,
        f.ann_date,
        f.report_type,
        COALESCE(f.or_yoy, f.tr_yoy, f.q_sales_yoy) AS revenue_yoy,
        COALESCE(f.netprofit_yoy, f.q_profit_yoy)    AS profit_yoy,
        f.roe,
        f.grossprofit_margin  AS gross_margin,
        f.debt_to_assets,
        f.ocfps,
        'cn_stock_fina_indicator' AS source
    FROM cn_stock_fina_indicator f
    WHERE f.end_date BETWEEN :lookback_start AND :chunk_end
      AND f.ann_date IS NOT NULL
    ON DUPLICATE KEY UPDATE
        ann_date = VALUES(ann_date),
        report_type = VALUES(report_type),
        revenue_yoy = VALUES(revenue_yoy),
        profit_yoy = VALUES(profit_yoy),
        roe = VALUES(roe),
        gross_margin = VALUES(gross_margin),
        debt_to_assets = VALUES(debt_to_assets),
        ocfps = VALUES(ocfps),
        source = VALUES(source),
        updated_at = CURRENT_TIMESTAMP
"""

# ---------------------------------------------------------------------------
# SQL: materialize daily rows from cn_local_stock_*_q
# ---------------------------------------------------------------------------

MATERIALIZE_PREP_SQLS = [
    (
        "tmp_price_universe",
        """
        CREATE TEMPORARY TABLE tmp_fund_price_universe AS
        SELECT DISTINCT
            p.SYMBOL COLLATE utf8mb4_unicode_ci AS symbol
        FROM cn_stock_daily_price p
        WHERE p.TRADE_DATE BETWEEN :chunk_start AND :chunk_end
        """,
    ),
    (
        "tmp_report_keys",
        """
        CREATE TEMPORARY TABLE tmp_fund_report_keys (
            symbol VARCHAR(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL,
            end_date DATE NOT NULL,
            PRIMARY KEY (symbol, end_date)
        ) ENGINE=MEMORY
        """,
    ),
    (
        "tmp_merged",
        """
        CREATE TEMPORARY TABLE tmp_fund_merged AS
        SELECT
            rk.symbol,
            rk.end_date,
            COALESCE(fi.ann_date, bs.f_ann_date, bs.ann_date, inc.f_ann_date, inc.ann_date) AS ann_date,
            fi.revenue_yoy,
            fi.profit_yoy,
            fi.roe,
            fi.gross_margin,
            fi.debt_to_assets,
            fi.ocfps,
            bs.inventory,
            bs.contract_liability,
            bs.fixed_assets
        FROM tmp_fund_report_keys rk
        LEFT JOIN cn_local_stock_fina_indicator_q fi
          ON fi.symbol COLLATE utf8mb4_unicode_ci = rk.symbol
         AND fi.end_date = rk.end_date
        LEFT JOIN cn_local_stock_balancesheet_q bs
          ON bs.symbol COLLATE utf8mb4_unicode_ci = rk.symbol
         AND bs.end_date = rk.end_date
        LEFT JOIN cn_local_stock_income_q inc
          ON inc.symbol COLLATE utf8mb4_unicode_ci = rk.symbol
         AND inc.end_date = rk.end_date
        WHERE COALESCE(fi.ann_date, bs.f_ann_date, bs.ann_date, inc.f_ann_date, inc.ann_date) IS NOT NULL
        """,
    ),
    (
        "tmp_effective",
        """
        CREATE TEMPORARY TABLE tmp_fund_effective AS
        SELECT
            m.*,
            LEAD(m.ann_date) OVER (
                PARTITION BY m.symbol
                ORDER BY m.ann_date, m.end_date
            ) AS next_ann_date
        FROM tmp_fund_merged m
        """,
    ),
]

MATERIALIZE_INDEX_SQLS = [
    "ALTER TABLE tmp_fund_price_universe ADD PRIMARY KEY (symbol)",
    None,
    "ALTER TABLE tmp_fund_merged ADD INDEX idx_tmp_fund_merged_symbol_ann (symbol, ann_date, end_date)",
    "ALTER TABLE tmp_fund_effective ADD INDEX idx_tmp_fund_effective_symbol_ann_next (symbol, ann_date, next_ann_date)",
]

MATERIALIZE_REPORT_KEY_INSERT_SQLS = [
    """
    INSERT IGNORE INTO tmp_fund_report_keys (symbol, end_date)
    SELECT inc.symbol, inc.end_date
    FROM cn_local_stock_income_q inc
    JOIN tmp_fund_price_universe pu ON pu.symbol = inc.symbol COLLATE utf8mb4_unicode_ci
    WHERE COALESCE(inc.f_ann_date, inc.ann_date) <= :chunk_end
    """,
    """
    INSERT IGNORE INTO tmp_fund_report_keys (symbol, end_date)
    SELECT bs.symbol, bs.end_date
    FROM cn_local_stock_balancesheet_q bs
    JOIN tmp_fund_price_universe pu ON pu.symbol = bs.symbol COLLATE utf8mb4_unicode_ci
    WHERE COALESCE(bs.f_ann_date, bs.ann_date) <= :chunk_end
    """,
    """
    INSERT IGNORE INTO tmp_fund_report_keys (symbol, end_date)
    SELECT fi.symbol, fi.end_date
    FROM cn_local_stock_fina_indicator_q fi
    JOIN tmp_fund_price_universe pu ON pu.symbol = fi.symbol COLLATE utf8mb4_unicode_ci
    WHERE fi.ann_date <= :chunk_end
    """,
]

MATERIALIZE_INSERT_SQL = """
    INSERT INTO cn_stock_fundamental_daily (
        symbol, trade_date, report_end_date, ann_date, revenue_yoy, profit_yoy, roe,
        gross_margin, debt_to_assets, ocfps, inventory, contract_liability, fixed_assets, source
    )
    SELECT
        p.SYMBOL,
        p.TRADE_DATE,
        e.end_date AS report_end_date,
        e.ann_date,
        e.revenue_yoy,
        e.profit_yoy,
        e.roe,
        e.gross_margin,
        e.debt_to_assets,
        e.ocfps,
        e.inventory,
        e.contract_liability,
        e.fixed_assets,
        'quarterly_snapshot' AS source
    FROM cn_stock_daily_price p
    JOIN tmp_fund_effective e
      ON e.symbol = p.SYMBOL COLLATE utf8mb4_unicode_ci
     AND p.TRADE_DATE >= e.ann_date
     AND p.TRADE_DATE <= COALESCE(DATE_SUB(e.next_ann_date, INTERVAL 1 DAY), :chunk_end)
    WHERE p.TRADE_DATE BETWEEN :chunk_start AND :chunk_end
    ON DUPLICATE KEY UPDATE
        report_end_date = VALUES(report_end_date),
        ann_date = VALUES(ann_date),
        revenue_yoy = VALUES(revenue_yoy),
        profit_yoy = VALUES(profit_yoy),
        roe = VALUES(roe),
        gross_margin = VALUES(gross_margin),
        debt_to_assets = VALUES(debt_to_assets),
        ocfps = VALUES(ocfps),
        inventory = VALUES(inventory),
        contract_liability = VALUES(contract_liability),
        fixed_assets = VALUES(fixed_assets),
        source = VALUES(source),
        updated_at = CURRENT_TIMESTAMP
"""

MATERIALIZE_TEMP_TABLES = [
    "tmp_fund_effective",
    "tmp_fund_merged",
    "tmp_fund_report_keys",
    "tmp_fund_price_universe",
]

CHUNK_UNIVERSE_COUNT_SQL = """
    SELECT
        COUNT(*) AS price_rows,
        COUNT(DISTINCT SYMBOL) AS symbol_count,
        COUNT(DISTINCT TRADE_DATE) AS trade_date_count
    FROM cn_stock_daily_price
    WHERE TRADE_DATE BETWEEN :chunk_start AND :chunk_end
"""




def _count_temp_table(conn, table_name: str) -> int:
    return int(conn.execute(text(f"SELECT COUNT(*) FROM {table_name}")).scalar() or 0)


def _materialize_daily_chunk(engine, chunk_start: date, chunk_end: date, logger) -> int:
    params = {"chunk_start": chunk_start, "chunk_end": chunk_end}
    with engine.begin() as conn:
        for table_name in MATERIALIZE_TEMP_TABLES:
            conn.execute(text(f"DROP TEMPORARY TABLE IF EXISTS {table_name}"))

        for stage_idx, (stage_name, stage_sql) in enumerate(MATERIALIZE_PREP_SQLS):
            stage_t0 = time.perf_counter()
            logger.info("daily_materialize_stage_start stage=%s start=%s end=%s", stage_name, chunk_start, chunk_end)
            conn.execute(text(stage_sql), params)
            if stage_name == "tmp_report_keys":
                insert_t0 = time.perf_counter()
                for insert_idx, insert_sql in enumerate(MATERIALIZE_REPORT_KEY_INSERT_SQLS, 1):
                    conn.execute(text(insert_sql), params)
                    logger.info(
                        "daily_materialize_report_keys_insert_done part=%s/3 seconds=%.3f",
                        insert_idx,
                        time.perf_counter() - insert_t0,
                    )

            rows = _count_temp_table(conn, f"tmp_fund_{stage_name.split('tmp_')[-1]}")
            logger.info(
                "daily_materialize_stage_done stage=%s rows=%s seconds=%.3f",
                stage_name,
                rows,
                time.perf_counter() - stage_t0,
            )

            index_sql = MATERIALIZE_INDEX_SQLS[stage_idx]
            if index_sql:
                index_t0 = time.perf_counter()
                try:
                    conn.execute(text(index_sql))
                    logger.info(
                        "daily_materialize_index_done stage=%s seconds=%.3f",
                        stage_name,
                        time.perf_counter() - index_t0,
                    )
                except Exception as exc:
                    logger.warning("daily_materialize_index_skip stage=%s reason=%s", stage_name, exc)

        insert_t0 = time.perf_counter()
        logger.info("daily_chunk_insert_start mode=temp_materialized start=%s end=%s", chunk_start, chunk_end)
        result = conn.execute(text(MATERIALIZE_INSERT_SQL), params)
        affected = int(result.rowcount or 0)
        logger.info(
            "daily_chunk_insert_done mode=temp_materialized affected=%s seconds=%.3f",
            affected,
            time.perf_counter() - insert_t0,
        )

        for table_name in MATERIALIZE_TEMP_TABLES:
            conn.execute(text(f"DROP TEMPORARY TABLE IF EXISTS {table_name}"))

    return affected


# ---------------------------------------------------------------------------
# Main build
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    date_range = resolve_date_range(args.start, args.end)
    logger = build_logger("build_cn_stock_fundamental_daily")
    engine = get_engine()
    apply_sql_file(engine, "docs/DDL/ga_mainline_data_backfill_system.sql")
    daily_state = BackfillState(engine=engine, job_name="build_cn_stock_fundamental_daily")
    resume = bool(getattr(args, "resume", False))
    replace = bool(getattr(args, "replace", False))

    # Look back 2 years for quarterly data (to cover the ann_date look-ahead rule)
    lookback_start = date(date_range.start.year - 2, 1, 1)
    periods = sorted(quarter_periods(date_range.start, date_range.end, lookback_years=2))

    logger.info(
        "build_cn_stock_fundamental_daily_start start=%s end=%s lookback_start=%s periods=%s resume=%s replace=%s",
        date_range.start,
        date_range.end,
        lookback_start,
        len(periods),
        resume,
        replace,
    )

    # Step 1: Populate cn_local_stock_*_q from existing cn_stock_* tables
    # These use INSERT ... ON DUPLICATE KEY UPDATE so they are idempotent.
    populate_tasks = [
        ("cn_local_stock_income_q", POPULATE_INCOME_SQL),
        ("cn_local_stock_balancesheet_q", POPULATE_BALANCE_SQL),
        ("cn_local_stock_fina_indicator_q", POPULATE_FINA_SQL),
    ]
    total_populated = 0
    for table_name, sql in populate_tasks:
        with engine.begin() as conn:
            result = conn.execute(
                text(sql),
                {"lookback_start": lookback_start, "chunk_end": date_range.end},
            )
        affected = int(result.rowcount or 0)
        total_populated += affected
        logger.info("populate_%s affected=%s", table_name, affected)

    # Verify we have data
    for table_name, _ in populate_tasks:
        with engine.connect() as conn:
            cnt = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}")).scalar()
        logger.info("verify_%s rows=%s", table_name, cnt)

    # Step 2: Materialize daily rows
    total_daily = 0
    for chunk_start, chunk_end in month_chunks(date_range.start, date_range.end, max(1, int(args.chunk_months))):
        chunk_key = f"daily:{chunk_start}:{chunk_end}"
        if resume and daily_state.is_completed(chunk_key):
            logger.info("resume_skip chunk=%s", chunk_key)
            continue
        daily_state.start(chunk_key, chunk_start, chunk_end)
        try:
            chunk_t0 = time.perf_counter()
            with engine.connect() as conn:
                stats = conn.execute(
                    text(CHUNK_UNIVERSE_COUNT_SQL),
                    {"chunk_start": chunk_start, "chunk_end": chunk_end},
                ).mappings().one()
            logger.info(
                "daily_chunk_universe chunk=%s price_rows=%s symbols=%s trade_dates=%s",
                chunk_key,
                int(stats["price_rows"] or 0),
                int(stats["symbol_count"] or 0),
                int(stats["trade_date_count"] or 0),
            )

            delete_t0 = time.perf_counter()
            _delete_daily_chunk_if_replace(engine, chunk_start, chunk_end, replace, logger)
            logger.info("daily_chunk_delete_elapsed chunk=%s seconds=%.3f", chunk_key, time.perf_counter() - delete_t0)

            insert_t0 = time.perf_counter()
            affected = _materialize_daily_chunk(engine, chunk_start, chunk_end, logger)
            insert_elapsed = time.perf_counter() - insert_t0
            total_daily += affected
            daily_state.complete(chunk_key, affected)
            logger.info(
                "daily_chunk_done chunk=%s affected=%s insert_seconds=%.3f total_chunk_seconds=%.3f",
                chunk_key,
                affected,
                insert_elapsed,
                time.perf_counter() - chunk_t0,
            )
        except Exception as exc:
            daily_state.fail(chunk_key, exc)
            raise

    logger.info(
        "build_stock_fundamental_daily_done populated=%s daily_affected=%s start=%s end=%s",
        total_populated,
        total_daily,
        date_range.start,
        date_range.end,
    )


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
