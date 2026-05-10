from __future__ import annotations

import argparse
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

MATERIALIZE_SQL = """
    INSERT INTO cn_stock_fundamental_daily (
        symbol, trade_date, report_end_date, ann_date, revenue_yoy, profit_yoy, roe,
        gross_margin, debt_to_assets, ocfps, inventory, contract_liability, fixed_assets, source
    )
    WITH report_keys AS (
        SELECT symbol, end_date FROM cn_local_stock_income_q
        UNION
        SELECT symbol, end_date FROM cn_local_stock_balancesheet_q
        UNION
        SELECT symbol, end_date FROM cn_local_stock_fina_indicator_q
    ),
    merged AS (
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
        FROM report_keys rk
        LEFT JOIN cn_local_stock_fina_indicator_q fi
          ON fi.symbol = rk.symbol
         AND fi.end_date = rk.end_date
        LEFT JOIN cn_local_stock_balancesheet_q bs
          ON bs.symbol = rk.symbol
         AND bs.end_date = rk.end_date
        LEFT JOIN cn_local_stock_income_q inc
          ON inc.symbol = rk.symbol
         AND inc.end_date = rk.end_date
    ),
    effective AS (
        SELECT
            merged.*,
            LEAD(merged.ann_date) OVER (
                PARTITION BY merged.symbol
                ORDER BY merged.ann_date, merged.end_date
            ) AS next_ann_date
        FROM merged
        WHERE merged.ann_date IS NOT NULL
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
    JOIN effective e
      ON e.symbol = p.SYMBOL
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
            _delete_daily_chunk_if_replace(engine, chunk_start, chunk_end, replace, logger)
            with engine.begin() as conn:
                result = conn.execute(
                    text(MATERIALIZE_SQL),
                    {"chunk_start": chunk_start, "chunk_end": chunk_end},
                )
            affected = int(result.rowcount or 0)
            total_daily += affected
            daily_state.complete(chunk_key, affected)
            logger.info("daily_chunk_done chunk=%s affected=%s", chunk_key, affected)
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
