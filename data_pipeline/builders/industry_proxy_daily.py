from __future__ import annotations

import argparse
from sqlalchemy import text

from data_pipeline.common.cli import add_shared_args, month_chunks, resolve_date_range
from data_pipeline.common.db import apply_sql_file, get_engine
from data_pipeline.common.logging_utils import build_logger
from data_pipeline.common.state import BackfillState


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build cn_local_industry_proxy_daily from stock prices and historical SW membership.")
    add_shared_args(parser)
    parser.add_argument("--industry-level", default="L1", choices=["L1", "L2", "L3"])
    return parser


def run(args: argparse.Namespace) -> None:
    date_range = resolve_date_range(args.start, args.end)
    logger = build_logger("build_industry_proxy_daily")
    engine = get_engine()
    apply_sql_file(engine, "docs/DDL/ga_mainline_data_backfill_system.sql")
    state = BackfillState(engine=engine, job_name="build_industry_proxy_daily")
    sql = """
    INSERT INTO cn_local_industry_proxy_daily (
        industry_id, trade_date, member_count, ret_eqw, amount_total, turnover_avg,
        market_cap_total, leader_return, top5_concentration
    )
    WITH base AS (
        SELECT
            m.industry_id,
            p.TRADE_DATE,
            p.SYMBOL,
            COALESCE(b.total_mv, b.circ_mv, 0) AS market_cap,
            p.AMOUNT,
            p.TURNOVER_RATE,
            CASE
                WHEN p.PRE_CLOSE IS NOT NULL AND p.PRE_CLOSE <> 0 THEN (p.CLOSE / p.PRE_CLOSE) - 1
                WHEN p.CHG_PCT IS NOT NULL AND ABS(p.CHG_PCT) > 1 THEN p.CHG_PCT / 100
                WHEN p.CHG_PCT IS NOT NULL THEN p.CHG_PCT
                ELSE NULL
            END AS stock_ret
        FROM cn_local_industry_map_hist m
        JOIN cn_local_industry_master im
          ON im.industry_id = m.industry_id
         AND im.industry_level = :industry_level
        JOIN cn_stock_daily_price p
          ON p.SYMBOL = m.symbol COLLATE utf8mb4_unicode_ci
         AND p.TRADE_DATE BETWEEN m.in_date AND COALESCE(m.out_date, DATE('2099-12-31'))
        LEFT JOIN cn_stock_daily_basic b
          ON b.symbol = p.SYMBOL COLLATE utf8mb4_unicode_ci
         AND b.trade_date = p.TRADE_DATE
        WHERE p.TRADE_DATE BETWEEN :chunk_start AND :chunk_end
    ),
    ranked AS (
        SELECT
            base.*,
            ROW_NUMBER() OVER (PARTITION BY industry_id, trade_date ORDER BY market_cap DESC, symbol) AS mc_rank
        FROM base
    )
    SELECT
        industry_id,
        trade_date,
        COUNT(*) AS member_count,
        AVG(stock_ret) AS ret_eqw,
        SUM(amount) AS amount_total,
        AVG(turnover_rate) AS turnover_avg,
        SUM(market_cap) AS market_cap_total,
        MAX(stock_ret) AS leader_return,
        CASE
            WHEN SUM(market_cap) = 0 THEN NULL
            ELSE SUM(CASE WHEN mc_rank <= 5 THEN market_cap ELSE 0 END) / SUM(market_cap)
        END AS top5_concentration
    FROM ranked
    GROUP BY industry_id, trade_date
    ON DUPLICATE KEY UPDATE
        member_count = VALUES(member_count),
        ret_eqw = VALUES(ret_eqw),
        amount_total = VALUES(amount_total),
        turnover_avg = VALUES(turnover_avg),
        market_cap_total = VALUES(market_cap_total),
        leader_return = VALUES(leader_return),
        top5_concentration = VALUES(top5_concentration),
        updated_at = CURRENT_TIMESTAMP
    """
    total_rows = 0
    for chunk_start, chunk_end in month_chunks(date_range.start, date_range.end, max(1, int(args.chunk_months))):
        chunk_key = f"{args.industry_level}:{chunk_start}:{chunk_end}"
        if args.resume and state.is_completed(chunk_key):
            logger.info("resume_skip chunk=%s", chunk_key)
            continue
        state.start(chunk_key, chunk_start, chunk_end)
        try:
            with engine.begin() as conn:
                result = conn.execute(text(sql), {"industry_level": args.industry_level, "chunk_start": chunk_start, "chunk_end": chunk_end})
            affected = int(result.rowcount or 0)
            total_rows += affected
            state.complete(chunk_key, affected)
            logger.info("chunk_done chunk=%s affected=%s", chunk_key, affected)
        except Exception as exc:
            state.fail(chunk_key, exc)
            raise
    logger.info("build_industry_proxy_daily_done affected=%s start=%s end=%s", total_rows, date_range.start, date_range.end)


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
