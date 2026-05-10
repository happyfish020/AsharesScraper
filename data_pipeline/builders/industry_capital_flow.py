from __future__ import annotations

import argparse
from datetime import timedelta

from sqlalchemy import text

from data_pipeline.common.cli import add_shared_args, month_chunks, resolve_date_range
from data_pipeline.common.db import apply_sql_file, get_engine
from data_pipeline.common.logging_utils import build_logger
from data_pipeline.common.state import BackfillState


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build cn_industry_capital_flow_daily from proxy daily and stock-level features.")
    add_shared_args(parser)
    parser.add_argument("--industry-level", default="L1", choices=["L1", "L2", "L3"])
    return parser


def run(args: argparse.Namespace) -> None:
    date_range = resolve_date_range(args.start, args.end)
    logger = build_logger("build_industry_capital_flow")
    engine = get_engine()
    apply_sql_file(engine, "docs/DDL/ga_mainline_data_backfill_system.sql")
    state = BackfillState(engine=engine, job_name="build_industry_capital_flow")
    sql = """
    INSERT INTO cn_industry_capital_flow_daily (
        trade_date, industry_id, industry_turnover, market_turnover_ratio, industry_return,
        relative_return, leader_count, breakout_count, trend_count, capital_concentration
    )
    WITH stock_features AS (
        SELECT
            p.SYMBOL,
            p.TRADE_DATE,
            p.CLOSE,
            p.AMOUNT,
            CASE
                WHEN p.PRE_CLOSE IS NOT NULL AND p.PRE_CLOSE <> 0 THEN (p.CLOSE / p.PRE_CLOSE) - 1
                WHEN p.CHG_PCT IS NOT NULL AND ABS(p.CHG_PCT) > 1 THEN p.CHG_PCT / 100
                WHEN p.CHG_PCT IS NOT NULL THEN p.CHG_PCT
                ELSE NULL
            END AS stock_ret,
            AVG(p.CLOSE) OVER (PARTITION BY p.SYMBOL ORDER BY p.TRADE_DATE ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS ma20,
            AVG(p.CLOSE) OVER (PARTITION BY p.SYMBOL ORDER BY p.TRADE_DATE ROWS BETWEEN 59 PRECEDING AND CURRENT ROW) AS ma60,
            MAX(p.CLOSE) OVER (PARTITION BY p.SYMBOL ORDER BY p.TRADE_DATE ROWS BETWEEN 60 PRECEDING AND 1 PRECEDING) AS prev60_high,
            AVG(p.AMOUNT) OVER (PARTITION BY p.SYMBOL ORDER BY p.TRADE_DATE ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS amt20
        FROM cn_stock_daily_price p
        WHERE p.TRADE_DATE BETWEEN :feature_start AND :chunk_end
    ),
    market_daily AS (
        SELECT TRADE_DATE, SUM(AMOUNT) AS market_amount, AVG(stock_ret) AS market_ret
        FROM stock_features
        WHERE trade_date BETWEEN :chunk_start AND :chunk_end
        GROUP BY trade_date
    ),
    industry_stats AS (
        SELECT
            m.industry_id,
            sf.trade_date,
            SUM(CASE WHEN sf.stock_ret >= 0.095 THEN 1 ELSE 0 END) AS leader_count,
            SUM(CASE WHEN sf.prev60_high IS NOT NULL AND sf.close > sf.prev60_high AND sf.amt20 IS NOT NULL AND sf.amt20 > 0 AND sf.amount >= sf.amt20 * 1.5 THEN 1 ELSE 0 END) AS breakout_count,
            SUM(CASE WHEN sf.ma20 IS NOT NULL AND sf.ma60 IS NOT NULL AND sf.close > sf.ma20 AND sf.ma20 > sf.ma60 THEN 1 ELSE 0 END) AS trend_count
        FROM stock_features sf
        JOIN cn_local_industry_map_hist m
          ON m.symbol = sf.symbol COLLATE utf8mb4_unicode_ci
         AND sf.trade_date BETWEEN m.in_date AND COALESCE(m.out_date, DATE('2099-12-31'))
        JOIN cn_local_industry_master im
          ON im.industry_id = m.industry_id
         AND im.industry_level = :industry_level
        WHERE sf.trade_date BETWEEN :chunk_start AND :chunk_end
        GROUP BY m.industry_id, sf.trade_date
    )
    SELECT
        p.trade_date,
        p.industry_id,
        p.amount_total AS industry_turnover,
        CASE WHEN md.market_amount IS NULL OR md.market_amount = 0 THEN NULL ELSE p.amount_total / md.market_amount END AS market_turnover_ratio,
        p.ret_eqw AS industry_return,
        CASE WHEN md.market_ret IS NULL THEN NULL ELSE p.ret_eqw - md.market_ret END AS relative_return,
        COALESCE(s.leader_count, 0) AS leader_count,
        COALESCE(s.breakout_count, 0) AS breakout_count,
        COALESCE(s.trend_count, 0) AS trend_count,
        p.top5_concentration AS capital_concentration
    FROM cn_local_industry_proxy_daily p
    JOIN cn_local_industry_master im
      ON im.industry_id = p.industry_id
     AND im.industry_level = :industry_level
    LEFT JOIN market_daily md
      ON md.trade_date = p.trade_date
    LEFT JOIN industry_stats s
      ON s.industry_id = p.industry_id
     AND s.trade_date = p.trade_date
    WHERE p.trade_date BETWEEN :chunk_start AND :chunk_end
    ON DUPLICATE KEY UPDATE
        industry_turnover = VALUES(industry_turnover),
        market_turnover_ratio = VALUES(market_turnover_ratio),
        industry_return = VALUES(industry_return),
        relative_return = VALUES(relative_return),
        leader_count = VALUES(leader_count),
        breakout_count = VALUES(breakout_count),
        trend_count = VALUES(trend_count),
        capital_concentration = VALUES(capital_concentration),
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
            feature_start = chunk_start - timedelta(days=90)
            with engine.begin() as conn:
                result = conn.execute(
                    text(sql),
                    {
                        "industry_level": args.industry_level,
                        "feature_start": feature_start,
                        "chunk_start": chunk_start,
                        "chunk_end": chunk_end,
                    },
                )
            affected = int(result.rowcount or 0)
            total_rows += affected
            state.complete(chunk_key, affected)
            logger.info("chunk_done chunk=%s affected=%s", chunk_key, affected)
        except Exception as exc:
            state.fail(chunk_key, exc)
            raise
    logger.info("build_industry_capital_flow_done affected=%s start=%s end=%s", total_rows, date_range.start, date_range.end)


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
