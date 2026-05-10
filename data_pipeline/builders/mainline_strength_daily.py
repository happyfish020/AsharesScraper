from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sqlalchemy import text

from data_pipeline.common.cli import add_shared_args, month_chunks, resolve_date_range
from data_pipeline.common.db import apply_sql_file, get_engine
from data_pipeline.common.logging_utils import build_logger
from data_pipeline.common.state import BackfillState


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build cn_mainline_strength_daily from capital flow and fundamentals.")
    add_shared_args(parser)
    parser.add_argument("--industry-level", default="L1", choices=["L1", "L2", "L3"])
    return parser


def _compute_scores(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["trade_date", "mainline_name", "strength_score", "leader_count", "capital_ratio", "earnings_score", "trend_days", "expansion_score", "lifecycle_state"])
    frame = frame.sort_values(["mainline_name", "trade_date"]).copy()
    leader_ratio = np.where(frame["member_count"] > 0, frame["leader_count"] / frame["member_count"], 0.0)
    capital_component = np.clip(frame["capital_ratio"].fillna(0.0) * 800.0, 0.0, 25.0)
    return_component = np.clip((frame["relative_return"].fillna(0.0) + 0.03) * 400.0, 0.0, 25.0)
    leader_component = np.clip(leader_ratio * 100.0, 0.0, 25.0)
    earnings_component = np.clip(frame["earnings_score"].fillna(0.0) * 0.15, 0.0, 15.0)
    expansion_component = np.clip(frame["expansion_score"].fillna(0.0) * 0.10, 0.0, 10.0)
    frame["strength_score"] = capital_component + return_component + leader_component + earnings_component + expansion_component

    trend_flags = ((frame["relative_return"].fillna(0.0) > 0) & (frame["capital_ratio"].fillna(0.0) > 0)).astype(int)
    streaks: list[int] = []
    current_name = None
    current_streak = 0
    for row in frame.itertuples(index=False):
        if row.mainline_name != current_name:
            current_name = row.mainline_name
            current_streak = 0
        if getattr(row, "relative_return") is not None and getattr(row, "relative_return") > 0 and getattr(row, "capital_ratio") is not None and getattr(row, "capital_ratio") > 0:
            current_streak += 1
        else:
            current_streak = 0
        streaks.append(current_streak)
    frame["trend_days"] = streaks

    frame["lifecycle_state"] = "NEUTRAL"
    frame.loc[(frame["strength_score"] >= 75) & (frame["trend_days"] >= 20), "lifecycle_state"] = "CONFIRM"
    frame.loc[(frame["strength_score"] >= 65) & (frame["trend_days"] >= 5), "lifecycle_state"] = "IGNITE"
    frame.loc[(frame["strength_score"] >= 55) & (frame["expansion_score"] >= 35), "lifecycle_state"] = "EXPAND"
    frame.loc[(frame["relative_return"].fillna(0.0) < 0) & (frame["trend_days"] == 0) & (frame["capital_ratio"].fillna(0.0) < 0.01), "lifecycle_state"] = "FADE"

    return frame[["trade_date", "mainline_name", "strength_score", "leader_count", "capital_ratio", "earnings_score", "trend_days", "expansion_score", "lifecycle_state"]].copy()


def run(args: argparse.Namespace) -> None:
    date_range = resolve_date_range(args.start, args.end)
    logger = build_logger("build_cn_mainline_strength_daily")
    engine = get_engine()
    apply_sql_file(engine, "docs/DDL/ga_mainline_data_backfill_system.sql")
    state = BackfillState(engine=engine, job_name="build_cn_mainline_strength_daily")
    select_sql = """
    WITH earnings AS (
        SELECT
            m.industry_id,
            fd.trade_date,
            AVG(
                (CASE WHEN fd.revenue_yoy > 0 THEN 20 ELSE 0 END) +
                (CASE WHEN fd.profit_yoy > 0 THEN 20 ELSE 0 END) +
                (CASE WHEN fd.roe >= 8 THEN 20 ELSE 0 END) +
                (CASE WHEN fd.gross_margin >= 15 THEN 20 ELSE 0 END) +
                (CASE WHEN fd.debt_to_assets IS NOT NULL AND fd.debt_to_assets <= 70 THEN 20 ELSE 0 END)
            ) AS earnings_score
        FROM cn_stock_fundamental_daily fd
        JOIN cn_local_industry_map_hist m
          ON m.symbol = fd.symbol COLLATE utf8mb4_unicode_ci
         AND fd.trade_date BETWEEN m.in_date AND COALESCE(m.out_date, DATE('2099-12-31'))
        JOIN cn_local_industry_master im
          ON im.industry_id = m.industry_id
         AND im.industry_level = :industry_level
        WHERE fd.trade_date BETWEEN :chunk_start AND :chunk_end
        GROUP BY m.industry_id, fd.trade_date
    )
    SELECT
        cf.trade_date,
        im.industry_name AS mainline_name,
        cf.leader_count,
        cf.market_turnover_ratio AS capital_ratio,
        COALESCE(e.earnings_score, 0) AS earnings_score,
        CASE
            WHEN proxy.member_count = 0 THEN 0
            ELSE LEAST(100, ((cf.breakout_count + cf.trend_count) / proxy.member_count) * 50)
        END AS expansion_score,
        proxy.member_count,
        cf.relative_return
    FROM cn_industry_capital_flow_daily cf
    JOIN cn_local_industry_master im
      ON im.industry_id = cf.industry_id
     AND im.industry_level = :industry_level
    LEFT JOIN cn_local_industry_proxy_daily proxy
      ON proxy.industry_id = cf.industry_id
     AND proxy.trade_date = cf.trade_date
    LEFT JOIN earnings e
      ON e.industry_id = cf.industry_id
     AND e.trade_date = cf.trade_date
    WHERE cf.trade_date BETWEEN :chunk_start AND :chunk_end
    ORDER BY cf.trade_date, im.industry_name
    """
    upsert_sql = """
    INSERT INTO cn_mainline_strength_daily (
        trade_date, mainline_name, strength_score, leader_count, capital_ratio,
        earnings_score, trend_days, expansion_score, lifecycle_state
    ) VALUES (
        :trade_date, :mainline_name, :strength_score, :leader_count, :capital_ratio,
        :earnings_score, :trend_days, :expansion_score, :lifecycle_state
    )
    ON DUPLICATE KEY UPDATE
        strength_score = VALUES(strength_score),
        leader_count = VALUES(leader_count),
        capital_ratio = VALUES(capital_ratio),
        earnings_score = VALUES(earnings_score),
        trend_days = VALUES(trend_days),
        expansion_score = VALUES(expansion_score),
        lifecycle_state = VALUES(lifecycle_state),
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
            with engine.connect() as conn:
                raw = pd.read_sql(text(select_sql), conn, params={"industry_level": args.industry_level, "chunk_start": chunk_start, "chunk_end": chunk_end})
            scored = _compute_scores(raw)
            rows = scored.astype(object).where(pd.notna(scored), None).to_dict(orient="records")
            with engine.begin() as conn:
                for index in range(0, len(rows), 4000):
                    conn.execute(text(upsert_sql), rows[index : index + 4000])
            total_rows += len(rows)
            state.complete(chunk_key, len(rows))
            logger.info("chunk_done chunk=%s rows=%s", chunk_key, len(rows))
        except Exception as exc:
            state.fail(chunk_key, exc)
            raise
    logger.info("build_cn_mainline_strength_daily_done rows=%s start=%s end=%s", total_rows, date_range.start, date_range.end)


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
