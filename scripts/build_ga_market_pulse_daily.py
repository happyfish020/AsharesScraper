"""
scripts/build_ga_market_pulse_daily.py
========================================
GrowthAlpha V8 — GA Layer — Part C.

Builds cn_ga_market_pulse_daily from:
  1. cn_ga_mainline_radar_daily  — mainline-level daily radar data
  2. cn_index_daily_price        — market index returns (HS300, ChiNext, SZ)

Key output columns used by build_unified_alpha_score_daily.py (Factor 8):
  bullish_industry_ratio  = fraction of mainlines with state CONFIRMED or FORMING
  bearish_industry_ratio  = fraction of mainlines with state FADE
  market_phase            = TREND_EXPANSION | DIFFUSION | BOTTOM_REPAIR |
                            DIVERGENCE | TOP_DECAY | RISK_OFF | CRISIS

market_state ENUM: TREND_STRONG | TREND_WEAK | RANGE | RISK_OFF

Run order: after build_ga_mainline_radar_daily.py

Usage:
  python scripts/build_ga_market_pulse_daily.py \\
      --start 2010-01-01 --end 2026-03-31 \\
      --db-name cn_market_red --replace

  python scripts/build_ga_market_pulse_daily.py \\
      --start 2010-01-01 --end 2026-03-31 \\
      --db-name cn_market_red --dry-run --verbose
"""

from __future__ import annotations

import argparse
import logging
import os
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, URL

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHUNK_MONTHS = 3

# Index codes in cn_index_daily_price
INDEX_HS300 = "sh000300"
INDEX_CHIEXT = "sh000688"  # STAR/ChiNext proxy
INDEX_SZ = "sz399001"

# State thresholds
BULLISH_STRONG_THRESH = 0.50   # > 50% mainlines bullish → TREND_STRONG
BULLISH_WEAK_THRESH = 0.30     # 30-50% → TREND_WEAK
BEARISH_RISK_OFF_THRESH = 0.50  # > 50% bearish → RISK_OFF

# market_phase derivation
PHASE_EXPANSION_BULL = 0.50    # bullish_ratio > 50% → TREND_EXPANSION
PHASE_DIFFUSION_BULL = 0.30    # 30-50% → DIFFUSION
PHASE_BOTTOM_BEAR = 0.30       # bearish > 30% but recovering → BOTTOM_REPAIR
PHASE_TOP_DECAY_SCORE = 70.0   # market_score falling from high → TOP_DECAY
PHASE_RISK_OFF_BEAR = 0.50     # bearish > 50% → RISK_OFF
PHASE_CRISIS_BEAR = 0.70       # bearish > 70% + HS300 < -3% → CRISIS

# Bullish mainline states
BULLISH_STATES = {"CONFIRMED", "FORMING"}
BEARISH_STATES = {"FADE"}
NEUTRAL_STATES = {"EARLY", "ROTATING"}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="GrowthAlpha V8 — Build cn_ga_market_pulse_daily"
    )
    p.add_argument("--start", default="2010-01-01")
    p.add_argument("--end", default="")
    p.add_argument("--db-name", default="cn_market_red")
    p.add_argument("--db-host", default="127.0.0.1")
    p.add_argument("--db-port", type=int, default=3306)
    p.add_argument("--db-user", default="cn_opr_red")
    p.add_argument("--db-password", default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--replace", action="store_true")
    p.add_argument("--chunk-months", type=int, default=CHUNK_MONTHS)
    p.add_argument("--verbose", action="store_true")
    return p


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------


def build_engine(host: str, port: int, user: str, password: str, db: str) -> Engine:
    url = URL.create(
        "mysql+pymysql", username=user, password=password,
        host=host, port=port, database=db,
        query={"charset": "utf8mb4"},
    )
    return create_engine(url, pool_pre_ping=True, future=True)


def fetch_df(engine: Engine, sql: str, params: dict[str, Any] | None = None) -> pd.DataFrame:
    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn, params=params or {})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_float(val: Any, default: float = 0.0) -> float:
    if val is None:
        return default
    try:
        v = float(val)
        return default if (np.isnan(v) or np.isinf(v)) else v
    except (ValueError, TypeError):
        return default


def _clip01(v: float) -> float:
    return max(0.0, min(1.0, v))


def _clip100(v: float) -> float:
    return max(0.0, min(100.0, v))


def _date_chunks(start: date, end: date, months: int) -> list[tuple[date, date]]:
    chunks = []
    cur = start
    while cur <= end:
        next_month = date(
            cur.year + (cur.month + months - 1) // 12,
            (cur.month + months - 1) % 12 + 1,
            1,
        )
        chunk_end = min(next_month - timedelta(days=1), end)
        chunks.append((cur, chunk_end))
        cur = next_month
    return chunks


def _market_state(bullish: float, bearish: float) -> str:
    if bullish > BULLISH_STRONG_THRESH:
        return "TREND_STRONG"
    if bearish > BEARISH_RISK_OFF_THRESH:
        return "RISK_OFF"
    if bullish > BULLISH_WEAK_THRESH:
        return "TREND_WEAK"
    return "RANGE"


def _market_phase(
    bullish: float,
    bearish: float,
    hs300_chg: float,
    market_score: float,
    prev_market_score: float | None,
) -> str:
    """Derive market phase string used by Factor 8 risk assessment."""
    score_declining = (
        prev_market_score is not None
        and prev_market_score > PHASE_TOP_DECAY_SCORE
        and market_score < prev_market_score - 5
    )

    if bearish > PHASE_CRISIS_BEAR and hs300_chg < -3.0:
        return "CRISIS"
    if bearish > PHASE_RISK_OFF_BEAR:
        return "RISK_OFF"
    if score_declining and market_score > 55:
        return "TOP_DECAY"
    if bullish > PHASE_EXPANSION_BULL:
        return "TREND_EXPANSION"
    if bullish > PHASE_DIFFUSION_BULL:
        return "DIFFUSION"
    if bearish > PHASE_BOTTOM_BEAR:
        return "BOTTOM_REPAIR"
    return "DIVERGENCE"


def _target_exposure(state: str) -> float:
    return {
        "TREND_STRONG": 0.90,
        "TREND_WEAK": 0.65,
        "RANGE": 0.45,
        "RISK_OFF": 0.20,
    }.get(state, 0.45)


def _risk_flag(state: str, bearish: float, hs300_chg: float) -> str:
    if state == "RISK_OFF" or bearish > 0.6:
        return "HIGH"
    if bearish > 0.4 or hs300_chg < -2.0:
        return "MEDIUM"
    if bearish > 0.2:
        return "LOW"
    return "NONE"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_radar(engine: Engine, start: date, end: date) -> pd.DataFrame:
    sql = """
        SELECT
            trade_date,
            mainline_id,
            mainline_state,
            mainline_score,
            up_ratio,
            leader_density,
            trend_alignment_score,
            rs_20d,
            amount_sum,
            amount_chg_5d
        FROM cn_ga_mainline_radar_daily
        WHERE trade_date BETWEEN :start AND :end
    """
    df = fetch_df(engine, sql, {"start": start, "end": end})
    if not df.empty:
        df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
        for col in ["mainline_score", "up_ratio", "leader_density",
                    "trend_alignment_score", "rs_20d", "amount_sum", "amount_chg_5d"]:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: _safe_float(x, 0.0))
    return df


def load_index_prices(engine: Engine, start: date, end: date) -> pd.DataFrame:
    sql = """
        SELECT
            TRADE_DATE as trade_date,
            INDEX_CODE as index_code,
            CHG_PCT as chg_pct,
            CLOSE as close_price
        FROM cn_index_daily_price
        WHERE TRADE_DATE BETWEEN :start AND :end
          AND INDEX_CODE IN (:hs300, :chiext, :sz)
    """
    df = fetch_df(engine, sql, {
        "start": start, "end": end,
        "hs300": INDEX_HS300, "chiext": INDEX_CHIEXT, "sz": INDEX_SZ,
    })
    if not df.empty:
        df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
        df["chg_pct"] = df["chg_pct"].apply(lambda x: _safe_float(x, 0.0))
    return df


# ---------------------------------------------------------------------------
# Pulse computation
# ---------------------------------------------------------------------------


def build_pulse(
    radar_df: pd.DataFrame,
    index_df: pd.DataFrame,
    output_start: date,
    output_end: date,
) -> pd.DataFrame:
    """Aggregate mainline radar into daily market pulse."""
    if radar_df.empty:
        return pd.DataFrame()

    # Pivot index prices
    index_lookup: dict[tuple[date, str], float] = {}
    if not index_df.empty:
        for _, row in index_df.iterrows():
            index_lookup[(row["trade_date"], row["index_code"])] = row["chg_pct"]

    rows_out = []
    prev_score: float | None = None
    trade_dates = sorted(d for d in radar_df["trade_date"].unique()
                         if output_start <= d <= output_end)

    for td in trade_dates:
        day_radar = radar_df[radar_df["trade_date"] == td]
        if day_radar.empty:
            continue

        n_total = len(day_radar)

        # State distribution
        states = day_radar["mainline_state"].fillna("EARLY").str.upper()
        n_bullish = states.isin(BULLISH_STATES).sum()
        n_bearish = states.isin(BEARISH_STATES).sum()
        n_neutral = n_total - n_bullish - n_bearish

        bullish_ratio = _clip01(n_bullish / n_total) if n_total > 0 else 0.5
        bearish_ratio = _clip01(n_bearish / n_total) if n_total > 0 else 0.0
        neutral_ratio = _clip01(n_neutral / n_total) if n_total > 0 else 0.5

        # Breadth
        breadth_up_ratio = _clip01(day_radar["up_ratio"].mean())
        breadth_down_ratio = _clip01(1.0 - breadth_up_ratio)
        trend_alignment_avg = _clip01(day_radar["trend_alignment_score"].mean())

        # Leader density and mainline stability
        avg_leader_density = _clip01(day_radar["leader_density"].mean())
        mainline_stability = _clip01(
            1.0 - day_radar["mainline_state"].apply(
                lambda s: 1.0 if str(s).upper() == "ROTATING" else 0.0
            ).mean()
        )

        # Industry expansion breadth: fraction of mainlines with positive rs_20d
        industry_expansion = _clip01(
            (day_radar["rs_20d"] > 0).sum() / n_total
        ) if n_total > 0 else 0.5

        # Amount score: 0-100 based on total market volume relative to moving avg
        amount_sum = day_radar["amount_sum"].sum()
        amount_score = _clip100(
            50.0 + (bullish_ratio - bearish_ratio) * 100.0 * 0.5
        )

        # Market score composite 0-100
        market_score = _clip100(
            breadth_up_ratio * 40.0
            + bullish_ratio * 35.0
            + trend_alignment_avg * 15.0
            + (1.0 - bearish_ratio) * 10.0
        )

        # Top mainlines (CONFIRMED or high-scoring)
        top_mainline_count = int(states.isin({"CONFIRMED"}).sum())

        # Index returns
        hs300_chg = _safe_float(index_lookup.get((td, INDEX_HS300)), 0.0)
        cyb_chg = _safe_float(index_lookup.get((td, INDEX_CHIEXT)), 0.0)
        sz_chg = _safe_float(index_lookup.get((td, INDEX_SZ)), 0.0)

        # Index RS score (0-100): how strong is the market index today
        index_rs_score = _clip100(50.0 + hs300_chg * 10.0)

        # Rotation speed: std dev of rs_20d across mainlines (higher = more rotation)
        rs_std = float(day_radar["rs_20d"].std()) if n_total > 1 else 0.0
        rotation_speed = _clip01(min(1.0, rs_std / 5.0))  # normalize 5% std → 1.0

        # Derived states
        state = _market_state(bullish_ratio, bearish_ratio)
        phase = _market_phase(bullish_ratio, bearish_ratio, hs300_chg,
                               market_score, prev_score)
        target_exp = _target_exposure(state)
        risk_f = _risk_flag(state, bearish_ratio, hs300_chg)

        reason = (
            f"breadth={breadth_up_ratio:.4f}; "
            f"amount_score={amount_score:.2f}; "
            f"index_rs_score={index_rs_score:.4f}"
        )

        rows_out.append({
            "trade_date": td,
            "market_score": round(market_score, 4),
            "market_state": state,
            "target_exposure": round(target_exp, 4),
            "breadth_up_ratio": round(breadth_up_ratio, 4),
            "breadth_down_ratio": round(breadth_down_ratio, 4),
            "amount_score": round(amount_score, 4),
            "index_rs_score": round(index_rs_score, 4),
            "hs300_pct_chg": round(hs300_chg, 4),
            "cyb_pct_chg": round(cyb_chg, 4),
            "sz_pct_chg": round(sz_chg, 4),
            "risk_flag": risk_f,
            "reason": reason,
            # Extended columns
            "bullish_industry_ratio": round(bullish_ratio, 4),
            "neutral_industry_ratio": round(neutral_ratio, 4),
            "bearish_industry_ratio": round(bearish_ratio, 4),
            "rotation_speed": round(rotation_speed, 4),
            "mainline_stability": round(mainline_stability, 4),
            "trend_alignment_avg": round(trend_alignment_avg, 4),
            "industry_expansion_breadth": round(industry_expansion, 4),
            "top_mainline_count": top_mainline_count,
            "market_phase": phase,
        })
        prev_score = market_score

    return pd.DataFrame(rows_out) if rows_out else pd.DataFrame()


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------


def delete_range(engine: Engine, start: date, end: date) -> None:
    with engine.begin() as conn:
        conn.execute(
            text("DELETE FROM cn_ga_market_pulse_daily WHERE trade_date BETWEEN :s AND :e"),
            {"s": start, "e": end},
        )


def write_rows(engine: Engine, df: pd.DataFrame, dry_run: bool) -> int:
    if df.empty:
        return 0
    if dry_run:
        logger.info("  [dry-run] would write %d rows", len(df))
        return 0

    cols = [
        "trade_date", "market_score", "market_state", "target_exposure",
        "breadth_up_ratio", "breadth_down_ratio", "amount_score", "index_rs_score",
        "hs300_pct_chg", "cyb_pct_chg", "sz_pct_chg", "risk_flag", "reason",
        "bullish_industry_ratio", "neutral_industry_ratio", "bearish_industry_ratio",
        "rotation_speed", "mainline_stability", "trend_alignment_avg",
        "industry_expansion_breadth", "top_mainline_count", "market_phase",
    ]
    write_cols = [c for c in cols if c in df.columns]
    records = df[write_cols].to_dict("records")

    col_list = ", ".join(write_cols)
    val_list = ", ".join(f":{c}" for c in write_cols)
    sql = f"INSERT INTO cn_ga_market_pulse_daily ({col_list}) VALUES ({val_list})"

    BATCH = 500
    written = 0
    with engine.begin() as conn:
        for i in range(0, len(records), BATCH):
            conn.execute(text(sql), records[i: i + BATCH])
            written += min(BATCH, len(records) - i)
    return written


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = build_parser().parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    pw = args.db_password or os.environ.get("ASHARE_MYSQL_PASSWORD", "")
    engine = build_engine(args.db_host, args.db_port, args.db_user, pw, args.db_name)

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end) if args.end else date.today()

    logger.info("build_ga_market_pulse_daily  %s ~ %s  dry_run=%s replace=%s",
                start, end, args.dry_run, args.replace)

    chunks = _date_chunks(start, end, args.chunk_months)
    total_written = 0
    prev_score_carry: float | None = None

    for chunk_start, chunk_end in chunks:
        logger.info("  chunk %s ~ %s", chunk_start, chunk_end)

        radar_df = load_radar(engine, chunk_start, chunk_end)
        if radar_df.empty:
            logger.info("    no radar data, skip")
            continue

        index_df = load_index_prices(engine, chunk_start, chunk_end)
        logger.debug("    radar=%d index=%d", len(radar_df), len(index_df))

        out = build_pulse(radar_df, index_df, chunk_start, chunk_end)
        if out.empty:
            logger.info("    no output, skip")
            continue

        if args.replace and not args.dry_run:
            delete_range(engine, chunk_start, chunk_end)

        n = write_rows(engine, out, args.dry_run)
        total_written += n
        logger.info("    wrote %d rows", n)

        if not out.empty:
            prev_score_carry = float(out["market_score"].iloc[-1])

    logger.info("Done. Total rows written: %d", total_written)


if __name__ == "__main__":
    main()
