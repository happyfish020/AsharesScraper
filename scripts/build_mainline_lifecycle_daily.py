"""
scripts/build_mainline_lifecycle_daily.py
===========================================
GrowthAlpha V8 — P2 Mainline Lifecycle Engine Builder.

Computes lifecycle state, lifecycle_score, phase_reason, risk_flag,
and rotation_rank for each mainline (industry) per trade_date.

Input sources (priority order):
  1. cn_ga_mainline_radar_daily       — mainline_score, breakout_ratio, new_high_ratio,
                                        leader_density, trend_alignment_score, rotation_rank
  2. cn_stock_mainline_strength_daily — mainline_strength_score (secondary strength)
  3. cn_industry_capital_flow_daily   — capital_concentration_score (optional)
  4. cn_ga_market_pulse_daily         — market_score, market_state, bullish/bearish ratios
  5. cn_local_industry_proxy_daily    — member_count, top5_concentration
  6. cn_mainline_strength_daily       — fallback only (may be empty)

Lifecycle States:
  BOTTOM_REPAIR    — Mainline recovering from low base; breakout_ratio & trend_alignment improving.
  TREND_EXPANSION  — Mainline in primary uptrend; capital, trend, breakout, new_high, leader density resonate.
  DIFFUSION        — Spreading from leaders to more chain links; strong_stock_count rising.
  DIVERGENCE       — Still strong but internal structure diverging; leader crowding or diffusion failure.
  TOP_DECAY        — Heat fading; breakout_ratio, new_high_ratio, rotation_rank worsening.
  RISK_OFF         — Market-wide risk contraction; mainline scores invalid or most sectors weakening.
  UNKNOWN          — Insufficient data or cannot determine.

Usage:
  python scripts/build_mainline_lifecycle_daily.py --start 2026-03-30 --end 2026-03-30 --db-name cn_market_red --replace --verbose
  python scripts/build_mainline_lifecycle_daily.py --start 2026-01-01 --end 2026-03-30 --db-name cn_market_red --replace
  python scripts/build_mainline_lifecycle_daily.py --start 2026-03-30 --end 2026-03-30 --db-name cn_market_red --dry-run --verbose
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, URL

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LIFECYCLE_STATES = [
    "BOTTOM_REPAIR",
    "TREND_EXPANSION",
    "DIFFUSION",
    "DIVERGENCE",
    "TOP_DECAY",
    "RISK_OFF",
    "UNKNOWN",
]

RISK_FLAGS = ["NONE", "CROWDING", "TOP_DECAY", "MARKET_RISK_OFF", "DATA_INSUFFICIENT"]

REPORT_DIR = Path("reports") / "mainline_lifecycle"

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="GrowthAlpha V8 — P2 Build cn_mainline_lifecycle_daily"
    )
    parser.add_argument("--start", default="2010-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default="", help="End date YYYY-MM-DD (default=today)")
    parser.add_argument("--db-name", default="cn_market_red", help="Database name")
    parser.add_argument("--db-host", default="127.0.0.1", help="MySQL host")
    parser.add_argument("--db-port", type=int, default=3306, help="MySQL port")
    parser.add_argument("--db-user", default="root", help="MySQL user")
    parser.add_argument("--db-password", default=None, help="MySQL password (default from env ASHARE_MYSQL_PASSWORD)")
    parser.add_argument("--dry-run", action="store_true", help="Compute but do not write to DB")
    parser.add_argument("--replace", action="store_true", help="Replace existing rows in date range")
    parser.add_argument("--output-dir", default=None, help="Override report output directory")
    parser.add_argument("--min-member-count", type=int, default=5, help="Minimum industry member count to consider")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    return parser


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------


def build_engine(db_host: str, db_port: int, db_user: str, db_password: str, db_name: str) -> Engine:
    conn_url = URL.create(
        drivername="mysql+pymysql",
        username=db_user,
        password=db_password,
        host=db_host,
        port=db_port,
        database=db_name,
        query={"charset": "utf8mb4"},
    )
    return create_engine(conn_url, pool_pre_ping=True, future=True)


def fetch_df(engine: Engine, sql: str, params: dict[str, Any] | None = None) -> pd.DataFrame:
    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn, params=params or {})


def table_exists(engine: Engine, db_name: str, table_name: str) -> bool:
    sql = """
        SELECT COUNT(*) AS cnt
        FROM INFORMATION_SCHEMA.TABLES
        WHERE TABLE_SCHEMA = :schema AND TABLE_NAME = :table
    """
    with engine.connect() as conn:
        return conn.execute(text(sql), {"schema": db_name, "table": table_name}).scalar() > 0


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_ga_mainline_radar(
    engine: Engine, start: date, end: date
) -> pd.DataFrame:
    """Load cn_ga_mainline_radar_daily for the date range (PRIMARY source)."""
    sql = """
    SELECT
        trade_date,
        mainline_id,
        mainline_name,
        mainline_score,
        rs_60d,
        rs_120d,
        trend_alignment_score,
        rotation_rank,
        heat_percentile_5d,
        breakout_ratio,
        new_high_ratio,
        strong_stock_count,
        leader_density,
        mainline_phase,
        mainline_confidence
    FROM cn_ga_mainline_radar_daily
    WHERE trade_date BETWEEN :start AND :end
    ORDER BY trade_date, mainline_id
    """
    return fetch_df(engine, sql, {"start": start, "end": end})


def load_stock_mainline_strength(
    engine: Engine, start: date, end: date
) -> pd.DataFrame:
    """Load cn_stock_mainline_strength_daily (secondary strength source)."""
    sql = """
    SELECT
        trade_date,
        industry_id,
        industry_name,
        mainline_strength_score,
        leader_density AS strength_leader_density,
        breakout_ratio AS strength_breakout_ratio,
        trend_alignment,
        breadth_score,
        acceleration_score,
        lifecycle_bonus,
        rank_in_market,
        is_active_mainline
    FROM cn_stock_mainline_strength_daily
    WHERE trade_date BETWEEN :start AND :end
    ORDER BY trade_date, industry_id
    """
    return fetch_df(engine, sql, {"start": start, "end": end})


def load_industry_capital_flow(
    engine: Engine, start: date, end: date
) -> pd.DataFrame:
    """Load cn_industry_capital_flow_daily (optional capital flow data)."""
    sql = """
    SELECT
        trade_date,
        industry_id,
        industry_name,
        total_amount,
        total_turnover,
        avg_change_pct,
        volume_ratio,
        market_share,
        amount_rank,
        flow_strength_score,
        rotation_speed_score,
        concentration_score,
        capital_flow_score
    FROM cn_industry_capital_flow_daily
    WHERE trade_date BETWEEN :start AND :end
    ORDER BY trade_date, industry_id
    """
    return fetch_df(engine, sql, {"start": start, "end": end})


def load_industry_proxy(
    engine: Engine, start: date, end: date
) -> pd.DataFrame:
    """Load cn_local_industry_proxy_daily for member count and concentration."""
    sql = """
    SELECT
        trade_date,
        industry_id,
        industry_name,
        member_count,
        ret_eqw,
        amount_total,
        top5_concentration
    FROM cn_local_industry_proxy_daily
    WHERE trade_date BETWEEN :start AND :end
    ORDER BY trade_date, industry_id
    """
    return fetch_df(engine, sql, {"start": start, "end": end})


def load_market_pulse(
    engine: Engine, start: date, end: date
) -> pd.DataFrame:
    """Load cn_ga_market_pulse_daily for market-level context."""
    sql = """
    SELECT
        trade_date,
        market_score,
        market_state,
        bullish_industry_ratio,
        neutral_industry_ratio,
        bearish_industry_ratio,
        rotation_speed,
        mainline_stability,
        trend_alignment_avg,
        industry_expansion_breadth,
        top_mainline_count,
        market_phase
    FROM cn_ga_market_pulse_daily
    WHERE trade_date BETWEEN :start AND :end
    ORDER BY trade_date
    """
    return fetch_df(engine, sql, {"start": start, "end": end})


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------


def _compute_lifecycle_state(
    row: dict[str, Any],
    prev_row: dict[str, Any] | None,
    market_pulse_map: dict[date, dict[str, Any]],
) -> tuple[str, float, str, str]:
    """
    Determine lifecycle_state, lifecycle_score, phase_reason, risk_flag
    for a single mainline on a single trade_date.

    Uses only data <= trade_date — no future function.

    Data scale notes (verified from actual DB):
      - radar mainline_score: 0~100 scale (avg=42, max=74)
      - stock mainline_strength_score: 0~1 scale (avg=0.35, max=0.73)
      - stock leader_density: 0~1 scale (avg=0.53)
      - stock trend_alignment: 0~1 scale (avg=0.53)
      - stock breadth_score: 0~1 scale (avg=0.08)
      - stock acceleration_score: 0~1 scale (avg=0.51)
      - stock rank_in_market: 1~331 (avg=110)
      - pulse market_score: 0~100 scale (avg=50)
      - pulse market_state: TREND_WEAK, RANGE, RISK_OFF
      - pulse bullish/bearish_ratio: ALL ZERO (unpopulated)
      - radar breakout_ratio, new_high_ratio, leader_density,
        trend_alignment_score, rotation_rank, strong_stock_count: ALL NULL
      - stock breakout_ratio: ALL ZERO
    """
    # --- Primary strength: mainline_score from radar (0~100 scale) ---
    mainline_score = row.get("mainline_score") or 0.0
    # --- Secondary strength: from stock_mainline_strength (0~1 scale) ---
    stock_strength = row.get("mainline_strength") or 0.0

    # Blend: prefer mainline_score (0~100), fall back to stock_strength * 100
    if mainline_score > 0.0:
        strength = mainline_score  # already 0~100 scale
    elif stock_strength > 0.0:
        strength = stock_strength * 100.0  # normalize 0~1 -> 0~100
    else:
        strength = 0.0

    # Stock strength features (0~1 scale, well-populated)
    stock_leader_density = row.get("strength_leader_density") or 0.0
    stock_trend_alignment = row.get("trend_alignment") or 0.0
    stock_breadth = row.get("breadth_score") or 0.0
    stock_acceleration = row.get("acceleration_score") or 0.0
    stock_rank = row.get("rank_in_market") or 999
    stock_active = row.get("is_active_mainline") or 0

    # Radar features (mostly NULL except mainline_score and leader_count)
    radar_leader_count = row.get("leader_count") or 0

    # Capital concentration (from capital flow or proxy)
    capital_conc = row.get("capital_concentration_score") or 0.0
    member_count = row.get("member_count") or 0

    # Previous day values for delta computation
    prev_strength = prev_row.get("mainline_score") if prev_row else None
    prev_stock_trend = prev_row.get("trend_alignment") if prev_row else None
    prev_stock_leader_density = prev_row.get("strength_leader_density") if prev_row else None
    prev_stock_breadth = prev_row.get("breadth_score") if prev_row else None
    prev_stock_acceleration = prev_row.get("acceleration_score") if prev_row else None
    prev_stock_rank = prev_row.get("rank_in_market") if prev_row else None

    # Market pulse for this date
    trade_date = row.get("trade_date")
    if isinstance(trade_date, str):
        trade_date = datetime.strptime(trade_date, "%Y-%m-%d").date()
    elif isinstance(trade_date, datetime):
        trade_date = trade_date.date()
    pulse = market_pulse_map.get(trade_date, {})
    market_score = pulse.get("market_score") or 0.0  # 0~100 scale
    market_state = pulse.get("market_state") or ""

    # ── Delta computations (safe: only uses prev data) ──────────────
    if prev_strength is not None:
        strength_delta = mainline_score - prev_strength  # 0~100 scale delta
    else:
        strength_delta = 0.0
    trend_alignment_delta = (stock_trend_alignment - prev_stock_trend) if prev_stock_trend is not None else 0.0
    leader_density_delta = (stock_leader_density - prev_stock_leader_density) if prev_stock_leader_density is not None else 0.0
    breadth_delta = (stock_breadth - prev_stock_breadth) if prev_stock_breadth is not None else 0.0
    acceleration_delta = (stock_acceleration - prev_stock_acceleration) if prev_stock_acceleration is not None else 0.0
    rank_improving = (stock_rank < prev_stock_rank) if prev_stock_rank is not None else False
    rank_worsening = (stock_rank > prev_stock_rank) if prev_stock_rank is not None else False

    # ── Feature thresholds ──────────────────────────────────────────
    # Strength on 0~100 scale (radar avg=42, stock avg=35 after *100)
    strength_high = strength >= 55.0    # top ~15% of radar scores
    strength_mid = 25.0 <= strength < 55.0  # middle band
    strength_low = strength < 25.0      # bottom
    strength_rising = strength_delta > 1.0
    strength_falling = strength_delta < -1.0

    # Stock-level features (0~1 scale)
    trend_alignment_high = stock_trend_alignment >= 0.50   # 3422 rows
    trend_alignment_improving = trend_alignment_delta > 0.02
    trend_alignment_falling = trend_alignment_delta < -0.02

    leader_density_high = stock_leader_density >= 0.40     # 7879 rows >= 0.30
    leader_density_falling = leader_density_delta < -0.02
    leader_density_flat = -0.02 <= leader_density_delta <= 0.02

    breadth_high = stock_breadth >= 0.05    # 3371 rows
    breadth_rising = breadth_delta > 0.005
    breadth_falling = breadth_delta < -0.005

    acceleration_high = stock_acceleration >= 0.50  # 4517 rows
    acceleration_rising = acceleration_delta > 0.05
    acceleration_falling = acceleration_delta < -0.05

    # Rank: lower is better (1 = best)
    rank_top10 = stock_rank <= 10
    rank_top30 = stock_rank <= 30    # 1651 rows
    rank_top50 = stock_rank <= 50
    rank_bottom = stock_rank > 200

    capital_conc_high = capital_conc >= 0.40

    # Market pulse: use market_state and market_score (0~100 scale)
    market_risk_off = (market_state == "RISK_OFF") or (market_score < 40.0)
    market_weak = (market_state in ("RISK_OFF", "TREND_WEAK")) or (market_score < 45.0)
    market_normal = not market_risk_off

    # ── State determination (priority order) ────────────────────────
    reasons: list[str] = []
    risk_flag: str = "NONE"

    # TREND_EXPANSION: strong mainline + high trend alignment + top rank + breadth rising
    if (
        strength_high
        and trend_alignment_high
        and rank_top10
        and breadth_rising
        and market_normal
    ):
        state = "TREND_EXPANSION"
        reasons = ["strength_high", "trend_alignment_high", "rank_top10", "breadth_rising"]
        if leader_density_high:
            risk_flag = "CROWDING"
        score = _score_for_state(state, strength, stock_trend_alignment, stock_breadth, stock_acceleration, stock_rank)

    # DIFFUSION: strong mainline + breadth high + acceleration high + leader density stable
    elif (
        strength_high
        and breadth_high
        and acceleration_high
        and (leader_density_flat or leader_density_high)
        and market_normal
    ):
        state = "DIFFUSION"
        reasons = ["strength_high", "breadth_high", "acceleration_high", "leader_density_stable"]
        score = _score_for_state(state, strength, stock_trend_alignment, stock_breadth, stock_acceleration, stock_rank)

    # DIVERGENCE: strong mainline + trend alignment falling + acceleration falling + high leader density
    elif (
        strength_high
        and trend_alignment_falling
        and acceleration_falling
        and leader_density_high
        and capital_conc_high
    ):
        state = "DIVERGENCE"
        reasons = ["strength_high", "trend_alignment_falling", "acceleration_falling", "leader_density_high", "capital_conc_high"]
        risk_flag = "CROWDING"
        score = _score_for_state(state, strength, stock_trend_alignment, stock_breadth, stock_acceleration, stock_rank)

    # TOP_DECAY: strength falling + breadth falling + rank worsening
    elif (
        strength_falling
        and breadth_falling
        and (rank_worsening or rank_bottom)
    ):
        state = "TOP_DECAY"
        reasons = ["strength_falling", "breadth_falling", "rank_worsening"]
        risk_flag = "TOP_DECAY"
        score = _score_for_state(state, strength, stock_trend_alignment, stock_breadth, stock_acceleration, stock_rank)

    # RISK_OFF: market risk-off + weak strength
    elif market_risk_off and (strength_low or strength_falling):
        state = "RISK_OFF"
        reasons = ["market_risk_off", "strength_low_or_falling"]
        risk_flag = "MARKET_RISK_OFF"
        score = _score_for_state(state, strength, stock_trend_alignment, stock_breadth, stock_acceleration, stock_rank)

    # BOTTOM_REPAIR: mid strength + trend alignment improving + breadth rising + rank improving
    elif (
        strength_mid
        and trend_alignment_improving
        and breadth_rising
        and (rank_improving or rank_top50)
        and market_normal
    ):
        state = "BOTTOM_REPAIR"
        reasons = ["strength_mid", "trend_alignment_improving", "breadth_rising", "rank_improving"]
        score = _score_for_state(state, strength, stock_trend_alignment, stock_breadth, stock_acceleration, stock_rank)

    # Catch-all: assign a default state based on strength + market conditions
    # This ensures most rows get a non-UNKNOWN state
    elif strength_high and market_normal:
        state = "DIFFUSION"
        reasons = ["strength_high_default"]
        score = _score_for_state(state, strength, stock_trend_alignment, stock_breadth, stock_acceleration, stock_rank)

    elif strength_mid and market_normal:
        state = "BOTTOM_REPAIR"
        reasons = ["strength_mid_default"]
        score = _score_for_state(state, strength, stock_trend_alignment, stock_breadth, stock_acceleration, stock_rank)

    elif strength_low and market_risk_off:
        state = "RISK_OFF"
        reasons = ["strength_low_market_risk_off"]
        risk_flag = "MARKET_RISK_OFF"
        score = _score_for_state(state, strength, stock_trend_alignment, stock_breadth, stock_acceleration, stock_rank)

    elif strength_low and not market_risk_off:
        state = "BOTTOM_REPAIR"
        reasons = ["strength_low_recovering"]
        score = _score_for_state(state, strength, stock_trend_alignment, stock_breadth, stock_acceleration, stock_rank)

    else:
        state = "UNKNOWN"
        reasons = ["insufficient_data_or_ambiguous"]
        risk_flag = "DATA_INSUFFICIENT"
        score = 0.50

    phase_reason = ";".join(reasons)
    return state, round(score, 4), phase_reason, risk_flag


def _score_for_state(
    state: str,
    strength: float,
    trend_alignment: float,
    breadth_score: float,
    acceleration_score: float,
    rank_in_market: int,
) -> float:
    """
    Compute lifecycle_score in [0, 1] based on state and feature values.

    Score ranges per state:
      TREND_EXPANSION: 0.85–1.00
      DIFFUSION:       0.70–0.85
      BOTTOM_REPAIR:   0.55–0.75
      DIVERGENCE:      0.35–0.60
      TOP_DECAY:       0.10–0.35
      RISK_OFF:        0.00–0.25
      UNKNOWN:         0.50
    """
    # Normalize features to [0,1] contribution
    s_norm = min(strength / 100.0, 1.0)
    t_norm = min(trend_alignment, 1.0)
    b_norm = min(breadth_score * 5.0, 1.0)  # breadth_score avg=0.08, so *5 brings to ~0.4
    a_norm = min(acceleration_score, 1.0)
    r_norm = 1.0 - min(rank_in_market / 100.0, 1.0)  # rank 1~331, lower = better

    if state == "TREND_EXPANSION":
        base = 0.85
        bonus = (s_norm * 0.05 + t_norm * 0.03 + b_norm * 0.04 + a_norm * 0.03)
        return min(base + bonus, 1.0)
    elif state == "DIFFUSION":
        base = 0.70
        bonus = (s_norm * 0.04 + b_norm * 0.05 + a_norm * 0.06)
        return min(base + bonus, 0.85)
    elif state == "BOTTOM_REPAIR":
        base = 0.55
        bonus = (t_norm * 0.06 + b_norm * 0.07 + r_norm * 0.07)
        return min(base + bonus, 0.75)
    elif state == "DIVERGENCE":
        base = 0.35
        bonus = (s_norm * 0.10 + t_norm * 0.10 + r_norm * 0.05)
        return min(base + bonus, 0.60)
    elif state == "TOP_DECAY":
        base = 0.10
        bonus = (s_norm * 0.10 + t_norm * 0.10 + r_norm * 0.05)
        return min(base + bonus, 0.35)
    elif state == "RISK_OFF":
        return round(0.10 + s_norm * 0.15, 4)
    else:  # UNKNOWN
        return 0.50


def _compute_capital_concentration_score(
    row: dict[str, Any],
) -> float:
    """
    Compute capital_concentration_score from industry capital flow data.
    Higher = more concentrated capital in top mainlines.
    Uses concentration_score and market_share from cn_industry_capital_flow_daily.
    Falls back to 0.0 if no capital flow data available.
    """
    concentration_score = row.get("concentration_score") or 0.0
    market_share = row.get("market_share") or 0.0
    # Blend raw concentration with market share
    score = concentration_score * 0.6 + min(market_share * 5.0, 1.0) * 0.4
    return round(min(score, 1.0), 4)


def _compute_trend_alignment_score(
    row: dict[str, Any],
) -> float:
    """
    Compute trend_alignment_score from radar data.
    Higher = more stocks in the industry aligned with uptrend.
    """
    raw = row.get("trend_alignment_score") or 0.0
    return round(min(raw, 1.0), 4)


# ---------------------------------------------------------------------------
# Main build logic
# ---------------------------------------------------------------------------


def run_build(args: argparse.Namespace) -> pd.DataFrame:
    """
    Execute the mainline lifecycle build for the given date range.
    Returns the computed DataFrame (all dates).
    """
    verbose = args.verbose
    dry_run = args.dry_run
    replace = args.replace
    min_member = args.min_member_count

    # Resolve date range
    start = datetime.strptime(args.start, "%Y-%m-%d").date() if "-" in args.start else datetime.strptime(args.start, "%Y%m%d").date()
    end = datetime.strptime(args.end, "%Y-%m-%d").date() if args.end and "-" in args.end else (
        datetime.strptime(args.end, "%Y%m%d").date() if args.end else date.today()
    )
    if start > end:
        print(f"ERROR: start {start} > end {end}")
        sys.exit(1)

    # Need lookback for delta computation
    lookback_start = start - timedelta(days=5)

    db_password = args.db_password or os.getenv("ASHARE_MYSQL_PASSWORD", "sec_Bobo123")
    engine = build_engine(args.db_host, args.db_port, args.db_user, db_password, args.db_name)

    if verbose:
        print(f"[INFO] Date range: {start} ~ {end}")
        print(f"[INFO] Lookback start: {lookback_start}")
        print(f"[INFO] Database: {args.db_name}")
        print(f"[INFO] Dry-run: {dry_run}, Replace: {replace}")

    # ── Load input data ─────────────────────────────────────────────
    if verbose:
        print("[INFO] Loading cn_ga_mainline_radar_daily (PRIMARY source) ...")
    radar_df = load_ga_mainline_radar(engine, lookback_start, end)
    if radar_df.empty:
        print("WARNING: cn_ga_mainline_radar_daily is empty for the range")
        return pd.DataFrame()
    if verbose:
        print(f"[INFO]   -> {len(radar_df)} rows loaded")

    if verbose:
        print("[INFO] Loading cn_stock_mainline_strength_daily (secondary strength) ...")
    stock_strength_df = load_stock_mainline_strength(engine, lookback_start, end)
    if verbose:
        print(f"[INFO]   -> {len(stock_strength_df)} rows loaded")

    if verbose:
        print("[INFO] Loading cn_industry_capital_flow_daily ...")
    capital_flow_df = load_industry_capital_flow(engine, lookback_start, end)
    if verbose:
        print(f"[INFO]   -> {len(capital_flow_df)} rows loaded")

    if verbose:
        print("[INFO] Loading cn_local_industry_proxy_daily ...")
    proxy_df = load_industry_proxy(engine, lookback_start, end)
    if verbose:
        print(f"[INFO]   -> {len(proxy_df)} rows loaded")

    if verbose:
        print("[INFO] Loading cn_ga_market_pulse_daily ...")
    pulse_df = load_market_pulse(engine, lookback_start, end)
    if verbose:
        print(f"[INFO]   -> {len(pulse_df)} rows loaded")

    # ── Merge data ──────────────────────────────────────────────────
    # Start with radar as base
    df = radar_df.copy()

    # Merge stock_mainline_strength on trade_date + mainline_id (secondary strength)
    if not stock_strength_df.empty:
        stock_str_renamed = stock_strength_df.rename(columns={
            "industry_id": "mainline_id",
            "industry_name": "mainline_name_strength",
            "mainline_strength_score": "mainline_strength",
        })
        df = df.merge(
            stock_str_renamed[[
                "trade_date", "mainline_id", "mainline_strength",
                "strength_leader_density", "strength_breakout_ratio",
                "trend_alignment", "breadth_score", "acceleration_score",
                "lifecycle_bonus", "rank_in_market", "is_active_mainline",
            ]],
            on=["trade_date", "mainline_id"],
            how="left",
            suffixes=("", "_strength"),
        )
    else:
        df["mainline_strength"] = np.nan
        df["strength_leader_density"] = np.nan
        df["strength_breakout_ratio"] = np.nan
        df["trend_alignment"] = np.nan
        df["breadth_score"] = np.nan
        df["acceleration_score"] = np.nan
        df["lifecycle_bonus"] = np.nan
        df["rank_in_market"] = np.nan
        df["is_active_mainline"] = 0

    # Merge industry_capital_flow on trade_date + mainline_id
    if not capital_flow_df.empty:
        cf_renamed = capital_flow_df.rename(columns={
            "industry_id": "mainline_id",
            "concentration_score": "concentration_score",
        })
        df = df.merge(
            cf_renamed[[
                "trade_date", "mainline_id", "total_amount", "total_turnover",
                "avg_change_pct", "volume_ratio", "market_share",
                "amount_rank", "flow_strength_score", "rotation_speed_score",
                "concentration_score", "capital_flow_score",
            ]],
            on=["trade_date", "mainline_id"],
            how="left",
            suffixes=("", "_cf"),
        )
    else:
        if verbose:
            print("[INFO] cn_industry_capital_flow_daily is empty, using defaults")
        df["total_amount"] = 0.0
        df["total_turnover"] = 0.0
        df["avg_change_pct"] = 0.0
        df["volume_ratio"] = 0.0
        df["market_share"] = 0.0
        df["amount_rank"] = 0
        df["flow_strength_score"] = 0.0
        df["rotation_speed_score"] = 0.0
        df["concentration_score"] = 0.0
        df["capital_flow_score"] = 0.0

    # Merge proxy for member_count
    if not proxy_df.empty:
        proxy_renamed = proxy_df.rename(columns={
            "industry_id": "mainline_id",
        })
        df = df.merge(
            proxy_renamed[["trade_date", "mainline_id", "member_count", "top5_concentration"]],
            on=["trade_date", "mainline_id"],
            how="left",
            suffixes=("", "_proxy"),
        )
    else:
        df["member_count"] = 0
        df["top5_concentration"] = 0.0

    # Filter by min member count
    df = df[df["member_count"].fillna(0) >= min_member].copy()

    # Build market pulse map
    market_pulse_map: dict[date, dict[str, Any]] = {}
    if not pulse_df.empty:
        for _, p_row in pulse_df.iterrows():
            td = p_row["trade_date"]
            if isinstance(td, str):
                td = datetime.strptime(td, "%Y-%m-%d").date()
            elif isinstance(td, datetime):
                td = td.date()
            market_pulse_map[td] = {
                "bullish_industry_ratio": p_row.get("bullish_industry_ratio"),
                "bearish_industry_ratio": p_row.get("bearish_industry_ratio"),
                "market_score": p_row.get("market_score"),
                "market_state": p_row.get("market_state"),
                "market_phase": p_row.get("market_phase"),
            }

    # ── Sort for sequential processing ──────────────────────────────
    df = df.sort_values(["mainline_id", "trade_date"]).reset_index(drop=True)

    # ── Compute derived scores ──────────────────────────────────────
    df["capital_concentration_score"] = df.apply(_compute_capital_concentration_score, axis=1)
    df["trend_alignment_score"] = df.apply(_compute_trend_alignment_score, axis=1)

    # ── Compute lifecycle state per row ─────────────────────────────
    results: list[dict[str, Any]] = []
    prev_by_mainline: dict[str, dict[str, Any]] = {}

    for idx, row in df.iterrows():
        mainline_id = row["mainline_id"]
        trade_date_val = row["trade_date"]
        if isinstance(trade_date_val, str):
            trade_date_val = datetime.strptime(trade_date_val, "%Y-%m-%d").date()
        elif isinstance(trade_date_val, datetime):
            trade_date_val = trade_date_val.date()

        # Skip lookback dates for final output
        if trade_date_val < start:
            # Still update prev state
            prev_by_mainline[mainline_id] = row.to_dict()
            continue

        prev_row = prev_by_mainline.get(mainline_id)
        row_dict = row.to_dict()
        row_dict["trade_date"] = trade_date_val

        state, score, reason, risk = _compute_lifecycle_state(
            row_dict, prev_row, market_pulse_map
        )

        # Compute rotation_rank per date
        # (will be recomputed per date group below)
        results.append({
            "trade_date": trade_date_val,
            "mainline_id": mainline_id,
            "mainline_name": row.get("mainline_name", ""),
            "mainline_score": row.get("mainline_score"),
            "mainline_strength": row.get("mainline_strength"),
            "capital_concentration_score": row_dict.get("capital_concentration_score"),
            "trend_alignment_score": row_dict.get("trend_alignment_score"),
            "breakout_ratio": row.get("breakout_ratio"),
            "new_high_ratio": row.get("new_high_ratio"),
            "leader_density": row.get("leader_density"),
            "rotation_rank": row.get("rotation_rank"),
            "lifecycle_state": state,
            "lifecycle_score": score,
            "phase_reason": reason,
            "risk_flag": risk,
            "member_count": row.get("member_count", 0),
            "strong_stock_count": row.get("strong_stock_count", 0),
        })

        # Update prev state
        prev_by_mainline[mainline_id] = row_dict

    if not results:
        print("WARNING: No results computed")
        return pd.DataFrame()

    result_df = pd.DataFrame(results)

    # ── Recompute rotation_rank per date based on lifecycle_score ───
    result_df["rotation_rank"] = (
        result_df.groupby("trade_date")["lifecycle_score"]
        .rank(ascending=False, method="min")
        .fillna(999)
        .astype(int)
    )

    # ── Filter to requested date range only ─────────────────────────
    result_df = result_df[result_df["trade_date"].between(start, end)].copy()

    if verbose:
        state_counts = result_df["lifecycle_state"].value_counts().to_dict()
        print(f"[INFO] Computed {len(result_df)} rows")
        print(f"[INFO] State distribution: {state_counts}")

    return result_df


# ---------------------------------------------------------------------------
# DB write
# ---------------------------------------------------------------------------


def write_to_db(
    engine: Engine,
    df: pd.DataFrame,
    db_name: str,
    replace: bool,
    dry_run: bool,
    verbose: bool,
) -> int:
    """Write computed DataFrame to cn_mainline_lifecycle_daily. Returns row count."""
    if df.empty:
        print("WARNING: No data to write")
        return 0

    if dry_run:
        print(f"[DRY-RUN] Would write {len(df)} rows to cn_mainline_lifecycle_daily")
        return len(df)

    # Ensure table exists
    if not table_exists(engine, db_name, "cn_mainline_lifecycle_daily"):
        print("ERROR: cn_mainline_lifecycle_daily table does not exist. Run DDL first.")
        return 0

    # If replace, delete existing rows in date range
    if replace:
        min_date = df["trade_date"].min()
        max_date = df["trade_date"].max()
        if isinstance(min_date, str):
            min_date = datetime.strptime(min_date, "%Y-%m-%d").date()
        elif isinstance(min_date, datetime):
            min_date = min_date.date()
        if isinstance(max_date, str):
            max_date = datetime.strptime(max_date, "%Y-%m-%d").date()
        elif isinstance(max_date, datetime):
            max_date = max_date.date()
        del_sql = """
        DELETE FROM cn_mainline_lifecycle_daily
        WHERE trade_date BETWEEN :start AND :end
        """
        with engine.begin() as conn:
            deleted = conn.execute(text(del_sql), {"start": min_date, "end": max_date}).rowcount
        if verbose:
            print(f"[INFO] Deleted {deleted} existing rows in [{min_date}, {max_date}]")

    # Prepare rows for upsert
    columns = [
        "trade_date", "mainline_id", "mainline_name", "mainline_strength",
        "capital_concentration_score", "trend_alignment_score",
        "breakout_ratio", "new_high_ratio", "leader_density",
        "rotation_rank", "lifecycle_state", "lifecycle_score",
        "phase_reason", "risk_flag",
    ]
    upsert_sql = """
    INSERT INTO cn_mainline_lifecycle_daily (
        trade_date, mainline_id, mainline_name, mainline_strength,
        capital_concentration_score, trend_alignment_score,
        breakout_ratio, new_high_ratio, leader_density,
        rotation_rank, lifecycle_state, lifecycle_score,
        phase_reason, risk_flag
    ) VALUES (
        :trade_date, :mainline_id, :mainline_name, :mainline_strength,
        :capital_concentration_score, :trend_alignment_score,
        :breakout_ratio, :new_high_ratio, :leader_density,
        :rotation_rank, :lifecycle_state, :lifecycle_score,
        :phase_reason, :risk_flag
    )
    ON DUPLICATE KEY UPDATE
        mainline_name = VALUES(mainline_name),
        mainline_strength = VALUES(mainline_strength),
        capital_concentration_score = VALUES(capital_concentration_score),
        trend_alignment_score = VALUES(trend_alignment_score),
        breakout_ratio = VALUES(breakout_ratio),
        new_high_ratio = VALUES(new_high_ratio),
        leader_density = VALUES(leader_density),
        rotation_rank = VALUES(rotation_rank),
        lifecycle_state = VALUES(lifecycle_state),
        lifecycle_score = VALUES(lifecycle_score),
        phase_reason = VALUES(phase_reason),
        risk_flag = VALUES(risk_flag),
        updated_at = CURRENT_TIMESTAMP
    """

    # Convert dates to string for SQL
    write_df = df[columns].copy()
    write_df["trade_date"] = write_df["trade_date"].apply(
        lambda d: d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d)
    )

    rows = write_df.astype(object).where(pd.notna(write_df), None).to_dict(orient="records")

    total = 0
    batch_size = 4000
    with engine.begin() as conn:
        for i in range(0, len(rows), batch_size):
            batch = rows[i : i + batch_size]
            conn.execute(text(upsert_sql), batch)
            total += len(batch)
        if verbose:
            print(f"[INFO] Wrote {total} rows to cn_mainline_lifecycle_daily")

    return total


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def generate_reports(
    df: pd.DataFrame,
    start: date,
    end: date,
    output_dir: str | None,
) -> tuple[Path, Path]:
    """Generate summary CSV and Markdown reports. Returns (csv_path, md_path)."""
    report_dir = Path(output_dir) if output_dir else REPORT_DIR
    report_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    start_str = start.strftime("%Y%m%d")
    end_str = end.strftime("%Y%m%d")
    base_name = f"mainline_lifecycle_summary_{start_str}_{end_str}_{timestamp}"

    csv_path = report_dir / f"{base_name}.csv"
    md_path = report_dir / f"{base_name}.md"

    # ── Summary statistics ──────────────────────────────────────────
    if df.empty:
        summary_rows = [{
            "trade_date": "N/A",
            "total_mainlines": 0,
            "state_counts": "N/A",
            "top_trend_expansion": "N/A",
            "top_top_decay": "N/A",
            "risk_off_flag": "N/A",
            "row_count_written": 0,
        }]
    else:
        summary_rows = []
        for trade_date_val, group in df.groupby("trade_date"):
            td = trade_date_val
            if hasattr(td, "strftime"):
                td_str = td.strftime("%Y-%m-%d")
            else:
                td_str = str(td)

            state_counts = group["lifecycle_state"].value_counts().to_dict()

            # Top TREND_EXPANSION mainlines
            te_group = group[group["lifecycle_state"] == "TREND_EXPANSION"].nsmallest(5, "rotation_rank")
            te_names = "; ".join(
                f"{r['mainline_name']}(rank={int(r['rotation_rank'])})"
                for _, r in te_group.iterrows()
            ) if not te_group.empty else "N/A"

            # Top TOP_DECAY mainlines
            td_group = group[group["lifecycle_state"] == "TOP_DECAY"].nsmallest(5, "rotation_rank")
            td_names = "; ".join(
                f"{r['mainline_name']}(rank={int(r['rotation_rank'])})"
                for _, r in td_group.iterrows()
            ) if not td_group.empty else "N/A"

            risk_off = "YES" if (group["risk_flag"] == "MARKET_RISK_OFF").any() else "NO"

            summary_rows.append({
                "trade_date": td_str,
                "total_mainlines": len(group),
                "state_counts": str(state_counts),
                "top_trend_expansion": te_names,
                "top_top_decay": td_names,
                "risk_off_flag": risk_off,
                "row_count_written": len(group),
            })

    summary_df = pd.DataFrame(summary_rows)

    # ── Write CSV ───────────────────────────────────────────────────
    summary_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"[REPORT] CSV -> {csv_path}")

    # ── Write Markdown ──────────────────────────────────────────────
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Mainline Lifecycle Summary\n\n")
        f.write(f"**Date Range:** {start_str} ~ {end_str}\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Total Rows:** {len(df)}\n\n")
        f.write(f"---\n\n")

        for _, srow in summary_df.iterrows():
            f.write(f"## {srow['trade_date']}\n\n")
            f.write(f"- Total Mainlines: {srow['total_mainlines']}\n")
            f.write(f"- State Distribution: {srow['state_counts']}\n")
            f.write(f"- Top TREND_EXPANSION: {srow['top_trend_expansion']}\n")
            f.write(f"- Top TOP_DECAY: {srow['top_top_decay']}\n")
            f.write(f"- Risk-Off Flag: {srow['risk_off_flag']}\n")
            f.write(f"- Rows Written: {srow['row_count_written']}\n\n")
            f.write(f"---\n\n")

        # ── Overall state distribution ──────────────────────────────
        f.write(f"## Overall State Distribution\n\n")
        overall_counts = df["lifecycle_state"].value_counts()
        for state_name, count in overall_counts.items():
            pct = count / len(df) * 100
            f.write(f"- **{state_name}**: {count} ({pct:.1f}%)\n")
        f.write("\n")

        # ── Risk flag summary ───────────────────────────────────────
        f.write(f"## Risk Flag Summary\n\n")
        risk_counts = df["risk_flag"].value_counts()
        for flag_name, count in risk_counts.items():
            f.write(f"- **{flag_name}**: {count}\n")
        f.write("\n")

        # ── Lifecycle State Definitions ─────────────────────────────
        f.write(f"## Lifecycle State Definitions\n\n")
        f.write(f"### BOTTOM_REPAIR\n")
        f.write(f"主线从低位修复，突破比例和趋势一致性从低位改善。\n\n")
        f.write(f"### TREND_EXPANSION\n")
        f.write(f"主线进入主升阶段，资金、趋势、突破、新高、龙头密度共振。\n\n")
        f.write(f"### DIFFUSION\n")
        f.write(f"主线从少数龙头扩散到产业链更多环节，强股数量扩大。\n\n")
        f.write(f"### DIVERGENCE\n")
        f.write(f"主线仍强，但内部结构开始分歧，龙头拥挤或扩散失败。\n\n")
        f.write(f"### TOP_DECAY\n")
        f.write(f"主线热度衰退，突破比例、新高比例、rotation rank 恶化。\n\n")
        f.write(f"### RISK_OFF\n")
        f.write(f"市场整体风险收缩，主线评分失效或多数行业同步转弱。\n\n")
        f.write(f"### UNKNOWN\n")
        f.write(f"数据不足或无法判断。\n\n")

    print(f"[REPORT] MD  -> {md_path}")

    return csv_path, md_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    print("=" * 60)
    print("  GrowthAlpha V8 — P2 Mainline Lifecycle Engine")
    print("=" * 60)
    print(f"  Start: {args.start}")
    print(f"  End:   {args.end or 'today'}")
    print(f"  DB:    {args.db_name}")
    print(f"  Dry-run: {args.dry_run}")
    print(f"  Replace: {args.replace}")
    print()

    # Resolve password
    db_password = args.db_password or os.getenv("ASHARE_MYSQL_PASSWORD", "sec_Bobo123")
    engine = build_engine(args.db_host, args.db_port, args.db_user, db_password, args.db_name)

    # Ensure DDL
    from pathlib import Path as _Path
    ddl_path = _Path(__file__).resolve().parents[1] / "sql" / "ddl" / "ga_p2_mainline_lifecycle_schema.sql"
    if ddl_path.exists():
        ddl_sql = ddl_path.read_text(encoding="utf-8")
        stmts = [s.strip() for s in ddl_sql.split(";") if s.strip()]
        with engine.begin() as conn:
            for stmt in stmts:
                if "CREATE TABLE" in stmt:
                    conn.execute(text(stmt))
        print(f"[DDL] Ensured cn_mainline_lifecycle_daily table exists")
    else:
        print(f"[WARNING] DDL file not found: {ddl_path}")

    # Run build
    result_df = run_build(args)

    if result_df.empty:
        print("No data computed. Exiting.")
        sys.exit(0)

    # Write to DB
    written = write_to_db(engine, result_df, args.db_name, args.replace, args.dry_run, args.verbose)

    # Generate reports
    start = datetime.strptime(args.start, "%Y-%m-%d").date() if "-" in args.start else datetime.strptime(args.start, "%Y%m%d").date()
    end = datetime.strptime(args.end, "%Y-%m-%d").date() if args.end and "-" in args.end else (
        datetime.strptime(args.end, "%Y%m%d").date() if args.end else date.today()
    )
    csv_path, md_path = generate_reports(result_df, start, end, args.output_dir)

    print()
    print("=" * 60)
    print(f"  Build Complete")
    print(f"  Rows computed: {len(result_df)}")
    print(f"  Rows written:  {written}")
    print(f"  CSV report:    {csv_path}")
    print(f"  MD  report:    {md_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
