"""
scripts/build_unified_alpha_score_daily.py
============================================
GrowthAlpha V8 — P3 Unified Alpha Engine Builder.

Computes cn_unified_alpha_score_daily by combining 8 alpha factors:
  1. quality_score              — Fundamental quality (from cn_stock_quality_score_daily)
  2. growth_acceleration_score  — Growth acceleration (from cn_stock_quality_score_daily)
  3. mainline_strength_score    — Mainline (industry) strength (from cn_stock_mainline_strength_daily)
  4. capital_concentration_score — Capital concentration (from cn_industry_capital_flow_daily)
  5. leader_dominance_score     — Leader dominance (from cn_ga_mainline_radar_daily)
  6. trend_quality_score        — Trend quality (from cn_stock_daily_price)
  7. lifecycle_position_score   — Lifecycle position (from cn_mainline_lifecycle_daily)
  8. risk_crowding_score        — Risk / crowding filter (from cn_ga_market_pulse_daily + cn_ga_mainline_radar_daily)

Final score = weighted average of 8 factors, with risk_crowding_score inverted
(risk_adjusted = 1 - risk_crowding_score), clipped to [0, 1].
Cross-sectional percentile bucket assignment per trade_date.

═══ RISK/CROWDING INVERSION RULE ════════════════════════════════════════════
The risk_crowding_score (Factor 8) is a RISK factor: higher values indicate
HIGHER risk (worse for the stock). Therefore, it MUST be inverted before
contributing to the final score:

    risk_adjusted_score = 1 - risk_crowding_score

This inversion is applied in _compute_final_score() at the point of weighted
aggregation. The inversion covers ALL code paths through _compute_risk_crowding_score(),
including all fallback values (0.5 → 1.0 - 0.5 = 0.5, which is neutral).

Direction proof:
  - _compute_risk_crowding_score() returns HIGHER values for:
    * bearish market (bearish_ratio > 0.5 → score=0.8)
    * crowded mainlines (leader_density > 0.7 → score=0.8)
    * RISK_OFF/CRISIS market phases (bonus +0.2)
  - _compute_final_score() inverts via: score = 1.0 - score (line ~596)
  - Result: high risk → low risk_adjusted_score → lower final_score → correct

P0 Schema Alignment:
  - cn_ga_stock_role_map_daily  (not stock_role_map)
  - cn_ga_market_pulse_daily    (not market_pulse_daily)
  - cn_stock_mainline_strength_daily  (industry-level, no symbol column)
  - cn_local_industry_map_hist.out_date  (not end_date)
  - cn_stock_daily_price     (UPPERCASE columns: SYMBOL, TRADE_DATE, CLOSE, PRE_CLOSE, CHG_PCT, AMOUNT, TURNOVER_RATE)

Usage:
  python scripts/build_unified_alpha_score_daily.py --start 2026-03-30 --end 2026-03-30 --db-name cn_market_red --replace --verbose
  python scripts/build_unified_alpha_score_daily.py --start 2026-01-01 --end 2026-03-30 --db-name cn_market_red --replace
  python scripts/build_unified_alpha_score_daily.py --start 2026-03-30 --end 2026-03-30 --db-name cn_market_red --dry-run --verbose
"""

from __future__ import annotations

import argparse
import csv
import json
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

# Base weights for 8 alpha factors (must sum to 1.0)
FACTOR_WEIGHTS: dict[str, float] = {
    "quality_score": 0.10,
    "growth_acceleration_score": 0.10,
    "mainline_strength_score": 0.20,
    "capital_concentration_score": 0.15,
    "leader_dominance_score": 0.15,
    "trend_quality_score": 0.15,
    "lifecycle_position_score": 0.10,
    "risk_crowding_score": 0.05,
}

FACTOR_NAMES = list(FACTOR_WEIGHTS.keys())

ALPHA_BUCKETS = [
    "TOP_1",
    "TOP_5",
    "TOP_10",
    "TOP_20",
    "WATCH",
    "NEUTRAL",
    "AVOID",
]

# Bucket percentile thresholds (cross-sectional)
BUCKET_THRESHOLDS: dict[str, float] = {
    "TOP_1": 0.99,
    "TOP_5": 0.95,
    "TOP_10": 0.90,
    "TOP_20": 0.80,
    "WATCH": 0.60,
    "NEUTRAL": 0.30,
    "AVOID": 0.00,
}

REPORT_DIR = Path("reports") / "unified_alpha"


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _fmt_seconds(seconds: float) -> str:
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds}s"
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {seconds}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m {seconds}s"


def _progress_line(label: str, current: int, total: int, started_at: float, extra: str = "") -> None:
    if total <= 0:
        return
    elapsed = time.time() - started_at
    pct = current * 100 // total
    eta = (elapsed / current * (total - current)) if current > 0 else 0.0
    suffix = f" | {extra}" if extra else ""
    print(
        f"[{_ts()}]   {label}: {current:,}/{total:,} ({pct}%) "
        f"elapsed={_fmt_seconds(elapsed)} eta={_fmt_seconds(eta)}{suffix}",
        flush=True,
    )


def _timed_load(label: str, loader, *args, **kwargs) -> pd.DataFrame:
    t0 = time.time()
    print(f"[{_ts()}]   Loading {label} ...", flush=True)
    df = loader(*args, **kwargs)
    print(f"[{_ts()}]   {label:<30} {len(df):>12,} rows ({_fmt_seconds(time.time() - t0)})", flush=True)
    return df

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="GrowthAlpha V8 — P3 Build cn_unified_alpha_score_daily"
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
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    parser.add_argument(
        "--chunk-months",
        type=int,
        default=3,
        help="Process output date range in chunks to avoid loading full source tables at once",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Skip report generation for large chunked builds",
    )
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
# Data loading helpers
# ---------------------------------------------------------------------------


def _safe_float(val: Any, default: float = 0.5) -> float:
    """Safely convert value to float, returning default if None/NaN."""
    if val is None:
        return default
    try:
        v = float(val)
        if np.isnan(v) or np.isinf(v):
            return default
        return v
    except (ValueError, TypeError):
        return default


def _clip_score(val: float) -> float:
    """Clip score to [0, 1]."""
    return max(0.0, min(1.0, val))


def load_stock_quality_scores(
    engine: Engine, start: date, end: date
) -> pd.DataFrame:
    """Load cn_stock_quality_score_daily for the date range."""
    sql = """
        SELECT
            trade_date,
            symbol,
            quality_score,
            growth_acceleration_score,
            cashflow_score,
            debt_control_score,
            margin_stability_score,
            profitability_score,
            fundamental_risk_flag
        FROM cn_stock_quality_score_daily
        WHERE trade_date BETWEEN :start AND :end
    """
    df = fetch_df(engine, sql, {"start": start, "end": end})
    if df.empty:
        return pd.DataFrame()
    score_cols = [
        "quality_score", "growth_acceleration_score",
        "cashflow_score", "debt_control_score",
        "margin_stability_score", "profitability_score",
    ]
    for col in score_cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: _safe_float(x, 0.5))
    return df


def load_mainline_strength(engine: Engine, start: date, end: date) -> pd.DataFrame:
    """Load cn_stock_mainline_strength_daily for the date range (industry-level, no symbol)."""
    sql = """
        SELECT
            trade_date,
            industry_name,
            mainline_strength_score
        FROM cn_stock_mainline_strength_daily
        WHERE trade_date BETWEEN :start AND :end
    """
    df = fetch_df(engine, sql, {"start": start, "end": end})
    if not df.empty:
        df["mainline_strength_score"] = df["mainline_strength_score"].apply(lambda x: _safe_float(x, 0.5))
    return df


def load_ga_mainline_radar(engine: Engine, start: date, end: date) -> pd.DataFrame:
    """Load cn_ga_mainline_radar_daily for leader dominance data."""
    sql = """
        SELECT
            trade_date,
            mainline_id,
            mainline_name,
            leader_density,
            new_high_ratio,
            breakout_ratio,
            rotation_rank,
            trend_alignment_score
        FROM cn_ga_mainline_radar_daily
        WHERE trade_date BETWEEN :start AND :end
    """
    df = fetch_df(engine, sql, {"start": start, "end": end})
    if not df.empty:
        for col in ["leader_density", "new_high_ratio", "breakout_ratio", "trend_alignment_score"]:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: _safe_float(x, 0.5))
    return df


def load_industry_capital_flow(engine: Engine, start: date, end: date) -> pd.DataFrame:
    """Load cn_industry_capital_flow_daily for capital concentration."""
    sql = """
        SELECT
            trade_date,
            industry_id,
            concentration_score
        FROM cn_industry_capital_flow_daily
        WHERE trade_date BETWEEN :start AND :end
    """
    df = fetch_df(engine, sql, {"start": start, "end": end})
    if not df.empty:
        if "concentration_score" in df.columns:
            df["concentration_score"] = df["concentration_score"].apply(
                lambda x: _safe_float(x, 0.5)
            )
    return df


def load_ga_stock_role_map(engine: Engine, start: date, end: date) -> pd.DataFrame:
    """Load cn_ga_stock_role_map_daily to map symbols to mainline_id and stock_role."""
    sql = """
        SELECT
            trade_date,
            symbol,
            mainline_id,
            stock_role
        FROM cn_ga_stock_role_map_daily
        WHERE trade_date BETWEEN :start AND :end
    """
    return fetch_df(engine, sql, {"start": start, "end": end})


def load_stock_daily_price(engine: Engine, start: date, end: date) -> pd.DataFrame:
    """Load cn_stock_daily_price for trend quality computation (UPPERCASE columns per P0)."""
    sql = """
        SELECT
            SYMBOL,
            TRADE_DATE,
            CLOSE,
            PRE_CLOSE,
            CHG_PCT,
            AMOUNT,
            TURNOVER_RATE
        FROM cn_stock_daily_price
        WHERE TRADE_DATE BETWEEN :start AND :end
    """
    df = fetch_df(engine, sql, {"start": start, "end": end})
    if not df.empty:
        for col in ["CLOSE", "PRE_CLOSE", "CHG_PCT", "AMOUNT", "TURNOVER_RATE"]:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: _safe_float(x, 0.0))
    return df


def load_mainline_lifecycle(engine: Engine, start: date, end: date) -> pd.DataFrame:
    """Load cn_mainline_lifecycle_daily for lifecycle position."""
    sql = """
        SELECT
            trade_date,
            mainline_id,
            mainline_name,
            lifecycle_state,
            lifecycle_score,
            risk_flag
        FROM cn_mainline_lifecycle_daily
        WHERE trade_date BETWEEN :start AND :end
    """
    df = fetch_df(engine, sql, {"start": start, "end": end})
    if not df.empty:
        if "lifecycle_score" in df.columns:
            df["lifecycle_score"] = df["lifecycle_score"].apply(lambda x: _safe_float(x, 0.5))
    return df


def load_ga_market_pulse(engine: Engine, start: date, end: date) -> pd.DataFrame:
    """Load cn_ga_market_pulse_daily for risk/crowding context."""
    sql = """
        SELECT
            trade_date,
            bullish_industry_ratio,
            bearish_industry_ratio,
            market_phase
        FROM cn_ga_market_pulse_daily
        WHERE trade_date BETWEEN :start AND :end
    """
    return fetch_df(engine, sql, {"start": start, "end": end})


def load_industry_map(engine: Engine) -> pd.DataFrame:
    """Load cn_local_industry_map_hist for symbol-to-industry mapping (uses in_date/out_date)."""
    sql = """
        SELECT
            symbol,
            industry_id,
            industry_name
        FROM cn_local_industry_map_hist
        WHERE (out_date IS NULL OR out_date >= CURDATE())
    """
    return fetch_df(engine, sql)


# ---------------------------------------------------------------------------
# Factor computation functions
# ---------------------------------------------------------------------------


def _compute_mainline_strength_score(
    row: pd.Series,
    ms_lookup: dict,
) -> float:
    """
    Factor 3: Mainline Strength Score.
    Uses mainline_strength_score from cn_stock_mainline_strength_daily (industry-level).
    Stock gets the strength score of its mainline.
    """
    # industry_name is resolved in run_build via role_lookup
    industry_name = row.get("_mainline_name", "")
    if not industry_name:
        return 0.5
    ms_row = ms_lookup.get(industry_name)
    if ms_row is not None:
        return _clip_score(_safe_float(ms_row.get("mainline_strength_score"), 0.5))
    return 0.5


def _compute_capital_concentration_score(
    row: pd.Series,
    icf_lookup: dict,
    role_lookup: dict,
) -> float:
    """
    Factor 4: Capital Concentration Score.
    Uses concentration_score from cn_industry_capital_flow_daily,
    adjusted by stock role (leader gets bonus).
    """
    sym_key = (row["trade_date"], row["symbol"])
    role_row = role_lookup.get(sym_key)
    mainline_id = None
    if role_row is not None:
        mainline_id = role_row.get("mainline_id")

    if mainline_id is None:
        return 0.5

    icf_key = (row["trade_date"], mainline_id)
    icf_row = icf_lookup.get(icf_key)

    if icf_row is not None:
        base_score = _safe_float(icf_row.get("concentration_score"), 0.5)
        stock_role = role_row.get("stock_role", "") if role_row else ""
        if stock_role == "LEADER":
            base_score = min(1.0, base_score + 0.10)
        elif stock_role == "FOLLOWER":
            base_score = max(0.0, base_score - 0.05)
        return _clip_score(base_score)
    return 0.5


def _compute_leader_dominance_score(
    row: pd.Series,
    radar_lookup: dict,
    role_lookup: dict,
) -> float:
    """
    Factor 5: Leader Dominance Score.
    Uses leader_density, new_high_ratio, breakout_ratio from cn_ga_mainline_radar_daily.
    """
    sym_key = (row["trade_date"], row["symbol"])
    role_row = role_lookup.get(sym_key)
    mainline_id = None
    if role_row is not None:
        mainline_id = role_row.get("mainline_id")

    if mainline_id is None:
        return 0.5

    radar_key = (row["trade_date"], mainline_id)
    radar_row = radar_lookup.get(radar_key)

    if radar_row is not None:
        leader_density = _safe_float(radar_row.get("leader_density"), 0.5)
        new_high_ratio = _safe_float(radar_row.get("new_high_ratio"), 0.5)
        breakout_ratio = _safe_float(radar_row.get("breakout_ratio"), 0.5)

        composite = (leader_density + new_high_ratio + breakout_ratio) / 3.0
        stock_role = role_row.get("stock_role", "") if role_row else ""
        if stock_role == "LEADER":
            composite = min(1.0, composite + 0.10)
        elif stock_role == "FOLLOWER":
            composite = max(0.0, composite - 0.05)
        return _clip_score(composite)
    return 0.5


def _compute_trend_quality_score(
    row: pd.Series,
    price_lookup: dict,
) -> float:
    """
    Factor 6: Trend Quality Score.
    Uses cn_stock_daily_price data (UPPERCASE columns per P0) to assess trend quality.
    """
    key = (row["trade_date"], row["symbol"])
    price_row = price_lookup.get(key)

    if price_row is None:
        return 0.5

    close = _safe_float(price_row.get("CLOSE"), 0.0)
    pre_close = _safe_float(price_row.get("PRE_CLOSE"), 0.0)
    chg_pct = _safe_float(price_row.get("CHG_PCT"), 0.0)
    amount = _safe_float(price_row.get("AMOUNT"), 0.0)

    if close <= 0 or pre_close <= 0:
        return 0.5

    # Price momentum (close vs pre_close)
    price_ratio = close / pre_close if pre_close > 0 else 1.0
    price_trend_score = _clip_score((price_ratio - 0.95) / 0.10)  # 0.95 -> 0, 1.05 -> 1

    # Return contribution (chg_pct is in percentage points, e.g. 3.5 for +3.5%)
    return_score = _clip_score((chg_pct + 10.0) / 20.0)  # -10% -> 0, +10% -> 1

    # Volume contribution (higher amount = more conviction)
    vol_score = _clip_score(min(1.0, amount / 1e9)) if amount > 0 else 0.5

    composite = price_trend_score * 0.5 + return_score * 0.3 + vol_score * 0.2
    return _clip_score(composite)


def _compute_lifecycle_position_score(
    row: pd.Series,
    lc_lookup: dict,
) -> float:
    """
    Factor 7: Lifecycle Position Score.
    Maps lifecycle_state to a score:
      TREND_EXPANSION -> 1.0
      DIFFUSION       -> 0.8
      BOTTOM_REPAIR   -> 0.6
      DIVERGENCE      -> 0.4
      TOP_DECAY       -> 0.2
      RISK_OFF        -> 0.1
      UNKNOWN         -> 0.5
    """
    # mainline_id is resolved in run_build
    mainline_id = row.get("_mainline_id", "")
    if not mainline_id:
        return 0.5

    lc_key = (row["trade_date"], mainline_id)
    lc_row = lc_lookup.get(lc_key)
    if lc_row is not None:
        lc_state = str(lc_row.get("lifecycle_state", ""))
        if lc_state == "TREND_EXPANSION":
            return 1.0
        elif lc_state == "DIFFUSION":
            return 0.8
        elif lc_state == "BOTTOM_REPAIR":
            return 0.6
        elif lc_state == "DIVERGENCE":
            return 0.4
        elif lc_state == "TOP_DECAY":
            return 0.2
        elif lc_state == "RISK_OFF":
            return 0.1
        else:
            return _safe_float(lc_row.get("lifecycle_score"), 0.5)
    return 0.5


def _compute_risk_crowding_score(
    row: pd.Series,
    radar_lookup: dict,
    pulse_lookup: dict,
    role_lookup: dict,
) -> float:
    """
    Factor 8: Risk/Crowding Score.

    ═══ DIRECTION: HIGHER = HIGHER RISK (worse for the stock) ═══
    This function returns values where HIGHER means MORE RISKY.
    The score is INVERTED in _compute_final_score() via:
        risk_adjusted_score = 1.0 - risk_crowding_score
    This ensures high risk → low risk_adjusted_score → lower final_score.

    Direction proof for ALL code paths:
    ┌──────────────────────────┬──────────┬──────────────────────────────┐
    │ Condition                │ Raw Score│ Interpretation              │
    ├──────────────────────────┼──────────┼──────────────────────────────┤
    │ bearish_ratio > 0.5      │ 0.8      │ Bearish market = high risk   │
    │ bullish_ratio < 0.2      │ 0.7      │ Weak bullish = elevated risk │
    │ bullish_ratio > 0.6      │ 0.3      │ Strong bullish = low risk    │
    │ default (neutral)        │ 0.5      │ Neutral market               │
    │ RISK_OFF/CRISIS phase    │ +0.2     │ Crisis = even higher risk    │
    │ leader_density > 0.7     │ 0.8      │ Crowded = high risk          │
    │ leader_density > 0.5     │ 0.6      │ Moderate crowding            │
    │ leader_density <= 0.5    │ 0.3      │ Not crowded = low risk       │
    │ default (no data)        │ 0.5      │ Neutral (fallback)           │
    └──────────────────────────┴──────────┴──────────────────────────────┘

    After inversion in _compute_final_score():
    - High risk (0.8) → risk_adjusted = 0.2 → penalizes final_score ✓
    - Low risk (0.3)  → risk_adjusted = 0.7 → boosts final_score ✓
    - Neutral (0.5)   → risk_adjusted = 0.5 → no effect ✓

    Uses:
    - cn_ga_market_pulse_daily: bullish_industry_ratio, bearish_industry_ratio, market_phase
    - cn_ga_mainline_radar_daily: leader_density (crowding proxy)
    """
    sym_key = (row["trade_date"], row["symbol"])
    role_row = role_lookup.get(sym_key)
    mainline_id = None
    if role_row is not None:
        mainline_id = role_row.get("mainline_id")

    # Market-level risk (from cn_ga_market_pulse_daily)
    pulse_key = row["trade_date"]
    pulse_row = pulse_lookup.get(pulse_key)

    market_risk_score = 0.5  # neutral
    if pulse_row is not None:
        bullish_ratio = _safe_float(pulse_row.get("bullish_industry_ratio"), 0.5)
        bearish_ratio = _safe_float(pulse_row.get("bearish_industry_ratio"), 0.5)
        market_phase = str(pulse_row.get("market_phase", "")).upper()

        # Low bullish ratio + high bearish ratio = risky market
        if bearish_ratio > 0.5:
            market_risk_score = 0.8
        elif bullish_ratio < 0.2:
            market_risk_score = 0.7
        elif bullish_ratio > 0.6:
            market_risk_score = 0.3
        else:
            market_risk_score = 0.5

        # Market phase adjustment
        if market_phase in ("RISK_OFF", "CRISIS"):
            market_risk_score = min(1.0, market_risk_score + 0.2)

    # Mainline-level crowding (from cn_ga_mainline_radar_daily)
    mainline_crowding_score = 0.5
    if mainline_id is not None:
        radar_key = (row["trade_date"], mainline_id)
        radar_row = radar_lookup.get(radar_key)
        if radar_row is not None:
            leader_density = _safe_float(radar_row.get("leader_density"), 0.5)
            # High leader_density (>0.7) indicates crowding -> higher risk
            if leader_density > 0.7:
                mainline_crowding_score = 0.8
            elif leader_density > 0.5:
                mainline_crowding_score = 0.6
            else:
                mainline_crowding_score = 0.3

    composite = market_risk_score * 0.6 + mainline_crowding_score * 0.4
    return _clip_score(composite)


def _compute_final_score(factor_scores: dict[str, float]) -> float:
    """
    Compute weighted final score from 8 factor scores.

    ═══ RISK/CROWDING INVERSION ═══════════════════════════════════════════
    risk_crowding_score is a RISK factor: higher = more risk (worse).
    It MUST be inverted before contributing to the final score:
        risk_adjusted_score = 1.0 - risk_crowding_score

    This inversion is applied HERE, at the point of weighted aggregation,
    AFTER _compute_risk_crowding_score() returns its raw value. This means
    ALL code paths through _compute_risk_crowding_score() are covered,
    including all fallback/neutral values (0.5 → 1.0 - 0.5 = 0.5).

    Direction proof:
      Raw risk_crowding_score = 0.8 (high risk, e.g. bearish market + crowding)
      → risk_adjusted = 1.0 - 0.8 = 0.2
      → weighted contribution = 0.2 * 0.05 = 0.01 (penalizes final score) ✓

      Raw risk_crowding_score = 0.3 (low risk, e.g. strong bullish, no crowding)
      → risk_adjusted = 1.0 - 0.3 = 0.7
      → weighted contribution = 0.7 * 0.05 = 0.035 (boosts final score) ✓

      Raw risk_crowding_score = 0.5 (neutral / no data fallback)
      → risk_adjusted = 1.0 - 0.5 = 0.5
      → weighted contribution = 0.5 * 0.05 = 0.025 (neutral) ✓
    ════════════════════════════════════════════════════════════════════════

    Missing factors default to 0.5.
    """
    total_weight = 0.0
    weighted_sum = 0.0

    for factor_name, weight in FACTOR_WEIGHTS.items():
        score = factor_scores.get(factor_name, 0.5)
        # ═══ RISK INVERSION: higher risk_crowding_score = higher risk = worse ═══
        # Invert so that high risk → low contribution to final_score
        if factor_name == "risk_crowding_score":
            score = 1.0 - score  # risk_adjusted_score
        weighted_sum += score * weight
        total_weight += weight

    if total_weight <= 0:
        return 0.5

    return _clip_score(weighted_sum / total_weight)


def _assign_alpha_bucket(
    final_score: float,
    all_scores: list[float],
) -> str:
    """
    Assign alpha bucket based on cross-sectional percentile.
    Uses all final scores for the same trade_date to determine percentile.
    """
    if not all_scores:
        return "NEUTRAL"

    sorted_scores = sorted(all_scores)
    n = len(sorted_scores)

    rank = sum(1 for s in sorted_scores if s <= final_score)
    percentile = rank / n if n > 0 else 0.5

    if percentile >= BUCKET_THRESHOLDS["TOP_1"]:
        return "TOP_1"
    elif percentile >= BUCKET_THRESHOLDS["TOP_5"]:
        return "TOP_5"
    elif percentile >= BUCKET_THRESHOLDS["TOP_10"]:
        return "TOP_10"
    elif percentile >= BUCKET_THRESHOLDS["TOP_20"]:
        return "TOP_20"
    elif percentile >= BUCKET_THRESHOLDS["WATCH"]:
        return "WATCH"
    elif percentile >= BUCKET_THRESHOLDS["NEUTRAL"]:
        return "NEUTRAL"
    else:
        return "AVOID"


def _generate_explanation(
    factor_scores: dict[str, float],
    alpha_bucket: str,
    lifecycle_state: str | None,
    mainline_name: str | None,
    fundamental_risk_flag: str | None,
) -> tuple[str, str, str, str]:
    """
    Generate natural language explanation, top_factors, weak_factors, and flags.

    Returns:
        (explanation, top_factors, weak_factors, flags)
    """
    sorted_factors = sorted(
        factor_scores.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    top_factors_list = [f"{name}={score:.3f}" for name, score in sorted_factors[:3]]
    weak_factors_list = [f"{name}={score:.3f}" for name, score in sorted_factors[-3:] if score < 0.5]

    top_factors = "; ".join(top_factors_list)
    weak_factors = "; ".join(weak_factors_list)

    flags_list: list[str] = []
    if fundamental_risk_flag and fundamental_risk_flag != "NONE":
        flags_list.append(f"FUNDAMENTAL_RISK:{fundamental_risk_flag}")
    if lifecycle_state:
        flags_list.append(f"LIFECYCLE:{lifecycle_state}")
    if alpha_bucket in ("TOP_1", "TOP_5"):
        flags_list.append("HIGH_ALPHA")
    elif alpha_bucket == "AVOID":
        flags_list.append("LOW_ALPHA")

    flags = "; ".join(flags_list) if flags_list else "NONE"

    bucket_desc = {
        "TOP_1": "top 1% alpha",
        "TOP_5": "top 5% alpha",
        "TOP_10": "top 10% alpha",
        "TOP_20": "top 20% alpha",
        "WATCH": "watch list",
        "NEUTRAL": "neutral",
        "AVOID": "avoid",
    }

    mainline_str = f" in {mainline_name}" if mainline_name else ""
    lifecycle_str = f" [{lifecycle_state}]" if lifecycle_state else ""

    explanation = (
        f"Stock{mainline_str}{lifecycle_str} classified as {bucket_desc.get(alpha_bucket, alpha_bucket)}. "
        f"Top factors: {top_factors}. "
        f"Weak factors: {weak_factors}. "
        f"Flags: {flags}."
    )

    return explanation, top_factors, weak_factors, flags


# ---------------------------------------------------------------------------
# Main build logic
# ---------------------------------------------------------------------------


def run_build(args: argparse.Namespace) -> pd.DataFrame:
    """
    Main build function. Loads all data, computes 8 factors, assigns buckets,
    and returns a DataFrame ready for DB write.
    """
    verbose = args.verbose

    start = datetime.strptime(args.start, "%Y-%m-%d").date() if "-" in args.start else datetime.strptime(args.start, "%Y%m%d").date()
    end = datetime.strptime(args.end, "%Y-%m-%d").date() if args.end and "-" in args.end else (
        datetime.strptime(args.end, "%Y%m%d").date() if args.end else date.today()
    )

    db_password = args.db_password or os.getenv("ASHARE_MYSQL_PASSWORD", "sec_Bobo123")
    engine = build_engine(args.db_host, args.db_port, args.db_user, db_password, args.db_name)

    build_started_at = time.time()
    print(f"[{_ts()}] Date range: {start} ~ {end}", flush=True)
    print(f"[{_ts()}] Database: {args.db_name} | dry_run={args.dry_run} | replace={args.replace}", flush=True)

    # ── Load all source data ──────────────────────────────────────────
    print(f"[{_ts()}] Loading source tables ...", flush=True)
    quality_df = _timed_load("stock_quality_scores", load_stock_quality_scores, engine, start, end)
    ms_df = _timed_load("mainline_strength", load_mainline_strength, engine, start, end)
    radar_df = _timed_load("ga_mainline_radar", load_ga_mainline_radar, engine, start, end)
    icf_df = _timed_load("industry_capital_flow", load_industry_capital_flow, engine, start, end)
    role_df = _timed_load("ga_stock_role_map", load_ga_stock_role_map, engine, start, end)
    price_df = _timed_load("cn_stock_daily_price", load_stock_daily_price, engine, start, end)
    lc_df = _timed_load("mainline_lifecycle", load_mainline_lifecycle, engine, start, end)
    pulse_df = _timed_load("ga_market_pulse", load_ga_market_pulse, engine, start, end)
    ind_map_df = _timed_load("industry_map", load_industry_map, engine)

    # ── Build lookup maps for fast access ─────────────────────────────
    lookup_started_at = time.time()
    print(f"[{_ts()}] Building lookup maps ...", flush=True)

    # Mainline strength lookup: industry_name -> row (industry-level)
    ms_lookup: dict[str, dict] = {}
    if not ms_df.empty:
        total = len(ms_df)
        report_every = max(5000, total // 10 or 1)
        for i, (_, row) in enumerate(ms_df.iterrows(), 1):
            ms_lookup[row["industry_name"]] = row.to_dict()
            if i % report_every == 0 or i == total:
                _progress_line("mainline strength lookup", i, total, lookup_started_at)

    # Radar lookup: (trade_date, mainline_id) -> row
    radar_lookup: dict[tuple[date, str], dict] = {}
    if not radar_df.empty:
        total = len(radar_df)
        report_every = max(5000, total // 10 or 1)
        for i, (_, row) in enumerate(radar_df.iterrows(), 1):
            radar_lookup[(row["trade_date"], row["mainline_id"])] = row.to_dict()
            if i % report_every == 0 or i == total:
                _progress_line("radar lookup", i, total, lookup_started_at)

    # Industry capital flow lookup: (trade_date, industry_id) -> row
    icf_lookup: dict[tuple[date, str], dict] = {}
    if not icf_df.empty:
        total = len(icf_df)
        report_every = max(5000, total // 10 or 1)
        for i, (_, row) in enumerate(icf_df.iterrows(), 1):
            icf_lookup[(row["trade_date"], row["industry_id"])] = row.to_dict()
            if i % report_every == 0 or i == total:
                _progress_line("capital flow lookup", i, total, lookup_started_at)

    # Role lookup: (trade_date, symbol) -> row (from cn_ga_stock_role_map_daily)
    role_lookup: dict[tuple[date, str], dict] = {}
    if not role_df.empty:
        total = len(role_df)
        report_every = max(5000, total // 10 or 1)
        for i, (_, row) in enumerate(role_df.iterrows(), 1):
            role_lookup[(row["trade_date"], row["symbol"])] = row.to_dict()
            if i % report_every == 0 or i == total:
                _progress_line("role lookup", i, total, lookup_started_at)

    # Price lookup: (TRADE_DATE, SYMBOL) -> row (UPPERCASE columns per P0)
    price_lookup: dict[tuple[date, str], dict] = {}
    if not price_df.empty:
        total = len(price_df)
        report_every = max(5000, total // 10 or 1)
        for i, (_, row) in enumerate(price_df.iterrows(), 1):
            price_lookup[(row["TRADE_DATE"], row["SYMBOL"])] = row.to_dict()
            if i % report_every == 0 or i == total:
                _progress_line("price lookup", i, total, lookup_started_at)

    # Lifecycle lookup: (trade_date, mainline_id) -> row
    lc_lookup: dict[tuple[date, str], dict] = {}
    if not lc_df.empty:
        total = len(lc_df)
        report_every = max(5000, total // 10 or 1)
        for i, (_, row) in enumerate(lc_df.iterrows(), 1):
            lc_lookup[(row["trade_date"], row["mainline_id"])] = row.to_dict()
            if i % report_every == 0 or i == total:
                _progress_line("lifecycle lookup", i, total, lookup_started_at)

    # Market pulse lookup: trade_date -> row
    pulse_lookup: dict[date, dict] = {}
    if not pulse_df.empty:
        total = len(pulse_df)
        report_every = max(5000, total // 10 or 1)
        for i, (_, row) in enumerate(pulse_df.iterrows(), 1):
            pulse_lookup[row["trade_date"]] = row.to_dict()
            if i % report_every == 0 or i == total:
                _progress_line("pulse lookup", i, total, lookup_started_at)

    # Industry map lookup: symbol -> (industry_id, industry_name)
    ind_map_lookup: dict[str, tuple[str, str]] = {}
    if not ind_map_df.empty:
        total = len(ind_map_df)
        report_every = max(5000, total // 10 or 1)
        for i, (_, row) in enumerate(ind_map_df.iterrows(), 1):
            ind_map_lookup[row["symbol"]] = (
                row.get("industry_id", ""),
                row.get("industry_name", ""),
            )
            if i % report_every == 0 or i == total:
                _progress_line("industry map lookup", i, total, lookup_started_at)
    print(f"[{_ts()}] Lookup maps ready in {_fmt_seconds(time.time() - lookup_started_at)}", flush=True)

    # ── Determine base universe ──────────────────────────────────────
    if quality_df.empty:
        print("[WARNING] No quality scores found. Using ga_stock_role_map symbols as base.")
        base_df = role_df.copy() if not role_df.empty else pd.DataFrame()
    else:
        base_df = quality_df.copy()

    if base_df.empty:
        print("[ERROR] No base data available. Cannot compute unified alpha scores.")
        return pd.DataFrame()

    print(f"[{_ts()}] Base universe: {base_df['symbol'].nunique():,} symbols across {base_df['trade_date'].nunique():,} dates", flush=True)

    # ── Compute factors for each row ──────────────────────────────────
    results: list[dict[str, Any]] = []
    total_rows = len(base_df)
    batch_size = max(1, total_rows // 20)

    compute_started_at = time.time()
    print(f"[{_ts()}] Computing factors for {total_rows:,} rows ...", flush=True)

    for idx, (_, row) in enumerate(base_df.iterrows()):
        trade_date = row["trade_date"]
        symbol = row["symbol"]

        # Resolve industry info from industry map
        ind_info = ind_map_lookup.get(symbol, ("", ""))
        industry_id = ind_info[0]
        mainline_name = ind_info[1]

        # Resolve mainline_id and stock_role from cn_ga_stock_role_map_daily
        role_key = (trade_date, symbol)
        role_row = role_lookup.get(role_key)
        mainline_id = role_row.get("mainline_id") if role_row is not None else None
        stock_role = role_row.get("stock_role") if role_row is not None else None

        # Resolve mainline_name from role if industry map doesn't have it
        if not mainline_name and mainline_id:
            # Try to get mainline_name from radar lookup
            radar_key = (trade_date, mainline_id)
            radar_row = radar_lookup.get(radar_key)
            if radar_row is not None:
                mainline_name = str(radar_row.get("mainline_name", ""))
            # Try lifecycle lookup
            if not mainline_name:
                lc_key = (trade_date, mainline_id)
                lc_row = lc_lookup.get(lc_key)
                if lc_row is not None:
                    mainline_name = str(lc_row.get("mainline_name", ""))

        # Resolve lifecycle state from mainline lifecycle
        lifecycle_state: str | None = None
        if mainline_id is not None:
            lc_key = (trade_date, mainline_id)
            lc_row = lc_lookup.get(lc_key)
            if lc_row is not None:
                lifecycle_state = str(lc_row.get("lifecycle_state", "")) or None

        # Store resolved IDs for factor computation
        row["_mainline_id"] = mainline_id or ""
        row["_mainline_name"] = mainline_name or ""
        row["_stock_role"] = stock_role or ""

        # Factor 1: quality_score (from cn_stock_quality_score_daily)
        quality_score = _safe_float(row.get("quality_score"), 0.5)

        # Factor 2: growth_acceleration_score (from cn_stock_quality_score_daily)
        growth_acceleration_score = _safe_float(row.get("growth_acceleration_score"), 0.5)

        # Factor 3: mainline_strength_score
        mainline_strength_score = _compute_mainline_strength_score(row, ms_lookup)

        # Factor 4: capital_concentration_score
        capital_concentration_score = _compute_capital_concentration_score(
            row, icf_lookup, role_lookup
        )

        # Factor 5: leader_dominance_score
        leader_dominance_score = _compute_leader_dominance_score(
            row, radar_lookup, role_lookup
        )

        # Factor 6: trend_quality_score
        trend_quality_score = _compute_trend_quality_score(row, price_lookup)

        # Factor 7: lifecycle_position_score
        lifecycle_position_score = _compute_lifecycle_position_score(row, lc_lookup)

        # Factor 8: risk_crowding_score
        risk_crowding_score = _compute_risk_crowding_score(
            row, radar_lookup, pulse_lookup, role_lookup
        )

        # Fundamental risk flag
        fundamental_risk_flag = str(row.get("fundamental_risk_flag", "NONE")) if "fundamental_risk_flag" in row else "NONE"

        # Build factor scores dict
        factor_scores = {
            "quality_score": quality_score,
            "growth_acceleration_score": growth_acceleration_score,
            "mainline_strength_score": mainline_strength_score,
            "capital_concentration_score": capital_concentration_score,
            "leader_dominance_score": leader_dominance_score,
            "trend_quality_score": trend_quality_score,
            "lifecycle_position_score": lifecycle_position_score,
            "risk_crowding_score": risk_crowding_score,
        }

        # Compute final score (risk_crowding_score inverted inside _compute_final_score)
        final_score = _compute_final_score(factor_scores)

        results.append({
            "trade_date": trade_date,
            "symbol": symbol,
            "industry_id": industry_id,
            "quality_score": quality_score,
            "growth_acceleration_score": growth_acceleration_score,
            "mainline_strength_score": mainline_strength_score,
            "capital_concentration_score": capital_concentration_score,
            "leader_dominance_score": leader_dominance_score,
            "trend_quality_score": trend_quality_score,
            "lifecycle_position_score": lifecycle_position_score,
            "risk_crowding_score": risk_crowding_score,
            "final_score": final_score,
            "lifecycle_state": lifecycle_state,
            "mainline_name": mainline_name,
            "fundamental_risk_flag": fundamental_risk_flag,
            "factor_scores": factor_scores,
        })

        # Progress
        if (idx + 1) % batch_size == 0 or idx == total_rows - 1:
            pct = (idx + 1) / total_rows * 100
            _progress_line("factor rows", idx + 1, total_rows, compute_started_at, f"output={len(results):,}")

    # ── Convert to DataFrame ──────────────────────────────────────────
    result_df = pd.DataFrame(results)
    if result_df.empty:
        print("[WARNING] No results computed.")
        return result_df

    print(f"[{_ts()}] Factor computation complete: {len(result_df):,} rows ({_fmt_seconds(time.time() - compute_started_at)})", flush=True)

    # ── Assign alpha buckets (cross-sectional per trade_date) ─────────
    bucket_started_at = time.time()
    print(f"[{_ts()}] Assigning alpha buckets ...", flush=True)
    result_df["alpha_bucket"] = "NEUTRAL"
    date_groups = list(result_df.groupby("trade_date"))
    total_date_groups = len(date_groups)
    report_every = max(20, total_date_groups // 10 or 1)
    for i, (dt, group) in enumerate(date_groups, 1):
        scores = group["final_score"].tolist()
        bucket_map = {}
        for _, row in group.iterrows():
            bucket = _assign_alpha_bucket(row["final_score"], scores)
            bucket_map[row["symbol"]] = bucket
        mask = result_df["trade_date"] == dt
        result_df.loc[mask, "alpha_bucket"] = result_df.loc[mask, "symbol"].map(bucket_map)
        if i % report_every == 0 or i == total_date_groups:
            _progress_line("bucket dates", i, total_date_groups, bucket_started_at)

    # ── Generate explanations ─────────────────────────────────────────
    explain_started_at = time.time()
    print(f"[{_ts()}] Generating explanations ...", flush=True)
    explanations = []
    top_factors_list = []
    weak_factors_list = []
    flags_list = []

    total_explain_rows = len(result_df)
    report_every = max(5000, total_explain_rows // 20 or 1)
    for i, (_, row) in enumerate(result_df.iterrows(), 1):
        factor_scores = row["factor_scores"]
        if isinstance(factor_scores, str):
            factor_scores = json.loads(factor_scores)
        elif not isinstance(factor_scores, dict):
            factor_scores = {}

        explanation, top_f, weak_f, flags = _generate_explanation(
            factor_scores,
            row["alpha_bucket"],
            row.get("lifecycle_state"),
            row.get("mainline_name"),
            row.get("fundamental_risk_flag"),
        )
        explanations.append(explanation)
        top_factors_list.append(top_f)
        weak_factors_list.append(weak_f)
        flags_list.append(flags)
        if i % report_every == 0 or i == total_explain_rows:
            _progress_line("explanations", i, total_explain_rows, explain_started_at)

    result_df["explanation"] = explanations
    result_df["top_factors"] = top_factors_list
    result_df["weak_factors"] = weak_factors_list
    result_df["flags"] = flags_list

    # Drop internal columns
    result_df = result_df.drop(columns=["factor_scores", "_mainline_id", "_mainline_name", "_stock_role",
                                         "fundamental_risk_flag"], errors="ignore")

    print(f"[{_ts()}] Build complete: {len(result_df):,} rows, {result_df['trade_date'].nunique():,} dates, "
          f"{result_df['symbol'].nunique():,} symbols ({_fmt_seconds(time.time() - build_started_at)})", flush=True)

    return result_df


# ---------------------------------------------------------------------------
# DB write
# ---------------------------------------------------------------------------


def write_to_db(
    engine: Engine,
    df: pd.DataFrame,
    db_name: str,
    replace: bool = False,
    dry_run: bool = False,
    verbose: bool = False,
) -> int:
    """Write computed DataFrame to cn_unified_alpha_score_daily. Returns row count."""
    if df.empty:
        print("[Write] No data to write.")
        return 0

    table_name = "cn_unified_alpha_score_daily"

    columns = [
        "trade_date",
        "symbol",
        "industry_id",
        "quality_score",
        "growth_acceleration_score",
        "mainline_strength_score",
        "capital_concentration_score",
        "leader_dominance_score",
        "trend_quality_score",
        "lifecycle_position_score",
        "risk_crowding_score",
        "final_score",
        "alpha_bucket",
        "lifecycle_state",
        "mainline_name",
        "explanation",
        "top_factors",
        "weak_factors",
        "flags",
    ]

    for col in columns:
        if col not in df.columns:
            df[col] = None

    write_df = df[columns].copy()
    write_df["trade_date"] = write_df["trade_date"].astype(str)
    write_df = write_df.where(pd.notna(write_df), None)

    if dry_run:
        print(f"[{_ts()}] [Dry-run] Would write {len(write_df):,} rows to {db_name}.{table_name}", flush=True)
        if verbose:
            print(f"  Sample rows:")
            print(f"  {write_df.head(3).to_string()}")
        return len(write_df)

    write_started_at = time.time()
    print(f"[{_ts()}] Writing to DB: {db_name}.{table_name} rows={len(write_df):,}", flush=True)

    if replace:
        date_min = write_df["trade_date"].min()
        date_max = write_df["trade_date"].max()
        del_sql = text(f"DELETE FROM {table_name} WHERE trade_date BETWEEN :start AND :end")
        with engine.begin() as conn:
            deleted = conn.execute(del_sql, {"start": date_min, "end": date_max}).rowcount
        print(f"[{_ts()}]   Deleted {deleted:,} existing rows in [{date_min} ~ {date_max}]", flush=True)

    # Build upsert SQL
    col_list = ", ".join(columns)
    placeholders = ", ".join([f":{c}" for c in columns])
    update_parts = [f"{c} = VALUES({c})" for c in columns if c not in ("trade_date", "symbol")]
    update_parts.append("updated_at = CURRENT_TIMESTAMP")
    update_clause = ", ".join(update_parts)

    upsert_sql = f"""
    INSERT INTO {table_name} ({col_list})
    VALUES ({placeholders})
    ON DUPLICATE KEY UPDATE {update_clause}
    """

    rows = write_df.astype(object).where(pd.notna(write_df), None).to_dict(orient="records")

    total = 0
    batch_size = 4000
    with engine.begin() as conn:
        for i in range(0, len(rows), batch_size):
            batch = rows[i : i + batch_size]
            conn.execute(text(upsert_sql), batch)
            total += len(batch)
            _progress_line("DB write", total, len(rows), write_started_at)

    print(f"[{_ts()}] DB write complete: {total:,} rows ({_fmt_seconds(time.time() - write_started_at)})", flush=True)
    return total


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------


def generate_reports(
    df: pd.DataFrame,
    start: date,
    end: date,
    output_dir: str | None = None,
) -> tuple[str, str]:
    """Generate summary CSV and Markdown reports. Returns (csv_path, md_path)."""
    if output_dir:
        report_dir = Path(output_dir)
    else:
        report_dir = REPORT_DIR

    report_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = report_dir / f"unified_alpha_{start}_{end}_{timestamp}.csv"
    md_path = report_dir / f"unified_alpha_{start}_{end}_{timestamp}.md"

    # CSV report
    report_cols = [
        "trade_date", "symbol", "industry_id", "final_score", "alpha_bucket",
        "quality_score", "growth_acceleration_score", "mainline_strength_score",
        "capital_concentration_score", "leader_dominance_score",
        "trend_quality_score", "lifecycle_position_score", "risk_crowding_score",
        "lifecycle_state", "mainline_name", "flags",
    ]
    csv_df = df[[c for c in report_cols if c in df.columns]].copy()
    csv_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"[Report] CSV: {csv_path}")

    # Markdown report
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Unified Alpha Score Report\n")
        f.write(f"**Date Range:** {start} ~ {end}\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Summary Statistics\n\n")
        f.write(f"- Total rows: {len(df)}\n")
        f.write(f"- Unique dates: {df['trade_date'].nunique()}\n")
        f.write(f"- Unique symbols: {df['symbol'].nunique()}\n")
        f.write(f"- Unique industries: {df['industry_id'].nunique() if 'industry_id' in df.columns else 'N/A'}\n\n")

        f.write("### Alpha Bucket Distribution\n\n")
        f.write("| Bucket | Count | Percentage |\n")
        f.write("|--------|-------|------------|\n")
        bucket_counts = df["alpha_bucket"].value_counts()
        for bucket in ALPHA_BUCKETS:
            count = bucket_counts.get(bucket, 0)
            pct = count / len(df) * 100 if len(df) > 0 else 0
            f.write(f"| {bucket} | {count} | {pct:.2f}% |\n")
        f.write("\n")

        f.write("### Top 20 by Final Score\n\n")
        f.write("| Rank | Trade Date | Symbol | Industry | Final Score | Bucket |\n")
        f.write("|------|------------|--------|----------|-------------|--------|\n")
        top20 = df.nlargest(20, "final_score")
        for rank, (_, row) in enumerate(top20.iterrows(), 1):
            f.write(f"| {rank} | {row['trade_date']} | {row['symbol']} | "
                    f"{row.get('industry_id', 'N/A')} | {row['final_score']:.4f} | "
                    f"{row['alpha_bucket']} |\n")
        f.write("\n")

        f.write("### Factor Weights\n\n")
        f.write("| Factor | Weight |\n")
        f.write("|--------|--------|\n")
        for factor_name, weight in FACTOR_WEIGHTS.items():
            f.write(f"| {factor_name} | {weight:.2f} |\n")
        f.write("\n")

        if "flags" in df.columns:
            f.write("### Flags Summary\n\n")
            all_flags: list[str] = []
            for flags_str in df["flags"].dropna():
                all_flags.extend([f.strip() for f in str(flags_str).split(";") if f.strip()])
            flag_counts = pd.Series(all_flags).value_counts().head(20)
            f.write("| Flag | Count |\n")
            f.write("|------|-------|\n")
            for flag, count in flag_counts.items():
                f.write(f"| {flag} | {count} |\n")
            f.write("\n")

        f.write("### Daily Statistics\n\n")
        f.write("| Date | Count | Mean Score | Top 1 | Top 5 | Top 10 | Top 20 | Watch | Neutral | Avoid |\n")
        f.write("|------|-------|------------|-------|-------|--------|--------|-------|---------|-------|\n")
        for dt, group in df.groupby("trade_date"):
            bucket_dist = group["alpha_bucket"].value_counts()
            f.write(f"| {dt} | {len(group)} | {group['final_score'].mean():.4f} | "
                    f"{bucket_dist.get('TOP_1', 0)} | {bucket_dist.get('TOP_5', 0)} | "
                    f"{bucket_dist.get('TOP_10', 0)} | {bucket_dist.get('TOP_20', 0)} | "
                    f"{bucket_dist.get('WATCH', 0)} | {bucket_dist.get('NEUTRAL', 0)} | "
                    f"{bucket_dist.get('AVOID', 0)} |\n")
        f.write("\n")

    print(f"[Report] MD:  {md_path}")

    return str(csv_path), str(md_path)




def _date_chunks(start: date, end: date, months: int) -> list[tuple[date, date]]:
    """Split [start, end] into month-based chunks."""
    if months <= 0:
        months = 3
    chunks: list[tuple[date, date]] = []
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




# ---------------------------------------------------------------------------
# Source data coverage preflight
# ---------------------------------------------------------------------------

REQUIRED_SOURCE_TABLES: list[dict[str, str]] = [
    {"table": "cn_stock_quality_score_daily", "date_col": "trade_date", "label": "stock_quality_scores"},
    {"table": "cn_stock_mainline_strength_daily", "date_col": "trade_date", "label": "mainline_strength"},
    {"table": "cn_ga_mainline_radar_daily", "date_col": "trade_date", "label": "ga_mainline_radar"},
    {"table": "cn_ga_stock_role_map_daily", "date_col": "trade_date", "label": "ga_stock_role_map"},
    {"table": "cn_stock_daily_price", "date_col": "TRADE_DATE", "label": "cn_stock_daily_price"},
    {"table": "cn_mainline_lifecycle_daily", "date_col": "trade_date", "label": "mainline_lifecycle"},
    {"table": "cn_ga_market_pulse_daily", "date_col": "trade_date", "label": "ga_market_pulse"},
]

OPTIONAL_SOURCE_TABLES: list[dict[str, str]] = [
    {"table": "cn_industry_capital_flow_daily", "date_col": "trade_date", "label": "industry_capital_flow"},
]

SOURCE_COVERAGE_SQL = """
    SELECT
        COUNT(*) AS row_count,
        MIN({date_col}) AS min_date,
        MAX({date_col}) AS max_date,
        COUNT(DISTINCT {date_col}) AS date_count
    FROM {table}
    WHERE {date_col} BETWEEN :start AND :end
"""


def _source_coverage_row(engine: Engine, table: str, date_col: str, start: date, end: date) -> dict[str, Any]:
    if not table_exists(engine, engine.url.database, table):
        return {
            "table": table,
            "date_col": date_col,
            "row_count": 0,
            "min_date": None,
            "max_date": None,
            "date_count": 0,
            "exists": False,
        }

    sql = SOURCE_COVERAGE_SQL.format(table=table, date_col=date_col)
    with engine.connect() as conn:
        row = conn.execute(text(sql), {"start": start, "end": end}).mappings().first()
    if row is None:
        return {
            "table": table,
            "date_col": date_col,
            "row_count": 0,
            "min_date": None,
            "max_date": None,
            "date_count": 0,
            "exists": True,
        }

    return {
        "table": table,
        "date_col": date_col,
        "row_count": int(row.get("row_count") or 0),
        "min_date": row.get("min_date"),
        "max_date": row.get("max_date"),
        "date_count": int(row.get("date_count") or 0),
        "exists": True,
    }


def _normalize_date_value(value: Any) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return datetime.strptime(value[:10], "%Y-%m-%d").date()
        except ValueError:
            return None
    return None


def audit_source_data_coverage(engine: Engine, start: date, end: date) -> None:
    """
    Fail-fast source data coverage audit.

    Required tables must:
      1. exist
      2. contain at least one row in the requested date range
      3. have min_date <= start and max_date >= end within the requested filter

    Optional tables are reported but do not block execution.
    """
    print(f"[{_ts()}] Source data coverage audit start: {start} ~ {end}", flush=True)

    missing_or_incomplete: list[str] = []

    for spec in REQUIRED_SOURCE_TABLES:
        table = spec["table"]
        date_col = spec["date_col"]
        label = spec["label"]
        t0 = time.time()
        info = _source_coverage_row(engine, table, date_col, start, end)

        min_date = _normalize_date_value(info.get("min_date"))
        max_date = _normalize_date_value(info.get("max_date"))
        row_count = info.get("row_count", 0)
        date_count = info.get("date_count", 0)

        status = "OK"
        reason = ""
        if not info.get("exists"):
            status = "MISSING_TABLE"
            reason = "table does not exist"
        elif row_count <= 0:
            status = "NO_ROWS"
            reason = "no rows in requested date range"
        elif min_date is None or max_date is None:
            status = "INVALID_DATES"
            reason = "min/max date is NULL"
        elif min_date > start or max_date < end:
            status = "RANGE_NOT_COVERED"
            reason = f"available={min_date}~{max_date}, required={start}~{end}"

        print(
            f"[{_ts()}]   required {label:<24} table={table:<36} "
            f"rows={row_count:,} dates={date_count:,} range={min_date}~{max_date} "
            f"status={status} ({_fmt_seconds(time.time() - t0)})",
            flush=True,
        )

        if status != "OK":
            missing_or_incomplete.append(
                f"- {table}.{date_col}: {status} | {reason}"
            )

    for spec in OPTIONAL_SOURCE_TABLES:
        table = spec["table"]
        date_col = spec["date_col"]
        label = spec["label"]
        t0 = time.time()
        info = _source_coverage_row(engine, table, date_col, start, end)

        min_date = _normalize_date_value(info.get("min_date"))
        max_date = _normalize_date_value(info.get("max_date"))
        row_count = info.get("row_count", 0)
        date_count = info.get("date_count", 0)

        status = "OK"
        if not info.get("exists"):
            status = "MISSING_TABLE_OPTIONAL"
        elif row_count <= 0:
            status = "NO_ROWS_OPTIONAL"
        elif min_date is None or max_date is None:
            status = "INVALID_DATES_OPTIONAL"
        elif min_date > start or max_date < end:
            status = "RANGE_NOT_COVERED_OPTIONAL"

        print(
            f"[{_ts()}]   optional {label:<24} table={table:<36} "
            f"rows={row_count:,} dates={date_count:,} range={min_date}~{max_date} "
            f"status={status} ({_fmt_seconds(time.time() - t0)})",
            flush=True,
        )

    if missing_or_incomplete:
        print("", flush=True)
        print("=" * 60, flush=True)
        print("[SOURCE DATA AUDIT FAILED]", flush=True)
        print("Required source tables do not cover the requested input date range.", flush=True)
        print("Missing / incomplete required sources:", flush=True)
        for item in missing_or_incomplete:
            print(item, flush=True)
        print("=" * 60, flush=True)
        sys.exit(2)

    print(f"[{_ts()}] Source data coverage audit PASS", flush=True)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    print("=" * 60)
    print("  GrowthAlpha V8 — P3 Unified Alpha Engine")
    print("=" * 60)
    print(f"  Start: {args.start}")
    print(f"  End:   {args.end or 'today'}")
    print(f"  DB:    {args.db_name}")
    print(f"  Dry-run: {args.dry_run}")
    print(f"  Replace: {args.replace}")
    print()

    db_password = args.db_password or os.getenv("ASHARE_MYSQL_PASSWORD", "sec_Bobo123")
    engine = build_engine(args.db_host, args.db_port, args.db_user, db_password, args.db_name)

    # Ensure DDL
    ddl_path = Path(__file__).resolve().parents[1] / "sql" / "create_unified_alpha_score_daily.sql"
    if ddl_path.exists():
        ddl_sql = ddl_path.read_text(encoding="utf-8")
        stmts = [s.strip() for s in ddl_sql.split(";") if s.strip()]
        with engine.begin() as conn:
            for stmt in stmts:
                if "CREATE TABLE" in stmt:
                    conn.execute(text(stmt))
        print(f"[DDL] Ensured cn_unified_alpha_score_daily table exists")
    else:
        print(f"[WARNING] DDL file not found: {ddl_path}")

    start = datetime.strptime(args.start, "%Y-%m-%d").date() if "-" in args.start else datetime.strptime(args.start, "%Y%m%d").date()
    end = datetime.strptime(args.end, "%Y-%m-%d").date() if args.end and "-" in args.end else (
        datetime.strptime(args.end, "%Y%m%d").date() if args.end else date.today()
    )

    audit_source_data_coverage(engine, start, end)

    chunks = _date_chunks(start, end, getattr(args, "chunk_months", 3))
    total_chunks = len(chunks)
    total_computed = 0
    total_written = 0
    chunk_reports: list[tuple[str, str]] = []
    overall_started_at = time.time()

    print(
        f"[{_ts()}] Chunked build enabled: chunks={total_chunks}, "
        f"chunk_months={getattr(args, 'chunk_months', 3)}",
        flush=True,
    )

    for chunk_idx, (chunk_start, chunk_end) in enumerate(chunks, 1):
        chunk_started_at = time.time()
        print("", flush=True)
        print("=" * 60, flush=True)
        print(
            f"[{_ts()}] Chunk {chunk_idx}/{total_chunks}: {chunk_start} ~ {chunk_end} "
            f"overall_elapsed={_fmt_seconds(time.time() - overall_started_at)}",
            flush=True,
        )
        print("=" * 60, flush=True)

        chunk_args = argparse.Namespace(**vars(args))
        chunk_args.start = chunk_start.strftime("%Y-%m-%d")
        chunk_args.end = chunk_end.strftime("%Y-%m-%d")

        result_df = run_build(chunk_args)

        if result_df.empty:
            print(f"[{_ts()}] Chunk {chunk_idx}/{total_chunks} produced no rows", flush=True)
            continue

        written = write_to_db(
            engine,
            result_df,
            args.db_name,
            args.replace,
            args.dry_run,
            args.verbose,
        )
        total_computed += len(result_df)
        total_written += written

        if not getattr(args, "no_report", False):
            report_started_at = time.time()
            print(f"[{_ts()}] Generating chunk report ...", flush=True)
            csv_path, md_path = generate_reports(result_df, chunk_start, chunk_end, args.output_dir)
            chunk_reports.append((csv_path, md_path))
            print(f"[{_ts()}] Chunk report complete ({_fmt_seconds(time.time() - report_started_at)})", flush=True)

        chunk_elapsed = time.time() - chunk_started_at
        remaining = total_chunks - chunk_idx
        eta = chunk_elapsed * remaining
        print(
            f"[{_ts()}] Chunk {chunk_idx}/{total_chunks} done: "
            f"computed={len(result_df):,}, written={written:,}, "
            f"chunk_elapsed={_fmt_seconds(chunk_elapsed)}, eta≈{_fmt_seconds(eta)}",
            flush=True,
        )

        del result_df

    if total_computed <= 0:
        print("No data computed. Exiting.")
        sys.exit(0)

    print()
    print("=" * 60)
    print(f"  Build Complete")
    print(f"  Chunks:         {total_chunks}")
    print(f"  Rows computed:  {total_computed}")
    print(f"  Rows written:   {total_written}")
    print(f"  Total elapsed:  {_fmt_seconds(time.time() - overall_started_at)}")
    if getattr(args, "no_report", False):
        print(f"  Reports:        skipped (--no-report)")
    else:
        print(f"  Chunk reports:  {len(chunk_reports)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
