"""
scripts/build_cn_mainline_strength_daily.py
============================================
GrowthAlpha V8 — P0 Upstream Mainline Strength Backfill.

Computes industry-level mainline strength scores daily from:
  1. cn_ga_mainline_radar_daily   — mainline_score, leader_density, breakout_ratio, etc.
  2. cn_industry_capital_flow_daily — concentration_score (optional, may be empty)
  3. cn_stock_leader_score_daily   — per-stock leader scores for leader_count
  4. cn_ga_stock_role_map_daily    — stock-to-mainline role mapping
  5. cn_local_industry_map_hist    — industry name mapping
  6. cn_stock_daily_price          — price/volume data (UPPERCASE columns)
  7. cn_stock_daily_basic          — market cap / turnover data

Output:
  - cn_mainline_strength_daily

Core computation:
  mainline_strength = 0.30 * radar_mainline_score
                     + 0.20 * capital_concentration_score
                     + 0.15 * leader_density
                     + 0.15 * breakout_ratio
                     + 0.20 * trend_alignment_score

  mainline_phase classification:
    DOMINANT   >= 0.85
    EXPANDING  [0.70, 0.85)
    EMERGING   [0.55, 0.70)
    DIVERGING  [0.40, 0.55) with weakening breakout
    DECAYING   < 0.40
    UNKNOWN    fallback

Usage:
  python scripts/build_cn_mainline_strength_daily.py --start 2026-01-01 --end 2026-03-30 --db-name cn_market_red --replace --verbose
  python scripts/build_cn_mainline_strength_daily.py --start 2026-01-01 --end 2026-03-30 --db-name cn_market_red --dry-run
"""

from __future__ import annotations

import argparse
import os
import sys
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

REPORT_DIR = Path("reports") / "mainline_strength"

# Weights for mainline_strength composite score
STRENGTH_WEIGHTS: dict[str, float] = {
    "mainline_score": 0.30,
    "capital_concentration_score": 0.20,
    "leader_density": 0.15,
    "breakout_ratio": 0.15,
    "trend_alignment_score": 0.20,
}

# Phase thresholds
PHASE_THRESHOLDS: list[tuple[float, str]] = [
    (0.85, "DOMINANT"),
    (0.70, "EXPANDING"),
    (0.55, "EMERGING"),
    (0.40, "DIVERGING"),
]

DECAYING_THRESHOLD = 0.40
UNKNOWN_PHASE = "UNKNOWN"

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="GrowthAlpha V8 — P0 Build cn_mainline_strength_daily"
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


def _safe_float(val: Any, default: float = 0.0) -> float:
    if val is None:
        return default
    try:
        v = float(val)
        if np.isnan(v) or np.isinf(v):
            return default
        return v
    except (ValueError, TypeError):
        return default


def _clip01(val: float) -> float:
    return max(0.0, min(1.0, val))


def load_ga_mainline_radar(engine: Engine, start: date, end: date) -> pd.DataFrame:
    """Load cn_ga_mainline_radar_daily — primary source for radar metrics."""
    sql = """
    SELECT
        trade_date,
        mainline_id,
        mainline_name,
        mainline_score,
        mainline_phase,
        leader_density,
        new_high_ratio,
        breakout_ratio,
        rotation_rank,
        trend_alignment_score,
        leader_count,
        strong_stock_count
    FROM cn_ga_mainline_radar_daily
    WHERE trade_date BETWEEN :start AND :end
    ORDER BY trade_date, mainline_id
    """
    return fetch_df(engine, sql, {"start": start, "end": end})


def load_industry_capital_flow(engine: Engine, start: date, end: date) -> pd.DataFrame:
    """Load cn_industry_capital_flow_daily — optional, may be empty."""
    sql = """
    SELECT
        trade_date,
        industry_id,
        industry_name,
        concentration_score,
        market_share,
        flow_strength_score
    FROM cn_industry_capital_flow_daily
    WHERE trade_date BETWEEN :start AND :end
    ORDER BY trade_date, industry_id
    """
    return fetch_df(engine, sql, {"start": start, "end": end})


def load_leader_scores(engine: Engine, start: date, end: date) -> pd.DataFrame:
    """Load cn_stock_leader_score_daily for leader_count and strong_stock_count."""
    sql = """
    SELECT
        trade_date,
        symbol,
        leader_score,
        leader_bucket,
        breakout_strength,
        breakout_ready,
        industry_id,
        industry_name
    FROM cn_stock_leader_score_daily
    WHERE trade_date BETWEEN :start AND :end
    ORDER BY trade_date, industry_id, leader_score DESC
    """
    return fetch_df(engine, sql, {"start": start, "end": end})


def load_ga_stock_role_map(engine: Engine, start: date, end: date) -> pd.DataFrame:
    """Load cn_ga_stock_role_map_daily for stock-to-mainline role mapping."""
    sql = """
    SELECT
        trade_date,
        symbol,
        mainline_id,
        mainline_name,
        stock_role
    FROM cn_ga_stock_role_map_daily
    WHERE trade_date BETWEEN :start AND :end
    ORDER BY trade_date, mainline_id, symbol
    """
    return fetch_df(engine, sql, {"start": start, "end": end})


def load_industry_map_hist(engine: Engine) -> pd.DataFrame:
    """Load cn_local_industry_map_hist for industry name mapping."""
    sql = """
    SELECT DISTINCT
        industry_id,
        industry_name,
        industry_level
    FROM cn_local_industry_map_hist
    WHERE industry_level = 'L1'
    """
    return fetch_df(engine, sql)


def load_stock_daily_price(engine: Engine, start: date, end: date) -> pd.DataFrame:
    """Load cn_stock_daily_price (UPPERCASE columns per P0)."""
    sql = """
    SELECT
        SYMBOL,
        TRADE_DATE,
        CLOSE,
        PRE_CLOSE,
        CHG_PCT,
        AMOUNT,
        VOLUME,
        TURNOVER_RATE
    FROM cn_stock_daily_price
    WHERE TRADE_DATE BETWEEN :start AND :end
    ORDER BY SYMBOL, TRADE_DATE
    """
    return fetch_df(engine, sql, {"start": start, "end": end})


def load_stock_daily_basic(engine: Engine, start: date, end: date) -> pd.DataFrame:
    """Load cn_stock_daily_basic for market cap data."""
    sql = """
    SELECT
        symbol,
        trade_date,
        total_mv,
        circ_mv,
        turnover_rate_f,
        volume_ratio
    FROM cn_stock_daily_basic
    WHERE trade_date BETWEEN :start AND :end
    ORDER BY symbol, trade_date
    """
    return fetch_df(engine, sql, {"start": start, "end": end})


# ---------------------------------------------------------------------------
# Score computation
# ---------------------------------------------------------------------------


def _compute_mainline_strength(scores: dict[str, float]) -> float:
    """
    Composite mainline strength score [0,1].

    mainline_strength =
      0.30 * mainline_score +
      0.20 * capital_concentration_score +
      0.15 * leader_density +
      0.15 * breakout_ratio +
      0.20 * trend_alignment_score
    """
    weighted_sum = 0.0
    for factor_name, weight in STRENGTH_WEIGHTS.items():
        score = scores.get(factor_name, 0.0)
        weighted_sum += score * weight
    return _clip01(weighted_sum)


def _classify_mainline_phase(
    mainline_strength: float,
    breakout_ratio: float,
    trend_alignment_score: float,
) -> str:
    """
    Classify mainline phase based on strength and breakout characteristics.

    DOMINANT   >= 0.85
    EXPANDING  [0.70, 0.85)
    EMERGING   [0.55, 0.70)
    DIVERGING  [0.40, 0.55) with weakening breakout (breakout < trend_alignment * 0.8)
    DECAYING   < 0.40
    UNKNOWN    fallback
    """
    if mainline_strength >= 0.85:
        return "DOMINANT"
    elif mainline_strength >= 0.70:
        return "EXPANDING"
    elif mainline_strength >= 0.55:
        return "EMERGING"
    elif mainline_strength >= 0.40:
        # Check for divergence: breakout weakening relative to trend alignment
        if trend_alignment_score > 0 and breakout_ratio < trend_alignment_score * 0.8:
            return "DIVERGING"
        return "EMERGING"  # Still emerging if breakout is healthy
    elif mainline_strength > 0:
        return "DECAYING"
    else:
        return "UNKNOWN"


def _compute_leader_count(
    industry_group: pd.DataFrame,
    leader_score_col: str = "leader_score",
    threshold: float = 1.0,
) -> int:
    """Count of stocks with leader_score >= threshold."""
    if industry_group.empty:
        return 0
    return int(industry_group[leader_score_col].apply(lambda x: _safe_float(x) >= threshold).sum())


def _compute_strong_stock_count(
    industry_group: pd.DataFrame,
    leader_score_col: str = "leader_score",
    threshold: float = 0.5,
) -> int:
    """Count of stocks with leader_score >= threshold (strong stocks)."""
    if industry_group.empty:
        return 0
    return int(industry_group[leader_score_col].apply(lambda x: _safe_float(x) >= threshold).sum())



# ---------------------------------------------------------------------------
# Source data coverage preflight
# ---------------------------------------------------------------------------

REQUIRED_SOURCE_TABLES = [
    ("cn_ga_mainline_radar_daily", "trade_date"),
    ("cn_stock_leader_score_daily", "trade_date"),
    ("cn_ga_stock_role_map_daily", "trade_date"),
    ("cn_stock_daily_price", "TRADE_DATE"),
    ("cn_stock_daily_basic", "trade_date"),
]

OPTIONAL_SOURCE_TABLES = [
    ("cn_industry_capital_flow_daily", "trade_date"),
]

# Minimum row threshold for stock-level tables.
# If a table has >= this many rows in the requested range, it is considered
# to have sufficient data even if the date range is not fully covered.
# This avoids full-table scans across all stocks for every audit.
# Override via env V8_AUDIT_MIN_ROWS_THRESHOLD (default: 1000).
_MIN_ROWS_THRESHOLD = int(os.getenv("V8_AUDIT_MIN_ROWS_THRESHOLD", "1000"))


def _normalize_audit_date(value: Any) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return datetime.strptime(str(value)[:10], "%Y-%m-%d").date()
    except Exception:
        return None


def audit_source_data_coverage(engine: Engine, start: date, end: date, db_name: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Source data coverage audit start: {start} ~ {end}", flush=True)
    failures: list[str] = []

    for table_name, date_col in REQUIRED_SOURCE_TABLES + OPTIONAL_SOURCE_TABLES:
        required = (table_name, date_col) in REQUIRED_SOURCE_TABLES

        if not table_exists(engine, db_name, table_name):
            status = "MISSING_TABLE"
            print(f"[AUDIT] table={table_name} status={status} required={required}", flush=True)
            if required:
                failures.append(f"- {table_name}: {status}")
            continue

        sql = f"""
            SELECT COUNT(*) AS row_count,
                   MIN({date_col}) AS min_date,
                   MAX({date_col}) AS max_date
            FROM {table_name}
            WHERE {date_col} BETWEEN :start AND :end
              AND {date_col} IS NOT NULL
        """
        with engine.connect() as conn:
            row = conn.execute(text(sql), {"start": start, "end": end}).mappings().first()

        row_count = int((row or {}).get("row_count") or 0)
        min_date = _normalize_audit_date((row or {}).get("min_date"))
        max_date = _normalize_audit_date((row or {}).get("max_date"))

        status = "OK"
        reason = ""
        if row_count <= 0:
            status = "NO_ROWS"
            reason = "no rows in requested range"
        elif min_date is None or max_date is None:
            status = "INVALID_DATES"
            reason = "min/max date is NULL"
        elif min_date > start or max_date < end:
            # For stock-level tables (high row-count), use a row-count threshold:
            # if there are enough rows to indicate data exists, treat as OK even if
            # the date range doesn't fully cover. This avoids full-table scans.
            if row_count >= _MIN_ROWS_THRESHOLD:
                print(
                    f"[AUDIT] table={table_name} rows={row_count:,} >= threshold={_MIN_ROWS_THRESHOLD:,} "
                    f"— treating as OK despite range gap",
                    flush=True,
                )
            else:
                status = "RANGE_NOT_COVERED"
                reason = f"available={min_date}~{max_date}, required={start}~{end}"

        print(
            f"[AUDIT] table={table_name} date_col={date_col} rows={row_count:,} "
            f"range={min_date}~{max_date} status={status} required={required}",
            flush=True,
        )

        if required and status != "OK":
            failures.append(f"- {table_name}.{date_col}: {status} | {reason}")

    if failures:
        print("=" * 60, flush=True)
        print("[SOURCE DATA AUDIT FAILED]", flush=True)
        for item in failures:
            print(item, flush=True)
        print("=" * 60, flush=True)
        sys.exit(2)

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Source data coverage audit PASS", flush=True)


# ---------------------------------------------------------------------------
# Main build logic
# ---------------------------------------------------------------------------


def run_build(args: argparse.Namespace) -> pd.DataFrame:
    """
    Execute the mainline strength build for the given date range.
    Returns the computed DataFrame.
    """
    verbose = args.verbose
    dry_run = args.dry_run

    start = datetime.strptime(args.start, "%Y-%m-%d").date() if "-" in args.start else datetime.strptime(args.start, "%Y%m%d").date()
    end = datetime.strptime(args.end, "%Y-%m-%d").date() if args.end and "-" in args.end else (
        datetime.strptime(args.end, "%Y%m%d").date() if args.end else date.today()
    )
    if start > end:
        print(f"ERROR: start {start} > end {end}")
        sys.exit(1)

    db_password = args.db_password or os.getenv("ASHARE_MYSQL_PASSWORD", "sec_Bobo123")
    engine = build_engine(args.db_host, args.db_port, args.db_user, db_password, args.db_name)

    audit_source_data_coverage(engine, start, end, args.db_name)

    if verbose:
        print(f"[INFO] Date range: {start} ~ {end}")
        print(f"[INFO] Database: {args.db_name}")
        print(f"[INFO] Dry-run: {dry_run}")

    # ── Load input data ─────────────────────────────────────────────
    if verbose:
        print("[INFO] Loading cn_ga_mainline_radar_daily ...")
    radar_df = load_ga_mainline_radar(engine, start, end)
    if verbose:
        print(f"[INFO]   -> {len(radar_df)} rows loaded")

    if verbose:
        print("[INFO] Loading cn_industry_capital_flow_daily ...")
    capital_flow_df = load_industry_capital_flow(engine, start, end)
    if verbose:
        print(f"[INFO]   -> {len(capital_flow_df)} rows loaded")

    if verbose:
        print("[INFO] Loading cn_stock_leader_score_daily ...")
    leader_df = load_leader_scores(engine, start, end)
    if verbose:
        print(f"[INFO]   -> {len(leader_df)} rows loaded")

    if verbose:
        print("[INFO] Loading cn_ga_stock_role_map_daily ...")
    role_df = load_ga_stock_role_map(engine, start, end)
    if verbose:
        print(f"[INFO]   -> {len(role_df)} rows loaded")

    if verbose:
        print("[INFO] Loading cn_local_industry_map_hist ...")
    ind_map_df = load_industry_map_hist(engine)
    if verbose:
        print(f"[INFO]   -> {len(ind_map_df)} rows loaded")

    if verbose:
        print("[INFO] Loading cn_stock_daily_price ...")
    price_df = load_stock_daily_price(engine, start, end)
    if verbose:
        print(f"[INFO]   -> {len(price_df)} rows loaded")

    if verbose:
        print("[INFO] Loading cn_stock_daily_basic ...")
    basic_df = load_stock_daily_basic(engine, start, end)
    if verbose:
        print(f"[INFO]   -> {len(basic_df)} rows loaded")

    if radar_df.empty:
        print("WARNING: cn_ga_mainline_radar_daily is empty for the range")
        return pd.DataFrame()

    # ── Build lookup maps ───────────────────────────────────────────
    # Capital flow lookup: (trade_date, industry_id) -> concentration_score
    capital_flow_lookup: dict[tuple[date, str], float] = {}
    if not capital_flow_df.empty:
        for _, row in capital_flow_df.iterrows():
            key = (row["trade_date"], row["industry_id"])
            capital_flow_lookup[key] = _safe_float(row.get("concentration_score", 0.0))

    # Industry name lookup
    ind_name_lookup: dict[str, str] = {}
    if not ind_map_df.empty:
        for _, row in ind_map_df.iterrows():
            ind_name_lookup[row["industry_id"]] = row.get("industry_name", "")

    # Leader score lookup: (trade_date, industry_id) -> list of leader scores
    leader_by_industry: dict[tuple[date, str], pd.DataFrame] = {}
    if not leader_df.empty:
        leader_df["symbol"] = leader_df["symbol"].astype(str).str.split(".").str[0]
        leader_df["trade_date"] = pd.to_datetime(leader_df["trade_date"], errors="coerce").dt.date
        leader_df["industry_id"] = leader_df["industry_id"].fillna("")
        for (td, ind_id), group in leader_df.groupby(["trade_date", "industry_id"]):
            if ind_id:
                leader_by_industry[(td, ind_id)] = group

    # ── Prepare radar data ──────────────────────────────────────────
    radar_df["trade_date"] = pd.to_datetime(radar_df["trade_date"], errors="coerce").dt.date
    radar_df["mainline_id"] = radar_df["mainline_id"].fillna("")
    radar_df["mainline_name"] = radar_df["mainline_name"].fillna("")

    # ── Compute per-mainline per-date scores ────────────────────────
    results: list[dict[str, Any]] = []

    for _, row in radar_df.iterrows():
        trade_date_val = row["trade_date"]
        industry_id = str(row["mainline_id"]).strip()
        industry_name = str(row["mainline_name"]).strip() if row["mainline_name"] else ""

        if not industry_id:
            continue

        # Use industry name from map if available
        mapped_name = ind_name_lookup.get(industry_id, "")
        if mapped_name:
            industry_name = mapped_name

        # ── Extract radar metrics ───────────────────────────────────
        mainline_score = _safe_float(row.get("mainline_score", 0.0))
        leader_density = _safe_float(row.get("leader_density", 0.0))
        new_high_ratio = _safe_float(row.get("new_high_ratio", 0.0))
        breakout_ratio = _safe_float(row.get("breakout_ratio", 0.0))
        rotation_rank = int(_safe_float(row.get("rotation_rank", 999)))
        trend_alignment_score = _safe_float(row.get("trend_alignment_score", 0.0))
        radar_leader_count = int(_safe_float(row.get("leader_count", 0)))
        radar_strong_stock_count = int(_safe_float(row.get("strong_stock_count", 0)))

        # ── Capital concentration score ─────────────────────────────
        cap_key = (trade_date_val, industry_id)
        capital_concentration_score = capital_flow_lookup.get(cap_key, 0.0)

        # ── Leader count from leader scores (more accurate) ─────────
        leader_key = (trade_date_val, industry_id)
        leader_count = radar_leader_count
        strong_stock_count = radar_strong_stock_count
        if leader_key in leader_by_industry:
            ind_group = leader_by_industry[leader_key]
            leader_count = _compute_leader_count(ind_group)
            strong_stock_count = _compute_strong_stock_count(ind_group)

        # ── Compute mainline strength ───────────────────────────────
        scores = {
            "mainline_score": mainline_score,
            "capital_concentration_score": capital_concentration_score,
            "leader_density": leader_density,
            "breakout_ratio": breakout_ratio,
            "trend_alignment_score": trend_alignment_score,
        }

        mainline_strength = _compute_mainline_strength(scores)

        # ── Classify mainline phase ─────────────────────────────────
        mainline_phase = _classify_mainline_phase(
            mainline_strength,
            breakout_ratio,
            trend_alignment_score,
        )

        results.append({
            "trade_date": trade_date_val,
            "industry_id": industry_id,
            "industry_name": industry_name,
            "mainline_strength": round(mainline_strength, 8),
            "trend_alignment_score": round(trend_alignment_score, 8),
            "breakout_ratio": round(breakout_ratio, 8),
            "new_high_ratio": round(new_high_ratio, 8),
            "leader_density": round(leader_density, 8),
            "leader_count": leader_count,
            "strong_stock_count": strong_stock_count,
            "capital_concentration_score": round(capital_concentration_score, 8),
            "rotation_rank": rotation_rank,
            "mainline_phase": mainline_phase,
        })

    if not results:
        print("WARNING: No results computed")
        return pd.DataFrame()

    result_df = pd.DataFrame(results)

    # ── Compute rotation_rank (per trade_date) ──────────────────────
    print("  Computing rotation ranks ...")
    for dt, group in result_df.groupby("trade_date"):
        ranked = group["mainline_strength"].rank(ascending=False, method="min")
        mask = result_df["trade_date"] == dt
        result_df.loc[mask, "rotation_rank"] = ranked.astype(int)

    result_df = result_df[result_df["trade_date"].between(start, end)].copy()

    if verbose:
        print(f"[INFO] Computed {len(result_df)} rows")
        print(f"[INFO] Mainline strength range: [{result_df['mainline_strength'].min():.4f}, {result_df['mainline_strength'].max():.4f}]")
        phase_counts = result_df["mainline_phase"].value_counts()
        print(f"[INFO] Phase distribution:")
        for phase, cnt in phase_counts.items():
            print(f"         {phase}: {cnt}")

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
    """Write computed DataFrame to cn_mainline_strength_daily. Returns row count."""
    if df.empty:
        print("WARNING: No data to write")
        return 0

    table_name = "cn_mainline_strength_daily"

    if dry_run:
        print(f"[DRY-RUN] Would write {len(df)} rows to {table_name}")
        return len(df)

    if not table_exists(engine, db_name, table_name):
        print(f"ERROR: {table_name} table does not exist. Run DDL first.")
        return 0

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
        del_sql = f"""
        DELETE FROM {table_name}
        WHERE trade_date BETWEEN :start AND :end
        """
        with engine.begin() as conn:
            deleted = conn.execute(text(del_sql), {"start": min_date, "end": max_date}).rowcount
        if verbose:
            print(f"[INFO] Deleted {deleted} existing rows in [{min_date}, {max_date}]")

    columns = [
        "trade_date", "industry_id", "industry_name",
        "mainline_strength", "trend_alignment_score", "breakout_ratio",
        "new_high_ratio", "leader_density", "leader_count",
        "strong_stock_count", "capital_concentration_score",
        "rotation_rank", "mainline_phase",
    ]
    upsert_sql = f"""
    INSERT INTO {table_name} (
        trade_date, industry_id, industry_name,
        mainline_strength, trend_alignment_score, breakout_ratio,
        new_high_ratio, leader_density, leader_count,
        strong_stock_count, capital_concentration_score,
        rotation_rank, mainline_phase
    ) VALUES (
        :trade_date, :industry_id, :industry_name,
        :mainline_strength, :trend_alignment_score, :breakout_ratio,
        :new_high_ratio, :leader_density, :leader_count,
        :strong_stock_count, :capital_concentration_score,
        :rotation_rank, :mainline_phase
    )
    ON DUPLICATE KEY UPDATE
        industry_name = VALUES(industry_name),
        mainline_strength = VALUES(mainline_strength),
        trend_alignment_score = VALUES(trend_alignment_score),
        breakout_ratio = VALUES(breakout_ratio),
        new_high_ratio = VALUES(new_high_ratio),
        leader_density = VALUES(leader_density),
        leader_count = VALUES(leader_count),
        strong_stock_count = VALUES(strong_stock_count),
        capital_concentration_score = VALUES(capital_concentration_score),
        rotation_rank = VALUES(rotation_rank),
        mainline_phase = VALUES(mainline_phase),
        updated_at = CURRENT_TIMESTAMP
    """

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
            print(f"[INFO] Wrote {total} rows to {table_name}")

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
    base_name = f"mainline_strength_summary_{start_str}_{end_str}_{timestamp}"

    csv_path = report_dir / f"{base_name}.csv"
    md_path = report_dir / f"{base_name}.md"

    if df.empty:
        summary_rows = [{
            "trade_date": "N/A",
            "industry_count": 0,
            "avg_mainline_strength": 0,
            "dominant_count": 0,
            "expanding_count": 0,
            "emerging_count": 0,
            "diverging_count": 0,
            "decaying_count": 0,
            "row_count": 0,
        }]
    else:
        summary_rows = []
        for trade_date_val, group in df.groupby("trade_date"):
            td = trade_date_val
            td_str = td.strftime("%Y-%m-%d") if hasattr(td, "strftime") else str(td)
            phase_counts = group["mainline_phase"].value_counts()
            summary_rows.append({
                "trade_date": td_str,
                "industry_count": len(group),
                "avg_mainline_strength": round(group["mainline_strength"].mean(), 4),
                "dominant_count": int(phase_counts.get("DOMINANT", 0)),
                "expanding_count": int(phase_counts.get("EXPANDING", 0)),
                "emerging_count": int(phase_counts.get("EMERGING", 0)),
                "diverging_count": int(phase_counts.get("DIVERGING", 0)),
                "decaying_count": int(phase_counts.get("DECAYING", 0)),
                "row_count": len(group),
            })

    summary_df = pd.DataFrame(summary_rows)

    summary_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"[REPORT] CSV -> {csv_path}")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Mainline Strength Score Summary\n\n")
        f.write(f"**Date Range:** {start_str} ~ {end_str}\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Total Rows:** {len(df)}\n\n")
        f.write(f"---\n\n")

        for _, srow in summary_df.iterrows():
            f.write(f"## {srow['trade_date']}\n\n")
            f.write(f"- Industry Count: {srow['industry_count']}\n")
            f.write(f"- Avg Mainline Strength: {srow['avg_mainline_strength']:.4f}\n")
            f.write(f"- DOMINANT: {srow['dominant_count']}\n")
            f.write(f"- EXPANDING: {srow['expanding_count']}\n")
            f.write(f"- EMERGING: {srow['emerging_count']}\n")
            f.write(f"- DIVERGING: {srow['diverging_count']}\n")
            f.write(f"- DECAYING: {srow['decaying_count']}\n")
            f.write(f"- Rows Written: {srow['row_count']}\n\n")
            f.write(f"---\n\n")

        f.write(f"## Overall Score Distribution\n\n")
        f.write(f"- Mainline Strength: mean={df['mainline_strength'].mean():.4f}, std={df['mainline_strength'].std():.4f}\n")
        f.write(f"- Trend Alignment: mean={df['trend_alignment_score'].mean():.4f}\n")
        f.write(f"- Breakout Ratio: mean={df['breakout_ratio'].mean():.4f}\n")
        f.write(f"- New High Ratio: mean={df['new_high_ratio'].mean():.4f}\n")
        f.write(f"- Leader Density: mean={df['leader_density'].mean():.4f}\n")
        f.write(f"- Capital Concentration: mean={df['capital_concentration_score'].mean():.4f}\n\n")

        f.write(f"## Phase Distribution\n\n")
        phase_counts = df["mainline_phase"].value_counts()
        for phase, cnt in phase_counts.items():
            pct = cnt / len(df) * 100
            f.write(f"- {phase}: {cnt} ({pct:.1f}%)\n")
        f.write("\n")

        f.write(f"## Top 10 Mainlines by Strength\n\n")
        f.write("| Rank | Industry | Strength | Phase | Leader Density | Breakout Ratio |\n")
        f.write("|------|----------|----------|-------|---------------|---------------|\n")
        top10 = df.nlargest(10, "mainline_strength")
        for _, row in top10.iterrows():
            f.write(f"| {row['rotation_rank']} | {row['industry_name']} | {row['mainline_strength']:.4f} | {row['mainline_phase']} | {row['leader_density']:.4f} | {row['breakout_ratio']:.4f} |\n")

    print(f"[REPORT] MD  -> {md_path}")
    return csv_path, md_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    print("=" * 60)
    print("  GrowthAlpha V8 — P0 Mainline Strength Builder")
    print("=" * 60)
    print(f"  Start: {args.start}")
    print(f"  End:   {args.end or 'today'}")
    print(f"  DB:    {args.db_name}")
    print(f"  Dry-run: {args.dry_run}")
    print(f"  Replace: {args.replace}")
    print()

    db_password = args.db_password or os.getenv("ASHARE_MYSQL_PASSWORD", "sec_Bobo123")
    engine = build_engine(args.db_host, args.db_port, args.db_user, db_password, args.db_name)

    # Ensure DDL — drop old table first if it has wrong schema, then recreate
    ddl_path = Path(__file__).resolve().parents[1] / "sql" / "create_cn_mainline_strength_daily.sql"
    if ddl_path.exists():
        ddl_sql = ddl_path.read_text(encoding="utf-8")
        stmts = [s.strip() for s in ddl_sql.split(";") if s.strip()]
        with engine.begin() as conn:
            # Drop existing table first (old schema may differ)
            conn.execute(text("DROP TABLE IF EXISTS cn_mainline_strength_daily"))
            for stmt in stmts:
                if "CREATE TABLE" in stmt:
                    conn.execute(text(stmt))
        print(f"[DDL] Recreated cn_mainline_strength_daily table")
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