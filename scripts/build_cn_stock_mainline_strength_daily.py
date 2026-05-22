"""
scripts/build_cn_stock_mainline_strength_daily.py
==================================================
GrowthAlpha V8 — P3 Unified Alpha Engine — Part B.

Computes industry-level mainline strength scores daily from:
  1. cn_stock_leader_score_daily   — per-stock leader scores
  2. cn_mainline_lifecycle_daily   — lifecycle state per mainline
  3. cn_board_member_map_d         — stock-to-industry membership
  4. cn_local_industry_map_hist    — historical industry mapping
  5. cn_stock_daily_price          — price/volume data (UPPERCASE columns)
  6. cn_stock_daily_basic          — market cap / turnover data
  7. cn_ga_mainline_radar_daily    — (optional) radar data

Output:
  - cn_stock_mainline_strength_daily

Sub-scores (all in [0,1]):
  - leader_density
  - avg_leader_score
  - top_leader_score
  - breakout_ratio
  - trend_alignment
  - breadth_score
  - acceleration_score
  - lifecycle_bonus
  - mainline_strength_score (composite)

Usage:
  python scripts/build_cn_stock_mainline_strength_daily.py --start 2026-01-01 --end 2026-03-30 --db-name cn_market_red --replace --verbose
  python scripts/build_cn_stock_mainline_strength_daily.py --start 2026-01-01 --end 2026-03-30 --db-name cn_market_red --dry-run
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

# Weights for mainline_strength_score composite
STRENGTH_WEIGHTS: dict[str, float] = {
    "avg_leader_score": 0.25,
    "leader_density": 0.20,
    "breakout_ratio": 0.15,
    "trend_alignment": 0.15,
    "breadth_score": 0.10,
    "acceleration_score": 0.10,
    "lifecycle_bonus": 0.05,
}

# Active mainline threshold
ACTIVE_MAINLINE_THRESHOLD = 0.65

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="GrowthAlpha V8 — P3 Build cn_stock_mainline_strength_daily"
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
    parser.add_argument("--chunk-months", type=int, default=1, help="Months per processing chunk (default=1)")
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


def load_leader_scores(engine: Engine, start: date, end: date) -> pd.DataFrame:
    """Load cn_stock_leader_score_daily for the date range."""
    sql = """
    SELECT
        trade_date,
        symbol,
        leader_score,
        leader_bucket,
        leader_structural,
        leader_liquidity,
        leader_trend,
        breakout_strength,
        breakout_ready,
        industry_id,
        industry_name
    FROM cn_stock_leader_score_daily
    WHERE trade_date BETWEEN :start AND :end
    ORDER BY trade_date, industry_id, leader_score DESC
    """
    return fetch_df(engine, sql, {"start": start, "end": end})


def load_mainline_lifecycle(engine: Engine, start: date, end: date) -> pd.DataFrame:
    """Load cn_mainline_lifecycle_daily for lifecycle state data."""
    sql = """
    SELECT
        trade_date,
        mainline_id,
        mainline_name,
        lifecycle_state,
        lifecycle_score,
        mainline_strength,
        capital_concentration_score,
        trend_alignment_score,
        breakout_ratio,
        new_high_ratio,
        leader_density,
        rotation_rank,
        risk_flag
    FROM cn_mainline_lifecycle_daily
    WHERE trade_date BETWEEN :start AND :end
    ORDER BY trade_date, mainline_id
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


def load_ga_mainline_radar(engine: Engine, start: date, end: date) -> pd.DataFrame:
    """Load cn_ga_mainline_radar_daily (optional)."""
    sql = """
    SELECT
        trade_date,
        mainline_id,
        mainline_name,
        mainline_state,
        leader_density,
        new_high_ratio,
        breakout_ratio,
        rotation_rank,
        trend_alignment_score,
        capital_concentration_score,
        strength_trend
    FROM cn_ga_mainline_radar_daily
    WHERE trade_date BETWEEN :start AND :end
    ORDER BY trade_date, mainline_id
    """
    return fetch_df(engine, sql, {"start": start, "end": end})


# ---------------------------------------------------------------------------
# Score computation
# ---------------------------------------------------------------------------


def _compute_leader_density(
    industry_group: pd.DataFrame,
    leader_score_col: str = "leader_score",
    threshold: float = 1.0,
) -> float:
    """Proportion of stocks in an industry that are leaders (leader_score >= threshold)."""
    total = len(industry_group)
    if total == 0:
        return 0.0
    leaders = industry_group[leader_score_col].apply(lambda x: _safe_float(x) >= threshold).sum()
    return _clip01(leaders / total)


def _compute_avg_leader_score(industry_group: pd.DataFrame, leader_score_col: str = "leader_score") -> float:
    """Average leader score across all stocks in the industry."""
    scores = industry_group[leader_score_col].apply(_safe_float)
    if len(scores) == 0:
        return 0.0
    return _clip01(scores.mean() / 3.0)  # leader_score is 0-3, normalize to 0-1


def _compute_top_leader_score(industry_group: pd.DataFrame, leader_score_col: str = "leader_score") -> float:
    """Maximum leader score in the industry."""
    scores = industry_group[leader_score_col].apply(_safe_float)
    if len(scores) == 0:
        return 0.0
    return _clip01(scores.max() / 3.0)


def _compute_breakout_ratio(industry_group: pd.DataFrame) -> float:
    """Proportion of stocks with breakout_ready=1 or breakout_strength > 0."""
    total = len(industry_group)
    if total == 0:
        return 0.0
    # Vectorized: avoid iterrows
    br = pd.to_numeric(
        industry_group.get("breakout_ready", pd.Series(0, index=industry_group.index)),
        errors="coerce",
    ).fillna(0.0)
    bs = pd.to_numeric(
        industry_group.get("breakout_strength", pd.Series(0, index=industry_group.index)),
        errors="coerce",
    ).fillna(0.0)
    breakout_count = ((br >= 1) | (bs > 0)).sum()
    return _clip01(breakout_count / total)


def _compute_trend_alignment(industry_group: pd.DataFrame) -> float:
    """Proportion of stocks with positive chg_pct (pre-merged into group)."""
    total = len(industry_group)
    if total == 0:
        return 0.5
    col = "price_chg_pct" if "price_chg_pct" in industry_group.columns else "chg_pct_fallback"
    if col not in industry_group.columns:
        return 0.5
    chg = industry_group[col].fillna(0).astype(float)
    return _clip01((chg > 0).sum() / total)


def _compute_breadth_score(member_count: int, total_market_stocks: int) -> float:
    """Industry member count relative to total market, log-normalised."""
    if total_market_stocks <= 0 or member_count <= 0:
        return 0.0
    ratio = member_count / total_market_stocks
    return _clip01(np.log1p(ratio * 100) / np.log(101))


def _compute_acceleration_score(industry_group: pd.DataFrame) -> float:
    """Average chg_pct mapped to [0,1]. Uses pre-merged price_chg_pct column."""
    col = "price_chg_pct" if "price_chg_pct" in industry_group.columns else None
    if col is None:
        return 0.5
    chg_values = industry_group[col].dropna().astype(float)
    if chg_values.empty:
        return 0.5
    return _clip01((chg_values.mean() + 10.0) / 20.0)


def _compute_lifecycle_bonus(lifecycle_state: str | None) -> float:
    """
    Lifecycle bonus based on lifecycle state:
      TREND_EXPANSION -> 1.0
      DIFFUSION       -> 0.8
      BOTTOM_REPAIR   -> 0.5
      DIVERGENCE      -> 0.3
      TOP_DECAY       -> 0.1
      RISK_OFF        -> 0.0
      None/UNKNOWN    -> 0.5
    """
    state_map = {
        "TREND_EXPANSION": 1.0,
        "DIFFUSION": 0.8,
        "BOTTOM_REPAIR": 0.5,
        "DIVERGENCE": 0.3,
        "TOP_DECAY": 0.1,
        "RISK_OFF": 0.0,
    }
    if lifecycle_state is None:
        return 0.5
    return state_map.get(lifecycle_state.upper(), 0.5)


def _compute_mainline_strength_score(scores: dict[str, float]) -> float:
    """
    Composite mainline strength score [0,1].

    mainline_strength_score =
      0.25 * avg_leader_score +
      0.20 * leader_density +
      0.15 * breakout_ratio +
      0.15 * trend_alignment +
      0.10 * breadth_score +
      0.10 * acceleration_score +
      0.05 * lifecycle_bonus
    """
    weighted_sum = 0.0
    for factor_name, weight in STRENGTH_WEIGHTS.items():
        score = scores.get(factor_name, 0.0)
        weighted_sum += score * weight
    return _clip01(weighted_sum)


# ---------------------------------------------------------------------------
# Main build logic
# ---------------------------------------------------------------------------


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


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


def _process_chunk(
    leader_df: pd.DataFrame,
    price_df: pd.DataFrame,
    lifecycle_df: pd.DataFrame,
    ind_name_lookup: dict[str, str],
) -> pd.DataFrame:
    """Compute mainline strength scores for one chunk. Returns result DataFrame."""
    if leader_df.empty:
        return pd.DataFrame()

    # Normalise leader_df
    leader_df = leader_df.copy()
    leader_df["symbol"] = leader_df["symbol"].astype(str).str.split(".").str[0]
    leader_df["trade_date"] = pd.to_datetime(leader_df["trade_date"], errors="coerce").dt.date
    leader_df["industry_id"] = leader_df["industry_id"].fillna("")
    leader_df = leader_df[leader_df["industry_id"] != ""]

    # ── Merge price CHG_PCT into leader_df (vectorised, no iterrows) ──
    if not price_df.empty:
        price_slim = price_df[["SYMBOL", "TRADE_DATE", "CHG_PCT"]].copy()
        price_slim.columns = ["symbol", "trade_date", "price_chg_pct"]
        price_slim["trade_date"] = pd.to_datetime(price_slim["trade_date"], errors="coerce").dt.date
        leader_df = leader_df.merge(price_slim, on=["symbol", "trade_date"], how="left")
    else:
        leader_df["price_chg_pct"] = np.nan

    # ── Lifecycle lookup: (trade_date, mainline_id) → lifecycle_state ──
    lifecycle_lookup: dict[tuple, str] = {}
    if not lifecycle_df.empty:
        for _, row in lifecycle_df.iterrows():
            lifecycle_lookup[(row["trade_date"], row["mainline_id"])] = str(row.get("lifecycle_state", ""))

    # ── Pre-compute total stocks per date (for breadth_score) ───────
    stocks_per_date: dict[date, int] = leader_df.groupby("trade_date")["symbol"].count().to_dict()

    # ── Group and compute ────────────────────────────────────────────
    grouped = leader_df.groupby(["trade_date", "industry_id"])
    n_groups = len(grouped)
    report_every = max(50, n_groups // 20)

    results: list[dict[str, Any]] = []

    for g_idx, ((trade_date_val, industry_id), group) in enumerate(grouped, 1):
        industry_name = ind_name_lookup.get(industry_id, "")
        if not industry_name and "industry_name" in group.columns:
            industry_name = group["industry_name"].iloc[0] or ""

        lc_state = lifecycle_lookup.get((trade_date_val, industry_id))

        leader_density    = _compute_leader_density(group)
        avg_leader_score  = _compute_avg_leader_score(group)
        top_leader_score  = _compute_top_leader_score(group)
        breakout_ratio    = _compute_breakout_ratio(group)
        trend_alignment   = _compute_trend_alignment(group)
        breadth_score     = _compute_breadth_score(len(group), stocks_per_date.get(trade_date_val, 1))
        acceleration_score = _compute_acceleration_score(group)
        lifecycle_bonus   = _compute_lifecycle_bonus(lc_state)

        mainline_strength_score = _compute_mainline_strength_score({
            "avg_leader_score": avg_leader_score,
            "leader_density": leader_density,
            "breakout_ratio": breakout_ratio,
            "trend_alignment": trend_alignment,
            "breadth_score": breadth_score,
            "acceleration_score": acceleration_score,
            "lifecycle_bonus": lifecycle_bonus,
        })

        results.append({
            "trade_date": trade_date_val,
            "industry_id": industry_id,
            "industry_name": industry_name,
            "mainline_strength_score": round(mainline_strength_score, 8),
            "leader_density": round(leader_density, 8),
            "avg_leader_score": round(avg_leader_score, 8),
            "top_leader_score": round(top_leader_score, 8),
            "breakout_ratio": round(breakout_ratio, 8),
            "trend_alignment": round(trend_alignment, 8),
            "breadth_score": round(breadth_score, 8),
            "acceleration_score": round(acceleration_score, 8),
            "lifecycle_bonus": round(lifecycle_bonus, 8),
        })

        if g_idx % report_every == 0:
            print(f"    [{_ts()}] computed {g_idx:,}/{n_groups:,} groups"
                  f"  ({g_idx * 100 // n_groups}%)"
                  f"  last_date={trade_date_val}")

    if not results:
        return pd.DataFrame()

    result_df = pd.DataFrame(results)

    # rank_in_market and is_active_mainline
    result_df["rank_in_market"] = (
        result_df.groupby("trade_date")["mainline_strength_score"]
        .rank(ascending=False, method="min")
        .astype(int)
    )
    result_df["is_active_mainline"] = (
        result_df["mainline_strength_score"] >= ACTIVE_MAINLINE_THRESHOLD
    ).astype(int)

    return result_df


def run_build(args: argparse.Namespace) -> pd.DataFrame:
    """
    Execute the mainline strength build in monthly chunks with progress output.
    Writes to DB per-chunk. Returns the combined DataFrame of all chunks.
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

    chunk_months = getattr(args, "chunk_months", 1)
    chunks = _date_chunks(start, end, chunk_months)
    n_chunks = len(chunks)

    print(f"[{_ts()}] Date range: {start} ~ {end}  ({n_chunks} chunks of {chunk_months} month(s))")

    # Industry name map — small, load once
    print(f"[{_ts()}] Loading industry name map ...")
    ind_map_df = load_industry_map_hist(engine)
    ind_name_lookup: dict[str, str] = {}
    if not ind_map_df.empty:
        for _, row in ind_map_df.iterrows():
            ind_name_lookup[row["industry_id"]] = row.get("industry_name", "")
    print(f"[{_ts()}]   -> {len(ind_name_lookup):,} industries")

    total_written = 0

    for chunk_idx, (chunk_start, chunk_end) in enumerate(chunks, 1):
        print()
        print(f"[{_ts()}] ── Chunk {chunk_idx}/{n_chunks}: {chunk_start} ~ {chunk_end} ──")
        t0 = datetime.now()

        # Load sources for this chunk
        t1 = datetime.now()
        print(f"  [{_ts()}] Loading leader scores ...")
        leader_df = load_leader_scores(engine, chunk_start, chunk_end)
        print(f"  [{_ts()}]   -> {len(leader_df):,} rows  ({(datetime.now()-t1).seconds}s)")

        if leader_df.empty:
            print(f"  [{_ts()}] No leader data, skip")
            continue

        t2 = datetime.now()
        print(f"  [{_ts()}] Loading price data ...")
        price_df = load_stock_daily_price(engine, chunk_start, chunk_end)
        print(f"  [{_ts()}]   -> {len(price_df):,} rows  ({(datetime.now()-t2).seconds}s)")

        t3 = datetime.now()
        print(f"  [{_ts()}] Loading lifecycle data ...")
        try:
            lifecycle_df = load_mainline_lifecycle(engine, chunk_start, chunk_end)
            print(f"  [{_ts()}]   -> {len(lifecycle_df):,} rows  ({(datetime.now()-t3).seconds}s)")
        except Exception as exc:
            lifecycle_df = pd.DataFrame()
            print(f"  [{_ts()}]   -> not available ({exc})")

        # Compute
        t4 = datetime.now()
        print(f"  [{_ts()}] Computing scores ...")
        chunk_df = _process_chunk(leader_df, price_df, lifecycle_df, ind_name_lookup)

        elapsed_compute = (datetime.now() - t4).seconds
        if chunk_df.empty:
            print(f"  [{_ts()}] No output for this chunk")
            continue

        dates_done = chunk_df["trade_date"].nunique()
        rows_out = len(chunk_df)
        print(f"  [{_ts()}] Computed {rows_out:,} rows  ({dates_done} dates)  ({elapsed_compute}s)")

        # Write
        if not dry_run:
            t5 = datetime.now()
            written = write_to_db(engine, chunk_df, args.db_name, args.replace, dry_run, verbose)
            total_written += written
            print(f"  [{_ts()}] Wrote {written:,} rows to DB  ({(datetime.now()-t5).seconds}s)")
        else:
            total_written += rows_out
            print(f"  [{_ts()}] [dry-run] would write {rows_out:,} rows")

        # Free chunk memory before next iteration
        del chunk_df

        chunk_elapsed = (datetime.now() - t0).seconds
        remaining_est = (n_chunks - chunk_idx) * chunk_elapsed
        print(f"  [{_ts()}] Chunk done in {chunk_elapsed}s  ~{remaining_est // 60}m remaining")

    if total_written == 0:
        return pd.DataFrame()

    print()
    print(f"[{_ts()}] All chunks complete. Total rows written: {total_written:,}")
    return pd.DataFrame()


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
    """Write computed DataFrame to cn_stock_mainline_strength_daily. Returns row count."""
    if df.empty:
        print("WARNING: No data to write")
        return 0

    table_name = "cn_stock_mainline_strength_daily"

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
        "mainline_strength_score", "leader_density", "avg_leader_score",
        "top_leader_score", "breakout_ratio", "trend_alignment",
        "breadth_score", "acceleration_score", "lifecycle_bonus",
        "rank_in_market", "is_active_mainline",
    ]
    upsert_sql = f"""
    INSERT INTO {table_name} (
        trade_date, industry_id, industry_name,
        mainline_strength_score, leader_density, avg_leader_score,
        top_leader_score, breakout_ratio, trend_alignment,
        breadth_score, acceleration_score, lifecycle_bonus,
        rank_in_market, is_active_mainline
    ) VALUES (
        :trade_date, :industry_id, :industry_name,
        :mainline_strength_score, :leader_density, :avg_leader_score,
        :top_leader_score, :breakout_ratio, :trend_alignment,
        :breadth_score, :acceleration_score, :lifecycle_bonus,
        :rank_in_market, :is_active_mainline
    )
    ON DUPLICATE KEY UPDATE
        industry_name = VALUES(industry_name),
        mainline_strength_score = VALUES(mainline_strength_score),
        leader_density = VALUES(leader_density),
        avg_leader_score = VALUES(avg_leader_score),
        top_leader_score = VALUES(top_leader_score),
        breakout_ratio = VALUES(breakout_ratio),
        trend_alignment = VALUES(trend_alignment),
        breadth_score = VALUES(breadth_score),
        acceleration_score = VALUES(acceleration_score),
        lifecycle_bonus = VALUES(lifecycle_bonus),
        rank_in_market = VALUES(rank_in_market),
        is_active_mainline = VALUES(is_active_mainline),
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
            "active_mainline_count": 0,
            "avg_leader_density": 0,
            "avg_breakout_ratio": 0,
            "row_count": 0,
        }]
    else:
        summary_rows = []
        for trade_date_val, group in df.groupby("trade_date"):
            td = trade_date_val
            td_str = td.strftime("%Y-%m-%d") if hasattr(td, "strftime") else str(td)
            summary_rows.append({
                "trade_date": td_str,
                "industry_count": len(group),
                "avg_mainline_strength": round(group["mainline_strength_score"].mean(), 4),
                "active_mainline_count": int(group["is_active_mainline"].sum()),
                "avg_leader_density": round(group["leader_density"].mean(), 4),
                "avg_breakout_ratio": round(group["breakout_ratio"].mean(), 4),
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
            f.write(f"- Active Mainlines: {srow['active_mainline_count']}\n")
            f.write(f"- Avg Leader Density: {srow['avg_leader_density']:.4f}\n")
            f.write(f"- Avg Breakout Ratio: {srow['avg_breakout_ratio']:.4f}\n")
            f.write(f"- Rows Written: {srow['row_count']}\n\n")
            f.write(f"---\n\n")

        f.write(f"## Overall Score Distribution\n\n")
        f.write(f"- Mainline Strength: mean={df['mainline_strength_score'].mean():.4f}, std={df['mainline_strength_score'].std():.4f}\n")
        f.write(f"- Leader Density: mean={df['leader_density'].mean():.4f}\n")
        f.write(f"- Avg Leader Score: mean={df['avg_leader_score'].mean():.4f}\n")
        f.write(f"- Breakout Ratio: mean={df['breakout_ratio'].mean():.4f}\n")
        f.write(f"- Trend Alignment: mean={df['trend_alignment'].mean():.4f}\n")
        f.write(f"- Breadth Score: mean={df['breadth_score'].mean():.4f}\n")
        f.write(f"- Acceleration Score: mean={df['acceleration_score'].mean():.4f}\n\n")

        f.write(f"## Active Mainlines Summary\n\n")
        active_df = df[df["is_active_mainline"] == 1]
        f.write(f"- Active Mainlines: {len(active_df)} ({len(active_df)/len(df)*100:.1f}% of total)\n\n")
        if not active_df.empty:
            f.write("### Top 10 Active Mainlines by Strength\n\n")
            f.write("| Rank | Industry | Strength Score | Leader Density | Breakout Ratio |\n")
            f.write("|------|----------|---------------|---------------|---------------|\n")
            top_active = active_df.nlargest(10, "mainline_strength_score")
            for _, row in top_active.iterrows():
                f.write(f"| {row['rank_in_market']} | {row['industry_name']} | {row['mainline_strength_score']:.4f} | {row['leader_density']:.4f} | {row['breakout_ratio']:.4f} |\n")

    print(f"[REPORT] MD  -> {md_path}")
    return csv_path, md_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    print("=" * 60)
    print("  GrowthAlpha V8 — P3 Mainline Strength Builder")
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
    ddl_path = Path(__file__).resolve().parents[1] / "sql" / "create_cn_stock_mainline_strength_daily.sql"
    if ddl_path.exists():
        ddl_sql = ddl_path.read_text(encoding="utf-8")
        stmts = [s.strip() for s in ddl_sql.split(";") if s.strip()]
        with engine.begin() as conn:
            for stmt in stmts:
                if "CREATE TABLE" in stmt:
                    conn.execute(text(stmt))
        print(f"[DDL] Ensured cn_stock_mainline_strength_daily table exists")
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
