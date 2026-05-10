"""
scripts/build_cn_industry_capital_flow_daily.py
================================================
GrowthAlpha V8 — P3 Unified Alpha Engine — Part C.

Computes industry-level capital flow and concentration metrics daily from:
  1. cn_stock_daily_price          — price/volume data (UPPERCASE columns)
  2. cn_stock_daily_basic          — market cap / turnover data
  3. cn_local_industry_map_hist    — historical industry mapping

Output:
  - cn_industry_capital_flow_daily

Sub-scores (all in [0,1]):
  - flow_strength_score
  - market_share
  - concentration_score
  - rotation_speed_score
  - capital_flow_score (composite)

Usage:
  python scripts/build_cn_industry_capital_flow_daily.py --start 2026-01-01 --end 2026-03-30 --db-name cn_market_red --replace --verbose
  python scripts/build_cn_industry_capital_flow_daily.py --start 2026-01-01 --end 2026-03-30 --db-name cn_market_red --dry-run
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

REPORT_DIR = Path("reports") / "industry_capital_flow"

# Weights for capital_flow_score composite
FLOW_WEIGHTS: dict[str, float] = {
    "flow_strength_score": 0.40,
    "market_share": 0.30,
    "concentration_score": 0.20,
    "rotation_speed_score": 0.10,
}

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="GrowthAlpha V8 — P3 Build cn_industry_capital_flow_daily"
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
    """Load cn_stock_daily_basic for market cap and turnover data."""
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


def load_industry_map_hist(engine: Engine) -> pd.DataFrame:
    """Load cn_local_industry_map_hist for stock-to-industry mapping with date ranges."""
    sql = """
    SELECT
        symbol,
        industry_id,
        industry_name,
        industry_level,
        in_date,
        out_date
    FROM cn_local_industry_map_hist
    WHERE industry_level = 'L1'
    ORDER BY symbol, in_date
    """
    return fetch_df(engine, sql)


# ---------------------------------------------------------------------------
# Score computation
# ---------------------------------------------------------------------------


def _compute_flow_strength_score(
    industry_amount: float,
    industry_turnover: float,
    industry_member_count: int,
    market_total_amount: float,
    market_total_turnover: float,
) -> float:
    """
    Flow strength score [0,1].
    Measures how strong the capital flow is in this industry relative to market.
    Combines amount share and turnover intensity.
    """
    if market_total_amount <= 0 or industry_member_count <= 0:
        return 0.5

    # Amount share relative to market
    amount_share = industry_amount / market_total_amount if market_total_amount > 0 else 0

    # Turnover intensity (per-member turnover relative to market average)
    avg_market_turnover = market_total_turnover / max(1, market_total_turnover)
    turnover_intensity = 0.5  # neutral default

    if market_total_turnover > 0 and industry_member_count > 0:
        industry_turnover_per_member = industry_turnover / industry_member_count
        market_turnover_per_member = market_total_turnover / max(1, market_total_turnover)
        if market_turnover_per_member > 0:
            turnover_ratio = industry_turnover_per_member / market_turnover_per_member
            turnover_intensity = _clip01(np.log1p(turnover_ratio) / np.log(6))

    # Composite: amount share (0-0.6) + turnover intensity (0-0.4)
    amount_component = _clip01(amount_share * 20)  # 5% share -> 1.0
    score = amount_component * 0.6 + turnover_intensity * 0.4
    return _clip01(score)


def _compute_market_share(
    industry_amount: float,
    market_total_amount: float,
) -> float:
    """
    Market share score [0,1].
    Industry's proportion of total market trading amount.
    Normalized using log scale.
    """
    if market_total_amount <= 0 or industry_amount <= 0:
        return 0.0
    ratio = industry_amount / market_total_amount
    # Log scale: ln(1 + ratio * 100) / ln(101)
    share = np.log1p(ratio * 100) / np.log(101)
    return _clip01(share)


def _compute_concentration_score(
    industry_amount: float,
    industry_member_count: int,
    industry_stocks_amounts: list[float],
) -> float:
    """
    Concentration score [0,1].
    Measures how concentrated the capital flow is within the industry.
    Higher = more concentrated (few stocks dominate).
    Uses Herfindahl-Hirschman Index (HHI) normalized.
    """
    if industry_member_count <= 1 or not industry_stocks_amounts:
        return 0.5

    total = sum(industry_stocks_amounts)
    if total <= 0:
        return 0.5

    # Compute HHI: sum of (share)^2
    hhi = sum((a / total) ** 2 for a in industry_stocks_amounts if a > 0)

    # Normalize HHI to [0,1]
    # HHI ranges from 1/N (equal distribution) to 1 (monopoly)
    min_hhi = 1.0 / industry_member_count
    if hhi <= min_hhi:
        return 0.0
    normalized = (hhi - min_hhi) / (1.0 - min_hhi)
    return _clip01(normalized)


def _compute_rotation_speed_score(
    industry_amount: float,
    industry_member_count: int,
    industry_turnover: float,
    prev_industry_amount: float | None,
) -> float:
    """
    Rotation speed score [0,1].
    Measures how quickly capital is rotating through this industry.
    Higher = faster rotation (more speculative / short-term).
    """
    if industry_member_count <= 0:
        return 0.5

    # Turnover per member as a measure of rotation speed
    turnover_per_member = industry_turnover / industry_member_count if industry_member_count > 0 else 0

    # Map to [0,1]: 0% -> 0, 50% -> 0.5, 100%+ -> 1.0
    speed = _clip01(turnover_per_member / 100.0)

    # If we have previous period data, add acceleration component
    if prev_industry_amount is not None and prev_industry_amount > 0:
        amount_change = (industry_amount - prev_industry_amount) / prev_industry_amount
        # Positive change = accelerating rotation
        accel = _clip01((amount_change + 0.5) / 1.0)  # -50% -> 0, +50% -> 1
        speed = speed * 0.7 + accel * 0.3

    return _clip01(speed)


def _compute_capital_flow_score(scores: dict[str, float]) -> float:
    """
    Composite capital flow score [0,1].

    capital_flow_score =
      0.40 * flow_strength_score +
      0.30 * market_share +
      0.20 * concentration_score +
      0.10 * rotation_speed_score
    """
    weighted_sum = 0.0
    for factor_name, weight in FLOW_WEIGHTS.items():
        score = scores.get(factor_name, 0.0)
        weighted_sum += score * weight
    return _clip01(weighted_sum)


# ---------------------------------------------------------------------------
# Main build logic
# ---------------------------------------------------------------------------


def run_build(args: argparse.Namespace) -> pd.DataFrame:
    """
    Execute the industry capital flow build for the given date range.
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

    if verbose:
        print(f"[INFO] Date range: {start} ~ {end}")
        print(f"[INFO] Database: {args.db_name}")
        print(f"[INFO] Dry-run: {dry_run}")

    # ── Load input data ─────────────────────────────────────────────
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

    if verbose:
        print("[INFO] Loading cn_local_industry_map_hist ...")
    ind_map_df = load_industry_map_hist(engine)
    if verbose:
        print(f"[INFO]   -> {len(ind_map_df)} rows loaded")

    if price_df.empty:
        print("WARNING: cn_stock_daily_price is empty for the range")
        return pd.DataFrame()

    if ind_map_df.empty:
        print("WARNING: cn_local_industry_map_hist is empty")
        return pd.DataFrame()

    # ── Prepare industry map ────────────────────────────────────────
    # Build a lookup: for each symbol, find the industry it belongs to on a given date
    ind_map_df["symbol"] = ind_map_df["symbol"].astype(str).str.split(".").str[0]
    ind_map_df["in_date"] = pd.to_datetime(ind_map_df["in_date"], errors="coerce").dt.date
    ind_map_df["out_date"] = pd.to_datetime(ind_map_df["out_date"], errors="coerce").dt.date

    # Build symbol -> list of (industry_id, industry_name, in_date, out_date)
    ind_records_by_symbol: dict[str, list[dict[str, Any]]] = {}
    for _, row in ind_map_df.iterrows():
        sym = row["symbol"]
        if sym not in ind_records_by_symbol:
            ind_records_by_symbol[sym] = []
        ind_records_by_symbol[sym].append({
            "industry_id": row["industry_id"],
            "industry_name": row.get("industry_name", ""),
            "in_date": row["in_date"],
            "out_date": row["out_date"],
        })

    def _find_industry_for_date(symbol: str, trade_date: date) -> tuple[str, str] | None:
        """Find the industry for a symbol on a given trade_date using in_date/out_date ranges."""
        records = ind_records_by_symbol.get(symbol, [])
        for rec in records:
            in_d = rec["in_date"]
            out_d = rec["out_date"]
            if in_d is not None and out_d is not None:
                if in_d <= trade_date and (out_d is None or out_d >= trade_date):
                    return (rec["industry_id"], rec["industry_name"])
            elif in_d is not None:
                if in_d <= trade_date:
                    return (rec["industry_id"], rec["industry_name"])
        return None

    # ── Prepare price data ──────────────────────────────────────────
    price_df["SYMBOL"] = price_df["SYMBOL"].astype(str).str.split(".").str[0]
    price_df["TRADE_DATE"] = pd.to_datetime(price_df["TRADE_DATE"], errors="coerce").dt.date

    # ── Prepare basic data ──────────────────────────────────────────
    if not basic_df.empty:
        basic_df["symbol"] = basic_df["symbol"].astype(str).str.split(".").str[0]
        basic_df["trade_date"] = pd.to_datetime(basic_df["trade_date"], errors="coerce").dt.date

    # Build basic lookup: (trade_date, symbol) -> row
    basic_lookup: dict[tuple[date, str], dict] = {}
    if not basic_df.empty:
        for _, row in basic_df.iterrows():
            basic_lookup[(row["trade_date"], row["symbol"])] = row.to_dict()

    # ── Assign industry to each price row ───────────────────────────
    print("  Assigning industry membership to price data ...")
    price_with_ind: list[dict[str, Any]] = []
    skipped_no_ind = 0
    for _, row in price_df.iterrows():
        sym = row["SYMBOL"]
        td = row["TRADE_DATE"]
        ind_info = _find_industry_for_date(sym, td)
        if ind_info is None:
            skipped_no_ind += 1
            continue
        industry_id, industry_name = ind_info

        # Get basic data
        basic_row = basic_lookup.get((td, sym), {})

        amount = _safe_float(row.get("AMOUNT"), 0.0)
        turnover = _safe_float(row.get("TURNOVER_RATE"), 0.0)
        chg_pct = _safe_float(row.get("CHG_PCT"), 0.0)
        volume_ratio = _safe_float(basic_row.get("volume_ratio"), 0.0)

        price_with_ind.append({
            "trade_date": td,
            "symbol": sym,
            "industry_id": industry_id,
            "industry_name": industry_name,
            "amount": amount,
            "turnover": turnover,
            "chg_pct": chg_pct,
            "volume_ratio": volume_ratio,
        })

    if skipped_no_ind > 0 and verbose:
        print(f"[INFO] Skipped {skipped_no_ind} price rows without industry mapping")

    if not price_with_ind:
        print("WARNING: No price data with industry mapping")
        return pd.DataFrame()

    ind_price_df = pd.DataFrame(price_with_ind)

    # ── Compute per-industry per-date aggregates ────────────────────
    print("  Computing industry-level aggregates ...")
    results: list[dict[str, Any]] = []

    # Track previous period amounts for rotation speed
    prev_amount_by_industry: dict[str, float] = {}

    grouped = ind_price_df.groupby(["trade_date", "industry_id"])

    for (trade_date_val, industry_id), group in grouped:
        industry_name = group["industry_name"].iloc[0]

        total_amount = group["amount"].sum()
        total_turnover = group["turnover"].sum()
        avg_chg_pct = group["chg_pct"].mean()
        avg_volume_ratio = group["volume_ratio"].mean()
        member_count = len(group)
        stock_amounts = group["amount"].tolist()

        # Market-wide totals for this date
        date_mask = ind_price_df["trade_date"] == trade_date_val
        market_total_amount = ind_price_df.loc[date_mask, "amount"].sum()
        market_total_turnover = ind_price_df.loc[date_mask, "turnover"].sum()

        # ── Compute sub-scores ──────────────────────────────────────
        flow_strength_score = _compute_flow_strength_score(
            total_amount, total_turnover, member_count,
            market_total_amount, market_total_turnover,
        )

        market_share = _compute_market_share(total_amount, market_total_amount)

        concentration_score = _compute_concentration_score(
            total_amount, member_count, stock_amounts,
        )

        prev_amount = prev_amount_by_industry.get(industry_id)
        rotation_speed_score = _compute_rotation_speed_score(
            total_amount, member_count, total_turnover, prev_amount,
        )

        # Update previous amount for next date
        prev_amount_by_industry[industry_id] = total_amount

        scores = {
            "flow_strength_score": flow_strength_score,
            "market_share": market_share,
            "concentration_score": concentration_score,
            "rotation_speed_score": rotation_speed_score,
        }

        capital_flow_score = _compute_capital_flow_score(scores)

        # Compute amount rank across industries for this date
        results.append({
            "trade_date": trade_date_val,
            "industry_id": industry_id,
            "industry_name": industry_name,
            "total_amount": total_amount,
            "total_turnover": total_turnover,
            "avg_change_pct": round(avg_chg_pct, 6),
            "volume_ratio": round(avg_volume_ratio, 6),
            "market_share": round(market_share, 8),
            "amount_rank": 0,  # will compute below
            "flow_strength_score": round(flow_strength_score, 8),
            "rotation_speed_score": round(rotation_speed_score, 8),
            "concentration_score": round(concentration_score, 8),
            "capital_flow_score": round(capital_flow_score, 8),
        })

    if not results:
        print("WARNING: No results computed")
        return pd.DataFrame()

    result_df = pd.DataFrame(results)

    # ── Compute amount_rank (per trade_date) ────────────────────────
    print("  Computing amount ranks ...")
    result_df["amount_rank"] = 0
    for dt, group in result_df.groupby("trade_date"):
        ranked = group["total_amount"].rank(ascending=False, method="min")
        mask = result_df["trade_date"] == dt
        result_df.loc[mask, "amount_rank"] = ranked.astype(int)

    # Filter to requested date range
    result_df = result_df[result_df["trade_date"].between(start, end)].copy()

    if verbose:
        print(f"[INFO] Computed {len(result_df)} rows")
        print(f"[INFO] Capital flow score range: [{result_df['capital_flow_score'].min():.4f}, {result_df['capital_flow_score'].max():.4f}]")

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
    """Write computed DataFrame to cn_industry_capital_flow_daily. Returns row count."""
    if df.empty:
        print("WARNING: No data to write")
        return 0

    table_name = "cn_industry_capital_flow_daily"

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
        "total_amount", "total_turnover", "avg_change_pct",
        "volume_ratio", "market_share", "amount_rank",
        "flow_strength_score", "rotation_speed_score",
        "concentration_score", "capital_flow_score",
    ]
    upsert_sql = f"""
    INSERT INTO {table_name} (
        trade_date, industry_id, industry_name,
        total_amount, total_turnover, avg_change_pct,
        volume_ratio, market_share, amount_rank,
        flow_strength_score, rotation_speed_score,
        concentration_score, capital_flow_score
    ) VALUES (
        :trade_date, :industry_id, :industry_name,
        :total_amount, :total_turnover, :avg_change_pct,
        :volume_ratio, :market_share, :amount_rank,
        :flow_strength_score, :rotation_speed_score,
        :concentration_score, :capital_flow_score
    )
    ON DUPLICATE KEY UPDATE
        industry_name = VALUES(industry_name),
        total_amount = VALUES(total_amount),
        total_turnover = VALUES(total_turnover),
        avg_change_pct = VALUES(avg_change_pct),
        volume_ratio = VALUES(volume_ratio),
        market_share = VALUES(market_share),
        amount_rank = VALUES(amount_rank),
        flow_strength_score = VALUES(flow_strength_score),
        rotation_speed_score = VALUES(rotation_speed_score),
        concentration_score = VALUES(concentration_score),
        capital_flow_score = VALUES(capital_flow_score),
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
    base_name = f"industry_capital_flow_summary_{start_str}_{end_str}_{timestamp}"

    csv_path = report_dir / f"{base_name}.csv"
    md_path = report_dir / f"{base_name}.md"

    if df.empty:
        summary_rows = [{
            "trade_date": "N/A",
            "industry_count": 0,
            "avg_capital_flow_score": 0,
            "avg_flow_strength": 0,
            "avg_market_share": 0,
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
                "avg_capital_flow_score": round(group["capital_flow_score"].mean(), 4),
                "avg_flow_strength": round(group["flow_strength_score"].mean(), 4),
                "avg_market_share": round(group["market_share"].mean(), 4),
                "row_count": len(group),
            })

    summary_df = pd.DataFrame(summary_rows)

    summary_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"[REPORT] CSV -> {csv_path}")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Industry Capital Flow Summary\n\n")
        f.write(f"**Date Range:** {start_str} ~ {end_str}\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Total Rows:** {len(df)}\n\n")
        f.write(f"---\n\n")

        for _, srow in summary_df.iterrows():
            f.write(f"## {srow['trade_date']}\n\n")
            f.write(f"- Industry Count: {srow['industry_count']}\n")
            f.write(f"- Avg Capital Flow Score: {srow['avg_capital_flow_score']:.4f}\n")
            f.write(f"- Avg Flow Strength: {srow['avg_flow_strength']:.4f}\n")
            f.write(f"- Avg Market Share: {srow['avg_market_share']:.4f}\n")
            f.write(f"- Rows Written: {srow['row_count']}\n\n")
            f.write(f"---\n\n")

        f.write(f"## Overall Score Distribution\n\n")
        f.write(f"- Capital Flow Score: mean={df['capital_flow_score'].mean():.4f}, std={df['capital_flow_score'].std():.4f}\n")
        f.write(f"- Flow Strength: mean={df['flow_strength_score'].mean():.4f}\n")
        f.write(f"- Market Share: mean={df['market_share'].mean():.4f}\n")
        f.write(f"- Concentration: mean={df['concentration_score'].mean():.4f}\n")
        f.write(f"- Rotation Speed: mean={df['rotation_speed_score'].mean():.4f}\n\n")

        f.write(f"### Top 10 Industries by Capital Flow\n\n")
        f.write("| Rank | Industry | Flow Score | Flow Strength | Market Share | Concentration |\n")
        f.write("|------|----------|------------|---------------|--------------|---------------|\n")
        top10 = df.nlargest(10, "capital_flow_score")
        for _, row in top10.iterrows():
            f.write(f"| {row['amount_rank']} | {row['industry_name']} | {row['capital_flow_score']:.4f} | "
                    f"{row['flow_strength_score']:.4f} | {row['market_share']:.4f} | {row['concentration_score']:.4f} |\n")

    print(f"[REPORT] MD  -> {md_path}")
    return csv_path, md_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    print("=" * 60)
    print("  GrowthAlpha V8 — P3 Industry Capital Flow Builder")
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
    ddl_path = Path(__file__).resolve().parents[1] / "sql" / "create_industry_capital_flow_daily.sql"
    if ddl_path.exists():
        ddl_sql = ddl_path.read_text(encoding="utf-8")
        stmts = [s.strip() for s in ddl_sql.split(";") if s.strip()]
        with engine.begin() as conn:
            for stmt in stmts:
                if "CREATE TABLE" in stmt:
                    conn.execute(text(stmt))
        print(f"[DDL] Ensured cn_industry_capital_flow_daily table exists")
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
