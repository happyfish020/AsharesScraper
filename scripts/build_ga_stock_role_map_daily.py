"""
scripts/build_ga_stock_role_map_daily.py
========================================
GrowthAlpha V8 — GA Layer — Part A.

Builds cn_ga_stock_role_map_daily from cn_stock_leader_score_daily.

Maps each stock's role within its mainline (industry) on each trade date.

Role mapping from leader_bucket (short form in source table):
  CORE  → stock_role=LEADER    (structural + liquidity + trend leader)
  NEAR  → stock_role=CORE      (strong but not full leader)
  EDGE  → stock_role=MOMENTUM  (marginal leadership)
  NON   → stock_role=NON_CORE  (non-leader)

role_score (0-100) = leader_component * 40 + rs_component * 35 + turnover_component * 25

Output:
  - cn_ga_stock_role_map_daily

Run order: before build_ga_mainline_radar_daily.py

Usage:
  python scripts/build_ga_stock_role_map_daily.py \\
      --start 2010-01-01 --end 2026-03-31 \\
      --db-name cn_market_red --replace

  python scripts/build_ga_stock_role_map_daily.py \\
      --start 2010-01-01 --end 2026-03-31 \\
      --db-name cn_market_red --dry-run --verbose
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, URL

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHUNK_MONTHS = 3

BUCKET_TO_ROLE: dict[str, str] = {
    "CORE": "LEADER",
    "NEAR": "CORE",
    "EDGE": "MOMENTUM",
    "NON": "NON_CORE",
}

BUCKET_FULL_NAME: dict[str, str] = {
    "CORE": "CORE_LEADER",
    "NEAR": "NEAR_LEADER",
    "EDGE": "EDGE_LEADER",
    "NON": "NON_LEADER",
}

# role_score weights (sum = 100)
W_LEADER = 40.0
W_RS = 35.0
W_TURNOVER = 25.0

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
        description="GrowthAlpha V8 — Build cn_ga_stock_role_map_daily"
    )
    p.add_argument("--start", default="2010-01-01", help="Start date YYYY-MM-DD")
    p.add_argument("--end", default="", help="End date YYYY-MM-DD (default=today)")
    p.add_argument("--db-name", default="cn_market_red")
    p.add_argument("--db-host", default="127.0.0.1")
    p.add_argument("--db-port", type=int, default=3306)
    p.add_argument("--db-user", default="cn_opr_red")
    p.add_argument("--db-password", default=None)
    p.add_argument("--dry-run", action="store_true", help="Compute but do not write")
    p.add_argument("--replace", action="store_true", help="Delete existing rows before insert")
    p.add_argument("--chunk-months", type=int, default=CHUNK_MONTHS)
    p.add_argument("--verbose", action="store_true")
    return p


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------


def build_engine(host: str, port: int, user: str, password: str, db: str) -> Engine:
    url = URL.create(
        "mysql+pymysql",
        username=user,
        password=password,
        host=host,
        port=port,
        database=db,
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


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_leader_scores(engine: Engine, start: date, end: date) -> pd.DataFrame:
    """Load cn_stock_leader_score_daily for the date range."""
    sql = """
        SELECT
            trade_date,
            symbol,
            name,
            industry_id,
            industry_name,
            leader_score,
            leader_bucket,
            rs_percentile,
            turnover_20d_percentile,
            turnover_rank_in_industry,
            industry_members
        FROM cn_stock_leader_score_daily
        WHERE trade_date BETWEEN :start AND :end
          AND industry_id IS NOT NULL
          AND industry_id != ''
    """
    df = fetch_df(engine, sql, {"start": start, "end": end})
    if df.empty:
        return df

    for col in ["rs_percentile", "turnover_20d_percentile"]:
        df[col] = df[col].apply(lambda x: _safe_float(x, 0.0))
    df["leader_score"] = df["leader_score"].apply(lambda x: _safe_float(x, 0.0))
    df["leader_bucket"] = df["leader_bucket"].fillna("NON").str.strip().str.upper()
    return df


# ---------------------------------------------------------------------------
# Score computation
# ---------------------------------------------------------------------------


def _compute_role_score(
    leader_score: float,
    rs_pct: float,
    turn_pct: float,
) -> float:
    """role_score in [0, 100]."""
    leader_component = (leader_score / 3.0) * W_LEADER
    rs_component = rs_pct * W_RS
    turn_component = turn_pct * W_TURNOVER
    return _clip100(leader_component + rs_component + turn_component)


def build_role_map(df: pd.DataFrame) -> pd.DataFrame:
    """Compute cn_ga_stock_role_map_daily records from leader score data."""
    if df.empty:
        return pd.DataFrame()

    # Rank by rs_percentile within (trade_date, industry_id)
    df = df.copy()
    df["rs_rank_in_mainline"] = (
        df.groupby(["trade_date", "industry_id"])["rs_percentile"]
        .rank(ascending=False, method="min")
        .astype("Int64")
    )

    # Normalise leader_bucket to short uppercase ('CORE', 'NEAR', 'EDGE', 'NON')
    # Source stores short form; map to full name for role_map
    def _norm_bucket(b: str) -> str:
        b = str(b).strip().upper()
        # Handle both short and full forms
        if "CORE" in b and "NEAR" not in b and "NON" not in b and "EDGE" not in b:
            return "CORE"
        if "NEAR" in b:
            return "NEAR"
        if "EDGE" in b:
            return "EDGE"
        return "NON"

    df["_bucket_short"] = df["leader_bucket"].apply(_norm_bucket)
    df["stock_role"] = df["_bucket_short"].map(BUCKET_TO_ROLE).fillna("NON_CORE")
    df["leader_bucket_full"] = df["_bucket_short"].map(BUCKET_FULL_NAME).fillna("NON_LEADER")

    df["role_score"] = df.apply(
        lambda r: _compute_role_score(
            _safe_float(r["leader_score"]),
            _safe_float(r["rs_percentile"]),
            _safe_float(r["turnover_20d_percentile"]),
        ),
        axis=1,
    )

    df["role_reason"] = df.apply(
        lambda r: (
            f"leader={r['leader_score']:.1f}; "
            f"rs_pct={r['rs_percentile']:.4f}; "
            f"turn_pct={r['turnover_20d_percentile']:.4f}; "
            f"amount_rank={r['turnover_rank_in_industry']}; "
            f"rs_rank={r['rs_rank_in_mainline']}"
        ),
        axis=1,
    )

    out = pd.DataFrame({
        "trade_date": df["trade_date"],
        "symbol": df["symbol"],
        "stock_name": df["name"].fillna(""),
        "mainline_id": df["industry_id"],
        "mainline_name": df["industry_name"],
        "leader_score": df["leader_score"],
        "leader_bucket": df["leader_bucket_full"],
        "rs_percentile": df["rs_percentile"],
        "turnover_20d_percentile": df["turnover_20d_percentile"],
        "amount_rank_in_mainline": df["turnover_rank_in_industry"].fillna(0).astype(int),
        "rs_rank_in_mainline": df["rs_rank_in_mainline"],
        "stock_role": df["stock_role"],
        "role_score": df["role_score"],
        "role_reason": df["role_reason"],
    })
    return out


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------


def delete_range(engine: Engine, start: date, end: date) -> None:
    with engine.begin() as conn:
        conn.execute(
            text(
                "DELETE FROM cn_ga_stock_role_map_daily "
                "WHERE trade_date BETWEEN :start AND :end"
            ),
            {"start": start, "end": end},
        )


def write_rows(engine: Engine, df: pd.DataFrame, dry_run: bool) -> int:
    if df.empty:
        return 0
    if dry_run:
        logger.info("  [dry-run] would write %d rows", len(df))
        return 0

    cols = [
        "trade_date", "symbol", "stock_name", "mainline_id", "mainline_name",
        "leader_score", "leader_bucket", "rs_percentile", "turnover_20d_percentile",
        "amount_rank_in_mainline", "rs_rank_in_mainline", "stock_role",
        "role_score", "role_reason",
    ]
    write_df = df[cols].copy()
    records = write_df.astype(object).where(pd.notna(write_df), None).to_dict(orient="records")

    sql = """
        INSERT INTO cn_ga_stock_role_map_daily
            (trade_date, symbol, stock_name, mainline_id, mainline_name,
             leader_score, leader_bucket, rs_percentile, turnover_20d_percentile,
             amount_rank_in_mainline, rs_rank_in_mainline, stock_role,
             role_score, role_reason)
        VALUES
            (:trade_date, :symbol, :stock_name, :mainline_id, :mainline_name,
             :leader_score, :leader_bucket, :rs_percentile, :turnover_20d_percentile,
             :amount_rank_in_mainline, :rs_rank_in_mainline, :stock_role,
             :role_score, :role_reason)
    """
    BATCH = 5000
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

    logger.info("build_ga_stock_role_map_daily  %s ~ %s  dry_run=%s replace=%s",
                start, end, args.dry_run, args.replace)

    chunks = _date_chunks(start, end, args.chunk_months)
    total_written = 0

    for chunk_start, chunk_end in chunks:
        logger.info("  chunk %s ~ %s", chunk_start, chunk_end)

        df = load_leader_scores(engine, chunk_start, chunk_end)
        if df.empty:
            logger.info("    no source data, skip")
            continue

        out = build_role_map(df)
        if out.empty:
            continue

        if args.replace and not args.dry_run:
            delete_range(engine, chunk_start, chunk_end)

        n = write_rows(engine, out, args.dry_run)
        total_written += n
        logger.info("    wrote %d rows (dates=%d stocks=%d)",
                    n, out["trade_date"].nunique(), out["symbol"].nunique())

    logger.info("Done. Total rows written: %d", total_written)


if __name__ == "__main__":
    main()
