"""
scripts/build_ga_mainline_radar_daily.py
=========================================
GrowthAlpha V8 — GA Layer — Part B.

Builds cn_ga_mainline_radar_daily from:
  1. cn_ga_stock_role_map_daily  — per-stock role and leader data
  2. cn_stock_daily_price        — price/amount metrics (UPPERCASE cols)
  3. cn_stock_mainline_strength_daily — pre-computed mainline scores

Key output columns used by build_unified_alpha_score_daily.py (Factors 5 & 8):
  leader_density      = leader_count / member_count
  new_high_ratio      = fraction of stocks with breakout_ready in role_map proxy
  breakout_ratio      = from cn_stock_mainline_strength_daily
  trend_alignment_score = fraction of stocks up today

Run order: after build_ga_stock_role_map_daily.py and
           after build_cn_stock_mainline_strength_daily.py

Usage:
  python scripts/build_ga_mainline_radar_daily.py \\
      --start 2010-01-01 --end 2026-03-31 \\
      --db-name cn_market_red --replace

  python scripts/build_ga_mainline_radar_daily.py \\
      --start 2010-01-01 --end 2026-03-31 \\
      --db-name cn_market_red --dry-run --verbose
"""

from __future__ import annotations

import argparse
import logging
import os
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

CHUNK_MONTHS = 1
LOOKBACK_DAYS = 30   # calendar days to load before chunk start for rolling metrics

# Mainline state thresholds (based on mainline_score 0–100)
STATE_CONFIRMED_MIN = 65.0
STATE_FORMING_MIN = 50.0
STATE_EARLY_MIN = 30.0
STATE_ROTATING_MIN = 15.0
# Below ROTATING_MIN → FADE

# LEADER-level roles used for leader_count
LEADER_ROLES = {"LEADER", "CORE"}

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
        description="GrowthAlpha V8 — Build cn_ga_mainline_radar_daily"
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


def _mainline_state(score: float) -> str:
    if score >= STATE_CONFIRMED_MIN:
        return "CONFIRMED"
    if score >= STATE_FORMING_MIN:
        return "FORMING"
    if score >= STATE_EARLY_MIN:
        return "EARLY"
    if score >= STATE_ROTATING_MIN:
        return "ROTATING"
    return "FADE"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_role_map(engine: Engine, start: date, end: date) -> pd.DataFrame:
    sql = """
        SELECT
            trade_date,
            symbol,
            mainline_id,
            mainline_name,
            stock_role,
            leader_score,
            rs_percentile,
            turnover_20d_percentile
        FROM cn_ga_stock_role_map_daily
        WHERE trade_date BETWEEN :start AND :end
    """
    return fetch_df(engine, sql, {"start": start, "end": end})


def load_price(engine: Engine, start: date, end: date) -> pd.DataFrame:
    sql = """
        SELECT
            SYMBOL,
            TRADE_DATE,
            CLOSE,
            PRE_CLOSE,
            CHG_PCT,
            AMOUNT
        FROM cn_stock_daily_price
        WHERE TRADE_DATE BETWEEN :start AND :end
    """
    df = fetch_df(engine, sql, {"start": start, "end": end})
    if not df.empty:
        df.rename(columns={
            "SYMBOL": "symbol", "TRADE_DATE": "trade_date",
            "CLOSE": "close", "PRE_CLOSE": "pre_close",
            "CHG_PCT": "chg_pct", "AMOUNT": "amount",
        }, inplace=True)
        for col in ["close", "pre_close", "chg_pct", "amount"]:
            df[col] = df[col].apply(lambda x: _safe_float(x, 0.0))
    return df


def load_mainline_strength(engine: Engine, start: date, end: date) -> pd.DataFrame:
    sql = """
        SELECT
            trade_date,
            industry_id,
            industry_name,
            mainline_strength_score,
            leader_density,
            breakout_ratio,
            trend_alignment,
            rank_in_market,
            is_active_mainline
        FROM cn_stock_mainline_strength_daily
        WHERE trade_date BETWEEN :start AND :end
    """
    df = fetch_df(engine, sql, {"start": start, "end": end})
    if not df.empty:
        for col in ["mainline_strength_score", "leader_density", "breakout_ratio", "trend_alignment"]:
            df[col] = df[col].apply(lambda x: _safe_float(x, 0.5))
    return df


def load_market_amounts(engine: Engine, start: date, end: date) -> pd.DataFrame:
    """Market-wide daily amount totals for relative strength computation."""
    sql = """
        SELECT
            TRADE_DATE as trade_date,
            SUM(AMOUNT) as market_amount_sum,
            AVG(CHG_PCT) as market_avg_ret
        FROM cn_stock_daily_price
        WHERE TRADE_DATE BETWEEN :start AND :end
        GROUP BY TRADE_DATE
    """
    df = fetch_df(engine, sql, {"start": start, "end": end})
    if not df.empty:
        df["market_amount_sum"] = df["market_amount_sum"].apply(lambda x: _safe_float(x, 0.0))
        df["market_avg_ret"] = df["market_avg_ret"].apply(lambda x: _safe_float(x, 0.0))
    return df


# ---------------------------------------------------------------------------
# Radar computation
# ---------------------------------------------------------------------------


def build_radar(
    role_df: pd.DataFrame,
    price_df: pd.DataFrame,
    strength_df: pd.DataFrame,
    market_df: pd.DataFrame,
    output_start: date,
    output_end: date,
) -> pd.DataFrame:
    """Compute radar records for dates in [output_start, output_end]."""
    if role_df.empty or price_df.empty:
        return pd.DataFrame()

    # Ensure date types are consistent
    for df in [role_df, price_df, strength_df if not strength_df.empty else pd.DataFrame()]:
        if not df.empty and "trade_date" in df.columns:
            df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date

    if not market_df.empty:
        market_df["trade_date"] = pd.to_datetime(market_df["trade_date"]).dt.date

    # Build strength lookup: (trade_date, industry_id) → row
    strength_lookup: dict[tuple, dict] = {}
    if not strength_df.empty:
        for _, row in strength_df.iterrows():
            strength_lookup[(row["trade_date"], row["industry_id"])] = row.to_dict()

    # Merge price into role_df for per-stock price metrics
    price_lookup: dict[tuple, dict] = {}
    if not price_df.empty:
        for _, row in price_df.iterrows():
            price_lookup[(row["trade_date"], row["symbol"])] = row.to_dict()

    market_lookup: dict[date, dict] = {}
    if not market_df.empty:
        for _, row in market_df.iterrows():
            market_lookup[row["trade_date"]] = row.to_dict()

    # Compute rolling 20d returns per stock for rs_20d
    # Build close price time-series per stock
    stock_close: dict[str, list[tuple[date, float]]] = {}
    if not price_df.empty:
        for _, row in price_df.iterrows():
            sym = row["symbol"]
            if sym not in stock_close:
                stock_close[sym] = []
            stock_close[sym].append((row["trade_date"], row["close"]))
        for sym in stock_close:
            stock_close[sym].sort()

    def _rs_20d(sym: str, td: date) -> float:
        series = stock_close.get(sym, [])
        if len(series) < 2:
            return 0.0
        dates = [s[0] for s in series]
        closes = [s[1] for s in series]
        try:
            idx = dates.index(td)
        except ValueError:
            return 0.0
        lookback_idx = max(0, idx - 20)
        base_close = closes[lookback_idx]
        if base_close <= 0:
            return 0.0
        return (closes[idx] - base_close) / base_close * 100  # as percentage

    rows_out = []
    trade_dates = sorted(set(role_df["trade_date"].unique()))
    output_dates = {d for d in trade_dates if output_start <= d <= output_end}

    # Group role_df by (trade_date, mainline_id)
    role_df_grp = role_df.groupby(["trade_date", "mainline_id"])

    for (td, mainline_id), grp in role_df_grp:
        if td not in output_dates:
            continue

        mainline_name = grp["mainline_name"].iloc[0] if "mainline_name" in grp.columns else mainline_id
        symbols = grp["symbol"].tolist()
        n = len(symbols)

        # Price metrics for this mainline on this date
        chg_list = []
        amount_list = []
        for sym in symbols:
            p = price_lookup.get((td, sym))
            if p:
                chg_list.append(_safe_float(p.get("chg_pct"), 0.0))
                amount_list.append(_safe_float(p.get("amount"), 0.0))

        up_count = sum(1 for c in chg_list if c > 0)
        down_count = sum(1 for c in chg_list if c < 0)
        up_ratio = _clip01(up_count / n) if n > 0 else 0.5
        avg_ret = float(np.mean(chg_list)) if chg_list else 0.0
        median_ret = float(np.median(chg_list)) if chg_list else 0.0
        amount_sum = sum(amount_list)
        amount_up_sum = sum(a for c, a in zip(chg_list, amount_list) if c > 0)

        # Amount change vs prior period (5-day proxy via market context)
        mkt = market_lookup.get(td, {})
        mkt_amount = _safe_float(mkt.get("market_amount_sum"), 1.0)
        amount_chg_5d = _safe_float(amount_sum / mkt_amount if mkt_amount > 0 else 0.0)

        # rs_20d: avg 20d return of mainline stocks vs market
        stock_rs = [_rs_20d(sym, td) for sym in symbols]
        mkt_rs_list = []
        for sym in symbols:
            p = price_lookup.get((td, sym))
        rs_20d = float(np.mean(stock_rs)) if stock_rs else 0.0

        # rs rank (will be computed after all mainlines are processed)
        # placeholder — filled in post-processing
        rs_rank_placeholder = 0

        # Leader metrics from role_map
        leader_roles_in_group = grp["stock_role"].isin(LEADER_ROLES)
        leader_count = int(leader_roles_in_group.sum())
        leader_density = _clip01(leader_count / n) if n > 0 else 0.0
        leader_score_avg = float(grp["leader_score"].apply(
            lambda x: _safe_float(x, 0.0) / 3.0
        ).mean()) if n > 0 else 0.0

        # Mainline-level metrics from strength table if available
        sk = (td, mainline_id)
        st = strength_lookup.get(sk, {})

        breakout_ratio = _clip01(_safe_float(st.get("breakout_ratio"), leader_density))
        trend_alignment_score = _clip01(_safe_float(st.get("trend_alignment"), up_ratio))
        mainline_strength_raw = _safe_float(st.get("mainline_strength_score"), -1.0)
        rank_in_market = int(st.get("rank_in_market") or 0)
        is_active = int(st.get("is_active_mainline") or 0)

        # mainline_score: derive from strength score (0-1 → 0-100)
        # When strength table not available, derive from role/price metrics
        if mainline_strength_raw >= 0:
            mainline_score = _clip100(mainline_strength_raw * 100.0)
        else:
            # Fallback: composite from available metrics
            mainline_score = _clip100(
                up_ratio * 40.0
                + leader_density * 30.0
                + (rs_20d + 5.0) / 10.0 * 20.0  # normalize rs_20d ≈ [-5,5]
                + breakout_ratio * 10.0
            )
            rank_in_market = 0

        mainline_state = _mainline_state(mainline_score)

        # new_high_ratio proxy: fraction of stocks with rs_percentile > 0.85
        new_high_ratio = _clip01(
            grp["rs_percentile"]
            .apply(lambda x: _safe_float(x, 0.0))
            .apply(lambda x: 1.0 if x >= 0.85 else 0.0)
            .mean()
        )

        strong_stock_count = int(
            grp["rs_percentile"]
            .apply(lambda x: _safe_float(x, 0.0) >= 0.70)
            .sum()
        )

        # mainline_confidence: how reliable this reading is (0-1)
        confidence = _clip01(
            0.4 * (1.0 if mainline_strength_raw >= 0 else 0.0)  # has strength data
            + 0.3 * min(1.0, n / 10.0)                           # enough members
            + 0.3 * _clip01(leader_density * 2)                  # has leaders
        )

        reason = (
            f"score={mainline_score:.2f}; up_ratio={up_ratio:.4f}; "
            f"rs20={rs_20d:.4f}; amount_sum={amount_sum:.0f}; "
            f"leader_density={leader_density:.4f}"
        )

        rows_out.append({
            "trade_date": td,
            "mainline_id": mainline_id,
            "mainline_name": mainline_name,
            "member_count": n,
            "up_count": up_count,
            "down_count": down_count,
            "up_ratio": round(up_ratio, 4),
            "avg_ret": round(avg_ret, 4),
            "median_ret": round(median_ret, 4),
            "amount_sum": round(amount_sum, 4),
            "amount_up_sum": round(amount_up_sum, 4),
            "amount_chg_5d": round(amount_chg_5d, 4),
            "rs_5d": round(rs_20d * 0.25, 4),   # approximation
            "rs_20d": round(rs_20d, 4),
            "rs_rank": rank_in_market,
            "leader_count": leader_count,
            "leader_score_avg": round(leader_score_avg, 4),
            "mainline_score": round(mainline_score, 4),
            "mainline_state": mainline_state,
            "rank_no": rank_in_market,
            "reason": reason,
            # Extended columns
            "leader_density": round(leader_density, 4),
            "new_high_ratio": round(new_high_ratio, 4),
            "breakout_ratio": round(breakout_ratio, 4),
            "trend_alignment_score": round(trend_alignment_score, 4),
            "strong_stock_count": strong_stock_count,
            "mainline_confidence": round(confidence, 4),
            "rotation_rank": rank_in_market,
            "mainline_phase": mainline_state,
        })

    if not rows_out:
        return pd.DataFrame()

    out = pd.DataFrame(rows_out)

    # Compute rs_rank across mainlines per trade_date
    if not out.empty:
        out["rs_rank"] = (
            out.groupby("trade_date")["rs_20d"]
            .rank(ascending=False, method="min")
            .astype(int)
        )
        out["rank_no"] = out["rs_rank"]
        out["rotation_rank"] = out["rs_rank"]

    return out


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------


def delete_range(engine: Engine, start: date, end: date) -> None:
    with engine.begin() as conn:
        conn.execute(
            text("DELETE FROM cn_ga_mainline_radar_daily WHERE trade_date BETWEEN :s AND :e"),
            {"s": start, "e": end},
        )


def write_rows(engine: Engine, df: pd.DataFrame, dry_run: bool) -> int:
    if df.empty:
        return 0
    if dry_run:
        logger.info("  [dry-run] would write %d rows", len(df))
        return 0

    cols = [
        "trade_date", "mainline_id", "mainline_name", "member_count",
        "up_count", "down_count", "up_ratio", "avg_ret", "median_ret",
        "amount_sum", "amount_up_sum", "amount_chg_5d", "rs_5d", "rs_20d",
        "rs_rank", "leader_count", "leader_score_avg", "mainline_score",
        "mainline_state", "rank_no", "reason",
        "leader_density", "new_high_ratio", "breakout_ratio",
        "trend_alignment_score", "strong_stock_count", "mainline_confidence",
        "rotation_rank", "mainline_phase",
    ]
    # Only include cols that exist in df
    write_cols = [c for c in cols if c in df.columns]
    records = df[write_cols].to_dict("records")

    col_list = ", ".join(write_cols)
    val_list = ", ".join(f":{c}" for c in write_cols)
    sql = f"INSERT INTO cn_ga_mainline_radar_daily ({col_list}) VALUES ({val_list})"

    BATCH = 2000
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

    logger.info("build_ga_mainline_radar_daily  %s ~ %s  dry_run=%s replace=%s",
                start, end, args.dry_run, args.replace)

    chunks = _date_chunks(start, end, args.chunk_months)
    total_written = 0

    for chunk_start, chunk_end in chunks:
        logger.info("  chunk %s ~ %s", chunk_start, chunk_end)

        # Load with lookback for rolling metrics
        load_start = chunk_start - timedelta(days=LOOKBACK_DAYS)

        role_df = load_role_map(engine, load_start, chunk_end)
        if role_df.empty:
            logger.info("    no role_map data, skip")
            continue

        price_df = load_price(engine, load_start, chunk_end)
        strength_df = load_mainline_strength(engine, load_start, chunk_end)
        market_df = load_market_amounts(engine, load_start, chunk_end)

        logger.debug("    role_map=%d price=%d strength=%d",
                     len(role_df), len(price_df), len(strength_df))

        out = build_radar(role_df, price_df, strength_df, market_df,
                          chunk_start, chunk_end)
        if out.empty:
            logger.info("    no output, skip")
            continue

        if args.replace and not args.dry_run:
            delete_range(engine, chunk_start, chunk_end)

        n = write_rows(engine, out, args.dry_run)
        total_written += n
        logger.info("    wrote %d rows (dates=%d mainlines=%d)",
                    n, out["trade_date"].nunique(), out["mainline_id"].nunique())

    logger.info("Done. Total rows written: %d", total_written)


if __name__ == "__main__":
    main()
