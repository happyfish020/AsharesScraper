"""
scripts/build_ga_market_pulse_daily.py
========================================
GrowthAlpha / AShareScraper P0I — Market Pulse Fact Rewire.

Builds cn_ga_market_pulse_daily from the trusted fact layer:
  1. cn_mainline_strength_fact_daily  — canonical mainline strength fact data
  2. cn_index_daily_price             — market index returns

This script intentionally DOES NOT read cn_ga_mainline_radar_daily.
Radar is legacy/display cache only and must not be a fact source.
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

CHUNK_MONTHS = 3
INDEX_HS300 = "sh000300"
INDEX_CHIEXT = "sh000688"
INDEX_SZ = "sz399001"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="P0I — Build cn_ga_market_pulse_daily from cn_mainline_strength_fact_daily")
    p.add_argument("--start", default="2024-01-01")
    p.add_argument("--end", default="")
    p.add_argument("--db-name", default="cn_market_red")
    p.add_argument("--db-host", default="127.0.0.1")
    p.add_argument("--db-port", type=int, default=3306)
    p.add_argument("--db-user", default="cn_opr_red")
    p.add_argument("--db-password", default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--replace", action="store_true")
    p.add_argument("--strict", action="store_true")
    p.add_argument("--min-mainlines", type=int, default=35)
    p.add_argument("--chunk-months", type=int, default=CHUNK_MONTHS)
    p.add_argument("--verbose", action="store_true")
    return p


def build_engine(host: str, port: int, user: str, password: str, db: str) -> Engine:
    url = URL.create("mysql+pymysql", username=user, password=password, host=host, port=port, database=db, query={"charset": "utf8mb4"})
    return create_engine(url, pool_pre_ping=True, future=True)


def fetch_df(engine: Engine, sql: str, params: dict[str, Any] | None = None) -> pd.DataFrame:
    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn, params=params or {})


def _safe_float(val: Any, default: float = 0.0) -> float:
    if val is None:
        return default
    try:
        v = float(val)
        return default if (np.isnan(v) or np.isinf(v)) else v
    except Exception:
        return default


def _clip01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


def _clip100(v: float) -> float:
    return max(0.0, min(100.0, float(v)))


def _date_chunks(start: date, end: date, months: int) -> list[tuple[date, date]]:
    chunks: list[tuple[date, date]] = []
    cur = start
    while cur <= end:
        next_month = date(cur.year + (cur.month + months - 1) // 12, (cur.month + months - 1) % 12 + 1, 1)
        chunk_end = min(next_month - timedelta(days=1), end)
        chunks.append((cur, chunk_end))
        cur = next_month
    return chunks


def ensure_schema(engine: Engine) -> None:
    ddl = """
    CREATE TABLE IF NOT EXISTS cn_ga_market_pulse_daily (
        trade_date DATE NOT NULL,
        market_score DOUBLE NULL,
        market_state VARCHAR(32) NULL,
        target_exposure DOUBLE NULL,
        breadth_up_ratio DOUBLE NULL,
        breadth_down_ratio DOUBLE NULL,
        amount_score DOUBLE NULL,
        index_rs_score DOUBLE NULL,
        hs300_pct_chg DOUBLE NULL,
        cyb_pct_chg DOUBLE NULL,
        sz_pct_chg DOUBLE NULL,
        risk_flag VARCHAR(32) NULL,
        reason TEXT NULL,
        bullish_industry_ratio DOUBLE NULL,
        neutral_industry_ratio DOUBLE NULL,
        bearish_industry_ratio DOUBLE NULL,
        rotation_speed DOUBLE NULL,
        mainline_stability DOUBLE NULL,
        trend_alignment_avg DOUBLE NULL,
        industry_expansion_breadth DOUBLE NULL,
        top_mainline_count INT NULL,
        market_phase VARCHAR(64) NULL,
        source_layer VARCHAR(64) NOT NULL DEFAULT 'FACT_META',
        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        PRIMARY KEY (trade_date),
        KEY idx_source_layer (source_layer)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """
    with engine.begin() as conn:
        conn.execute(text(ddl))
        # Older table may not have source_layer. Add it only if absent.
        cnt = conn.execute(text("""
            SELECT COUNT(*) FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME='cn_ga_market_pulse_daily' AND COLUMN_NAME='source_layer'
        """)).scalar()
        if not cnt:
            conn.execute(text("ALTER TABLE cn_ga_market_pulse_daily ADD COLUMN source_layer VARCHAR(64) NOT NULL DEFAULT 'FACT_META'"))


def load_fact(engine: Engine, start: date, end: date) -> pd.DataFrame:
    sql = """
        SELECT trade_date, mainline_id, mainline_name, source_layer,
               active_member_count, leader_count, core_count,
               up_ratio, rs_20d, rs_60d, rs_120d,
               amount_total, amount_rank_pct, amount_delta_5d,
               strong_stock_count, new_high_20d_count, new_high_52w_count,
               breakout_count, breakout_ratio,
               leader_strength_score, breadth_score, capital_score, trend_score,
               mainline_strength_score, rank_no, data_quality_flag
        FROM cn_mainline_strength_fact_daily
        WHERE trade_date BETWEEN :start AND :end
          AND data_quality_flag <> 'UNMAPPED_SOURCE'
    """
    df = fetch_df(engine, sql, {"start": start, "end": end})
    if not df.empty:
        df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
        numeric_cols = [c for c in df.columns if c not in ("trade_date", "mainline_id", "mainline_name", "source_layer", "data_quality_flag")]
        for c in numeric_cols:
            df[c] = df[c].apply(lambda x: _safe_float(x, 0.0))
    return df


def load_index_prices(engine: Engine, start: date, end: date) -> pd.DataFrame:
    sql = """
        SELECT TRADE_DATE as trade_date, INDEX_CODE as index_code, CHG_PCT as chg_pct, CLOSE as close_price
        FROM cn_index_daily_price
        WHERE TRADE_DATE BETWEEN :start AND :end
          AND INDEX_CODE IN (:hs300, :chiext, :sz)
    """
    df = fetch_df(engine, sql, {"start": start, "end": end, "hs300": INDEX_HS300, "chiext": INDEX_CHIEXT, "sz": INDEX_SZ})
    if not df.empty:
        df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
        df["chg_pct"] = df["chg_pct"].apply(lambda x: _safe_float(x, 0.0))
    return df


def _market_state(bullish: float, bearish: float) -> str:
    if bullish >= 0.45:
        return "TREND_STRONG"
    if bearish >= 0.45:
        return "RISK_OFF"
    if bullish >= 0.28:
        return "TREND_WEAK"
    return "RANGE"


def _market_phase(bullish: float, bearish: float, hs300_chg: float, market_score: float, prev_score: float | None) -> str:
    declining_from_high = prev_score is not None and prev_score > 70 and market_score < prev_score - 5
    if bearish >= 0.65 and hs300_chg < -3.0:
        return "CRISIS"
    if bearish >= 0.45:
        return "RISK_OFF"
    if declining_from_high and market_score > 55:
        return "TOP_DECAY"
    if bullish >= 0.45:
        return "TREND_EXPANSION"
    if bullish >= 0.28:
        return "DIFFUSION"
    if bearish >= 0.28:
        return "BOTTOM_REPAIR"
    return "DIVERGENCE"


def _target_exposure(state: str) -> float:
    return {"TREND_STRONG": 0.90, "TREND_WEAK": 0.65, "RANGE": 0.45, "RISK_OFF": 0.20}.get(state, 0.45)


def _risk_flag(state: str, bearish: float, hs300_chg: float) -> str:
    if state == "RISK_OFF" or bearish > 0.55:
        return "HIGH"
    if bearish > 0.35 or hs300_chg < -2.0:
        return "MEDIUM"
    if bearish > 0.20:
        return "LOW"
    return "NONE"


def build_pulse(fact_df: pd.DataFrame, index_df: pd.DataFrame, output_start: date, output_end: date) -> pd.DataFrame:
    if fact_df.empty:
        return pd.DataFrame()
    index_lookup: dict[tuple[date, str], float] = {}
    if not index_df.empty:
        for _, row in index_df.iterrows():
            index_lookup[(row["trade_date"], row["index_code"])] = _safe_float(row["chg_pct"], 0.0)

    rows: list[dict[str, Any]] = []
    prev_score: float | None = None
    for td in sorted(d for d in fact_df["trade_date"].unique() if output_start <= d <= output_end):
        day = fact_df[fact_df["trade_date"] == td]
        n_total = len(day)
        if n_total == 0:
            continue
        score = day["mainline_strength_score"].apply(lambda x: _safe_float(x, 50.0))
        trend = day["trend_score"].apply(lambda x: _safe_float(x, 0.5))
        breadth = day["breadth_score"].apply(lambda x: _safe_float(x, 0.5))

        bullish = _clip01(((score >= 60) | ((score >= 55) & (trend >= 0.60))).sum() / n_total)
        bearish = _clip01(((score <= 35) | (trend <= 0.30)).sum() / n_total)
        neutral = _clip01(1.0 - bullish - bearish)
        breadth_up = _clip01(day["up_ratio"].apply(lambda x: _safe_float(x, 0.5)).mean())
        breadth_down = _clip01(1.0 - breadth_up)
        trend_avg = _clip01(trend.mean())
        stability = _clip01(1.0 - min(1.0, float(score.std() if n_total > 1 else 0.0) / 30.0))
        expansion = _clip01((day["rs_20d"].apply(lambda x: _safe_float(x, 0.0)) >= 0.55).sum() / n_total)
        amount_score = _clip100(day["amount_rank_pct"].apply(lambda x: _safe_float(x, 0.5)).mean() * 100)
        market_score = _clip100(0.35 * breadth_up * 100 + 0.30 * bullish * 100 + 0.20 * trend_avg * 100 + 0.15 * (1 - bearish) * 100)
        hs300 = _safe_float(index_lookup.get((td, INDEX_HS300)), 0.0)
        cyb = _safe_float(index_lookup.get((td, INDEX_CHIEXT)), 0.0)
        sz = _safe_float(index_lookup.get((td, INDEX_SZ)), 0.0)
        index_rs_score = _clip100(50 + hs300 * 10)
        rotation_speed = _clip01(float(score.std() if n_total > 1 else 0.0) / 35.0)
        top_count = int((score >= 70).sum())
        state = _market_state(bullish, bearish)
        phase = _market_phase(bullish, bearish, hs300, market_score, prev_score)
        reason = f"fact_layer=cn_mainline_strength_fact_daily; breadth={breadth_up:.4f}; bullish={bullish:.4f}; bearish={bearish:.4f}; n={n_total}"
        rows.append({
            "trade_date": td,
            "market_score": round(market_score, 4),
            "market_state": state,
            "target_exposure": round(_target_exposure(state), 4),
            "breadth_up_ratio": round(breadth_up, 4),
            "breadth_down_ratio": round(breadth_down, 4),
            "amount_score": round(amount_score, 4),
            "index_rs_score": round(index_rs_score, 4),
            "hs300_pct_chg": round(hs300, 4),
            "cyb_pct_chg": round(cyb, 4),
            "sz_pct_chg": round(sz, 4),
            "risk_flag": _risk_flag(state, bearish, hs300),
            "reason": reason,
            "bullish_industry_ratio": round(bullish, 4),
            "neutral_industry_ratio": round(neutral, 4),
            "bearish_industry_ratio": round(bearish, 4),
            "rotation_speed": round(rotation_speed, 4),
            "mainline_stability": round(stability, 4),
            "trend_alignment_avg": round(trend_avg, 4),
            "industry_expansion_breadth": round(expansion, 4),
            "top_mainline_count": top_count,
            "market_phase": phase,
            "source_layer": "FACT_META",
        })
        prev_score = market_score
    return pd.DataFrame(rows)


def write_rows(engine: Engine, df: pd.DataFrame, dry_run: bool, replace: bool, start: date, end: date) -> int:
    if df.empty:
        return 0
    if dry_run:
        logger.info("[dry-run] would write %d rows", len(df))
        return 0
    cols = list(df.columns)
    placeholders = ", ".join(f":{c}" for c in cols)
    columns = ", ".join(f"`{c}`" for c in cols)
    updates = ", ".join(f"`{c}`=VALUES(`{c}`)" for c in cols if c != "trade_date")
    sql = text(f"INSERT INTO cn_ga_market_pulse_daily ({columns}) VALUES ({placeholders}) ON DUPLICATE KEY UPDATE {updates}")
    records = df.replace({np.nan: None}).to_dict("records")
    with engine.begin() as conn:
        if replace:
            conn.execute(text("DELETE FROM cn_ga_market_pulse_daily WHERE trade_date BETWEEN :s AND :e"), {"s": start, "e": end})
        conn.execute(sql, records)
    return len(records)


def main() -> None:
    args = build_parser().parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    pw = args.db_password or os.environ.get("ASHARE_MYSQL_PASSWORD", "")
    engine = build_engine(args.db_host, args.db_port, args.db_user, pw, args.db_name)
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end) if args.end else date.today()
    ensure_schema(engine)
    total_written = 0
    all_out = []
    for cs, ce in _date_chunks(start, end, args.chunk_months):
        fact = load_fact(engine, cs, ce)
        if fact.empty:
            logger.warning("no fact data for %s ~ %s", cs, ce)
            continue
        counts = fact.groupby("trade_date")["mainline_id"].nunique()
        if args.strict and not counts.empty and counts.min() < args.min_mainlines:
            bad = counts[counts < args.min_mainlines].head(10).to_dict()
            raise SystemExit(f"[MARKET PULSE FACT FAILED] fact mainline count below {args.min_mainlines}: {bad}")
        idx = load_index_prices(engine, cs, ce)
        out = build_pulse(fact, idx, cs, ce)
        all_out.append(out)
        total_written += write_rows(engine, out, args.dry_run, args.replace, cs, ce)
        logger.info("chunk %s~%s fact_rows=%d output_rows=%d", cs, ce, len(fact), len(out))
    final = pd.concat(all_out, ignore_index=True) if all_out else pd.DataFrame()
    print("[MARKET PULSE FACT]", {"output_rows": len(final), "written": total_written, "dry_run": bool(args.dry_run), "replace": bool(args.replace)})
    if not final.empty:
        print("[MARKET PULSE FACT LATEST]")
        print(final.sort_values("trade_date").tail(5)[["trade_date", "market_score", "market_state", "market_phase", "bullish_industry_ratio", "bearish_industry_ratio", "source_layer"]].to_string(index=False))


if __name__ == "__main__":
    main()
