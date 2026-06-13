"""
scripts/build_cn_mainline_strength_fact_daily.py
================================================
GrowthAlpha / AShareScraper — Mainline Strength Fact Layer.

Builds cn_mainline_strength_fact_daily as a clean fact table for GrowthAlpha.
This builder MUST NOT read cn_ga_mainline_radar_daily.  Radar can later be
rebuilt from this fact table, but not the reverse.

Primary sources:
  - cn_ga_stock_role_map_daily        stock -> mainline mapping and role scores
  - cn_stock_daily_price              price/amount daily facts
  - cn_stock_daily_basic              market-cap / turnover facts, optional join
  - cn_industry_capital_flow_daily    optional capital evidence

Usage:
  python scripts/build_cn_mainline_strength_fact_daily.py --start 2024-01-01 --end 2026-06-12 --replace
  python scripts/build_cn_mainline_strength_fact_daily.py --start 2024-01-01 --end 2026-06-12 --dry-run --verbose
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import date, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, URL

CHUNK_MONTHS = 1
LOOKBACK_CALENDAR_DAYS = 210  # enough for 120 trading-day rolling windows
LEADER_ROLES = {"LEADER", "CORE"}
CORE_ROLES = {"LEADER", "CORE", "MOMENTUM"}

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build clean cn_mainline_strength_fact_daily without radar dependency")
    p.add_argument("--start", default="2024-01-01")
    p.add_argument("--end", default="")
    p.add_argument("--db-name", default="cn_market_red")
    p.add_argument("--db-host", default="127.0.0.1")
    p.add_argument("--db-port", type=int, default=3306)
    p.add_argument("--db-user", default="cn_opr_red")
    p.add_argument("--db-password", default=None)
    p.add_argument("--replace", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--chunk-months", type=int, default=CHUNK_MONTHS)
    p.add_argument("--verbose", action="store_true")
    return p


def _parse_date(v: str) -> date:
    return datetime.strptime(v, "%Y-%m-%d" if "-" in v else "%Y%m%d").date()


def _date_chunks(start: date, end: date, months: int) -> list[tuple[date, date]]:
    chunks: list[tuple[date, date]] = []
    cur = start
    while cur <= end:
        next_month = date(cur.year + (cur.month + months - 1) // 12, (cur.month + months - 1) % 12 + 1, 1)
        chunk_end = min(next_month - timedelta(days=1), end)
        chunks.append((cur, chunk_end))
        cur = next_month
    return chunks


def build_engine(host: str, port: int, user: str, password: str, db: str) -> Engine:
    url = URL.create("mysql+pymysql", username=user, password=password, host=host, port=port, database=db, query={"charset": "utf8mb4"})
    return create_engine(url, pool_pre_ping=True, future=True)


def fetch_df(engine: Engine, sql: str, params: dict[str, Any] | None = None) -> pd.DataFrame:
    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn, params=params or {})


def table_exists(engine: Engine, db_name: str, table_name: str) -> bool:
    sql = """
        SELECT COUNT(*) FROM INFORMATION_SCHEMA.TABLES
        WHERE TABLE_SCHEMA=:db AND TABLE_NAME=:table
    """
    with engine.connect() as conn:
        return int(conn.execute(text(sql), {"db": db_name, "table": table_name}).scalar() or 0) > 0


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return default
        x = float(v)
        if np.isnan(x) or np.isinf(x):
            return default
        return x
    except Exception:
        return default


def _clip01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


def _clip100(v: float) -> float:
    return max(0.0, min(100.0, float(v)))


def ensure_schema(engine: Engine) -> None:
    ddl = """
    CREATE TABLE IF NOT EXISTS cn_mainline_strength_fact_daily (
        trade_date DATE NOT NULL,
        mainline_id VARCHAR(64) NOT NULL,
        mainline_name VARCHAR(128) NULL,
        source_layer VARCHAR(64) NOT NULL DEFAULT 'FACT_FROM_ROLE_PRICE',
        member_count INT NOT NULL DEFAULT 0,
        active_member_count INT NOT NULL DEFAULT 0,
        leader_count INT NOT NULL DEFAULT 0,
        core_count INT NOT NULL DEFAULT 0,
        ret_1d DOUBLE NULL,
        ret_5d DOUBLE NULL,
        ret_20d DOUBLE NULL,
        ret_60d DOUBLE NULL,
        ret_120d DOUBLE NULL,
        rs_20d DOUBLE NULL,
        rs_60d DOUBLE NULL,
        rs_120d DOUBLE NULL,
        amount_total DOUBLE NULL,
        amount_rank_pct DOUBLE NULL,
        amount_delta_5d DOUBLE NULL,
        turnover_avg DOUBLE NULL,
        up_ratio DOUBLE NULL,
        strong_stock_count INT NOT NULL DEFAULT 0,
        new_high_20d_count INT NOT NULL DEFAULT 0,
        new_high_52w_count INT NOT NULL DEFAULT 0,
        breakout_count INT NOT NULL DEFAULT 0,
        breakout_ratio DOUBLE NULL,
        leader_strength_score DOUBLE NULL,
        breadth_score DOUBLE NULL,
        capital_score DOUBLE NULL,
        trend_score DOUBLE NULL,
        mainline_strength_score DOUBLE NULL,
        rank_no INT NULL,
        data_quality_flag VARCHAR(64) NOT NULL DEFAULT 'UNKNOWN',
        coverage_start_date DATE NULL,
        is_backtest_eligible TINYINT NOT NULL DEFAULT 0,
        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        PRIMARY KEY (trade_date, mainline_id),
        KEY idx_cmsfd_date_rank (trade_date, rank_no),
        KEY idx_cmsfd_mainline_date (mainline_id, trade_date),
        KEY idx_cmsfd_quality (data_quality_flag, is_backtest_eligible)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """
    with engine.begin() as conn:
        conn.execute(text(ddl))


def load_role_map(engine: Engine, start: date, end: date) -> pd.DataFrame:
    sql = """
        SELECT trade_date, symbol, mainline_id, mainline_name, stock_role,
               role_score, leader_score, rs_percentile, turnover_20d_percentile
        FROM cn_ga_stock_role_map_daily
        WHERE trade_date BETWEEN :start AND :end
          AND mainline_id IS NOT NULL AND mainline_id <> ''
    """
    return fetch_df(engine, sql, {"start": start, "end": end})


def load_price(engine: Engine, start: date, end: date) -> pd.DataFrame:
    sql = """
        SELECT SYMBOL AS symbol, TRADE_DATE AS trade_date, CLOSE AS close,
               PRE_CLOSE AS pre_close, CHG_PCT AS chg_pct, AMOUNT AS amount,
               TURNOVER_RATE AS turnover_rate
        FROM cn_stock_daily_price
        WHERE TRADE_DATE BETWEEN :start AND :end
    """
    return fetch_df(engine, sql, {"start": start, "end": end})


def load_basic(engine: Engine, start: date, end: date, db_name: str) -> pd.DataFrame:
    if not table_exists(engine, db_name, "cn_stock_daily_basic"):
        return pd.DataFrame()
    sql = """
        SELECT symbol, trade_date, total_mv, circ_mv, turnover_rate_f
        FROM cn_stock_daily_basic
        WHERE trade_date BETWEEN :start AND :end
    """
    return fetch_df(engine, sql, {"start": start, "end": end})


def load_capital_flow(engine: Engine, start: date, end: date, db_name: str) -> pd.DataFrame:
    if not table_exists(engine, db_name, "cn_industry_capital_flow_daily"):
        return pd.DataFrame()
    sql = """
        SELECT trade_date, industry_id AS mainline_id, concentration_score, flow_strength_score
        FROM cn_industry_capital_flow_daily
        WHERE trade_date BETWEEN :start AND :end
    """
    return fetch_df(engine, sql, {"start": start, "end": end})


def _normalize_inputs(role: pd.DataFrame, price: pd.DataFrame, basic: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for df in (role, price, basic):
        if not df.empty and "trade_date" in df.columns:
            df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
    if not role.empty:
        role["symbol"] = role["symbol"].astype(str)
        role["stock_role"] = role["stock_role"].fillna("NON_CORE").astype(str)
        for c in ["role_score", "leader_score", "rs_percentile", "turnover_20d_percentile"]:
            if c in role.columns:
                role[c] = role[c].apply(_safe_float)
    if not price.empty:
        price["symbol"] = price["symbol"].astype(str)
        for c in ["close", "pre_close", "chg_pct", "amount", "turnover_rate"]:
            if c in price.columns:
                price[c] = price[c].apply(_safe_float)
        price["ret_1d_stock"] = price["chg_pct"] / 100.0
    if not basic.empty:
        basic["symbol"] = basic["symbol"].astype(str)
        for c in ["total_mv", "circ_mv", "turnover_rate_f"]:
            if c in basic.columns:
                basic[c] = basic[c].apply(_safe_float)
    return role, price, basic


def _add_stock_breakout_flags(price: pd.DataFrame) -> pd.DataFrame:
    if price.empty:
        return price
    price = price.sort_values(["symbol", "trade_date"]).copy()
    g = price.groupby("symbol", group_keys=False)
    price["high_20d_prev"] = g["close"].transform(lambda s: s.shift(1).rolling(20, min_periods=10).max())
    price["high_52w_prev"] = g["close"].transform(lambda s: s.shift(1).rolling(252, min_periods=120).max())
    price["new_high_20d"] = (price["close"] >= price["high_20d_prev"]).fillna(False)
    price["new_high_52w"] = (price["close"] >= price["high_52w_prev"]).fillna(False)
    price["breakout"] = ((price["ret_1d_stock"] >= 0.03) & price["new_high_20d"]).fillna(False)
    return price


def build_fact_frame(engine: Engine, chunk_start: date, chunk_end: date, db_name: str, verbose: bool = False) -> pd.DataFrame:
    load_start = chunk_start - timedelta(days=LOOKBACK_CALENDAR_DAYS)
    role = load_role_map(engine, load_start, chunk_end)
    price = load_price(engine, load_start, chunk_end)
    basic = load_basic(engine, load_start, chunk_end, db_name)
    capital = load_capital_flow(engine, chunk_start, chunk_end, db_name)
    role, price, basic = _normalize_inputs(role, price, basic)
    if role.empty or price.empty:
        return pd.DataFrame()
    price = _add_stock_breakout_flags(price)
    df = role.merge(price, on=["trade_date", "symbol"], how="inner")
    if not basic.empty:
        df = df.merge(basic, on=["trade_date", "symbol"], how="left")
    else:
        df["turnover_rate_f"] = np.nan
        df["total_mv"] = np.nan
    if df.empty:
        return pd.DataFrame()

    df["is_up"] = df["ret_1d_stock"] > 0
    df["is_active"] = df["close"] > 0
    df["is_leader"] = df["stock_role"].isin(LEADER_ROLES)
    df["is_core"] = df["stock_role"].isin(CORE_ROLES)
    df["is_strong"] = (df["leader_score"] >= 0.5) | (df["rs_percentile"] >= 0.80)

    grouped = df.groupby(["trade_date", "mainline_id", "mainline_name"], dropna=False)
    out = grouped.agg(
        member_count=("symbol", "nunique"),
        active_member_count=("is_active", "sum"),
        leader_count=("is_leader", "sum"),
        core_count=("is_core", "sum"),
        ret_1d=("ret_1d_stock", "mean"),
        amount_total=("amount", "sum"),
        turnover_avg=("turnover_rate_f", "mean"),
        up_ratio=("is_up", "mean"),
        strong_stock_count=("is_strong", "sum"),
        new_high_20d_count=("new_high_20d", "sum"),
        new_high_52w_count=("new_high_52w", "sum"),
        breakout_count=("breakout", "sum"),
        avg_role_score=("role_score", "mean"),
        avg_leader_score=("leader_score", "mean"),
        avg_rs_percentile=("rs_percentile", "mean"),
    ).reset_index()
    out = out.sort_values(["mainline_id", "trade_date"])
    g = out.groupby("mainline_id", group_keys=False)
    min_periods_by_window = {5: 3, 20: 10, 60: 30, 120: 60}
    for win in (5, 20, 60, 120):
        min_p = min_periods_by_window[win]
        out[f"ret_{win}d"] = g["ret_1d"].transform(
            lambda s, w=win, mp=min_p: (1.0 + s.fillna(0)).rolling(w, min_periods=mp).apply(np.prod, raw=True) - 1.0
        )
    # Cross-sectional RS percentile by date. 0..1, higher is stronger.
    for win in (20, 60, 120):
        ret_col = f"ret_{win}d"
        rs_col = f"rs_{win}d"
        out[rs_col] = out.groupby("trade_date")[ret_col].rank(pct=True, method="average")
    out["amount_rank_pct"] = out.groupby("trade_date")["amount_total"].rank(pct=True, method="average")
    out["amount_delta_5d"] = g["amount_total"].transform(lambda s: s / s.shift(5) - 1.0)
    out["leader_strength_score"] = (0.6 * out["avg_leader_score"].fillna(0) + 0.4 * out["avg_role_score"].fillna(0) / 100.0).clip(0, 1)
    out["breakout_ratio"] = (out["breakout_count"] / out["active_member_count"].replace(0, np.nan)).fillna(0).clip(0, 1)
    nh20_ratio = (out["new_high_20d_count"] / out["active_member_count"].replace(0, np.nan)).fillna(0).clip(0, 1)
    nh52_ratio = (out["new_high_52w_count"] / out["active_member_count"].replace(0, np.nan)).fillna(0).clip(0, 1)
    strong_ratio = (out["strong_stock_count"] / out["active_member_count"].replace(0, np.nan)).fillna(0).clip(0, 1)
    out["breadth_score"] = (0.35 * out["up_ratio"].fillna(0) + 0.25 * strong_ratio + 0.20 * nh20_ratio + 0.20 * nh52_ratio).clip(0, 1)
    out["capital_score"] = (0.65 * out["amount_rank_pct"].fillna(0) + 0.35 * ((out["amount_delta_5d"].fillna(0).clip(-1, 1) + 1) / 2)).clip(0, 1)
    out["trend_score"] = (0.50 * out["rs_20d"].fillna(0) + 0.30 * out["rs_60d"].fillna(0) + 0.20 * out["rs_120d"].fillna(0)).clip(0, 1)

    if not capital.empty:
        capital["trade_date"] = pd.to_datetime(capital["trade_date"]).dt.date
        for c in ["concentration_score", "flow_strength_score"]:
            if c in capital.columns:
                capital[c] = capital[c].apply(_safe_float)
        out = out.merge(capital, on=["trade_date", "mainline_id"], how="left")
        flow = out.get("flow_strength_score", pd.Series(0.0, index=out.index)).fillna(0).clip(0, 1)
        conc = out.get("concentration_score", pd.Series(0.0, index=out.index)).fillna(0).clip(0, 1)
        out["capital_score"] = (0.70 * out["capital_score"] + 0.20 * flow + 0.10 * conc).clip(0, 1)

    out["mainline_strength_score"] = (
        0.35 * out["trend_score"] +
        0.25 * out["breadth_score"] +
        0.20 * out["capital_score"] +
        0.20 * out["leader_strength_score"]
    ).clip(0, 1) * 100.0
    out["rank_no"] = out.groupby("trade_date")["mainline_strength_score"].rank(ascending=False, method="first").astype(int)
    out["source_layer"] = "FACT_FROM_ROLE_PRICE"
    out["coverage_start_date"] = load_start
    out["is_backtest_eligible"] = ((out["rs_60d"].notna()) & (out["rs_120d"].notna()) & (out["active_member_count"] >= 3)).astype(int)
    out["data_quality_flag"] = np.select(
        [out["active_member_count"] < 3, out["rs_120d"].isna(), out["rs_60d"].isna()],
        ["LOW_MEMBER_COVERAGE", "INSUFFICIENT_120D_HISTORY", "INSUFFICIENT_60D_HISTORY"],
        default="OK",
    )
    keep = [
        "trade_date", "mainline_id", "mainline_name", "source_layer", "member_count", "active_member_count",
        "leader_count", "core_count", "ret_1d", "ret_5d", "ret_20d", "ret_60d", "ret_120d",
        "rs_20d", "rs_60d", "rs_120d", "amount_total", "amount_rank_pct", "amount_delta_5d",
        "turnover_avg", "up_ratio", "strong_stock_count", "new_high_20d_count", "new_high_52w_count",
        "breakout_count", "breakout_ratio", "leader_strength_score", "breadth_score", "capital_score",
        "trend_score", "mainline_strength_score", "rank_no", "data_quality_flag", "coverage_start_date", "is_backtest_eligible",
    ]
    out = out[(out["trade_date"] >= chunk_start) & (out["trade_date"] <= chunk_end)][keep].copy()
    if verbose:
        logger.info("fact frame %s~%s rows=%s", chunk_start, chunk_end, len(out))
    return out


def write_fact(engine: Engine, df: pd.DataFrame, start: date, end: date, replace: bool) -> None:
    if df.empty:
        return
    records = df.replace({np.nan: None}).to_dict("records")
    cols = list(df.columns)
    col_sql = ", ".join(cols)
    val_sql = ", ".join(f":{c}" for c in cols)
    update_sql = ", ".join(f"{c}=VALUES({c})" for c in cols if c not in {"trade_date", "mainline_id"})
    sql = f"INSERT INTO cn_mainline_strength_fact_daily ({col_sql}) VALUES ({val_sql}) ON DUPLICATE KEY UPDATE {update_sql}"
    with engine.begin() as conn:
        if replace:
            conn.execute(text("DELETE FROM cn_mainline_strength_fact_daily WHERE trade_date BETWEEN :start AND :end"), {"start": start, "end": end})
        conn.execute(text(sql), records)


def run(args: argparse.Namespace) -> pd.DataFrame:
    start = _parse_date(args.start)
    end = _parse_date(args.end) if args.end else date.today()
    if start > end:
        raise SystemExit(f"start {start} > end {end}")
    password = args.db_password or os.getenv("ASHARE_MYSQL_PASSWORD", "sec_Bobo123")
    engine = build_engine(args.db_host, args.db_port, args.db_user, password, args.db_name)
    for table in ["cn_ga_stock_role_map_daily", "cn_stock_daily_price"]:
        if not table_exists(engine, args.db_name, table):
            raise SystemExit(f"Required source table missing: {table}")
    ensure_schema(engine)
    frames: list[pd.DataFrame] = []
    for s, e in _date_chunks(start, end, max(1, args.chunk_months)):
        df = build_fact_frame(engine, s, e, args.db_name, args.verbose)
        frames.append(df)
        if args.dry_run:
            logger.info("DRY-RUN skip write rows=%s chunk=%s~%s", len(df), s, e)
        else:
            write_fact(engine, df, s, e, args.replace)
            logger.info("WROTE cn_mainline_strength_fact_daily rows=%s chunk=%s~%s", len(df), s, e)
    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if result.empty:
        logger.warning("No fact rows produced for %s~%s", start, end)
    else:
        q = result["data_quality_flag"].value_counts(dropna=False).to_dict()
        logger.info("DONE rows=%s quality=%s backtest_eligible=%s", len(result), q, int(result["is_backtest_eligible"].sum()))
    return result


def main() -> None:
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
