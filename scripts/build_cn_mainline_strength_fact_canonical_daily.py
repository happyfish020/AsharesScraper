"""Build canonical cn_mainline_strength_fact_daily from source facts.

This is the Fact Layer after introducing the system canonical MAINLINE_ID.
The output table still uses cn_mainline_strength_fact_daily for compatibility,
but its mainline_id is the canonical ID (e.g. ML_CN_SWL1_801080), not an
external source code (801080.SI / 850xxx.SI / BKxxxx).

Hard rule: this script MUST NOT read cn_ga_mainline_radar_daily.

Current V1 canonical universe:
  source: cn_local_industry_proxy_daily SW-L1 rows
  map:    cn_source_mainline_code_map
  output: cn_mainline_strength_fact_daily with source_layer='FACT_CANONICAL_SW_L1'

Usage:
  python scripts/build_cn_canonical_mainline_registry.py --start 2024-01-01 --end 2026-06-12 --replace-sw-l1 --strict
  python scripts/build_cn_mainline_strength_fact_canonical_daily.py --start 2024-01-01 --end 2026-06-12 --replace --strict
"""
from __future__ import annotations

import argparse
import os
from datetime import date, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, URL

LOOKBACK_CALENDAR_DAYS = 210


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build canonical mainline strength fact table without radar dependency")
    p.add_argument("--start", default="2024-01-01")
    p.add_argument("--end", default="")
    p.add_argument("--db-name", default="cn_market_red")
    p.add_argument("--db-host", default="127.0.0.1")
    p.add_argument("--db-port", type=int, default=3306)
    p.add_argument("--db-user", default="cn_opr_red")
    p.add_argument("--db-password", default=None)
    p.add_argument("--replace", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--strict", action="store_true")
    p.add_argument("--min-daily-mainlines", type=int, default=25)
    p.add_argument("--max-daily-mainlines", type=int, default=40)
    return p


def _parse_date(v: str) -> date:
    return datetime.strptime(v, "%Y-%m-%d" if "-" in v else "%Y%m%d").date()


def build_engine(host: str, port: int, user: str, password: str, db: str) -> Engine:
    url = URL.create("mysql+pymysql", username=user, password=password, host=host, port=port, database=db, query={"charset": "utf8mb4"})
    return create_engine(url, pool_pre_ping=True, future=True)


def fetch_df(engine: Engine, sql: str, params: dict[str, Any] | None = None) -> pd.DataFrame:
    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn, params=params or {})


def ensure_schema(engine: Engine) -> None:
    ddl = """
    CREATE TABLE IF NOT EXISTS cn_mainline_strength_fact_daily (
        trade_date DATE NOT NULL,
        mainline_id VARCHAR(64) NOT NULL,
        mainline_name VARCHAR(128) NULL,
        source_layer VARCHAR(64) NOT NULL DEFAULT 'FACT_CANONICAL_SW_L1',
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


def load_canonical_l1_proxy(engine: Engine, start: date, end: date) -> pd.DataFrame:
    sql = """
      SELECT p.trade_date,
             m.canonical_mainline_id AS mainline_id,
             m.canonical_mainline_name AS mainline_name,
             p.industry_id AS source_code,
             p.industry_name AS source_name,
             p.member_count,
             p.ret_eqw,
             p.amount_total,
             p.turnover_avg,
             p.market_cap_total,
             p.leader_return,
             p.top5_concentration
      FROM cn_local_industry_proxy_daily p
      JOIN cn_source_mainline_code_map m
        ON m.source_system='SW'
       AND m.source_code=p.industry_id
       AND m.is_active=1
       AND m.canonical_level='L1'
       AND p.trade_date BETWEEN m.effective_start_date AND COALESCE(m.effective_end_date, '2999-12-31')
      WHERE p.trade_date BETWEEN :start AND :end
        AND (p.industry_level = 'L1' OR p.industry_id REGEXP '^801[0-9]{3}\\.SI$')
    """
    return fetch_df(engine, sql, {"start": start, "end": end})


def build_frame(engine: Engine, start: date, end: date, strict: bool, min_daily: int, max_daily: int) -> pd.DataFrame:
    load_start = start - timedelta(days=LOOKBACK_CALENDAR_DAYS)
    df = load_canonical_l1_proxy(engine, load_start, end)
    if df.empty:
        raise SystemExit("NO_CANONICAL_L1_PROXY_ROWS: run build_cn_canonical_mainline_registry.py after cn_local_industry_proxy_daily is populated")
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
    for c in ["member_count", "ret_eqw", "amount_total", "turnover_avg", "market_cap_total", "leader_return", "top5_concentration"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # Source map should be one-to-one for active SW-L1. If not, fail loudly.
    dup = df.groupby(["trade_date", "mainline_id"]).size().reset_index(name="n")
    bad = dup[dup["n"] > 1]
    if not bad.empty:
        raise SystemExit(f"DUPLICATE_CANONICAL_ROWS sample={bad.head(5).to_dict('records')}")
    df = df.sort_values(["mainline_id", "trade_date"]).copy()
    df["ret_1d"] = df["ret_eqw"].fillna(0.0)
    g = df.groupby("mainline_id", group_keys=False)
    for win, min_p in [(5, 3), (20, 10), (60, 30), (120, 60)]:
        df[f"ret_{win}d"] = g["ret_1d"].transform(lambda s, w=win, mp=min_p: (1.0 + s.fillna(0)).rolling(w, min_periods=mp).apply(np.prod, raw=True) - 1.0)
    for win in (20, 60, 120):
        df[f"rs_{win}d"] = df.groupby("trade_date")[f"ret_{win}d"].rank(pct=True, method="average")
    df["amount_rank_pct"] = df.groupby("trade_date")["amount_total"].rank(pct=True, method="average")
    df["amount_delta_5d"] = g["amount_total"].transform(lambda s: s / s.shift(5) - 1.0)
    df["active_member_count"] = df["member_count"].fillna(0).astype(int)
    df["leader_count"] = 0
    df["core_count"] = 0
    df["up_ratio"] = (df["ret_1d"] > 0).astype(float)
    df["strong_stock_count"] = 0
    df["new_high_20d_count"] = 0
    df["new_high_52w_count"] = 0
    df["breakout_count"] = 0
    df["breakout_ratio"] = 0.0
    df["leader_strength_score"] = 0.0
    conc = pd.to_numeric(df["top5_concentration"], errors="coerce").fillna(0.5).clip(0, 1)
    df["breadth_score"] = (0.65 * (1.0 - conc) + 0.35 * df["up_ratio"].fillna(0)).clip(0, 1)
    amt_mom = ((df["amount_delta_5d"].fillna(0).clip(-1, 1) + 1.0) / 2.0)
    df["capital_score"] = (0.70 * df["amount_rank_pct"].fillna(0) + 0.30 * amt_mom).clip(0, 1)
    df["trend_score"] = (0.50 * df["rs_20d"].fillna(0) + 0.30 * df["rs_60d"].fillna(0) + 0.20 * df["rs_120d"].fillna(0)).clip(0, 1)
    df["mainline_strength_score"] = (0.45 * df["trend_score"] + 0.25 * df["breadth_score"] + 0.30 * df["capital_score"]).clip(0, 1) * 100.0
    df["rank_no"] = df.groupby("trade_date")["mainline_strength_score"].rank(ascending=False, method="first").astype(int)
    df["source_layer"] = "FACT_CANONICAL_SW_L1"
    df["coverage_start_date"] = load_start
    df["is_backtest_eligible"] = ((df["rs_60d"].notna()) & (df["rs_120d"].notna()) & (df["active_member_count"] >= 3)).astype(int)
    df["data_quality_flag"] = np.select(
        [df["active_member_count"] < 3, df["rs_120d"].isna(), df["rs_60d"].isna()],
        ["LOW_MEMBER_COVERAGE", "INSUFFICIENT_120D_HISTORY", "INSUFFICIENT_60D_HISTORY"],
        default="OK",
    )
    df = df[(df["trade_date"] >= start) & (df["trade_date"] <= end)].copy()
    daily = df.groupby("trade_date")["mainline_id"].nunique()
    if not daily.empty:
        print(f"[CANONICAL FACT DAILY_MAINLINES] min={int(daily.min())} max={int(daily.max())} latest={int(daily.iloc[-1])}")
        if strict and (int(daily.min()) < min_daily or int(daily.max()) > max_daily):
            raise SystemExit(f"CANONICAL_UNIVERSE_DRIFT min={int(daily.min())} max={int(daily.max())} expected={min_daily}..{max_daily}")
    keep = [
        "trade_date", "mainline_id", "mainline_name", "source_layer", "member_count", "active_member_count",
        "leader_count", "core_count", "ret_1d", "ret_5d", "ret_20d", "ret_60d", "ret_120d",
        "rs_20d", "rs_60d", "rs_120d", "amount_total", "amount_rank_pct", "amount_delta_5d",
        "turnover_avg", "up_ratio", "strong_stock_count", "new_high_20d_count", "new_high_52w_count",
        "breakout_count", "breakout_ratio", "leader_strength_score", "breadth_score", "capital_score",
        "trend_score", "mainline_strength_score", "rank_no", "data_quality_flag", "coverage_start_date", "is_backtest_eligible",
    ]
    return df[keep]


def write_fact(engine: Engine, df: pd.DataFrame, start: date, end: date, replace: bool) -> int:
    if df.empty:
        return 0
    rows = df.astype(object).where(pd.notna(df), None).to_dict("records")
    cols = list(df.columns)
    sql = f"""
    INSERT INTO cn_mainline_strength_fact_daily ({', '.join(cols)})
    VALUES ({', '.join(':'+c for c in cols)})
    ON DUPLICATE KEY UPDATE {', '.join(f'{c}=VALUES({c})' for c in cols if c not in {'trade_date','mainline_id'})}
    """
    with engine.begin() as conn:
        if replace:
            conn.execute(text("DELETE FROM cn_mainline_strength_fact_daily WHERE trade_date BETWEEN :start AND :end"), {"start": start, "end": end})
        for i in range(0, len(rows), 5000):
            conn.execute(text(sql), rows[i:i+5000])
    return len(rows)


def main() -> None:
    args = build_parser().parse_args()
    start = _parse_date(args.start)
    end = _parse_date(args.end) if args.end else datetime.today().date()
    password = args.db_password if args.db_password is not None else os.getenv("MYSQL_PASSWORD", "")
    engine = build_engine(args.db_host, args.db_port, args.db_user, password, args.db_name)
    ensure_schema(engine)
    out = build_frame(engine, start, end, args.strict, args.min_daily_mainlines, args.max_daily_mainlines)
    print(f"[CANONICAL FACT] output_rows={len(out)} range={start}~{end}")
    latest = out[out["trade_date"] == out["trade_date"].max()].sort_values("rank_no").head(15)
    if not latest.empty:
        print("[CANONICAL FACT LATEST TOP]")
        print(latest[["trade_date","rank_no","mainline_id","mainline_name","mainline_strength_score","rs_20d","rs_60d","rs_120d","data_quality_flag"]].to_string(index=False))
    if args.dry_run:
        print("[CANONICAL FACT DRY-RUN] no write")
        return
    n = write_fact(engine, out, start, end, args.replace)
    print(f"[CANONICAL FACT WROTE] rows={n}")


if __name__ == "__main__":
    main()
