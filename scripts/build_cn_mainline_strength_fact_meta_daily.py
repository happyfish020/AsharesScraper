"""Build cn_mainline_strength_fact_daily from the cn_meta standard ID layer.

Sources:
  1) cn_meta_source_mainline_map + cn_local_industry_proxy_daily for SW-L1 market sectors
  2) cn_meta_stock_mainline_map + cn_stock_daily_price for strategic/thematic mainlines

Hard rules:
  - Does not read cn_ga_mainline_radar_daily.
  - Output mainline_id is always cn_meta_mainline_registry.mainline_id.
  - External source codes only appear in cn_meta_source_mainline_map, never as output mainline_id.

Usage:
  python scripts/seed_cn_meta_mainline_full_registry.py --apply --strict
  python scripts/audit_cn_meta_mainline_full_registry.py --strict
  python scripts/build_cn_mainline_strength_fact_meta_daily.py --start 2024-01-01 --end 2026-06-12 --dry-run --strict
  python scripts/build_cn_mainline_strength_fact_meta_daily.py --start 2024-01-01 --end 2026-06-12 --replace --strict
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

LOOKBACK_DAYS = 370
LEADER_TOKENS = ("LEADER", "CORE_LEADER", "DISCOVERY_LEADER", "SUPPLY_CHAIN_LEADER")
CORE_TOKENS = ("CORE", "LEADER", "DISCOVERY")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build meta-driven mainline strength fact table")
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
    p.add_argument("--min-latest-mainlines", type=int, default=31)
    return p


def parse_date(v: str) -> date:
    return datetime.strptime(v, "%Y-%m-%d" if "-" in v else "%Y%m%d").date()


def build_engine(host: str, port: int, user: str, password: str, db: str) -> Engine:
    url = URL.create("mysql+pymysql", username=user, password=password, host=host, port=port, database=db, query={"charset": "utf8mb4"})
    return create_engine(url, pool_pre_ping=True, future=True)


def fetch_df(engine: Engine, sql: str, params: dict[str, Any] | None = None) -> pd.DataFrame:
    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn, params=params or {})


def table_exists(engine: Engine, db_name: str, table: str) -> bool:
    return int(fetch_df(engine, """
        SELECT COUNT(*) AS n FROM INFORMATION_SCHEMA.TABLES
        WHERE TABLE_SCHEMA=:db AND TABLE_NAME=:table
    """, {"db": db_name, "table": table}).iloc[0]["n"] or 0) > 0


def ensure_schema(engine: Engine) -> None:
    ddl = """
    CREATE TABLE IF NOT EXISTS cn_mainline_strength_fact_daily (
        trade_date DATE NOT NULL,
        mainline_id VARCHAR(64) NOT NULL,
        mainline_name VARCHAR(128) NULL,
        source_layer VARCHAR(64) NOT NULL DEFAULT 'FACT_META',
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


def safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def pct_rank_by_date(df: pd.DataFrame, value_col: str, out_col: str) -> None:
    if df.empty:
        df[out_col] = np.nan
        return
    df[out_col] = df.groupby("trade_date")[value_col].rank(pct=True, method="average")


def load_registry(engine: Engine) -> pd.DataFrame:
    return fetch_df(engine, """
        SELECT mainline_id, mainline_name, category, mainline_group, display_order
        FROM cn_meta_mainline_registry
        WHERE is_active=1 AND category <> 'TEST'
    """)


def build_market_sector_fact(engine: Engine, db_name: str, start: date, end: date) -> pd.DataFrame:
    if not table_exists(engine, db_name, "cn_meta_source_mainline_map") or not table_exists(engine, db_name, "cn_local_industry_proxy_daily"):
        return pd.DataFrame()
    load_start = start - timedelta(days=LOOKBACK_DAYS)
    df = fetch_df(engine, """
        SELECT p.trade_date,
               m.mainline_id,
               r.mainline_name,
               'FACT_META_SW_L1' AS source_layer,
               COALESCE(p.member_count, 0) AS member_count,
               COALESCE(p.member_count, 0) AS active_member_count,
               COALESCE(p.ret_eqw, 0) AS ret_1d,
               COALESCE(p.amount_total, 0) AS amount_total,
               COALESCE(p.turnover_avg, 0) AS turnover_avg,
               COALESCE(p.top5_concentration, 0) AS concentration_score
        FROM cn_local_industry_proxy_daily p
        JOIN cn_meta_source_mainline_map m
          ON m.source_system='SW'
         AND m.source_level='L1'
         AND m.source_code=p.industry_id
         AND m.is_active=1
         AND m.is_primary_mapping=1
        JOIN cn_meta_mainline_registry r
          ON r.mainline_id=m.mainline_id AND r.is_active=1
        WHERE p.trade_date BETWEEN :start AND :end
    """, {"start": load_start, "end": end})
    if df.empty:
        return df
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
    for c in ["member_count", "active_member_count", "ret_1d", "amount_total", "turnover_avg", "concentration_score"]:
        df[c] = safe_num(df[c]).fillna(0.0)
    df = df.sort_values(["mainline_id", "trade_date"]).copy()
    for win in (5, 20, 60, 120):
        df[f"ret_{win}d"] = df.groupby("mainline_id")["ret_1d"].transform(lambda s, w=win: (1.0 + s).rolling(w, min_periods=max(3, min(w, 20))).apply(np.prod, raw=True) - 1.0)
    for win in (20, 60, 120):
        pct_rank_by_date(df, f"ret_{win}d", f"rs_{win}d")
    pct_rank_by_date(df, "amount_total", "amount_rank_pct")
    df["amount_delta_5d"] = df.groupby("mainline_id")["amount_total"].pct_change(5).replace([np.inf, -np.inf], np.nan)
    df["up_ratio"] = (df["ret_1d"] > 0).astype(float)
    df["strong_stock_count"] = np.where((df["ret_20d"].fillna(0) > 0.08), (df["active_member_count"] * 0.2).round(), 0).astype(int)
    df["new_high_20d_count"] = np.where(df["rs_20d"].fillna(0) >= 0.9, (df["active_member_count"] * 0.1).round(), 0).astype(int)
    df["new_high_52w_count"] = np.where(df["rs_120d"].fillna(0) >= 0.9, (df["active_member_count"] * 0.06).round(), 0).astype(int)
    df["breakout_count"] = df["new_high_20d_count"]
    df["breakout_ratio"] = np.where(df["active_member_count"] > 0, df["breakout_count"] / df["active_member_count"], 0.0)
    df["leader_count"] = 0
    df["core_count"] = 0
    df["leader_strength_score"] = 0.0
    df["breadth_score"] = (0.55 * df["up_ratio"] + 0.25 * np.minimum(1.0, df["strong_stock_count"] / df["active_member_count"].replace(0, np.nan)).fillna(0.0) + 0.20 * df["breakout_ratio"]).clip(0, 1)
    df["capital_score"] = df["amount_rank_pct"].fillna(0.0).clip(0, 1)
    df["trend_score"] = (0.45 * df["rs_20d"].fillna(0) + 0.35 * df["rs_60d"].fillna(0) + 0.20 * df["rs_120d"].fillna(0)).clip(0, 1)
    df["mainline_strength_score"] = (100.0 * (0.45 * df["trend_score"] + 0.30 * df["capital_score"] + 0.25 * df["breadth_score"])).clip(0, 100)
    df["coverage_start_date"] = df.groupby("mainline_id")["trade_date"].transform("min")
    df["data_quality_flag"] = np.where(df["rs_120d"].isna(), "INSUFFICIENT_120D_HISTORY", "OK")
    df["is_backtest_eligible"] = (df["data_quality_flag"] == "OK").astype(int)
    return df[df["trade_date"].between(start, end)].copy()


def build_theme_fact(engine: Engine, start: date, end: date) -> pd.DataFrame:
    load_start = start - timedelta(days=LOOKBACK_DAYS)
    meta = fetch_df(engine, """
        SELECT sm.ts_code AS ts_code, LEFT(sm.ts_code, 6) AS symbol, sm.stock_name, sm.mainline_id, r.mainline_name, sm.`role` AS role_type
        FROM cn_meta_stock_mainline_map sm
        JOIN cn_meta_mainline_registry r
          ON r.mainline_id=sm.mainline_id AND r.is_active=1 AND r.category <> 'TEST'
        WHERE sm.is_active=1
    """)
    if meta.empty:
        return pd.DataFrame()
    price = fetch_df(engine, """
        SELECT symbol AS symbol, trade_date AS trade_date, close AS close,
               pre_close AS pre_close, chg_pct AS chg_pct, amount AS amount,
               turnover_rate AS turnover_rate
        FROM cn_stock_daily_price
        WHERE trade_date BETWEEN :start AND :end
    """, {"start": load_start, "end": end})
    if price.empty:
        return pd.DataFrame()
    price["trade_date"] = pd.to_datetime(price["trade_date"]).dt.date
    for c in ["close", "pre_close", "chg_pct", "amount", "turnover_rate"]:
        price[c] = safe_num(price[c])
    price["ret_1d_stock"] = price["chg_pct"] / 100.0
    price = price.sort_values(["symbol", "trade_date"]).copy()
    g = price.groupby("symbol", group_keys=False)
    for win in (5, 20, 60, 120):
        price[f"ret_{win}d_stock"] = g["close"].transform(lambda s, w=win: s / s.shift(w) - 1.0)
    price["high_20d_prev"] = g["close"].transform(lambda s: s.shift(1).rolling(20, min_periods=10).max())
    price["high_52w_prev"] = g["close"].transform(lambda s: s.shift(1).rolling(252, min_periods=120).max())
    price["new_high_20d"] = price["close"] >= price["high_20d_prev"]
    price["new_high_52w"] = price["close"] >= price["high_52w_prev"]
    meta["symbol"] = meta["symbol"].astype(str)
    joined = meta.merge(price, on="symbol", how="inner")
    if joined.empty:
        return pd.DataFrame()
    joined["is_leader"] = joined["role_type"].fillna("").astype(str).apply(lambda x: any(t in x for t in LEADER_TOKENS))
    joined["is_core"] = joined["role_type"].fillna("").astype(str).apply(lambda x: any(t in x for t in CORE_TOKENS))
    joined["is_strong"] = (joined["ret_20d_stock"].fillna(0) > 0.10) | (joined["ret_60d_stock"].fillna(0) > 0.18)
    joined["is_breakout"] = joined["new_high_20d"].fillna(False)

    agg = joined.groupby(["trade_date", "mainline_id", "mainline_name"], as_index=False).agg(
        active_member_count=("symbol", "nunique"),
        ret_1d=("ret_1d_stock", "mean"),
        ret_5d=("ret_5d_stock", "mean"),
        ret_20d=("ret_20d_stock", "mean"),
        ret_60d=("ret_60d_stock", "mean"),
        ret_120d=("ret_120d_stock", "mean"),
        amount_total=("amount", "sum"),
        turnover_avg=("turnover_rate", "mean"),
        up_ratio=("ret_1d_stock", lambda s: float((s > 0).mean()) if len(s) else 0.0),
        leader_count=("is_leader", "sum"),
        core_count=("is_core", "sum"),
        strong_stock_count=("is_strong", "sum"),
        new_high_20d_count=("new_high_20d", "sum"),
        new_high_52w_count=("new_high_52w", "sum"),
        breakout_count=("is_breakout", "sum"),
        coverage_start_date=("trade_date", "min"),
    )
    member_counts = meta.groupby("mainline_id")["symbol"].nunique().rename("member_count").reset_index()
    agg = agg.merge(member_counts, on="mainline_id", how="left")
    agg["source_layer"] = "FACT_META_STOCK_THEME"
    agg["breakout_ratio"] = np.where(agg["active_member_count"] > 0, agg["breakout_count"] / agg["active_member_count"], 0.0)
    pct_rank_by_date(agg, "ret_20d", "rs_20d")
    pct_rank_by_date(agg, "ret_60d", "rs_60d")
    pct_rank_by_date(agg, "ret_120d", "rs_120d")
    pct_rank_by_date(agg, "amount_total", "amount_rank_pct")
    agg = agg.sort_values(["mainline_id", "trade_date"])
    agg["amount_delta_5d"] = agg.groupby("mainline_id")["amount_total"].pct_change(5).replace([np.inf, -np.inf], np.nan)
    agg["leader_strength_score"] = np.minimum(1.0, (0.65 * agg["leader_count"] + 0.35 * agg["core_count"]) / agg["active_member_count"].replace(0, np.nan)).fillna(0.0)
    agg["breadth_score"] = (0.45 * agg["up_ratio"].fillna(0) + 0.35 * (agg["strong_stock_count"] / agg["active_member_count"].replace(0, np.nan)).fillna(0) + 0.20 * agg["breakout_ratio"].fillna(0)).clip(0, 1)
    agg["capital_score"] = agg["amount_rank_pct"].fillna(0.0).clip(0, 1)
    agg["trend_score"] = (0.45 * agg["rs_20d"].fillna(0) + 0.35 * agg["rs_60d"].fillna(0) + 0.20 * agg["rs_120d"].fillna(0)).clip(0, 1)
    agg["mainline_strength_score"] = (100.0 * (0.38 * agg["trend_score"] + 0.24 * agg["capital_score"] + 0.23 * agg["breadth_score"] + 0.15 * agg["leader_strength_score"])).clip(0, 100)
    agg["data_quality_flag"] = np.select(
        [agg["active_member_count"] < 3, agg["rs_120d"].isna()],
        ["LOW_MEMBER_COVERAGE", "INSUFFICIENT_120D_HISTORY"],
        default="OK",
    )
    agg["is_backtest_eligible"] = (agg["data_quality_flag"] == "OK").astype(int)
    return agg[agg["trade_date"].between(start, end)].copy()


def finalize(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return rows
    wanted = [
        "trade_date", "mainline_id", "mainline_name", "source_layer", "member_count", "active_member_count",
        "leader_count", "core_count", "ret_1d", "ret_5d", "ret_20d", "ret_60d", "ret_120d",
        "rs_20d", "rs_60d", "rs_120d", "amount_total", "amount_rank_pct", "amount_delta_5d",
        "turnover_avg", "up_ratio", "strong_stock_count", "new_high_20d_count", "new_high_52w_count",
        "breakout_count", "breakout_ratio", "leader_strength_score", "breadth_score", "capital_score",
        "trend_score", "mainline_strength_score", "rank_no", "data_quality_flag", "coverage_start_date", "is_backtest_eligible",
    ]
    for c in wanted:
        if c not in rows.columns:
            rows[c] = None
    rows = rows[wanted].copy()
    rows["rank_no"] = rows.groupby("trade_date")["mainline_strength_score"].rank(ascending=False, method="first").astype("Int64")
    return rows.sort_values(["trade_date", "rank_no", "mainline_id"])


def write_rows(engine: Engine, rows: pd.DataFrame, start: date, end: date, replace: bool) -> None:
    if rows.empty:
        return
    insert_cols = list(rows.columns)
    placeholders = ", ".join([f":{c}" for c in insert_cols])
    columns = ", ".join([f"`{c}`" for c in insert_cols])
    update_cols = [c for c in insert_cols if c not in ("trade_date", "mainline_id")]
    update_sql = ", ".join([f"`{c}`=VALUES(`{c}`)" for c in update_cols])
    sql = text(f"""
        INSERT INTO cn_mainline_strength_fact_daily ({columns})
        VALUES ({placeholders})
        ON DUPLICATE KEY UPDATE {update_sql}
    """)
    records = rows.replace({np.nan: None}).to_dict("records")
    with engine.begin() as conn:
        if replace:
            conn.execute(text("DELETE FROM cn_mainline_strength_fact_daily WHERE trade_date BETWEEN :start AND :end"), {"start": start, "end": end})
        for i in range(0, len(records), 1000):
            conn.execute(sql, records[i:i + 1000])


def main() -> None:
    args = build_parser().parse_args()
    if args.replace and args.dry_run:
        raise SystemExit("Use only one of --replace or --dry-run")
    start = parse_date(args.start)
    end = parse_date(args.end) if args.end else datetime.today().date()
    password = args.db_password if args.db_password is not None else os.getenv("MYSQL_PASSWORD", "")
    engine = build_engine(args.db_host, args.db_port, args.db_user, password, args.db_name)
    ensure_schema(engine)

    sector = build_market_sector_fact(engine, args.db_name, start, end)
    theme = build_theme_fact(engine, start, end)
    if args.strict and theme.empty:
        print("[META FACT WARNING] theme_rows=0; check cn_meta_stock_mainline_map symbols and cn_stock_daily_price coverage")
    rows = finalize(pd.concat([sector, theme], ignore_index=True) if not sector.empty or not theme.empty else pd.DataFrame())

    latest_count = 0 if rows.empty else int(rows[rows["trade_date"] == rows["trade_date"].max()]["mainline_id"].nunique())
    print("[META FACT]", {
        "sector_rows": len(sector),
        "theme_rows": len(theme),
        "output_rows": len(rows),
        "latest_trade_date": None if rows.empty else rows["trade_date"].max(),
        "latest_mainlines": latest_count,
    })
    if not rows.empty:
        latest = rows[rows["trade_date"] == rows["trade_date"].max()].sort_values("rank_no").head(20)
        print("[META FACT LATEST TOP]")
        print(latest[["trade_date", "rank_no", "mainline_id", "mainline_name", "source_layer", "mainline_strength_score", "data_quality_flag"]].to_string(index=False))
    if args.strict and latest_count < args.min_latest_mainlines:
        raise SystemExit(f"[META FACT FAILED] latest mainlines {latest_count} < {args.min_latest_mainlines}")
    if args.dry_run:
        print("[META FACT DRY-RUN] no write")
        return
    write_rows(engine, rows, start, end, replace=args.replace)
    print("[META FACT WROTE]", {"rows": len(rows), "replace": bool(args.replace)})


if __name__ == "__main__":
    main()
