"""Audit clean cn_mainline_strength_fact_daily before replacing radar consumers.

This script is read-only. It checks coverage, daily stability, NULL ratios,
Top-N mainlines at checkpoint dates, and optional comparison against the legacy
cn_ga_mainline_radar_daily table.

Examples:
  python scripts/audit_cn_mainline_strength_fact_daily.py --start 2024-01-01 --end 2026-06-12 --strict
  python scripts/audit_cn_mainline_strength_fact_daily.py --checkpoints 2025-03-31,2025-09-30,2026-03-31,2026-06-12
"""
from __future__ import annotations

import argparse
import os
from datetime import date, datetime
from typing import Any

import pandas as pd
from sqlalchemy import URL, create_engine, text
from sqlalchemy.engine import Engine


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Audit cn_mainline_strength_fact_daily quality and radar independence")
    p.add_argument("--start", default="2024-01-01")
    p.add_argument("--end", default="")
    p.add_argument("--db-name", default="cn_market_red")
    p.add_argument("--db-host", default="127.0.0.1")
    p.add_argument("--db-port", type=int, default=3306)
    p.add_argument("--db-user", default="cn_opr_red")
    p.add_argument("--db-password", default=None)
    p.add_argument("--top-n", type=int, default=10)
    p.add_argument("--checkpoints", default="2025-03-31,2025-09-30,2026-03-31,2026-06-12")
    p.add_argument("--max-daily-mainline-drift", type=int, default=8)
    p.add_argument("--max-rs60-null-ratio", type=float, default=0.10)
    p.add_argument("--max-rs120-null-ratio", type=float, default=0.15)
    p.add_argument("--strict", action="store_true")
    return p


def _parse_date(v: str) -> date:
    return datetime.strptime(v, "%Y-%m-%d" if "-" in v else "%Y%m%d").date()


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


def table_exists(engine: Engine, db_name: str, table_name: str) -> bool:
    sql = "SELECT COUNT(*) FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_SCHEMA=:db AND TABLE_NAME=:table"
    with engine.connect() as conn:
        return int(conn.execute(text(sql), {"db": db_name, "table": table_name}).scalar() or 0) > 0


def previous_available_date(engine: Engine, checkpoint: date) -> date | None:
    sql = """
      SELECT MAX(trade_date) AS d
      FROM cn_mainline_strength_fact_daily
      WHERE trade_date <= :checkpoint
    """
    df = fetch_df(engine, sql, {"checkpoint": checkpoint})
    if df.empty or pd.isna(df.iloc[0]["d"]):
        return None
    return df.iloc[0]["d"]


def main() -> None:
    args = build_parser().parse_args()
    start = _parse_date(args.start)
    end = _parse_date(args.end) if args.end else date.today()
    password = args.db_password or os.getenv("ASHARE_MYSQL_PASSWORD", "sec_Bobo123")
    engine = build_engine(args.db_host, args.db_port, args.db_user, password, args.db_name)

    failures: list[str] = []
    if not table_exists(engine, args.db_name, "cn_mainline_strength_fact_daily"):
        raise SystemExit("MISSING_TABLE cn_mainline_strength_fact_daily")

    summary_sql = """
      SELECT COUNT(*) AS row_count,
             COUNT(DISTINCT trade_date) AS trade_days,
             COUNT(DISTINCT mainline_id) AS mainlines,
             SUM(is_backtest_eligible=1) AS eligible_rows,
             SUM(data_quality_flag='OK') AS ok_rows,
             SUM(rs_60d IS NULL) AS rs60_null,
             SUM(rs_120d IS NULL) AS rs120_null,
             MIN(trade_date) AS min_date,
             MAX(trade_date) AS max_date
      FROM cn_mainline_strength_fact_daily
      WHERE trade_date BETWEEN :start AND :end
    """
    summary = fetch_df(engine, summary_sql, {"start": start, "end": end}).iloc[0].to_dict()
    row_count = int(summary.get("row_count") or 0)
    rs60_null = int(summary.get("rs60_null") or 0)
    rs120_null = int(summary.get("rs120_null") or 0)
    rs60_ratio = rs60_null / max(row_count, 1)
    rs120_ratio = rs120_null / max(row_count, 1)
    print("[FACT AUDIT SUMMARY]", summary)
    print(f"[FACT AUDIT NULL_RATIO] rs_60d={rs60_ratio:.4%} rs_120d={rs120_ratio:.4%}")

    if args.strict:
        if row_count <= 0:
            failures.append("row_count=0")
        if rs60_ratio > args.max_rs60_null_ratio:
            failures.append(f"rs_60d null ratio {rs60_ratio:.2%} > {args.max_rs60_null_ratio:.2%}")
        if rs120_ratio > args.max_rs120_null_ratio:
            failures.append(f"rs_120d null ratio {rs120_ratio:.2%} > {args.max_rs120_null_ratio:.2%}")

    daily_sql = """
      SELECT trade_date, COUNT(DISTINCT mainline_id) AS mainline_count
      FROM cn_mainline_strength_fact_daily
      WHERE trade_date BETWEEN :start AND :end
      GROUP BY trade_date
      ORDER BY trade_date
    """
    daily = fetch_df(engine, daily_sql, {"start": start, "end": end})
    if not daily.empty:
        min_cnt = int(daily["mainline_count"].min())
        max_cnt = int(daily["mainline_count"].max())
        drift = max_cnt - min_cnt
        print(f"[FACT AUDIT DAILY_MAINLINES] min={min_cnt} max={max_cnt} drift={drift}")
        print("[FACT AUDIT DAILY_MAINLINES LATEST]\n" + daily.tail(20).to_string(index=False))
        if args.strict and drift > args.max_daily_mainline_drift:
            failures.append(f"daily mainline count drift {drift} > {args.max_daily_mainline_drift}")

    quality = fetch_df(engine, """
      SELECT data_quality_flag, COUNT(*) AS row_count
      FROM cn_mainline_strength_fact_daily
      WHERE trade_date BETWEEN :start AND :end
      GROUP BY data_quality_flag
      ORDER BY row_count DESC
    """, {"start": start, "end": end})
    print("[FACT AUDIT QUALITY]\n" + quality.to_string(index=False))

    checkpoints = [_parse_date(x.strip()) for x in args.checkpoints.split(",") if x.strip()]
    for cp in checkpoints:
        actual = previous_available_date(engine, cp)
        if actual is None:
            print(f"[FACT AUDIT TOP] checkpoint={cp} no available date")
            if args.strict:
                failures.append(f"no available data <= checkpoint {cp}")
            continue
        top = fetch_df(engine, """
          SELECT trade_date, rank_no, mainline_id, mainline_name,
                 ROUND(mainline_strength_score, 2) AS strength,
                 ROUND(rs_20d, 4) AS rs_20d,
                 ROUND(rs_60d, 4) AS rs_60d,
                 ROUND(rs_120d, 4) AS rs_120d,
                 active_member_count, leader_count, strong_stock_count,
                 ROUND(breadth_score, 2) AS breadth_score,
                 ROUND(capital_score, 2) AS capital_score,
                 ROUND(trend_score, 2) AS trend_score,
                 data_quality_flag
          FROM cn_mainline_strength_fact_daily
          WHERE trade_date = :d
          ORDER BY rank_no ASC, mainline_strength_score DESC
          LIMIT :n
        """, {"d": actual, "n": args.top_n})
        print(f"[FACT AUDIT TOP] checkpoint={cp} actual_trade_date={actual}")
        print(top.to_string(index=False))

    if table_exists(engine, args.db_name, "cn_ga_mainline_radar_daily"):
        radar_cmp = fetch_df(engine, """
          SELECT f.trade_date,
                 COUNT(*) AS matched_rows,
                 ROUND(AVG(ABS(COALESCE(f.mainline_strength_score,0) - COALESCE(r.mainline_score,0))), 4) AS avg_abs_score_gap,
                 SUM(r.rs_60d IS NULL) AS radar_rs60_null,
                 SUM(r.rs_120d IS NULL) AS radar_rs120_null
          FROM cn_mainline_strength_fact_daily f
          JOIN cn_ga_mainline_radar_daily r
            ON r.trade_date = f.trade_date AND r.mainline_id = f.mainline_id
          WHERE f.trade_date BETWEEN :start AND :end
          GROUP BY f.trade_date
          ORDER BY f.trade_date DESC
          LIMIT 20
        """, {"start": start, "end": end})
        print("[FACT AUDIT RADAR_COMPARE LATEST]\n" + (radar_cmp.to_string(index=False) if not radar_cmp.empty else "NO_MATCH"))
    else:
        print("[FACT AUDIT RADAR_COMPARE] cn_ga_mainline_radar_daily not found; skipped")

    if failures:
        print("[FACT AUDIT FAILED]")
        for f in failures:
            print("-", f)
        raise SystemExit(2)
    print("[FACT AUDIT PASS]")


if __name__ == "__main__":
    main()
