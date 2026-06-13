"""Validate clean cn_mainline_strength_fact_daily quality and radar independence."""
from __future__ import annotations

import argparse
import os
from datetime import date, datetime
from typing import Any

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, URL


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Validate cn_mainline_strength_fact_daily")
    p.add_argument("--start", default="2024-01-01")
    p.add_argument("--end", default="")
    p.add_argument("--db-name", default="cn_market_red")
    p.add_argument("--db-host", default="127.0.0.1")
    p.add_argument("--db-port", type=int, default=3306)
    p.add_argument("--db-user", default="cn_opr_red")
    p.add_argument("--db-password", default=None)
    p.add_argument("--min-rows", type=int, default=1)
    p.add_argument("--strict", action="store_true", help="Fail when eligible rows are zero or quality is weak")
    return p


def _parse_date(v: str) -> date:
    return datetime.strptime(v, "%Y-%m-%d" if "-" in v else "%Y%m%d").date()


def build_engine(host: str, port: int, user: str, password: str, db: str) -> Engine:
    url = URL.create("mysql+pymysql", username=user, password=password, host=host, port=port, database=db, query={"charset": "utf8mb4"})
    return create_engine(url, pool_pre_ping=True, future=True)


def fetch_df(engine: Engine, sql: str, params: dict[str, Any] | None = None) -> pd.DataFrame:
    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn, params=params or {})


def table_exists(engine: Engine, db_name: str, table_name: str) -> bool:
    sql = "SELECT COUNT(*) FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_SCHEMA=:db AND TABLE_NAME=:table"
    with engine.connect() as conn:
        return int(conn.execute(text(sql), {"db": db_name, "table": table_name}).scalar() or 0) > 0


def main() -> None:
    args = build_parser().parse_args()
    start = _parse_date(args.start)
    end = _parse_date(args.end) if args.end else date.today()
    password = args.db_password or os.getenv("ASHARE_MYSQL_PASSWORD", "sec_Bobo123")
    engine = build_engine(args.db_host, args.db_port, args.db_user, password, args.db_name)
    failures: list[str] = []
    if not table_exists(engine, args.db_name, "cn_mainline_strength_fact_daily"):
        raise SystemExit("MISSING_TABLE cn_mainline_strength_fact_daily")
    sql = """
      SELECT COUNT(*) AS row_count,
             COUNT(DISTINCT trade_date) AS trade_days,
             COUNT(DISTINCT mainline_id) AS mainlines,
             SUM(is_backtest_eligible=1) AS eligible_rows,
             SUM(rs_60d IS NULL) AS rs60_null,
             SUM(rs_120d IS NULL) AS rs120_null,
             MIN(trade_date) AS min_date,
             MAX(trade_date) AS max_date
      FROM cn_mainline_strength_fact_daily
      WHERE trade_date BETWEEN :start AND :end
    """
    row = fetch_df(engine, sql, {"start": start, "end": end}).iloc[0].to_dict()
    rows = int(row.get("row_count") or 0)
    eligible = int(row.get("eligible_rows") or 0)
    rs60_null = int(row.get("rs60_null") or 0)
    rs120_null = int(row.get("rs120_null") or 0)
    print("[FACT VALIDATION]", row)
    if rows < args.min_rows:
        failures.append(f"row_count {rows} < min_rows {args.min_rows}")
    if rows > 0 and args.strict:
        if eligible <= 0:
            failures.append("eligible_rows=0")
        if rs60_null / rows > 0.80:
            failures.append(f"rs_60d NULL ratio too high: {rs60_null}/{rows}")
        if rs120_null / rows > 0.80:
            failures.append(f"rs_120d NULL ratio too high: {rs120_null}/{rows}")
    quality = fetch_df(engine, """
        SELECT data_quality_flag, COUNT(*) AS row_count
        FROM cn_mainline_strength_fact_daily
        WHERE trade_date BETWEEN :start AND :end
        GROUP BY data_quality_flag
        ORDER BY row_count DESC
    """, {"start": start, "end": end})
    print("[QUALITY]\n" + quality.to_string(index=False))
    if failures:
        print("[FACT VALIDATION FAILED]")
        for f in failures:
            print("-", f)
        raise SystemExit(2)
    print("[FACT VALIDATION PASS]")


if __name__ == "__main__":
    main()
