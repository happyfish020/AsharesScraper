"""Audit cn_mainline_lifecycle_daily after P0H fact-layer rewire."""
from __future__ import annotations

import argparse
import os
from datetime import date, datetime
from typing import Any

import pandas as pd
from sqlalchemy import URL, create_engine, text
from sqlalchemy.engine import Engine


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Audit fact-based cn_mainline_lifecycle_daily")
    p.add_argument("--start", default="2024-01-01")
    p.add_argument("--end", default="")
    p.add_argument("--db-name", default="cn_market_red")
    p.add_argument("--db-host", default="127.0.0.1")
    p.add_argument("--db-port", type=int, default=3306)
    p.add_argument("--db-user", default="cn_opr_red")
    p.add_argument("--db-password", default=None)
    p.add_argument("--strict", action="store_true")
    p.add_argument("--expected-daily-mainlines", type=int, default=41)
    return p


def parse_date(v: str) -> date:
    return datetime.strptime(v, "%Y-%m-%d" if "-" in v else "%Y%m%d").date()


def build_engine(host: str, port: int, user: str, password: str, db: str) -> Engine:
    url = URL.create("mysql+pymysql", username=user, password=password, host=host, port=port, database=db, query={"charset": "utf8mb4"})
    return create_engine(url, pool_pre_ping=True, future=True)


def fetch_df(engine: Engine, sql: str, params: dict[str, Any] | None = None) -> pd.DataFrame:
    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn, params=params or {})


def main() -> None:
    args = build_parser().parse_args()
    start = parse_date(args.start)
    end = parse_date(args.end) if args.end else datetime.today().date()
    password = args.db_password if args.db_password is not None else (os.getenv("ASHARE_MYSQL_PASSWORD") or os.getenv("MYSQL_PASSWORD") or "")
    engine = build_engine(args.db_host, args.db_port, args.db_user, password, args.db_name)

    summary = fetch_df(engine, """
        SELECT COUNT(*) AS row_count,
               COUNT(DISTINCT trade_date) AS trade_days,
               COUNT(DISTINCT mainline_id) AS mainlines,
               MIN(trade_date) AS min_date,
               MAX(trade_date) AS max_date,
               SUM(source_layer IS NULL OR source_layer NOT LIKE 'FACT_META%') AS non_fact_rows
        FROM cn_mainline_lifecycle_daily
        WHERE trade_date BETWEEN :start AND :end
    """, {"start": start, "end": end}).iloc[0].to_dict()
    print("[LIFECYCLE AUDIT SUMMARY]", summary)

    daily = fetch_df(engine, """
        SELECT trade_date, COUNT(DISTINCT mainline_id) AS mainline_count
        FROM cn_mainline_lifecycle_daily
        WHERE trade_date BETWEEN :start AND :end
        GROUP BY trade_date
        ORDER BY trade_date DESC
    """, {"start": start, "end": end})
    print("[LIFECYCLE AUDIT DAILY LATEST]")
    print(daily.head(20).to_string(index=False))

    states = fetch_df(engine, """
        SELECT lifecycle_state, COUNT(*) AS row_count
        FROM cn_mainline_lifecycle_daily
        WHERE trade_date BETWEEN :start AND :end
        GROUP BY lifecycle_state
        ORDER BY row_count DESC
    """, {"start": start, "end": end})
    print("[LIFECYCLE AUDIT STATES]")
    print(states.to_string(index=False))

    compare = fetch_df(engine, """
        SELECT f.trade_date,
               COUNT(DISTINCT f.mainline_id) AS fact_mainlines,
               COUNT(DISTINCT l.mainline_id) AS lifecycle_mainlines
        FROM cn_mainline_strength_fact_daily f
        LEFT JOIN cn_mainline_lifecycle_daily l
          ON l.trade_date=f.trade_date AND l.mainline_id=f.mainline_id
        WHERE f.trade_date BETWEEN :start AND :end
        GROUP BY f.trade_date
        HAVING fact_mainlines <> lifecycle_mainlines
        ORDER BY f.trade_date
        LIMIT 20
    """, {"start": start, "end": end})
    if not compare.empty:
        print("[LIFECYCLE AUDIT MISMATCH SAMPLE]")
        print(compare.to_string(index=False))

    issues: list[str] = []
    if int(summary.get("row_count") or 0) <= 0:
        issues.append("lifecycle table has 0 rows")
    if int(summary.get("non_fact_rows") or 0) > 0:
        issues.append(f"non-fact lifecycle rows detected: {int(summary.get('non_fact_rows') or 0)}")
    bad_daily = daily[daily["mainline_count"] != args.expected_daily_mainlines]
    if not bad_daily.empty:
        issues.append(f"daily mainline count not equal {args.expected_daily_mainlines}: bad_days={len(bad_daily)}")
    if not compare.empty:
        issues.append("fact/lifecycle row coverage mismatch")

    if args.strict and issues:
        print("[LIFECYCLE AUDIT FAILED]")
        for issue in issues:
            print(f"- {issue}")
        raise SystemExit(1)
    print("[LIFECYCLE AUDIT PASS]")


if __name__ == "__main__":
    main()
