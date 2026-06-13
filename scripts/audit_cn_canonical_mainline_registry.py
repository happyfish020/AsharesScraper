"""Audit canonical mainline registry and source-code mapping coverage."""
from __future__ import annotations

import argparse
import os
from datetime import date, datetime
from typing import Any

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, URL


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Audit canonical mainline mapping coverage")
    p.add_argument("--start", default="2024-01-01")
    p.add_argument("--end", default="")
    p.add_argument("--db-name", default="cn_market_red")
    p.add_argument("--db-host", default="127.0.0.1")
    p.add_argument("--db-port", type=int, default=3306)
    p.add_argument("--db-user", default="cn_opr_red")
    p.add_argument("--db-password", default=None)
    p.add_argument("--strict", action="store_true")
    p.add_argument("--min-active-l1", type=int, default=25)
    p.add_argument("--max-unmapped-l1", type=int, default=0)
    return p


def _parse_date(v: str) -> date:
    return datetime.strptime(v, "%Y-%m-%d" if "-" in v else "%Y%m%d").date()


def build_engine(host: str, port: int, user: str, password: str, db: str) -> Engine:
    url = URL.create("mysql+pymysql", username=user, password=password, host=host, port=port, database=db, query={"charset": "utf8mb4"})
    return create_engine(url, pool_pre_ping=True, future=True)


def fetch_df(engine: Engine, sql: str, params: dict[str, Any] | None = None) -> pd.DataFrame:
    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn, params=params or {})


def main() -> None:
    args = build_parser().parse_args()
    start = _parse_date(args.start)
    end = _parse_date(args.end) if args.end else datetime.today().date()
    password = args.db_password if args.db_password is not None else os.getenv("MYSQL_PASSWORD", "")
    engine = build_engine(args.db_host, args.db_port, args.db_user, password, args.db_name)
    summary = fetch_df(engine, """
        SELECT COUNT(*) AS map_count,
               COUNT(DISTINCT canonical_mainline_id) AS canonical_count,
               SUM(is_active=1) AS active_count,
               SUM(source_level='L1' AND is_active=1) AS active_l1_count
        FROM cn_source_mainline_code_map
    """).iloc[0].to_dict()
    unmapped = fetch_df(engine, """
        SELECT p.industry_id AS source_code,
               MAX(COALESCE(p.industry_name, p.industry_id)) AS source_name,
               COUNT(DISTINCT p.trade_date) AS trade_days
        FROM cn_local_industry_proxy_daily p
        LEFT JOIN cn_source_mainline_code_map m
          ON m.source_system='SW'
         AND m.source_code=p.industry_id
         AND m.is_active=1
         AND p.trade_date BETWEEN m.effective_start_date AND COALESCE(m.effective_end_date, '2999-12-31')
        WHERE p.trade_date BETWEEN :start AND :end
          AND (p.industry_level='L1' OR p.industry_id REGEXP '^801[0-9]{3}\\.SI$')
          AND m.canonical_mainline_id IS NULL
        GROUP BY p.industry_id
        ORDER BY p.industry_id
    """, {"start": start, "end": end})
    dup = fetch_df(engine, """
        SELECT source_system, source_code, COUNT(*) AS active_mappings
        FROM cn_source_mainline_code_map
        WHERE is_active=1 AND effective_end_date IS NULL
        GROUP BY source_system, source_code
        HAVING COUNT(*) > 1
    """)
    print(f"[CANONICAL AUDIT SUMMARY] {summary}")
    print(f"[CANONICAL AUDIT UNMAPPED_L1] rows={len(unmapped)}")
    if not unmapped.empty:
        print(unmapped.head(40).to_string(index=False))
    print(f"[CANONICAL AUDIT DUP_ACTIVE] rows={len(dup)}")
    if not dup.empty:
        print(dup.head(40).to_string(index=False))
    errors: list[str] = []
    if int(summary.get("active_l1_count") or 0) < args.min_active_l1:
        errors.append(f"active L1 mappings {int(summary.get('active_l1_count') or 0)} < {args.min_active_l1}")
    if len(unmapped) > args.max_unmapped_l1:
        errors.append(f"unmapped L1 source codes {len(unmapped)} > {args.max_unmapped_l1}")
    if len(dup) > 0:
        errors.append(f"duplicate active mappings {len(dup)}")
    if errors:
        print("[CANONICAL AUDIT FAILED]")
        for e in errors:
            print(f"- {e}")
        if args.strict:
            raise SystemExit(2)
    else:
        print("[CANONICAL AUDIT PASS]")


if __name__ == "__main__":
    main()
