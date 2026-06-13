"""Audit GrowthAlpha cn_meta mainline/subline/source mapping registry.

Checks the existing cn_meta_* metadata layer that GrowthAlpha should consume.
This script does not read cn_ga_mainline_radar_daily.

Usage:
  python scripts/audit_cn_meta_mainline_full_registry.py --strict
"""
from __future__ import annotations

import argparse
import os
from typing import Any

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, URL


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Audit full cn_meta mainline registry")
    p.add_argument("--db-name", default="cn_market_red")
    p.add_argument("--db-host", default="127.0.0.1")
    p.add_argument("--db-port", type=int, default=3306)
    p.add_argument("--db-user", default="cn_opr_red")
    p.add_argument("--db-password", default=None)
    p.add_argument("--strict", action="store_true")
    p.add_argument("--min-market-sectors", type=int, default=31)
    p.add_argument("--min-strategic-mainlines", type=int, default=12)
    p.add_argument("--min-sublines", type=int, default=35)
    return p


def build_engine(host: str, port: int, user: str, password: str, db: str) -> Engine:
    url = URL.create("mysql+pymysql", username=user, password=password, host=host, port=port, database=db, query={"charset": "utf8mb4"})
    return create_engine(url, pool_pre_ping=True, future=True)


def fetch_df(engine: Engine, sql: str, params: dict[str, Any] | None = None) -> pd.DataFrame:
    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn, params=params or {})


def table_exists(engine: Engine, db_name: str, table_name: str) -> bool:
    sql = """
        SELECT COUNT(*) AS n FROM INFORMATION_SCHEMA.TABLES
        WHERE TABLE_SCHEMA=:db AND TABLE_NAME=:table
    """
    return int(fetch_df(engine, sql, {"db": db_name, "table": table_name}).iloc[0]["n"] or 0) > 0


def main() -> None:
    args = build_parser().parse_args()
    password = args.db_password if args.db_password is not None else os.getenv("MYSQL_PASSWORD", "")
    engine = build_engine(args.db_host, args.db_port, args.db_user, password, args.db_name)

    failures: list[str] = []

    summary = fetch_df(engine, """
        SELECT COUNT(*) AS total_mainlines,
               SUM(is_active=1) AS active_mainlines,
               SUM(is_active=1 AND category='MARKET_SECTOR') AS market_sectors,
               SUM(is_active=1 AND category<>'MARKET_SECTOR' AND category<>'TEST') AS strategic_mainlines,
               SUM(is_active=1 AND category='TEST') AS active_test_mainlines
        FROM cn_meta_mainline_registry
    """).iloc[0].to_dict()
    sub_summary = fetch_df(engine, """
        SELECT COUNT(*) AS total_sublines,
               SUM(is_active=1) AS active_sublines,
               COUNT(DISTINCT mainline_id) AS subline_parent_mainlines
        FROM cn_meta_subline_registry
    """).iloc[0].to_dict()
    stock_map_summary = fetch_df(engine, """
        SELECT COUNT(*) AS stock_map_rows,
               SUM(is_active=1) AS active_stock_map_rows,
               COUNT(DISTINCT mainline_id) AS stock_map_mainlines
        FROM cn_meta_stock_mainline_map
    """).iloc[0].to_dict()

    print("[META AUDIT SUMMARY]", summary)
    print("[META AUDIT SUBLINES]", sub_summary)
    print("[META AUDIT STOCK_MAP]", stock_map_summary)

    missing_stock_parents = fetch_df(engine, """
        SELECT sm.mainline_id, COUNT(*) AS row_count
        FROM cn_meta_stock_mainline_map sm
        LEFT JOIN cn_meta_mainline_registry mr
          ON mr.mainline_id=sm.mainline_id AND mr.is_active=1
        WHERE sm.is_active=1 AND mr.mainline_id IS NULL
        GROUP BY sm.mainline_id
        ORDER BY row_count DESC, sm.mainline_id
    """)
    missing_subline_parents = fetch_df(engine, """
        SELECT sr.mainline_id, COUNT(*) AS row_count
        FROM cn_meta_subline_registry sr
        LEFT JOIN cn_meta_mainline_registry mr
          ON mr.mainline_id=sr.mainline_id AND mr.is_active=1
        WHERE sr.is_active=1 AND mr.mainline_id IS NULL
        GROUP BY sr.mainline_id
        ORDER BY row_count DESC, sr.mainline_id
    """)
    missing_subline_stock_parents = fetch_df(engine, """
        SELECT ssm.subline_id, COUNT(*) AS row_count
        FROM cn_meta_subline_stock_map ssm
        LEFT JOIN cn_meta_subline_registry sr
          ON sr.subline_id=ssm.subline_id AND sr.is_active=1
        WHERE ssm.is_active=1 AND sr.subline_id IS NULL
        GROUP BY ssm.subline_id
        ORDER BY row_count DESC, ssm.subline_id
    """)

    if not missing_stock_parents.empty:
        print("[META AUDIT MISSING_STOCK_MAINLINE_PARENTS]")
        print(missing_stock_parents.to_string(index=False))
        failures.append("active stock-mainline map references missing/inactive mainline_id")
    if not missing_subline_parents.empty:
        print("[META AUDIT MISSING_SUBLINE_MAINLINE_PARENTS]")
        print(missing_subline_parents.to_string(index=False))
        failures.append("active subline registry references missing/inactive mainline_id")
    if not missing_subline_stock_parents.empty:
        print("[META AUDIT MISSING_SUBLINE_STOCK_PARENTS]")
        print(missing_subline_stock_parents.to_string(index=False))
        failures.append("active subline-stock map references missing/inactive subline_id")

    if table_exists(engine, args.db_name, "cn_meta_source_mainline_map"):
        source_summary = fetch_df(engine, """
            SELECT source_system, source_level, COUNT(*) AS map_count,
                   COUNT(DISTINCT mainline_id) AS mapped_mainlines
            FROM cn_meta_source_mainline_map
            WHERE is_active=1
            GROUP BY source_system, source_level
            ORDER BY source_system, source_level
        """)
        print("[META AUDIT SOURCE_MAP]")
        print(source_summary.to_string(index=False))
        duplicate_source = fetch_df(engine, """
            SELECT source_system, source_level, source_code, COUNT(*) AS active_primary_count
            FROM cn_meta_source_mainline_map
            WHERE is_active=1 AND is_primary_mapping=1
            GROUP BY source_system, source_level, source_code
            HAVING COUNT(*) > 1
            ORDER BY active_primary_count DESC, source_system, source_level, source_code
        """)
        missing_source_parent = fetch_df(engine, """
            SELECT m.mainline_id, COUNT(*) AS row_count
            FROM cn_meta_source_mainline_map m
            LEFT JOIN cn_meta_mainline_registry r
              ON r.mainline_id=m.mainline_id AND r.is_active=1
            WHERE m.is_active=1 AND r.mainline_id IS NULL
            GROUP BY m.mainline_id
            ORDER BY row_count DESC, m.mainline_id
        """)
        if not duplicate_source.empty:
            print("[META AUDIT DUPLICATE_PRIMARY_SOURCE_MAP]")
            print(duplicate_source.to_string(index=False))
            failures.append("duplicate active primary source mappings")
        if not missing_source_parent.empty:
            print("[META AUDIT MISSING_SOURCE_MAP_PARENTS]")
            print(missing_source_parent.to_string(index=False))
            failures.append("source map references missing/inactive mainline_id")
    else:
        print("[META AUDIT SOURCE_MAP] cn_meta_source_mainline_map missing")
        failures.append("cn_meta_source_mainline_map table missing")

    market_sectors = int(summary.get("market_sectors") or 0)
    strategic_mainlines = int(summary.get("strategic_mainlines") or 0)
    active_sublines = int(sub_summary.get("active_sublines") or 0)
    active_test = int(summary.get("active_test_mainlines") or 0)

    if market_sectors < args.min_market_sectors:
        failures.append(f"active MARKET_SECTOR mainlines {market_sectors} < {args.min_market_sectors}")
    if strategic_mainlines < args.min_strategic_mainlines:
        failures.append(f"active strategic mainlines {strategic_mainlines} < {args.min_strategic_mainlines}")
    if active_sublines < args.min_sublines:
        failures.append(f"active sublines {active_sublines} < {args.min_sublines}")
    if active_test > 0:
        failures.append("active TEST mainlines exist")

    if failures:
        print("[META AUDIT FAILED]")
        for f in failures:
            print("-", f)
        if args.strict:
            raise SystemExit(1)
    else:
        print("[META AUDIT PASS]")


if __name__ == "__main__":
    main()
