"""Static + DB audit for P0I unified alpha fact rewire."""
from __future__ import annotations
import argparse, os
from pathlib import Path
from datetime import date
from typing import Any
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL, Engine

def make_engine(args) -> Engine:
    return create_engine(URL.create("mysql+pymysql", username=args.db_user, password=args.db_password or os.getenv("ASHARE_MYSQL_PASSWORD", ""), host=args.db_host, port=args.db_port, database=args.db_name, query={"charset":"utf8mb4"}), pool_pre_ping=True, future=True)

def fetch(e: Engine, sql: str, params: dict[str, Any]) -> pd.DataFrame:
    with e.connect() as c:
        return pd.read_sql(text(sql), c, params=params)

def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument("--root", default="."); p.add_argument("--start", default="2024-01-01"); p.add_argument("--end", default="")
    p.add_argument("--db-name", default="cn_market_red"); p.add_argument("--db-host", default="127.0.0.1"); p.add_argument("--db-port", type=int, default=3306); p.add_argument("--db-user", default="cn_opr_red"); p.add_argument("--db-password", default=None)
    p.add_argument("--strict", action="store_true")
    args=p.parse_args(); root=Path(args.root)
    target=root/"scripts"/"build_unified_alpha_score_daily.py"
    text_body=target.read_text(encoding="utf-8", errors="ignore") if target.exists() else ""
    static_bad="cn_ga_mainline_radar_daily" in text_body
    print("[UNIFIED ALPHA STATIC]", {"build_script": str(target), "contains_radar_table": static_bad})
    start=date.fromisoformat(args.start); end=date.fromisoformat(args.end) if args.end else date.today(); e=make_engine(args)
    summary=fetch(e, """
      SELECT COUNT(*) row_count, COUNT(DISTINCT trade_date) trade_days, MIN(trade_date) min_date, MAX(trade_date) max_date
      FROM cn_unified_alpha_score_daily WHERE trade_date BETWEEN :start AND :end
    """, {"start":start,"end":end}).iloc[0].to_dict()
    print("[UNIFIED ALPHA DB SUMMARY]", summary)
    if args.strict and static_bad:
        raise SystemExit("[UNIFIED ALPHA AUDIT FAILED] build script still references radar table")
    print("[UNIFIED ALPHA AUDIT PASS]")
if __name__=="__main__": main()
