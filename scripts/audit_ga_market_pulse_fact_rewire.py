"""DB audit for P0I market pulse fact rewire."""
from __future__ import annotations
import argparse, os
from datetime import date
from typing import Any
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL, Engine

def engine(args) -> Engine:
    return create_engine(URL.create("mysql+pymysql", username=args.db_user, password=args.db_password or os.getenv("ASHARE_MYSQL_PASSWORD", ""), host=args.db_host, port=args.db_port, database=args.db_name, query={"charset":"utf8mb4"}), pool_pre_ping=True, future=True)

def fetch(e: Engine, sql: str, params: dict[str, Any]) -> pd.DataFrame:
    with e.connect() as c:
        return pd.read_sql(text(sql), c, params=params)

def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument("--start", default="2024-01-01"); p.add_argument("--end", default="")
    p.add_argument("--db-name", default="cn_market_red"); p.add_argument("--db-host", default="127.0.0.1"); p.add_argument("--db-port", type=int, default=3306); p.add_argument("--db-user", default="cn_opr_red"); p.add_argument("--db-password", default=None)
    p.add_argument("--strict", action="store_true")
    args=p.parse_args(); start=date.fromisoformat(args.start); end=date.fromisoformat(args.end) if args.end else date.today(); e=engine(args)
    summary=fetch(e, """
      SELECT COUNT(*) row_count, COUNT(DISTINCT trade_date) trade_days, MIN(trade_date) min_date, MAX(trade_date) max_date,
             SUM(CASE WHEN COALESCE(source_layer,'') <> 'FACT_META' THEN 1 ELSE 0 END) non_fact_rows
      FROM cn_ga_market_pulse_daily WHERE trade_date BETWEEN :start AND :end
    """, {"start":start,"end":end}).iloc[0].to_dict()
    missing=fetch(e, """
      SELECT f.trade_date
      FROM (SELECT DISTINCT trade_date FROM cn_mainline_strength_fact_daily WHERE trade_date BETWEEN :start AND :end) f
      LEFT JOIN cn_ga_market_pulse_daily p ON p.trade_date=f.trade_date
      WHERE p.trade_date IS NULL
      ORDER BY f.trade_date LIMIT 20
    """, {"start":start,"end":end})
    print("[MARKET PULSE AUDIT SUMMARY]", summary)
    if not missing.empty:
        print("[MARKET PULSE AUDIT MISSING]"); print(missing.to_string(index=False))
    if args.strict and (summary.get("row_count",0)==0 or float(summary.get("non_fact_rows") or 0)>0 or not missing.empty):
        raise SystemExit("[MARKET PULSE AUDIT FAILED]")
    print("[MARKET PULSE AUDIT PASS]")
if __name__=="__main__": main()
