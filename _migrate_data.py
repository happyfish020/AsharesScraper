"""Copy data from cn_ prefixed tables to non-prefixed tables for builder compatibility.

DEPRECATED: All builders now use cn_* tables directly.
This script is kept for reference only. Use scripts/migrate_local_tables_to_cn_prefix.py instead.
"""
import os
os.environ["ASHARE_MYSQL_USER"] = "cn_opr_red"
os.environ["ASHARE_MYSQL_PASSWORD"] = "sec_Bobo123"
os.environ["ASHARE_MYSQL_HOST"] = "localhost"
os.environ["ASHARE_MYSQL_PORT"] = "3306"
os.environ["ASHARE_MYSQL_DB"] = "cn_market_red"

from app.settings import build_engine
from sqlalchemy import text

e = build_engine()

with e.begin() as conn:
    # 1. Copy cn_local_industry_map_hist -> local_industry_map_hist
    # Map: valid_from->in_date, valid_to->out_date
    print("Copying cn_local_industry_map_hist -> local_industry_map_hist...")
    rs = conn.execute(text("""
        INSERT IGNORE INTO local_industry_map_hist
            (symbol, industry_id, industry_name, industry_level, in_date, out_date, is_current, updated_at)
        SELECT 
            symbol, 
            industry_id, 
            industry_name, 
            industry_level, 
            valid_from AS in_date, 
            COALESCE(valid_to, DATE('2099-12-31')) AS out_date,
            1 AS is_current,
            updated_at
        FROM cn_local_industry_map_hist
    """))
    print(f"  Inserted {rs.rowcount} rows into local_industry_map_hist")

    # 2. Copy cn_local_industry_proxy_daily -> local_industry_proxy_daily
    print("Copying cn_local_industry_proxy_daily -> local_industry_proxy_daily...")
    rs = conn.execute(text("""
        INSERT IGNORE INTO local_industry_proxy_daily
            (industry_id, trade_date, member_count, ret_eqw, amount_total, 
             turnover_avg, market_cap_total, leader_return, top5_concentration, updated_at)
        SELECT 
            industry_id,
            trade_date,
            member_count,
            ret_eqw,
            amount_total,
            turnover_avg,
            market_cap_total,
            leader_return,
            top5_concentration,
            updated_at
        FROM cn_local_industry_proxy_daily
    """))
    print(f"  Inserted {rs.rowcount} rows into local_industry_proxy_daily")

print("Migration complete!")
