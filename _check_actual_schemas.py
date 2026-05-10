"""Check actual column names for cn_stock_daily_price and cn_local_industry_map_hist."""
import sys
import os
import traceback

try:
    from app.settings import build_engine
    from sqlalchemy import text

    engine = build_engine()
    with engine.connect() as conn:
        for tbl in ['cn_stock_daily_price', 'cn_local_industry_map_hist']:
            print(f"\n=== {tbl} ===")
            rows = conn.execute(text(f"""
                SELECT COLUMN_NAME, COLUMN_TYPE, IS_NULLABLE, COLUMN_DEFAULT, COLUMN_KEY
                FROM information_schema.COLUMNS
                WHERE TABLE_SCHEMA = DATABASE()
                  AND TABLE_NAME = '{tbl}'
                ORDER BY ORDINAL_POSITION
            """)).fetchall()
            for r in rows:
                print(f"  {r[0]:30s} {r[1]:30s} {r[2]:10s} {str(r[3]):20s} {r[4] or ''}")
except Exception as e:
    print(f"ERROR: {e}")
    traceback.print_exc()
