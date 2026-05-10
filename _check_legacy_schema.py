"""Check legacy cn_local_industry_map_hist schema."""
import sys
import os
import traceback

try:
    from app.settings import build_engine
    from sqlalchemy import text

    engine = build_engine()
    with engine.connect() as conn:
        for tbl in ['cn_local_industry_map_hist', 'cn_local_industry_master', 'cn_stock_fundamental_daily']:
            print(f"\n=== {tbl} ===")
            try:
                rows = conn.execute(text(f"""
                    SELECT COLUMN_NAME, COLUMN_TYPE, IS_NULLABLE, COLUMN_DEFAULT, COLUMN_KEY
                    FROM information_schema.COLUMNS
                    WHERE TABLE_SCHEMA = DATABASE()
                      AND TABLE_NAME = '{tbl}'
                    ORDER BY ORDINAL_POSITION
                """)).fetchall()
                if not rows:
                    print("  (table not found)")
                for r in rows:
                    print(f"  {r[0]:30s} {r[1]:30s} {r[2]:10s} {str(r[3]):20s} {r[4] or ''}")
            except Exception as e:
                print(f"  ERROR: {e}")
except Exception as e:
    print(f"ERROR: {e}")
    traceback.print_exc()
