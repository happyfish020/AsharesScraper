"""Check which database we're connected to."""
import sys
import os
import traceback

try:
    from app.settings import build_engine
    from sqlalchemy import text

    engine = build_engine()
    with engine.connect() as conn:
        db = conn.execute(text("SELECT DATABASE()")).scalar()
        print(f"Connected to database: {db}")
        
        # Check cn_local_industry_map_hist in this DB
        rows = conn.execute(text(f"""
            SELECT COLUMN_NAME, COLUMN_TYPE
            FROM information_schema.COLUMNS
            WHERE TABLE_SCHEMA = '{db}'
              AND TABLE_NAME = 'cn_local_industry_map_hist'
            ORDER BY ORDINAL_POSITION
        """)).fetchall()
        print(f"\ncn_local_industry_map_hist columns in {db}:")
        for r in rows:
            print(f"  {r[0]:30s} {r[1]}")
except Exception as e:
    print(f"ERROR: {e}")
    traceback.print_exc()
