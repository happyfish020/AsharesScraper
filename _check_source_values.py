"""Check what source values exist in cn_local_industry_proxy_daily."""
import sys
import os
import traceback

try:
    from app.settings import build_engine
    from sqlalchemy import text

    engine = build_engine()
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT DISTINCT source FROM cn_local_industry_proxy_daily LIMIT 20
        """)).fetchall()
        print("Distinct source values:")
        for r in rows:
            print(f"  '{r[0]}'")
        
        cnt = conn.execute(text("SELECT COUNT(*) FROM cn_local_industry_proxy_daily")).scalar()
        print(f"\nTotal rows: {cnt}")
        
        # Also check what source is used in the map_hist table
        rows2 = conn.execute(text("""
            SELECT DISTINCT source FROM cn_local_industry_map_hist LIMIT 20
        """)).fetchall()
        print("\nDistinct source values in cn_local_industry_map_hist:")
        for r in rows2:
            print(f"  '{r[0]}'")
except Exception as e:
    print(f"ERROR: {e}")
    traceback.print_exc()
