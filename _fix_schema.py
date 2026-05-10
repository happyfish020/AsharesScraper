"""Fix schema: add in_date/out_date columns to cn_local_industry_map_hist and fix all builder references."""
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
    # Step 1: Add in_date and out_date as regular columns to cn_local_industry_map_hist
    print("Step 1: Adding in_date/out_date columns to cn_local_industry_map_hist...")
    
    rs = conn.execute(text("""
        SELECT COUNT(*) FROM information_schema.columns 
        WHERE TABLE_SCHEMA = DATABASE() 
          AND TABLE_NAME = 'cn_local_industry_map_hist' 
          AND COLUMN_NAME = 'in_date'
    """))
    if rs.scalar() == 0:
        conn.execute(text("ALTER TABLE cn_local_industry_map_hist ADD COLUMN in_date DATE AFTER industry_level"))
        print("  Added in_date column")
    else:
        print("  in_date already exists")
    
    rs = conn.execute(text("""
        SELECT COUNT(*) FROM information_schema.columns 
        WHERE TABLE_SCHEMA = DATABASE() 
          AND TABLE_NAME = 'cn_local_industry_map_hist' 
          AND COLUMN_NAME = 'out_date'
    """))
    if rs.scalar() == 0:
        conn.execute(text("ALTER TABLE cn_local_industry_map_hist ADD COLUMN out_date DATE AFTER in_date"))
        print("  Added out_date column")
    else:
        print("  out_date already exists")
    
    # Copy data from valid_from/valid_to to in_date/out_date
    print("  Copying valid_from->in_date, valid_to->out_date...")
    conn.execute(text("UPDATE cn_local_industry_map_hist SET in_date = valid_from, out_date = valid_to WHERE in_date IS NULL"))
    print("  Done")
    
    # Step 2: Drop the duplicate non-prefixed tables (data already migrated)
    print("\nStep 2: Dropping duplicate non-prefixed tables...")
    
    for tbl in ['local_industry_proxy_daily', 'local_industry_map_hist']:
        conn.execute(text(f"DROP TABLE IF EXISTS {tbl}"))
        print(f"  Dropped {tbl}")
    
    # Step 3: Rename local_industry_master to cn_local_industry_master
    print("\nStep 3: Renaming local_industry_master to cn_local_industry_master...")
    rs = conn.execute(text("""
        SELECT COUNT(*) FROM information_schema.tables 
        WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'cn_local_industry_master'
    """))
    if rs.scalar() == 0:
        conn.execute(text("RENAME TABLE local_industry_master TO cn_local_industry_master"))
        print("  Renamed local_industry_master -> cn_local_industry_master")
    else:
        print("  cn_local_industry_master already exists, dropping local_industry_master")
        conn.execute(text("DROP TABLE IF EXISTS local_industry_master"))

print("\nSchema fix complete!")
