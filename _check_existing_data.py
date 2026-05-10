"""Check existing cn_stock_* tables and their schemas for the materialization."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from app.settings import build_engine
from sqlalchemy import text

engine = build_engine()

with engine.connect() as c:
    # Check cn_stock_income
    r = c.execute(text("SELECT COLUMN_NAME, DATA_TYPE FROM information_schema.COLUMNS WHERE TABLE_SCHEMA=DATABASE() AND TABLE_NAME='cn_stock_income' ORDER BY ORDINAL_POSITION"))
    print("=== cn_stock_income ===")
    for row in r:
        print(f"  {row[0]} ({row[1]})")
    r = c.execute(text("SELECT COUNT(*) FROM cn_stock_income"))
    print(f"  rows: {r.scalar()}")

    # Check cn_stock_balancesheet
    r = c.execute(text("SELECT COLUMN_NAME, DATA_TYPE FROM information_schema.COLUMNS WHERE TABLE_SCHEMA=DATABASE() AND TABLE_NAME='cn_stock_balancesheet' ORDER BY ORDINAL_POSITION"))
    print("\n=== cn_stock_balancesheet ===")
    for row in r:
        print(f"  {row[0]} ({row[1]})")
    r = c.execute(text("SELECT COUNT(*) FROM cn_stock_balancesheet"))
    print(f"  rows: {r.scalar()}")

    # Check cn_stock_fina_indicator
    r = c.execute(text("SELECT COLUMN_NAME, DATA_TYPE FROM information_schema.COLUMNS WHERE TABLE_SCHEMA=DATABASE() AND TABLE_NAME='cn_stock_fina_indicator' ORDER BY ORDINAL_POSITION"))
    print("\n=== cn_stock_fina_indicator ===")
    for row in r:
        print(f"  {row[0]} ({row[1]})")
    r = c.execute(text("SELECT COUNT(*) FROM cn_stock_fina_indicator"))
    print(f"  rows: {r.scalar()}")

    # Check cn_stock_fundamental_daily
    r = c.execute(text("SELECT COLUMN_NAME, DATA_TYPE FROM information_schema.COLUMNS WHERE TABLE_SCHEMA=DATABASE() AND TABLE_NAME='cn_stock_fundamental_daily' ORDER BY ORDINAL_POSITION"))
    print("\n=== cn_stock_fundamental_daily ===")
    for row in r:
        print(f"  {row[0]} ({row[1]})")
    r = c.execute(text("SELECT COUNT(*) FROM cn_stock_fundamental_daily"))
    print(f"  rows: {r.scalar()}")

    # Check cn_local_stock_*_q tables
    for tbl in ['cn_local_stock_income_q', 'cn_local_stock_balancesheet_q', 'cn_local_stock_fina_indicator_q']:
        r = c.execute(text(f"SELECT COUNT(*) FROM information_schema.TABLES WHERE TABLE_SCHEMA=DATABASE() AND TABLE_NAME='{tbl}'"))
        exists = r.scalar() > 0
        print(f"\n=== {tbl} exists={exists} ===")
        if exists:
            r = c.execute(text(f"SELECT COLUMN_NAME, DATA_TYPE FROM information_schema.COLUMNS WHERE TABLE_SCHEMA=DATABASE() AND TABLE_NAME='{tbl}' ORDER BY ORDINAL_POSITION"))
            for row in r:
                print(f"  {row[0]} ({row[1]})")
            r = c.execute(text(f"SELECT COUNT(*) FROM {tbl}"))
            print(f"  rows: {r.scalar()}")

    # Sample data from cn_stock_income
    r = c.execute(text("SELECT * FROM cn_stock_income LIMIT 3"))
    print("\n=== cn_stock_income sample ===")
    for row in r:
        print(f"  {dict(row)}")

    # Sample data from cn_stock_fina_indicator
    r = c.execute(text("SELECT * FROM cn_stock_fina_indicator LIMIT 3"))
    print("\n=== cn_stock_fina_indicator sample ===")
    for row in r:
        print(f"  {dict(row)}")
