"""Check data relationship between cn_stock_* and cn_local_stock_*_q tables."""
import sys
sys.path.insert(0, '.')
from app.settings import build_engine
from sqlalchemy import text

engine = build_engine()

with engine.connect() as conn:
    print("=" * 70)
    print("TABLE ROW COUNTS")
    print("=" * 70)
    for tbl in ['cn_stock_income', 'cn_stock_balancesheet', 'cn_stock_cashflow', 
                'cn_stock_daily_basic', 'cn_stock_fina_indicator']:
        r = conn.execute(text(f'SELECT COUNT(*) FROM {tbl}'))
        print(f"  {tbl:<40s} {r.scalar():>10,} rows")
    
    print()
    for tbl in ['cn_local_stock_income_q', 'cn_local_stock_balancesheet_q', 'cn_local_stock_fina_indicator_q']:
        r = conn.execute(text(f'SELECT COUNT(*) FROM {tbl}'))
        print(f"  {tbl:<40s} {r.scalar():>10,} rows")
    
    print()
    r = conn.execute(text('SELECT COUNT(*) FROM cn_stock_fundamental_daily'))
    print(f"  {'cn_stock_fundamental_daily':<40s} {r.scalar():>10,} rows")
    
    print()
    print("=" * 70)
    print("DISTINCT SYMBOLS (date range 2026-01-01 ~ 2026-03-30)")
    print("=" * 70)
    pairs = [
        ('cn_stock_income', 'cn_local_stock_income_q'),
        ('cn_stock_balancesheet', 'cn_local_stock_balancesheet_q'),
        ('cn_stock_fina_indicator', 'cn_local_stock_fina_indicator_q'),
    ]
    for cn, loc in pairs:
        r1 = conn.execute(text(
            f"SELECT COUNT(DISTINCT symbol) FROM {cn} "
            f"WHERE end_date BETWEEN '2026-01-01' AND '2026-03-30'"
        ))
        r2 = conn.execute(text(f"SELECT COUNT(DISTINCT symbol) FROM {loc}"))
        print(f"  {cn:<35s} {r1.scalar():>6} symbols")
        print(f"  {loc:<35s} {r2.scalar():>6} symbols")
        print()
    
    print("=" * 70)
    print("COLUMN COMPARISON: cn_stock_income vs cn_local_stock_income_q")
    print("=" * 70)
    r = conn.execute(text("DESC cn_local_stock_income_q"))
    print("  cn_local_stock_income_q columns:")
    for row in r:
        print(f"    {row[0]:<25s} {row[1]}")
    
    print()
    r = conn.execute(text("SELECT COLUMN_NAME, COLUMN_TYPE FROM information_schema.COLUMNS "
                          "WHERE TABLE_SCHEMA = (SELECT DATABASE()) AND TABLE_NAME = 'cn_stock_income' "
                          "ORDER BY ORDINAL_POSITION"))
    print("  cn_stock_income columns (first 15):")
    for i, row in enumerate(r):
        if i >= 15:
            print("    ...")
            break
        print(f"    {row[0]:<25s} {row[1]}")
