"""Investigate why build_stock_quality_score_daily.py produces 0 rows."""
from app.settings import build_engine
from sqlalchemy import text

engine = build_engine()

with engine.connect() as conn:
    # 1. Check cn_stock_fundamental_daily (the primary gate)
    r = conn.execute(text(
        "SELECT COUNT(*) FROM information_schema.TABLES "
        "WHERE TABLE_SCHEMA = 'cn_market_red' AND TABLE_NAME = 'cn_stock_fundamental_daily'"
    )).fetchone()
    print(f"[1] cn_stock_fundamental_daily exists: {r[0] > 0}")
    if r[0] > 0:
        r2 = conn.execute(text("SELECT COUNT(*) FROM cn_stock_fundamental_daily")).fetchone()
        print(f"    row count: {r2[0]}")
        r3 = conn.execute(text(
            "SELECT MIN(trade_date), MAX(trade_date) FROM cn_stock_fundamental_daily"
        )).fetchone()
        print(f"    date range: {r3[0]} ~ {r3[1]}")
        # Check if there's data in 2026-01-01 ~ 2026-03-30
        r4 = conn.execute(text(
            "SELECT COUNT(*) FROM cn_stock_fundamental_daily "
            "WHERE trade_date BETWEEN '2026-01-01' AND '2026-03-30'"
        )).fetchone()
        print(f"    rows in [2026-01-01, 2026-03-30]: {r4[0]}")
    else:
        print("    TABLE NOT FOUND in DB!")

    # 2. Check cn_stock_fina_indicator
    r = conn.execute(text(
        "SELECT COUNT(*) FROM information_schema.TABLES "
        "WHERE TABLE_SCHEMA = 'cn_market_red' AND TABLE_NAME = 'cn_stock_fina_indicator'"
    )).fetchone()
    print(f"\n[2] cn_stock_fina_indicator exists: {r[0] > 0}")
    if r[0] > 0:
        r2 = conn.execute(text("SELECT COUNT(*) FROM cn_stock_fina_indicator")).fetchone()
        print(f"    row count: {r2[0]}")
        r3 = conn.execute(text(
            "SELECT MIN(ann_date), MAX(ann_date) FROM cn_stock_fina_indicator"
        )).fetchone()
        print(f"    ann_date range: {r3[0]} ~ {r3[1]}")

    # 3. Check cn_stock_income
    r = conn.execute(text(
        "SELECT COUNT(*) FROM information_schema.TABLES "
        "WHERE TABLE_SCHEMA = 'cn_market_red' AND TABLE_NAME = 'cn_stock_income'"
    )).fetchone()
    print(f"\n[3] cn_stock_income exists: {r[0] > 0}")
    if r[0] > 0:
        r2 = conn.execute(text("SELECT COUNT(*) FROM cn_stock_income")).fetchone()
        print(f"    row count: {r2[0]}")
        r3 = conn.execute(text(
            "SELECT MIN(ann_date), MAX(ann_date) FROM cn_stock_income"
        )).fetchone()
        print(f"    ann_date range: {r3[0]} ~ {r3[1]}")

    # 4. Check cn_stock_balancesheet
    r = conn.execute(text(
        "SELECT COUNT(*) FROM information_schema.TABLES "
        "WHERE TABLE_SCHEMA = 'cn_market_red' AND TABLE_NAME = 'cn_stock_balancesheet'"
    )).fetchone()
    print(f"\n[4] cn_stock_balancesheet exists: {r[0] > 0}")
    if r[0] > 0:
        r2 = conn.execute(text("SELECT COUNT(*) FROM cn_stock_balancesheet")).fetchone()
        print(f"    row count: {r2[0]}")
        r3 = conn.execute(text(
            "SELECT MIN(ann_date), MAX(ann_date) FROM cn_stock_balancesheet"
        )).fetchone()
        print(f"    ann_date range: {r3[0]} ~ {r3[1]}")

    # 5. Check if cn_stock_fundamental_daily has the right columns
    if r[0] > 0:
        r = conn.execute(text(
            "SELECT COLUMN_NAME FROM information_schema.COLUMNS "
            "WHERE TABLE_SCHEMA = 'cn_market_red' AND TABLE_NAME = 'cn_stock_fundamental_daily'"
        )).fetchall()
        cols = [row[0] for row in r]
        print(f"\n[5] cn_stock_fundamental_daily columns: {cols}")

    # 6. Check if there's a cn_stock_fundamental_daily table (with cn_ prefix)
    r = conn.execute(text(
        "SELECT COUNT(*) FROM information_schema.TABLES "
        "WHERE TABLE_SCHEMA = 'cn_market_red' AND TABLE_NAME = 'cn_stock_fundamental_daily'"
    )).fetchone()
    print(f"\n[6] cn_stock_fundamental_daily exists: {r[0] > 0}")
    if r[0] > 0:
        r2 = conn.execute(text("SELECT COUNT(*) FROM cn_stock_fundamental_daily")).fetchone()
        print(f"    row count: {r2[0]}")
