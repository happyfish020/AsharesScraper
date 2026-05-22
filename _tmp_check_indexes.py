"""检查索引和表统计信息"""
from app.settings import build_engine
from sqlalchemy import text

engine = build_engine()
with engine.connect() as conn:
    print("=" * 60)
    print("1. cn_board_member_map_d 索引")
    print("=" * 60)
    rows = conn.execute(text("SHOW INDEX FROM cn_board_member_map_d")).fetchall()
    for r in rows:
        print(f"  {r[2]:30s}  {r[4]:20s}  seq={r[3]}  non_unique={r[1]}")

    print()
    print("=" * 60)
    print("2. cn_stock_daily_price 索引")
    print("=" * 60)
    rows = conn.execute(text("SHOW INDEX FROM cn_stock_daily_price")).fetchall()
    for r in rows:
        print(f"  {r[2]:30s}  {r[4]:20s}  seq={r[3]}  non_unique={r[1]}")

    print()
    print("=" * 60)
    print("3. cn_stock_daily_basic 索引")
    print("=" * 60)
    rows = conn.execute(text("SHOW INDEX FROM cn_stock_daily_basic")).fetchall()
    for r in rows:
        print(f"  {r[2]:30s}  {r[4]:20s}  seq={r[3]}  non_unique={r[1]}")

    print()
    print("=" * 60)
    print("4. 表行数统计")
    print("=" * 60)
    
    r = conn.execute(text("SELECT COUNT(*) FROM cn_stock_leader_score_daily")).fetchone()
    print(f"  cn_stock_leader_score_daily: {r[0]} 行")
    
    r = conn.execute(text("SELECT COUNT(*) FROM cn_board_member_map_d WHERE sector_type='INDUSTRY'")).fetchone()
    print(f"  cn_board_member_map_d (INDUSTRY): {r[0]} 行")
    
    r = conn.execute(text("SELECT COUNT(*) FROM cn_stock_daily_price")).fetchone()
    print(f"  cn_stock_daily_price: {r[0]} 行")
    
    r = conn.execute(text("SELECT COUNT(*) FROM cn_stock_daily_basic")).fetchone()
    print(f"  cn_stock_daily_basic: {r[0]} 行")
    
    r = conn.execute(text("SELECT COUNT(*) FROM cn_board_industry_master")).fetchone()
    print(f"  cn_board_industry_master: {r[0]} 行")

    print()
    print("=" * 60)
    print("5. cn_board_member_map_d 日期分布 (INDUSTRY)")
    print("=" * 60)
    rows = conn.execute(text("""
        SELECT YEAR(trade_date) AS yr, COUNT(DISTINCT trade_date) AS d_cnt, COUNT(*) AS row_cnt
        FROM cn_board_member_map_d
        WHERE sector_type='INDUSTRY'
        GROUP BY YEAR(trade_date)
        ORDER BY yr
    """)).fetchall()
    for r in rows:
        print(f"  {r[0]}: {r[1]} 天, {r[2]} 行")

    print()
    print("=" * 60)
    print("6. cn_stock_daily_price 日期分布")
    print("=" * 60)
    rows = conn.execute(text("""
        SELECT YEAR(trade_date) AS yr, COUNT(DISTINCT trade_date) AS d_cnt, COUNT(*) AS row_cnt
        FROM cn_stock_daily_price
        GROUP BY YEAR(trade_date)
        ORDER BY yr
    """)).fetchall()
    for r in rows:
        print(f"  {r[0]}: {r[1]} 天, {r[2]} 行")
