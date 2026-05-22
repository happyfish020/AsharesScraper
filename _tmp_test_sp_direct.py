"""直接测试 SP，捕获所有输出"""
import pymysql
from pymysql.cursors import DictCursor

conn = pymysql.connect(host='localhost', port=3306, user='cn_opr_red', password='sec_Bobo123', database='cn_market_red', charset='utf8mb4')
cur = conn.cursor()

# 先检查 5 月有哪些交易日有 industry_map 数据
cur.execute("""
    SELECT DISTINCT m.trade_date 
    FROM cn_board_member_map_d m
    WHERE m.trade_date >= '2026-05-01' AND m.trade_date <= '2026-05-21'
      AND m.sector_type = 'INDUSTRY'
      AND (m.sector_id LIKE 'BK%%' OR m.sector_id LIKE '801%%.SI')
    ORDER BY m.trade_date
""")
dates = [r[0] for r in cur.fetchall()]
print("5月有industry_map数据的交易日:", dates)

# 检查 stock_daily_price 5月数据
cur.execute("""
    SELECT DISTINCT TRADE_DATE 
    FROM cn_stock_daily_price 
    WHERE TRADE_DATE >= '2026-05-01' AND TRADE_DATE <= '2026-05-21'
    ORDER BY TRADE_DATE
""")
price_dates = [r[0] for r in cur.fetchall()]
print("5月有price数据的交易日:", price_dates)

# 检查 stock_daily_basic 5月数据
cur.execute("""
    SELECT DISTINCT trade_date 
    FROM cn_stock_daily_basic 
    WHERE trade_date >= '2026-05-01' AND trade_date <= '2026-05-21'
    ORDER BY trade_date
""")
basic_dates = [r[0] for r in cur.fetchall()]
print("5月有basic数据的交易日:", basic_dates)

# 直接调用 SP 测试 5-20 ~ 5-21
print("\n--- 调用 SP 测试 2026-05-20 ~ 2026-05-21 ---")
cur.callproc("sp_materialize_leader_score", ("2026-05-20", "2026-05-21"))

# 读取所有结果集
result_set = 0
while True:
    rows = cur.fetchall()
    if rows:
        print(f"结果集 {result_set}: {rows}")
    result_set += 1
    if not cur.nextset():
        break

conn.rollback()  # 回滚，不实际写入

# 再测试 2026-05-01 ~ 2026-05-21
print("\n--- 调用 SP 测试 2026-05-01 ~ 2026-05-21 ---")
cur.callproc("sp_materialize_leader_score", ("2026-05-01", "2026-05-21"))

result_set = 0
while True:
    rows = cur.fetchall()
    if rows:
        print(f"结果集 {result_set}: {rows}")
    result_set += 1
    if not cur.nextset():
        break

conn.rollback()

cur.close()
conn.close()
