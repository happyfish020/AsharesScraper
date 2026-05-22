import pymysql
conn = pymysql.connect(host='localhost', port=3306, user='cn_opr_red', password='sec_Bobo123', database='cn_market_red', charset='utf8mb4')
cur = conn.cursor()

# 1. board_member_map_d 最新日期
cur.execute("SELECT MAX(trade_date) FROM cn_board_member_map_d")
r = cur.fetchone()
print("board_member_map_d MAX(trade_date):", r[0])

# 2. board_member_map_d 在目标范围的数据
cur.execute("SELECT MIN(trade_date), MAX(trade_date), COUNT(*) FROM cn_board_member_map_d WHERE trade_date >= '2026-04-22' AND trade_date <= '2026-05-21'")
r = cur.fetchone()
print("board_member_map_d 0422~0521: min=%s max=%s count=%s" % (r[0], r[1], r[2]))

# 3. board_member_map_d INDUSTRY 在目标范围的数据
cur.execute("SELECT MIN(trade_date), MAX(trade_date), COUNT(*) FROM cn_board_member_map_d WHERE trade_date >= '2026-04-22' AND trade_date <= '2026-05-21' AND sector_type='INDUSTRY' AND (sector_id LIKE 'BK%%' OR sector_id LIKE '801%%.SI')")
r = cur.fetchone()
print("board_member_map_d INDUSTRY 0422~0521: min=%s max=%s count=%s" % (r[0], r[1], r[2]))

# 4. leader_score_daily 最新日期
cur.execute("SELECT MAX(trade_date) FROM cn_stock_leader_score_daily")
r = cur.fetchone()
print("leader_score_daily MAX(trade_date):", r[0])

# 5. stock_daily_price 最新日期
cur.execute("SELECT MAX(TRADE_DATE) FROM cn_stock_daily_price")
r = cur.fetchone()
print("stock_daily_price MAX(TRADE_DATE):", r[0])

cur.close()
conn.close()
