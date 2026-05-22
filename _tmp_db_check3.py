import pymysql

conn = pymysql.connect(host='192.168.1.240', port=3306, user='cn_opr_red', password='Op@2025#Red', db='cn_market_red')
cur = conn.cursor()

queries = [
    ("cn_stock_leader_score_daily (VIEW)", "SELECT COUNT(*), MIN(trade_date), MAX(trade_date) FROM cn_stock_leader_score_daily WHERE trade_date >= '2026-05-01'"),
    ("cn_stock_leader_score_daily_bak", "SELECT COUNT(*), MIN(trade_date), MAX(trade_date) FROM cn_stock_leader_score_daily_bak WHERE trade_date >= '2026-05-01'"),
    ("cn_stock_quality_score_daily", "SELECT COUNT(*), MIN(trade_date), MAX(trade_date) FROM cn_stock_quality_score_daily WHERE trade_date >= '2026-05-01'"),
    ("cn_stock_mainline_strength_daily", "SELECT COUNT(*), MIN(trade_date), MAX(trade_date) FROM cn_stock_mainline_strength_daily WHERE trade_date >= '2026-04-01'"),
    ("cn_ga_stock_role_map_daily", "SELECT COUNT(*), MIN(trade_date), MAX(trade_date) FROM cn_ga_stock_role_map_daily WHERE trade_date >= '2026-04-01'"),
    ("cn_ga_mainline_radar_daily", "SELECT COUNT(*), MIN(trade_date), MAX(trade_date) FROM cn_ga_mainline_radar_daily WHERE trade_date >= '2026-04-01'"),
    ("cn_ga_market_pulse_daily", "SELECT COUNT(*), MIN(trade_date), MAX(trade_date) FROM cn_ga_market_pulse_daily WHERE trade_date >= '2026-04-01'"),
]
for name, sql in queries:
    cur.execute(sql)
    r = cur.fetchone()
    print(f"{name}: rows={r[0]}, range={r[1]}~{r[2]}")

cur.close()
conn.close()
