"""直接测试 SP v2，捕获完整输出"""
import pymysql

conn = pymysql.connect(host='localhost', port=3306, user='cn_opr_red', password='sec_Bobo123', database='cn_market_red', charset='utf8mb4')
cur = conn.cursor()

# 先删除 5 月数据，确保没有冲突
print("删除 2026-05-01 ~ 2026-05-21 的旧数据...")
cur.execute("DELETE FROM cn_stock_leader_score_daily WHERE trade_date >= '2026-05-01' AND trade_date <= '2026-05-21'")
print(f"  删除了 {cur.rowcount} 行")
conn.commit()

# 调用 SP
print("\n调用 SP: 2026-05-20 ~ 2026-05-21")
cur.callproc("sp_materialize_leader_score", ("2026-05-20", "2026-05-21"))

# 读取所有结果集
result_set = 0
while True:
    rows = cur.fetchall()
    if rows:
        for row in rows:
            # 尝试解码 GBK 友好的内容
            try:
                text = str(row)
                print(f"  结果集 {result_set}: {text}")
            except:
                print(f"  结果集 {result_set}: {row}")
    result_set += 1
    if not cur.nextset():
        break

conn.commit()

# 验证
cur.execute("SELECT COUNT(*), MIN(trade_date), MAX(trade_date) FROM cn_stock_leader_score_daily WHERE trade_date >= '2026-05-01'")
r = cur.fetchone()
print(f"\n验证: count={r[0]}, min={r[1]}, max={r[2]}")

cur.close()
conn.close()
