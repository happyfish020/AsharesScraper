"""检查数据库中 SP 的定义"""
import pymysql

conn = pymysql.connect(host='localhost', port=3306, user='cn_opr_red', password='sec_Bobo123', database='cn_market_red', charset='utf8mb4')
cur = conn.cursor()

cur.execute("SHOW CREATE PROCEDURE sp_materialize_leader_score")
row = cur.fetchone()
body = row[2]  # 第三列是 body

# 检查是否包含 source = t2.source
if 'source = t2.source' in body:
    print("ERROR: 数据库中的 SP 仍包含 'source = t2.source'")
else:
    print("OK: 数据库中的 SP 已移除 'source = t2.source'")

# 检查是否包含 VALUES(
if 'VALUES(' in body:
    print("ERROR: 数据库中的 SP 仍包含 VALUES() 语法")
else:
    print("OK: 数据库中的 SP 已使用 AS alias 语法")

# 打印最后 500 字符
print("\n--- SP 末尾 500 字符 ---")
print(body[-500:])

cur.close()
conn.close()
