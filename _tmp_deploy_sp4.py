"""部署 sp_materialize_leader_score - 直接去掉 $$ 和 DELIMITER"""
import pymysql

conn = pymysql.connect(
    host="localhost", port=3306,
    user="cn_opr_red", password="sec_Bobo123",
    database="cn_market_red",
    charset="utf8mb4",
    connect_timeout=30,
)

with open("sql/sp_materialize_leader_score.sql", encoding="utf-8") as f:
    raw_sql = f.read()

# 1. 去掉所有 DELIMITER 行
# 2. 去掉所有 $$
# 3. 去掉注释行
lines = []
for line in raw_sql.split("\n"):
    s = line.strip()
    if s.upper().startswith("DELIMITER"):
        continue
    if s == "$$":
        continue
    # 保留非注释行（但保留 CREATE 中的注释）
    lines.append(line)

sp_sql = "\n".join(lines)

# 验证：不应该有 $$ 或 DELIMITER
assert "$$" not in sp_sql, "仍有 $$ 残留"
assert "DELIMITER" not in sp_sql.upper(), "仍有 DELIMITER 残留"

print(f"SP SQL 长度: {len(sp_sql)} 字符")
print(f"开头: {sp_sql[:100]}")
print(f"结尾: {sp_sql[-100:]}")

try:
    with conn.cursor() as cur:
        cur.execute(sp_sql)
    conn.commit()
    print("SP 创建成功!")
except Exception as e:
    print(f"创建失败: {e}")
    conn.rollback()

# 验证
with conn.cursor() as cur:
    cur.execute("""
        SELECT COUNT(*) FROM information_schema.ROUTINES
        WHERE ROUTINE_SCHEMA = 'cn_market_red'
          AND ROUTINE_NAME = 'sp_materialize_leader_score'
          AND ROUTINE_TYPE = 'PROCEDURE'
    """)
    exists = cur.fetchone()[0] > 0
    print(f"sp_materialize_leader_score 存在: {exists}")

    if exists:
        cur.execute("""
            SELECT ROUTINE_DEFINITION
            FROM information_schema.ROUTINES
            WHERE ROUTINE_SCHEMA = 'cn_market_red'
              AND ROUTINE_NAME = 'sp_materialize_leader_score'
        """)
        defn = cur.fetchone()[0]
        print(f"SP 定义长度: {len(defn)} 字符")
        print(f"包含 #tmp_v1: {'#tmp_v1' in defn}")
        print(f"包含 #tmp_v2: {'#tmp_v2' in defn}")

conn.close()
