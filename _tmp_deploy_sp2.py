"""部署 sp_materialize_leader_score - 使用 pymysql 原生连接"""
import pymysql

conn = pymysql.connect(
    host="localhost", port=3306,
    user="cn_opr_red", password="sec_Bobo123",
    database="cn_market_red",
    charset="utf8mb4",
    connect_timeout=30,
)

# 读取 SQL 文件
with open("sql/sp_materialize_leader_score.sql", encoding="utf-8") as f:
    raw_sql = f.read()

# 提取 DELIMITER $$ ... $$ 之间的内容
# 去掉注释和 DELIMITER 语句
lines = raw_sql.split("\n")
cleaned_lines = []
in_sp = False
for line in lines:
    stripped = line.strip()
    if stripped.upper().startswith("DELIMITER"):
        continue
    if stripped == "$$":
        continue
    if stripped.upper().startswith("DROP PROCEDURE") or stripped.upper().startswith("CREATE PROCEDURE"):
        in_sp = True
    if in_sp:
        cleaned_lines.append(line)

sp_sql = "\n".join(cleaned_lines)
print(f"SP SQL 长度: {len(sp_sql)} 字符")
print(f"前100字符: {sp_sql[:100]}")
print(f"后100字符: {sp_sql[-100:]}")

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
        # 查看 SP 定义
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
