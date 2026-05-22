"""部署 sp_materialize_leader_score 到数据库"""
from app.settings import build_engine
from sqlalchemy import text

engine = build_engine()
sql = open("sql/sp_materialize_leader_score.sql", encoding="utf-8").read()

# 按 DELIMITER 分割多个语句
statements = []
current = []
for line in sql.split("\n"):
    stripped = line.strip()
    if stripped.upper() == "DELIMITER $$":
        continue
    if stripped.upper() == "DELIMITER ;":
        continue
    if stripped == "$$":
        if current:
            statements.append("\n".join(current))
            current = []
        continue
    current.append(line)
if current:
    statements.append("\n".join(current))

print(f"共 {len(statements)} 个语句块")

with engine.begin() as conn:
    for i, stmt in enumerate(statements):
        stmt = stmt.strip()
        if not stmt:
            continue
        try:
            conn.execute(text(stmt))
            print(f"  [{i+1}] 执行成功 ({len(stmt)} chars)")
        except Exception as e:
            print(f"  [{i+1}] 执行失败: {e}")
            print(f"    前200字符: {stmt[:200]}")

print("\n验证存储过程...")
with engine.connect() as conn:
    r = conn.execute(text("""
        SELECT COUNT(*) FROM information_schema.ROUTINES
        WHERE ROUTINE_SCHEMA = 'cn_market_red'
          AND ROUTINE_NAME = 'sp_materialize_leader_score'
          AND ROUTINE_TYPE = 'PROCEDURE'
    """)).fetchone()
    print(f"  sp_materialize_leader_score 存在: {r[0] > 0}")
