"""直接通过 pymysql 部署 SP，正确处理 DELIMITER 和 $$"""
import pymysql
import re

# 读取 SQL 文件
with open('sql/sp_materialize_leader_score.sql', 'r', encoding='utf-8') as f:
    sql = f.read()

# 方法: 去掉 DELIMITER 行，将 $$ 替换为 ;
cleaned = sql.replace('DELIMITER $$\n', '')
cleaned = cleaned.replace('DELIMITER ;\n', '')
# 将行尾的 $$ 替换为 ;（保留行内容，如 END sp_main $$ → END sp_main ;）
cleaned = re.sub(r'\$\$\s*$', ';', cleaned, flags=re.MULTILINE)

print(f"清理后 SQL 长度: {len(cleaned)} 字符")

if '$$' in cleaned:
    print("ERROR: 仍有 $$ 残留！")
else:
    print("OK: 无 $$ 残留")

# 分离 DROP 和 CREATE 语句
# DROP PROCEDURE IF EXISTS sp_materialize_leader_score ;
# CREATE PROCEDURE sp_materialize_leader_score( ... ) ... END sp_main ;
# 找到 CREATE 的位置
create_idx = cleaned.find('CREATE PROCEDURE')
if create_idx < 0:
    print("ERROR: 未找到 CREATE PROCEDURE")
    exit(1)

drop_stmt = cleaned[:create_idx].strip()  # DROP PROCEDURE IF EXISTS ...
create_stmt = cleaned[create_idx:].strip()  # CREATE PROCEDURE ...

print(f"\nDROP 语句: {drop_stmt[:80]}...")
print(f"CREATE 语句长度: {len(create_stmt)} 字符")

conn = pymysql.connect(host='localhost', port=3306, user='cn_opr_red', password='sec_Bobo123', database='cn_market_red', charset='utf8mb4')
cur = conn.cursor()

print("\nDROP 旧 SP...")
cur.execute(drop_stmt)
print("CREATE 新 SP...")
cur.execute(create_stmt)
print("COMMIT...")
conn.commit()

# 验证
cur.execute("SHOW CREATE PROCEDURE sp_materialize_leader_score")
row = cur.fetchone()
body = row[2]

if 'VALUES(' in body:
    print("ERROR: 部署后仍包含 VALUES() 语法！")
else:
    print("OK: VALUES() 已移除")

if 'source = t2.source' in body:
    print("ERROR: 部署后仍包含 source = t2.source！")
else:
    print("OK: source = t2.source 已移除")

if 't2.industry_id' in body:
    print("OK: 使用了 AS alias 语法 (t2.xxx)")
else:
    print("ERROR: 未使用 AS alias 语法")

# 打印 INSERT ... ON DUPLICATE KEY UPDATE 部分
idx = body.find('ON DUPLICATE KEY UPDATE')
if idx >= 0:
    print("\n--- ON DUPLICATE KEY UPDATE 部分 ---")
    print(body[idx:idx+800])

cur.close()
conn.close()
