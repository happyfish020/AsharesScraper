"""直接通过 pymysql 部署 SP，正确处理 DELIMITER"""
import pymysql

# 读取 SQL 文件
with open('sql/sp_materialize_leader_score.sql', 'r', encoding='utf-8') as f:
    sql = f.read()

# 提取 CREATE PROCEDURE 部分（去掉 DELIMITER 行和 $$ 行）
lines = sql.split('\n')
proc_lines = []
in_proc = False
for line in lines:
    stripped = line.strip()
    
    # 跳过 DELIMITER 行
    if stripped.startswith('DELIMITER '):
        continue
    
    # 检测 CREATE PROCEDURE 开始
    if stripped.startswith('CREATE PROCEDURE'):
        in_proc = True
    
    if in_proc:
        # 跳过单独的 $$ 行（结束符）
        if stripped == '$$':
            continue
        proc_lines.append(line)

proc_sql = '\n'.join(proc_lines)

print(f"提取的 SQL 长度: {len(proc_sql)} 字符")
print(f"最后 100 字符: ...{proc_sql[-100:]}")

# 检查是否还有 $$
if '$$' in proc_sql:
    print("ERROR: 仍有 $$ 残留！")
    # 找到位置
    idx = proc_sql.find('$$')
    print(f"  位置 {idx}: ...{proc_sql[max(0,idx-50):idx+50]}...")
else:
    print("OK: 无 $$ 残留")

# 先 DROP 再 CREATE
conn = pymysql.connect(host='localhost', port=3306, user='cn_opr_red', password='sec_Bobo123', database='cn_market_red', charset='utf8mb4')
cur = conn.cursor()

print("\nDROP 旧 SP...")
cur.execute("DROP PROCEDURE IF EXISTS sp_materialize_leader_score")
print("CREATE 新 SP...")
cur.execute(proc_sql)
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
