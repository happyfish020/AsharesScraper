"""Rename tables to add cn_ prefix and update all related files."""
import pymysql
import sys

conn = pymysql.connect(host='127.0.0.1', user='cn_opr_red', password='sec_Bobo123', database='cn_market_red')
cur = conn.cursor()

# Check current state
cur.execute("SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_SCHEMA='cn_market_red'")
all_tables = [r[0] for r in cur.fetchall()]
print(f"Total tables: {len(all_tables)}")

targets = ['mainline_lifecycle_daily', 'stock_quality_score_daily', 'unified_alpha_score_daily']
for t in targets:
    cn_t = f"cn_{t}"
    if t in all_tables and cn_t not in all_tables:
        cur.execute(f"RENAME TABLE `{t}` TO `{cn_t}`")
        print(f"RENAMED: {t} -> {cn_t}")
    elif cn_t in all_tables:
        print(f"ALREADY EXISTS: {cn_t}")
        if t in all_tables:
            # Drop the non-prefixed version if cn_ already exists
            cur.execute(f"DROP TABLE IF EXISTS `{t}`")
            print(f"DROPPED: {t} (duplicate)")
    else:
        print(f"NOT FOUND: {t}")

conn.commit()

# Verify
cur.execute("SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_SCHEMA='cn_market_red'")
updated = [r[0] for r in cur.fetchall()]
for t in targets:
    cn_t = f"cn_{t}"
    print(f"  {cn_t}: {'EXISTS' if cn_t in updated else 'MISSING'}")

cur.close()
conn.close()
print("Done")
