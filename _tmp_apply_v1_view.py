"""Temporary script to apply updated v1 view."""
from app.settings import build_engine, load_sql_for_current_db
from sqlalchemy import text

engine = build_engine()

# Read the SQL
sql = load_sql_for_current_db("docs/DDL/cn_market.cn_stock_leader_score_v1.sql")

# Replace CREATE OR REPLACE with explicit DROP + CREATE to avoid DEFINER issues
sql = sql.replace(
    "CREATE OR REPLACE\nALGORITHM = UNDEFINED VIEW",
    "CREATE OR REPLACE ALGORITHM = UNDEFINED DEFINER = CURRENT_USER SQL SECURITY DEFINER VIEW",
)

print(f"SQL length: {len(sql)}")
print(f"Has 801%.SI: {'801%.SI' in sql}")
print(f"Has BK%: {'BK%' in sql}")

with engine.begin() as conn:
    conn.execute(text(sql))
    print("V1 view applied successfully")

# Verify
with engine.connect() as conn:
    r = conn.execute(text("SHOW CREATE VIEW cn_stock_leader_score_v1")).fetchone()
    view_def = r[1]
    print(f"View has 801%.SI: {'801%.SI' in view_def}")
    print(f"View has BK%: {'BK%' in view_def}")
