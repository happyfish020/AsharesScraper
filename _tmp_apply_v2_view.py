"""Temporary script to re-apply v2 view after v1 update."""
from app.settings import build_engine, load_sql_for_current_db
from sqlalchemy import text

engine = build_engine()

sql = load_sql_for_current_db("docs/DDL/cn_market.cn_stock_leader_score_v2.sql")
sql = sql.replace(
    "CREATE OR REPLACE\nALGORITHM = UNDEFINED VIEW",
    "CREATE OR REPLACE ALGORITHM = UNDEFINED DEFINER = CURRENT_USER SQL SECURITY DEFINER VIEW",
)

print(f"SQL length: {len(sql)}")

with engine.begin() as conn:
    conn.execute(text(sql))
    print("V2 view applied successfully")

# Verify v2 can query data
with engine.connect() as conn:
    r = conn.execute(text("SELECT COUNT(*), MAX(trade_date) FROM cn_stock_leader_score_v2")).fetchone()
    print(f"v2 count={r[0]} max={r[1]}")
