"""Fast re-apply v2 view."""
from app.settings import build_engine, load_sql_for_current_db
from sqlalchemy import text

engine = build_engine()

# Read v2 DDL and replace schema
sql = load_sql_for_current_db("docs/DDL/cn_market.cn_stock_leader_score_v2.sql")

# Add DEFINER = CURRENT_USER to avoid permission issues
sql = sql.replace(
    "CREATE OR REPLACE\nALGORITHM = UNDEFINED VIEW",
    "CREATE OR REPLACE ALGORITHM = UNDEFINED DEFINER = CURRENT_USER SQL SECURITY DEFINER VIEW",
)

print(f"SQL length: {len(sql)}")
print("Executing CREATE OR REPLACE VIEW...")

with engine.begin() as conn:
    conn.execute(text(sql))
    print("V2 view applied successfully!")

# Verify
with engine.connect() as conn:
    r = conn.execute(text("SHOW CREATE VIEW cn_stock_leader_score_v2")).fetchone()
    sql_def = r[1]
    print(f"Has 801 in v2: {'801' in sql_def}")
    print(f"Definer: {'DEFINER=' in sql_def}")
    print("Done!")
