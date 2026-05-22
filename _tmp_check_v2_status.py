"""Check v2 view status."""
from app.settings import build_engine
from sqlalchemy import text

e = build_engine()
with e.connect() as c:
    r = c.execute(text("SHOW CREATE VIEW cn_stock_leader_score_v2")).fetchone()
    sql = r[1]
    # Check if it references v1
    if "cn_stock_leader_score_v1" in sql:
        print("v2 references v1: YES")
    else:
        print("v2 references v1: NO")
    # Check definer
    if "DEFINER=" in sql:
        import re
        m = re.search(r"DEFINER=\S+", sql)
        print(f"Definer: {m.group(0) if m else 'N/A'}")
    # Check for 801
    if "801" in sql:
        print("v2 contains 801: YES")
    else:
        print("v2 contains 801: NO")
    print(f"v2 SQL length: {len(sql)}")
    print("First 300 chars:", sql[:300])
c.close()
