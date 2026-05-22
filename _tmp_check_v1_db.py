"""Check v1 view DDL in database."""
from app.settings import build_engine
from sqlalchemy import text

e = build_engine()
with e.connect() as c:
    r = c.execute(text("SHOW CREATE VIEW cn_stock_leader_score_v1")).fetchone()
    sql = r[1]
    print(f"Has 801: {'801' in sql}")
    print(f"Has BK: {'BK' in sql}")
    import re
    m = re.search(r"DEFINER=\S+", sql)
    print(f"Definer: {m.group(0) if m else 'N/A'}")
    # Check the industry_map filter section
    if "801" in sql:
        idx = sql.find("801")
        print(f"Context around 801: ...{sql[max(0,idx-50):idx+100]}...")
c.close()
