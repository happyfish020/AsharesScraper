from app.settings import build_engine
from sqlalchemy import text

e = build_engine()
with e.connect() as c:
    # cn_local_industry_map_hist
    cols = c.execute(text("DESCRIBE cn_local_industry_map_hist")).fetchall()
    print("cn_local_industry_map_hist cols:", [(r[0], r[1]) for r in cols])
    cnt = c.execute(text("SELECT COUNT(*) FROM cn_local_industry_map_hist")).scalar()
    print("Row count:", cnt)
    sample = c.execute(text("SELECT * FROM cn_local_industry_map_hist LIMIT 3")).fetchall()
    print("Sample:", sample)
    
    # local_industry_map_hist
    cols2 = c.execute(text("DESCRIBE local_industry_map_hist")).fetchall()
    print("local_industry_map_hist cols:", [(r[0], r[1]) for r in cols2])
    cnt2 = c.execute(text("SELECT COUNT(*) FROM local_industry_map_hist")).scalar()
    print("local_industry_map_hist count:", cnt2)
    
    # cn_board_member_map_d
    cols3 = c.execute(text("DESCRIBE cn_board_member_map_d")).fetchall()
    print("cn_board_member_map_d cols:", [(r[0], r[1]) for r in cols3])
    cnt3 = c.execute(text("SELECT COUNT(*) FROM cn_board_member_map_d")).scalar()
    print("cn_board_member_map_d count:", cnt3)
    sample3 = c.execute(text("SELECT * FROM cn_board_member_map_d LIMIT 3")).fetchall()
    print("cn_board_member_map_d sample:", sample3)
