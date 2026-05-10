from app.settings import build_engine
from sqlalchemy import text
engine = build_engine()
tables = [
    "cn_mainline_strength_daily",
    "cn_mainline_radar_daily",
    "cn_ga_mainline_radar_daily",
    "cn_industry_capital_flow_daily",
    "cn_local_industry_map_hist",
    "cn_stock_leader_score_daily",
    "cn_ga_stock_role_map_daily",
    "cn_stock_daily_price",
    "cn_stock_daily_basic",
]
with engine.connect() as conn:
    for t in tables:
        r = conn.execute(text(f"SELECT COUNT(*) FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_SCHEMA = 'cn_market_red' AND TABLE_NAME = '{t}'")).scalar()
        print(f"{t}: {'EXISTS' if r > 0 else 'MISSING'}")

    # Check cn_mainline_strength_daily columns
    r = conn.execute(text("SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA = 'cn_market_red' AND TABLE_NAME = 'cn_mainline_strength_daily' ORDER BY ORDINAL_POSITION")).fetchall()
    if r:
        print("\ncn_mainline_strength_daily columns:")
        for row in r:
            print(f"  {row[0]}")

    # Check cn_ga_mainline_radar_daily columns
    r = conn.execute(text("SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA = 'cn_market_red' AND TABLE_NAME = 'cn_ga_mainline_radar_daily' ORDER BY ORDINAL_POSITION")).fetchall()
    if r:
        print("\ncn_ga_mainline_radar_daily columns:")
        for row in r:
            print(f"  {row[0]}")

    # Check cn_industry_capital_flow_daily columns
    r = conn.execute(text("SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA = 'cn_market_red' AND TABLE_NAME = 'cn_industry_capital_flow_daily' ORDER BY ORDINAL_POSITION")).fetchall()
    if r:
        print("\ncn_industry_capital_flow_daily columns:")
        for row in r:
            print(f"  {row[0]}")
