"""Quick check of radar data values."""
import os
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

conn_url = URL.create(
    drivername="mysql+pymysql",
    username="cn_opr_red",
    password=os.getenv("ASHARE_MYSQL_PASSWORD", "sec_Bobo123"),
    host="localhost",
    port=3306,
    database="cn_market_red",
    query={"charset": "utf8mb4"},
)
engine = create_engine(conn_url, pool_pre_ping=True, future=True)

with engine.connect() as conn:
    # Sample rows
    df = pd.read_sql(
        text("SELECT mainline_score, leader_density, breakout_ratio, trend_alignment_score, leader_count, strong_stock_count, rotation_rank FROM cn_ga_mainline_radar_daily WHERE trade_date = '2026-01-05' LIMIT 10"),
        conn
    )
    print("=== Sample rows 2026-01-05 ===")
    print(df.to_string())
    print()

    # Stats
    df2 = pd.read_sql(
        text("""
        SELECT 
            MIN(mainline_score) as min_s, MAX(mainline_score) as max_s, AVG(mainline_score) as avg_s,
            MIN(leader_density) as min_ld, MAX(leader_density) as max_ld, AVG(leader_density) as avg_ld,
            MIN(breakout_ratio) as min_br, MAX(breakout_ratio) as max_br, AVG(breakout_ratio) as avg_br,
            MIN(trend_alignment_score) as min_ta, MAX(trend_alignment_score) as max_ta, AVG(trend_alignment_score) as avg_ta
        FROM cn_ga_mainline_radar_daily 
        WHERE trade_date BETWEEN '2026-01-05' AND '2026-01-10'
        """),
        conn
    )
    print("=== Stats 2026-01-05 ~ 2026-01-10 ===")
    print(df2.to_string())
    print()

    # Check output table
    df3 = pd.read_sql(
        text("SELECT mainline_strength, mainline_phase, rotation_rank FROM cn_mainline_strength_daily WHERE trade_date = '2026-01-05' ORDER BY rotation_rank LIMIT 10"),
        conn
    )
    print("=== Output sample 2026-01-05 ===")
    print(df3.to_string())
    print()

    # Check distinct values
    df4 = pd.read_sql(
        text("SELECT mainline_strength, COUNT(*) as cnt FROM cn_mainline_strength_daily WHERE trade_date = '2026-01-05' GROUP BY mainline_strength ORDER BY cnt DESC LIMIT 10"),
        conn
    )
    print("=== Output strength distribution 2026-01-05 ===")
    print(df4.to_string())
