"""Check more radar columns."""
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
    # Check member_count, leader_score_avg, mainline_state
    df = pd.read_sql(
        text("""
        SELECT trade_date, mainline_id, mainline_name, mainline_score, mainline_state,
               member_count, leader_count, leader_score_avg, rank_no, rs_rank,
               up_ratio, avg_ret, heat_percentile_5d, mainline_confidence
        FROM cn_ga_mainline_radar_daily
        WHERE trade_date = '2026-01-05'
        LIMIT 10
        """),
        conn
    )
    print("=== Sample with more columns ===")
    print(df.to_string())
    print()

    # Check leader_score_avg range
    df2 = pd.read_sql(
        text("""
        SELECT 
            MIN(mainline_score) as min_ms, MAX(mainline_score) as max_ms,
            MIN(leader_score_avg) as min_lsa, MAX(leader_score_avg) as max_lsa, AVG(leader_score_avg) as avg_lsa,
            MIN(member_count) as min_mc, MAX(member_count) as max_mc, AVG(member_count) as avg_mc,
            MIN(leader_count) as min_lc, MAX(leader_count) as max_lc, AVG(leader_count) as avg_lc,
            MIN(up_ratio) as min_ur, MAX(up_ratio) as max_ur, AVG(up_ratio) as avg_ur,
            MIN(avg_ret) as min_ar, MAX(avg_ret) as max_ar, AVG(avg_ret) as avg_ar
        FROM cn_ga_mainline_radar_daily
        WHERE trade_date BETWEEN '2026-01-05' AND '2026-01-10'
        """),
        conn
    )
    print("=== Stats ===")
    print(df2.to_string())
    print()

    # Check stock leader scores for breakout_ratio computation
    df3 = pd.read_sql(
        text("""
        SELECT trade_date, industry_id, 
               COUNT(*) as total,
               SUM(CASE WHEN breakout_ready = 1 OR breakout_strength > 0 THEN 1 ELSE 0 END) as breakout_count,
               SUM(CASE WHEN leader_score >= 1.0 THEN 1 ELSE 0 END) as leader_count,
               AVG(leader_score) as avg_leader_score
        FROM cn_stock_leader_score_daily
        WHERE trade_date = '2026-01-05'
        GROUP BY trade_date, industry_id
        LIMIT 10
        """),
        conn
    )
    print("=== Stock leader scores by industry ===")
    print(df3.to_string())
