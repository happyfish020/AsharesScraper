"""Check radar data column types and non-null values."""
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
    # Check column types
    df = pd.read_sql(
        text("""
        SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = 'cn_market_red' AND TABLE_NAME = 'cn_ga_mainline_radar_daily'
        ORDER BY ORDINAL_POSITION
        """),
        conn
    )
    print("=== Column types ===")
    print(df.to_string())
    print()

    # Check non-null counts for key columns
    df2 = pd.read_sql(
        text("""
        SELECT 
            COUNT(*) as total,
            SUM(CASE WHEN mainline_score IS NOT NULL THEN 1 ELSE 0 END) as mainline_score_nn,
            SUM(CASE WHEN leader_density IS NOT NULL THEN 1 ELSE 0 END) as leader_density_nn,
            SUM(CASE WHEN breakout_ratio IS NOT NULL THEN 1 ELSE 0 END) as breakout_ratio_nn,
            SUM(CASE WHEN new_high_ratio IS NOT NULL THEN 1 ELSE 0 END) as new_high_ratio_nn,
            SUM(CASE WHEN trend_alignment_score IS NOT NULL THEN 1 ELSE 0 END) as trend_alignment_nn,
            SUM(CASE WHEN rotation_rank IS NOT NULL THEN 1 ELSE 0 END) as rotation_rank_nn,
            SUM(CASE WHEN leader_count IS NOT NULL THEN 1 ELSE 0 END) as leader_count_nn,
            SUM(CASE WHEN strong_stock_count IS NOT NULL THEN 1 ELSE 0 END) as strong_stock_nn
        FROM cn_ga_mainline_radar_daily
        WHERE trade_date BETWEEN '2026-01-05' AND '2026-01-10'
        """),
        conn
    )
    print("=== Non-null counts ===")
    print(df2.to_string())
    print()

    # Sample with non-null values
    df3 = pd.read_sql(
        text("""
        SELECT trade_date, mainline_id, mainline_name, mainline_score, mainline_phase,
               leader_density, breakout_ratio, new_high_ratio, trend_alignment_score,
               rotation_rank, leader_count, strong_stock_count
        FROM cn_ga_mainline_radar_daily
        WHERE trade_date = '2026-01-05'
          AND leader_density IS NOT NULL
        LIMIT 5
        """),
        conn
    )
    print("=== Sample with non-null leader_density ===")
    print(df3.to_string())
    print()

    # Check if there's any data with non-null values at all
    df4 = pd.read_sql(
        text("""
        SELECT trade_date, COUNT(*) as total,
               SUM(CASE WHEN leader_density IS NOT NULL THEN 1 ELSE 0 END) as ld_nn
        FROM cn_ga_mainline_radar_daily
        WHERE trade_date BETWEEN '2026-01-05' AND '2026-03-30'
        GROUP BY trade_date
        HAVING ld_nn > 0
        ORDER BY trade_date
        LIMIT 10
        """),
        conn
    )
    print("=== Dates with non-null leader_density ===")
    print(df4.to_string())
