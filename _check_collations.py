"""Check table collations."""
import sys
sys.path.insert(0, ".")
from app.settings import build_engine
from sqlalchemy import text

engine = build_engine()
with engine.connect() as conn:
    tables = [
        "cn_local_industry_map_hist",
        "cn_stock_daily_price",
        "cn_stock_daily_basic",
        "cn_stock_leader_score_daily",
        "cn_local_industry_proxy_daily",
    ]
    for t in tables:
        try:
            row = conn.execute(
                text(
                    f"SELECT TABLE_NAME, TABLE_COLLATION "
                    f"FROM INFORMATION_SCHEMA.TABLES "
                    f"WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = '{t}'"
                )
            ).one()
            print(f"{t:40s} collation={row[1]}")
        except Exception as e:
            print(f"{t:40s} error={e}")
