from app.settings import build_engine
from sqlalchemy import text
engine = build_engine()
with engine.connect() as conn:
    r = conn.execute(text('SELECT COUNT(*) FROM cn_mainline_strength_daily WHERE trade_date BETWEEN "2026-01-01" AND "2026-03-30"')).scalar()
    print(f'cn_mainline_strength_daily rows (2026-01-01 ~ 2026-03-30): {r}')
    
    r = conn.execute(text('SELECT COUNT(*) FROM cn_ga_mainline_radar_daily WHERE trade_date BETWEEN "2026-01-01" AND "2026-03-30"')).scalar()
    print(f'cn_ga_mainline_radar_daily rows: {r}')
    
    r = conn.execute(text('SELECT COUNT(*) FROM cn_industry_capital_flow_daily WHERE trade_date BETWEEN "2026-01-01" AND "2026-03-30"')).scalar()
    print(f'cn_industry_capital_flow_daily rows: {r}')
    
    r = conn.execute(text('SELECT COUNT(*) FROM cn_stock_leader_score_daily WHERE trade_date BETWEEN "2026-01-01" AND "2026-03-30"')).scalar()
    print(f'cn_stock_leader_score_daily rows: {r}')
    
    r = conn.execute(text('SELECT COUNT(*) FROM cn_ga_stock_role_map_daily WHERE trade_date BETWEEN "2026-01-01" AND "2026-03-30"')).scalar()
    print(f'cn_ga_stock_role_map_daily rows: {r}')
    
    r = conn.execute(text('SELECT COUNT(*) FROM cn_local_industry_map_hist')).scalar()
    print(f'cn_local_industry_map_hist rows: {r}')
    
    r = conn.execute(text('SELECT COUNT(*) FROM cn_stock_daily_price WHERE TRADE_DATE BETWEEN "2026-01-01" AND "2026-03-30"')).scalar()
    print(f'cn_stock_daily_price rows: {r}')
    
    r = conn.execute(text('SELECT COUNT(*) FROM cn_stock_daily_basic WHERE trade_date BETWEEN "2026-01-01" AND "2026-03-30"')).scalar()
    print(f'cn_stock_daily_basic rows: {r}')
    
    # Sample cn_ga_mainline_radar_daily
    r = conn.execute(text('SELECT trade_date, mainline_id, mainline_name, mainline_score, mainline_phase, leader_density, breakout_ratio, new_high_ratio, rotation_rank, trend_alignment_score, capital_concentration_score FROM cn_ga_mainline_radar_daily WHERE trade_date = "2026-01-05" LIMIT 5')).fetchall()
    print('\nSample cn_ga_mainline_radar_daily (2026-01-05):')
    for row in r:
        print(f'  {row}')
    
    # Check if cn_ga_mainline_radar_daily has capital_concentration_score
    r = conn.execute(text("SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA = 'cn_market_red' AND TABLE_NAME = 'cn_ga_mainline_radar_daily' AND COLUMN_NAME = 'capital_concentration_score'")).scalar()
    print(f'\ncn_ga_mainline_radar_daily has capital_concentration_score: {r is not None}')
