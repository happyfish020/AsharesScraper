"""Check cn_mainline_strength_daily and upstream tables."""
from app.settings import build_engine
from sqlalchemy import text
import sys

engine = build_engine()
with engine.connect() as conn:
    # Check cn_mainline_strength_daily
    for tbl in ['cn_mainline_strength_daily', 'mainline_strength_daily', 'ga_mainline_strength_daily']:
        try:
            r = conn.execute(text(f'SELECT COUNT(*) FROM {tbl}')).fetchone()
            print(f'{tbl}: {r[0]} rows')
            r2 = conn.execute(text(f'SELECT MIN(trade_date), MAX(trade_date) FROM {tbl}')).fetchone()
            print(f'  date range: {r2[0]} ~ {r2[1]}')
        except Exception as e:
            print(f'{tbl}: NOT FOUND - {e}')
    
    print()
    print('--- Upstream tables for mainline_strength ---')
    for tbl in ['cn_mainline_radar_daily', 'cn_ga_mainline_radar_daily', 'mainline_radar_daily',
                'cn_industry_capital_flow_daily', 'industry_capital_flow_daily',
                'cn_mainline_lifecycle_daily']:
        try:
            r = conn.execute(text(f'SELECT COUNT(*) FROM {tbl}')).fetchone()
            print(f'{tbl}: {r[0]} rows')
        except Exception as e:
            print(f'{tbl}: NOT FOUND - {e}')
    
    print()
    print('--- Mainline lifecycle lifecycle_state distribution ---')
    try:
        r = conn.execute(text('SELECT lifecycle_state, COUNT(*) FROM cn_mainline_lifecycle_daily GROUP BY lifecycle_state ORDER BY COUNT(*) DESC')).fetchall()
        for row in r:
            print(f'  {row[0]}: {row[1]}')
    except Exception as e:
        print(f'  Error: {e}')
    
    print()
    print('--- Does cn_mainline_lifecycle_daily have mainline_strength column? ---')
    try:
        r = conn.execute(text("SHOW COLUMNS FROM cn_mainline_lifecycle_daily LIKE 'mainline_strength'")).fetchone()
        if r:
            print(f'  YES - column exists: {r}')
        else:
            print('  NO - mainline_strength column not found')
    except Exception as e:
        print(f'  Error: {e}')
    
    # Check if mainline_strength values are non-null
    try:
        r = conn.execute(text('SELECT COUNT(*) FROM cn_mainline_lifecycle_daily WHERE mainline_strength IS NOT NULL')).fetchone()
        print(f'  rows with non-null mainline_strength: {r[0]}')
    except Exception as e:
        print(f'  Error: {e}')

print('DONE')
