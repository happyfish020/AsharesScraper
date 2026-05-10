"""P2/P3 Acceptance Check - Query database and output results."""
import sys
import os
from datetime import datetime

# Ensure we can import from project
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.settings import build_engine
from sqlalchemy import text

engine = build_engine()
results = {}

with engine.connect() as conn:
    # ── P2: Mainline Lifecycle ──
    for tbl in ['cn_mainline_lifecycle_daily', 'mainline_lifecycle_daily', 'ga_mainline_lifecycle_daily']:
        try:
            r = conn.execute(text(f"SELECT COUNT(*) FROM {tbl}")).fetchone()
            results['P2_table'] = tbl
            results['P2_rows'] = r[0]
            
            # Get date range
            r2 = conn.execute(text(f"SELECT MIN(trade_date), MAX(trade_date) FROM {tbl}")).fetchone()
            results['P2_min_date'] = str(r2[0]) if r2[0] else None
            results['P2_max_date'] = str(r2[1]) if r2[1] else None
            
            # Check lifecycle_state distribution
            r3 = conn.execute(text(f"SELECT lifecycle_state, COUNT(*) as cnt FROM {tbl} GROUP BY lifecycle_state ORDER BY cnt DESC")).fetchall()
            results['P2_state_dist'] = {row[0]: row[1] for row in r3}
            
            # Check lifecycle_score range
            r4 = conn.execute(text(f"SELECT MIN(lifecycle_score), MAX(lifecycle_score), AVG(lifecycle_score) FROM {tbl}")).fetchone()
            results['P2_score_min'] = float(r4[0]) if r4[0] is not None else None
            results['P2_score_max'] = float(r4[1]) if r4[1] is not None else None
            results['P2_score_avg'] = float(r4[2]) if r4[2] is not None else None
            
            # Check columns
            r5 = conn.execute(text(f"SHOW COLUMNS FROM {tbl}")).fetchall()
            results['P2_columns'] = [row[0] for row in r5]
            break
        except Exception as e:
            continue
    
    if 'P2_table' not in results:
        results['P2_table'] = None
        results['P2_error'] = 'No P2 table found'
    
    # ── P3: Quality Score ──
    for tbl in ['cn_stock_quality_score_daily', 'stock_quality_score_daily', 'ga_stock_quality_score_daily']:
        try:
            r = conn.execute(text(f"SELECT COUNT(*) FROM {tbl}")).fetchone()
            results['P3Q_table'] = tbl
            results['P3Q_rows'] = r[0]
            
            r2 = conn.execute(text(f"SELECT MIN(trade_date), MAX(trade_date) FROM {tbl}")).fetchone()
            results['P3Q_min_date'] = str(r2[0]) if r2[0] else None
            results['P3Q_max_date'] = str(r2[1]) if r2[1] else None
            
            r3 = conn.execute(text(f"SELECT MIN(quality_score), MAX(quality_score), AVG(quality_score) FROM {tbl}")).fetchone()
            results['P3Q_score_min'] = float(r3[0]) if r3[0] is not None else None
            results['P3Q_score_max'] = float(r3[1]) if r3[1] is not None else None
            results['P3Q_score_avg'] = float(r3[2]) if r3[2] is not None else None
            
            r4 = conn.execute(text(f"SHOW COLUMNS FROM {tbl}")).fetchall()
            results['P3Q_columns'] = [row[0] for row in r4]
            break
        except Exception as e:
            continue
    
    if 'P3Q_table' not in results:
        results['P3Q_table'] = None
        results['P3Q_error'] = 'No P3 Quality table found'
    
    # ── P3: Unified Alpha ──
    for tbl in ['cn_unified_alpha_score_daily', 'unified_alpha_score_daily', 'ga_unified_alpha_score_daily']:
        try:
            r = conn.execute(text(f"SELECT COUNT(*) FROM {tbl}")).fetchone()
            results['P3U_table'] = tbl
            results['P3U_rows'] = r[0]
            
            r2 = conn.execute(text(f"SELECT MIN(trade_date), MAX(trade_date) FROM {tbl}")).fetchone()
            results['P3U_min_date'] = str(r2[0]) if r2[0] else None
            results['P3U_max_date'] = str(r2[1]) if r2[1] else None
            
            # Symbol count
            r3 = conn.execute(text(f"SELECT COUNT(DISTINCT symbol) FROM {tbl}")).fetchone()
            results['P3U_symbol_count'] = r3[0]
            
            # final_score range
            r4 = conn.execute(text(f"SELECT MIN(final_score), MAX(final_score), AVG(final_score) FROM {tbl}")).fetchone()
            results['P3U_score_min'] = float(r4[0]) if r4[0] is not None else None
            results['P3U_score_max'] = float(r4[1]) if r4[1] is not None else None
            results['P3U_score_avg'] = float(r4[2]) if r4[2] is not None else None
            
            # alpha_bucket distribution
            r5 = conn.execute(text(f"SELECT alpha_bucket, COUNT(*) as cnt FROM {tbl} GROUP BY alpha_bucket ORDER BY cnt DESC")).fetchall()
            results['P3U_bucket_dist'] = {row[0]: row[1] for row in r5}
            
            # Check columns
            r6 = conn.execute(text(f"SHOW COLUMNS FROM {tbl}")).fetchall()
            results['P3U_columns'] = [row[0] for row in r6]
            
            # Top 20 by final_score
            r7 = conn.execute(text(f"SELECT trade_date, symbol, final_score, alpha_bucket FROM {tbl} ORDER BY final_score DESC LIMIT 20")).fetchall()
            results['P3U_top20'] = [(str(row[0]), row[1], float(row[2]), row[3]) for row in r7]
            
            # Check explanation null rate
            if 'explanation' in results['P3U_columns']:
                r8 = conn.execute(text(f"SELECT COUNT(*) FROM {tbl} WHERE explanation IS NULL OR explanation = ''")).fetchone()
                results['P3U_explanation_null'] = r8[0]
            
            break
        except Exception as e:
            continue
    
    if 'P3U_table' not in results:
        results['P3U_table'] = None
        results['P3U_error'] = 'No P3 Unified Alpha table found'

# Print results
print("=" * 70)
print("P2/P3 ACCEPTANCE CHECK RESULTS")
print(f"Generated: {datetime.now().isoformat()}")
print("=" * 70)

# P2
print("\n--- P2: Mainline Lifecycle ---")
print(f"  Table: {results.get('P2_table', 'NOT FOUND')}")
if results.get('P2_table'):
    print(f"  Rows: {results.get('P2_rows')}")
    print(f"  Date Range: {results.get('P2_min_date')} ~ {results.get('P2_max_date')}")
    print(f"  Score Range: {results.get('P2_score_min')} ~ {results.get('P2_score_max')} (avg: {results.get('P2_score_avg')})")
    print(f"  State Distribution: {results.get('P2_state_dist')}")
    print(f"  Columns: {results.get('P2_columns')}")
else:
    print(f"  Error: {results.get('P2_error')}")

# P3 Quality
print("\n--- P3: Quality Score ---")
print(f"  Table: {results.get('P3Q_table', 'NOT FOUND')}")
if results.get('P3Q_table'):
    print(f"  Rows: {results.get('P3Q_rows')}")
    print(f"  Date Range: {results.get('P3Q_min_date')} ~ {results.get('P3Q_max_date')}")
    print(f"  Score Range: {results.get('P3Q_score_min')} ~ {results.get('P3Q_score_max')} (avg: {results.get('P3Q_score_avg')})")
    print(f"  Columns: {results.get('P3Q_columns')}")
else:
    print(f"  Error: {results.get('P3Q_error')}")

# P3 Unified Alpha
print("\n--- P3: Unified Alpha ---")
print(f"  Table: {results.get('P3U_table', 'NOT FOUND')}")
if results.get('P3U_table'):
    print(f"  Rows: {results.get('P3U_rows')}")
    print(f"  Date Range: {results.get('P3U_min_date')} ~ {results.get('P3U_max_date')}")
    print(f"  Symbol Count: {results.get('P3U_symbol_count')}")
    print(f"  Score Range: {results.get('P3U_score_min')} ~ {results.get('P3U_score_max')} (avg: {results.get('P3U_score_avg')})")
    print(f"  Bucket Distribution: {results.get('P3U_bucket_dist')}")
    print(f"  Columns: {results.get('P3U_columns')}")
    if 'P3U_explanation_null' in results:
        print(f"  Explanation Null Count: {results.get('P3U_explanation_null')}")
    print("\n  Top 20 by final_score:")
    for i, row in enumerate(results.get('P3U_top20', []), 1):
        print(f"    {i:2d}. {row[0]} | {row[1]} | score={row[2]:.4f} | bucket={row[3]}")
else:
    print(f"  Error: {results.get('P3U_error')}")

print("\n" + "=" * 70)
