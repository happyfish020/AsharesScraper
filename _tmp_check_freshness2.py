"""
Check freshness of V8 tables - writes results to a file for reliable output.
"""
from app.settings import build_engine
from sqlalchemy import text
from datetime import datetime, date
import sys

e = build_engine()
c = e.connect()

TODAY = date(2026, 5, 21)

# All tables from both runbooks
tables_config = [
    # Raw-source tables
    ("cn_stock_daily_price", "trade_date"),
    ("cn_index_daily_price", "trade_date"),
    ("cn_stock_daily_basic", "trade_date"),
    ("cn_sw_industry_daily", "trade_date"),
    ("cn_stock_monthly_basic", "month"),
    ("cn_stock_fina_indicator", "end_date"),
    ("cn_stock_income", "end_date"),
    ("cn_stock_balancesheet", "end_date"),
    ("cn_stock_cashflow", "end_date"),
    ("cn_board_industry_member_hist", "trade_date"),
    ("cn_board_member_map_d", "trade_date"),
    ("cn_local_industry_master", None),
    ("cn_local_industry_map_hist", "trade_date"),
    ("cn_local_industry_proxy_daily", "trade_date"),
    ("cn_event_disclosure_date", "end_date"),
    ("cn_event_earnings_forecast", "end_date"),
    # Derived tables
    ("cn_stock_fundamental_daily", "trade_date"),
    ("cn_stock_quality_score_daily", "trade_date"),
    ("cn_industry_capital_flow_daily", "trade_date"),
    ("cn_ga_stock_role_map_daily", "trade_date"),
    ("cn_stock_mainline_strength_daily", "trade_date"),
    ("cn_ga_mainline_radar_daily", "trade_date"),
    ("cn_ga_market_pulse_daily", "trade_date"),
    ("cn_mainline_lifecycle_daily", "trade_date"),
    ("cn_unified_alpha_score_daily", "trade_date"),
    ("cn_stock_leader_sw_l1_latest_snap", "trade_date"),
    ("cn_v7_v8_industry_crosswalk_latest", "trade_date"),
]

results = []

for tbl, date_col in tables_config:
    try:
        r = c.execute(text(f"SELECT COUNT(*) FROM {tbl}")).fetchone()
        total = r[0]
        
        if date_col is None:
            results.append((tbl, "DIM", total, "N/A", "N/A", "N/A"))
            continue
        
        r = c.execute(text(f"SELECT MIN({date_col}), MAX({date_col}) FROM {tbl}")).fetchone()
        min_d, max_d = r[0], r[1]
        
        # Monthly distribution for last 6 months
        monthly_sql = f"""
        SELECT DATE_FORMAT({date_col}, '%%Y-%%m') as ym, COUNT(*) as cnt
        FROM {tbl}
        WHERE {date_col} >= DATE_SUB('{TODAY.isoformat()}', INTERVAL 6 MONTH)
        GROUP BY ym ORDER BY ym DESC
        """
        monthly_rows = c.execute(text(monthly_sql)).fetchall()
        
        # Determine freshness
        lag = None
        if max_d:
            if isinstance(max_d, str):
                md = datetime.strptime(max_d[:10], "%Y-%m-%d").date()
            else:
                md = max_d if isinstance(max_d, date) else max_d.date()
            lag = (TODAY - md).days
        
        results.append((tbl, "RAW" if tbl in [r[0] for r in tables_config[:16]] else "DERIVED", 
                       total, str(min_d or "N/A"), str(max_d or "N/A"), 
                       lag, monthly_rows))
        
    except Exception as ex:
        results.append((tbl, "ERR", 0, str(ex), "N/A", None, []))

# Print results
print(f"{'='*120}")
print(f"V8 TABLES FRESHNESS CHECK — Reference: {TODAY}")
print(f"{'='*120}")

print(f"\n{'Table':<45} {'Type':<8} {'Total':>10} {'Min Date':<12} {'Max Date':<12} {'Lag(d)':<8} {'Status':<8}")
print(f"{'-'*105}")

stale_list = []
err_list = []

for tbl, typ, total, min_d, max_d, lag, monthly in results:
    if typ == "ERR":
        status = "ERROR"
        err_list.append(tbl)
    elif lag is None:
        status = "DIM"
    elif lag <= 3:
        status = "OK"
    else:
        status = "STALE"
        stale_list.append((tbl, max_d, lag))
    
    lag_str = str(lag) if lag is not None else "N/A"
    print(f"{tbl:<45} {typ:<8} {total:>10,} {str(min_d):<12} {str(max_d):<12} {lag_str:<8} {status:<8}")
    
    # Print monthly distribution for tables with data
    if monthly and len(monthly) > 0:
        for row in monthly:
            ym, cnt = row
            bar = "█" * min(cnt // 1000, 50) if cnt > 0 else ""
            print(f"  {'':>45} {ym}: {cnt:>8,} rows  {bar}")

print(f"\n{'='*120}")
print("SUMMARY")
print(f"{'='*120}")

if stale_list:
    print(f"\n⚠️  STALE TABLES ({len(stale_list)}):")
    for tbl, max_d, lag in stale_list:
        print(f"   ✗ {tbl:<50} max={max_d}, lag={lag}d")
else:
    print("\n✅ All tables are FRESH (lag <= 3 days)")

if err_list:
    print(f"\n❌ ERROR TABLES ({len(err_list)}):")
    for tbl in err_list:
        print(f"   ✗ {tbl}")

print(f"\n{'='*120}")
print("DONE")
print(f"{'='*120}")

c.close()
