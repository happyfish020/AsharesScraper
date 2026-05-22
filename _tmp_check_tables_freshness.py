"""
Check freshness of all V8 tables listed in the runbooks.
Reports MAX(date) and monthly record counts for recent months.
"""
from app.settings import build_engine
from sqlalchemy import text
from datetime import datetime, date

e = build_engine()
c = e.connect()

TODAY = date(2026, 5, 21)

# Tables grouped by type with their date column
tables_config = [
    # === Raw-source tables (from V8_DATASET_RUNBOOK §1) ===
    ("cn_stock_daily_price", "trade_date", "raw"),
    ("cn_index_daily_price", "trade_date", "raw"),
    ("cn_stock_daily_basic", "trade_date", "raw"),
    ("cn_sw_industry_daily", "trade_date", "raw"),
    ("cn_stock_monthly_basic", "month", "raw"),
    ("cn_stock_fina_indicator", "end_date", "raw"),
    ("cn_stock_income", "end_date", "raw"),
    ("cn_stock_balancesheet", "end_date", "raw"),
    ("cn_stock_cashflow", "end_date", "raw"),
    ("cn_board_industry_member_hist", "trade_date", "raw"),
    ("cn_board_member_map_d", "trade_date", "raw"),
    ("cn_local_industry_master", None, "raw"),  # dimension table
    ("cn_local_industry_map_hist", "trade_date", "raw"),
    ("cn_local_industry_proxy_daily", "trade_date", "raw"),
    ("cn_event_disclosure_date", "end_date", "raw"),
    ("cn_event_earnings_forecast", "end_date", "raw"),
    # === Derived tables (from V8_DATASET_RUNBOOK §1) ===
    ("cn_stock_fundamental_daily", "trade_date", "derived"),
    ("cn_stock_quality_score_daily", "trade_date", "derived"),
    ("cn_industry_capital_flow_daily", "trade_date", "derived"),
    ("cn_ga_stock_role_map_daily", "trade_date", "derived"),
    ("cn_stock_mainline_strength_daily", "trade_date", "derived"),
    ("cn_ga_mainline_radar_daily", "trade_date", "derived"),
    ("cn_ga_market_pulse_daily", "trade_date", "derived"),
    ("cn_mainline_lifecycle_daily", "trade_date", "derived"),
    ("cn_unified_alpha_score_daily", "trade_date", "derived"),
    ("cn_stock_leader_sw_l1_latest_snap", "trade_date", "derived"),
    ("cn_v7_v8_industry_crosswalk_latest", "trade_date", "derived"),
    # === Future-system primary tables (from V8_FUTURE_SYSTEM_UPDATE_RUNBOOK) ===
    ("cn_event_stock_weekly", "trade_date", "raw"),
    ("cn_event_stock_monthly", "trade_date", "raw"),
]

def fmt(d):
    if d is None:
        return "N/A"
    if isinstance(d, date):
        return d.isoformat()
    return str(d)

def check_monthly_distribution(tbl, date_col):
    """Check record counts by month for the last 6 months."""
    sql = f"""
    SELECT 
        DATE_FORMAT({date_col}, '%Y-%m') as ym,
        COUNT(*) as cnt,
        MIN({date_col}) as min_d,
        MAX({date_col}) as max_d
    FROM {tbl}
    WHERE {date_col} >= DATE_SUB('{TODAY.isoformat()}', INTERVAL 6 MONTH)
    GROUP BY DATE_FORMAT({date_col}, '%Y-%m')
    ORDER BY ym DESC
    """
    rows = c.execute(text(sql)).fetchall()
    return rows

print("=" * 120)
print(f"V8 Tables Freshness Check — Reference Date: {TODAY.isoformat()}")
print("=" * 120)

results = []

for tbl, date_col, tbl_type in tables_config:
    try:
        # Check existence and total count
        r = c.execute(text(f"SELECT COUNT(*) FROM {tbl}")).fetchone()
        total = r[0]
        
        if date_col is None:
            # Dimension table, no date to check
            print(f"\n[{tbl_type.upper()}] {tbl}")
            print(f"  Total rows: {total}")
            print(f"  (dimension table, no date column)")
            results.append((tbl, tbl_type, "DIM", total, None, None, None))
            continue
        
        # MAX date
        r = c.execute(text(f"SELECT MAX({date_col}) FROM {tbl}")).fetchone()
        max_date = r[0]
        
        # MIN date
        r = c.execute(text(f"SELECT MIN({date_col}) FROM {tbl}")).fetchone()
        min_date = r[0]
        
        # Check if max_date >= today (or today is a trading day)
        is_fresh = False
        if max_date:
            if isinstance(max_date, str):
                md = datetime.strptime(max_date[:10], "%Y-%m-%d").date()
            else:
                md = max_date if isinstance(max_date, date) else max_date
            # For daily tables, expect today or yesterday
            diff = (TODAY - md).days
            is_fresh = diff <= 3  # within 3 days is considered fresh
        
        # Monthly distribution for last 6 months
        monthly = check_monthly_distribution(tbl, date_col)
        
        print(f"\n[{tbl_type.upper()}] {tbl}")
        print(f"  Total rows: {total}")
        print(f"  Date range: {fmt(min_date)} ~ {fmt(max_date)}")
        print(f"  Freshness: {'✓ FRESH' if is_fresh else '✗ STALE'} (diff={diff if max_date else 'N/A'}d)")
        print(f"  Monthly distribution (last 6 months):")
        if monthly:
            for row in monthly:
                ym, cnt, mn, mx = row
                bar = "#" * min(cnt // 1000, 60) if cnt > 0 else ""
                print(f"    {ym}: {cnt:>8,} rows  [{fmt(mn)} ~ {fmt(mx)}]  {bar}")
        else:
            print(f"    (no data in last 6 months)")
        
        results.append((tbl, tbl_type, "OK" if is_fresh else "STALE", total, fmt(min_date), fmt(max_date), diff if max_date else None))
        
    except Exception as ex:
        print(f"\n[{tbl_type.upper()}] {tbl}: ERROR — {ex}")
        results.append((tbl, tbl_type, "ERROR", 0, None, None, None))

c.close()

print("\n" + "=" * 120)
print("SUMMARY")
print("=" * 120)
print(f"{'Table':<45} {'Type':<10} {'Status':<8} {'Total':>12} {'Min Date':<12} {'Max Date':<12} {'Lag(d)':<8}")
print("-" * 120)
for tbl, tbl_type, status, total, min_d, max_d, lag in results:
    lag_str = str(lag) if lag is not None else "N/A"
    print(f"{tbl:<45} {tbl_type:<10} {status:<8} {total:>12,} {str(min_d or 'N/A'):<12} {str(max_d or 'N/A'):<12} {lag_str:<8}")

# Count issues
stale = [r for r in results if r[2] == "STALE"]
errors = [r for r in results if r[2] == "ERROR"]
print("\n" + "=" * 120)
if stale:
    print(f"⚠️  STALE tables ({len(stale)}):")
    for r in stale:
        print(f"   - {r[0]}: max_date={r[5]}, lag={r[6]}d")
if errors:
    print(f"❌ ERROR tables ({len(errors)}):")
    for r in errors:
        print(f"   - {r[0]}: {r[5]}")
if not stale and not errors:
    print("✅ All tables are fresh!")
print("=" * 120)
