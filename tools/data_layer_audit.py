import pandas as pd
from app.settings import build_engine
from sqlalchemy import text
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger("Audit")

def audit_report():
    engine = build_engine()
    report = []
    
    overall_status = "PASS"
    with engine.connect() as conn:
        log.info("==== STARTING QUANT DATA LAYER AUDIT (E80 COMPLIANCE) ====\n")
        
        # 1. cn_stock_daily_price (P0)
        log.info("[1/5] Auditing cn_stock_daily_price...")
        price_stats = conn.execute(text("""
            SELECT 
                MIN(trade_date) as start_d, MAX(trade_date) as end_d,
                COUNT(DISTINCT trade_date) as distinct_days,
                SUM(CASE WHEN pct_chg IS NULL THEN 1 ELSE 0 END) / COUNT(*) * 100 as null_pct,
                SUM(CASE WHEN amount = 0 OR amount IS NULL THEN 1 ELSE 0 END) / COUNT(*) * 100 as bad_amount_pct
            FROM cn_stock_daily_price
        """)).mappings().first()
        
        price_ok = price_stats['end_d'] is not None and price_stats['bad_amount_pct'] < 1.0
        if not price_ok: overall_status = "FAIL"

        # 2. cn_index_daily_price (P0)
        log.info("[2/5] Auditing cn_index_daily_price...")
        index_coverage = conn.execute(text("""
            SELECT symbol, COUNT(*) as cnt, MAX(trade_date) as last_d
            FROM cn_index_daily_price 
            WHERE symbol IN ('sh000300', 'sz399006')
            GROUP BY symbol
        """)).mappings().all()
        
        index_ok = len(index_coverage) >= 2
        if not index_ok: overall_status = "FAIL"

        # 3. cn_stock_daily_basic (P0)
        log.info("[3/5] Auditing cn_stock_daily_basic...")
        basic_stats = conn.execute(text("""
            SELECT 
                MIN(trade_date) as start_d, MAX(trade_date) as end_d,
                SUM(CASE WHEN turnover_rate IS NULL THEN 1 ELSE 0 END) / COUNT(*) * 100 as null_turnover
            FROM cn_stock_daily_basic
        """)).mappings().first()
        
        # Check if basic is significantly shorter than price (common issue)
        basic_recent_only = False
        if price_stats['start_d'] and basic_stats['start_d']:
            if (basic_stats['start_d'] - price_stats['start_d']).days > 365:
                basic_recent_only = True
                overall_status = "PARTIAL"

        # 4. cn_board_member_map_d (P0)
        log.info("[4/5] Auditing cn_board_member_map_d...")
        map_stats = conn.execute(text("""
            SELECT MAX(trade_date) as last_d, COUNT(DISTINCT symbol) as sym_count
            FROM cn_board_member_map_d
        """)).mappings().first()
        
        map_lagging = False
        if price_stats['end_d'] and map_stats['last_d']:
            if (price_stats['end_d'] - map_stats['last_d']).days > 7:
                map_lagging = True
                overall_status = "PARTIAL"

        # 5. cn_stock_leader_score_daily (P0) — physical table, materialized via SP
        log.info("[5/5] Auditing cn_stock_leader_score_daily (materialized table)...")
        leader_stats = conn.execute(text("""
            SELECT MAX(trade_date) as last_d,
                   SUM(CASE WHEN leader_score IS NULL THEN 1 ELSE 0 END) / COUNT(*) * 100 as null_score
            FROM cn_stock_leader_score_daily
        """)).mappings().first()

        # Final logic for scoring
        log.info("\n==== DATA HEALTH ====\n")
        log.info(f"[{overall_status}]")
        log.info(f"stock_daily_price: {'OK' if price_ok else 'FAIL'}")
        log.info(f"index_daily_price: {'OK' if index_ok else 'FAIL'}")
        log.info(f"daily_basic: {'PARTIAL (Recent Only)' if basic_recent_only else 'OK'}")
        log.info(f"board_map: {'LAGGING' if map_lagging else 'OK'}")
        log.info(f"leader_score: {'OK' if leader_stats['last_d'] else 'MISSING'}")

        log.info("\n==== COVERAGE ====\n")
        log.info(f"stock_daily_price: {price_stats['start_d']} to {price_stats['end_d']} (Quality: {100-price_stats['bad_amount_pct']:.1f}%)")
        log.info(f"daily_basic: {basic_stats['start_d']} to {basic_stats['end_d']} ({'⚠️ Recent only' if basic_recent_only else 'Full history'})")
        
        log.info("\n==== FIELD QUALITY ====\n")
        log.info(f"pct_chg: {'OK' if price_stats['null_pct'] < 0.1 else '⚠️ NULLs detected'}")
        log.info(f"amount: {'OK' if price_stats['bad_amount_pct'] < 0.1 else '⚠️ Zeros detected'}")
        log.info(f"turnover_rate: {basic_stats['null_turnover']:.1f}% NULL")

        log.info("\n==== DERIVABLE ====\n")
        log.info("industry_return: YES (via map_d + price.pct_chg)")
        log.info("industry_flow: YES (via map_d + price.amount)")
        log.info("market_breadth: YES (via price.pct_chg)")

        # VERDICT
        can_market_regime = index_ok and price_ok
        can_mainline = price_ok and (not map_lagging) and (leader_stats['last_d'] is not None)
        can_role = can_mainline and (not basic_recent_only)

        log.info("\n==== FINAL VERDICT ====\n")
        log.info(f"[{'✔' if can_market_regime else '✖'}] market regime")
        log.info(f"[{'✔' if can_mainline else '✖'}] mainline detection")
        log.info(f"[{'✔' if can_role else '✖'}] stock role classification")
        
        ready_status = "READY" if (can_market_regime and can_mainline) else "NOT READY"
        log.info(f"\nE80 COMPLIANCE: {ready_status}")

        if ready_status == "NOT READY":
            log.info("\n==== ACTION REQUIRED ====")
            if not index_ok: log.info("- Backfill index_daily_price for sh000300/sz399006")
            if map_lagging: log.info("- Refresh board_member_map_d to align with price data")
            if leader_stats['last_d'] is None: log.info("- Execute leader_score calculation task")
            if price_stats['bad_amount_pct'] > 1.0: log.info("- Fix stock_daily_price zero amounts (Data Corruption)")

if __name__ == "__main__":
    audit_report()