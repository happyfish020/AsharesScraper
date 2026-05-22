from __future__ import annotations

import argparse
import calendar
import re
from datetime import datetime, timedelta
from typing import List, Optional
from sqlalchemy import text

from app.tasks.rotation_sector_snapshot_task import SectorRotationSnapshotTask
from app.utils.logger import setup_logging, get_logger
from app.utils.wireguard_helper import activate_tunnel, deactivate_tunnel, toggle_vpn

from app.context import RunnerConfig
from app.base import RunContext
from app.runner_app import RunnerApp
from app.defaults import DEFAULT_INDEX_SYMBOLS, DEFAULT_BASE_INDEX
from app.settings import build_engine
from app.trading_day import get_latest_trade_date

from app.tasks.db_init_task import DbInitTask
from app.tasks.stock_loader_task import StockLoaderTask
from app.tasks.etf_loader_task import EtfLoaderTask
from app.tasks.index_loader_task import IndexLoaderTask
from app.tasks.futures_loader_task import FuturesLoaderTask
from app.tasks.options_loader_task import OptionsLoaderTask
from app.tasks.coverage_audit_task import CoverageAuditTask
from app.tasks.his_stocks_loader_task import HisStocksLoaderTask
from app.tasks.board_membership_refresh_task import BoardMembershipRefreshTask
from app.tasks.stock_basic_weekly_task import StockBasicWeeklyTask
from app.tasks.stock_fundamental_monthly_task import StockFundamentalMonthlyTask
from app.tasks.stock_quality_snapshot_task import StockQualitySnapshotTask
from app.tasks.stock_working_capital_alert_task import StockWorkingCapitalAlertTask
from app.tasks.inst_fund_hold_summary_task import InstFundHoldSummaryTask
from app.tasks.event_loader_task import EventLoaderTask
from app.tasks.sw_industry_daily_task import SwIndustryDailyTask
from app.tasks.v8_dataset_ops_task import (
    V8BoardRefreshTask,
    V8DailyAuditTask,
    V8DailyDerivedAlphaTask,
    V8DailyDerivedFoundationTask,
    V8DailyDerivedMainlineTask,
    V8DailyOpsTask,
    V8EventDailyTask,
    V8EventPeriodicTask,
    V8HistoricalBackfillTask,
    V8IndexTask,
    V8DailyMarketRawTask,
    V8DailyReferenceRefreshTask,
    V8RotationAuditTask,
    V8RotationRepairTask,
    V8RotationTask,
    V8StockBasicTask,
    V8StockFundamentalRefreshTask,
    V8StockTask,
    V8MonthlyOpsTask,
    V8MonthlyAuditTask,
    V8MonthlyDerivedTask,
    V8MonthlyRefreshTask,
    V8SwIndustryDailyTask,
    V8WeeklyAuditTask,
    V8WeeklyAuditIndexTask,
    V8WeeklyAuditMarketTask,
    V8WeeklyAuditStockTask,
    V8WeeklyFinalizeTask,
    V8WeeklyRefreshTask,
    V8WeeklyOpsTask,
)


def _parse_args(argv: Optional[List[str]] = None):
    p = argparse.ArgumentParser(description="AsharesScraper Runner")
    p.add_argument("--asof", default="latest", help="run as-of trading day: latest or YYYYMMDD")
    p.add_argument("--days", type=int, default=1, help="lookback window days (used when --asof=latest and you want backfill)")
    p.add_argument("--start-date", "--start_date", dest="start_date", default=None, help="explicit inclusive start date: YYYYMMDD or YYYY-MM-DD. Overrides --days when supplied.")
    p.add_argument("--end-date", "--end_date", dest="end_date", default=None, help="explicit inclusive end date: latest, YYYYMMDD or YYYY-MM-DD. Defaults to --asof when omitted.")
    p.add_argument("--flag", default="tu", choices=["tu", "ak"], help="data source flag: tu=tushare (default), ak=akshare")
    p.add_argument("--tasks", default="stock", help="comma-separated: stock,his_stocks,board,stock_basic,sw_industry,stock_fundamental,stock_quality_snapshot,stock_working_capital,inst_fund_hold,rotation,etf,index,futures,options,event,event_daily,event_periodic,audit,v8_stock,v8_index,v8_board_refresh,v8_stock_basic,v8_sw_industry_daily,v8_rotation,v8_rotation_audit,v8_rotation_repair,v8_event_daily,v8_event_periodic,v8_stock_fundamental_refresh,v8_daily,v8_daily_market_raw,v8_daily_reference,v8_daily_audit,v8_daily_derived_foundation,v8_daily_derived_mainline,v8_daily_derived_alpha,v8_weekly,v8_weekly_refresh,v8_weekly_audit,v8_weekly_audit_stock,v8_weekly_audit_index,v8_weekly_audit_market,v8_weekly_finalize,v8_monthly,v8_monthly_refresh,v8_monthly_audit,v8_monthly_derived,v8_backfill,all")
    p.add_argument("--history-start", default="20000101", help="his_stocks only: YYYYMMDD or YYYY-MM")
    p.add_argument("--history-end", default="latest", help="his_stocks only: latest or YYYYMMDD or YYYY-MM")
    p.add_argument("--history-source-order", default="ak,yf", help="his_stocks only: comma-separated source priority, default ak,yf")
    p.add_argument("--history-max-symbols", type=int, default=0, help="his_stocks only: limit symbols for smoke test; 0 means all")
    p.add_argument("--history-symbols", default="", help="his_stocks only: comma-separated symbols, e.g. 000001,600000; when set, skip universe scan")
    p.add_argument("--history-ignore-state", action="store_true", help="his_stocks only: ignore scanned/failed state and force re-check")
    p.add_argument("--history-alternate-bs-ak", action="store_true", help="his_stocks only: deprecated compatibility flag; no effect")
    p.add_argument("--history-universe-frequency", default="monthly", choices=["monthly", "weekly"], help="his_stocks only: anchor frequency for query_all_stock(date)")
    p.add_argument("--refresh", action="store_true", help="clear state files and refresh")
    p.add_argument("--no-vpn", action="store_true", help="skip wireguard start/stop")
    return p.parse_args(argv)


def _normalize_runner_date(raw: str, latest_asof: str, label: str) -> str:
    s = str(raw or "").strip()
    if not s:
        raise SystemExit(f"{label} is empty")
    if s.lower() == "latest":
        return latest_asof
    if re.fullmatch(r"\d{8}", s):
        datetime.strptime(s, "%Y%m%d")
        return s
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        return datetime.strptime(s, "%Y-%m-%d").strftime("%Y%m%d")
    raise SystemExit(f"Invalid {label} format: {raw}. Expected latest, YYYYMMDD, or YYYY-MM-DD.")


def _normalize_history_date(raw: str, asof: str, is_end: bool) -> str:
    s = str(raw or "").strip()
    low = s.lower()
    if low == "latest":
        return asof

    if re.fullmatch(r"\d{8}", s):
        datetime.strptime(s, "%Y%m%d")
        return s
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        dt = datetime.strptime(s, "%Y-%m-%d")
        return dt.strftime("%Y%m%d")

    ym = None
    if re.fullmatch(r"\d{6}", s):
        ym = (int(s[:4]), int(s[4:6]))
    elif re.fullmatch(r"\d{4}-\d{2}", s):
        ym = tuple(map(int, s.split("-")))

    if ym is None:
        raise SystemExit(f"Invalid history date format: {raw}. Expected latest, YYYYMMDD, or YYYY-MM.")

    y, m = ym
    if m < 1 or m > 12:
        raise SystemExit(f"Invalid month in history date: {raw}")
    day = calendar.monthrange(y, m)[1] if is_end else 1
    return f"{y:04d}{m:02d}{day:02d}"


def run(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)

    # Log file naming rule:
    # - if tasks=all/* or empty -> tag=all
    # - if multiple tasks -> tag=first task name
    # - otherwise -> tag=single task
    import sys
    raw_tasks = (args.tasks or "").strip().lower()
    # Detect whether user explicitly provided --tasks
    argv_list = sys.argv[1:] if argv is None else list(argv)
    has_tasks_flag = "--tasks" in argv_list
    if not raw_tasks or raw_tasks in ("all", "*"):
        task_tag = "all"
    elif not has_tasks_flag:
        # When task selection is default (no --tasks flag), tag as all per requirement.
        task_tag = "all"
    else:
        parts = [t.strip() for t in raw_tasks.split(",") if t.strip()]
        task_tag = (parts[0] if len(parts) > 1 else (parts[0] if parts else "all"))
    setup_logging(market="cn", mode=f"scraper_{task_tag}")
    log = get_logger("Runner")

    if not args.no_vpn:
        log.info("wireguard: start")
        activate_tunnel("cn")
    else:
       log.info("wireguard: stop")
       deactivate_tunnel("cn")


 

    raw = (args.tasks or "stock").strip().lower()
    selected = []
    if raw in ("all", "*"):
        selected = ["stock", "board", "stock_basic", "sw_industry", "stock_fundamental", "inst_fund_hold", "rotation", "etf", "index", "futures", "options", "event_daily", "event_periodic", "audit"]
    else:
        selected = [t.strip() for t in raw.split(",") if t.strip()]

    # resolve asof date
    if isinstance(args.asof, str) and args.asof.lower() == "latest":
        asof = get_latest_trade_date()
    else:
        asof = str(args.asof).strip()

    # RunnerConfig: date-window semantics
    # - --start-date/--end-date: explicit inclusive range, preferred for manual backfill
    # - otherwise --asof latest --days N: auto recent backfill [asof-(N-1), asof]
    # - otherwise --asof YYYYMMDD --days N: explicit asof-centered recent window
    cfg = RunnerConfig(
        look_back_days=args.days,
        manual_stock_symbols=[],
        index_symbols=DEFAULT_INDEX_SYMBOLS,
        base_index=DEFAULT_BASE_INDEX,
        refresh_state=args.refresh,
    )
    cfg.data_source_flag = str(args.flag or "tu").strip().lower()

    if args.start_date or args.end_date:
        end_raw = args.end_date or asof
        start_raw = args.start_date or end_raw
        cfg.end_date = _normalize_runner_date(end_raw, asof, "--end-date")
        cfg.start_date = _normalize_runner_date(start_raw, asof, "--start-date")
        if cfg.start_date > cfg.end_date:
            raise SystemExit(f"date window invalid: {cfg.start_date} > {cfg.end_date}")
    else:
        cfg.end_date = asof
        if args.days and args.days > 1:
            asof_dt = datetime.strptime(asof, "%Y%m%d")
            cfg.start_date = (asof_dt - timedelta(days=args.days - 1)).strftime("%Y%m%d")
        else:
            cfg.start_date = asof

    if "his_stocks" in selected:
        history_start = _normalize_history_date(args.history_start, asof, is_end=False)
        history_end = _normalize_history_date(args.history_end, asof, is_end=True)
        if history_start > history_end:
            raise SystemExit(f"history window invalid: {history_start} > {history_end}")

        cfg.his_start_date = history_start
        cfg.his_end_date = history_end
        cfg.his_source_order = str(args.history_source_order).strip().lower()
        cfg.his_max_symbols = max(0, int(args.history_max_symbols))
        raw_symbols = str(args.history_symbols).replace(",", " ").split()
        cfg.his_symbols = [s.strip() for s in raw_symbols if s.strip()]
        cfg.his_ignore_state = bool(args.history_ignore_state)
        cfg.his_alternate_bs_ak = bool(args.history_alternate_bs_ak)
        cfg.his_universe_frequency = str(args.history_universe_frequency).strip().lower()
        cfg.start_date = history_start
        cfg.end_date = history_end

    tasks_map = {
        "stock": StockLoaderTask,
        "his_stocks": HisStocksLoaderTask,
        "board": BoardMembershipRefreshTask,
        "stock_basic": StockBasicWeeklyTask,
        "sw_industry": SwIndustryDailyTask,
        "stock_fundamental": StockFundamentalMonthlyTask,
        "stock_quality_snapshot": StockQualitySnapshotTask,
        "stock_working_capital": StockWorkingCapitalAlertTask,
        "inst_fund_hold": InstFundHoldSummaryTask,
        "event": EventLoaderTask,
        "event_daily": lambda: EventLoaderTask(name="EventLoaderDaily", frequency_tag="daily"),
        "event_periodic": lambda: EventLoaderTask(name="EventLoaderPeriodic", frequency_tag="periodic"),
        "etf": EtfLoaderTask,
        "index": IndexLoaderTask,
        "futures": FuturesLoaderTask,
        "options": OptionsLoaderTask,
        "audit": CoverageAuditTask,
        "rotation": SectorRotationSnapshotTask,
        "v8_stock": V8StockTask,
        "v8_index": V8IndexTask,
        "v8_board_refresh": V8BoardRefreshTask,
        "v8_stock_basic": V8StockBasicTask,
        "v8_sw_industry_daily": V8SwIndustryDailyTask,
        "v8_rotation": V8RotationTask,
        "v8_rotation_audit": V8RotationAuditTask,
        "v8_rotation_repair": V8RotationRepairTask,
        "v8_event_daily": V8EventDailyTask,
        "v8_event_periodic": V8EventPeriodicTask,
        "v8_stock_fundamental_refresh": V8StockFundamentalRefreshTask,
        "v8_daily": V8DailyOpsTask,
        "v8_daily_market_raw": V8DailyMarketRawTask,
        "v8_daily_reference": V8DailyReferenceRefreshTask,
        "v8_daily_audit": V8DailyAuditTask,
        "v8_daily_derived_foundation": V8DailyDerivedFoundationTask,
        "v8_daily_derived_mainline": V8DailyDerivedMainlineTask,
        "v8_daily_derived_alpha": V8DailyDerivedAlphaTask,
        "v8_weekly": V8WeeklyOpsTask,
        "v8_weekly_refresh": V8WeeklyRefreshTask,
        "v8_weekly_audit": V8WeeklyAuditTask,
        "v8_weekly_audit_stock": V8WeeklyAuditStockTask,
        "v8_weekly_audit_index": V8WeeklyAuditIndexTask,
        "v8_weekly_audit_market": V8WeeklyAuditMarketTask,
        "v8_weekly_finalize": V8WeeklyFinalizeTask,
        "v8_monthly": V8MonthlyOpsTask,
        "v8_monthly_refresh": V8MonthlyRefreshTask,
        "v8_monthly_audit": V8MonthlyAuditTask,
        "v8_monthly_derived": V8MonthlyDerivedTask,
        "v8_backfill": V8HistoricalBackfillTask,
    }

    unknown = [t for t in selected if t not in tasks_map]
    if unknown:
        raise SystemExit(f"Unknown task(s): {unknown}. Allowed: {sorted(tasks_map.keys())} or 'all'.")
    if "his_stocks" in selected and len(selected) > 1:
        raise SystemExit("Task 'his_stocks' must run alone for safety. Example: --tasks his_stocks")

    # DbInitTask is always executed first
    tasks = [DbInitTask()]

    engine = build_engine()
    # Force an early MySQL connectivity check so failures are explicit at startup.
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    log.info("mysql: connection check ok")
    ctx = RunContext(config=cfg, engine=engine, log=log)

    # Preserve a stable execution order + rotation
    order = ["v8_backfill", "v8_stock", "v8_index", "v8_board_refresh", "v8_stock_basic", "v8_sw_industry_daily", "v8_rotation", "v8_rotation_audit", "v8_rotation_repair", "v8_event_daily", "v8_event_periodic", "v8_stock_fundamental_refresh", "v8_daily", "v8_daily_market_raw", "v8_daily_reference", "v8_daily_audit", "v8_daily_derived_foundation", "v8_daily_derived_mainline", "v8_daily_derived_alpha", "v8_weekly", "v8_weekly_refresh", "v8_weekly_audit", "v8_weekly_audit_stock", "v8_weekly_audit_index", "v8_weekly_audit_market", "v8_weekly_finalize", "v8_monthly", "v8_monthly_refresh", "v8_monthly_audit", "v8_monthly_derived", "his_stocks", "stock", "board", "stock_basic", "sw_industry", "stock_fundamental", "stock_quality_snapshot", "stock_working_capital", "inst_fund_hold", "rotation", "event", "event_daily", "event_periodic", "etf", "index", "futures", "options", "audit"]
    for name in order:
        if name in selected:
            tasks.append(tasks_map[name]())


    RunnerApp(ctx, tasks).run()
  
