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


def _parse_args(argv: Optional[List[str]] = None):
    p = argparse.ArgumentParser(description="AsharesScraper Runner")
    p.add_argument("--asof", default="latest", help="run as-of trading day: latest or YYYYMMDD")
    p.add_argument("--days", type=int, default=1, help="lookback window days (used when --asof=latest and you want backfill)")
    p.add_argument("--tasks", default="stock", help="comma-separated: stock,his_stocks,board,rotation,etf,index,futures,options,audit,all")
    p.add_argument("--history-start", default="20000101", help="his_stocks only: YYYYMMDD or YYYY-MM")
    p.add_argument("--history-end", default="latest", help="his_stocks only: latest or YYYYMMDD or YYYY-MM")
    p.add_argument("--history-source-order", default="baostock,ak", help="his_stocks only: comma-separated source priority, default baostock,ak")
    p.add_argument("--history-max-symbols", type=int, default=0, help="his_stocks only: limit symbols for smoke test; 0 means all")
    p.add_argument("--history-symbols", default="", help="his_stocks only: comma-separated symbols, e.g. 000001,600000; when set, skip universe scan")
    p.add_argument("--history-ignore-state", action="store_true", help="his_stocks only: ignore scanned/failed state and force re-check")
    p.add_argument("--history-alternate-bs-ak", action="store_true", help="his_stocks only: odd symbol->baostock first, even symbol->ak first")
    p.add_argument("--history-universe-frequency", default="monthly", choices=["monthly", "weekly"], help="his_stocks only: anchor frequency for query_all_stock(date)")
    p.add_argument("--refresh", action="store_true", help="clear state files and refresh")
    p.add_argument("--no-vpn", action="store_true", help="skip wireguard start/stop")
    return p.parse_args(argv)


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
        selected = ["stock", "board", "rotation", "etf", "index", "futures", "options", "audit"]
    else:
        selected = [t.strip() for t in raw.split(",") if t.strip()]

    # resolve asof date
    if isinstance(args.asof, str) and args.asof.lower() == "latest":
        asof = get_latest_trade_date()
    else:
        asof = str(args.asof).strip()

    # RunnerConfig: drive an asof-centric window.
    # - --asof YYYYMMDD: strict single-day
    # - --asof latest --days N: multi-day backfill [asof-(N-1), asof]
    cfg = RunnerConfig(
        look_back_days=args.days,
        manual_stock_symbols=[],
        index_symbols=DEFAULT_INDEX_SYMBOLS,
        base_index=DEFAULT_BASE_INDEX,
        refresh_state=args.refresh,
    )

    # If user explicitly targets a day, keep strict single-day.
    if asof and asof != "latest":
        cfg.start_date = asof
        cfg.end_date = asof
    else:
        # latest with 1-day or multi-day option
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
        "etf": EtfLoaderTask,
        "index": IndexLoaderTask,
        "futures": FuturesLoaderTask,
        "options": OptionsLoaderTask,
        "audit": CoverageAuditTask,
        "rotation": SectorRotationSnapshotTask,
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
    order = ["his_stocks", "stock", "board", "rotation", "etf", "index", "futures", "options", "audit"]
    for name in order:
        if name in selected:
            tasks.append(tasks_map[name]())


    RunnerApp(ctx, tasks).run()
  
