from __future__ import annotations

import argparse
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


def _parse_args(argv: Optional[List[str]] = None):
    p = argparse.ArgumentParser(description="AsharesScraper Runner")
    p.add_argument("--asof", default="latest", help="run as-of trading day: latest or YYYYMMDD")
    p.add_argument("--days", type=int, default=1, help="lookback window days (used when --asof=latest and you want backfill)")
    p.add_argument("--tasks", default="stock", help="comma-separated: stock,etf,index,futures,options,audit,all")
    p.add_argument("--refresh", action="store_true", help="clear state files and refresh")
    p.add_argument("--no-vpn", action="store_true", help="skip wireguard start/stop")
    return p.parse_args(argv)


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
    # resolve asof date
    if isinstance(args.asof, str) and args.asof.lower() == "latest":
        asof = get_latest_trade_date()
    else:
        asof = str(args.asof).strip()

    # RunnerConfig: keep tasks logic unchanged; we just drive an asof-centric window
    # - start_date=end_date=asof for true daily run
    # - if user wants backfill, they can pass --days and we set start based on finalize_dates
    cfg = RunnerConfig(
        look_back_days=args.days,
        manual_stock_symbols=[],
        index_symbols=DEFAULT_INDEX_SYMBOLS,
        base_index=DEFAULT_BASE_INDEX,
        refresh_state=args.refresh,
    )

    # If user explicitly targets a day, make runner strictly daily and clean.
    if asof and asof != "latest":
        cfg.start_date = asof
        cfg.end_date = asof
    else:
        # latest
        cfg.start_date = asof
        cfg.end_date = asof

    tasks_map = {
        "stock": StockLoaderTask,
        "etf": EtfLoaderTask,
        "index": IndexLoaderTask,
        "futures": FuturesLoaderTask,
        "options": OptionsLoaderTask,
        "audit": CoverageAuditTask,
        "rotation": SectorRotationSnapshotTask,
    }

    # DbInitTask is always executed first
    tasks = [DbInitTask()]

    raw = (args.tasks or "stock").strip().lower()
    selected = []
    if raw in ("all", "*"):
        selected = ["stock", "etf", "index", "futures", "options", "audit"]
    else:
        selected = [t.strip() for t in raw.split(",") if t.strip()]

    unknown = [t for t in selected if t not in tasks_map]
    if unknown:
        raise SystemExit(f"Unknown task(s): {unknown}. Allowed: {sorted(tasks_map.keys())} or 'all'.")

    engine = build_engine()
    # Force an early MySQL connectivity check so failures are explicit at startup.
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    log.info("mysql: connection check ok")
    ctx = RunContext(config=cfg, engine=engine, log=log)

    # Preserve a stable execution order + rotation
    order = ["stock", "rotation", "etf", "index", "futures", "options", "audit"]
    for name in order:
        if name in selected:
            tasks.append(tasks_map[name]())


    RunnerApp(ctx, tasks).run()
  
