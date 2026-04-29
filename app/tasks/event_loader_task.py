from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta

import tushare as ts
from sqlalchemy import text

from app.tools.sync_cn_stock_event_tushare import (
    ensure_tables,
    load_forecast,
    load_express,
    load_fina_indicator,
    load_disclosure_date,
    load_dividend,
    load_anns_d,
    rebuild_event_signal_daily,
    save_quality_report,
)
from app.tools.sync_cn_stock_daily_price_from_tushare import (
    _parse_ymd,
    patch_pandas_fillna_method_compat,
    resolve_tushare_token,
)


def _parse_yyyymmdd(s: str) -> date:
    return datetime.strptime(str(s), "%Y%m%d").date()


def _max_date(engine, table: str, col: str) -> date | None:
    with engine.connect() as conn:
        value = conn.execute(text(f"SELECT MAX({col}) FROM {table}")).scalar()
    if value is None:
        return None
    if isinstance(value, date):
        return value
    dt = datetime.strptime(str(value), "%Y-%m-%d")
    return dt.date()


@dataclass
class EventLoaderTask:
    name: str = "EventLoader"
    frequency_tag: str = "all"  # all | daily | periodic

    def run(self, ctx) -> None:
        cfg = ctx.config
        cfg.finalize_dates()
        data_source_flag = str(getattr(cfg, "data_source_flag", "tu") or "tu").strip().lower()
        frequency_tag = str(self.frequency_tag or "all").strip().lower()
        run_daily = frequency_tag in {"all", "daily"}
        run_periodic = frequency_tag in {"all", "periodic", "monthly", "quarterly"}

        enabled = str(os.getenv("EVENT_ENABLED", "1")).strip().lower()
        if enabled in {"0", "false", "no", "off"}:
            ctx.log.info("[event] skip by env EVENT_ENABLED")
            return

        if data_source_flag != "tu":
            raise RuntimeError("[event] flag=ak is not supported yet; use --flag tu")

        if not run_daily and not run_periodic:
            raise RuntimeError(f"[event] unknown frequency_tag={frequency_tag}; expected all/daily/periodic")

        patch_pandas_fillna_method_compat()
        token, tried_files = resolve_tushare_token("", "")
        if not token:
            msg = "[event] Tushare token missing"
            if tried_files:
                msg += f"; tried_files={', '.join(str(p) for p in tried_files)}"
            raise RuntimeError(msg)

        end_date = _parse_yyyymmdd(cfg.end_date)
        default_full_start = str(os.getenv("EVENT_FULL_START", "2008-01-01")).strip()
        full_start = _parse_ymd(default_full_start)
        buffer_days = int(os.getenv("EVENT_LOOKBACK_BUFFER_DAYS", "7"))
        force_full = str(os.getenv("EVENT_FORCE_FULL", "0")).strip().lower() in {"1", "true", "yes", "on"}
        with_anns = str(os.getenv("EVENT_WITH_ANNS", "0")).strip().lower() in {"1", "true", "yes", "on"}
        with_signal = str(os.getenv("EVENT_WITH_SIGNAL", "1")).strip().lower() in {"1", "true", "yes", "on"}
        audit_dir = str(os.getenv("EVENT_AUDIT_DIR", "audit_reports")).strip() or "audit_reports"

        ensure_tables(ctx.engine)
        pro = ts.pro_api(token)

        requested_start = _parse_yyyymmdd(cfg.start_date)

        def resolve_start(table: str, col: str) -> date:
            if force_full:
                return max(full_start, requested_start)
            db_max = _max_date(ctx.engine, table, col)
            if db_max is None:
                return max(full_start, requested_start)
            return max(full_start, requested_start, db_max - timedelta(days=buffer_days))

        start_forecast = resolve_start("cn_event_earnings_forecast", "ann_date") if run_daily else end_date
        start_express = resolve_start("cn_event_earnings_express", "ann_date") if run_daily else end_date
        start_disclosure = resolve_start("cn_event_disclosure_date", "end_date") if run_daily else end_date
        start_fina = resolve_start("cn_event_fina_indicator", "ann_date") if run_periodic else end_date
        start_dividend = resolve_start("cn_event_dividend", "ann_date") if run_periodic else end_date
        start_anns = resolve_start("cn_event_announcement_meta", "ann_date") if run_daily and with_anns else end_date

        rows_forecast = aff_forecast = 0
        rows_express = aff_express = 0
        rows_disclosure = aff_disclosure = 0
        rows_fina = aff_fina = 0
        rows_dividend = aff_dividend = 0
        rows_anns = aff_anns = 0
        signal_sources: set[str] = set()
        signal_start_dates: list[date] = []

        if run_daily:
            rows_forecast, aff_forecast = load_forecast(ctx.engine, pro, start_forecast, end_date, "tushare_forecast", log=ctx.log)
            rows_express, aff_express = load_express(ctx.engine, pro, start_express, end_date, "tushare_express", log=ctx.log)
            rows_disclosure, aff_disclosure = load_disclosure_date(ctx.engine, pro, start_disclosure, end_date, "tushare_disclosure_date", log=ctx.log)
            signal_sources.update({"forecast", "express"})
            signal_start_dates.extend([start_forecast, start_express])
            if with_anns:
                try:
                    rows_anns, aff_anns = load_anns_d(ctx.engine, pro, start_anns, end_date, "tushare_anns_d", log=ctx.log)
                except Exception as e:
                    ctx.log.warning("[event] anns_d skipped: %s", e)

        if run_periodic:
            raw_symbols = str(os.getenv("EVENT_SYMBOLS", "")).strip()
            max_symbols = int(os.getenv("EVENT_MAX_SYMBOLS", "0"))
            symbols = []
            if raw_symbols:
                symbols = [x.strip() for x in raw_symbols.replace(",", " ").split() if x.strip()]
            elif max_symbols > 0:
                with ctx.engine.connect() as conn:
                    rows = conn.execute(text("SELECT DISTINCT symbol FROM cn_stock_daily_price ORDER BY symbol LIMIT :n"), {"n": max_symbols}).fetchall()
                symbols = [str(row[0]).strip() for row in rows if str(row[0]).strip()]

            rows_fina, aff_fina = load_fina_indicator(ctx.engine, pro, start_fina, end_date, "tushare_fina_indicator", symbols=symbols if symbols else None, log=ctx.log)
            rows_dividend, aff_dividend = load_dividend(ctx.engine, pro, start_dividend, end_date, "tushare_dividend", log=ctx.log)
            signal_sources.update({"fina_indicator", "dividend"})
            signal_start_dates.extend([start_fina, start_dividend])

        if with_signal and signal_sources:
            rebuild_event_signal_daily(ctx.engine, min(signal_start_dates), end_date, include_sources=signal_sources)

        report_path = save_quality_report(ctx.engine, audit_dir)
        ctx.log.info(
            "[event:%s] done forecast=%s/%s express=%s/%s disclosure=%s/%s fina=%s/%s dividend=%s/%s anns=%s/%s report=%s",
            frequency_tag,
            rows_forecast, aff_forecast,
            rows_express, aff_express,
            rows_disclosure, aff_disclosure,
            rows_fina, aff_fina,
            rows_dividend, aff_dividend,
            rows_anns, aff_anns,
            report_path,
        )
