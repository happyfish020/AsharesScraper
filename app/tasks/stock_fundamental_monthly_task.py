from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta

from sqlalchemy import text

from app.tools.sync_cn_stock_daily_price_from_tushare import (
    patch_pandas_fillna_method_compat,
    resolve_tushare_token,
)
from app.tools.sync_cn_stock_fundamental_monthly import (
    apply_ddl,
    ensure_tables,
    load_balancesheet_tushare,
    load_cashflow_tushare,
    load_income_tushare,
    load_fina_indicator_akshare,
    load_fina_indicator_tushare,
    load_monthly_basic_akshare,
    load_monthly_basic_tushare,
    rebuild_quality_snapshot,
)
from app.tasks.stock_working_capital_alert_task import StockWorkingCapitalAlertTask


def _parse_yyyymmdd(s: str) -> date:
    return datetime.strptime(str(s), "%Y%m%d").date()


def _quarter_start(dt: date) -> date:
    month = ((dt.month - 1) // 3) * 3 + 1
    return date(dt.year, month, 1)


def _next_day_from_db_date(value, fallback: date) -> date:
    if value is None:
        return fallback
    if isinstance(value, date):
        dt = value
    else:
        dt = datetime.strptime(str(value), "%Y-%m-%d").date()
    return dt + timedelta(days=1)


@dataclass
class StockFundamentalMonthlyTask:
    name: str = "StockFundamentalMonthlyTask"

    def run(self, ctx) -> None:
        cfg = ctx.config
        cfg.finalize_dates()
        data_source_flag = str(getattr(cfg, "data_source_flag", "tu") or "tu").strip().lower()

        enabled = str(os.getenv("STOCK_FUNDAMENTAL_MONTHLY_ENABLED", "1")).strip().lower()
        if enabled in {"0", "false", "no", "off"}:
            ctx.log.info("[stock_fundamental_monthly] skip by env STOCK_FUNDAMENTAL_MONTHLY_ENABLED")
            return

        end_date = _parse_yyyymmdd(str(cfg.end_date))
        force = str(os.getenv("STOCK_FUNDAMENTAL_MONTHLY_FORCE", "0")).strip().lower() in {"1", "true", "yes", "on"}
        full_rebuild = str(os.getenv("STOCK_FUNDAMENTAL_MONTHLY_FULL_REBUILD", "0")).strip().lower() in {"1", "true", "yes", "on"}
        run_day = max(1, min(31, int(os.getenv("STOCK_FUNDAMENTAL_MONTHLY_MONTHDAY", "1"))))
        provider = str(os.getenv("STOCK_FUNDAMENTAL_MONTHLY_PROVIDER", "")).strip().lower()
        if not provider:
            provider = "tushare" if data_source_flag == "tu" else "akshare"
        calendar_source = str(os.getenv("STOCK_FUNDAMENTAL_MONTHLY_CALENDAR_SOURCE", "price")).strip().lower() or "price"
        history_start_raw = str(os.getenv("STOCK_FUNDAMENTAL_MONTHLY_HISTORY_START", "20080101")).strip() or "20080101"
        basic_source_label = str(os.getenv("STOCK_FUNDAMENTAL_MONTHLY_BASIC_SOURCE_LABEL", "tushare_monthly_basic")).strip() or "tushare_monthly_basic"
        income_source_label = str(os.getenv("STOCK_FUNDAMENTAL_MONTHLY_INCOME_SOURCE_LABEL", "tushare_income")).strip() or "tushare_income"
        balance_source_label = str(os.getenv("STOCK_FUNDAMENTAL_MONTHLY_BALANCE_SOURCE_LABEL", "tushare_balancesheet")).strip() or "tushare_balancesheet"
        fina_source_label = str(os.getenv("STOCK_FUNDAMENTAL_MONTHLY_FINA_SOURCE_LABEL", "tushare_fina_indicator")).strip() or "tushare_fina_indicator"
        cashflow_source_label = str(os.getenv("STOCK_FUNDAMENTAL_MONTHLY_CASHFLOW_SOURCE_LABEL", "tushare_cashflow")).strip() or "tushare_cashflow"
        by_symbol = str(os.getenv("STOCK_FUNDAMENTAL_MONTHLY_BY_SYMBOL", "0")).strip().lower() in {"1", "true", "yes", "on"}
        akshare_workers = max(1, int(os.getenv("STOCK_FUNDAMENTAL_MONTHLY_AKSHARE_WORKERS", "8")))
        akshare_timeout = float(os.getenv("STOCK_FUNDAMENTAL_MONTHLY_AKSHARE_TIMEOUT", "15"))
        skip_quality_snapshot = str(os.getenv("STOCK_FUNDAMENTAL_MONTHLY_SKIP_QUALITY_SNAPSHOT", "0")).strip().lower() in {"1", "true", "yes", "on"}
        frequency_mode = str(os.getenv("STOCK_FUNDAMENTAL_MODE", "all")).strip().lower() or "all"
        if frequency_mode not in {"all", "monthly", "monthly_basic", "quarterly", "financial", "financials"}:
            raise RuntimeError(f"[stock_fundamental_monthly] unsupported STOCK_FUNDAMENTAL_MODE={frequency_mode}; expected all/monthly/quarterly")
        run_monthly_basic = frequency_mode in {"all", "monthly", "monthly_basic"}
        run_quarterly_financials = frequency_mode in {"all", "quarterly", "financial", "financials"}

        if data_source_flag == "ak":
            raise RuntimeError("[stock_fundamental_monthly] flag=ak is not supported yet; use --flag tu")

        if not force and end_date.day != run_day:
            ctx.log.info(
                "[stock_fundamental_monthly] skip: end_date=%s day=%s target_day=%s",
                end_date,
                end_date.day,
                run_day,
            )
            return

        patch_pandas_fillna_method_compat()
        token = ""
        tried_files = []
        if provider == "tushare":
            token, tried_files = resolve_tushare_token("", "")
            if not token:
                msg = "Tushare token is required for stock_fundamental_monthly"
                if tried_files:
                    msg += f"; tried_files={', '.join(str(p) for p in tried_files)}"
                raise RuntimeError(msg)
        elif provider != "akshare":
            raise RuntimeError(f"[stock_fundamental_monthly] unsupported provider={provider}")

        history_start = _parse_yyyymmdd(history_start_raw)
        ensure_tables(ctx.engine)

        with ctx.engine.connect() as conn:
            existing_basic_max = conn.execute(text("SELECT MAX(trade_date) FROM cn_stock_monthly_basic")).scalar()
            existing_income_max = conn.execute(text("SELECT MAX(end_date) FROM cn_stock_income")).scalar()
            existing_balance_max = conn.execute(text("SELECT MAX(end_date) FROM cn_stock_balancesheet")).scalar()
            existing_fina_max = conn.execute(text("SELECT MAX(end_date) FROM cn_stock_fina_indicator")).scalar()
            existing_income_ann_max = conn.execute(text("SELECT MAX(ann_date) FROM cn_stock_income")).scalar()
            existing_balance_ann_max = conn.execute(text("SELECT MAX(ann_date) FROM cn_stock_balancesheet")).scalar()
            existing_fina_ann_max = conn.execute(text("SELECT MAX(ann_date) FROM cn_stock_fina_indicator")).scalar()

        if full_rebuild or existing_basic_max is None:
            basic_start = history_start
        else:
            requested_start = _parse_yyyymmdd(str(cfg.start_date))
            default_start = max(history_start, requested_start)
            existing_basic_dt = existing_basic_max if isinstance(existing_basic_max, date) else datetime.strptime(str(existing_basic_max), "%Y-%m-%d").date()
            next_month = (existing_basic_dt.replace(day=1) + timedelta(days=32)).replace(day=1)
            basic_start = max(default_start, next_month)

        # Financial tables are quarterly by report period, but incremental loading is driven by
        # announcement/disclosure date. Do not derive start from MAX(end_date); otherwise
        # a normal April run scans the whole earnings season from 04-01.
        if full_rebuild:
            fina_start = history_start
            income_start = history_start
            balance_start = history_start
        else:
            requested_start = _parse_yyyymmdd(str(cfg.start_date))
            default_start = max(history_start, requested_start)
            fina_start = max(default_start, _next_day_from_db_date(existing_fina_ann_max, default_start))
            income_start = max(default_start, _next_day_from_db_date(existing_income_ann_max, default_start))
            balance_start = max(default_start, _next_day_from_db_date(existing_balance_ann_max, default_start))

        ctx.log.info(
            "[stock_fundamental_monthly] mode=%s history_start=%s end_date=%s force=%s full_rebuild=%s "
            "basic_existing_max=%s income_existing_max=%s balance_existing_max=%s fina_existing_max=%s "
            "income_ann_existing_max=%s balance_ann_existing_max=%s fina_ann_existing_max=%s "
            "basic_start=%s income_start=%s balance_start=%s fina_start=%s",
            frequency_mode,
            history_start,
            end_date,
            force,
            full_rebuild,
            existing_basic_max,
            existing_income_max,
            existing_balance_max,
            existing_fina_max,
            existing_income_ann_max,
            existing_balance_ann_max,
            existing_fina_ann_max,
            basic_start,
            income_start,
            balance_start,
            fina_start,
        )

        basic_rows = basic_affected = income_rows = income_affected = balance_rows = balance_affected = 0
        fina_rows = fina_affected = cashflow_rows = cashflow_affected = 0
        basic_dates = []
        income_periods = []
        balance_periods = []
        fina_periods = []
        cashflow_periods = []
        used_provider = provider
        ak_failures = 0

        if run_monthly_basic and basic_start <= end_date:
            if provider == "akshare":
                basic_rows, basic_affected, basic_dates, used_provider, ak_failures = load_monthly_basic_akshare(
                    engine=ctx.engine,
                    end_date=end_date,
                    source_label="akshare_monthly_basic",
                    max_workers=akshare_workers,
                    timeout=akshare_timeout,
                )
            else:
                if not token:
                    raise RuntimeError("tushare token missing")
                basic_rows, basic_affected, basic_dates, used_provider = load_monthly_basic_tushare(
                    engine=ctx.engine,
                    start_date=basic_start,
                    end_date=end_date,
                    calendar_source=calendar_source,
                    source_label=basic_source_label,
                    token=token,
                    log=ctx.log,
                )

        if run_quarterly_financials and income_start <= end_date:
            if provider == "akshare":
                ctx.log.info("[stock_fundamental_monthly] income skipped: akshare provider has no income fallback")
            else:
                if not token:
                    raise RuntimeError("tushare token missing")
                income_rows, income_affected, income_periods, _ = load_income_tushare(
                    engine=ctx.engine,
                    start_date=income_start,
                    end_date=end_date,
                    source_label=income_source_label,
                    token=token,
                    log=ctx.log,
                    by_symbol=by_symbol,
                )

        if run_quarterly_financials and balance_start <= end_date:
            if provider == "akshare":
                ctx.log.info("[stock_fundamental_monthly] balancesheet skipped: akshare provider has no balancesheet fallback")
            else:
                if not token:
                    raise RuntimeError("tushare token missing")
                balance_rows, balance_affected, balance_periods, _ = load_balancesheet_tushare(
                    engine=ctx.engine,
                    start_date=balance_start,
                    end_date=end_date,
                    source_label=balance_source_label,
                    token=token,
                    log=ctx.log,
                    by_symbol=by_symbol,
                )

        if run_quarterly_financials and fina_start <= end_date:
            if provider == "akshare":
                fina_rows, fina_affected, fina_periods, _, fina_failures = load_fina_indicator_akshare(
                    engine=ctx.engine,
                    start_date=fina_start,
                    end_date=end_date,
                    source_label="akshare_fina_indicator",
                    max_workers=akshare_workers,
                )
                ak_failures += fina_failures
            else:
                if not token:
                    raise RuntimeError("tushare token missing")
                fina_rows, fina_affected, fina_periods, _ = load_fina_indicator_tushare(
                    engine=ctx.engine,
                    start_date=fina_start,
                    end_date=end_date,
                    source_label=fina_source_label,
                    token=token,
                    log=ctx.log,
                    by_symbol=by_symbol,
                )

        if run_quarterly_financials and fina_start <= end_date:
            if provider == "akshare":
                ctx.log.info("[stock_fundamental_monthly] cashflow skipped: akshare provider has no cashflow fallback")
            else:
                if not token:
                    raise RuntimeError("tushare token missing")
                cashflow_rows, cashflow_affected, cashflow_periods, _ = load_cashflow_tushare(
                    engine=ctx.engine,
                    start_date=fina_start,
                    end_date=end_date,
                    source_label=cashflow_source_label,
                    token=token,
                    log=ctx.log,
                    by_symbol=by_symbol,
                )

        apply_ddl(ctx.engine, "docs/DDL/cn_market.cn_stock_fundamental_quality_v1.sql")
        apply_ddl(ctx.engine, "docs/DDL/cn_market.cn_stock_fundamental_quality_hist_v1.sql")
        snap_rows = -1
        if skip_quality_snapshot:
            ctx.log.info("[stock_fundamental_monthly] skip quality snapshot rebuild by env STOCK_FUNDAMENTAL_MONTHLY_SKIP_QUALITY_SNAPSHOT")
        else:
            snap_rows = rebuild_quality_snapshot(ctx.engine)
        StockWorkingCapitalAlertTask().run(ctx)

        ctx.log.info(
            "[stock_fundamental_monthly] done provider=%s basic_start=%s basic_end=%s basic_dates=%s basic_rows=%s basic_affected=%s "
            "income_start=%s income_end=%s income_periods=%s income_rows=%s income_affected=%s "
            "balance_start=%s balance_end=%s balance_periods=%s balance_rows=%s balance_affected=%s "
            "fina_start=%s fina_end=%s fina_periods=%s fina_rows=%s fina_affected=%s "
            "cashflow_periods=%s cashflow_rows=%s cashflow_affected=%s ak_failures=%s "
            "full_rebuild=%s skip_quality_snapshot=%s views_applied=3 snap_rows=%s",
            used_provider,
            basic_start,
            end_date,
            len(basic_dates),
            basic_rows,
            basic_affected,
            income_start,
            end_date,
            len(income_periods),
            income_rows,
            income_affected,
            balance_start,
            end_date,
            len(balance_periods),
            balance_rows,
            balance_affected,
            fina_start,
            end_date,
            len(fina_periods),
            fina_rows,
            fina_affected,
            len(cashflow_periods),
            cashflow_rows,
            cashflow_affected,
            ak_failures,
            full_rebuild,
            skip_quality_snapshot,
            snap_rows,
        )
