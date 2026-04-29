from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta

from sqlalchemy import text

from app.tools.sync_cn_stock_daily_basic_from_tushare import (
    apply_view,
    ensure_table,
    load_daily_basic_akshare,
    load_daily_basic_tushare,
)
from app.tools.sync_cn_stock_daily_price_from_tushare import (
    resolve_tushare_token,
    patch_pandas_fillna_method_compat,
)


def _parse_yyyymmdd(s: str) -> date:
    return datetime.strptime(str(s), "%Y%m%d").date()


@dataclass
class StockBasicWeeklyTask:
    name: str = "StockBasicWeeklyTask"

    def run(self, ctx) -> None:
        cfg = ctx.config
        cfg.finalize_dates()
        data_source_flag = str(getattr(cfg, "data_source_flag", "tu") or "tu").strip().lower()

        enabled = str(os.getenv("STOCK_BASIC_WEEKLY_ENABLED", "1")).strip().lower()
        if enabled in {"0", "false", "no", "off"}:
            ctx.log.info("[stock_basic_weekly] skip by env STOCK_BASIC_WEEKLY_ENABLED")
            return

        end_date = _parse_yyyymmdd(str(cfg.end_date))
        force = str(os.getenv("STOCK_BASIC_WEEKLY_FORCE", "0")).strip().lower() in {"1", "true", "yes", "on"}
        run_weekday = int(os.getenv("STOCK_BASIC_WEEKLY_WEEKDAY", "0"))
        calendar_source = str(os.getenv("STOCK_BASIC_WEEKLY_CALENDAR_SOURCE", "board-map")).strip().lower() or "board-map"
        fallback_days = max(1, int(os.getenv("STOCK_BASIC_WEEKLY_LOOKBACK_DAYS", "14")))
        provider = str(os.getenv("STOCK_BASIC_WEEKLY_PROVIDER", "")).strip().lower()
        if not provider:
            provider = "tushare" if data_source_flag == "tu" else "akshare"
        source_label = str(os.getenv("STOCK_BASIC_WEEKLY_SOURCE_LABEL", "tushare_daily_basic")).strip() or "tushare_daily_basic"
        akshare_workers = max(1, int(os.getenv("STOCK_BASIC_WEEKLY_AKSHARE_WORKERS", "12")))
        akshare_timeout = float(os.getenv("STOCK_BASIC_WEEKLY_AKSHARE_TIMEOUT", "15"))

        if data_source_flag == "ak":
            raise RuntimeError("[stock_basic_weekly] flag=ak is not supported yet; use --flag tu")

        if not force and end_date.weekday() != run_weekday:
            ctx.log.info(
                "[stock_basic_weekly] skip: end_date=%s weekday=%s target_weekday=%s",
                end_date,
                end_date.weekday(),
                run_weekday,
            )
            return

        patch_pandas_fillna_method_compat()
        token = ""
        tried_files = []
        if provider == "tushare":
            token, tried_files = resolve_tushare_token("", "")
            if not token:
                msg = "Tushare token is required for stock_basic_weekly"
                if tried_files:
                    msg += f"; tried_files={', '.join(str(p) for p in tried_files)}"
                raise RuntimeError(msg)
        elif provider != "akshare":
            raise RuntimeError(f"[stock_basic_weekly] unsupported provider={provider}")

        ensure_table(ctx.engine)

        with ctx.engine.connect() as conn:
            existing_max = conn.execute(text("SELECT MAX(trade_date) FROM cn_stock_daily_basic")).scalar()

        requested_start = _parse_yyyymmdd(str(cfg.start_date))
        if existing_max is None:
            start_date = max(requested_start, end_date - timedelta(days=fallback_days - 1))
        else:
            existing_max = existing_max if isinstance(existing_max, date) else datetime.strptime(str(existing_max), "%Y-%m-%d").date()
            start_date = max(requested_start, existing_max + timedelta(days=1))

        if start_date > end_date:
            ctx.log.info(
                "[stock_basic_weekly] no new range to load: start=%s end=%s existing_max=%s",
                start_date,
                end_date,
                existing_max,
            )
            apply_view(ctx.engine, "docs/DDL/cn_market.cn_stock_leader_score_v1.sql")
            apply_view(ctx.engine, "docs/DDL/cn_market.cn_stock_leader_score_v2.sql")
            ctx.log.info("[stock_basic_weekly] views refreshed only")
            return

        used_provider = provider
        ak_failures = 0
        if provider == "akshare":
            total_rows, total_affected, trade_dates, used_provider, ak_failures = load_daily_basic_akshare(
                engine=ctx.engine,
                start_date=start_date,
                end_date=end_date,
                source_label="akshare_individual_info_em",
                max_workers=akshare_workers,
                timeout=akshare_timeout,
                log=ctx.log,
            )
        else:
            if not token:
                raise RuntimeError("tushare token missing")
            total_rows, total_affected, trade_dates, used_provider = load_daily_basic_tushare(
                engine=ctx.engine,
                start_date=start_date,
                end_date=end_date,
                calendar_source=calendar_source,
                source_label=source_label,
                token=token,
                log=ctx.log,
            )

        if not trade_dates:
            ctx.log.info(
                "[stock_basic_weekly] no trade dates matched: start=%s end=%s provider=%s calendar_source=%s",
                start_date,
                end_date,
                used_provider,
                calendar_source,
            )
            return

        apply_view(ctx.engine, "docs/DDL/cn_market.cn_stock_leader_score_v1.sql")
        apply_view(ctx.engine, "docs/DDL/cn_market.cn_stock_leader_score_v2.sql")

        ctx.log.info(
            "[stock_basic_weekly] done provider=%s start=%s end=%s dates=%s rows=%s affected=%s ak_failures=%s views_refreshed=1",
            used_provider,
            trade_dates[0],
            trade_dates[-1],
            len(trade_dates),
            total_rows,
            total_affected,
            ak_failures,
        )
