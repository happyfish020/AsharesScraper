from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta

from sqlalchemy import text

from app.tools.sync_cn_stock_daily_basic_from_tushare import (
    ensure_table,
    load_daily_basic_akshare,
    load_daily_basic_tushare,
)
from app.tools.sync_cn_stock_daily_price_from_tushare import (
    patch_pandas_fillna_method_compat,
    resolve_tushare_token,
)


def _parse_yyyymmdd(s: str) -> date:
    return datetime.strptime(str(s), "%Y%m%d").date()


def _env_value(*names: str, default: str = "") -> str:
    for name in names:
        value = os.getenv(name)
        if value is not None and str(value).strip() != "":
            return str(value).strip()
    return default


def _env_bool(*names: str, default: bool = False) -> bool:
    raw = _env_value(*names, default="")
    if raw == "":
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


def _env_int(*names: str, default: int) -> int:
    raw = _env_value(*names, default="")
    if raw == "":
        return default
    return int(raw)


def _env_float(*names: str, default: float) -> float:
    raw = _env_value(*names, default="")
    if raw == "":
        return default
    return float(raw)


@dataclass
class StockBasicTask:
    name: str = "StockBasicTask"

    def run(self, ctx) -> None:
        cfg = ctx.config
        cfg.finalize_dates()
        data_source_flag = str(getattr(cfg, "data_source_flag", "tu") or "tu").strip().lower()

        enabled = _env_bool("STOCK_BASIC_ENABLED", "STOCK_BASIC_WEEKLY_ENABLED", default=True)
        if not enabled:
            ctx.log.info("[stock_basic] skip by env STOCK_BASIC_ENABLED")
            return

        requested_start = _parse_yyyymmdd(str(cfg.start_date))
        end_date = _parse_yyyymmdd(str(cfg.end_date))
        force = _env_bool("STOCK_BASIC_FORCE", "STOCK_BASIC_WEEKLY_FORCE", default=False)
        calendar_source = _env_value("STOCK_BASIC_CALENDAR_SOURCE", "STOCK_BASIC_WEEKLY_CALENDAR_SOURCE", default="price").lower() or "price"
        provider = _env_value("STOCK_BASIC_PROVIDER", "STOCK_BASIC_WEEKLY_PROVIDER", default="").lower()
        if not provider:
            provider = "tushare" if data_source_flag == "tu" else "akshare"
        source_label = _env_value("STOCK_BASIC_SOURCE_LABEL", "STOCK_BASIC_WEEKLY_SOURCE_LABEL", default="tushare_daily_basic") or "tushare_daily_basic"
        refresh_window_days = max(1, _env_int("STOCK_BASIC_LOOKBACK_DAYS", "STOCK_BASIC_WEEKLY_LOOKBACK_DAYS", default=7))
        date_order = _env_value("STOCK_BASIC_DATE_ORDER", default="asc").lower() or "asc"
        batch_size = max(0, _env_int("STOCK_BASIC_BATCH_SIZE", default=0))
        akshare_workers = max(1, _env_int("STOCK_BASIC_AKSHARE_WORKERS", "STOCK_BASIC_WEEKLY_AKSHARE_WORKERS", default=12))
        akshare_timeout = _env_float("STOCK_BASIC_AKSHARE_TIMEOUT", "STOCK_BASIC_WEEKLY_AKSHARE_TIMEOUT", default=15.0)

        if data_source_flag == "ak":
            raise RuntimeError("[stock_basic] flag=ak is not supported yet; use --flag tu")
        if provider not in {"tushare", "akshare"}:
            raise RuntimeError(f"[stock_basic] unsupported provider={provider}")
        if calendar_source not in {"board-map", "price"}:
            raise RuntimeError(f"[stock_basic] unsupported calendar_source={calendar_source}")
        if date_order not in {"asc", "desc"}:
            raise RuntimeError(f"[stock_basic] unsupported date_order={date_order}")

        patch_pandas_fillna_method_compat()
        token = ""
        tried_files = []
        if provider == "tushare":
            token, tried_files = resolve_tushare_token("", "")
            if not token:
                msg = "Tushare token is required for stock_basic"
                if tried_files:
                    msg += f"; tried_files={', '.join(str(p) for p in tried_files)}"
                raise RuntimeError(msg)

        ensure_table(ctx.engine)

        if force:
            start_date = requested_start
        elif date_order == "desc":
            start_date = max(requested_start, end_date - timedelta(days=refresh_window_days - 1))
        else:
            start_date = max(requested_start, end_date - timedelta(days=refresh_window_days - 1))

        if start_date > end_date:
            ctx.log.info("[stock_basic] no range to load: start=%s end=%s", start_date, end_date)
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
            total_rows, total_affected, trade_dates, used_provider = load_daily_basic_tushare(
                engine=ctx.engine,
                start_date=start_date,
                end_date=end_date,
                calendar_source=calendar_source,
                source_label=source_label,
                token=token,
                descending=date_order == "desc",
                batch_size=batch_size,
                log=ctx.log,
            )

        if not trade_dates:
            ctx.log.info(
                "[stock_basic] no trade dates matched: start=%s end=%s provider=%s calendar_source=%s",
                start_date,
                end_date,
                used_provider,
                calendar_source,
            )
            return

        # cn_stock_leader_score_daily is a BASE TABLE (partitioned, materialized).
        # Materialize via stored procedure (replaces old v1/v2 view chain).
        materialize_start = start_date.strftime("%Y-%m-%d")
        materialize_end = end_date.strftime("%Y-%m-%d")
        ctx.log.info(
            "[stock_basic] materializing cn_stock_leader_score_daily via SP: %s ~ %s",
            materialize_start, materialize_end,
        )
        _materialize_leader_score_daily(ctx.engine, materialize_start, materialize_end)

        ctx.log.info(
            "[stock_basic] done provider=%s start=%s end=%s dates=%s rows=%s affected=%s ak_failures=%s calendar_source=%s date_order=%s batch_size=%s leader_materialized=1",
            used_provider,
            trade_dates[0],
            trade_dates[-1],
            len(trade_dates),
            total_rows,
            total_affected,
            ak_failures,
            calendar_source,
            date_order,
            batch_size,
        )


def _materialize_leader_score_daily(engine, start_date: str, end_date: str) -> int:
    """Materialize cn_stock_leader_score_daily via stored procedure.

    Calls sp_materialize_leader_score(start_date, end_date) which uses
    temporary tables to replace the old v1/v2 view chain.
    Returns the number of affected rows.
    """
    sql = "CALL sp_materialize_leader_score(:start_date, :end_date)"
    with engine.begin() as conn:
        result = conn.execute(text(sql), {"start_date": start_date, "end_date": end_date})
        # For CALL statements, rowcount may not reflect inserted rows reliably.
        # We return 0 and let the caller rely on before/after comparison if needed.
        affected = result.rowcount if result.rowcount > 0 else 0
    return affected


# Backward-compatible alias for existing imports.
StockBasicWeeklyTask = StockBasicTask
