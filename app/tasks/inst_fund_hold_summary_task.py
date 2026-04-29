from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, datetime

from app.tools.sync_cn_inst_fund_hold_summary import (
    compute_incremental_start,
    ensure_tables,
    load_inst_fund_hold_summary_tushare,
)
from app.tools.sync_cn_stock_daily_price_from_tushare import (
    patch_pandas_fillna_method_compat,
    resolve_tushare_token,
)


def _parse_yyyymmdd(s: str) -> date:
    return datetime.strptime(str(s), "%Y%m%d").date()


@dataclass
class InstFundHoldSummaryTask:
    name: str = "InstFundHoldSummaryTask"

    def run(self, ctx) -> None:
        cfg = ctx.config
        cfg.finalize_dates()

        enabled = str(os.getenv("INST_FUND_HOLD_SUMMARY_ENABLED", "1")).strip().lower()
        if enabled in {"0", "false", "no", "off"}:
            ctx.log.info("[inst_fund_hold_summary] skip by env INST_FUND_HOLD_SUMMARY_ENABLED")
            return

        history_start = _parse_yyyymmdd(str(os.getenv("INST_FUND_HOLD_SUMMARY_HISTORY_START", "20100101")).strip() or "20100101")
        requested_start = _parse_yyyymmdd(str(cfg.start_date))
        end_date = _parse_yyyymmdd(str(cfg.end_date))
        lookback_quarters = max(1, int(os.getenv("INST_FUND_HOLD_SUMMARY_LOOKBACK_QUARTERS", "2")))
        force_full = str(os.getenv("INST_FUND_HOLD_SUMMARY_FORCE_FULL", "0")).strip().lower() in {"1", "true", "yes", "on"}
        page_size = max(1, int(os.getenv("INST_FUND_HOLD_SUMMARY_PAGE_SIZE", "5000")))
        sleep_seconds = max(0.0, float(os.getenv("INST_FUND_HOLD_SUMMARY_SLEEP_SECONDS", "0.2")))
        source_label = str(os.getenv("INST_FUND_HOLD_SUMMARY_SOURCE_LABEL", "tushare_fund_portfolio")).strip() or "tushare_fund_portfolio"

        patch_pandas_fillna_method_compat()
        token, tried_files = resolve_tushare_token("", "")
        if not token:
            msg = "Tushare token is required for inst_fund_hold_summary"
            if tried_files:
                msg += f"; tried_files={', '.join(str(p) for p in tried_files)}"
            raise RuntimeError(msg)

        ensure_tables(ctx.engine)
        start_date = max(history_start, requested_start) if force_full else compute_incremental_start(
            ctx.engine,
            requested_start=requested_start,
            history_start=history_start,
            lookback_quarters=lookback_quarters,
        )

        raw_rows, summary_rows, periods_done = load_inst_fund_hold_summary_tushare(
            engine=ctx.engine,
            start_date=start_date,
            end_date=end_date,
            token=token,
            source_label=source_label,
            page_size=page_size,
            sleep_seconds=sleep_seconds,
            log=ctx.log,
        )
        ctx.log.info(
            "[inst_fund_hold_summary] done start=%s end=%s periods=%s raw_rows=%s summary_rows=%s page_size=%s",
            start_date,
            end_date,
            len(periods_done),
            raw_rows,
            summary_rows,
            page_size,
        )
