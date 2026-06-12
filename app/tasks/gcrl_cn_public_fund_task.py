from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime

from app.tools.sync_cn_gcrl_public_fund_from_tushare import (
    DEFAULT_MAX_FUNDS,
    _latest_completed_report_period,
    run_gcrl_cn_public_fund_sync,
)
from app.tools.sync_cn_stock_daily_price_from_tushare import resolve_tushare_token


def _parse_yyyymmdd(s: str):
    return datetime.strptime(str(s), "%Y%m%d").date()


@dataclass
class GcrlCnPublicFundTask:
    name: str = "GcrlCnPublicFundTask"

    def run(self, ctx) -> None:
        report_period_raw = str(os.getenv("GCRL_CN_REPORT_PERIOD", "")).strip()
        report_period = _parse_yyyymmdd(report_period_raw) if report_period_raw else _latest_completed_report_period()
        token, _ = resolve_tushare_token(os.getenv("TUSHARE_TOKEN", ""), "")
        result = run_gcrl_cn_public_fund_sync(
            engine=ctx.engine,
            report_period=report_period,
            token=token or "",
            max_funds=max(0, int(os.getenv("GCRL_CN_MAX_FUNDS", str(DEFAULT_MAX_FUNDS)))),
            eastmoney_sleep_seconds=max(0.0, float(os.getenv("GCRL_CN_EASTMONEY_SLEEP_SECONDS", "0.08"))),
            dry_run=str(os.getenv("GCRL_CN_DRY_RUN", "0")).strip().lower() in {"1", "true", "yes", "on"},
            allow_empty=str(os.getenv("GCRL_CN_ALLOW_EMPTY", "0")).strip().lower() in {"1", "true", "yes", "on"},
            log=ctx.log,
        )
        ctx.log.info("[GCRL-CN] public fund sync result=%s", result)
