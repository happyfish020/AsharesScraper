from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import date, timedelta

from app.tools.sync_cn_stock_daily_price_from_tushare import (
    patch_pandas_fillna_method_compat,
    resolve_tushare_token,
)
from app.tools.sync_cn_sw_industry_daily_from_tushare import (
    ensure_table,
    fetch_sw_l1_codes_from_tushare,
    fetch_sw_daily_with_retry,
    get_sw_l1_codes,
    normalize_sw_daily,
    resolve_start_dates,
    upsert_dataframe,
)

import tushare as ts


def _parse_yyyymmdd(s: str) -> date:
    from datetime import datetime
    return datetime.strptime(str(s), "%Y%m%d").date()


@dataclass
class SwIndustryDailyTask:
    name: str = "SwIndustryDailyTask"

    def run(self, ctx) -> None:
        enabled = str(os.getenv("SW_INDUSTRY_DAILY_ENABLED", "1")).strip().lower()
        if enabled in {"0", "false", "no", "off"}:
            ctx.log.info("[sw_industry_daily] skip by env SW_INDUSTRY_DAILY_ENABLED")
            return

        cfg = ctx.config
        cfg.finalize_dates()

        end_date = _parse_yyyymmdd(str(cfg.end_date))
        lookback_days = max(1, int(os.getenv("SW_INDUSTRY_DAILY_LOOKBACK_DAYS", "3")))
        force = str(os.getenv("SW_INDUSTRY_DAILY_FORCE", "0")).strip().lower() in {"1", "true", "yes", "on"}
        src = str(os.getenv("SW_INDUSTRY_DAILY_SRC", "SW2021")).strip() or "SW2021"
        master_source = str(os.getenv("SW_INDUSTRY_DAILY_MASTER_SOURCE", "TUSHARE_SW2021_L1")).strip() or "TUSHARE_SW2021_L1"
        sleep_between = float(os.getenv("SW_INDUSTRY_DAILY_SLEEP", "0.15"))
        source_label = str(os.getenv("SW_INDUSTRY_DAILY_SOURCE_LABEL", "tushare_sw_daily")).strip() or "tushare_sw_daily"

        patch_pandas_fillna_method_compat()
        token, tried_files = resolve_tushare_token("", "")
        if not token:
            msg = "Tushare token required for sw_industry_daily"
            if tried_files:
                msg += f"; tried_files={', '.join(str(p) for p in tried_files)}"
            raise RuntimeError(msg)

        pro = ts.pro_api(token)
        ensure_table(ctx.engine)

        codes = get_sw_l1_codes(ctx.engine, master_source)
        if not codes:
            codes = fetch_sw_l1_codes_from_tushare(pro, src=src)
        if not codes:
            ctx.log.warning("[sw_industry_daily] no SW L1 codes found; skip")
            return

        default_start = end_date - timedelta(days=lookback_days - 1)
        if force:
            start_dates = {code: default_start for code in codes}
        else:
            start_dates = resolve_start_dates(ctx.engine, codes, default_start)
            start_dates = {
                code: max(default_start, dt if isinstance(dt, date) else default_start)
                for code, dt in start_dates.items()
            }

        fields = "ts_code,trade_date,name,open,high,low,close,change,pct_change,vol,amount,pe,pb,float_mv"
        total_rows = total_affected = touched = 0

        for i, code in enumerate(codes, start=1):
            code_start = start_dates.get(code, default_start)
            if code_start > end_date:
                ctx.log.debug("[sw_industry_daily] %s/%s %s skip up-to-date", i, len(codes), code)
                continue
            ctx.log.debug("[sw_industry_daily] %s/%s %s %s..%s", i, len(codes), code, code_start, end_date)
            try:
                raw = fetch_sw_daily_with_retry(pro, ts_code=code, start_date=code_start, end_date=end_date, fields=fields)
            except Exception as exc:
                ctx.log.warning("[sw_industry_daily] %s fetch failed: %s", code, exc)
                continue
            df = normalize_sw_daily(raw, source_label=source_label)
            if df.empty:
                continue
            affected = upsert_dataframe(ctx.engine, df)
            total_rows += len(df)
            total_affected += affected
            touched += 1
            if sleep_between > 0:
                time.sleep(sleep_between)

        ctx.log.info(
            "[sw_industry_daily] done codes=%s touched=%s rows=%s affected=%s end_date=%s",
            len(codes),
            touched,
            total_rows,
            total_affected,
            end_date,
        )
