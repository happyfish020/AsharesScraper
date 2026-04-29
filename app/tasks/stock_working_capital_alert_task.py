from __future__ import annotations

import os
from dataclasses import dataclass

from sqlalchemy import text

from app.tools.sync_cn_stock_fundamental_monthly import apply_ddl


@dataclass
class StockWorkingCapitalAlertTask:
    name: str = "StockWorkingCapitalAlertTask"

    def run(self, ctx) -> None:
        enabled = str(os.getenv("STOCK_WORKING_CAPITAL_ALERT_ENABLED", "1")).strip().lower()
        if enabled in {"0", "false", "no", "off"}:
            ctx.log.info("[stock_working_capital_alert] skip by env STOCK_WORKING_CAPITAL_ALERT_ENABLED")
            return

        apply_ddl(ctx.engine, "docs/DDL/cn_market.cn_stock_working_capital_alert_v1.sql")

        summary_sql = """
            SELECT
                COUNT(*) AS total_rows,
                SUM(CASE WHEN working_capital_alert_level = 'high' THEN 1 ELSE 0 END) AS high_rows,
                SUM(CASE WHEN working_capital_alert_level = 'watch' THEN 1 ELSE 0 END) AS watch_rows
            FROM cn_stock_working_capital_alert_v1
        """
        with ctx.engine.connect() as conn:
            row = conn.execute(text(summary_sql)).fetchone()

        total_rows = int(row[0] or 0) if row else 0
        high_rows = int(row[1] or 0) if row else 0
        watch_rows = int(row[2] or 0) if row else 0
        ctx.log.info(
            "[stock_working_capital_alert] refreshed view=cn_stock_working_capital_alert_v1 total=%s high=%s watch=%s",
            total_rows,
            high_rows,
            watch_rows,
        )
