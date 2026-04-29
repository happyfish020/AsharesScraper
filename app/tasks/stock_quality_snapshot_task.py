from __future__ import annotations

import os
from dataclasses import dataclass

from app.tools.sync_cn_stock_fundamental_monthly import (
    apply_ddl,
    rebuild_quality_snapshot,
)


@dataclass
class StockQualitySnapshotTask:
    name: str = "StockQualitySnapshotTask"

    def run(self, ctx) -> None:
        enabled = str(os.getenv("STOCK_QUALITY_SNAPSHOT_ENABLED", "1")).strip().lower()
        if enabled in {"0", "false", "no", "off"}:
            ctx.log.info("[stock_quality_snapshot] skip by env STOCK_QUALITY_SNAPSHOT_ENABLED")
            return
        include_payload = str(os.getenv("STOCK_QUALITY_SNAPSHOT_INCLUDE_PAYLOAD", "0")).strip().lower() not in {
            "0",
            "false",
            "no",
            "off",
        }

        apply_ddl(ctx.engine, "docs/DDL/cn_market.cn_stock_fundamental_quality_v1.sql")
        apply_ddl(ctx.engine, "docs/DDL/cn_market.cn_stock_fundamental_quality_hist_v1.sql")
        rows = rebuild_quality_snapshot(ctx.engine, log=ctx.log, include_payload=include_payload)
        ctx.log.info(
            "[stock_quality_snapshot] rebuilt cn_stock_fundamental_quality_snap rows=%s mode=monthly_batch payload=%s",
            rows,
            "on" if include_payload else "off",
        )
