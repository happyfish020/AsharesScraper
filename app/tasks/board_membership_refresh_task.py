from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, datetime

from sqlalchemy import text


def _parse_yyyymmdd(s: str) -> date:
    return datetime.strptime(s, "%Y%m%d").date()


@dataclass
class BoardMembershipRefreshTask:
    name: str = "BoardMembershipRefreshTask"

    def run(self, ctx) -> None:
        cfg = ctx.config
        cfg.finalize_dates()

        start = _parse_yyyymmdd(str(cfg.start_date))
        end = _parse_yyyymmdd(str(cfg.end_date))
        source = os.getenv("BOARD_MEMBERSHIP_SOURCE", "tushare").strip() or "tushare"
        apply_concept = int(os.getenv("BOARD_APPLY_CONCEPT", "1"))
        apply_industry = int(os.getenv("BOARD_APPLY_INDUSTRY", "1"))

        if str(getattr(ctx.engine.dialect, "name", "")).lower() != "mysql":
            raise RuntimeError("[board_refresh] task='board' must run on MySQL.")

        ctx.log.info(
            "[board_refresh] start=%s end=%s source=%s apply_concept=%s apply_industry=%s",
            start,
            end,
            source,
            apply_concept,
            apply_industry,
        )

        with ctx.engine.begin() as conn:
            conn.execute(
                text("CALL sp_refresh_board_member_hist(:asof, :src, :ac, :ai)"),
                {
                    "asof": end,
                    "src": source,
                    "ac": apply_concept,
                    "ai": apply_industry,
                },
            )
            conn.execute(
                text("CALL sp_build_board_member_map(:d1, :d2)"),
                {
                    "d1": start,
                    "d2": end,
                },
            )

            stg_concept = conn.execute(
                text("SELECT COUNT(*) FROM cn_board_concept_member_stg WHERE asof_date=:d"),
                {"d": end},
            ).scalar() or 0
            stg_industry = conn.execute(
                text("SELECT COUNT(*) FROM cn_board_industry_member_stg WHERE asof_date=:d"),
                {"d": end},
            ).scalar() or 0
            map_rows = conn.execute(
                text(
                    """
                    SELECT COUNT(*)
                    FROM cn_board_member_map_d
                    WHERE trade_date BETWEEN :d1 AND :d2
                    """
                ),
                {"d1": start, "d2": end},
            ).scalar() or 0

        ctx.log.info(
            "[board_refresh] done asof=%s stg_concept=%s stg_industry=%s map_rows_in_range=%s",
            end,
            int(stg_concept),
            int(stg_industry),
            int(map_rows),
        )
