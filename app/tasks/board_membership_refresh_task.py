from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, datetime
import time

import pandas as pd
import tushare as ts
from sqlalchemy import text

from app.tools.sync_cn_stock_daily_price_from_tushare import resolve_tushare_token, patch_pandas_fillna_method_compat

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

        # Step A: Load external snapshot into staging before refreshing history
        if source == "tushare":
            patch_pandas_fillna_method_compat()
            token, _ = resolve_tushare_token("", "")
            if not token:
                raise RuntimeError("Tushare token is required for board membership staging load")
            pro = ts.pro_api(token)

            if apply_concept:
                ctx.log.info(f"[board_refresh] fetching Tushare concepts staging start -  {start}")
                ctx.log.info(f"[board_refresh] fetching Tushare concepts staging end {end}")
                concepts = pro.concept(src='ts')
                all_details = []
                if concepts is not None and not concepts.empty:
                    for _, row in concepts.iterrows():
                        try:
                            cid = row['code']
                            detail = pro.concept_detail(id=cid)
                            if detail is not None and not detail.empty:
                                detail['concept_id'] = cid
                                all_details.append(detail[['concept_id', 'ts_code']])
                            time.sleep(0.3)  # Respect rate limits
                        except Exception as e:
                            ctx.log.warning(f"[board_refresh] failed to fetch detail for concept {cid}: {e}")

                if all_details:
                    stg_df = pd.concat(all_details)
                    stg_df['asof_date'] = end
                    stg_df['symbol'] = stg_df['ts_code'].str[:6]
                    stg_df['source'] = 'tushare'

                    # Deduplicate to prevent IntegrityError (1062) and implement "Update or Insert"
                    stg_df = stg_df[['asof_date', 'concept_id', 'symbol', 'source']].drop_duplicates()
                    records = stg_df.to_dict('records')

                    upsert_sql = """
                        INSERT INTO cn_board_concept_member_stg (asof_date, concept_id, symbol, source)
                        VALUES (:asof_date, :concept_id, :symbol, :source)
                        ON DUPLICATE KEY UPDATE
                            source = VALUES(source)
                    """
                    with ctx.engine.begin() as conn:
                        conn.execute(text(upsert_sql), records)

                    ctx.log.info(f"[board_refresh] merged {len(records)} concept staging rows")

            if apply_industry:
                ctx.log.info(f"[board_refresh] fetching Tushare industries (SW L3) staging for {end}")
                industries = pro.index_classify(level='L3', src='SW2021')
                all_m = []
                if industries is not None and not industries.empty:
                    for _, row in industries.iterrows():
                        try:
                            icode = row['index_code']
                            m = pro.index_member(index_code=icode)
                            if m is not None and not m.empty:
                                m['board_id'] = icode
                                all_m.append(m[['board_id', 'con_code']])
                            time.sleep(0.3)  # Respect rate limits
                        except Exception as e:
                            ctx.log.warning(f"[board_refresh] failed to fetch members for industry {icode}: {e}")

                if all_m:
                    stg_df = pd.concat(all_m)
                    stg_df['asof_date'] = end
                    stg_df['symbol'] = stg_df['con_code'].str[:6]
                    stg_df['source'] = 'tushare'

                    # Deduplicate to prevent IntegrityError (1062) and implement "Update or Insert"
                    stg_df = stg_df[['asof_date', 'board_id', 'symbol', 'source']].drop_duplicates()
                    records = stg_df.to_dict('records')

                    upsert_sql = """
                        INSERT INTO cn_board_industry_member_stg (asof_date, board_id, symbol, source)
                        VALUES (:asof_date, :board_id, :symbol, :source)
                        ON DUPLICATE KEY UPDATE
                            source = VALUES(source)
                    """
                    with ctx.engine.begin() as conn:
                        conn.execute(text(upsert_sql), records)

                    ctx.log.info(f"[board_refresh] merged {len(records)} industry staging rows")

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
