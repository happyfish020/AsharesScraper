from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, datetime
import time
import threading
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from app.tools.sync_cn_stock_daily_price_from_tushare import resolve_tushare_token, patch_pandas_fillna_method_compat
from app.utils.tushare_pro_client import build_tushare_pro_client

def _parse_yyyymmdd(s: str) -> date:
    return datetime.strptime(s, "%Y%m%d").date()


@dataclass
class BoardMembershipRefreshTask:
    name: str = "BoardMembershipRefreshTask"

    def _tushare_min_interval_seconds(self) -> float:
        return max(0.0, float(os.getenv("BOARD_TUSHARE_MIN_INTERVAL_SECONDS", "0.8")))

    def _respect_tushare_spacing(self, last_call_at: float | None) -> float:
        min_interval = self._tushare_min_interval_seconds()
        now = time.monotonic()
        if last_call_at is not None:
            elapsed = now - last_call_at
            remaining = min_interval - elapsed
            if remaining > 0:
                time.sleep(remaining)
        return time.monotonic()

    def _audit_dir(self, ctx) -> Path:
        return Path(getattr(ctx.config, "audit_reports_dir", "audit_reports"))

    @contextmanager
    def _heartbeat(self, ctx, label: str, interval_seconds: float | None = None):
        """Emit periodic progress logs while a long blocking SQL/API step is running."""
        interval = interval_seconds
        if interval is None:
            interval = max(10.0, float(os.getenv("BOARD_SQL_HEARTBEAT_SECONDS", "30")))
        stop_event = threading.Event()
        started = time.monotonic()

        def _worker() -> None:
            while not stop_event.wait(interval):
                elapsed = time.monotonic() - started
                ctx.log.info("[board_refresh] %s still running elapsed=%.1fs", label, elapsed)

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()
        try:
            yield
        finally:
            stop_event.set()
            thread.join(timeout=1.0)
            elapsed = time.monotonic() - started
            ctx.log.info("[board_refresh] %s finished elapsed=%.1fs", label, elapsed)

    def _execute_with_heartbeat(self, ctx, conn, sql_text: str, params: dict, label: str):
        ctx.log.info("[board_refresh] %s start params=%s", label, params)
        with self._heartbeat(ctx, label):
            result = conn.execute(text(sql_text), params)
        return result

    def _fetch_concept_detail_with_retry(
        self,
        pro,
        concept_id: str,
        ctx,
        last_call_at: float | None,
    ) -> tuple[pd.DataFrame | None, str | None, float | None]:
        max_attempts = max(1, int(os.getenv("BOARD_CONCEPT_DETAIL_MAX_ATTEMPTS", "3")))
        retry_sleep_seconds = max(1.0, float(os.getenv("BOARD_CONCEPT_DETAIL_RETRY_SLEEP_SECONDS", "3")))

        last_error: str | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                last_call_at = self._respect_tushare_spacing(last_call_at)
                detail = pro.concept_detail(id=concept_id)
                return detail, None, last_call_at
            except Exception as exc:
                last_error = str(exc)
                if attempt >= max_attempts:
                    break
                ctx.log.warning(
                    "[board_refresh] concept %s attempt %s/%s failed: %s; retry in %.1fs",
                    concept_id,
                    attempt,
                    max_attempts,
                    exc,
                    retry_sleep_seconds,
                )
                time.sleep(retry_sleep_seconds)
        return None, last_error, last_call_at

    def _write_failed_concepts_report(self, ctx, end: date, failed_rows: list[dict[str, str]]) -> None:
        audit_dir = self._audit_dir(ctx)
        audit_dir.mkdir(parents=True, exist_ok=True)
        report_path = audit_dir / f"board_refresh_failed_concepts_{end.strftime('%Y%m%d')}.csv"
        frame = pd.DataFrame(failed_rows, columns=["asof_date", "concept_id", "error"])
        frame.to_csv(report_path, index=False, encoding="utf-8-sig")
        ctx.log.warning("[board_refresh] wrote failed concept report: %s rows=%s", report_path, len(frame.index))

    def _merge_concept_staging_rows(self, ctx, end: date, details_frames: list[pd.DataFrame]) -> int:
        if not details_frames:
            return 0
        stg_df = pd.concat(details_frames, ignore_index=True)
        stg_df["asof_date"] = end
        stg_df["symbol"] = stg_df["ts_code"].astype(str).str[:6]
        stg_df["source"] = "tushare"
        stg_df = stg_df[["asof_date", "concept_id", "symbol", "source"]].drop_duplicates()
        records = stg_df.to_dict("records")
        if not records:
            return 0
        upsert_sql = """
            INSERT INTO cn_board_concept_member_stg (asof_date, concept_id, symbol, source)
            VALUES (:asof_date, :concept_id, :symbol, :source)
            ON DUPLICATE KEY UPDATE
                source = VALUES(source)
        """
        with ctx.engine.begin() as conn:
            conn.execute(text(upsert_sql), records)
        return len(records)

    def _merge_industry_staging_rows(self, ctx, end: date, member_frames: list[pd.DataFrame]) -> int:
        if not member_frames:
            return 0
        stg_df = pd.concat(member_frames, ignore_index=True)
        stg_df["asof_date"] = end
        stg_df["symbol"] = stg_df["con_code"].astype(str).str[:6]
        stg_df["source"] = "tushare"
        stg_df = stg_df[["asof_date", "board_id", "symbol", "source"]].drop_duplicates()
        records = stg_df.to_dict("records")
        if not records:
            return 0
        upsert_sql = """
            INSERT INTO cn_board_industry_member_stg (asof_date, board_id, symbol, source)
            VALUES (:asof_date, :board_id, :symbol, :source)
            ON DUPLICATE KEY UPDATE
                source = VALUES(source)
        """
        with ctx.engine.begin() as conn:
            conn.execute(text(upsert_sql), records)
        return len(records)

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
            pro_timeout = max(5.0, float(os.getenv("BOARD_TUSHARE_TIMEOUT_SECONDS", os.getenv("TUSHARE_PRO_TIMEOUT_SECONDS", "30"))))
            pro = build_tushare_pro_client(token, timeout=pro_timeout)
            last_call_at: float | None = None
            try:
                if apply_concept:
                    ctx.log.info(f"[board_refresh] fetching Tushare concepts staging start -  {start}")
                    ctx.log.info(f"[board_refresh] fetching Tushare concepts staging end {end}")
                    last_call_at = self._respect_tushare_spacing(last_call_at)
                    concepts = pro.concept(src='ts')
                    all_details = []
                    failed_concepts: list[dict[str, str]] = []
                    if concepts is not None and not concepts.empty:
                        concept_ids = [str(v).strip() for v in concepts["code"].tolist() if str(v).strip()]
                        ctx.log.info("[board_refresh] concept universe loaded count=%s timeout=%.1fs base_url=%s", len(concept_ids), pro_timeout, os.getenv("TUSHARE_PRO_BASE_URL", "http://api.waditu.com/dataapi"))
                        progress_every = max(1, int(os.getenv("BOARD_CONCEPT_PROGRESS_EVERY", "50")))
                        flush_every = max(1, int(os.getenv("BOARD_CONCEPT_FLUSH_EVERY", "100")))
                        merged_concept_rows = 0
                        second_pass_ids: list[str] = []
                        for idx, cid in enumerate(concept_ids, start=1):
                            try:
                                detail, error, last_call_at = self._fetch_concept_detail_with_retry(pro, cid, ctx, last_call_at)
                                if detail is not None and not detail.empty:
                                    detail['concept_id'] = cid
                                    all_details.append(detail[['concept_id', 'ts_code']])
                                elif error:
                                    second_pass_ids.append(cid)
                            except Exception as e:
                                second_pass_ids.append(cid)
                                ctx.log.warning(f"[board_refresh] failed to fetch detail for concept {cid}: {e}")
                            if idx % progress_every == 0 or idx == len(concept_ids):
                                ctx.log.info(
                                    "[board_refresh] concept detail progress %s/%s buffered=%s retry_queue=%s merged_rows=%s",
                                    idx,
                                    len(concept_ids),
                                    len(all_details),
                                    len(second_pass_ids),
                                    merged_concept_rows,
                                )
                            if len(all_details) >= flush_every:
                                merged_now = self._merge_concept_staging_rows(ctx, end, all_details)
                                merged_concept_rows += merged_now
                                ctx.log.info(
                                    "[board_refresh] concept staging flushed rows=%s merged_total=%s",
                                    merged_now,
                                    merged_concept_rows,
                                )
                                all_details = []

                        if second_pass_ids:
                            retry_round_sleep = max(1.0, float(os.getenv("BOARD_CONCEPT_DETAIL_SECOND_PASS_SLEEP_SECONDS", "10")))
                            ctx.log.warning(
                                "[board_refresh] concept first pass incomplete, second pass retry concepts=%s sleep=%.1fs",
                                len(second_pass_ids),
                                retry_round_sleep,
                            )
                            time.sleep(retry_round_sleep)
                            for idx, cid in enumerate(second_pass_ids, start=1):
                                detail, error, last_call_at = self._fetch_concept_detail_with_retry(pro, cid, ctx, last_call_at)
                                if detail is not None and not detail.empty:
                                    detail["concept_id"] = cid
                                    all_details.append(detail[["concept_id", "ts_code"]])
                                else:
                                    failed_concepts.append(
                                        {
                                            "asof_date": end.strftime("%Y-%m-%d"),
                                            "concept_id": cid,
                                            "error": error or "unknown_error",
                                        }
                                    )
                                if idx % progress_every == 0 or idx == len(second_pass_ids):
                                    ctx.log.info(
                                        "[board_refresh] concept second-pass progress %s/%s buffered=%s failed=%s merged_rows=%s",
                                        idx,
                                        len(second_pass_ids),
                                        len(all_details),
                                        len(failed_concepts),
                                        merged_concept_rows,
                                    )
                                if len(all_details) >= flush_every:
                                    merged_now = self._merge_concept_staging_rows(ctx, end, all_details)
                                    merged_concept_rows += merged_now
                                    ctx.log.info(
                                        "[board_refresh] concept staging flushed rows=%s merged_total=%s",
                                        merged_now,
                                        merged_concept_rows,
                                    )
                                    all_details = []

                    if all_details:
                        merged_now = self._merge_concept_staging_rows(ctx, end, all_details)
                        ctx.log.info("[board_refresh] concept staging final flush rows=%s", merged_now)

                    if failed_concepts:
                        self._write_failed_concepts_report(ctx, end, failed_concepts)
                        if str(os.getenv("BOARD_CONCEPT_DETAIL_FAIL_ON_REMAINING", "0")).strip().lower() in {"1", "true", "yes", "on"}:
                            sample_ids = ",".join(row["concept_id"] for row in failed_concepts[:10])
                            raise RuntimeError(
                                f"[board_refresh] concept detail still failed after retries count={len(failed_concepts)} sample={sample_ids}"
                            )

                if apply_industry:
                    ctx.log.info(f"[board_refresh] fetching Tushare industries (SW L3) staging for {end}")
                    last_call_at = self._respect_tushare_spacing(last_call_at)
                    industries = pro.index_classify(level='L3', src='SW2021')
                    all_m = []
                    if industries is not None and not industries.empty:
                        progress_every = max(1, int(os.getenv("BOARD_INDUSTRY_PROGRESS_EVERY", "50")))
                        flush_every = max(1, int(os.getenv("BOARD_INDUSTRY_FLUSH_EVERY", "100")))
                        merged_industry_rows = 0
                        total_industries = len(industries.index)
                        for idx, (_, row) in enumerate(industries.iterrows(), start=1):
                            try:
                                icode = row['index_code']
                                last_call_at = self._respect_tushare_spacing(last_call_at)
                                m = pro.index_member(index_code=icode)
                                if m is not None and not m.empty:
                                    m['board_id'] = icode
                                    all_m.append(m[['board_id', 'con_code']])
                            except Exception as e:
                                ctx.log.warning(f"[board_refresh] failed to fetch members for industry {icode}: {e}")
                            if idx % progress_every == 0 or idx == total_industries:
                                ctx.log.info(
                                    "[board_refresh] industry member progress %s/%s buffered=%s merged_rows=%s",
                                    idx,
                                    total_industries,
                                    len(all_m),
                                    merged_industry_rows,
                                )
                            if len(all_m) >= flush_every:
                                merged_now = self._merge_industry_staging_rows(ctx, end, all_m)
                                merged_industry_rows += merged_now
                                ctx.log.info(
                                    "[board_refresh] industry staging flushed rows=%s merged_total=%s",
                                    merged_now,
                                    merged_industry_rows,
                                )
                                all_m = []

                    if all_m:
                        merged_now = self._merge_industry_staging_rows(ctx, end, all_m)
                        ctx.log.info("[board_refresh] industry staging final flush rows=%s", merged_now)
            finally:
                if hasattr(pro, "close"):
                    pro.close()

        ctx.log.info(
            "[board_refresh] start=%s end=%s source=%s apply_concept=%s apply_industry=%s",
            start,
            end,
            source,
            apply_concept,
            apply_industry,
        )

        with ctx.engine.begin() as conn:
            self._execute_with_heartbeat(
                ctx,
                conn,
                "CALL sp_refresh_board_member_hist(:asof, :src, :ac, :ai)",
                {
                    "asof": end,
                    "src": source,
                    "ac": apply_concept,
                    "ai": apply_industry,
                },
                "CALL sp_refresh_board_member_hist",
            )
            self._execute_with_heartbeat(
                ctx,
                conn,
                "CALL sp_build_board_member_map(:d1, :d2)",
                {
                    "d1": start,
                    "d2": end,
                },
                "CALL sp_build_board_member_map",
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
