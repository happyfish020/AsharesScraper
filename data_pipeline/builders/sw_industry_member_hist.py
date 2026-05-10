from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime

import pandas as pd
from sqlalchemy import text

from data_pipeline.common.cli import add_shared_args, resolve_date_range
from data_pipeline.common.db import apply_sql_file, chunked_rows, fetch_df, get_engine
from data_pipeline.common.logging_utils import build_logger
from data_pipeline.common.state import BackfillState
from data_pipeline.tushare.client import TushareClient, resolve_tushare_token


def _to_date(value: object) -> date | None:
    if value in (None, "", "None"):
        return None
    dt = pd.to_datetime(value, format="%Y%m%d", errors="coerce")
    if pd.isna(dt):
        return None
    return dt.date()


def _normalize_symbol(value: object) -> str:
    text = str(value or "").strip()
    return text.split(".", 1)[0] if "." in text else text


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build cn_local_industry_map_hist from Tushare SW member history.")
    add_shared_args(parser)
    parser.add_argument("--src", default="SW2021", help="Industry source in cn_local_industry_master")
    parser.add_argument("--industry-level", default="L1", choices=["L1", "L2", "L3"], help="Industry level to backfill")
    return parser


def run(args: argparse.Namespace) -> None:
    date_range = resolve_date_range(args.start, args.end)
    logger = build_logger("build_sw_industry_member_hist")
    engine = get_engine()
    apply_sql_file(engine, "docs/DDL/ga_mainline_data_backfill_system.sql")
    state = BackfillState(engine=engine, job_name="build_sw_industry_member_hist")
    token = resolve_tushare_token(args.token, args.config)
    client = TushareClient(token=token, logger=logger)
    master = fetch_df(
        engine,
        """
        SELECT industry_id, industry_name, industry_level, src
        FROM cn_local_industry_master
        WHERE src = :src
          AND industry_level = :industry_level
        ORDER BY industry_id
        """,
        {"src": args.src, "industry_level": args.industry_level},
    )
    if master.empty:
        raise SystemExit("cn_local_industry_master is empty for the requested src/level. Run build_sw_industry_master.py first.")

    def worker(industry_id: str) -> list[dict]:
        frame = client.paginate(
            "index_member_all",
            {"ts_code": industry_id},
            "index_code,con_code,in_date,out_date,is_new",
            page_size=5000,
            key_prefix=f"index_member_all|industry_id={industry_id}",
        )
        rows: list[dict] = []
        for item in frame.to_dict(orient="records"):
            in_date = _to_date(item.get("in_date"))
            out_date = _to_date(item.get("out_date"))
            if in_date is None:
                continue
            if in_date > date_range.end:
                continue
            if out_date is not None and out_date < date_range.start:
                continue
            rows.append(
                {
                    "symbol": _normalize_symbol(item.get("con_code")),
                    "industry_id": industry_id,
                    "in_date": in_date,
                    "out_date": out_date,
                    "is_current": 1 if out_date is None or out_date >= date.today() else 0,
                }
            )
        return rows

    insert_sql = """
    INSERT INTO cn_local_industry_map_hist
        (symbol, industry_id, industry_name, industry_level, in_date, out_date, is_current)
    VALUES
        (:symbol, :industry_id, :industry_name, :industry_level, :in_date, :out_date, :is_current)
    ON DUPLICATE KEY UPDATE
        industry_name = VALUES(industry_name),
        industry_level = VALUES(industry_level),
        out_date = VALUES(out_date),
        is_current = VALUES(is_current),
        updated_at = CURRENT_TIMESTAMP
    """
    total_rows = 0
    lookup = {row.industry_id: row for row in master.itertuples(index=False)}
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        futures = {}
        for industry_id in lookup:
            chunk_key = f"{args.src}:{args.industry_level}:{industry_id}"
            if args.resume and state.is_completed(chunk_key):
                logger.info("resume_skip chunk=%s", chunk_key)
                continue
            state.start(chunk_key, date_range.start, date_range.end)
            futures[pool.submit(worker, industry_id)] = chunk_key
        for future in as_completed(futures):
            chunk_key = futures[future]
            industry_id = chunk_key.rsplit(":", 1)[-1]
            info = lookup[industry_id]
            try:
                raw_rows = future.result()
                rows = []
                for row in raw_rows:
                    row["industry_name"] = info.industry_name
                    row["industry_level"] = info.industry_level
                    rows.append(row)
                if rows:
                    with engine.begin() as conn:
                        for batch in chunked_rows(rows, 5000):
                            conn.execute(text(insert_sql), batch)
                total_rows += len(rows)
                state.complete(chunk_key, len(rows))
                logger.info("chunk_done chunk=%s rows=%s", chunk_key, len(rows))
            except Exception as exc:
                state.fail(chunk_key, exc)
                raise
    logger.info(
        "build_sw_industry_member_hist_done total_rows=%s level=%s start=%s end=%s finished_at=%s",
        total_rows,
        args.industry_level,
        date_range.start,
        date_range.end,
        datetime.now().isoformat(timespec="seconds"),
    )


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
