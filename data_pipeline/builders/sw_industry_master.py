from __future__ import annotations

import argparse
from datetime import date

from sqlalchemy import text

from data_pipeline.common.cli import add_shared_args, resolve_date_range
from data_pipeline.common.db import apply_sql_file, chunked_rows, get_engine
from data_pipeline.common.logging_utils import build_logger
from data_pipeline.common.state import BackfillState
from data_pipeline.tushare.client import TushareClient, resolve_tushare_token


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build cn_local_industry_master from Tushare SW2021 classify.")
    add_shared_args(parser)
    parser.add_argument("--src", default="SW2021", help="Tushare index_classify src")
    return parser


def run(args: argparse.Namespace) -> None:
    date_range = resolve_date_range(args.start, args.end)
    logger = build_logger("build_sw_industry_master")
    engine = get_engine()
    apply_sql_file(engine, "docs/DDL/ga_mainline_data_backfill_system.sql")
    state = BackfillState(engine=engine, job_name="build_sw_industry_master")
    token = resolve_tushare_token(args.token, args.config)
    client = TushareClient(token=token, logger=logger)

    all_rows: list[dict] = []
    for level in ("L1", "L2", "L3"):
        chunk_key = f"{args.src}:{level}"
        if args.resume and state.is_completed(chunk_key):
            logger.info("resume_skip chunk=%s", chunk_key)
            continue
        state.start(chunk_key, date_range.start, date_range.end)
        try:
            frame = client.call(
                "index_classify",
                {"src": args.src, "level": level},
                "index_code,industry_name,level,parent_code,src",
                cache_key=f"index_classify|src={args.src}|level={level}",
            )
            rows = []
            for item in frame.to_dict(orient="records"):
                industry_id = str(item.get("index_code") or "").strip()
                if not industry_id:
                    continue
                rows.append(
                    {
                        "industry_id": industry_id,
                        "industry_name": str(item.get("industry_name") or industry_id).strip(),
                        "industry_level": str(item.get("level") or level).strip(),
                        "parent_id": str(item.get("parent_code") or "").strip() or None,
                        "src": str(item.get("src") or args.src).strip(),
                    }
                )
            sql = """
            INSERT INTO cn_local_industry_master
                (industry_id, industry_name, industry_level, parent_id, src)
            VALUES
                (:industry_id, :industry_name, :industry_level, :parent_id, :src)
            ON DUPLICATE KEY UPDATE
                industry_name = VALUES(industry_name),
                industry_level = VALUES(industry_level),
                parent_id = VALUES(parent_id),
                src = VALUES(src)
            """
            with engine.begin() as conn:
                for batch in chunked_rows(rows, 500):
                    conn.execute(text(sql), batch)
            state.complete(chunk_key, len(rows))
            all_rows.extend(rows)
            logger.info("chunk_done chunk=%s rows=%s", chunk_key, len(rows))
        except Exception as exc:
            state.fail(chunk_key, exc)
            raise
    logger.info("build_sw_industry_master_done total_rows=%s date_start=%s date_end=%s", len(all_rows), date_range.start, date_range.end)


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
