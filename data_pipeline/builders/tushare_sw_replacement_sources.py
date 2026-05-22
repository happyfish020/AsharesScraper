from __future__ import annotations

import argparse
import json
from datetime import date, datetime
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from data_pipeline.common.cli import add_shared_args, resolve_date_range
from data_pipeline.common.db import apply_sql_file, chunked_rows, get_engine
from data_pipeline.common.logging_utils import build_logger
from data_pipeline.common.state import BackfillState
from data_pipeline.tushare.client import TushareClient, resolve_tushare_token


LEVEL_TO_MEMBER_PARAM = {
    "L1": "l1_code",
    "L2": "l2_code",
    "L3": "l3_code",
}


def _normalize_symbol(value: object) -> str:
    text_value = str(value or "").strip()
    return text_value.split(".", 1)[0] if "." in text_value else text_value


def _to_date(value: object) -> date | None:
    if value in (None, "", "None", "nan", "NaT"):
        return None
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return None
    return ts.date()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build official Tushare SW replacement source tables for V8 replacement work."
    )
    add_shared_args(parser, default_start="2023-01-01")
    parser.add_argument("--srcs", nargs="+", default=["SW2021", "SW2014"], help="Shenwan classification sources")
    parser.add_argument("--levels", nargs="+", default=["L1"], choices=["L1", "L2", "L3"], help="Shenwan levels to fetch")
    parser.add_argument("--output-dir", default="reports/analysis/tushare_sw_replacement_sources", help="Output directory")
    parser.add_argument("--replace-members", action="store_true", help="Delete existing member rows for the requested src/level/date overlap before insert")
    parser.add_argument("--replace-master", action="store_true", help="Delete existing master rows for the requested src/level before insert")
    return parser


def _load_master_rows(client: TushareClient, src: str, level: str, asof_date: date) -> list[dict]:
    frame = client.call(
        "index_classify",
        {"src": src, "level": level},
        "index_code,industry_name,level,parent_code,is_pub,src",
        cache_key=f"index_classify|src={src}|level={level}",
    )
    rows: list[dict] = []
    for item in frame.to_dict(orient="records"):
        industry_id = str(item.get("index_code") or "").strip()
        if not industry_id:
            continue
        rows.append(
            {
                "src": str(item.get("src") or src).strip(),
                "industry_level": str(item.get("level") or level).strip(),
                "industry_id": industry_id,
                "industry_name": str(item.get("industry_name") or industry_id).strip(),
                "parent_id": str(item.get("parent_code") or "").strip() or None,
                "is_pub": str(item.get("is_pub") or "").strip() or None,
                "provider": "TUSHARE",
                "asof_date": asof_date,
                "raw_json": json.dumps(item, ensure_ascii=False, separators=(",", ":")),
            }
        )
    if not rows:
        raise SystemExit(f"index_classify returned no rows for src={src} level={level}.")
    return rows


def _load_member_rows(
    client: TushareClient,
    src: str,
    level: str,
    industry_id: str,
    industry_name: str,
    start: date,
    end: date,
) -> list[dict]:
    param_name = LEVEL_TO_MEMBER_PARAM[level]
    frame = client.paginate(
        "index_member_all",
        {param_name: industry_id},
        "",
        page_size=5000,
        key_prefix=f"index_member_all|{param_name}={industry_id}|src={src}|level={level}",
    )
    rows: list[dict] = []
    for item in frame.to_dict(orient="records"):
        in_date = _to_date(item.get("in_date"))
        out_date = _to_date(item.get("out_date"))
        if in_date is None:
            continue
        if in_date > end:
            continue
        if out_date is not None and out_date < start:
            continue
        rows.append(
            {
                "src": src,
                "industry_level": level,
                "industry_id": industry_id,
                "industry_name": industry_name,
                "symbol": _normalize_symbol(item.get("ts_code")),
                "ts_code": str(item.get("ts_code") or "").strip() or None,
                "stock_name": str(item.get("name") or "").strip() or None,
                "in_date": in_date,
                "out_date": out_date,
                "is_new": str(item.get("is_new") or "").strip() or None,
                "provider": "TUSHARE",
            }
        )
    return rows


def _write_master_rows(engine, rows: list[dict], replace_master: bool) -> int:
    if not rows:
        return 0
    if replace_master:
        by_scope = {(row["src"], row["industry_level"]) for row in rows}
        with engine.begin() as conn:
            for src, level in by_scope:
                conn.execute(
                    text(
                        """
                        DELETE FROM cn_ts_sw_industry_master
                        WHERE src = :src
                          AND industry_level = :industry_level
                        """
                    ),
                    {"src": src, "industry_level": level},
                )
    sql = """
    INSERT INTO cn_ts_sw_industry_master
        (src, industry_level, industry_id, industry_name, parent_id, is_pub, provider, asof_date, raw_json)
    VALUES
        (:src, :industry_level, :industry_id, :industry_name, :parent_id, :is_pub, :provider, :asof_date, :raw_json)
    ON DUPLICATE KEY UPDATE
        industry_name = VALUES(industry_name),
        parent_id = VALUES(parent_id),
        is_pub = VALUES(is_pub),
        provider = VALUES(provider),
        asof_date = VALUES(asof_date),
        raw_json = VALUES(raw_json),
        updated_at = CURRENT_TIMESTAMP
    """
    affected = 0
    with engine.begin() as conn:
        for batch in chunked_rows(rows, 500):
            result = conn.execute(text(sql), batch)
            affected += int(result.rowcount or 0)
    return affected


def _write_member_rows(engine, rows: list[dict], start: date, end: date, replace_members: bool) -> int:
    if not rows:
        return 0
    if replace_members:
        by_scope = {(row["src"], row["industry_level"]) for row in rows}
        with engine.begin() as conn:
            for src, level in by_scope:
                conn.execute(
                    text(
                        """
                        DELETE FROM cn_ts_sw_industry_member_hist
                        WHERE src = :src
                          AND industry_level = :industry_level
                          AND in_date <= :end
                          AND (out_date IS NULL OR out_date >= :start)
                        """
                    ),
                    {"src": src, "industry_level": level, "start": start, "end": end},
                )
    sql = """
    INSERT INTO cn_ts_sw_industry_member_hist
        (src, industry_level, industry_id, industry_name, symbol, ts_code, stock_name, in_date, out_date, is_new, provider)
    VALUES
        (:src, :industry_level, :industry_id, :industry_name, :symbol, :ts_code, :stock_name, :in_date, :out_date, :is_new, :provider)
    ON DUPLICATE KEY UPDATE
        industry_name = VALUES(industry_name),
        ts_code = VALUES(ts_code),
        stock_name = VALUES(stock_name),
        out_date = VALUES(out_date),
        is_new = VALUES(is_new),
        provider = VALUES(provider),
        updated_at = CURRENT_TIMESTAMP
    """
    affected = 0
    with engine.begin() as conn:
        for batch in chunked_rows(rows, 2000):
            result = conn.execute(text(sql), batch)
            affected += int(result.rowcount or 0)
    return affected


def _write_summary(output_dir: Path, summary_rows: list[dict], args: argparse.Namespace) -> None:
    frame = pd.DataFrame(summary_rows)
    frame.to_csv(output_dir / "tushare_sw_replacement_sources_summary.csv", index=False, encoding="utf-8-sig")
    lines = [
        "# Tushare SW Replacement Sources",
        "",
        f"- start: `{args.start}`",
        f"- end: `{args.end or 'today'}`",
        f"- srcs: `{', '.join(args.srcs)}`",
        f"- levels: `{', '.join(args.levels)}`",
        "",
        "## Summary",
        "",
    ]
    for row in summary_rows:
        lines.append(
            f"- `{row['src']}` / `{row['industry_level']}`: master_rows=`{row['master_rows']}`, "
            f"member_rows=`{row['member_rows']}`, industries=`{row['industry_count']}`"
        )
    (output_dir / "tushare_sw_replacement_sources_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    date_range = resolve_date_range(args.start, args.end)
    logger = build_logger("build_tushare_sw_replacement_sources")
    engine = get_engine()
    apply_sql_file(engine, "docs/DDL/cn_market.cn_ts_sw_industry_master.sql")
    apply_sql_file(engine, "docs/DDL/cn_market.cn_ts_sw_industry_member_hist.sql")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    state = BackfillState(engine=engine, job_name="build_tushare_sw_replacement_sources")
    token = resolve_tushare_token(args.token, args.config)
    client = TushareClient(token=token, logger=logger)

    summary_rows: list[dict] = []
    asof_date = date_range.end

    for src in args.srcs:
        for level in args.levels:
            chunk_key = f"{src}:{level}:{date_range.start}:{date_range.end}"
            if args.resume and state.is_completed(chunk_key):
                logger.info("resume_skip chunk=%s", chunk_key)
                continue
            state.start(chunk_key, date_range.start, date_range.end)
            try:
                master_rows = _load_master_rows(client, src, level, asof_date)
                master_affected = _write_master_rows(engine, master_rows, replace_master=args.replace_master)
                member_rows: list[dict] = []
                for row in master_rows:
                    member_rows.extend(
                        _load_member_rows(
                            client=client,
                            src=src,
                            level=level,
                            industry_id=row["industry_id"],
                            industry_name=row["industry_name"],
                            start=date_range.start,
                            end=date_range.end,
                        )
                    )
                member_affected = _write_member_rows(
                    engine,
                    member_rows,
                    start=date_range.start,
                    end=date_range.end,
                    replace_members=args.replace_members,
                )
                summary_rows.append(
                    {
                        "src": src,
                        "industry_level": level,
                        "master_rows": len(master_rows),
                        "master_affected": master_affected,
                        "member_rows": len(member_rows),
                        "member_affected": member_affected,
                        "industry_count": len({row["industry_id"] for row in master_rows}),
                    }
                )
                state.complete(chunk_key, len(member_rows))
                logger.info(
                    "chunk_done chunk=%s master_rows=%s member_rows=%s",
                    chunk_key,
                    len(master_rows),
                    len(member_rows),
                )
            except Exception as exc:
                state.fail(chunk_key, exc)
                raise

    _write_summary(output_dir, summary_rows, args)
    logger.info("build_tushare_sw_replacement_sources_done output_dir=%s", output_dir)


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
