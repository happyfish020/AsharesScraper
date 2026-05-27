from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from data_pipeline.common.cli import add_shared_args, resolve_date_range
from data_pipeline.common.db import apply_sql_file, chunked_rows, fetch_df, get_engine
from data_pipeline.common.logging_utils import build_logger
from data_pipeline.tushare.client import TushareClient, resolve_tushare_token


# Compatibility filter used by the current V7/V8 crosswalk builder.
# This builder is not the V8 production source of truth; it only maps the
# subset of LOCAL_FINE industries that participate in the legacy crosswalk.
V8_LOCAL_PATTERN = "85%.SI"
BENCHMARK_INDEX_CODES = ("sh000300", "sh000001", "sz399001")


@dataclass(slots=True)
class SwIndustryInfo:
    industry_id: str
    industry_name: str
    industry_level: str
    parent_id: str | None
    parent_name: str | None
    src: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build V7/V8 compatibility crosswalk from the V8 LOCAL_FINE local "
            "industry layer and Tushare SW L1 memberships."
        )
    )
    add_shared_args(parser, default_start="2023-01-01")
    parser.add_argument("--output-dir", default="reports/analysis/sw_v7_v8_crosswalk", help="Output directory root")
    parser.add_argument("--srcs", nargs="+", default=["SW2021", "SW2014"], help="Tushare SW classify sources to compare")
    parser.add_argument("--top-k", type=int, default=3, help="Candidate rows to keep per V8 code/date/src")
    parser.add_argument("--replace", action="store_true", help="Delete existing crosswalk rows in the target date range before insert")
    parser.add_argument(
        "--v8-local-scope",
        choices=["compat", "full"],
        default="compat",
        help=(
            "Which V8 LOCAL_FINE universe to compare against SW L1. "
            "'compat' keeps the legacy 85%%.SI subset; 'full' uses all LOCAL_FINE "
            "rows from map_hist L3."
        ),
    )
    parser.add_argument(
        "--allow-full-db-write",
        action="store_true",
        help=(
            "Allow --v8-local-scope full results to be written into "
            "cn_v7_v8_industry_crosswalk. By default full-scope runs are "
            "report-only to avoid mixing compatibility and full-universe semantics "
            "in the same history table."
        ),
    )
    parser.add_argument(
        "--source-mode",
        choices=["db", "tushare"],
        default="db",
        help="Read official SW source from local cn_ts_sw_* tables or directly from Tushare",
    )
    return parser


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


def _fetch_effective_dates(engine, start: date, end: date) -> list[date]:
    for index_code in BENCHMARK_INDEX_CODES:
        frame = fetch_df(
            engine,
            """
            SELECT MAX(trade_date) AS effective_date
            FROM cn_index_daily_price
            WHERE index_code = :index_code
              AND trade_date BETWEEN :start AND :end
            GROUP BY YEAR(trade_date), MONTH(trade_date)
            ORDER BY effective_date
            """,
            {"index_code": index_code, "start": start, "end": end},
        )
        if not frame.empty:
            return [_to_date(value) for value in frame["effective_date"].tolist() if _to_date(value) is not None]
    raise SystemExit("No usable monthly trading dates found in cn_index_daily_price for the requested range.")


def _scope_filter(scope: str) -> tuple[str, str]:
    if scope == "full":
        return "", "full LOCAL_FINE universe"
    return "AND industry_id LIKE :industry_pattern", "compatibility subset (85%.SI)"


def _fetch_v8_local_rows(engine, start: date, end: date, scope: str) -> pd.DataFrame:
    """Load V8 local-industry memberships for the compatibility crosswalk.

    Under the current V8 semantic contract:
    - `cn_local_industry_map_hist.industry_level = 'L3'` is the LOCAL_FINE
      391-industry production layer
    - this builder can consume either the legacy compatibility subset or the
      full LOCAL_FINE universe
    """
    extra_filter, scope_label = _scope_filter(scope)
    frame = fetch_df(
        engine,
        f"""
        SELECT
            symbol,
            industry_id,
            industry_name,
            industry_level,
            in_date,
            out_date
        FROM cn_local_industry_map_hist
        WHERE industry_level = 'L3'
          {extra_filter}
          AND in_date <= :end
          AND (out_date IS NULL OR out_date >= :start)
        ORDER BY industry_id, symbol, in_date
        """,
        {"industry_pattern": V8_LOCAL_PATTERN, "start": start, "end": end},
    )
    if frame.empty:
        raise SystemExit(
            "No V8 LOCAL_FINE rows found in cn_local_industry_map_hist "
            f"for scope={scope} ({scope_label}) in the requested range."
        )
    frame["symbol"] = frame["symbol"].map(_normalize_symbol)
    frame["in_date"] = pd.to_datetime(frame["in_date"], errors="coerce").dt.date
    frame["out_date"] = pd.to_datetime(frame["out_date"], errors="coerce").dt.date
    return frame


def _fetch_sw_l1_master(client: TushareClient, src: str) -> dict[str, SwIndustryInfo]:
    frame = client.call(
        "index_classify",
        {"src": src, "level": "L1"},
        "index_code,industry_name,level,parent_code,src",
        cache_key=f"index_classify|src={src}|level=L1",
    )
    if frame.empty:
        raise SystemExit(f"Tushare index_classify returned no L1 rows for src={src}.")
    return {
        str(row.get("index_code") or "").strip(): SwIndustryInfo(
            industry_id=str(row.get("index_code") or "").strip(),
            industry_name=str(row.get("industry_name") or "").strip(),
            industry_level=str(row.get("level") or "L1").strip(),
            parent_id=str(row.get("parent_code") or "").strip() or None,
            parent_name=None,
            src=str(row.get("src") or src).strip(),
        )
        for row in frame.to_dict(orient="records")
        if str(row.get("index_code") or "").strip()
    }


def _fetch_sw_l1_master_from_db(engine, src: str) -> dict[str, SwIndustryInfo]:
    frame = fetch_df(
        engine,
        """
        SELECT src, industry_level, industry_id, industry_name, parent_id
        FROM cn_ts_sw_industry_master
        WHERE src = :src
          AND industry_level = 'L1'
        ORDER BY industry_id
        """,
        {"src": src},
    )
    if frame.empty:
        raise SystemExit(f"cn_ts_sw_industry_master has no L1 rows for src={src}. Run build_tushare_sw_replacement_sources.py first.")
    return {
        str(row.get("industry_id") or "").strip(): SwIndustryInfo(
            industry_id=str(row.get("industry_id") or "").strip(),
            industry_name=str(row.get("industry_name") or "").strip(),
            industry_level=str(row.get("industry_level") or "L1").strip(),
            parent_id=str(row.get("parent_id") or "").strip() or None,
            parent_name=None,
            src=str(row.get("src") or src).strip(),
        )
        for row in frame.to_dict(orient="records")
        if str(row.get("industry_id") or "").strip()
    }


def _fetch_sw_memberships(
    client: TushareClient,
    src: str,
    master: dict[str, SwIndustryInfo],
    start: date,
    end: date,
) -> pd.DataFrame:
    rows: list[dict] = []
    for industry_id, info in master.items():
        frame = client.paginate(
            "index_member_all",
            {"l1_code": industry_id},
            "",
            page_size=5000,
            key_prefix=f"index_member_all|mode=l1_code|src={src}|industry_id={industry_id}",
        )
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
                    "industry_id": industry_id,
                    "industry_name": info.industry_name,
                    "industry_level": info.industry_level,
                    "parent_id": info.parent_id,
                    "parent_name": info.parent_name,
                    "symbol": _normalize_symbol(item.get("ts_code")),
                    "in_date": in_date,
                    "out_date": out_date,
                }
            )
    if not rows:
        raise SystemExit(f"No Tushare index_member_all rows found for src={src} in the requested range.")
    return pd.DataFrame(rows)


def _fetch_sw_memberships_from_db(engine, src: str, start: date, end: date) -> pd.DataFrame:
    frame = fetch_df(
        engine,
        """
        SELECT
            src,
            industry_id,
            industry_name,
            industry_level,
            symbol,
            in_date,
            out_date
        FROM cn_ts_sw_industry_member_hist
        WHERE src = :src
          AND industry_level = 'L1'
          AND in_date <= :end
          AND (out_date IS NULL OR out_date >= :start)
        ORDER BY industry_id, symbol, in_date
        """,
        {"src": src, "start": start, "end": end},
    )
    if frame.empty:
        raise SystemExit(f"cn_ts_sw_industry_member_hist has no L1 rows for src={src} in the requested range.")
    frame["symbol"] = frame["symbol"].map(_normalize_symbol)
    frame["in_date"] = pd.to_datetime(frame["in_date"], errors="coerce").dt.date
    frame["out_date"] = pd.to_datetime(frame["out_date"], errors="coerce").dt.date
    frame["parent_id"] = None
    frame["parent_name"] = None
    return frame


def _active_memberships_by_date(
    frame: pd.DataFrame,
    effective_dates: list[date],
    industry_column: str,
    extra_columns: list[str] | None = None,
) -> tuple[dict[date, dict[str, set[str]]], dict[str, dict[str, object]]]:
    memberships: dict[date, dict[str, set[str]]] = {effective_date: defaultdict(set) for effective_date in effective_dates}
    meta: dict[str, dict[str, object]] = {}
    extra_columns = extra_columns or []
    for row in frame.itertuples(index=False):
        industry_id = str(getattr(row, industry_column))
        symbol = str(getattr(row, "symbol"))
        in_date = getattr(row, "in_date")
        out_date = getattr(row, "out_date")
        if pd.isna(in_date):
            continue
        if pd.isna(out_date):
            out_date = None
        for effective_date in effective_dates:
            if in_date <= effective_date and (out_date is None or effective_date <= out_date):
                memberships[effective_date][industry_id].add(symbol)
        if industry_id not in meta:
            meta[industry_id] = {}
            for column in extra_columns:
                meta[industry_id][column] = getattr(row, column, None)
    return memberships, meta


def _classify_mapping(best: dict | None, second: dict | None) -> tuple[str, int, str]:
    if best is None or int(best.get("shared_symbol_count") or 0) <= 0:
        return "UNMATCHED", 1, "no_shared_symbols"
    best_jaccard = float(best["jaccard_score"])
    best_cov_v8 = float(best["coverage_vs_v8"])
    best_cov_v7 = float(best["coverage_vs_v7"])
    second_jaccard = float(second["jaccard_score"]) if second is not None else 0.0
    second_cov_v8 = float(second["coverage_vs_v8"]) if second is not None else 0.0
    if best_cov_v8 >= 0.85 and best_cov_v7 >= 0.60 and second_jaccard <= max(0.15, best_jaccard - 0.08):
        return "ONE_TO_ONE", 0, "high_v8_coverage_clear_parent"
    if best_cov_v8 >= 0.85 and second_cov_v8 <= max(0.50, best_cov_v8 - 0.20):
        return "ONE_TO_ONE", 0, "high_v8_coverage_single_parent"
    if best_cov_v8 >= 0.60 and second is not None and second_cov_v8 >= max(0.55, best_cov_v8 - 0.10):
        return "ONE_TO_MANY", 1, "multiple_close_parent_candidates"
    if best_cov_v8 >= 0.60:
        return "ONE_TO_ONE", 0, "usable_parent_match"
    if best_jaccard < 0.20 or best_cov_v8 < 0.35:
        return "MANUAL_REVIEW", 1, "low_overlap"
    return "MANUAL_REVIEW", 1, "borderline_overlap"


def _build_crosswalk_rows(
    effective_dates: list[date],
    v8_memberships: dict[date, dict[str, set[str]]],
    v8_meta: dict[str, dict[str, object]],
    sw_memberships_by_src: dict[str, dict[date, dict[str, set[str]]]],
    sw_meta_by_src: dict[str, dict[str, SwIndustryInfo]],
    top_k: int,
) -> list[dict]:
    rows: list[dict] = []
    for effective_date in effective_dates:
        v8_at_date = v8_memberships.get(effective_date, {})
        for src, sw_by_date in sw_memberships_by_src.items():
            sw_at_date = sw_by_date.get(effective_date, {})
            for v8_id, v8_symbols in sorted(v8_at_date.items()):
                candidates: list[dict] = []
                v8_symbol_count = len(v8_symbols)
                for sw_id, sw_symbols in sw_at_date.items():
                    shared = len(v8_symbols & sw_symbols)
                    if shared <= 0:
                        continue
                    v7_symbol_count = len(sw_symbols)
                    union_count = len(v8_symbols | sw_symbols)
                    candidates.append(
                        {
                            "effective_date": effective_date,
                            "v8_local_industry_id": v8_id,
                            "v8_local_industry_name": v8_meta.get(v8_id, {}).get("industry_name"),
                            "v8_parent_id": v8_meta.get(v8_id, {}).get("parent_id"),
                            "v8_parent_name": v8_meta.get(v8_id, {}).get("parent_name"),
                            "v7_sw_l1_code": sw_id,
                            "v7_sw_l1_name": sw_meta_by_src[src][sw_id].industry_name,
                            "v7_parent_code": sw_meta_by_src[src][sw_id].parent_id,
                            "v7_parent_name": sw_meta_by_src[src][sw_id].parent_name,
                            "src": src,
                            "shared_symbol_count": shared,
                            "v8_symbol_count": v8_symbol_count,
                            "v7_symbol_count": v7_symbol_count,
                            "jaccard_score": shared / union_count if union_count else 0.0,
                            "coverage_vs_v8": shared / v8_symbol_count if v8_symbol_count else 0.0,
                            "coverage_vs_v7": shared / v7_symbol_count if v7_symbol_count else 0.0,
                        }
                    )
                candidates.sort(
                    key=lambda item: (
                        item["jaccard_score"],
                        item["coverage_vs_v8"],
                        item["shared_symbol_count"],
                        item["coverage_vs_v7"],
                    ),
                    reverse=True,
                )
                best = candidates[0] if candidates else None
                second = candidates[1] if len(candidates) > 1 else None
                mapping_type, manual_review_flag, reason = _classify_mapping(best, second)
                if not candidates:
                    rows.append(
                        {
                            "effective_date": effective_date,
                            "v8_local_industry_id": v8_id,
                            "v8_local_industry_name": v8_meta.get(v8_id, {}).get("industry_name"),
                            "v8_parent_id": v8_meta.get(v8_id, {}).get("parent_id"),
                            "v8_parent_name": v8_meta.get(v8_id, {}).get("parent_name"),
                            "v7_sw_l1_code": None,
                            "v7_sw_l1_name": None,
                            "v7_parent_code": None,
                            "v7_parent_name": None,
                            "src": src,
                            "candidate_rank": 1,
                            "best_match_flag": 1,
                            "mapping_type": mapping_type,
                            "manual_review_flag": manual_review_flag,
                            "shared_symbol_count": 0,
                            "v8_symbol_count": v8_symbol_count,
                            "v7_symbol_count": 0,
                            "jaccard_score": 0.0,
                            "coverage_vs_v8": 0.0,
                            "coverage_vs_v7": 0.0,
                            "decision_reason": reason,
                        }
                    )
                    continue
                for rank_no, candidate in enumerate(candidates[:top_k], start=1):
                    candidate.update(
                        {
                            "candidate_rank": rank_no,
                            "best_match_flag": 1 if rank_no == 1 else 0,
                            "mapping_type": mapping_type,
                            "manual_review_flag": manual_review_flag,
                            "decision_reason": reason,
                        }
                    )
                    rows.append(candidate)
    return rows


def _write_report(best_frame: pd.DataFrame, candidate_frame: pd.DataFrame, output_dir: Path, args: argparse.Namespace) -> None:
    summary = {
        "row_count": len(best_frame.index),
        "one_to_one": int((best_frame["mapping_type"] == "ONE_TO_ONE").sum()),
        "one_to_many": int((best_frame["mapping_type"] == "ONE_TO_MANY").sum()),
        "unmatched": int((best_frame["mapping_type"] == "UNMATCHED").sum()),
        "manual_review": int((best_frame["mapping_type"] == "MANUAL_REVIEW").sum()),
    }
    top_review = best_frame.loc[best_frame["mapping_type"] != "ONE_TO_ONE"].copy()
    top_review = top_review.sort_values(["shared_symbol_count", "jaccard_score", "coverage_vs_v8"], ascending=[True, True, True]).head(20)
    by_src = best_frame.groupby(["src", "mapping_type"]).size().reset_index(name="count")
    lines = [
        "# V7/V8 Industry Crosswalk Report",
        "",
        f"- start: `{args.start}`",
        f"- end: `{args.end or 'today'}`",
        f"- srcs: `{', '.join(args.srcs)}`",
        f"- top_k: `{args.top_k}`",
        f"- v8_local_scope: `{args.v8_local_scope}`",
        "",
        "## Summary",
        "",
        f"- best rows: `{summary['row_count']}`",
        f"- one_to_one: `{summary['one_to_one']}`",
        f"- one_to_many: `{summary['one_to_many']}`",
        f"- unmatched: `{summary['unmatched']}`",
        f"- manual_review: `{summary['manual_review']}`",
        "",
        "## By Source",
        "",
    ]
    if by_src.empty:
        lines.append("- no rows")
    else:
        for row in by_src.itertuples(index=False):
            lines.append(f"- `{row.src}` / `{row.mapping_type}`: `{row.count}`")
    lines.extend(["", "## Review / Unmatched Examples", ""])
    if top_review.empty:
        lines.append("- none")
    else:
        for row in top_review.itertuples(index=False):
            lines.append(
                f"- `{row.effective_date}` `{row.v8_local_industry_id}` -> `{row.v7_sw_l1_code}` "
                f"({row.mapping_type}, jaccard={row.jaccard_score:.4f}, cov_v8={row.coverage_vs_v8:.4f}, reason={row.decision_reason})"
            )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- candidates: `{(output_dir / 'v7_v8_industry_crosswalk.csv').as_posix()}`",
            f"- best: `{(output_dir / 'v7_v8_industry_crosswalk_best.csv').as_posix()}`",
            f"- one_to_many: `{(output_dir / 'v7_v8_industry_crosswalk_one_to_many.csv').as_posix()}`",
            f"- unmatched: `{(output_dir / 'v7_v8_industry_crosswalk_unmatched.csv').as_posix()}`",
            f"- manual_review: `{(output_dir / 'v7_v8_industry_crosswalk_manual_review.csv').as_posix()}`",
        ]
    )
    (output_dir / "v7_v8_industry_crosswalk_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    date_range = resolve_date_range(args.start, args.end)
    logger = build_logger("build_v7_v8_industry_crosswalk")
    engine = get_engine()
    apply_sql_file(engine, "docs/DDL/cn_market.cn_v7_v8_industry_crosswalk.sql")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "crosswalk_start start=%s end=%s srcs=%s top_k=%s replace=%s scope=%s",
        date_range.start,
        date_range.end,
        args.srcs,
        args.top_k,
        args.replace,
        args.v8_local_scope,
    )
    effective_dates = _fetch_effective_dates(engine, date_range.start, date_range.end)
    logger.info("effective_dates count=%s first=%s last=%s", len(effective_dates), effective_dates[0], effective_dates[-1])
    v8_frame = _fetch_v8_local_rows(engine, date_range.start, date_range.end, args.v8_local_scope)
    v8_memberships, v8_meta = _active_memberships_by_date(
        v8_frame,
        effective_dates,
        "industry_id",
        ["industry_name"],
    )

    sw_memberships_by_src: dict[str, dict[date, dict[str, set[str]]]] = {}
    sw_meta_by_src: dict[str, dict[str, SwIndustryInfo]] = {}
    client: TushareClient | None = None
    if args.source_mode == "tushare":
        token = resolve_tushare_token(args.token, args.config)
        client = TushareClient(token=token, logger=logger)

    for src in args.srcs:
        if args.source_mode == "db":
            master = _fetch_sw_l1_master_from_db(engine, src)
            membership_frame = _fetch_sw_memberships_from_db(engine, src, date_range.start, date_range.end)
        else:
            if client is None:
                raise SystemExit("Tushare client was not initialized.")
            master = _fetch_sw_l1_master(client, src)
            membership_frame = _fetch_sw_memberships(client, src, master, date_range.start, date_range.end)
        sw_meta_by_src[src] = master
        memberships, _ = _active_memberships_by_date(membership_frame, effective_dates, "industry_id")
        sw_memberships_by_src[src] = memberships
        logger.info("sw_memberships src=%s source_mode=%s industries=%s rows=%s", src, args.source_mode, len(master), len(membership_frame.index))

    rows = _build_crosswalk_rows(
        effective_dates=effective_dates,
        v8_memberships=v8_memberships,
        v8_meta=v8_meta,
        sw_memberships_by_src=sw_memberships_by_src,
        sw_meta_by_src=sw_meta_by_src,
        top_k=max(1, int(args.top_k)),
    )
    if not rows:
        raise SystemExit("Crosswalk builder produced no rows.")

    frame = pd.DataFrame(rows)
    frame.sort_values(["effective_date", "v8_local_industry_id", "src", "candidate_rank"], inplace=True)
    best_frame = frame.loc[frame["best_match_flag"] == 1].copy()
    one_to_many_frame = best_frame.loc[best_frame["mapping_type"] == "ONE_TO_MANY"].copy()
    unmatched_frame = best_frame.loc[best_frame["mapping_type"] == "UNMATCHED"].copy()
    manual_frame = best_frame.loc[best_frame["manual_review_flag"] == 1].copy()

    frame.to_csv(output_dir / "v7_v8_industry_crosswalk.csv", index=False, encoding="utf-8-sig")
    best_frame.to_csv(output_dir / "v7_v8_industry_crosswalk_best.csv", index=False, encoding="utf-8-sig")
    one_to_many_frame.to_csv(output_dir / "v7_v8_industry_crosswalk_one_to_many.csv", index=False, encoding="utf-8-sig")
    unmatched_frame.to_csv(output_dir / "v7_v8_industry_crosswalk_unmatched.csv", index=False, encoding="utf-8-sig")
    manual_frame.to_csv(output_dir / "v7_v8_industry_crosswalk_manual_review.csv", index=False, encoding="utf-8-sig")
    _write_report(best_frame, frame, output_dir, args)

    if args.v8_local_scope == "full" and not args.allow_full_db_write:
        logger.warning(
            "crosswalk_skip_db_write scope=full allow_full_db_write=0 "
            "output_dir=%s candidate_rows=%s best_rows=%s",
            output_dir,
            len(frame.index),
            len(best_frame.index),
        )
        return

    if args.replace:
        with engine.begin() as conn:
            conn.execute(
                text(
                    """
                    DELETE FROM cn_v7_v8_industry_crosswalk
                    WHERE effective_date BETWEEN :start AND :end
                    """
                ),
                {"start": date_range.start, "end": date_range.end},
            )
    insert_sql = """
    INSERT INTO cn_v7_v8_industry_crosswalk
        (
            effective_date, v8_local_industry_id, v8_local_industry_name, v8_parent_id, v8_parent_name,
            v7_sw_l1_code, v7_sw_l1_name, v7_parent_code, v7_parent_name, src,
            candidate_rank, best_match_flag, mapping_type, manual_review_flag,
            shared_symbol_count, v8_symbol_count, v7_symbol_count,
            jaccard_score, coverage_vs_v8, coverage_vs_v7, decision_reason
        )
    VALUES
        (
            :effective_date, :v8_local_industry_id, :v8_local_industry_name, :v8_parent_id, :v8_parent_name,
            :v7_sw_l1_code, :v7_sw_l1_name, :v7_parent_code, :v7_parent_name, :src,
            :candidate_rank, :best_match_flag, :mapping_type, :manual_review_flag,
            :shared_symbol_count, :v8_symbol_count, :v7_symbol_count,
            :jaccard_score, :coverage_vs_v8, :coverage_vs_v7, :decision_reason
        )
    ON DUPLICATE KEY UPDATE
        v8_local_industry_name = VALUES(v8_local_industry_name),
        v8_parent_id = VALUES(v8_parent_id),
        v8_parent_name = VALUES(v8_parent_name),
        v7_sw_l1_code = VALUES(v7_sw_l1_code),
        v7_sw_l1_name = VALUES(v7_sw_l1_name),
        v7_parent_code = VALUES(v7_parent_code),
        v7_parent_name = VALUES(v7_parent_name),
        best_match_flag = VALUES(best_match_flag),
        mapping_type = VALUES(mapping_type),
        manual_review_flag = VALUES(manual_review_flag),
        shared_symbol_count = VALUES(shared_symbol_count),
        v8_symbol_count = VALUES(v8_symbol_count),
        v7_symbol_count = VALUES(v7_symbol_count),
        jaccard_score = VALUES(jaccard_score),
        coverage_vs_v8 = VALUES(coverage_vs_v8),
        coverage_vs_v7 = VALUES(coverage_vs_v7),
        decision_reason = VALUES(decision_reason),
        updated_at = CURRENT_TIMESTAMP
    """
    records = frame.to_dict(orient="records")
    with engine.begin() as conn:
        for batch in chunked_rows(records, 1000):
            conn.execute(text(insert_sql), batch)
    logger.info(
        "crosswalk_done candidate_rows=%s best_rows=%s one_to_many=%s unmatched=%s manual_review=%s output_dir=%s",
        len(frame.index),
        len(best_frame.index),
        len(one_to_many_frame.index),
        len(unmatched_frame.index),
        len(manual_frame.index),
        output_dir,
    )


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
