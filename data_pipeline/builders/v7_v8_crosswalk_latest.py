from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from data_pipeline.common.db import apply_sql_file, chunked_rows, fetch_df, get_engine
from data_pipeline.common.logging_utils import build_logger


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build latest best-snapshot table from cn_v7_v8_industry_crosswalk.")
    parser.add_argument("--asof-date", default="", help="YYYY-MM-DD. Default uses latest effective_date in cn_v7_v8_industry_crosswalk.")
    parser.add_argument("--output-dir", default="reports/analysis/v7_v8_crosswalk_latest", help="Output directory")
    parser.add_argument("--replace", action="store_true", help="Delete existing latest snapshot rows for the asof date before insert")
    return parser


def _resolve_asof_date(engine, asof_date_text: str) -> str:
    if str(asof_date_text or "").strip():
        return str(asof_date_text).strip()
    frame = fetch_df(engine, "SELECT MAX(effective_date) AS asof_date FROM cn_v7_v8_industry_crosswalk")
    if frame.empty or pd.isna(frame.iloc[0]["asof_date"]):
        raise SystemExit("cn_v7_v8_industry_crosswalk is empty. Run the crosswalk builder first.")
    return str(pd.to_datetime(frame.iloc[0]["asof_date"]).date())


def run(args: argparse.Namespace) -> None:
    logger = build_logger("build_v7_v8_crosswalk_latest")
    engine = get_engine()
    apply_sql_file(engine, "docs/DDL/cn_market.cn_v7_v8_industry_crosswalk_latest.sql")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    asof_date = _resolve_asof_date(engine, args.asof_date)

    frame = fetch_df(
        engine,
        """
        SELECT
            effective_date AS asof_date,
            src,
            v8_local_industry_id,
            v8_local_industry_name,
            v8_parent_id,
            v8_parent_name,
            v7_sw_l1_code,
            v7_sw_l1_name,
            v7_parent_code,
            v7_parent_name,
            mapping_type,
            manual_review_flag,
            shared_symbol_count,
            v8_symbol_count,
            v7_symbol_count,
            jaccard_score,
            coverage_vs_v8,
            coverage_vs_v7,
            decision_reason
        FROM cn_v7_v8_industry_crosswalk
        WHERE effective_date = :asof_date
          AND best_match_flag = 1
        ORDER BY src, v8_local_industry_id
        """,
        {"asof_date": asof_date},
    )
    if frame.empty:
        raise SystemExit(f"No best-match crosswalk rows found for asof_date={asof_date}.")

    if args.replace:
        with engine.begin() as conn:
            conn.execute(
                text("DELETE FROM cn_v7_v8_industry_crosswalk_latest WHERE asof_date = :asof_date"),
                {"asof_date": asof_date},
            )

    sql = """
    INSERT INTO cn_v7_v8_industry_crosswalk_latest
        (
            asof_date, src, v8_local_industry_id, v8_local_industry_name, v8_parent_id, v8_parent_name,
            v7_sw_l1_code, v7_sw_l1_name, v7_parent_code, v7_parent_name, mapping_type, manual_review_flag,
            shared_symbol_count, v8_symbol_count, v7_symbol_count, jaccard_score, coverage_vs_v8, coverage_vs_v7, decision_reason
        )
    VALUES
        (
            :asof_date, :src, :v8_local_industry_id, :v8_local_industry_name, :v8_parent_id, :v8_parent_name,
            :v7_sw_l1_code, :v7_sw_l1_name, :v7_parent_code, :v7_parent_name, :mapping_type, :manual_review_flag,
            :shared_symbol_count, :v8_symbol_count, :v7_symbol_count, :jaccard_score, :coverage_vs_v8, :coverage_vs_v7, :decision_reason
        )
    ON DUPLICATE KEY UPDATE
        v8_local_industry_name = VALUES(v8_local_industry_name),
        v8_parent_id = VALUES(v8_parent_id),
        v8_parent_name = VALUES(v8_parent_name),
        v7_sw_l1_code = VALUES(v7_sw_l1_code),
        v7_sw_l1_name = VALUES(v7_sw_l1_name),
        v7_parent_code = VALUES(v7_parent_code),
        v7_parent_name = VALUES(v7_parent_name),
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
            conn.execute(text(sql), batch)

    frame.to_csv(output_dir / f"v7_v8_crosswalk_latest_{asof_date}.csv", index=False, encoding="utf-8-sig")
    summary = frame.groupby(["src", "mapping_type"]).size().reset_index(name="count")
    summary.to_csv(output_dir / f"v7_v8_crosswalk_latest_{asof_date}_summary.csv", index=False, encoding="utf-8-sig")
    logger.info("build_v7_v8_crosswalk_latest_done asof_date=%s rows=%s output_dir=%s", asof_date, len(frame.index), output_dir)


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
