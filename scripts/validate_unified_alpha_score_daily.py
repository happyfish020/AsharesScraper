"""
scripts/validate_unified_alpha_score_daily.py
===============================================
GrowthAlpha V8 — P3 Unified Alpha Engine Validation Script.

Validates the cn_unified_alpha_score_daily table for data integrity,
completeness, and correctness.

Pre-checks (source table/column validation):
  0a. Source tables exist
  0b. Source columns exist (case-sensitive check)
  0c. P0 schema drift detection

Data checks:
  1. Output table exists
  2. Required columns exist
  3. Data exists for specified date range
  4. final_score in [0, 1]
  5. All 8 factor scores in [0, 1]
  6. alpha_bucket values are valid
  7. No duplicate primary keys
  8. Explanation is not NULL
  9. Factor weight consistency (risk_crowding_score inverted: 1 - score)
  10. No future data (ann_date <= trade_date)
  11. Bucket distribution summary

Usage:
  python scripts/validate_unified_alpha_score_daily.py --start 2026-01-01 --end 2026-03-30 --db-name cn_market_red --min-rows 10 --fail-on-empty
  python scripts/validate_unified_alpha_score_daily.py --start 2026-01-01 --end 2026-03-30 --db-name cn_market_red --skip-source-checks
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date, datetime
from typing import Any

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, URL

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS = [
    "trade_date",
    "symbol",
    "industry_id",
    "quality_score",
    "growth_acceleration_score",
    "mainline_strength_score",
    "capital_concentration_score",
    "leader_dominance_score",
    "trend_quality_score",
    "lifecycle_position_score",
    "risk_crowding_score",
    "final_score",
    "alpha_bucket",
    "lifecycle_state",
    "mainline_name",
    "explanation",
    "top_factors",
    "weak_factors",
    "flags",
]

FACTOR_SCORE_COLUMNS = [
    "quality_score",
    "growth_acceleration_score",
    "mainline_strength_score",
    "capital_concentration_score",
    "leader_dominance_score",
    "trend_quality_score",
    "lifecycle_position_score",
    "risk_crowding_score",
]

VALID_ALPHA_BUCKETS = [
    "TOP_1",
    "TOP_5",
    "TOP_10",
    "TOP_20",
    "WATCH",
    "NEUTRAL",
    "AVOID",
]

FACTOR_WEIGHTS = {
    "quality_score": 0.10,
    "growth_acceleration_score": 0.10,
    "mainline_strength_score": 0.20,
    "capital_concentration_score": 0.15,
    "leader_dominance_score": 0.15,
    "trend_quality_score": 0.15,
    "lifecycle_position_score": 0.10,
    "risk_crowding_score": 0.05,
}

MAX_NULL_EXPLANATION_RATIO = 0.05  # 5% max NULL explanations allowed

# ── Source tables & their critical columns for pre-checks ────────────────
SOURCE_TABLES: dict[str, list[str]] = {
    "cn_stock_quality_score_daily": [
        "trade_date", "symbol", "quality_score", "growth_acceleration_score",
    ],
    "cn_stock_mainline_strength_daily": [
        "trade_date", "industry_id", "industry_name", "mainline_strength_score",
    ],
    "cn_ga_mainline_radar_daily": [
        "trade_date", "mainline_id", "mainline_name", "leader_density",
        "leader_count", "mainline_score",
    ],
    "cn_industry_capital_flow_daily": [
        "trade_date", "industry_id", "industry_name", "concentration_score",
        "market_share", "flow_strength_score",
    ],
    "cn_ga_stock_role_map_daily": [
        "trade_date", "symbol", "mainline_id", "mainline_name", "stock_role",
    ],
    "cn_stock_daily_price": [
        "SYMBOL", "TRADE_DATE", "CLOSE", "PRE_CLOSE", "CHG_PCT", "AMOUNT", "TURNOVER_RATE",
    ],
    "cn_mainline_lifecycle_daily": [
        "trade_date", "mainline_id", "lifecycle_state", "lifecycle_score",
    ],
    "cn_ga_market_pulse_daily": [
        "trade_date", "bullish_industry_ratio", "bearish_industry_ratio", "market_phase",
    ],
    "cn_local_industry_map_hist": [
        "symbol", "industry_id", "industry_name", "in_date", "out_date",
    ],
}

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="GrowthAlpha V8 — P3 Validate cn_unified_alpha_score_daily"
    )
    parser.add_argument("--start", default="2010-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default="", help="End date YYYY-MM-DD (default=today)")
    parser.add_argument("--db-name", default="cn_market_red", help="Database name")
    parser.add_argument("--db-host", default="127.0.0.1", help="MySQL host")
    parser.add_argument("--db-port", type=int, default=3306, help="MySQL port")
    parser.add_argument("--db-user", default="root", help="MySQL user")
    parser.add_argument("--db-password", default=None, help="MySQL password (default from env)")
    parser.add_argument("--min-rows", type=int, default=1, help="Minimum rows expected")
    parser.add_argument("--fail-on-empty", action="store_true", help="Exit with error if no data")
    parser.add_argument("--skip-source-checks", action="store_true", help="Skip source table/column pre-checks")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    return parser


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------


def build_engine(db_host: str, db_port: int, db_user: str, db_password: str, db_name: str) -> Engine:
    conn_url = URL.create(
        drivername="mysql+pymysql",
        username=db_user,
        password=db_password,
        host=db_host,
        port=db_port,
        database=db_name,
        query={"charset": "utf8mb4"},
    )
    return create_engine(conn_url, pool_pre_ping=True, future=True)


def fetch_df(engine: Engine, sql: str, params: dict[str, Any] | None = None) -> pd.DataFrame:
    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn, params=params or {})


def table_exists(engine: Engine, db_name: str, table_name: str) -> bool:
    sql = """
        SELECT COUNT(*) AS cnt
        FROM INFORMATION_SCHEMA.TABLES
        WHERE TABLE_SCHEMA = :schema AND TABLE_NAME = :table
    """
    with engine.connect() as conn:
        return conn.execute(text(sql), {"schema": db_name, "table": table_name}).scalar() > 0


def column_exists(engine: Engine, db_name: str, table_name: str, column_name: str) -> bool:
    sql = """
        SELECT COUNT(*) AS cnt
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = :schema
          AND TABLE_NAME = :table
          AND COLUMN_NAME = :column
    """
    with engine.connect() as conn:
        return conn.execute(text(sql), {"schema": db_name, "table": table_name, "column": column_name}).scalar() > 0


def get_actual_columns(engine: Engine, db_name: str, table_name: str) -> list[str]:
    """Fetch actual column names for a table (preserving case)."""
    sql = """
        SELECT COLUMN_NAME
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = :schema AND TABLE_NAME = :table
        ORDER BY ORDINAL_POSITION
    """
    with engine.connect() as conn:
        rows = conn.execute(text(sql), {"schema": db_name, "table": table_name}).fetchall()
        return [row[0] for row in rows]


def check_column_case_mismatch(
    engine: Engine, db_name: str, table_name: str, expected_columns: list[str]
) -> list[str]:
    """
    Detect case mismatches between expected column names and actual column names.
    Returns list of mismatches as 'expected_col -> actual_col' strings.
    """
    actual_cols = get_actual_columns(engine, db_name, table_name)
    actual_lower = {c.lower(): c for c in actual_cols}
    mismatches: list[str] = []
    for expected in expected_columns:
        expected_lower = expected.lower()
        if expected_lower in actual_lower:
            actual = actual_lower[expected_lower]
            if actual != expected:
                mismatches.append(f"{expected} -> {actual}")
        else:
            mismatches.append(f"{expected} -> MISSING")
    return mismatches


# ---------------------------------------------------------------------------
# Validation checks
# ---------------------------------------------------------------------------


class ValidationResult:
    """Collects validation check results."""

    def __init__(self) -> None:
        self.checks: list[dict[str, Any]] = []
        self.failures: int = 0

    def add(self, name: str, passed: bool, detail: str = "") -> None:
        self.checks.append({"name": name, "passed": passed, "detail": detail})
        if not passed:
            self.failures += 1
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))

    def summary(self) -> str:
        total = len(self.checks)
        passed = total - self.failures
        return f"Validation: {passed}/{total} passed, {self.failures} failed"


def check_source_tables(
    engine: Engine, db_name: str, result: ValidationResult, verbose: bool
) -> bool:
    """
    Pre-check all source tables exist and have required columns.
    Returns False if any source table is missing (fatal).
    """
    print()
    print("── Source Table Pre-checks ──────────────────────────────────")
    all_tables_ok = True

    for table_name, required_cols in SOURCE_TABLES.items():
        # Check table existence
        tbl_ok = table_exists(engine, db_name, table_name)
        result.add(
            f"Source table [{table_name}] exists",
            tbl_ok,
        )
        if not tbl_ok:
            all_tables_ok = False
            continue

        # Check column existence
        missing_cols: list[str] = []
        for col in required_cols:
            if not column_exists(engine, db_name, table_name, col):
                missing_cols.append(col)
        cols_ok = len(missing_cols) == 0
        result.add(
            f"  Columns in [{table_name}]",
            cols_ok,
            detail=f"missing={missing_cols}" if missing_cols else "all present",
        )

        # Case sensitivity check
        case_mismatches = check_column_case_mismatch(engine, db_name, table_name, required_cols)
        case_ok = all("MISSING" not in m for m in case_mismatches) and len(case_mismatches) == 0
        if case_mismatches:
            result.add(
                f"  Case check [{table_name}]",
                case_ok,
                detail=f"mismatches={case_mismatches}",
            )

        # P0 schema drift: show actual vs expected columns if verbose
        if verbose and tbl_ok:
            actual_cols = get_actual_columns(engine, db_name, table_name)
            extra_cols = [c for c in actual_cols if c.lower() not in {x.lower() for x in required_cols}]
            if extra_cols:
                print(f"    [INFO] Extra columns in [{table_name}]: {extra_cols}")

    return all_tables_ok


def validate(args: argparse.Namespace) -> int:
    """
    Run all validation checks. Returns 0 on success, 1 on failure.
    """
    verbose = args.verbose
    fail_on_empty = args.fail_on_empty
    min_rows = args.min_rows
    skip_source = args.skip_source_checks

    # Resolve date range
    start = datetime.strptime(args.start, "%Y-%m-%d").date() if "-" in args.start else datetime.strptime(args.start, "%Y%m%d").date()
    end = datetime.strptime(args.end, "%Y-%m-%d").date() if args.end and "-" in args.end else (
        datetime.strptime(args.end, "%Y%m%d").date() if args.end else date.today()
    )

    db_password = args.db_password or os.getenv("ASHARE_MYSQL_PASSWORD", "sec_Bobo123")
    engine = build_engine(args.db_host, args.db_port, args.db_user, db_password, args.db_name)

    result = ValidationResult()

    print(f"Validating cn_unified_alpha_score_daily [{start} ~ {end}] on {args.db_name}")

    # ── Pre-check 0: Source table/column validation ──────────────────
    if not skip_source:
        source_ok = check_source_tables(engine, args.db_name, result, verbose)
        if not source_ok:
            print()
            print("  ⚠ Some source tables are missing. The output table may be incomplete.")
            print("  Use --skip-source-checks to bypass this check.")
            print()
    else:
        if verbose:
            print("  [SKIP] Source table pre-checks (--skip-source-checks)")

    print()
    print("── Output Table Checks ──────────────────────────────────────")

    # ── Check 1: Output table exists ─────────────────────────────────
    tbl_exists = table_exists(engine, args.db_name, "cn_unified_alpha_score_daily")
    result.add("Output table [cn_unified_alpha_score_daily] exists", tbl_exists)
    if not tbl_exists:
        print("\n" + result.summary())
        return 1

    # ── Check 2: Required columns exist ──────────────────────────────
    missing_cols: list[str] = []
    for col in REQUIRED_COLUMNS:
        if not column_exists(engine, args.db_name, "cn_unified_alpha_score_daily", col):
            missing_cols.append(col)
    cols_ok = len(missing_cols) == 0
    result.add(
        "Required columns exist",
        cols_ok,
        detail=f"missing={missing_cols}" if missing_cols else "all present",
    )
    if not cols_ok:
        print("\n" + result.summary())
        return 1

    # ── Check 2b: Column case sensitivity ────────────────────────────
    case_mismatches = check_column_case_mismatch(
        engine, args.db_name, "cn_unified_alpha_score_daily", REQUIRED_COLUMNS
    )
    case_ok = len(case_mismatches) == 0
    result.add(
        "Column case match",
        case_ok,
        detail=f"mismatches={case_mismatches}" if case_mismatches else "all match",
    )

    # ── Check 3: Data exists for date range ──────────────────────────
    count_sql = """
        SELECT COUNT(*) AS cnt
        FROM cn_unified_alpha_score_daily
        WHERE trade_date BETWEEN :start AND :end
    """
    row_count = fetch_df(engine, count_sql, {"start": start, "end": end}).iloc[0, 0]
    has_data = row_count >= min_rows
    result.add(
        f"Data exists (min_rows={min_rows})",
        has_data,
        detail=f"found={row_count} rows",
    )
    if not has_data and fail_on_empty:
        print("\n" + result.summary())
        return 1

    if row_count == 0:
        print("\n" + result.summary())
        return 0 if not fail_on_empty else 1

    # ── Load data for detailed checks ────────────────────────────────
    data_sql = """
        SELECT *
        FROM cn_unified_alpha_score_daily
        WHERE trade_date BETWEEN :start AND :end
        ORDER BY trade_date, symbol
    """
    df = fetch_df(engine, data_sql, {"start": start, "end": end})

    # ── Check 4: final_score in [0, 1] ───────────────────────────────
    score_col = df["final_score"]
    out_of_range = ((score_col < 0.0) | (score_col > 1.0)).sum()
    score_in_range = out_of_range == 0
    result.add(
        "final_score in [0, 1]",
        score_in_range,
        detail=f"out_of_range={out_of_range}" if out_of_range > 0 else "all valid",
    )

    # ── Check 5: All 8 factor scores in [0, 1] ──────────────────────
    factor_out_of_range = 0
    for col in FACTOR_SCORE_COLUMNS:
        if col in df.columns:
            factor_out_of_range += ((df[col] < 0.0) | (df[col] > 1.0)).sum()
    factors_in_range = factor_out_of_range == 0
    result.add(
        "All factor scores in [0, 1]",
        factors_in_range,
        detail=f"out_of_range={factor_out_of_range}" if factor_out_of_range > 0 else "all valid",
    )

    # ── Check 6: alpha_bucket values are valid ───────────────────────
    if "alpha_bucket" in df.columns and not df["alpha_bucket"].isna().all():
        invalid_buckets = df[~df["alpha_bucket"].isin(VALID_ALPHA_BUCKETS)]["alpha_bucket"].unique().tolist()
        buckets_valid = len(invalid_buckets) == 0
        result.add(
            "alpha_bucket values valid",
            buckets_valid,
            detail=f"invalid={invalid_buckets}" if invalid_buckets else "all valid",
        )
    else:
        result.add("alpha_bucket values valid", False, detail="all NULL or column missing")

    # ── Check 6b: Bucket existence (adaptive to sample size) ─────────
    # Small samples may not populate all top buckets; adapt thresholds.
    if "alpha_bucket" in df.columns and not df["alpha_bucket"].isna().all():
        bucket_counts = df["alpha_bucket"].value_counts()
        n = len(df)
        if n >= 500:
            # Full production: TOP_1 and TOP_5 must have rows
            top1_ok = bucket_counts.get("TOP_1", 0) > 0
            top5_ok = bucket_counts.get("TOP_5", 0) > 0
            bucket_existence_ok = top1_ok and top5_ok
            detail_parts = []
            if not top1_ok:
                detail_parts.append("TOP_1 empty")
            if not top5_ok:
                detail_parts.append("TOP_5 empty")
            result.add(
                "Bucket existence (TOP_1 + TOP_5 required, n>=500)",
                bucket_existence_ok,
                detail="; ".join(detail_parts) if detail_parts else f"TOP_1={bucket_counts.get('TOP_1',0)} TOP_5={bucket_counts.get('TOP_5',0)}",
            )
        elif n >= 100:
            # Medium sample: at least one of TOP_5 or TOP_10 must have rows
            top5_ok = bucket_counts.get("TOP_5", 0) > 0
            top10_ok = bucket_counts.get("TOP_10", 0) > 0
            bucket_existence_ok = top5_ok or top10_ok
            result.add(
                "Bucket existence (TOP_5 or TOP_10 required, n>=100)",
                bucket_existence_ok,
                detail=f"TOP_5={bucket_counts.get('TOP_5',0)} TOP_10={bucket_counts.get('TOP_10',0)}",
            )
        else:
            # Small sample: skip strict bucket existence checks
            result.add(
                "Bucket existence (skipped, n<100)",
                True,
                detail=f"n={n} too small for reliable bucket distribution",
            )
    else:
        result.add("Bucket existence check", False, detail="alpha_bucket column missing or all NULL")

    # ── Check 7: No duplicate primary keys ───────────────────────────
    dup_count = df.duplicated(subset=["trade_date", "symbol"]).sum()
    no_dupes = dup_count == 0
    result.add(
        "No duplicate PKs",
        no_dupes,
        detail=f"duplicates={dup_count}" if dup_count > 0 else "all unique",
    )

    # ── Check 8: Explanation not NULL ────────────────────────────────
    if "explanation" in df.columns:
        null_explanation = df["explanation"].isna().sum()
        null_ratio = null_explanation / len(df) if len(df) > 0 else 0.0
        explanation_ok = null_ratio <= MAX_NULL_EXPLANATION_RATIO
        result.add(
            f"Explanation NULL ratio <= {MAX_NULL_EXPLANATION_RATIO:.0%}",
            explanation_ok,
            detail=f"null={null_explanation}/{len(df)} ({null_ratio:.2%})",
        )
    else:
        result.add("Explanation column exists", False, detail="column missing")

    # ── Check 9: Factor weight consistency (final_score ≈ weighted avg) ──
    # NOTE: risk_crowding_score is inverted in the builder (1 - score),
    # so the validation must apply the same inversion when computing expected score.
    if all(col in df.columns for col in FACTOR_SCORE_COLUMNS):
        weighted_sum = pd.Series(0.0, index=df.index)
        total_weight = sum(FACTOR_WEIGHTS.values())
        for factor_name, weight in FACTOR_WEIGHTS.items():
            if factor_name in df.columns:
                score = df[factor_name].fillna(0.5)
                # Apply same inversion as builder's _compute_final_score
                if factor_name == "risk_crowding_score":
                    score = 1.0 - score
                weighted_sum += score * weight
        if total_weight > 0:
            expected_score = weighted_sum / total_weight
            deviation = (df["final_score"].fillna(0.5) - expected_score).abs()
            max_dev = deviation.max()
            weight_ok = max_dev < 0.01  # Allow 0.01 tolerance for floating point
            result.add(
                "Final score consistent with weighted factors (risk inverted)",
                weight_ok,
                detail=f"max_deviation={max_dev:.6f}",
            )
        else:
            result.add("Final score consistency check", False, detail="total_weight=0")
    else:
        result.add("Final score consistency check", False, detail="missing factor columns")

    # ── Check 10b: Verify no future data (ann_date <= trade_date) ────
    # This check validates that the cn_stock_quality_score_daily builder correctly
    # filtered out rows where ann_date > trade_date (future-function safety).
    if "ann_date" in df.columns:
        # ann_date is optional in the output table; only check if present
        ann_date_col = df["ann_date"]
        if ann_date_col.notna().any():
            # Compare ann_date with trade_date
            trade_dates = pd.to_datetime(df["trade_date"])
            ann_dates = pd.to_datetime(ann_date_col, errors="coerce")
            future_data = (ann_dates.notna() & (ann_dates > trade_dates)).sum()
            no_future_data = future_data == 0
            result.add(
                "No future data (ann_date <= trade_date)",
                no_future_data,
                detail=f"rows_with_future_data={future_data}" if future_data > 0 else "all valid",
            )
        else:
            result.add("No future data check", True, detail="ann_date all NULL (no data to check)")
    else:
        result.add("No future data check", True, detail="ann_date column not present in output")

    # ── Check 11: Bucket distribution summary ────────────────────────
    if verbose and "alpha_bucket" in df.columns:
        print()
        print("  Alpha Bucket Distribution:")
        bucket_counts = df["alpha_bucket"].value_counts()
        for bucket in VALID_ALPHA_BUCKETS:
            count = bucket_counts.get(bucket, 0)
            pct = count / len(df) * 100 if len(df) > 0 else 0
            print(f"    {bucket}: {count} ({pct:.1f}%)")
        print()

    # ── Summary ──────────────────────────────────────────────────────
    print()
    print(f"  Total rows checked: {len(df)}")
    print(f"  Date range: {df['trade_date'].min()} ~ {df['trade_date'].max()}")
    print(f"  Unique symbols: {df['symbol'].nunique()}")
    print(f"  Unique dates: {df['trade_date'].nunique()}")
    print(f"  Mean final_score: {df['final_score'].mean():.4f}")
    print(f"  Median final_score: {df['final_score'].median():.4f}")
    print()
    print(f"  {result.summary()}")

    return 0 if result.failures == 0 else 1


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    print("=" * 60)
    print("  GrowthAlpha V8 — P3 Unified Alpha Engine Validation")
    print("=" * 60)

    exit_code = validate(args)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
