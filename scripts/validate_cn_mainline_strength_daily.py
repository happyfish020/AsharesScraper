"""
scripts/validate_cn_mainline_strength_daily.py
===============================================
GrowthAlpha V8 — P0 Upstream Mainline Strength Backfill Validation Script.

Validates the cn_mainline_strength_daily table for data integrity,
completeness, and correctness.

Pre-checks (source table/column validation):
  0a. Source tables exist
  0b. Source columns exist (case-sensitive check)
  0c. P0 schema drift detection

Data checks:
  1. Output table exists
  2. Required columns exist
  3. Data exists for specified date range
  4. mainline_strength in [0, 1]
  5. All sub-scores in [0, 1]
  6. mainline_phase values are valid
  7. No duplicate primary keys
  8. rotation_rank is sequential per date
  9. mainline_phase consistency with strength thresholds
  10. Phase distribution summary

Usage:
  python scripts/validate_cn_mainline_strength_daily.py --start 2026-01-01 --end 2026-03-30 --db-name cn_market_red --min-rows 10 --fail-on-empty
  python scripts/validate_cn_mainline_strength_daily.py --start 2026-01-01 --end 2026-03-30 --db-name cn_market_red --skip-source-checks
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
    "industry_id",
    "industry_name",
    "mainline_strength",
    "trend_alignment_score",
    "breakout_ratio",
    "new_high_ratio",
    "leader_density",
    "leader_count",
    "strong_stock_count",
    "capital_concentration_score",
    "rotation_rank",
    "mainline_phase",
]

SCORE_COLUMNS = [
    "mainline_strength",
    "trend_alignment_score",
    "breakout_ratio",
    "new_high_ratio",
    "leader_density",
    "capital_concentration_score",
]

VALID_PHASES = [
    "DOMINANT",
    "EXPANDING",
    "EMERGING",
    "DIVERGING",
    "DECAYING",
    "UNKNOWN",
]

# Phase thresholds (same as builder)
PHASE_THRESHOLDS: dict[str, float] = {
    "DOMINANT": 0.85,
    "EXPANDING": 0.70,
    "EMERGING": 0.55,
    "DIVERGING": 0.40,
}

# ── Source tables & their critical columns for pre-checks ────────────────
SOURCE_TABLES: dict[str, list[str]] = {
    "cn_ga_mainline_radar_daily": [
        "trade_date", "mainline_id", "mainline_name", "mainline_score",
        "mainline_phase", "leader_density", "new_high_ratio", "breakout_ratio",
        "rotation_rank", "trend_alignment_score", "leader_count", "strong_stock_count",
    ],
    "cn_industry_capital_flow_daily": [
        "trade_date", "industry_id", "industry_name", "concentration_score",
        "market_share", "flow_strength_score",
    ],
    "cn_stock_leader_score_daily": [
        "trade_date", "symbol", "leader_score", "leader_bucket",
        "breakout_strength", "industry_id",
    ],
    "cn_ga_stock_role_map_daily": [
        "trade_date", "symbol", "mainline_id", "mainline_name", "stock_role",
    ],
    "cn_local_industry_map_hist": [
        "symbol", "industry_id", "industry_name", "in_date", "out_date",
    ],
    "cn_stock_daily_price": [
        "SYMBOL", "TRADE_DATE", "CLOSE", "PRE_CLOSE", "CHG_PCT", "AMOUNT", "TURNOVER_RATE",
    ],
    "cn_stock_daily_basic": [
        "symbol", "trade_date", "total_mv", "circ_mv", "turnover_rate_f", "volume_ratio",
    ],
}

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="GrowthAlpha V8 — P0 Validate cn_mainline_strength_daily"
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

    print(f"Validating cn_mainline_strength_daily [{start} ~ {end}] on {args.db_name}")

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
    tbl_exists = table_exists(engine, args.db_name, "cn_mainline_strength_daily")
    result.add("Output table [cn_mainline_strength_daily] exists", tbl_exists)
    if not tbl_exists:
        print("\n" + result.summary())
        return 1

    # ── Check 2: Required columns exist ──────────────────────────────
    missing_cols: list[str] = []
    for col in REQUIRED_COLUMNS:
        if not column_exists(engine, args.db_name, "cn_mainline_strength_daily", col):
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
        engine, args.db_name, "cn_mainline_strength_daily", REQUIRED_COLUMNS
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
        FROM cn_mainline_strength_daily
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
        FROM cn_mainline_strength_daily
        WHERE trade_date BETWEEN :start AND :end
        ORDER BY trade_date, rotation_rank
    """
    df = fetch_df(engine, data_sql, {"start": start, "end": end})

    # ── Check 4: mainline_strength in [0, 1] ─────────────────────────
    score_col = df["mainline_strength"]
    out_of_range = ((score_col < 0.0) | (score_col > 1.0)).sum()
    score_in_range = out_of_range == 0
    result.add(
        "mainline_strength in [0, 1]",
        score_in_range,
        detail=f"out_of_range={out_of_range}" if out_of_range > 0 else "all valid",
    )

    # ── Check 5: All sub-scores in [0, 1] ────────────────────────────
    subscore_out_of_range = 0
    for col in SCORE_COLUMNS:
        if col in df.columns:
            subscore_out_of_range += ((df[col] < 0.0) | (df[col] > 1.0)).sum()
    subscores_in_range = subscore_out_of_range == 0
    result.add(
        "All sub-scores in [0, 1]",
        subscores_in_range,
        detail=f"out_of_range={subscore_out_of_range}" if subscore_out_of_range > 0 else "all valid",
    )

    # ── Check 6: mainline_phase values are valid ─────────────────────
    if "mainline_phase" in df.columns and not df["mainline_phase"].isna().all():
        invalid_phases = df[~df["mainline_phase"].isin(VALID_PHASES)]["mainline_phase"].unique().tolist()
        phases_valid = len(invalid_phases) == 0
        result.add(
            "mainline_phase values valid",
            phases_valid,
            detail=f"invalid={invalid_phases}" if invalid_phases else "all valid",
        )
    else:
        result.add("mainline_phase values valid", False, detail="all NULL or column missing")

    # ── Check 7: No duplicate primary keys ───────────────────────────
    dup_count = df.duplicated(subset=["trade_date", "industry_id"]).sum()
    no_dupes = dup_count == 0
    result.add(
        "No duplicate PKs",
        no_dupes,
        detail=f"duplicates={dup_count}" if dup_count > 0 else "all unique",
    )

    # ── Check 8: rotation_rank is sequential per date ────────────────
    rank_issues = 0
    for dt, group in df.groupby("trade_date"):
        ranks = group["rotation_rank"].dropna().unique()
        expected = set(range(1, len(ranks) + 1))
        actual = set(int(r) for r in ranks)
        if actual != expected:
            rank_issues += 1
    rank_ok = rank_issues == 0
    result.add(
        "rotation_rank sequential per date",
        rank_ok,
        detail=f"dates_with_issues={rank_issues}" if rank_issues > 0 else "all sequential",
    )

    # ── Check 9: mainline_phase consistency with strength thresholds ─
    phase_issues = 0
    for _, row in df.iterrows():
        strength = row["mainline_strength"]
        phase = str(row["mainline_phase"]).strip().upper() if pd.notna(row["mainline_phase"]) else "UNKNOWN"

        if phase == "UNKNOWN":
            continue  # UNKNOWN is always valid

        expected_phase = "DECAYING"
        for threshold, p in sorted(PHASE_THRESHOLDS.items(), key=lambda x: -x[1]):
            if strength >= p:
                expected_phase = p
                break

        # DIVERGING is a special case: strength in [0.40, 0.55) with weakening breakout
        # We accept either EMERGING or DIVERGING in this range
        if expected_phase == "EMERGING" and phase == "DIVERGING":
            continue
        if expected_phase == "DIVERGING" and phase == "EMERGING":
            continue

        if phase != expected_phase:
            phase_issues += 1

    phase_consistent = phase_issues == 0
    result.add(
        "mainline_phase consistent with strength thresholds",
        phase_consistent,
        detail=f"inconsistent={phase_issues}" if phase_issues > 0 else "all consistent",
    )

    # ── Check 10: Phase distribution summary ─────────────────────────
    if verbose and "mainline_phase" in df.columns:
        print()
        print("  Phase Distribution:")
        phase_counts = df["mainline_phase"].value_counts()
        for phase in VALID_PHASES:
            count = phase_counts.get(phase, 0)
            pct = count / len(df) * 100 if len(df) > 0 else 0
            print(f"    {phase}: {count} ({pct:.1f}%)")
        print()

    # ── Summary ──────────────────────────────────────────────────────
    print()
    print(f"  Total rows checked: {len(df)}")
    print(f"  Date range: {df['trade_date'].min()} ~ {df['trade_date'].max()}")
    print(f"  Unique industries: {df['industry_id'].nunique()}")
    print(f"  Unique dates: {df['trade_date'].nunique()}")
    print(f"  Mean mainline_strength: {df['mainline_strength'].mean():.4f}")
    print(f"  Median mainline_strength: {df['mainline_strength'].median():.4f}")
    print()
    print(f"  {result.summary()}")

    return 0 if result.failures == 0 else 1


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    print("=" * 60)
    print("  GrowthAlpha V8 — P0 Mainline Strength Validation")
    print("=" * 60)

    exit_code = validate(args)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
