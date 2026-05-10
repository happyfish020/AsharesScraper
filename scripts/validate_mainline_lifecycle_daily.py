"""
scripts/validate_mainline_lifecycle_daily.py
==============================================
GrowthAlpha V8 — P2 Mainline Lifecycle Validation Script.

Validates the cn_mainline_lifecycle_daily table for data integrity,
completeness, and correctness.

Checks:
  1. Table exists
  2. Required columns exist
  3. Data exists for specified date range
  4. lifecycle_state is not NULL
  5. lifecycle_score in [0, 1]
  6. No duplicate primary keys
  7. UNKNOWN ratio not too high (FAIL if >= 80%)
  8. Non-UNKNOWN state counts reported
  9. At least one non-UNKNOWN state exists
 10. Report can be generated

Usage:
  python scripts/validate_mainline_lifecycle_daily.py --start 2026-01-01 --end 2026-03-30 --db-name cn_market_red --min-rows 10 --fail-on-empty
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
    "mainline_id",
    "mainline_name",
    "mainline_strength",
    "capital_concentration_score",
    "trend_alignment_score",
    "breakout_ratio",
    "new_high_ratio",
    "leader_density",
    "rotation_rank",
    "lifecycle_state",
    "lifecycle_score",
    "phase_reason",
    "risk_flag",
]

VALID_LIFECYCLE_STATES = [
    "BOTTOM_REPAIR",
    "TREND_EXPANSION",
    "DIFFUSION",
    "DIVERGENCE",
    "TOP_DECAY",
    "RISK_OFF",
    "UNKNOWN",
]

NON_UNKNOWN_STATES = [
    "BOTTOM_REPAIR",
    "TREND_EXPANSION",
    "DIFFUSION",
    "DIVERGENCE",
    "TOP_DECAY",
    "RISK_OFF",
]

VALID_RISK_FLAGS = [
    "NONE",
    "CROWDING",
    "TOP_DECAY",
    "MARKET_RISK_OFF",
    "DATA_INSUFFICIENT",
]

# FAIL threshold: UNKNOWN ratio >= 80% causes FAIL
FAIL_UNKNOWN_RATIO = 0.80

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="GrowthAlpha V8 — P2 Validate cn_mainline_lifecycle_daily"
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


def validate(args: argparse.Namespace) -> int:
    """
    Run all validation checks. Returns 0 on success, 1 on failure.
    """
    verbose = args.verbose
    fail_on_empty = args.fail_on_empty
    min_rows = args.min_rows

    # Resolve date range
    start = datetime.strptime(args.start, "%Y-%m-%d").date() if "-" in args.start else datetime.strptime(args.start, "%Y%m%d").date()
    end = datetime.strptime(args.end, "%Y-%m-%d").date() if args.end and "-" in args.end else (
        datetime.strptime(args.end, "%Y%m%d").date() if args.end else date.today()
    )

    db_password = args.db_password or os.getenv("ASHARE_MYSQL_PASSWORD", "sec_Bobo123")
    engine = build_engine(args.db_host, args.db_port, args.db_user, db_password, args.db_name)

    result = ValidationResult()

    print(f"Validating cn_mainline_lifecycle_daily [{start} ~ {end}] on {args.db_name}")
    print()

    # ── Check 1: Table exists ───────────────────────────────────────
    tbl_exists = table_exists(engine, args.db_name, "cn_mainline_lifecycle_daily")
    result.add("Table exists", tbl_exists)
    if not tbl_exists:
        print("\n" + result.summary())
        return 1

    # ── Check 2: Required columns exist ─────────────────────────────
    missing_cols: list[str] = []
    for col in REQUIRED_COLUMNS:
        if not column_exists(engine, args.db_name, "cn_mainline_lifecycle_daily", col):
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

    # ── Check 3: Data exists for date range ─────────────────────────
    count_sql = """
        SELECT COUNT(*) AS cnt
        FROM cn_mainline_lifecycle_daily
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

    # ── Load data for detailed checks ───────────────────────────────
    data_sql = """
        SELECT *
        FROM cn_mainline_lifecycle_daily
        WHERE trade_date BETWEEN :start AND :end
        ORDER BY trade_date, mainline_id
    """
    df = fetch_df(engine, data_sql, {"start": start, "end": end})

    # ── Check 4: lifecycle_state not NULL ───────────────────────────
    null_states = df["lifecycle_state"].isna().sum()
    state_not_null = null_states == 0
    result.add(
        "lifecycle_state not NULL",
        state_not_null,
        detail=f"null_count={null_states}" if null_states > 0 else "all valid",
    )

    # ── Check 5: lifecycle_state values are valid ───────────────────
    if not df["lifecycle_state"].isna().all():
        invalid_states = df[~df["lifecycle_state"].isin(VALID_LIFECYCLE_STATES)]["lifecycle_state"].unique().tolist()
        states_valid = len(invalid_states) == 0
        result.add(
            "lifecycle_state values valid",
            states_valid,
            detail=f"invalid={invalid_states}" if invalid_states else "all valid",
        )
    else:
        result.add("lifecycle_state values valid", False, detail="all NULL")

    # ── Check 6: lifecycle_score in [0, 1] ──────────────────────────
    score_col = df["lifecycle_score"]
    out_of_range = ((score_col < 0.0) | (score_col > 1.0)).sum()
    score_in_range = out_of_range == 0
    result.add(
        "lifecycle_score in [0, 1]",
        score_in_range,
        detail=f"out_of_range={out_of_range}" if out_of_range > 0 else "all valid",
    )

    # ── Check 7: No duplicate primary keys ──────────────────────────
    dup_count = df.duplicated(subset=["trade_date", "mainline_id"]).sum()
    no_dupes = dup_count == 0
    result.add(
        "No duplicate PKs",
        no_dupes,
        detail=f"duplicates={dup_count}" if dup_count > 0 else "all unique",
    )

    # ── Check 8: UNKNOWN ratio report ───────────────────────────────
    unknown_count = (df["lifecycle_state"] == "UNKNOWN").sum()
    unknown_ratio = unknown_count / len(df) if len(df) > 0 else 0.0
    unknown_ok = unknown_ratio < FAIL_UNKNOWN_RATIO
    result.add(
        f"UNKNOWN ratio < {FAIL_UNKNOWN_RATIO:.0%}",
        unknown_ok,
        detail=f"ratio={unknown_ratio:.2%} ({unknown_count}/{len(df)})",
    )

    # ── Check 9: Non-UNKNOWN state counts ───────────────────────────
    non_unknown_count = len(df) - unknown_count
    non_unknown_states_present = non_unknown_count > 0
    result.add(
        "Non-UNKNOWN states exist",
        non_unknown_states_present,
        detail=f"non_unknown_count={non_unknown_count}" if non_unknown_states_present else "all UNKNOWN",
    )

    # ── Check 10: At least one specific non-UNKNOWN state ───────────
    specific_states_found = []
    for state in NON_UNKNOWN_STATES:
        cnt = (df["lifecycle_state"] == state).sum()
        if cnt > 0:
            specific_states_found.append(f"{state}={cnt}")
    has_specific_state = len(specific_states_found) > 0
    result.add(
        "Has at least one specific lifecycle state",
        has_specific_state,
        detail="; ".join(specific_states_found) if specific_states_found else "none found",
    )

    # ── Check 11: risk_flag values are valid ────────────────────────
    if not df["risk_flag"].isna().all():
        invalid_flags = df[~df["risk_flag"].isin(VALID_RISK_FLAGS)]["risk_flag"].unique().tolist()
        flags_valid = len(invalid_flags) == 0
        result.add(
            "risk_flag values valid",
            flags_valid,
            detail=f"invalid={invalid_flags}" if invalid_flags else "all valid",
        )
    else:
        result.add("risk_flag values valid", True, detail="all NONE or NULL")

    # ── Check 12: rotation_rank is positive integer ─────────────────
    rank_col = df["rotation_rank"]
    invalid_rank = (rank_col.isna() | (rank_col < 1)).sum()
    rank_ok = invalid_rank == 0
    result.add(
        "rotation_rank positive integer",
        rank_ok,
        detail=f"invalid={invalid_rank}" if invalid_rank > 0 else "all valid",
    )

    # ── Check 13: State distribution summary ────────────────────────
    if verbose:
        print()
        print("  State Distribution:")
        state_counts = df["lifecycle_state"].value_counts()
        for state_name, count in state_counts.items():
            pct = count / len(df) * 100
            print(f"    {state_name}: {count} ({pct:.1f}%)")
        print()
        print("  Risk Flag Distribution:")
        risk_counts = df["risk_flag"].value_counts()
        for flag_name, count in risk_counts.items():
            print(f"    {flag_name}: {count}")
        print()
        print("  UNKNOWN Ratio Report:")
        print(f"    UNKNOWN: {unknown_count} / {len(df)} = {unknown_ratio:.2%}")
        print(f"    Non-UNKNOWN: {non_unknown_count} / {len(df)} = {(1-unknown_ratio):.2%}")
        print(f"    Threshold: < {FAIL_UNKNOWN_RATIO:.0%}")
        print(f"    Status: {'PASS' if unknown_ok else 'FAIL'}")

    # ── Summary ─────────────────────────────────────────────────────
    print()
    print(f"  Total rows checked: {len(df)}")
    print(f"  Date range: {df['trade_date'].min()} ~ {df['trade_date'].max()}")
    print(f"  Unique mainlines: {df['mainline_id'].nunique()}")
    print(f"  Unique dates: {df['trade_date'].nunique()}")
    print(f"  UNKNOWN ratio: {unknown_ratio:.2%}")
    print(f"  Non-UNKNOWN states: {non_unknown_count}")
    print()
    print(f"  {result.summary()}")

    return 0 if result.failures == 0 else 1


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    print("=" * 60)
    print("  GrowthAlpha V8 — P2 Mainline Lifecycle Validation")
    print("=" * 60)

    exit_code = validate(args)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
