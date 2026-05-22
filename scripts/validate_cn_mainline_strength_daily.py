"""
scripts/validate_cn_mainline_strength_daily.py
===============================================
GrowthAlpha V8 — Mainline Strength Validation Script.

Compatibility note
------------------
The historical script name is kept for runner compatibility, but GrowthAlpha V8's
current production mainline-strength output table is:

    cn_stock_mainline_strength_daily

The legacy/fallback table cn_mainline_strength_daily may be empty and is no
longer treated as the required validation target.

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

OUTPUT_TABLE = "cn_stock_mainline_strength_daily"
LEGACY_TABLE = "cn_mainline_strength_daily"

REQUIRED_COLUMNS = [
    "trade_date",
    "industry_id",
    "industry_name",
    "mainline_strength_score",
    "leader_density",
    "avg_leader_score",
    "top_leader_score",
    "breakout_ratio",
    "trend_alignment",
    "breadth_score",
    "acceleration_score",
    "lifecycle_bonus",
    "rank_in_market",
    "is_active_mainline",
]

SCORE_COLUMNS = [
    "mainline_strength_score",
    "leader_density",
    "avg_leader_score",
    "top_leader_score",
    "breakout_ratio",
    "trend_alignment",
    "breadth_score",
    "acceleration_score",
    "lifecycle_bonus",
]

# Source tables used by scripts/build_cn_stock_mainline_strength_daily.py.
# cn_mainline_lifecycle_daily is optional in the builder, but if present it should
# have these columns so lifecycle_bonus can be applied.
SOURCE_TABLES: dict[str, list[str]] = {
    "cn_stock_leader_score_daily": [
        "trade_date", "symbol", "leader_score", "leader_bucket",
        "breakout_strength", "breakout_ready", "industry_id", "industry_name",
    ],
    "cn_stock_daily_price": [
        "SYMBOL", "TRADE_DATE", "CHG_PCT",
    ],
    "cn_local_industry_map_hist": [
        "industry_id", "industry_name", "industry_level",
    ],
}

OPTIONAL_SOURCE_TABLES: dict[str, list[str]] = {
    "cn_mainline_lifecycle_daily": [
        "trade_date", "mainline_id", "mainline_name", "lifecycle_state",
        "lifecycle_score", "mainline_strength", "rotation_rank",
    ],
}

ACTIVE_MAINLINE_THRESHOLD = 0.65

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="GrowthAlpha V8 — Validate cn_stock_mainline_strength_daily"
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
    def __init__(self) -> None:
        self.checks: list[dict[str, Any]] = []
        self.failures: int = 0

    def add(self, name: str, passed: bool, detail: str = "") -> None:
        self.checks.append({"name": name, "passed": passed, "detail": detail})
        if not passed:
            self.failures += 1
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))

    def info(self, name: str, detail: str = "") -> None:
        print(f"  [INFO] {name}" + (f" — {detail}" if detail else ""))

    def summary(self) -> str:
        total = len(self.checks)
        passed = total - self.failures
        return f"Validation: {passed}/{total} passed, {self.failures} failed"


def _check_one_source_table(
    engine: Engine,
    db_name: str,
    result: ValidationResult,
    table_name: str,
    required_cols: list[str],
    *,
    optional: bool,
    verbose: bool,
) -> bool:
    tbl_ok = table_exists(engine, db_name, table_name)
    if optional and not tbl_ok:
        result.info(f"Optional source table [{table_name}] missing", "builder can run without it")
        return True

    result.add(f"Source table [{table_name}] exists", tbl_ok)
    if not tbl_ok:
        return False

    missing_cols = [col for col in required_cols if not column_exists(engine, db_name, table_name, col)]
    cols_ok = len(missing_cols) == 0
    if optional and not cols_ok:
        result.info(
            f"Optional source columns [{table_name}] incomplete",
            f"missing={missing_cols}; builder can run without lifecycle bonus",
        )
        return True

    result.add(
        f"  Columns in [{table_name}]",
        cols_ok,
        detail=f"missing={missing_cols}" if missing_cols else "all present",
    )

    case_mismatches = check_column_case_mismatch(engine, db_name, table_name, required_cols)
    case_ok = all("MISSING" not in m for m in case_mismatches) and len(case_mismatches) == 0
    if case_mismatches and not optional:
        result.add(
            f"  Case check [{table_name}]",
            case_ok,
            detail=f"mismatches={case_mismatches}",
        )

    if verbose:
        actual_cols = get_actual_columns(engine, db_name, table_name)
        extra_cols = [c for c in actual_cols if c.lower() not in {x.lower() for x in required_cols}]
        if extra_cols:
            print(f"    [INFO] Extra columns in [{table_name}]: {extra_cols}")
    return cols_ok or optional


def check_source_tables(
    engine: Engine, db_name: str, result: ValidationResult, verbose: bool
) -> bool:
    print()
    print("── Source Table Pre-checks ──────────────────────────────────")
    all_tables_ok = True

    for table_name, required_cols in SOURCE_TABLES.items():
        ok = _check_one_source_table(
            engine, db_name, result, table_name, required_cols, optional=False, verbose=verbose
        )
        all_tables_ok = all_tables_ok and ok

    for table_name, required_cols in OPTIONAL_SOURCE_TABLES.items():
        _check_one_source_table(
            engine, db_name, result, table_name, required_cols, optional=True, verbose=verbose
        )

    return all_tables_ok


def _parse_date(value: str, default_today: bool = False) -> date:
    if value:
        return datetime.strptime(value, "%Y-%m-%d").date() if "-" in value else datetime.strptime(value, "%Y%m%d").date()
    if default_today:
        return date.today()
    raise ValueError("date value is required")


def validate(args: argparse.Namespace) -> int:
    verbose = args.verbose
    fail_on_empty = args.fail_on_empty
    min_rows = args.min_rows
    skip_source = args.skip_source_checks

    start = _parse_date(args.start)
    end = _parse_date(args.end, default_today=True)

    db_password = args.db_password or os.getenv("ASHARE_MYSQL_PASSWORD", "sec_Bobo123")
    engine = build_engine(args.db_host, args.db_port, args.db_user, db_password, args.db_name)

    result = ValidationResult()

    print(f"Validating {OUTPUT_TABLE} [{start} ~ {end}] on {args.db_name}")
    print(f"Legacy note: {LEGACY_TABLE} is treated as fallback/legacy and is not the validation target.")

    if not skip_source:
        source_ok = check_source_tables(engine, args.db_name, result, verbose)
        if not source_ok:
            print()
            print("  ⚠ Some required source tables are missing. The output table may be incomplete.")
            print("  Use --skip-source-checks to bypass this check.")
            print()
    elif verbose:
        print("  [SKIP] Source table pre-checks (--skip-source-checks)")

    print()
    print("── Output Table Checks ──────────────────────────────────────")

    tbl_exists = table_exists(engine, args.db_name, OUTPUT_TABLE)
    result.add(f"Output table [{OUTPUT_TABLE}] exists", tbl_exists)
    if not tbl_exists:
        print("\n" + result.summary())
        return 1

    missing_cols = [col for col in REQUIRED_COLUMNS if not column_exists(engine, args.db_name, OUTPUT_TABLE, col)]
    cols_ok = len(missing_cols) == 0
    result.add(
        "Required columns exist",
        cols_ok,
        detail=f"missing={missing_cols}" if missing_cols else "all present",
    )
    if not cols_ok:
        print("\n" + result.summary())
        return 1

    case_mismatches = check_column_case_mismatch(engine, args.db_name, OUTPUT_TABLE, REQUIRED_COLUMNS)
    case_ok = len(case_mismatches) == 0
    result.add(
        "Column case match",
        case_ok,
        detail=f"mismatches={case_mismatches}" if case_mismatches else "all match",
    )

    count_sql = f"""
        SELECT COUNT(*) AS cnt
        FROM {OUTPUT_TABLE}
        WHERE trade_date BETWEEN :start AND :end
    """
    row_count = int(fetch_df(engine, count_sql, {"start": start, "end": end}).iloc[0, 0])
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

    data_sql = f"""
        SELECT *
        FROM {OUTPUT_TABLE}
        WHERE trade_date BETWEEN :start AND :end
        ORDER BY trade_date, rank_in_market
    """
    df = fetch_df(engine, data_sql, {"start": start, "end": end})

    # Main score and sub-scores should be normalized to [0, 1].
    score_col = df["mainline_strength_score"]
    out_of_range = int(((score_col < 0.0) | (score_col > 1.0)).sum())
    result.add(
        "mainline_strength_score in [0, 1]",
        out_of_range == 0,
        detail=f"out_of_range={out_of_range}" if out_of_range > 0 else "all valid",
    )

    subscore_out_of_range = 0
    for col in SCORE_COLUMNS:
        if col in df.columns:
            subscore_out_of_range += int(((df[col] < 0.0) | (df[col] > 1.0)).sum())
    result.add(
        "All sub-scores in [0, 1]",
        subscore_out_of_range == 0,
        detail=f"out_of_range={subscore_out_of_range}" if subscore_out_of_range > 0 else "all valid",
    )

    dup_count = int(df.duplicated(subset=["trade_date", "industry_id"]).sum())
    result.add(
        "No duplicate PKs",
        dup_count == 0,
        detail=f"duplicates={dup_count}" if dup_count > 0 else "all unique",
    )

    # rank_in_market is produced with pandas rank(method="min") in the builder.
    # Tied scores are therefore valid and can create non-contiguous ranks such as
    # 1, 2, 2, 4.  Validate rank integrity instead of requiring strict sequence.
    rank_issue_dates = 0
    rank_null_rows = 0
    rank_nonpositive_rows = 0
    rank_overflow_rows = 0
    for _, group in df.groupby("trade_date"):
        ranks_raw = group["rank_in_market"]
        rank_null_rows += int(ranks_raw.isna().sum())
        ranks = ranks_raw.dropna().astype(int)
        n_rows = len(group)
        nonpositive = int((ranks < 1).sum())
        overflow = int((ranks > n_rows).sum())
        if nonpositive or overflow or ranks_raw.isna().any():
            rank_issue_dates += 1
            rank_nonpositive_rows += nonpositive
            rank_overflow_rows += overflow
    rank_ok = rank_issue_dates == 0
    result.add(
        "rank_in_market valid per date",
        rank_ok,
        detail=(
            f"dates_with_issues={rank_issue_dates}, null_rows={rank_null_rows}, "
            f"nonpositive_rows={rank_nonpositive_rows}, overflow_rows={rank_overflow_rows}"
            if not rank_ok
            else "all ranks non-null and within [1, daily_row_count]"
        ),
    )

    if "is_active_mainline" in df.columns:
        expected_active = (df["mainline_strength_score"] >= ACTIVE_MAINLINE_THRESHOLD).astype(int)
        active_mismatch = int((df["is_active_mainline"].fillna(0).astype(int) != expected_active).sum())
        result.add(
            "is_active_mainline matches threshold",
            active_mismatch == 0,
            detail=f"mismatches={active_mismatch}" if active_mismatch > 0 else "all valid",
        )

    if verbose:
        print()
        print("  Active Mainline Distribution:")
        active_counts = df["is_active_mainline"].value_counts(dropna=False).to_dict()
        print(f"    {active_counts}")

    print()
    print(f"  Total rows checked: {len(df)}")
    print(f"  Date range: {df['trade_date'].min()} ~ {df['trade_date'].max()}")
    print(f"  Unique industries: {df['industry_id'].nunique()}")
    print(f"  Unique dates: {df['trade_date'].nunique()}")
    print(f"  Mean mainline_strength_score: {df['mainline_strength_score'].mean():.4f}")
    print(f"  Median mainline_strength_score: {df['mainline_strength_score'].median():.4f}")
    print()
    print(f"  {result.summary()}")

    return 0 if result.failures == 0 else 1


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    print("=" * 60)
    print("  GrowthAlpha V8 — Mainline Strength Validation")
    print("=" * 60)

    exit_code = validate(args)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
