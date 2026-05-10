"""
scripts/validate_cn_prefix_table_migration.py
===============================================
Validate that all non-cn local_* tables have been migrated to cn_local_* prefix.

Checks:
1. Row count comparison: cn_* table row_count >= non-cn table row_count
2. cn_* table key columns exist
3. cn_* table primary key has no duplicates
4. cn_stock_fundamental_daily can be generated from cn_* raw tables
5. Source code scan: no non-cn local_* references remain in scripts/ or docs/DDL/

Usage:
  python scripts/validate_cn_prefix_table_migration.py --db-name cn_market_red
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

# ---------------------------------------------------------------------------
# Table pairs: (non-cn, cn_*)
# ---------------------------------------------------------------------------

TABLE_PAIRS: list[tuple[str, str]] = [
    ("local_industry_map_hist", "cn_local_industry_map_hist"),
    ("local_industry_master", "cn_local_industry_master"),
    ("local_industry_proxy_daily", "cn_local_industry_proxy_daily"),
    ("local_stock_balancesheet_q", "cn_local_stock_balancesheet_q"),
    ("local_stock_fina_indicator_q", "cn_local_stock_fina_indicator_q"),
    ("local_stock_income_q", "cn_local_stock_income_q"),
]

# Key columns to verify exist in cn_* tables
CN_KEY_COLUMNS: dict[str, list[str]] = {
    "cn_local_industry_master": ["industry_id", "industry_name", "industry_level", "src"],
    "cn_local_industry_map_hist": ["symbol", "industry_id", "in_date", "out_date", "is_current"],
    "cn_local_industry_proxy_daily": ["industry_id", "trade_date", "member_count", "ret_eqw"],
    "cn_local_stock_income_q": ["symbol", "end_date", "ann_date", "total_revenue", "n_income_attr_p"],
    "cn_local_stock_balancesheet_q": ["symbol", "end_date", "ann_date", "inventory", "fixed_assets", "total_assets", "total_liab"],
    "cn_local_stock_fina_indicator_q": ["symbol", "end_date", "ann_date", "revenue_yoy", "profit_yoy", "roe"],
}

# Forbidden patterns in source code
FORBIDDEN_PATTERNS: list[str] = [
    "local_stock_income_q",
    "local_stock_balancesheet_q",
    "local_stock_fina_indicator_q",
    "local_industry_map_hist",
    "local_industry_master",
    "local_industry_proxy_daily",
]

# Directories to scan for forbidden patterns
SCAN_DIRS: list[str] = ["scripts", "docs/DDL"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def build_engine_from_args(args: argparse.Namespace) -> Engine:
    user = args.user or "cn_opr_red"
    password = args.password or "sec_Bobo123"
    host = args.host or "localhost"
    port = args.port or 3306
    db = args.db_name or "cn_market_red"
    url = f"mysql+pymysql://{user}:{password}@{host}:{port}/{db}?charset=utf8mb4"
    return create_engine(url, pool_pre_ping=True)


def table_exists(engine: Engine, table_name: str) -> bool:
    with engine.connect() as conn:
        rs = conn.execute(
            text(
                "SELECT COUNT(*) FROM information_schema.tables "
                "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = :t"
            ),
            {"t": table_name},
        )
        return rs.scalar() > 0


def row_count(engine: Engine, table_name: str) -> int:
    with engine.connect() as conn:
        rs = conn.execute(text(f"SELECT COUNT(*) FROM `{table_name}`"))
        return int(rs.scalar() or 0)


def has_duplicate_pk(engine: Engine, table_name: str) -> bool:
    """Check if primary key has duplicates by comparing COUNT(*) vs COUNT(DISTINCT pk_columns)."""
    pk_map = {
        "cn_local_industry_master": "industry_id",
        "cn_local_industry_map_hist": "CONCAT(symbol, '-', industry_id, '-', in_date)",
        "cn_local_industry_proxy_daily": "CONCAT(industry_id, '-', trade_date)",
        "cn_local_stock_income_q": "CONCAT(symbol, '-', end_date)",
        "cn_local_stock_balancesheet_q": "CONCAT(symbol, '-', end_date)",
        "cn_local_stock_fina_indicator_q": "CONCAT(symbol, '-', end_date)",
    }
    pk_expr = pk_map.get(table_name)
    if not pk_expr:
        return False
    with engine.connect() as conn:
        total = conn.execute(text(f"SELECT COUNT(*) FROM `{table_name}`")).scalar()
        distinct = conn.execute(
            text(f"SELECT COUNT(DISTINCT {pk_expr}) FROM `{table_name}`")
        ).scalar()
    return total > distinct


def check_stock_fundamental_daily_readiness(engine: Engine) -> list[str]:
    """Check if cn_stock_fundamental_daily can be generated from cn_* raw tables."""
    issues: list[str] = []
    required_tables = [
        "cn_local_stock_income_q",
        "cn_local_stock_balancesheet_q",
        "cn_local_stock_fina_indicator_q",
        "cn_stock_daily_price",
    ]
    for tbl in required_tables:
        if not table_exists(engine, tbl):
            issues.append(f"Required table `{tbl}` does not exist")

    # Check that cn_local_stock_*_q tables have data
    for tbl in ["cn_local_stock_income_q", "cn_local_stock_balancesheet_q", "cn_local_stock_fina_indicator_q"]:
        if table_exists(engine, tbl):
            cnt = row_count(engine, tbl)
            if cnt == 0:
                issues.append(f"Table `{tbl}` exists but has 0 rows")

    # Check cn_stock_fundamental_daily exists
    if not table_exists(engine, "cn_stock_fundamental_daily"):
        issues.append("Table `cn_stock_fundamental_daily` does not exist (will be created by builder)")

    return issues


def scan_source_files() -> list[str]:
    """Scan scripts/ and docs/DDL/ for forbidden non-cn table references."""
    violations: list[str] = []
    for scan_dir in SCAN_DIRS:
        scan_path = ROOT / scan_dir
        if not scan_path.is_dir():
            continue
        for file_path in sorted(scan_path.rglob("*")):
            if not file_path.is_file():
                continue
            # Skip binary files and non-source files
            if file_path.suffix not in (".py", ".sql", ".bat", ".ps1", ".sh"):
                continue
            try:
                content = file_path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            for pattern in FORBIDDEN_PATTERNS:
                # Only match non-cn prefixed occurrences (not cn_local_*)
                for match in re.finditer(rf"(?<!cn_){re.escape(pattern)}", content):
                    line_num = content[: match.start()].count("\n") + 1
                    rel_path = file_path.relative_to(ROOT)
                    violations.append(
                        f"  {rel_path}:{line_num}: found '{pattern}'"
                    )
    return violations


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate cn_* prefix table migration."
    )
    parser.add_argument("--db-name", default="cn_market_red", help="Database name")
    parser.add_argument("--user", default="", help="MySQL user")
    parser.add_argument("--password", default="", help="MySQL password")
    parser.add_argument("--host", default="localhost", help="MySQL host")
    parser.add_argument("--port", type=int, default=3306, help="MySQL port")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    engine = build_engine_from_args(args)

    all_passed = True
    total_checks = 0
    failed_checks = 0

    print("=" * 70)
    print("VALIDATE CN_PREFIX TABLE MIGRATION")
    print("=" * 70)
    print(f"  Database: {args.db_name}")
    print()

    # -----------------------------------------------------------------------
    # Check 1: Row count comparison
    # -----------------------------------------------------------------------
    print("--- Check 1: Row count comparison (cn_* >= non-cn) ---")
    total_checks += 1
    check1_failed = False
    for src, dst in TABLE_PAIRS:
        src_exists = table_exists(engine, src)
        dst_exists = table_exists(engine, dst)
        src_cnt = row_count(engine, src) if src_exists else 0
        dst_cnt = row_count(engine, dst) if dst_exists else 0

        if not dst_exists:
            print(f"  [FAIL] `{dst}` does not exist")
            check1_failed = True
            continue

        if dst_cnt >= src_cnt:
            print(f"  [OK] `{src}` ({src_cnt:,}) -> `{dst}` ({dst_cnt:,})")
        else:
            print(
                f"  [FAIL] `{dst}` ({dst_cnt:,}) < `{src}` ({src_cnt:,}) — data loss risk"
            )
            check1_failed = True

    if check1_failed:
        all_passed = False
        failed_checks += 1
    print()

    # -----------------------------------------------------------------------
    # Check 2: Key columns exist in cn_* tables
    # -----------------------------------------------------------------------
    print("--- Check 2: Key columns in cn_* tables ---")
    total_checks += 1
    check2_failed = False
    for tbl, expected_cols in CN_KEY_COLUMNS.items():
        if not table_exists(engine, tbl):
            print(f"  [FAIL] `{tbl}` does not exist — skipping column check")
            check2_failed = True
            continue

        with engine.connect() as conn:
            rs = conn.execute(
                text(
                    "SELECT COLUMN_NAME FROM information_schema.columns "
                    "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = :t"
                ),
                {"t": tbl},
            )
            actual_cols = {row[0].lower() for row in rs}

        missing = [c for c in expected_cols if c.lower() not in actual_cols]
        if missing:
            print(f"  [FAIL] `{tbl}` missing columns: {missing}")
            check2_failed = True
        else:
            print(f"  [OK] `{tbl}` has all expected columns")

    if check2_failed:
        all_passed = False
        failed_checks += 1
    print()

    # -----------------------------------------------------------------------
    # Check 3: No duplicate primary keys in cn_* tables
    # -----------------------------------------------------------------------
    print("--- Check 3: Primary key uniqueness ---")
    total_checks += 1
    check3_failed = False
    for tbl, _ in TABLE_PAIRS:
        dst = tbl.replace("local_", "cn_local_", 1) if tbl.startswith("local_") else f"cn_{tbl}"
        if not table_exists(engine, dst):
            continue
        if has_duplicate_pk(engine, dst):
            print(f"  [FAIL] `{dst}` has duplicate primary keys")
            check3_failed = True
        else:
            print(f"  [OK] `{dst}` primary key is unique")

    if check3_failed:
        all_passed = False
        failed_checks += 1
    print()

    # -----------------------------------------------------------------------
    # Check 4: cn_stock_fundamental_daily readiness
    # -----------------------------------------------------------------------
    print("--- Check 4: cn_stock_fundamental_daily readiness ---")
    total_checks += 1
    check4_issues = check_stock_fundamental_daily_readiness(engine)
    if check4_issues:
        for issue in check4_issues:
            print(f"  [FAIL] {issue}")
        all_passed = False
        failed_checks += 1
    else:
        print("  [OK] cn_stock_fundamental_daily can be generated from cn_* raw tables")
    print()

    # -----------------------------------------------------------------------
    # Check 5: Source code scan for forbidden patterns
    # -----------------------------------------------------------------------
    print("--- Check 5: Source code scan (no non-cn local_* references) ---")
    total_checks += 1
    violations = scan_source_files()
    if violations:
        print(f"  [FAIL] Found {len(violations)} forbidden non-cn references:")
        for v in violations:
            print(f"    {v}")
        all_passed = False
        failed_checks += 1
    else:
        print("  [OK] No forbidden non-cn local_* references found in scripts/ or docs/DDL/")
    print()

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("=" * 70)
    if all_passed:
        print(f"RESULT: PASSED ({total_checks}/{total_checks} checks passed)")
    else:
        print(
            f"RESULT: FAILED ({failed_checks}/{total_checks} checks failed)"
        )
    print("=" * 70)

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
