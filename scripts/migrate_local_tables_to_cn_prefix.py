"""
scripts/migrate_local_tables_to_cn_prefix.py
==============================================
Migrate data from non-cn local_* tables to cn_local_* tables.

Table mapping:
  local_industry_map_hist      -> cn_local_industry_map_hist
  local_industry_master        -> cn_local_industry_master
  local_industry_proxy_daily   -> cn_local_industry_proxy_daily
  local_stock_balancesheet_q   -> cn_local_stock_balancesheet_q
  local_stock_fina_indicator_q -> cn_local_stock_fina_indicator_q
  local_stock_income_q         -> cn_local_stock_income_q

Usage:
  # Dry-run (show what would happen)
  python scripts/migrate_local_tables_to_cn_prefix.py --db-name cn_market_red --dry-run

  # Execute migration
  python scripts/migrate_local_tables_to_cn_prefix.py --db-name cn_market_red

  # Execute and drop old tables after verification
  python scripts/migrate_local_tables_to_cn_prefix.py --db-name cn_market_red --drop-old-after-verify
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

# ---------------------------------------------------------------------------
# Table mapping: non-cn -> cn_*
# ---------------------------------------------------------------------------

TABLE_MAP: dict[str, str] = {
    "local_industry_map_hist": "cn_local_industry_map_hist",
    "local_industry_master": "cn_local_industry_master",
    "local_industry_proxy_daily": "cn_local_industry_proxy_daily",
    "local_stock_balancesheet_q": "cn_local_stock_balancesheet_q",
    "local_stock_fina_indicator_q": "cn_local_stock_fina_indicator_q",
    "local_stock_income_q": "cn_local_stock_income_q",
}

# ---------------------------------------------------------------------------
# DDL for cn_* tables (CREATE IF NOT EXISTS)
# ---------------------------------------------------------------------------

CN_DDL: dict[str, str] = {
    "cn_local_industry_master": """
        CREATE TABLE IF NOT EXISTS `cn_local_industry_master` (
            `industry_id` varchar(32) NOT NULL,
            `industry_name` varchar(128) NOT NULL,
            `industry_level` varchar(8) NOT NULL,
            `parent_id` varchar(32) DEFAULT NULL,
            `src` varchar(32) NOT NULL,
            `created_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
            `updated_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            PRIMARY KEY (`industry_id`),
            KEY `idx_cn_local_industry_master_level` (`industry_level`),
            KEY `idx_cn_local_industry_master_parent` (`parent_id`),
            KEY `idx_cn_local_industry_master_src` (`src`)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """,
    "cn_local_industry_map_hist": """
        CREATE TABLE IF NOT EXISTS `cn_local_industry_map_hist` (
            `symbol` varchar(10) NOT NULL,
            `industry_id` varchar(32) NOT NULL,
            `industry_name` varchar(128) NOT NULL,
            `industry_level` varchar(8) NOT NULL,
            `in_date` date NOT NULL,
            `out_date` date DEFAULT NULL,
            `is_current` tinyint(1) NOT NULL DEFAULT 0,
            `updated_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            PRIMARY KEY (`symbol`, `industry_id`, `in_date`),
            KEY `idx_cn_local_industry_map_hist_industry_date` (`industry_id`, `in_date`, `out_date`),
            KEY `idx_cn_local_industry_map_hist_symbol_date` (`symbol`, `in_date`, `out_date`),
            KEY `idx_cn_local_industry_map_hist_current` (`industry_id`, `is_current`)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """,
    "cn_local_industry_proxy_daily": """
        CREATE TABLE IF NOT EXISTS `cn_local_industry_proxy_daily` (
            `industry_id` varchar(32) NOT NULL,
            `trade_date` date NOT NULL,
            `member_count` int NOT NULL,
            `ret_eqw` decimal(18,8) DEFAULT NULL,
            `amount_total` decimal(24,4) DEFAULT NULL,
            `turnover_avg` decimal(18,6) DEFAULT NULL,
            `market_cap_total` decimal(24,4) DEFAULT NULL,
            `leader_return` decimal(18,8) DEFAULT NULL,
            `top5_concentration` decimal(18,8) DEFAULT NULL,
            `created_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
            `updated_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            PRIMARY KEY (`industry_id`, `trade_date`),
            KEY `idx_cn_local_industry_proxy_daily_trade_date` (`trade_date`)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """,
    "cn_local_stock_income_q": """
        CREATE TABLE IF NOT EXISTS `cn_local_stock_income_q` (
            `symbol` varchar(10) NOT NULL,
            `end_date` date NOT NULL,
            `ann_date` date DEFAULT NULL,
            `f_ann_date` date DEFAULT NULL,
            `report_type` varchar(32) DEFAULT NULL,
            `total_revenue` decimal(24,4) DEFAULT NULL,
            `revenue` decimal(24,4) DEFAULT NULL,
            `n_income_attr_p` decimal(24,4) DEFAULT NULL,
            `source` varchar(64) DEFAULT NULL,
            `created_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
            `updated_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            PRIMARY KEY (`symbol`, `end_date`),
            KEY `idx_cn_local_stock_income_q_ann_date` (`ann_date`)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """,
    "cn_local_stock_balancesheet_q": """
        CREATE TABLE IF NOT EXISTS `cn_local_stock_balancesheet_q` (
            `symbol` varchar(10) NOT NULL,
            `end_date` date NOT NULL,
            `ann_date` date DEFAULT NULL,
            `f_ann_date` date DEFAULT NULL,
            `report_type` varchar(32) DEFAULT NULL,
            `inventory` decimal(24,4) DEFAULT NULL,
            `contract_liability` decimal(24,4) DEFAULT NULL,
            `fixed_assets` decimal(24,4) DEFAULT NULL,
            `total_assets` decimal(24,4) DEFAULT NULL,
            `total_liab` decimal(24,4) DEFAULT NULL,
            `source` varchar(64) DEFAULT NULL,
            `created_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
            `updated_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            PRIMARY KEY (`symbol`, `end_date`),
            KEY `idx_cn_local_stock_balancesheet_q_ann_date` (`ann_date`)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """,
    "cn_local_stock_fina_indicator_q": """
        CREATE TABLE IF NOT EXISTS `cn_local_stock_fina_indicator_q` (
            `symbol` varchar(10) NOT NULL,
            `end_date` date NOT NULL,
            `ann_date` date DEFAULT NULL,
            `report_type` varchar(32) DEFAULT NULL,
            `revenue_yoy` decimal(18,6) DEFAULT NULL,
            `profit_yoy` decimal(18,6) DEFAULT NULL,
            `roe` decimal(18,6) DEFAULT NULL,
            `gross_margin` decimal(18,6) DEFAULT NULL,
            `debt_to_assets` decimal(18,6) DEFAULT NULL,
            `ocfps` decimal(18,6) DEFAULT NULL,
            `source` varchar(64) DEFAULT NULL,
            `created_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
            `updated_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            PRIMARY KEY (`symbol`, `end_date`),
            KEY `idx_cn_local_stock_fina_indicator_q_ann_date` (`ann_date`)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """,
}


def build_engine_from_args(args: argparse.Namespace) -> Engine:
    """Build SQLAlchemy engine from args or environment."""
    user = args.user or "cn_opr_red"
    password = args.password or "sec_Bobo123"
    host = args.host or "localhost"
    port = args.port or 3306
    db = args.db_name or "cn_market_red"
    url = f"mysql+pymysql://{user}:{password}@{host}:{port}/{db}?charset=utf8mb4"
    return create_engine(url, pool_pre_ping=True)


def table_exists(engine: Engine, table_name: str) -> bool:
    """Check if a table exists in the current database."""
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
    """Get row count for a table."""
    with engine.connect() as conn:
        rs = conn.execute(text(f"SELECT COUNT(*) FROM `{table_name}`"))
        return int(rs.scalar() or 0)


def ensure_cn_table(engine: Engine, cn_table: str, dry_run: bool) -> bool:
    """Ensure cn_* table exists. Returns True if created."""
    if table_exists(engine, cn_table):
        return False
    ddl = CN_DDL.get(cn_table)
    if not ddl:
        print(f"  [SKIP] No DDL defined for {cn_table}")
        return False
    if dry_run:
        print(f"  [DRY-RUN] Would CREATE TABLE IF NOT EXISTS `{cn_table}`")
        return False
    with engine.begin() as conn:
        conn.execute(text(ddl))
    print(f"  [OK] Created table `{cn_table}`")
    return True


def migrate_table(
    engine: Engine,
    src_table: str,
    dst_table: str,
    dry_run: bool,
) -> tuple[int, int]:
    """Migrate data from src_table to dst_table. Returns (src_count, dst_count_before)."""
    src_count = row_count(engine, src_table) if table_exists(engine, src_table) else 0
    dst_count_before = row_count(engine, dst_table) if table_exists(engine, dst_table) else 0

    if src_count == 0:
        print(f"  [SKIP] Source `{src_table}` has 0 rows or does not exist")
        return (src_count, dst_count_before)

    if dry_run:
        print(
            f"  [DRY-RUN] Would migrate {src_count:,} rows from `{src_table}` -> `{dst_table}`"
        )
        return (src_count, dst_count_before)

    # Use INSERT IGNORE to avoid duplicate key conflicts
    with engine.begin() as conn:
        rs = conn.execute(
            text(f"INSERT IGNORE INTO `{dst_table}` SELECT * FROM `{src_table}`")
        )
        inserted = int(rs.rowcount or 0)

    dst_count_after = row_count(engine, dst_table)
    print(
        f"  [OK] `{src_table}` -> `{dst_table}`: "
        f"src={src_count:,} dst_before={dst_count_before:,} "
        f"inserted={inserted:,} dst_after={dst_count_after:,}"
    )
    return (src_count, dst_count_before)


def drop_table(engine: Engine, table_name: str) -> None:
    """Drop a table."""
    with engine.begin() as conn:
        conn.execute(text(f"DROP TABLE IF EXISTS `{table_name}`"))
    print(f"  [OK] Dropped `{table_name}`")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Migrate non-cn local_* tables to cn_local_* prefix."
    )
    parser.add_argument("--db-name", default="cn_market_red", help="Database name")
    parser.add_argument("--user", default="", help="MySQL user")
    parser.add_argument("--password", default="", help="MySQL password")
    parser.add_argument("--host", default="localhost", help="MySQL host")
    parser.add_argument("--port", type=int, default=3306, help="MySQL port")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )
    parser.add_argument(
        "--drop-old-after-verify",
        action="store_true",
        help="Drop non-cn tables after verifying row counts match",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    engine = build_engine_from_args(args)

    print(f"=== Migration: non-cn local_* -> cn_local_* ===")
    print(f"  Database: {args.db_name}")
    print(f"  Dry-run:  {args.dry_run}")
    print(f"  Drop old: {args.drop_old_after_verify}")
    print()

    # Phase 1: Ensure cn_* tables exist
    print("--- Phase 1: Ensure cn_* tables ---")
    for src, dst in TABLE_MAP.items():
        ensure_cn_table(engine, dst, args.dry_run)
    print()

    # Phase 2: Migrate data
    print("--- Phase 2: Migrate data ---")
    migration_results: dict[str, tuple[int, int]] = {}
    for src, dst in TABLE_MAP.items():
        print(f"  Table: {src} -> {dst}")
        result = migrate_table(engine, src, dst, args.dry_run)
        migration_results[src] = result
    print()

    # Phase 3: Verify row counts
    print("--- Phase 3: Verify row counts ---")
    all_aligned = True
    for src, dst in TABLE_MAP.items():
        src_count = migration_results[src][0]
        dst_count = row_count(engine, dst) if table_exists(engine, dst) else 0
        aligned = dst_count >= src_count
        status = "OK" if aligned else "MISMATCH"
        if not aligned:
            all_aligned = False
        print(f"  {src:<40s} src={src_count:>12,}  {dst:<40s} dst={dst_count:>12,}  [{status}]")
    print()

    # Phase 4: Drop old tables (only if --drop-old-after-verify and counts align)
    if args.drop_old_after_verify and all_aligned:
        print("--- Phase 4: Drop non-cn tables (--drop-old-after-verify) ---")
        for src in TABLE_MAP:
            if table_exists(engine, src):
                drop_table(engine, src)
            else:
                print(f"  [SKIP] `{src}` does not exist")
        print()
    elif args.drop_old_after_verify and not all_aligned:
        print("[WARN] --drop-old-after-verify specified but row counts do not align. Not dropping.")
        print()

    print("=== Migration complete ===")


if __name__ == "__main__":
    main()
