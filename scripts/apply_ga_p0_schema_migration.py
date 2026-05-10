"""
scripts/apply_ga_p0_schema_migration.py
=========================================
GrowthAlpha V8 — P0/P2 Mainline Context & Lifecycle Schema Migration.

Idempotent migration script that:

  1. Checks if GA-layer tables exist (cn_ga_mainline_radar_daily,
     cn_ga_market_pulse_daily, cn_ga_stock_role_map_daily)
  2. Adds missing columns to each table (checks INFORMATION_SCHEMA.COLUMNS first)
  3. Creates cn_ga_market_context_daily if not exists
  4. Creates mainline_lifecycle_daily (P2) if not exists
  5. Creates indexes if not exist

Safe to run multiple times — never drops data or columns.

Usage:
  python scripts/apply_ga_p0_schema_migration.py ^
      --db-host 127.0.0.1 --db-port 3306 --db-user root ^
      --db-password YOUR_PASSWORD --db-name cn_market_red
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, URL

# ---------------------------------------------------------------------------
# Progress Tracker
# ---------------------------------------------------------------------------


class ProgressTracker:
    """Simple progress bar for CLI operations."""

    def __init__(self, total: int, prefix: str = "Progress", bar_len: int = 40):
        self.total = total
        self.prefix = prefix
        self.bar_len = bar_len
        self.current = 0
        self._start_time = time.time()

    def update(self, n: int = 1, suffix: str = "") -> None:
        """Advance by n steps and redraw."""
        self.current += n
        self._draw(suffix)

    def set_suffix(self, suffix: str) -> None:
        """Redraw with a new suffix without advancing."""
        self._draw(suffix)

    def _draw(self, suffix: str) -> None:
        frac = self.current / max(self.total, 1)
        filled = int(self.bar_len * frac)
        bar = "#" * filled + "-" * (self.bar_len - filled)
        elapsed = time.time() - self._start_time
        pct = frac * 100
        print(
            f"\r  {self.prefix} |{bar}| {pct:5.1f}%  [{self.current}/{self.total}]  {suffix}  ({elapsed:.1f}s)",
            end="",
            flush=True,
        )

    def finish(self, suffix: str = "Done") -> None:
        self.current = self.total
        self._draw(suffix)
        print()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_DB_NAME = "cn_market_red"
SQL_FILE = Path(__file__).resolve().parents[1] / "sql" / "ddl" / "ga_p0_mainline_context_schema.sql"
P2_LIFECYCLE_SQL_FILE = Path(__file__).resolve().parents[1] / "sql" / "ddl" / "ga_p2_mainline_lifecycle_schema.sql"

# ---------------------------------------------------------------------------
# Column definitions per table
# ---------------------------------------------------------------------------
# Each entry: (table_name, [(column_name, column_type), ...])

MAINLINE_RADAR_COLUMNS: list[tuple[str, str]] = [
    ("rs_60d", "DECIMAL(10,4) DEFAULT NULL"),
    ("rs_120d", "DECIMAL(10,4) DEFAULT NULL"),
    ("trend_alignment_score", "DECIMAL(10,4) DEFAULT NULL"),
    ("rotation_rank", "INT DEFAULT NULL"),
    ("heat_percentile_5d", "DECIMAL(10,4) DEFAULT NULL"),
    ("breakout_ratio", "DECIMAL(10,4) DEFAULT NULL"),
    ("new_high_ratio", "DECIMAL(10,4) DEFAULT NULL"),
    ("strong_stock_count", "INT DEFAULT NULL"),
    ("leader_density", "DECIMAL(10,4) DEFAULT NULL"),
    ("mainline_phase", "VARCHAR(32) DEFAULT NULL"),
    ("mainline_confidence", "DECIMAL(10,4) DEFAULT NULL"),
]

MARKET_PULSE_COLUMNS: list[tuple[str, str]] = [
    ("bullish_industry_ratio", "DECIMAL(10,4) DEFAULT NULL"),
    ("neutral_industry_ratio", "DECIMAL(10,4) DEFAULT NULL"),
    ("bearish_industry_ratio", "DECIMAL(10,4) DEFAULT NULL"),
    ("rotation_speed", "DECIMAL(10,4) DEFAULT NULL"),
    ("mainline_stability", "DECIMAL(10,4) DEFAULT NULL"),
    ("trend_alignment_avg", "DECIMAL(10,4) DEFAULT NULL"),
    ("industry_expansion_breadth", "DECIMAL(10,4) DEFAULT NULL"),
    ("top_mainline_count", "INT DEFAULT NULL"),
    ("market_phase", "VARCHAR(32) DEFAULT NULL"),
]

STOCK_ROLE_COLUMNS: list[tuple[str, str]] = [
    ("breakout_strength", "DECIMAL(10,4) DEFAULT NULL"),
    ("new_high_flag", "TINYINT(1) DEFAULT NULL"),
    ("trend_structure_score", "DECIMAL(10,4) DEFAULT NULL"),
    ("volume_expansion_score", "DECIMAL(10,4) DEFAULT NULL"),
    ("role_lifecycle_state", "VARCHAR(32) DEFAULT NULL"),
    ("candidate_action", "VARCHAR(64) DEFAULT NULL"),
]

# Index definitions: (table_name, index_name, columns)
INDEX_DEFINITIONS: list[tuple[str, str, str]] = [
    ("cn_ga_mainline_radar_daily", "idx_cn_ga_mainline_radar_daily_date", "(`trade_date`)"),
    ("cn_ga_mainline_radar_daily", "idx_cn_ga_mainline_radar_daily_mainline", "(`mainline_id`, `trade_date`)"),
    ("cn_ga_mainline_radar_daily", "idx_cn_ga_mainline_radar_daily_state", "(`mainline_state`, `trade_date`)"),
    ("cn_ga_market_pulse_daily", "idx_cn_ga_market_pulse_daily_date", "(`trade_date`)"),
    ("cn_ga_market_pulse_daily", "idx_cn_ga_market_pulse_daily_state", "(`market_state`, `trade_date`)"),
    ("cn_ga_stock_role_map_daily", "idx_cn_ga_stock_role_map_daily_date", "(`trade_date`)"),
    ("cn_ga_stock_role_map_daily", "idx_cn_ga_stock_role_map_daily_symbol", "(`symbol`, `trade_date`)"),
    ("cn_ga_stock_role_map_daily", "idx_cn_ga_stock_role_map_daily_role", "(`stock_role`, `trade_date`)"),
    ("cn_ga_stock_role_map_daily", "idx_cn_ga_stock_role_map_daily_mainline", "(`mainline_id`, `trade_date`)"),
    # P2 — cn_mainline_lifecycle_daily indexes
    ("cn_mainline_lifecycle_daily", "idx_mainline_lifecycle_daily_date", "(`trade_date`)"),
    ("cn_mainline_lifecycle_daily", "idx_mainline_lifecycle_daily_state", "(`lifecycle_state`, `trade_date`)"),
    ("cn_mainline_lifecycle_daily", "idx_mainline_lifecycle_daily_rank", "(`rotation_rank`, `trade_date`)"),
    ("cn_mainline_lifecycle_daily", "idx_mainline_lifecycle_daily_strength", "(`mainline_strength`, `trade_date`)"),
]

# ---------------------------------------------------------------------------
# Helpers
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


def table_exists(conn: Any, db_name: str, table_name: str) -> bool:
    sql = """
        SELECT COUNT(*) AS cnt
        FROM INFORMATION_SCHEMA.TABLES
        WHERE TABLE_SCHEMA = :schema AND TABLE_NAME = :table
    """
    return conn.execute(text(sql), {"schema": db_name, "table": table_name}).scalar() > 0


def column_exists(conn: Any, db_name: str, table_name: str, column_name: str) -> bool:
    sql = """
        SELECT COUNT(*) AS cnt
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = :schema
          AND TABLE_NAME = :table
          AND COLUMN_NAME = :column
    """
    return conn.execute(text(sql), {"schema": db_name, "table": table_name, "column": column_name}).scalar() > 0


def index_exists(conn: Any, db_name: str, table_name: str, index_name: str) -> bool:
    sql = """
        SELECT COUNT(*) AS cnt
        FROM INFORMATION_SCHEMA.STATISTICS
        WHERE TABLE_SCHEMA = :schema
          AND TABLE_NAME = :table
          AND INDEX_NAME = :index
    """
    return conn.execute(text(sql), {"schema": db_name, "table": table_name, "index": index_name}).scalar() > 0


def add_columns_if_missing(
    conn: Any,
    db_name: str,
    table_name: str,
    columns: list[tuple[str, str]],
) -> list[str]:
    """Add columns that don't exist yet. Returns list of added column names."""
    added: list[str] = []
    for col_name, col_type in columns:
        if not column_exists(conn, db_name, table_name, col_name):
            alter_sql = f"ALTER TABLE `{table_name}` ADD COLUMN `{col_name}` {col_type}"
            conn.execute(text(alter_sql))
            added.append(col_name)
            print(f"    + ADDED COLUMN `{col_name}` {col_type}")
    return added


def create_index_if_missing(
    conn: Any,
    db_name: str,
    table_name: str,
    index_name: str,
    columns_clause: str,
) -> bool:
    """Create index if it doesn't exist. Returns True if created."""
    if not index_exists(conn, db_name, table_name, index_name):
        sql = f"CREATE INDEX `{index_name}` ON `{table_name}` {columns_clause}"
        conn.execute(text(sql))
        print(f"    + CREATED INDEX `{index_name}` ON `{table_name}` {columns_clause}")
        return True
    return False


# ---------------------------------------------------------------------------
# Migration logic
# ---------------------------------------------------------------------------


def run_migration(conn: Any, db_name: str) -> int:
    """
    Execute all migration steps. Returns count of changes made.
    """
    changes = 0
    prog = ProgressTracker(total=6, prefix="Migration steps")

    # ── Step 1: cn_ga_mainline_radar_daily ──────────────────────────────────
    prog.update(0, "cn_ga_mainline_radar_daily")
    if table_exists(conn, db_name, "cn_ga_mainline_radar_daily"):
        added = add_columns_if_missing(conn, db_name, "cn_ga_mainline_radar_daily", MAINLINE_RADAR_COLUMNS)
        changes += len(added)
        suffix = f"cn_ga_mainline_radar_daily → {len(added)} col(s) added" if added else "cn_ga_mainline_radar_daily → OK"
        prog.update(1, suffix)
    else:
        prog.update(1, "cn_ga_mainline_radar_daily → TABLE NOT FOUND (skipped)")

    # ── Step 2: cn_ga_market_pulse_daily ────────────────────────────────────
    prog.update(0, "cn_ga_market_pulse_daily")
    if table_exists(conn, db_name, "cn_ga_market_pulse_daily"):
        added = add_columns_if_missing(conn, db_name, "cn_ga_market_pulse_daily", MARKET_PULSE_COLUMNS)
        changes += len(added)
        suffix = f"cn_ga_market_pulse_daily → {len(added)} col(s) added" if added else "cn_ga_market_pulse_daily → OK"
        prog.update(1, suffix)
    else:
        prog.update(1, "cn_ga_market_pulse_daily → TABLE NOT FOUND (skipped)")

    # ── Step 3: cn_ga_stock_role_map_daily ──────────────────────────────────
    prog.update(0, "cn_ga_stock_role_map_daily")
    if table_exists(conn, db_name, "cn_ga_stock_role_map_daily"):
        added = add_columns_if_missing(conn, db_name, "cn_ga_stock_role_map_daily", STOCK_ROLE_COLUMNS)
        changes += len(added)
        suffix = f"cn_ga_stock_role_map_daily → {len(added)} col(s) added" if added else "cn_ga_stock_role_map_daily → OK"
        prog.update(1, suffix)
    else:
        prog.update(1, "cn_ga_stock_role_map_daily → TABLE NOT FOUND (skipped)")

    # ── Step 4: cn_ga_market_context_daily ──────────────────────────────────
    prog.update(0, "cn_ga_market_context_daily")
    if table_exists(conn, db_name, "cn_ga_market_context_daily"):
        prog.update(1, "cn_ga_market_context_daily → already exists")
    else:
        # Execute CREATE TABLE from SQL file
        if SQL_FILE.exists():
            sql_text = SQL_FILE.read_text(encoding="utf-8")
            # Extract only the CREATE TABLE statement for cn_ga_market_context_daily
            stmts = [s.strip() for s in sql_text.split(";") if s.strip()]
            for stmt in stmts:
                if "CREATE TABLE" in stmt and "cn_ga_market_context_daily" in stmt:
                    conn.execute(text(stmt))
                    changes += 1
                    prog.update(1, "cn_ga_market_context_daily → CREATED")
                    break
            else:
                prog.update(1, "cn_ga_market_context_daily → WARNING: CREATE TABLE not found in SQL file")
        else:
            prog.update(1, f"cn_ga_market_context_daily → WARNING: SQL file not found")
    conn.commit()

    # ── Step 5: cn_mainline_lifecycle_daily (P2) ─────────────────────────
    prog.update(0, "cn_mainline_lifecycle_daily")
    if table_exists(conn, db_name, "cn_mainline_lifecycle_daily"):
        prog.update(1, "cn_mainline_lifecycle_daily → already exists")
    else:
        if P2_LIFECYCLE_SQL_FILE.exists():
            sql_text = P2_LIFECYCLE_SQL_FILE.read_text(encoding="utf-8")
            stmts = [s.strip() for s in sql_text.split(";") if s.strip()]
            for stmt in stmts:
                if "CREATE TABLE" in stmt and "cn_mainline_lifecycle_daily" in stmt:
                    conn.execute(text(stmt))
                    changes += 1
                    prog.update(1, "cn_mainline_lifecycle_daily → CREATED")
                    break
            else:
                prog.update(1, "cn_mainline_lifecycle_daily → WARNING: CREATE TABLE not found in SQL file")
        else:
            prog.update(1, "cn_mainline_lifecycle_daily → WARNING: SQL file not found")
    conn.commit()

    # ── Step 6: Indexes ──────────────────────────────────────────────────
    prog.update(0, "Indexes")
    idx_created = 0
    for table_name, index_name, columns_clause in INDEX_DEFINITIONS:
        if table_exists(conn, db_name, table_name):
            if create_index_if_missing(conn, db_name, table_name, index_name, columns_clause):
                idx_created += 1
    suffix = f"Indexes → {idx_created} created" if idx_created else "Indexes → all exist"
    prog.update(1, suffix)

    prog.finish(f"Changes: {changes}")
    return changes


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="GrowthAlpha V8 — P0 Mainline Context Schema Migration"
    )
    parser.add_argument("--db-host", default="127.0.0.1", help="MySQL host")
    parser.add_argument("--db-port", type=int, default=3306, help="MySQL port")
    parser.add_argument("--db-user", default="root", help="MySQL user")
    parser.add_argument("--db-password", required=True, help="MySQL password")
    parser.add_argument("--db-name", default=DEFAULT_DB_NAME, help="Database name (default: cn_market_red)")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    print(f"=== GrowthAlpha V8 — P0 Schema Migration ===")
    print(f"  Database: {args.db_name}@{args.db_host}:{args.db_port}")
    print()

    engine = build_engine(args.db_host, args.db_port, args.db_user, args.db_password, args.db_name)

    with engine.connect() as conn:
        changes = run_migration(conn, args.db_name)

    print(f"\n=== Migration Complete ===")
    print(f"  Changes made: {changes}")
    if changes == 0:
        print("  Schema is already up-to-date. No changes needed.")


if __name__ == "__main__":
    main()
