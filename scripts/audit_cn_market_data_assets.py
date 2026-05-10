"""
scripts/audit_cn_market_data_assets.py
========================================
P0 Data Asset Auditor for cn_market_red (GrowthAlpha V8).

Audits 20+ tables across 5 tiers (P0_CRITICAL, P1_IMPORTANT, P2_STRUCTURE,
P3_REPORTING, GA_LAYER) and produces:

  - CSV report   -> reports/data_audit/data_asset_audit_<timestamp>.csv
  - Markdown     -> reports/data_audit/data_asset_audit_<timestamp>.md
  - Latest link  -> reports/data_audit/data_asset_audit_latest.md

Optional --write-db writes results into cn_ga_data_readiness_daily.
Optional --fail-on-critical exits with code 1 if any P0 table is FAIL/STALE/EMPTY/MISSING_TABLE.

Supports --as-of-date for historical audit reference (e.g. 2026-03-31).
When --as-of-date is set, freshness checks compare against that date instead of today,
so missing data after the as-of date does not cause STALE failures.

Usage:
  python scripts/audit_cn_market_data_assets.py ^
      --db-host 127.0.0.1 --db-port 3306 --db-user root ^
      --db-password YOUR_PASSWORD --db-name cn_market_red ^
      --output-dir reports/data_audit --write-db --fail-on-critical

  # Historical audit (ignore data after 2026-03-31)
  python scripts/audit_cn_market_data_assets.py ^
      --db-host 127.0.0.1 --db-port 3306 --db-user root ^
      --db-password YOUR_PASSWORD --db-name cn_market_red ^
      --as-of-date 2026-03-31 --write-db
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass, asdict
from datetime import date, datetime, timedelta
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
DEFAULT_OUTPUT_DIR = "reports/data_audit"

# Priority-ordered date column candidates (lower index = higher priority)
DATE_COLUMN_CANDIDATES: list[str] = [
    "trade_date",
    "TRADE_DATE",
    "ann_date",
    "ANN_DATE",
    "end_date",
    "END_DATE",
    "date",
    "DATE",
    "created_at",
    "CREATED_AT",
]

# ---------------------------------------------------------------------------
# Table Registry
# ---------------------------------------------------------------------------
# Each entry: (table_name, tier, expected_columns, freshness_threshold_days, preferred_date_column)
# freshness_threshold_days = None means skip freshness check.

TABLE_REGISTRY: list[tuple[str, str, list[str], int | None, str | None]] = [
    # -- P0_CRITICAL ----------------------------------------------------------
    (
        "cn_stock_daily_price",
        "P0_CRITICAL",
        ["SYMBOL", "TRADE_DATE", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME", "AMOUNT", "CHG_PCT"],
        5,
        "TRADE_DATE",
    ),
    (
        "cn_index_daily_price",
        "P0_CRITICAL",
        ["INDEX_CODE", "TRADE_DATE", "OPEN", "CLOSE", "HIGH", "LOW", "VOLUME", "AMOUNT", "PRE_CLOSE", "CHG_PCT"],
        5,
        "TRADE_DATE",
    ),
    (
        "cn_sw_industry_daily",
        "P0_CRITICAL",
        ["ts_code", "name", "trade_date", "close", "amount"],
        5,
        "trade_date",
    ),
    (
        "cn_stock_monthly_basic",
        "P0_CRITICAL",
        ["symbol", "trade_date", "month_key", "total_mv", "circ_mv", "pe_ttm", "pb"],
        45,
        "trade_date",
    ),
    (
        "cn_stock_fina_indicator",
        "P0_CRITICAL",
        [
            "symbol",
            "end_date",
            "ann_date",
            "report_type",
            "netprofit_yoy",
            "q_profit_yoy",
            "or_yoy",
            "eps",
            "roe",
            "grossprofit_margin",
        ],
        90,
        "end_date",
    ),
    # -- P1_IMPORTANT --------------------------------------------------------
    (
        "cn_stock_daily_basic",
        "P1_IMPORTANT",
        ["symbol", "trade_date", "pe_ttm", "pb", "total_mv", "circ_mv", "volume_ratio"],
        5,
        "trade_date",
    ),
    (
        "cn_stock_leader_score_daily",
        "P1_IMPORTANT",
        ["trade_date", "symbol", "leader_score", "leader_bucket"],
        5,
        "trade_date",
    ),
    (
        "cn_stock_leader_sw_l1_latest_snap",
        "P1_IMPORTANT",
        ["trade_date", "symbol", "leader_score", "sw_l1_id"],
        5,
        "trade_date",
    ),
    (
        "cn_stock_universe_status_t",
        "P1_IMPORTANT",
        ["symbol", "is_active", "last_trade_date"],
        None,
        "last_trade_date",
    ),
    # -- P2_STRUCTURE --------------------------------------------------------
    (
        "cn_board_member_map_d",
        "P2_STRUCTURE",
        ["trade_date", "sector_type", "sector_id", "symbol"],
        7,
        "trade_date",
    ),
    (
        "cn_local_industry_map_hist",
        "P2_STRUCTURE",
        ["symbol", "industry_id", "in_date", "out_date", "is_current"],
        None,
        "in_date",
    ),
    (
        "cn_local_industry_proxy_daily",
        "P2_STRUCTURE",
        ["industry_id", "trade_date", "member_count", "ret_eqw", "amount_total"],
        5,
        "trade_date",
    ),
    # -- P3_REPORTING --------------------------------------------------------
    (
        "cn_stock_income",
        "P3_REPORTING",
        ["symbol", "end_date", "ann_date", "report_type", "total_revenue", "n_income_attr_p"],
        None,
        "end_date",
    ),
    (
        "cn_stock_balancesheet",
        "P3_REPORTING",
        ["symbol", "end_date", "ann_date", "report_type", "total_assets", "total_liab"],
        None,
        "end_date",
    ),
    (
        "cn_event_disclosure_date",
        "P3_REPORTING",
        ["symbol", "end_date", "pre_date", "actual_date"],
        None,
        "end_date",
    ),
    (
        "cn_event_earnings_forecast",
        "P3_REPORTING",
        ["symbol", "ann_date", "end_date", "forecast_type", "p_change_min", "p_change_max"],
        None,
        "ann_date",
    ),
    # -- GA_LAYER ------------------------------------------------------------
    (
        "cn_ga_mainline_radar_daily",
        "GA_LAYER",
        [
            "trade_date",
            "mainline_id",
            "mainline_name",
            "member_count",
            "leader_count",
            "mainline_score",
            "mainline_state",
            "rank_no",
            "reason",
        ],
        5,
        "trade_date",
    ),
    (
        "cn_ga_market_pulse_daily",
        "GA_LAYER",
        [
            "trade_date",
            "market_score",
            "market_state",
            "target_exposure",
            "breadth_up_ratio",
            "risk_flag",
            "reason",
        ],
        5,
        "trade_date",
    ),
    (
        "cn_ga_stock_role_map_daily",
        "GA_LAYER",
        [
            "trade_date",
            "symbol",
            "stock_name",
            "mainline_id",
            "mainline_name",
            "leader_score",
            "stock_role",
            "role_score",
            "role_reason",
        ],
        5,
        "trade_date",
    ),
    (
        "cn_ga_data_readiness_daily",
        "GA_LAYER",
        ["trade_date", "table_name", "status", "severity", "row_count", "max_trade_date", "null_rate_summary"],
        None,
        "trade_date",
    ),
]


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class AuditRow:
    table_name: str = ""
    tier: str = ""
    exists_flag: str = "N"
    object_type: str = ""
    row_count: int = 0
    min_trade_date: str = ""
    max_trade_date: str = ""
    distinct_trade_days: int = 0
    expected_key_columns: str = ""
    missing_key_columns: str = ""
    latest_lag_days: int = -1
    null_rate_summary: str = ""
    status: str = "MISSING_TABLE"
    severity: str = "CRITICAL"
    recommendation: str = ""


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


def resolve_as_of_date(as_of_text: str | None) -> date:
    if as_of_text:
        try:
            return datetime.strptime(as_of_text.strip(), "%Y-%m-%d").date()
        except ValueError:
            pass
        try:
            return datetime.strptime(as_of_text.strip(), "%Y%m%d").date()
        except ValueError:
            pass
    return date.today()


def detect_date_column(conn: Any, db_name: str, table_name: str, preferred: str | None) -> str | None:
    """Find the first existing date column from candidates, preferring the given one."""
    sql = """
        SELECT COLUMN_NAME, DATA_TYPE
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = :schema
          AND TABLE_NAME = :table
          AND DATA_TYPE IN ('date', 'datetime', 'timestamp')
        ORDER BY ORDINAL_POSITION
    """
    rows = conn.execute(text(sql), {"schema": db_name, "table": table_name}).mappings().all()
    existing_cols = {r["COLUMN_NAME"] for r in rows}

    if preferred and preferred in existing_cols:
        return preferred

    for candidate in DATE_COLUMN_CANDIDATES:
        if candidate in existing_cols:
            return candidate

    # fallback: first date-type column
    for r in rows:
        return r["COLUMN_NAME"]

    return None


def table_exists(conn: Any, db_name: str, table_name: str) -> bool:
    sql = """
        SELECT COUNT(*) AS cnt
        FROM INFORMATION_SCHEMA.TABLES
        WHERE TABLE_SCHEMA = :schema AND TABLE_NAME = :table
    """
    return conn.execute(text(sql), {"schema": db_name, "table": table_name}).scalar() > 0


def get_object_type(conn: Any, db_name: str, table_name: str) -> str:
    sql = """
        SELECT TABLE_TYPE
        FROM INFORMATION_SCHEMA.TABLES
        WHERE TABLE_SCHEMA = :schema AND TABLE_NAME = :table
    """
    result = conn.execute(text(sql), {"schema": db_name, "table": table_name}).scalar()
    return result or ""


def get_columns(conn: Any, db_name: str, table_name: str) -> set[str]:
    sql = """
        SELECT COLUMN_NAME
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = :schema AND TABLE_NAME = :table
    """
    return {r[0] for r in conn.execute(text(sql), {"schema": db_name, "table": table_name}).fetchall()}


def compute_null_rates(conn: Any, db_name: str, table_name: str, columns: list[str]) -> dict[str, float]:
    """Return {column: null_pct} for each column."""
    if not columns:
        return {}
    rates: dict[str, float] = {}
    for col in columns:
        safe_col = f"`{col}`"
        sql = f"""
            SELECT COUNT(*) AS total,
                   SUM(CASE WHEN {safe_col} IS NULL THEN 1 ELSE 0 END) AS null_cnt
            FROM `{table_name}`
        """
        row = conn.execute(text(sql)).mappings().first()
        total = row["total"] or 0
        null_cnt = row["null_cnt"] or 0
        rates[col] = round(null_cnt / total * 100, 2) if total > 0 else 0.0
    return rates


# ---------------------------------------------------------------------------
# Status determination
# ---------------------------------------------------------------------------


def determine_status(
    exists: bool,
    row_count: int,
    missing_cols: list[str],
    lag_days: int | None,
    tier: str,
    freshness_threshold: int | None,
    null_rates: dict[str, float],
    table_name: str,
) -> tuple[str, str, str]:
    """
    Returns (status, severity, recommendation).
    """
    if not exists:
        return (
            "MISSING_TABLE",
            _tier_severity(tier, "CRITICAL"),
            f"Table `{table_name}` does not exist in database.",
        )

    if row_count == 0:
        return (
            "EMPTY",
            _tier_severity(tier, "WARN"),
            "Table exists but contains zero rows.",
        )

    if missing_cols:
        cols_str = ", ".join(missing_cols)
        return (
            "MISSING_COLUMNS",
            _tier_severity(tier, "WARN"),
            f"Missing key columns: {cols_str}.",
        )

    if freshness_threshold is not None and lag_days is not None and lag_days > freshness_threshold:
        return (
            "STALE",
            _tier_severity(tier, "WARN"),
            f"Latest data is {lag_days} days behind as-of date (threshold: {freshness_threshold}d).",
        )

    # Check null rates -- warn if any key column > 20% null
    high_null_cols = [col for col, rate in null_rates.items() if rate > 20.0]
    if high_null_cols:
        cols_str = ", ".join(f"{c}={null_rates[c]:.1f}%" for c in high_null_cols)
        return (
            "WARN",
            _tier_severity(tier, "WARN"),
            f"High null rates on key columns: {cols_str}.",
        )

    return ("OK", _tier_severity(tier, "INFO"), "Table is healthy and up-to-date.")


def _tier_severity(tier: str, fallback: str) -> str:
    mapping = {
        "P0_CRITICAL": "CRITICAL",
        "P1_IMPORTANT": "WARN",
        "P2_STRUCTURE": "WARN",
        "P3_REPORTING": "INFO",
        "GA_LAYER": "INFO",
    }
    return mapping.get(tier, fallback)


# ---------------------------------------------------------------------------
# Core audit logic
# ---------------------------------------------------------------------------


def audit_table(
    conn: Any,
    db_name: str,
    table_name: str,
    tier: str,
    expected_columns: list[str],
    freshness_threshold: int | None,
    preferred_date_col: str | None,
    as_of_date: date,
) -> AuditRow:
    row = AuditRow(table_name=table_name, tier=tier)
    row.expected_key_columns = ", ".join(expected_columns)

    # 1. Existence
    exists = table_exists(conn, db_name, table_name)
    if not exists:
        row.exists_flag = "N"
        row.status = "MISSING_TABLE"
        row.severity = _tier_severity(tier, "CRITICAL")
        row.recommendation = f"Table `{table_name}` does not exist. Must be created before engine can use it."
        return row

    row.exists_flag = "Y"
    row.object_type = get_object_type(conn, db_name, table_name)

    # 2. Row count
    row.row_count = conn.execute(text(f"SELECT COUNT(*) FROM `{table_name}`")).scalar() or 0

    if row.row_count == 0:
        row.status = "EMPTY"
        row.severity = _tier_severity(tier, "WARN")
        row.recommendation = f"Table `{table_name}` exists but is empty."
        return row

    # 3. Detect date column
    date_col = detect_date_column(conn, db_name, table_name, preferred_date_col)

    # 4. Date range
    if date_col:
        safe_date_col = f"`{date_col}`"
        date_stats = (
            conn.execute(
                text(
                    f"SELECT MIN({safe_date_col}) AS min_d, MAX({safe_date_col}) AS max_d, "
                    f"COUNT(DISTINCT {safe_date_col}) AS distinct_days FROM `{table_name}`"
                )
            )
            .mappings()
            .first()
        )
        if date_stats:
            row.min_trade_date = str(date_stats["min_d"] or "")
            row.max_trade_date = str(date_stats["max_d"] or "")
            row.distinct_trade_days = date_stats["distinct_days"] or 0

            if date_stats["max_d"]:
                max_dt = date_stats["max_d"]
                if isinstance(max_dt, datetime):
                    max_dt = max_dt.date()
                row.latest_lag_days = (as_of_date - max_dt).days

    # 5. Column check
    existing_cols = get_columns(conn, db_name, table_name)
    missing_cols = [c for c in expected_columns if c not in existing_cols]
    row.missing_key_columns = ", ".join(missing_cols) if missing_cols else ""

    # 6. Null rates on expected columns that exist
    checkable = [c for c in expected_columns if c in existing_cols]
    null_rates = compute_null_rates(conn, db_name, table_name, checkable)
    row.null_rate_summary = "; ".join(f"{c}={v}%" for c, v in null_rates.items())

    # 7. Status
    status, severity, recommendation = determine_status(
        exists=True,
        row_count=row.row_count,
        missing_cols=missing_cols,
        lag_days=row.latest_lag_days,
        tier=tier,
        freshness_threshold=freshness_threshold,
        null_rates=null_rates,
        table_name=table_name,
    )
    row.status = status
    row.severity = severity
    row.recommendation = recommendation

    return row


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def write_csv(results: list[AuditRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    field_names = [
        "table_name",
        "tier",
        "exists_flag",
        "object_type",
        "row_count",
        "min_trade_date",
        "max_trade_date",
        "distinct_trade_days",
        "expected_key_columns",
        "missing_key_columns",
        "latest_lag_days",
        "null_rate_summary",
        "status",
        "severity",
        "recommendation",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=field_names)
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))
    print(f"  [CSV]  {path}")


def write_markdown(results: list[AuditRow], path: Path, as_of_date: date, db_name: str) -> None:
    """Write a comprehensive markdown audit report."""
    path.parent.mkdir(parents=True, exist_ok=True)

    tiers_in_order = ["P0_CRITICAL", "P1_IMPORTANT", "P2_STRUCTURE", "P3_REPORTING", "GA_LAYER"]
    tier_labels = {
        "P0_CRITICAL": "P0 -- Critical (Core Market Data)",
        "P1_IMPORTANT": "P1 -- Important (Derived / Quality)",
        "P2_STRUCTURE": "P2 -- Structure (Mapping / Proxy)",
        "P3_REPORTING": "P3 -- Reporting (Financial Events)",
        "GA_LAYER": "GA Layer (GrowthAlpha Engine)",
    }

    # Group by tier
    by_tier: dict[str, list[AuditRow]] = {t: [] for t in tiers_in_order}
    for r in results:
        by_tier.setdefault(r.tier, []).append(r)

    # Count statuses
    total = len(results)
    ok_count = sum(1 for r in results if r.status == "OK")
    warn_count = sum(1 for r in results if r.status in ("WARN", "STALE", "MISSING_COLUMNS"))
    fail_count = sum(1 for r in results if r.status in ("FAIL", "MISSING_TABLE", "EMPTY"))
    p0_fail = sum(
        1
        for r in results
        if r.tier == "P0_CRITICAL" and r.status in ("FAIL", "MISSING_TABLE", "MISSING_COLUMNS", "EMPTY", "STALE")
    )

    lines: list[str] = []
    _w = lines.append

    _w(f"# Data Asset Audit Report -- `{db_name}`")
    _w(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    _w(f"**As-Of Date**: {as_of_date}")
    _w(f"**Tables Audited**: {total}")
    _w("")
    _w("---")
    _w("")

    # -- 1. Executive Summary ------------------------------------------------
    _w("## 1. Executive Summary")
    _w("")
    _w("| Metric | Value |")
    _w("|--------|-------|")
    _w(f"| **Total Tables** | {total} |")
    _w(f"| **Healthy (OK)** | {ok_count} |")
    _w(f"| **Warning (WARN/STALE/MISSING_COLUMNS)** | {warn_count} |")
    _w(f"| **Failed (FAIL/MISSING_TABLE/EMPTY)** | {fail_count} |")
    _w(f"| **P0 Critical Failures** | {p0_fail} |")
    _w("")
    if p0_fail > 0:
        _w("> :warning: **P0 CRITICAL tables have failures. Immediate action required.**")
    elif warn_count > 0:
        _w("> :warning: **Some tables have warnings. Review recommended.**")
    else:
        _w("> :white_check_mark: **All tables healthy.**")
    _w("")

    # -- 2. Critical Table Status --------------------------------------------
    _w("## 2. Critical Table Status")
    _w("")
    _w("| Table | Tier | Status | Severity | Rows | Max Date | Lag (d) |")
    _w("|-------|------|--------|----------|------|----------|---------|")
    for r in results:
        if r.tier in ("P0_CRITICAL", "P1_IMPORTANT"):
            lag_str = str(r.latest_lag_days) if r.latest_lag_days >= 0 else "N/A"
            _w(
                f"| `{r.table_name}` | {r.tier} | {r.status} | {r.severity} "
                f"| {r.row_count:,} | {r.max_trade_date} | {lag_str} |"
            )
    _w("")

    # -- 3. Per-Tier Detail --------------------------------------------------
    _w("## 3. Per-Tier Detail")
    _w("")
    for tier in tiers_in_order:
        tier_rows = by_tier.get(tier, [])
        if not tier_rows:
            continue
        _w(f"### {tier_labels.get(tier, tier)}")
        _w("")
        _w("| Table | Status | Rows | Min Date | Max Date | Lag (d) | Missing Columns | Null Rates |")
        _w("|-------|--------|------|----------|----------|---------|-----------------|------------|")
        for r in tier_rows:
            lag_str = str(r.latest_lag_days) if r.latest_lag_days >= 0 else "N/A"
            missing = r.missing_key_columns if r.missing_key_columns else "--"
            nulls = r.null_rate_summary if r.null_rate_summary else "--"
            _w(
                f"| `{r.table_name}` | {r.status} | {r.row_count:,} "
                f"| {r.min_trade_date} | {r.max_trade_date} | {lag_str} "
                f"| {missing} | {nulls} |"
            )
        _w("")

    # -- 4. Date Coverage Summary --------------------------------------------
    _w("## 4. Date Coverage Summary")
    _w("")
    _w("| Table | Tier | Min Date | Max Date | Distinct Days | Lag (d) | Threshold (d) |")
    _w("|-------|------|----------|----------|---------------|---------|---------------|")
    for r in results:
        threshold = ""
        for tname, _tier, _cols, thr, _pref in TABLE_REGISTRY:
            if tname == r.table_name:
                threshold = str(thr) if thr else "N/A"
                break
        lag_str = str(r.latest_lag_days) if r.latest_lag_days >= 0 else "N/A"
        _w(
            f"| `{r.table_name}` | {r.tier} | {r.min_trade_date} | {r.max_trade_date} "
            f"| {r.distinct_trade_days:,} | {lag_str} | {threshold} |"
        )
    _w("")

    # -- 5. Missing Columns --------------------------------------------------
    _w("## 5. Missing Columns")
    _w("")
    missing_cols_tables = [r for r in results if r.missing_key_columns]
    if missing_cols_tables:
        _w("| Table | Tier | Missing Columns |")
        _w("|-------|------|-----------------|")
        for r in missing_cols_tables:
            _w(f"| `{r.table_name}` | {r.tier} | `{r.missing_key_columns}` |")
    else:
        _w("No missing key columns detected.")
    _w("")

    # -- 6. Stale Tables -----------------------------------------------------
    _w("## 6. Stale Tables")
    _w("")
    stale_tables = [r for r in results if r.status == "STALE"]
    if stale_tables:
        _w("| Table | Tier | Max Date | Lag (d) | Threshold (d) |")
        _w("|-------|------|----------|---------|---------------|")
        for r in stale_tables:
            threshold = ""
            for tname, _tier, _cols, thr, _pref in TABLE_REGISTRY:
                if tname == r.table_name:
                    threshold = str(thr) if thr else "N/A"
                    break
            _w(f"| `{r.table_name}` | {r.tier} | {r.max_trade_date} | {r.latest_lag_days} | {threshold} |")
    else:
        _w("No stale tables detected.")
    _w("")

    # -- 7. Null Rate Analysis -----------------------------------------------
    _w("## 7. Null Rate Analysis")
    _w("")
    high_null_tables = [r for r in results if r.status == "WARN" and r.null_rate_summary]
    if high_null_tables:
        _w("Tables with high null rates on key columns (>20%):")
        _w("")
        for r in high_null_tables:
            _w(f"- `{r.table_name}` ({r.tier}): {r.null_rate_summary}")
    else:
        _w("No significant null rate issues detected on key columns.")
    _w("")

    # -- 8. Recommended Next Actions -----------------------------------------
    _w("## 8. Recommended Next Actions")
    _w("")
    non_ok = [r for r in results if r.status != "OK"]
    if non_ok:
        _w("| Priority | Table | Issue | Recommendation |")
        _w("|----------|-------|-------|----------------|")
        for i, r in enumerate(non_ok, 1):
            _w(f"| {i}. | `{r.table_name}` | {r.status} | {r.recommendation} |")
    else:
        _w("No action required. All tables are healthy.")
    _w("")

    # -- 9. Readiness Assessment ---------------------------------------------
    _w("## 9. Readiness Assessment")
    _w("")

    # Mainline Strength Engine readiness
    mainline_tables = [
        "cn_stock_daily_price",
        "cn_board_member_map_d",
        "cn_local_industry_map_hist",
        "cn_local_industry_proxy_daily",
        "cn_ga_mainline_radar_daily",
    ]
    mainline_ready = _check_readiness(results, mainline_tables)
    _w("### Mainline Strength Engine")
    _w(f"**Status**: {'READY' if mainline_ready else 'NOT READY'}")
    if not mainline_ready:
        _w(f"**Blockers**: {_list_blockers(results, mainline_tables)}")
    _w("")

    # Market Breadth Engine readiness
    breadth_tables = [
        "cn_stock_daily_price",
        "cn_index_daily_price",
        "cn_sw_industry_daily",
        "cn_ga_market_pulse_daily",
    ]
    breadth_ready = _check_readiness(results, breadth_tables)
    _w("### Market Breadth Engine")
    _w(f"**Status**: {'READY' if breadth_ready else 'NOT READY'}")
    if not breadth_ready:
        _w(f"**Blockers**: {_list_blockers(results, breadth_tables)}")
    _w("")

    # Narrative / Context Layer readiness
    narrative_tables = [
        "cn_ga_mainline_radar_daily",
        "cn_ga_market_pulse_daily",
        "cn_ga_stock_role_map_daily",
        "cn_stock_fina_indicator",
    ]
    narrative_ready = _check_readiness(results, narrative_tables)
    _w("### Narrative / Context Layer")
    _w(f"**Status**: {'READY' if narrative_ready else 'NOT READY'}")
    if not narrative_ready:
        _w(f"**Blockers**: {_list_blockers(results, narrative_tables)}")
    _w("")

    # -- Footer --------------------------------------------------------------
    _w("---")
    _w(f"*Report generated by `audit_cn_market_data_assets.py` at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  [MD]   {path}")


def _check_readiness(results: list[AuditRow], required_tables: list[str]) -> bool:
    """Check if all required tables are in OK status."""
    result_map = {r.table_name: r for r in results}
    for tname in required_tables:
        r = result_map.get(tname)
        if not r or r.status != "OK":
            return False
    return True


def _list_blockers(results: list[AuditRow], required_tables: list[str]) -> str:
    """List tables that are blocking readiness."""
    result_map = {r.table_name: r for r in results}
    blockers: list[str] = []
    for tname in required_tables:
        r = result_map.get(tname)
        if not r:
            blockers.append(f"`{tname}` (missing from audit)")
        elif r.status != "OK":
            blockers.append(f"`{tname}` ({r.status})")
    return "; ".join(blockers) if blockers else "None"


# ---------------------------------------------------------------------------
# DB writer
# ---------------------------------------------------------------------------


def write_to_readiness_table(conn: Any, results: list[AuditRow], trade_date: date) -> None:
    """Upsert results into cn_ga_data_readiness_daily (existing schema).

    Matches the existing table schema:
      check_date, table_name, status(ENUM), row_count, max_trade_date,
      null_rate_json(JSON), issue_json(JSON), remark
    """
    insert_sql = text(
        """
        INSERT INTO `cn_ga_data_readiness_daily`
            (`check_date`, `table_name`, `status`, `row_count`,
             `max_trade_date`, `null_rate_json`, `issue_json`, `remark`)
        VALUES
            (:check_date, :table_name, :status, :row_count,
             :max_trade_date, :null_rate_json, :issue_json, :remark)
        ON DUPLICATE KEY UPDATE
            `status` = VALUES(`status`),
            `row_count` = VALUES(`row_count`),
            `max_trade_date` = VALUES(`max_trade_date`),
            `null_rate_json` = VALUES(`null_rate_json`),
            `issue_json` = VALUES(`issue_json`),
            `remark` = VALUES(`remark`),
            `updated_at` = CURRENT_TIMESTAMP
        """
    )

    for r in results:
        max_date = None
        if r.max_trade_date:
            try:
                max_date = datetime.strptime(r.max_trade_date[:10], "%Y-%m-%d").date()
            except ValueError:
                pass

        # Map status to existing ENUM: 'PASS', 'PARTIAL', 'FAIL'
        if r.status == "OK":
            mapped_status = "PASS"
        elif r.status in ("WARN", "STALE", "MISSING_COLUMNS"):
            mapped_status = "PARTIAL"
        else:
            mapped_status = "FAIL"

        # Build null_rate_json from summary string
        null_rate_json = None
        if r.null_rate_summary:
            pairs = [p.strip() for p in r.null_rate_summary.split(";") if p.strip()]
            null_dict = {}
            for pair in pairs:
                if "=" in pair:
                    k, v = pair.split("=", 1)
                    null_dict[k.strip()] = v.strip()
            if null_dict:
                import json as _json
                null_rate_json = _json.dumps(null_dict)

        # Build issue_json
        issues = []
        if r.status not in ("OK",):
            issues.append(f"status={r.status}")
        if r.recommendation:
            issues.append(r.recommendation)
        issue_json = None
        if issues:
            import json as _json
            issue_json = _json.dumps(issues)

        conn.execute(
            insert_sql,
            {
                "check_date": trade_date,
                "table_name": r.table_name,
                "status": mapped_status,
                "row_count": r.row_count,
                "max_trade_date": max_date,
                "null_rate_json": null_rate_json,
                "issue_json": issue_json,
                "remark": r.recommendation[:500] if r.recommendation else None,
            },
        )
    conn.commit()
    print(f"  [DB]   Upserted {len(results)} rows into cn_ga_data_readiness_daily")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="P0 Data Asset Auditor for cn_market_red (GrowthAlpha V8)"
    )
    parser.add_argument("--db-host", default="127.0.0.1", help="MySQL host")
    parser.add_argument("--db-port", type=int, default=3306, help="MySQL port")
    parser.add_argument("--db-user", default="root", help="MySQL user")
    parser.add_argument("--db-password", required=True, help="MySQL password")
    parser.add_argument("--db-name", default=DEFAULT_DB_NAME, help="Database name (default: cn_market_red)")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output directory for reports")
    parser.add_argument("--as-of-date", default="", help="Reference date YYYY-MM-DD (default: today)")
    parser.add_argument("--write-db", action="store_true", help="Write results to cn_ga_data_readiness_daily")
    parser.add_argument("--fail-on-critical", action="store_true", help="Exit code 1 if P0 tables fail")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    as_of_date = resolve_as_of_date(args.as_of_date)
    output_dir = Path(args.output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"=== P0 Data Asset Audit ===")
    print(f"  Database:   {args.db_name}@{args.db_host}:{args.db_port}")
    print(f"  As-Of Date: {as_of_date}")
    print(f"  Output Dir: {output_dir}")
    print(f"  Write DB:   {args.write_db}")
    print(f"  Fail On Critical: {args.fail_on_critical}")
    print()

    engine = build_engine(args.db_host, args.db_port, args.db_user, args.db_password, args.db_name)

    results: list[AuditRow] = []
    total_tables = len(TABLE_REGISTRY)
    prog = ProgressTracker(total=total_tables, prefix="Auditing tables")

    with engine.connect() as conn:
        for idx, (table_name, tier, expected_cols, freshness_threshold, preferred_date_col) in enumerate(TABLE_REGISTRY):
            prog.update(0, f"{table_name} ({tier})")
            row = audit_table(
                conn=conn,
                db_name=args.db_name,
                table_name=table_name,
                tier=tier,
                expected_columns=expected_cols,
                freshness_threshold=freshness_threshold,
                preferred_date_col=preferred_date_col,
                as_of_date=as_of_date,
            )
            results.append(row)
            prog.update(1, f"{table_name} → {row.status}")
    prog.finish(f"{total_tables} tables audited")

    # Write CSV
    csv_path = output_dir / f"data_asset_audit_{timestamp}.csv"
    write_csv(results, csv_path)

    # Write Markdown
    md_path = output_dir / f"data_asset_audit_{timestamp}.md"
    write_markdown(results, md_path, as_of_date, args.db_name)

    # Write Latest
    latest_path = output_dir / "data_asset_audit_latest.md"
    write_markdown(results, latest_path, as_of_date, args.db_name)

    # Write to DB
    if args.write_db:
        with engine.connect() as conn:
            write_to_readiness_table(conn, results, as_of_date)

    # Summary
    print()
    print("=== Audit Complete ===")
    ok_count = sum(1 for r in results if r.status == "OK")
    warn_count = sum(1 for r in results if r.status in ("WARN", "STALE", "MISSING_COLUMNS"))
    fail_count = sum(1 for r in results if r.status in ("FAIL", "MISSING_TABLE", "EMPTY"))
    print(f"  OK:   {ok_count}")
    print(f"  WARN: {warn_count}")
    print(f"  FAIL: {fail_count}")

    # Fail-on-critical check
    if args.fail_on_critical:
        p0_fail = sum(
            1
            for r in results
            if r.tier == "P0_CRITICAL" and r.status in ("FAIL", "MISSING_TABLE", "MISSING_COLUMNS", "EMPTY", "STALE")
        )
        if p0_fail > 0:
            print(f"\nERROR: {p0_fail} P0_CRITICAL table(s) failed. Exiting with code 1.")
            sys.exit(1)


if __name__ == "__main__":
    main()