"""
scripts/build_stock_quality_score_daily.py
============================================
GrowthAlpha V8 — P3 Unified Alpha Engine — Part A.

Computes stock-level fundamental quality scores daily.

Input sources (ALL 4 MUST be filtered by ann_date <= trade_date):
  1. cn_stock_fundamental_daily  — each row has its own ann_date; rows with ann_date > trade_date are SKIPPED
  2. cn_stock_fina_indicator     — records sorted by ann_date; binary search picks the latest with ann_date <= trade_date
  3. cn_stock_income             — same binary-search approach as fina_indicator
  4. cn_stock_balancesheet       — same binary-search approach as fina_indicator

  ⚠ CRITICAL: Every financial data field used in score computation MUST pass through
    an ann_date <= trade_date filter. If ann_date is NULL for a record, that record
    is treated as unavailable (returns None) — NO fallback to end_date is allowed,
    because end_date is the fiscal period end, not the disclosure date, and using it
    would leak future financial data.

Output:
  - cn_stock_quality_score_daily

Sub-scores (all in [0,1]):
  - growth_acceleration_score
  - cashflow_score
  - debt_control_score
  - margin_stability_score
  - profitability_score
  - quality_score (composite)

Usage:
  python scripts/build_stock_quality_score_daily.py --start 2026-03-30 --end 2026-03-30 --db-name cn_market_red --replace --verbose
  python scripts/build_stock_quality_score_daily.py --start 2026-01-01 --end 2026-03-30 --db-name cn_market_red --dry-run
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, URL

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPORT_DIR = Path("reports") / "stock_quality"


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _fmt_seconds(seconds: float) -> str:
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds}s"
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {seconds}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m {seconds}s"


def _progress_line(label: str, current: int, total: int, started_at: float, extra: str = "") -> None:
    if total <= 0:
        return
    elapsed = time.time() - started_at
    pct = current * 100 // total
    eta = (elapsed / current * (total - current)) if current > 0 else 0.0
    suffix = f" | {extra}" if extra else ""
    print(
        f"[{_ts()}]   {label}: {current:,}/{total:,} ({pct}%) "
        f"elapsed={_fmt_seconds(elapsed)} eta={_fmt_seconds(eta)}{suffix}",
        flush=True,
    )


def _timed_load(label: str, loader, *args, **kwargs) -> pd.DataFrame:
    t0 = time.time()
    print(f"[{_ts()}]   Loading {label} ...", flush=True)
    df = loader(*args, **kwargs)
    print(f"[{_ts()}]   {label:<30} {len(df):>12,} rows ({_fmt_seconds(time.time() - t0)})", flush=True)
    return df

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="GrowthAlpha V8 — P3 Build cn_stock_quality_score_daily"
    )
    parser.add_argument("--start", default="2010-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default="", help="End date YYYY-MM-DD (default=today)")
    parser.add_argument("--db-name", default="cn_market_red", help="Database name")
    parser.add_argument("--db-host", default="127.0.0.1", help="MySQL host")
    parser.add_argument("--db-port", type=int, default=3306, help="MySQL port")
    parser.add_argument("--db-user", default="root", help="MySQL user")
    parser.add_argument("--db-password", default=None, help="MySQL password (default from env ASHARE_MYSQL_PASSWORD)")
    parser.add_argument("--dry-run", action="store_true", help="Compute but do not write to DB")
    parser.add_argument("--replace", action="store_true", help="Replace existing rows in date range")
    parser.add_argument("--output-dir", default=None, help="Override report output directory")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    parser.add_argument(
        "--chunk-months",
        type=int,
        default=3,
        help="Process output date range in chunks to avoid loading full cn_stock_fundamental_daily history at once",
    )
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


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
# ═══════════════════════════════════════════════════════════════════════════
# FUTURE-FUNCTION SAFETY — ann_date <= trade_date CONSTRAINT
# ═══════════════════════════════════════════════════════════════════════════
# ALL 4 financial source tables MUST be filtered by ann_date <= trade_date
# to prevent look-ahead bias (using financial data not yet disclosed):
#
# Source 1: cn_stock_fundamental_daily
#   - Each row has its own ann_date.
#   - In run_build(), rows where ann_date > trade_date are SKIPPED (line ~790).
#   - If ann_date is NULL, the row is treated as unavailable (skipped).
#
# Source 2: cn_stock_fina_indicator
#   - Records sorted by ann_date ascending.
#   - _find_latest_before() binary search picks the latest record with
#     ann_date <= trade_date. If ann_date is NULL, returns None.
#
# Source 3: cn_stock_income
#   - Same binary-search approach as fina_indicator.
#
# Source 4: cn_stock_balancesheet
#   - Same binary-search approach as fina_indicator.
#
# ⚠ NO fallback to end_date is allowed anywhere — end_date is the fiscal
#   period end, not the disclosure date. Using it would leak future data.
# ═══════════════════════════════════════════════════════════════════════════


def load_stock_fundamental_daily(
    engine: Engine, start: date, end: date
) -> pd.DataFrame:
    """Load cn_stock_fundamental_daily for the date range (with ann_date)."""
    sql = """
    SELECT
        symbol,
        trade_date,
        report_end_date,
        ann_date,
        revenue_yoy,
        profit_yoy,
        roe,
        gross_margin,
        debt_to_assets,
        ocfps,
        inventory,
        contract_liability,
        fixed_assets
    FROM cn_stock_fundamental_daily
    WHERE trade_date BETWEEN :start AND :end
    ORDER BY symbol, trade_date
    """
    return fetch_df(engine, sql, {"start": start, "end": end})


def load_fina_indicator(
    engine: Engine, lookback_start: date, end: date
) -> pd.DataFrame:
    """Load cn_stock_fina_indicator for lookback period (wide range for ann_date filtering)."""
    sql = """
    SELECT
        symbol,
        end_date,
        ann_date,
        report_type,
        or_yoy,
        tr_yoy,
        netprofit_yoy,
        roe,
        grossprofit_margin,
        debt_to_assets,
        ocfps,
        eps,
        bps,
        dt_eps,
        q_profit_yoy,
        q_sales_yoy,
        q_ocf_yoy
    FROM cn_stock_fina_indicator
    WHERE COALESCE(ann_date, end_date) BETWEEN :start AND :end
    ORDER BY symbol, end_date
    """
    return fetch_df(engine, sql, {"start": lookback_start, "end": end})


def load_income(
    engine: Engine, lookback_start: date, end: date
) -> pd.DataFrame:
    """Load cn_stock_income for lookback period."""
    sql = """
    SELECT
        symbol,
        end_date,
        ann_date,
        f_ann_date,
        report_type,
        total_revenue,
        revenue,
        n_income_attr_p,
        operate_profit,
        total_profit,
        n_income
    FROM cn_stock_income
    WHERE COALESCE(ann_date, end_date) BETWEEN :start AND :end
    ORDER BY symbol, end_date
    """
    return fetch_df(engine, sql, {"start": lookback_start, "end": end})


def load_balancesheet(
    engine: Engine, lookback_start: date, end: date
) -> pd.DataFrame:
    """Load cn_stock_balancesheet for lookback period."""
    sql = """
    SELECT
        symbol,
        end_date,
        ann_date,
        f_ann_date,
        report_type,
        total_assets,
        total_liab,
        total_hldr_eqy_exc_min_int AS total_equity,
        inventories AS inventory,
        accounts_receiv AS account_receivable,
        fix_assets AS fixed_assets,
        intan_assets AS intangible_assets,
        goodwill,
        notes_payable,
        acct_payable AS accounts_payable,
        lt_borr AS longterm_loans,
        st_borr AS shortterm_loans
    FROM cn_stock_balancesheet
    WHERE COALESCE(ann_date, end_date) BETWEEN :start AND :end
    ORDER BY symbol, end_date
    """
    return fetch_df(engine, sql, {"start": lookback_start, "end": end})


# ---------------------------------------------------------------------------
# Score computation
# ---------------------------------------------------------------------------


def _safe_float(val: Any, default: float = 0.0) -> float:
    if val is None:
        return default
    try:
        v = float(val)
        if np.isnan(v) or np.isinf(v):
            return default
        return v
    except (ValueError, TypeError):
        return default


def _clip01(val: float) -> float:
    return max(0.0, min(1.0, val))


def _compute_growth_acceleration(
    row: dict[str, Any],
    prev_row: dict[str, Any] | None,
) -> tuple[float, str]:
    """
    Growth Acceleration Score [0,1].

    Evaluates:
      - Revenue YoY growth level and acceleration
      - Profit YoY growth level and acceleration
      - Consistency (both revenue and profit positive)
    """
    rev_yoy = _safe_float(row.get("revenue_yoy"))
    profit_yoy = _safe_float(row.get("profit_yoy"))

    prev_rev = _safe_float(prev_row.get("revenue_yoy")) if prev_row else 0.0
    prev_profit = _safe_float(prev_row.get("profit_yoy")) if prev_row else 0.0

    reasons: list[str] = []

    # Revenue growth component (0-0.5)
    rev_score = 0.0
    if rev_yoy > 30:
        rev_score = 0.5
        reasons.append("rev_growth_strong")
    elif rev_yoy > 15:
        rev_score = 0.4
        reasons.append("rev_growth_moderate")
    elif rev_yoy > 5:
        rev_score = 0.25
        reasons.append("rev_growth_mild")
    elif rev_yoy > 0:
        rev_score = 0.15
        reasons.append("rev_growth_positive")
    else:
        reasons.append("rev_growth_negative_or_zero")

    # Profit growth component (0-0.35)
    profit_score = 0.0
    if profit_yoy > 30:
        profit_score = 0.35
        reasons.append("profit_growth_strong")
    elif profit_yoy > 15:
        profit_score = 0.28
        reasons.append("profit_growth_moderate")
    elif profit_yoy > 5:
        profit_score = 0.18
        reasons.append("profit_growth_mild")
    elif profit_yoy > 0:
        profit_score = 0.10
        reasons.append("profit_growth_positive")
    else:
        reasons.append("profit_growth_negative_or_zero")

    # Acceleration bonus (0-0.15)
    accel_bonus = 0.0
    if prev_row is not None:
        rev_accel = rev_yoy - prev_rev
        profit_accel = profit_yoy - prev_profit
        if rev_accel > 5 and profit_accel > 5:
            accel_bonus = 0.15
            reasons.append("dual_acceleration")
        elif rev_accel > 5 or profit_accel > 5:
            accel_bonus = 0.08
            reasons.append("single_acceleration")

    score = _clip01(rev_score + profit_score + accel_bonus)
    reason_str = ";".join(reasons) if reasons else "data_insufficient"
    return score, reason_str


def _compute_cashflow_score(
    row: dict[str, Any],
) -> tuple[float, str]:
    """
    Cashflow Score [0,1].

    Evaluates:
      - OCFPS (operating cash flow per share) level
      - OCF to net profit ratio (quality of earnings)
    """
    ocfps = _safe_float(row.get("ocfps"))
    n_income = _safe_float(row.get("n_income_attr_p"))
    ocf_net = _safe_float(row.get("net_cashflow_act"))

    reasons: list[str] = []

    # OCFPS component (0-0.6)
    ocfps_score = 0.0
    if ocfps > 1.0:
        ocfps_score = 0.6
        reasons.append("ocfps_high")
    elif ocfps > 0.5:
        ocfps_score = 0.45
        reasons.append("ocfps_moderate")
    elif ocfps > 0.1:
        ocfps_score = 0.25
        reasons.append("ocfps_low")
    elif ocfps > 0:
        ocfps_score = 0.10
        reasons.append("ocfps_positive")
    else:
        reasons.append("ocfps_negative_or_zero")

    # OCF / Net Profit ratio (0-0.4)
    quality_score = 0.0
    if n_income != 0 and ocf_net != 0:
        ratio = ocf_net / abs(n_income)
        if ratio > 1.0:
            quality_score = 0.4
            reasons.append("ocf_quality_strong")
        elif ratio > 0.5:
            quality_score = 0.25
            reasons.append("ocf_quality_moderate")
        elif ratio > 0:
            quality_score = 0.10
            reasons.append("ocf_quality_weak")
        else:
            reasons.append("ocf_quality_negative")
    else:
        reasons.append("ocf_quality_unknown")

    score = _clip01(ocfps_score + quality_score)
    reason_str = ";".join(reasons) if reasons else "data_insufficient"
    return score, reason_str


def _compute_debt_control_score(
    row: dict[str, Any],
) -> tuple[float, str]:
    """
    Debt Control Score [0,1].

    Evaluates:
      - Debt-to-assets ratio (lower is better)
      - Interest debt level
      - Short-term vs long-term debt structure
    """
    debt_to_assets = _safe_float(row.get("debt_to_assets"))
    total_assets = _safe_float(row.get("total_assets"))
    total_liab = _safe_float(row.get("total_liab"))
    interest_debt = _safe_float(row.get("interest_debt"))
    shortterm_loans = _safe_float(row.get("shortterm_loans"))
    longterm_loans = _safe_float(row.get("longterm_loans"))

    reasons: list[str] = []

    # Debt-to-assets component (0-0.5)
    dta_score = 0.0
    if debt_to_assets > 0:
        if debt_to_assets < 30:
            dta_score = 0.5
            reasons.append("debt_low")
        elif debt_to_assets < 50:
            dta_score = 0.35
            reasons.append("debt_moderate")
        elif debt_to_assets < 70:
            dta_score = 0.15
            reasons.append("debt_elevated")
        else:
            dta_score = 0.05
            reasons.append("debt_high")
    elif total_liab > 0 and total_assets > 0:
        ratio = total_liab / total_assets * 100
        if ratio < 30:
            dta_score = 0.5
            reasons.append("debt_low")
        elif ratio < 50:
            dta_score = 0.35
            reasons.append("debt_moderate")
        elif ratio < 70:
            dta_score = 0.15
            reasons.append("debt_elevated")
        else:
            dta_score = 0.05
            reasons.append("debt_high")
    else:
        reasons.append("debt_data_insufficient")

    # Interest debt component (0-0.3)
    interest_score = 0.3
    if interest_debt > 0 and total_assets > 0:
        id_ratio = interest_debt / total_assets
        if id_ratio > 0.5:
            interest_score = 0.05
            reasons.append("interest_debt_high")
        elif id_ratio > 0.3:
            interest_score = 0.15
            reasons.append("interest_debt_moderate")
        else:
            interest_score = 0.3
            reasons.append("interest_debt_low")

    # Short-term debt concentration penalty (0-0.2)
    st_penalty = 0.2
    if shortterm_loans > 0 and interest_debt > 0:
        st_ratio = shortterm_loans / interest_debt
        if st_ratio > 0.8:
            st_penalty = 0.0
            reasons.append("st_debt_high")
        elif st_ratio > 0.5:
            st_penalty = 0.10
            reasons.append("st_debt_moderate")
        else:
            st_penalty = 0.2
            reasons.append("st_debt_low")

    score = _clip01(dta_score + interest_score + st_penalty)
    reason_str = ";".join(reasons) if reasons else "data_insufficient"
    return score, reason_str


def _compute_margin_stability_score(
    row: dict[str, Any],
    prev_row: dict[str, Any] | None,
) -> tuple[float, str]:
    """
    Margin Stability Score [0,1].

    Evaluates:
      - Gross margin level
      - Gross margin stability (change from previous period)
    """
    gross_margin = _safe_float(row.get("gross_margin"))
    prev_gm = _safe_float(prev_row.get("gross_margin")) if prev_row else None

    reasons: list[str] = []

    # Gross margin level (0-0.6)
    gm_level_score = 0.0
    if gross_margin > 60:
        gm_level_score = 0.6
        reasons.append("gm_high")
    elif gross_margin > 40:
        gm_level_score = 0.45
        reasons.append("gm_moderate_high")
    elif gross_margin > 20:
        gm_level_score = 0.25
        reasons.append("gm_moderate")
    elif gross_margin > 10:
        gm_level_score = 0.15
        reasons.append("gm_low")
    elif gross_margin > 0:
        gm_level_score = 0.05
        reasons.append("gm_positive")
    else:
        reasons.append("gm_negative_or_zero")

    # Stability component (0-0.4)
    stability_score = 0.0
    if prev_gm is not None and gross_margin > 0:
        gm_change = gross_margin - prev_gm
        if abs(gm_change) < 2:
            stability_score = 0.4
            reasons.append("gm_stable")
        elif abs(gm_change) < 5:
            stability_score = 0.25
            reasons.append("gm_slightly_changed")
        elif gm_change > 0:
            stability_score = 0.15
            reasons.append("gm_improving")
        else:
            stability_score = 0.05
            reasons.append("gm_deteriorating")
    elif gross_margin > 0:
        stability_score = 0.2
        reasons.append("gm_no_prior_data")

    score = _clip01(gm_level_score + stability_score)
    reason_str = ";".join(reasons) if reasons else "data_insufficient"
    return score, reason_str


def _compute_profitability_score(
    row: dict[str, Any],
) -> tuple[float, str]:
    """
    Profitability Score [0,1].

    Evaluates:
      - ROE level
      - Net profit margin (implied from profit_yoy and revenue_yoy)
      - Earnings per share
    """
    roe = _safe_float(row.get("roe"))
    profit_yoy = _safe_float(row.get("profit_yoy"))
    revenue_yoy = _safe_float(row.get("revenue_yoy"))
    eps = _safe_float(row.get("eps"))

    reasons: list[str] = []

    # ROE component (0-0.5)
    roe_score = 0.0
    if roe > 20:
        roe_score = 0.5
        reasons.append("roe_high")
    elif roe > 15:
        roe_score = 0.4
        reasons.append("roe_moderate_high")
    elif roe > 10:
        roe_score = 0.3
        reasons.append("roe_moderate")
    elif roe > 5:
        roe_score = 0.15
        reasons.append("roe_low")
    elif roe > 0:
        roe_score = 0.05
        reasons.append("roe_positive")
    else:
        reasons.append("roe_negative_or_zero")

    # EPS component (0-0.3)
    eps_score = 0.0
    if eps > 2.0:
        eps_score = 0.3
        reasons.append("eps_high")
    elif eps > 1.0:
        eps_score = 0.2
        reasons.append("eps_moderate")
    elif eps > 0.5:
        eps_score = 0.12
        reasons.append("eps_low")
    elif eps > 0:
        eps_score = 0.05
        reasons.append("eps_positive")
    else:
        reasons.append("eps_negative_or_zero")

    # Profitability consistency (0-0.2)
    consistency_score = 0.0
    if profit_yoy > 0 and revenue_yoy > 0:
        consistency_score = 0.2
        reasons.append("dual_profitable")
    elif profit_yoy > 0 or revenue_yoy > 0:
        consistency_score = 0.08
        reasons.append("single_profitable")

    score = _clip01(roe_score + eps_score + consistency_score)
    reason_str = ";".join(reasons) if reasons else "data_insufficient"
    return score, reason_str


def _compute_quality_score(
    growth_accel: float,
    cashflow: float,
    debt_control: float,
    margin_stability: float,
    profitability: float,
) -> float:
    """
    Composite Quality Score [0,1].

    Equal-weighted average of the five sub-scores.
    """
    avg = (growth_accel + cashflow + debt_control + margin_stability + profitability) / 5.0
    return _clip01(avg)


def _determine_risk_flag(
    debt_control: float,
    profitability: float,
    cashflow: float,
    growth_accel: float,
) -> str:
    """Determine fundamental risk flag based on sub-scores."""
    if debt_control < 0.3:
        return "HIGH_DEBT"
    if profitability < 0.2 and growth_accel < 0.2:
        return "NEGATIVE_EARNINGS"
    if cashflow < 0.25:
        return "CASHFLOW_WEAK"
    if debt_control < 0.5 and profitability < 0.3:
        return "HIGH_DEBT"
    return "NONE"



# ---------------------------------------------------------------------------
# Source data coverage preflight
# ---------------------------------------------------------------------------

REQUIRED_SOURCE_TABLES = [
    ("cn_stock_fundamental_daily", "trade_date"),
    ("cn_stock_fina_indicator", "ann_date"),
    ("cn_stock_income", "ann_date"),
    ("cn_stock_balancesheet", "ann_date"),
]

def _normalize_date_value(value):
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return None

def audit_source_data_coverage(engine: Engine, start: date, end: date, lookback_start: date) -> None:
    print(f"[{_ts()}] Source data coverage audit start: {lookback_start} ~ {end}", flush=True)
    failures = []
    for table_name, date_col in REQUIRED_SOURCE_TABLES:
        if not table_exists(engine, engine.url.database, table_name):
            failures.append(f"- {table_name}: table does not exist")
            continue
        sql = f"""
        SELECT COUNT(*) AS row_count, MIN({date_col}) AS min_date, MAX({date_col}) AS max_date
        FROM {table_name}
        WHERE {date_col} BETWEEN :start AND :end
          AND {date_col} IS NOT NULL
        """
        with engine.connect() as conn:
            row = conn.execute(text(sql), {"start": lookback_start, "end": end}).mappings().first()
        row_count = int((row or {}).get("row_count") or 0)
        min_date = _normalize_date_value((row or {}).get("min_date"))
        max_date = _normalize_date_value((row or {}).get("max_date"))
        print(f"[{_ts()}]   {table_name:<30} rows={row_count:,} range={min_date}~{max_date}", flush=True)
        if row_count <= 0 or min_date is None or max_date is None or min_date > lookback_start or max_date < end:
            failures.append(f"- {table_name}.{date_col}: available={min_date}~{max_date}, required={lookback_start}~{end}")
    if failures:
        print("=" * 60, flush=True)
        print("[SOURCE DATA AUDIT FAILED]", flush=True)
        for item in failures:
            print(item, flush=True)
        print("=" * 60, flush=True)
        sys.exit(2)
    print(f"[{_ts()}] Source data coverage audit PASS", flush=True)


# ---------------------------------------------------------------------------
# Main build logic
# ---------------------------------------------------------------------------




def _date_chunks(start: date, end: date, months: int) -> list[tuple[date, date]]:
    """Split [start, end] into month-based chunks."""
    if months <= 0:
        months = 3
    chunks: list[tuple[date, date]] = []
    cur = start
    while cur <= end:
        next_month = date(
            cur.year + (cur.month + months - 1) // 12,
            (cur.month + months - 1) % 12 + 1,
            1,
        )
        chunk_end = min(next_month - timedelta(days=1), end)
        chunks.append((cur, chunk_end))
        cur = next_month
    return chunks


def run_build(args: argparse.Namespace) -> pd.DataFrame:
    """
    Execute the stock quality score build for the given date range.
    Returns the computed DataFrame.
    """
    verbose = args.verbose
    dry_run = args.dry_run

    # Resolve date range
    start = datetime.strptime(args.start, "%Y-%m-%d").date() if "-" in args.start else datetime.strptime(args.start, "%Y%m%d").date()
    end = datetime.strptime(args.end, "%Y-%m-%d").date() if args.end and "-" in args.end else (
        datetime.strptime(args.end, "%Y%m%d").date() if args.end else date.today()
    )
    if start > end:
        print(f"ERROR: start {start} > end {end}")
        sys.exit(1)

    # Need lookback for delta computation and ann_date filtering.
    # Financial data needs ~3 years lookback so early dates have enough
    # historical records with ann_date <= trade_date.
    lookback_start = start - timedelta(days=1100)  # ~3 years

    db_password = args.db_password or os.getenv("ASHARE_MYSQL_PASSWORD", "sec_Bobo123")
    engine = build_engine(args.db_host, args.db_port, args.db_user, db_password, args.db_name)

    build_started_at = time.time()
    print(f"[{_ts()}] Date range: {start} ~ {end}", flush=True)
    print(f"[{_ts()}] Lookback start: {lookback_start}", flush=True)
    print(f"[{_ts()}] Database: {args.db_name} | dry_run={dry_run} | replace={args.replace}", flush=True)

    # ── Load input data ─────────────────────────────────────────────
    # Financial data lookback must be wide enough so that for every trade_date
    # in the target range, we have records with ann_date <= trade_date.
    fina_lookback = start - timedelta(days=1100)  # ~3 years for financial data

    print(f"[{_ts()}] Loading source tables ...", flush=True)
    fund_df = _timed_load("cn_stock_fundamental_daily", load_stock_fundamental_daily, engine, lookback_start, end)
    if fund_df.empty:
        print(f"[{_ts()}] WARNING: cn_stock_fundamental_daily is empty for the range", flush=True)
        return pd.DataFrame()

    fina_df = _timed_load("cn_stock_fina_indicator", load_fina_indicator, engine, fina_lookback, end)
    income_df = _timed_load("cn_stock_income", load_income, engine, fina_lookback, end)
    bs_df = _timed_load("cn_stock_balancesheet", load_balancesheet, engine, fina_lookback, end)

    # ── Prepare fina_indicator data (with ann_date filtering) ───────
    # Build a per-symbol list of records sorted by ann_date, so we can
    # pick the most recent record with ann_date <= trade_date at each row.
    print(f"[{_ts()}] Preparing financial lookup maps ...", flush=True)
    prep_started_at = time.time()
    fina_records_by_symbol: dict[str, list[dict[str, Any]]] = {}
    if not fina_df.empty:
        fina_df["symbol"] = fina_df["symbol"].astype(str).str.split(".").str[0]
        fina_df["end_date"] = pd.to_datetime(fina_df["end_date"], errors="coerce").dt.date
        fina_df["ann_date"] = pd.to_datetime(fina_df["ann_date"], errors="coerce").dt.date
        fina_df = fina_df.sort_values(["symbol", "end_date", "ann_date"], ascending=[True, False, False])
        fina_df = fina_df.drop_duplicates(subset=["symbol", "end_date"], keep="first")
        # Sort by ann_date ascending so we can binary-search for the right record
        fina_df = fina_df.sort_values(["symbol", "ann_date"], ascending=[True, True]).reset_index(drop=True)
        fina_groups = list(fina_df.groupby("symbol"))
        total_fina_groups = len(fina_groups)
        report_every = max(500, total_fina_groups // 10 or 1)
        for i, (sym, grp) in enumerate(fina_groups, 1):
            fina_records_by_symbol[sym] = grp.to_dict(orient="records")
            if i % report_every == 0 or i == total_fina_groups:
                _progress_line("fina lookup", i, total_fina_groups, prep_started_at)

    # ── Prepare income data (with ann_date filtering) ───────────────
    income_records_by_symbol: dict[str, list[dict[str, Any]]] = {}
    if not income_df.empty:
        income_df["symbol"] = income_df["symbol"].astype(str).str.split(".").str[0]
        income_df["end_date"] = pd.to_datetime(income_df["end_date"], errors="coerce").dt.date
        income_df["ann_date"] = pd.to_datetime(income_df["ann_date"], errors="coerce").dt.date
        income_df = income_df.sort_values(["symbol", "end_date", "ann_date"], ascending=[True, False, False])
        income_df = income_df.drop_duplicates(subset=["symbol", "end_date"], keep="first")
        income_df = income_df.sort_values(["symbol", "ann_date"], ascending=[True, True]).reset_index(drop=True)
        income_groups = list(income_df.groupby("symbol"))
        total_income_groups = len(income_groups)
        report_every = max(500, total_income_groups // 10 or 1)
        for i, (sym, grp) in enumerate(income_groups, 1):
            income_records_by_symbol[sym] = grp.to_dict(orient="records")
            if i % report_every == 0 or i == total_income_groups:
                _progress_line("income lookup", i, total_income_groups, prep_started_at)

    # ── Prepare balancesheet data (with ann_date filtering) ─────────
    bs_records_by_symbol: dict[str, list[dict[str, Any]]] = {}
    if not bs_df.empty:
        bs_df["symbol"] = bs_df["symbol"].astype(str).str.split(".").str[0]
        bs_df["end_date"] = pd.to_datetime(bs_df["end_date"], errors="coerce").dt.date
        bs_df["ann_date"] = pd.to_datetime(bs_df["ann_date"], errors="coerce").dt.date
        bs_df = bs_df.sort_values(["symbol", "end_date", "ann_date"], ascending=[True, False, False])
        bs_df = bs_df.drop_duplicates(subset=["symbol", "end_date"], keep="first")
        bs_df = bs_df.sort_values(["symbol", "ann_date"], ascending=[True, True]).reset_index(drop=True)
        bs_groups = list(bs_df.groupby("symbol"))
        total_bs_groups = len(bs_groups)
        report_every = max(500, total_bs_groups // 10 or 1)
        for i, (sym, grp) in enumerate(bs_groups, 1):
            bs_records_by_symbol[sym] = grp.to_dict(orient="records")
            if i % report_every == 0 or i == total_bs_groups:
                _progress_line("balancesheet lookup", i, total_bs_groups, prep_started_at)
    print(f"[{_ts()}] Lookup maps ready in {_fmt_seconds(time.time() - prep_started_at)}", flush=True)

    # ── Helper: find latest record with ann_date <= trade_date ──────
    # ═══════════════════════════════════════════════════════════════════
    # CRITICAL: This function STRICTLY enforces ann_date <= trade_date.
    #   - Records are sorted by ann_date ascending.
    #   - Binary search finds the most recent record with ann_date <= trade_date.
    #   - If ann_date is NULL, the record is treated as UNAVAILABLE (returns None).
    #   - NO fallback to end_date is allowed — end_date is the fiscal period end,
    #     not the disclosure date, and using it would leak future financial data.
    #
    # This constraint applies to ALL 4 financial source tables:
    #   cn_stock_fina_indicator, cn_stock_income, cn_stock_balancesheet
    # ═══════════════════════════════════════════════════════════════════
    def _find_latest_before(
        records: list[dict[str, Any]], trade_date: date
    ) -> dict[str, Any] | None:
        """
        Binary-search for the most recent record with ann_date <= trade_date.
        
        STRICT ann_date enforcement:
        - If ann_date is NULL or NaT, returns None (data treated as unavailable).
        - NO end_date fallback — end_date is fiscal period end, not disclosure date.
        """
        if not records:
            return None
        lo, hi = 0, len(records) - 1
        best_idx = -1
        while lo <= hi:
            mid = (lo + hi) // 2
            rec_ann = records[mid].get("ann_date")
            # STRICT: if ann_date is NULL/NaT, treat as unavailable — skip this record
            if rec_ann is None or pd.isna(rec_ann):
                hi = mid - 1
                continue
            if isinstance(rec_ann, pd.Timestamp):
                rec_ann = rec_ann.date()
            if rec_ann <= trade_date:
                best_idx = mid
                lo = mid + 1
            else:
                hi = mid - 1
        if best_idx >= 0:
            return records[best_idx]
        return None

    # ── Compute scores per stock per trade_date ─────────────────────
    results: list[dict[str, Any]] = []
    prev_by_symbol: dict[str, dict[str, Any]] = {}

    fund_df = fund_df.sort_values(["symbol", "trade_date"]).reset_index(drop=True)
    total_rows = len(fund_df)
    target_rows = len(fund_df[fund_df["trade_date"] >= start]) if "trade_date" in fund_df.columns else total_rows
    report_every = max(5000, total_rows // 20 or 1)
    compute_started_at = time.time()
    print(f"[{_ts()}] Computing quality scores ... total_input={total_rows:,}, target_rows≈{target_rows:,}", flush=True)

    for idx, row in fund_df.iterrows():
        symbol = row["symbol"]
        trade_date_val = row["trade_date"]
        if isinstance(trade_date_val, str):
            trade_date_val = datetime.strptime(trade_date_val, "%Y-%m-%d").date()
        elif isinstance(trade_date_val, datetime):
            trade_date_val = trade_date_val.date()

        # Skip lookback dates for final output
        if trade_date_val < start:
            prev_by_symbol[symbol] = row.to_dict()
            continue

        # ── Future-function safety: skip if ann_date is NULL or > trade_date ──
        # SOURCE 1: cn_stock_fundamental_daily
        #   Constraint: ann_date <= trade_date
        #   Each row in cn_stock_fundamental_daily has its own ann_date.
        #   If ann_date is NULL, data availability is uncertain → SKIP.
        #   If ann_date > trade_date, the financial data was not yet disclosed
        #   on this trade_date → SKIP.
        row_ann_date = row.get("ann_date")
        if row_ann_date is None:
            # ann_date is NULL — data availability uncertain, skip to be safe
            if verbose:
                print(f"    [SKIP] {symbol} @ {trade_date_val}: ann_date is NULL")
            continue
        if isinstance(row_ann_date, str):
            row_ann_date = datetime.strptime(row_ann_date, "%Y-%m-%d").date()
        elif isinstance(row_ann_date, datetime):
            row_ann_date = row_ann_date.date()
        if row_ann_date > trade_date_val:
            # This row's financial data was announced after this trade_date
            if verbose:
                print(f"    [SKIP] {symbol} @ {trade_date_val}: ann_date {row_ann_date} > trade_date")
            continue

        # Build enriched row with data from all sources
        enriched = row.to_dict()
        enriched["trade_date"] = trade_date_val

        # ── Merge fina_indicator data ─────────────────────────────────
        # SOURCE 2: cn_stock_fina_indicator
        #   Constraint: ann_date <= trade_date (enforced by _find_latest_before)
        #   If ann_date is NULL → returns None → all fields remain None
        #   → defaults to 0.5 in score computation (neutral, no future data leak)
        fina_records = fina_records_by_symbol.get(symbol, [])
        fi = _find_latest_before(fina_records, trade_date_val)
        if fi is not None:
            enriched["eps"] = fi.get("eps")
            enriched["gross_margin"] = enriched.get("gross_margin") or fi.get("grossprofit_margin")
            enriched["debt_to_assets"] = enriched.get("debt_to_assets") or fi.get("debt_to_assets")
            enriched["ocfps"] = enriched.get("ocfps") or fi.get("ocfps")
            enriched["roe"] = enriched.get("roe") or fi.get("roe")
            if pd.isna(enriched.get("revenue_yoy")) or enriched.get("revenue_yoy") is None:
                enriched["revenue_yoy"] = fi.get("or_yoy") or fi.get("tr_yoy") or fi.get("q_sales_yoy")
            if pd.isna(enriched.get("profit_yoy")) or enriched.get("profit_yoy") is None:
                enriched["profit_yoy"] = fi.get("netprofit_yoy") or fi.get("q_profit_yoy")
            enriched["report_end_date"] = enriched.get("report_end_date") or fi.get("end_date")
            enriched["ann_date"] = enriched.get("ann_date") or fi.get("ann_date")

        # ── Merge income data ─────────────────────────────────────────
        # SOURCE 3: cn_stock_income
        #   Constraint: ann_date <= trade_date (enforced by _find_latest_before)
        #   If ann_date is NULL → returns None → n_income_attr_p remains None
        #   → defaults to 0.5 in cashflow_score computation (neutral)
        income_records = income_records_by_symbol.get(symbol, [])
        inc = _find_latest_before(income_records, trade_date_val)
        if inc is not None:
            enriched["n_income_attr_p"] = enriched.get("n_income_attr_p") or inc.get("n_income_attr_p")
            enriched["report_end_date"] = enriched.get("report_end_date") or inc.get("end_date")
            enriched["ann_date"] = enriched.get("ann_date") or inc.get("ann_date")

        # ── Merge balancesheet data ────────────────────────────────────
        # SOURCE 4: cn_stock_balancesheet
        #   Constraint: ann_date <= trade_date (enforced by _find_latest_before)
        #   If ann_date is NULL → returns None → all fields remain None
        #   → defaults to 0.5 in debt_control/cashflow score computation (neutral)
        bs_records = bs_records_by_symbol.get(symbol, [])
        b = _find_latest_before(bs_records, trade_date_val)
        if b is not None:
            enriched["total_assets"] = b.get("total_assets")
            enriched["total_liab"] = b.get("total_liab")
            enriched["interest_debt"] = b.get("interest_debt")
            enriched["shortterm_loans"] = b.get("shortterm_loans")
            enriched["longterm_loans"] = b.get("longterm_loans")
            enriched["net_cashflow_act"] = b.get("net_cashflow_act")
            enriched["inventory"] = enriched.get("inventory") or b.get("inventory")
            enriched["fixed_assets"] = enriched.get("fixed_assets") or b.get("fixed_assets")

        # Get previous row for delta computation
        prev_row = prev_by_symbol.get(symbol)

        # ── Compute sub-scores ──────────────────────────────────────
        growth_accel, growth_reason = _compute_growth_acceleration(enriched, prev_row)
        cashflow, cf_reason = _compute_cashflow_score(enriched)
        debt_control, debt_reason = _compute_debt_control_score(enriched)
        margin_stability, margin_reason = _compute_margin_stability_score(enriched, prev_row)
        profitability, profit_reason = _compute_profitability_score(enriched)

        # Composite quality score
        quality = _compute_quality_score(growth_accel, cashflow, debt_control, margin_stability, profitability)

        # Risk flag
        risk_flag = _determine_risk_flag(debt_control, profitability, cashflow, growth_accel)

        # Build reason string
        all_reasons = [growth_reason, cf_reason, debt_reason, margin_reason, profit_reason]
        reason_str = ";".join(all_reasons)

        results.append({
            "trade_date": trade_date_val,
            "symbol": symbol,
            "quality_score": round(quality, 6),
            "growth_acceleration_score": round(growth_accel, 6),
            "cashflow_score": round(cashflow, 6),
            "debt_control_score": round(debt_control, 6),
            "margin_stability_score": round(margin_stability, 6),
            "profitability_score": round(profitability, 6),
            "report_end_date": enriched.get("report_end_date"),
            "ann_date": enriched.get("ann_date"),
            "fundamental_risk_flag": risk_flag,
            "reason": reason_str,
        })

        # Update previous state
        prev_by_symbol[symbol] = enriched

        if (idx + 1) % report_every == 0 or idx == total_rows - 1:
            _progress_line("quality rows", idx + 1, total_rows, compute_started_at, f"output={len(results):,}")

    print(f"[{_ts()}] Quality score computation complete: output={len(results):,} rows ({_fmt_seconds(time.time() - compute_started_at)})", flush=True)

    if not results:
        print("WARNING: No results computed")
        return pd.DataFrame()

    result_df = pd.DataFrame(results)

    # Filter to requested date range only
    result_df = result_df[result_df["trade_date"].between(start, end)].copy()

    print(f"[{_ts()}] Computed {len(result_df):,} rows", flush=True)
    print(f"[{_ts()}] Quality score range: [{result_df['quality_score'].min():.4f}, {result_df['quality_score'].max():.4f}]", flush=True)
    print(f"[{_ts()}] Risk flag distribution: {result_df['fundamental_risk_flag'].value_counts().to_dict()}", flush=True)
    print(f"[{_ts()}] Build stage elapsed: {_fmt_seconds(time.time() - build_started_at)}", flush=True)

    return result_df


# ---------------------------------------------------------------------------
# DB write
# ---------------------------------------------------------------------------


def write_to_db(
    engine: Engine,
    df: pd.DataFrame,
    db_name: str,
    replace: bool,
    dry_run: bool,
    verbose: bool,
) -> int:
    """Write computed DataFrame to cn_stock_quality_score_daily. Returns row count."""
    if df.empty:
        print("WARNING: No data to write")
        return 0

    if dry_run:
        print(f"[{_ts()}] [DRY-RUN] Would write {len(df):,} rows to cn_stock_quality_score_daily", flush=True)
        return len(df)

    write_started_at = time.time()
    print(f"[{_ts()}] Writing to DB: cn_stock_quality_score_daily rows={len(df):,}", flush=True)

    # Ensure table exists
    if not table_exists(engine, db_name, "cn_stock_quality_score_daily"):
        print("ERROR: cn_stock_quality_score_daily table does not exist. Run DDL first.")
        return 0

    # If replace, delete existing rows in date range
    if replace:
        min_date = df["trade_date"].min()
        max_date = df["trade_date"].max()
        if isinstance(min_date, str):
            min_date = datetime.strptime(min_date, "%Y-%m-%d").date()
        elif isinstance(min_date, datetime):
            min_date = min_date.date()
        if isinstance(max_date, str):
            max_date = datetime.strptime(max_date, "%Y-%m-%d").date()
        elif isinstance(max_date, datetime):
            max_date = max_date.date()
        del_sql = """
        DELETE FROM cn_stock_quality_score_daily
        WHERE trade_date BETWEEN :start AND :end
        """
        with engine.begin() as conn:
            deleted = conn.execute(text(del_sql), {"start": min_date, "end": max_date}).rowcount
        print(f"[{_ts()}]   Deleted {deleted:,} existing rows in [{min_date}, {max_date}]", flush=True)

    # Prepare rows for upsert
    columns = [
        "trade_date", "symbol", "quality_score", "growth_acceleration_score",
        "cashflow_score", "debt_control_score", "margin_stability_score",
        "profitability_score", "report_end_date", "ann_date",
        "fundamental_risk_flag", "reason",
    ]
    upsert_sql = """
    INSERT INTO cn_stock_quality_score_daily (
        trade_date, symbol, quality_score, growth_acceleration_score,
        cashflow_score, debt_control_score, margin_stability_score,
        profitability_score, report_end_date, ann_date,
        fundamental_risk_flag, reason
    ) VALUES (
        :trade_date, :symbol, :quality_score, :growth_acceleration_score,
        :cashflow_score, :debt_control_score, :margin_stability_score,
        :profitability_score, :report_end_date, :ann_date,
        :fundamental_risk_flag, :reason
    )
    ON DUPLICATE KEY UPDATE
        quality_score = VALUES(quality_score),
        growth_acceleration_score = VALUES(growth_acceleration_score),
        cashflow_score = VALUES(cashflow_score),
        debt_control_score = VALUES(debt_control_score),
        margin_stability_score = VALUES(margin_stability_score),
        profitability_score = VALUES(profitability_score),
        report_end_date = VALUES(report_end_date),
        ann_date = VALUES(ann_date),
        fundamental_risk_flag = VALUES(fundamental_risk_flag),
        reason = VALUES(reason),
        updated_at = CURRENT_TIMESTAMP
    """

    write_df = df[columns].copy()
    write_df["trade_date"] = write_df["trade_date"].apply(
        lambda d: d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d)
    )

    rows = write_df.astype(object).where(pd.notna(write_df), None).to_dict(orient="records")

    total = 0
    batch_size = 4000
    with engine.begin() as conn:
        for i in range(0, len(rows), batch_size):
            batch = rows[i : i + batch_size]
            conn.execute(text(upsert_sql), batch)
            total += len(batch)
            _progress_line("DB write", total, len(rows), write_started_at)

    print(f"[{_ts()}] DB write complete: {total:,} rows ({_fmt_seconds(time.time() - write_started_at)})", flush=True)
    return total


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def generate_reports(
    df: pd.DataFrame,
    start: date,
    end: date,
    output_dir: str | None,
) -> tuple[Path, Path]:
    """Generate summary CSV and Markdown reports. Returns (csv_path, md_path)."""
    report_dir = Path(output_dir) if output_dir else REPORT_DIR
    report_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    start_str = start.strftime("%Y%m%d")
    end_str = end.strftime("%Y%m%d")
    base_name = f"stock_quality_summary_{start_str}_{end_str}_{timestamp}"

    csv_path = report_dir / f"{base_name}.csv"
    md_path = report_dir / f"{base_name}.md"

    if df.empty:
        summary_rows = [{
            "trade_date": "N/A",
            "symbol_count": 0,
            "avg_quality_score": 0,
            "avg_growth_acceleration": 0,
            "avg_cashflow": 0,
            "avg_debt_control": 0,
            "avg_margin_stability": 0,
            "avg_profitability": 0,
            "risk_flag_distribution": "N/A",
            "row_count": 0,
        }]
    else:
        summary_rows = []
        for trade_date_val, group in df.groupby("trade_date"):
            td = trade_date_val
            td_str = td.strftime("%Y-%m-%d") if hasattr(td, "strftime") else str(td)
            risk_dist = group["fundamental_risk_flag"].value_counts().to_dict()
            summary_rows.append({
                "trade_date": td_str,
                "symbol_count": len(group),
                "avg_quality_score": round(group["quality_score"].mean(), 4),
                "avg_growth_acceleration": round(group["growth_acceleration_score"].mean(), 4),
                "avg_cashflow": round(group["cashflow_score"].mean(), 4),
                "avg_debt_control": round(group["debt_control_score"].mean(), 4),
                "avg_margin_stability": round(group["margin_stability_score"].mean(), 4),
                "avg_profitability": round(group["profitability_score"].mean(), 4),
                "risk_flag_distribution": str(risk_dist),
                "row_count": len(group),
            })

    summary_df = pd.DataFrame(summary_rows)

    # Write CSV
    summary_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"[REPORT] CSV -> {csv_path}")

    # Write Markdown
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Stock Quality Score Summary\n\n")
        f.write(f"**Date Range:** {start_str} ~ {end_str}\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Total Rows:** {len(df)}\n\n")
        f.write(f"---\n\n")

        for _, srow in summary_df.iterrows():
            f.write(f"## {srow['trade_date']}\n\n")
            f.write(f"- Symbol Count: {srow['symbol_count']}\n")
            f.write(f"- Avg Quality Score: {srow['avg_quality_score']}\n")
            f.write(f"- Avg Growth Acceleration: {srow['avg_growth_acceleration']}\n")
            f.write(f"- Avg Cashflow: {srow['avg_cashflow']}\n")
            f.write(f"- Avg Debt Control: {srow['avg_debt_control']}\n")
            f.write(f"- Avg Margin Stability: {srow['avg_margin_stability']}\n")
            f.write(f"- Avg Profitability: {srow['avg_profitability']}\n")
            f.write(f"- Risk Flag Distribution: {srow['risk_flag_distribution']}\n")
            f.write(f"- Rows Written: {srow['row_count']}\n\n")
            f.write(f"---\n\n")

        f.write(f"## Overall Score Distribution\n\n")
        f.write(f"- Quality Score: mean={df['quality_score'].mean():.4f}, std={df['quality_score'].std():.4f}\n")
        f.write(f"- Growth Acceleration: mean={df['growth_acceleration_score'].mean():.4f}\n")
        f.write(f"- Cashflow: mean={df['cashflow_score'].mean():.4f}\n")
        f.write(f"- Debt Control: mean={df['debt_control_score'].mean():.4f}\n")
        f.write(f"- Margin Stability: mean={df['margin_stability_score'].mean():.4f}\n")
        f.write(f"- Profitability: mean={df['profitability_score'].mean():.4f}\n\n")

        f.write(f"## Risk Flag Distribution\n\n")
        risk_counts = df["fundamental_risk_flag"].value_counts()
        for flag_name, count in risk_counts.items():
            pct = count / len(df) * 100
            f.write(f"- **{flag_name}**: {count} ({pct:.1f}%)\n")

    print(f"[REPORT] MD  -> {md_path}")
    return csv_path, md_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    print("=" * 60)
    print("  GrowthAlpha V8 — P3 Stock Quality Score Builder")
    print("=" * 60)
    print(f"  Start: {args.start}")
    print(f"  End:   {args.end or 'today'}")
    print(f"  DB:    {args.db_name}")
    print(f"  Dry-run: {args.dry_run}")
    print(f"  Replace: {args.replace}")
    print()

    db_password = args.db_password or os.getenv("ASHARE_MYSQL_PASSWORD", "sec_Bobo123")
    engine = build_engine(args.db_host, args.db_port, args.db_user, db_password, args.db_name)

    # Ensure DDL
    ddl_path = Path(__file__).resolve().parents[1] / "sql" / "create_stock_quality_score_daily.sql"
    if ddl_path.exists():
        ddl_sql = ddl_path.read_text(encoding="utf-8")
        stmts = [s.strip() for s in ddl_sql.split(";") if s.strip()]
        with engine.begin() as conn:
            for stmt in stmts:
                if "CREATE TABLE" in stmt:
                    conn.execute(text(stmt))
        print(f"[DDL] Ensured cn_stock_quality_score_daily table exists")
    else:
        print(f"[WARNING] DDL file not found: {ddl_path}")

    # Run build in chunks so cn_stock_fundamental_daily is not loaded for the full range at once.
    start = datetime.strptime(args.start, "%Y-%m-%d").date() if "-" in args.start else datetime.strptime(args.start, "%Y%m%d").date()
    end = datetime.strptime(args.end, "%Y-%m-%d").date() if args.end and "-" in args.end else (
        datetime.strptime(args.end, "%Y%m%d").date() if args.end else date.today()
    )

    lookback_start = start - timedelta(days=1100)
    audit_source_data_coverage(engine, start, end, lookback_start)

    chunks = _date_chunks(start, end, getattr(args, "chunk_months", 3))
    total_chunks = len(chunks)
    total_written = 0
    total_computed = 0
    all_results: list[pd.DataFrame] = []
    overall_started_at = time.time()

    print(f"[{_ts()}] Chunked build enabled: chunks={total_chunks}, chunk_months={getattr(args, 'chunk_months', 3)}", flush=True)

    for chunk_idx, (chunk_start, chunk_end) in enumerate(chunks, 1):
        chunk_started_at = time.time()
        remaining_before = total_chunks - chunk_idx
        print("", flush=True)
        print("=" * 60, flush=True)
        print(
            f"[{_ts()}] Chunk {chunk_idx}/{total_chunks}: {chunk_start} ~ {chunk_end} "
            f"(elapsed={_fmt_seconds(time.time() - overall_started_at)})",
            flush=True,
        )
        print("=" * 60, flush=True)

        chunk_args = argparse.Namespace(**vars(args))
        chunk_args.start = chunk_start.strftime("%Y-%m-%d")
        chunk_args.end = chunk_end.strftime("%Y-%m-%d")

        chunk_df = run_build(chunk_args)
        if chunk_df.empty:
            print(f"[{_ts()}] Chunk {chunk_idx}/{total_chunks} produced no rows", flush=True)
            continue

        written = write_to_db(engine, chunk_df, args.db_name, args.replace, args.dry_run, args.verbose)
        total_written += written
        total_computed += len(chunk_df)
        all_results.append(chunk_df)

        chunk_elapsed = time.time() - chunk_started_at
        eta_seconds = remaining_before * chunk_elapsed
        print(
            f"[{_ts()}] Chunk {chunk_idx}/{total_chunks} done: "
            f"computed={len(chunk_df):,}, written={written:,}, "
            f"chunk_elapsed={_fmt_seconds(chunk_elapsed)}, eta≈{_fmt_seconds(eta_seconds)}",
            flush=True,
        )

    if not all_results:
        print("No data computed. Exiting.")
        sys.exit(0)

    report_started_at = time.time()
    print(f"[{_ts()}] Generating reports from chunk outputs ...", flush=True)
    result_df = pd.concat(all_results, ignore_index=True)
    csv_path, md_path = generate_reports(result_df, start, end, args.output_dir)
    print(f"[{_ts()}] Reports complete ({_fmt_seconds(time.time() - report_started_at)})", flush=True)

    print()
    print("=" * 60)
    print(f"  Build Complete")
    print(f"  Chunks:         {total_chunks}")
    print(f"  Rows computed:  {total_computed}")
    print(f"  Rows written:   {total_written}")
    print(f"  Total elapsed:  {_fmt_seconds(time.time() - overall_started_at)}")
    print(f"  CSV report:     {csv_path}")
    print(f"  MD  report:     {md_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()