from __future__ import annotations

import argparse
import calendar
import subprocess
from datetime import date, datetime
from typing import Iterable, List, Set
from pathlib import Path
import sys

# Ensure project root is on sys.path when run directly.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import akshare as ak
from sqlalchemy import text

from app.defaults import DEFAULT_INDEX_SYMBOLS
from app.settings import build_engine


def _parse_ymd(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def _fmt_ymd(d: date) -> str:
    return d.strftime("%Y-%m-%d")


def _month_start(d: date) -> date:
    return date(d.year, d.month, 1)


def _month_end(d: date) -> date:
    last = calendar.monthrange(d.year, d.month)[1]
    return date(d.year, d.month, last)


def _iter_trade_dates(start: date, end: date) -> List[date]:
    df = ak.tool_trade_date_hist_sina()
    if df is None or df.empty:
        raise RuntimeError("tool_trade_date_hist_sina returned empty")
    col = df.columns[0]

    def to_date(v):
        if isinstance(v, date):
            return v
        if hasattr(v, "date"):
            return v.date()
        s = str(v).strip()
        if len(s) == 10:
            return date.fromisoformat(s)
        return date(int(s[:4]), int(s[4:6]), int(s[6:8]))

    all_dates = sorted([to_date(v) for v in df[col].tolist()])
    return [d for d in all_dates if start <= d <= end]


def _get_existing_index_dates(index_code: str, start: date, end: date) -> Set[date]:
    engine = build_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT TRADE_DATE
                FROM cn_index_daily_price
                WHERE index_code = :idx
                  AND TRADE_DATE BETWEEN :s AND :e
                """
            ),
            {"idx": index_code, "s": _fmt_ymd(start), "e": _fmt_ymd(end)},
        ).fetchall()
    return {r[0] for r in rows}


def _missing_months(
    index_codes: Iterable[str], trade_days: List[date], start: date, end: date
) -> Set[str]:
    base_set = set(trade_days)
    missing_months: Set[str] = set()
    for code in index_codes:
        existing = _get_existing_index_dates(code, start, end)
        missing = base_set - existing
        for d in missing:
            missing_months.add(d.strftime("%Y-%m"))
    return missing_months


def _months_sorted(months: Set[str]) -> List[str]:
    return sorted(months)


def _run_month(python_exe: str, month: str, no_vpn: bool) -> None:
    y, m = map(int, month.split("-"))
    m_start = date(y, m, 1)
    m_end = _month_end(m_start)
    days = (m_end - m_start).days + 1
    asof = m_end.strftime("%Y%m%d")
    args = [
        python_exe,
        "runner.py",
        "--tasks",
        "index",
        "--asof",
        asof,
        "--days",
        str(days),
    ]
    if no_vpn:
        args.append("--no-vpn")
    subprocess.run(args, check=False)


def main() -> int:
    p = argparse.ArgumentParser(description="Audit index gaps and backfill missing months.")
    p.add_argument("--start", required=True, help="YYYY-MM-DD")
    p.add_argument("--end", required=True, help="YYYY-MM-DD")
    p.add_argument("--no-vpn", action="store_true", help="pass --no-vpn to runner")
    p.add_argument("--dry-run", action="store_true", help="only print missing months, do not run backfill")
    args = p.parse_args()

    start = _parse_ymd(args.start)
    end = _parse_ymd(args.end)
    if start > end:
        raise SystemExit("start must be <= end")

    trade_days = _iter_trade_dates(start, end)
    months = _months_sorted(_missing_months(DEFAULT_INDEX_SYMBOLS, trade_days, start, end))

    if not months:
        print("no missing months detected")
        return 0

    print(f"missing months: {len(months)}")
    for m in months:
        print(m)

    if args.dry_run:
        return 0

    python_exe = sys.executable
    for m in months:
        _run_month(python_exe, m, args.no_vpn)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
