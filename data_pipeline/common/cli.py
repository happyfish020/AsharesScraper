from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, datetime, timedelta


def parse_ymd(value: str) -> date:
    text = str(value or "").strip()
    if len(text) == 10 and text[4] == "-" and text[7] == "-":
        return datetime.strptime(text, "%Y-%m-%d").date()
    return datetime.strptime(text, "%Y%m%d").date()


def format_ymd(value: date) -> str:
    return value.strftime("%Y%m%d")


def month_chunks(start: date, end: date, months_per_chunk: int = 1) -> list[tuple[date, date]]:
    out: list[tuple[date, date]] = []
    months_per_chunk = max(1, int(months_per_chunk))
    cur = start.replace(day=1)
    while cur <= end:
        chunk_start = max(cur, start)
        year = cur.year
        month = cur.month + months_per_chunk - 1
        year += (month - 1) // 12
        month = ((month - 1) % 12) + 1
        if month == 12:
            chunk_end = date(year, 12, 31)
        else:
            chunk_end = date(year, month + 1, 1) - timedelta(days=1)
        if chunk_end > end:
            chunk_end = end
        out.append((chunk_start, chunk_end))
        next_month = cur.month + months_per_chunk
        next_year = cur.year + (next_month - 1) // 12
        next_month = ((next_month - 1) % 12) + 1
        cur = date(next_year, next_month, 1)
    return out


def quarter_periods(start: date, end: date, lookback_years: int = 2) -> list[str]:
    out: list[str] = []
    begin_year = start.year - max(0, lookback_years)
    finish_year = end.year
    for year in range(begin_year, finish_year + 1):
        for month, day in ((3, 31), (6, 30), (9, 30), (12, 31)):
            period = date(year, month, day)
            if period <= end:
                out.append(period.strftime("%Y%m%d"))
    return out


def add_shared_args(parser: argparse.ArgumentParser, *, default_start: str = "2010-01-01") -> argparse.ArgumentParser:
    parser.add_argument("--start", default=default_start, help="YYYY-MM-DD or YYYYMMDD")
    parser.add_argument("--end", default="", help="YYYY-MM-DD or YYYYMMDD, default=today")
    parser.add_argument("--resume", action="store_true", help="Skip completed chunks in cn_mainline_backfill_job_state")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers")
    parser.add_argument("--chunk-months", type=int, default=1, help="Month count per chunk for daily builders")
    parser.add_argument("--token", default="", help="Tushare token")
    parser.add_argument("--config", default="", help="Optional config path for Tushare token")
    return parser


@dataclass(slots=True)
class DateRange:
    start: date
    end: date


def resolve_date_range(start_text: str, end_text: str) -> DateRange:
    start = parse_ymd(start_text)
    end = parse_ymd(end_text) if str(end_text or "").strip() else date.today()
    if start > end:
        raise SystemExit(f"invalid date range: {start} > {end}")
    return DateRange(start=start, end=end)
