#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GrowthAlpha V8 — Build board member map full range.

Purpose
-------
Build cn_board_member_map_d independently and serially by calling:

    CALL sp_build_board_member_map(:d1, :d2)

This avoids multiple daily.bat processes trying to auto-backfill the same
board-member map dates in parallel, which can cause MySQL lock wait timeout
errors.

Typical usage
-------------
python scripts/build_board_member_map_full.py --start 2010-01-01 --end 2026-03-31 --resume ^
  --db-host 127.0.0.1 --db-port 3306 --db-user cn_opr_red --db-password sec_Bobo123 --db-name cn_market_red

Force rebuild the range:
python scripts/build_board_member_map_full.py --start 2010-01-01 --end 2026-03-31 --replace ^
  --db-host 127.0.0.1 --db-port 3306 --db-user cn_opr_red --db-password sec_Bobo123 --db-name cn_market_red
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import date, datetime
from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, URL
from sqlalchemy.exc import OperationalError


def _parse_date(value: str) -> date:
    s = str(value).strip()
    if len(s) == 8 and s.isdigit():
        return datetime.strptime(s, "%Y%m%d").date()
    return datetime.strptime(s, "%Y-%m-%d").date()


def _ensure_date(value: Any) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return _parse_date(str(value))


def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="GrowthAlpha V8 — serial full builder for cn_board_member_map_d"
    )
    parser.add_argument("--start", required=True, help="Start date, YYYY-MM-DD or YYYYMMDD")
    parser.add_argument("--end", required=True, help="End date, YYYY-MM-DD or YYYYMMDD")
    parser.add_argument("--db-host", default="127.0.0.1", help="MySQL host")
    parser.add_argument("--db-port", type=int, default=3306, help="MySQL port")
    parser.add_argument("--db-user", default="root", help="MySQL user")
    parser.add_argument(
        "--db-password",
        default=None,
        help="MySQL password. Defaults to ASHARE_MYSQL_PASSWORD env var.",
    )
    parser.add_argument("--db-name", default="cn_market_red", help="MySQL database name")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Only build price trade dates missing in cn_board_member_map_d.",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Delete cn_board_member_map_d rows in the date range before rebuilding all price trade dates.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show target dates without deleting or calling the stored procedure.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Retry count for lock wait timeout / deadlock errors.",
    )
    parser.add_argument(
        "--retry-sleep",
        type=float,
        default=5.0,
        help="Base sleep seconds between retries. Actual sleep increases linearly.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=50,
        help="Print progress every N trade dates.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop immediately on a per-date failure instead of continuing after retries.",
    )
    return parser


def build_engine(args: argparse.Namespace) -> Engine:
    password = args.db_password if args.db_password is not None else os.getenv("ASHARE_MYSQL_PASSWORD", "")
    url = URL.create(
        drivername="mysql+pymysql",
        username=args.db_user,
        password=password,
        host=args.db_host,
        port=args.db_port,
        database=args.db_name,
        query={"charset": "utf8mb4"},
    )
    return create_engine(url, pool_pre_ping=True, future=True)


def _scalar(engine: Engine, sql: str, params: dict[str, Any] | None = None) -> Any:
    with engine.connect() as conn:
        return conn.execute(text(sql), params or {}).scalar()


def _table_exists(engine: Engine, table_name: str) -> bool:
    n = _scalar(
        engine,
        """
        SELECT COUNT(*)
        FROM information_schema.tables
        WHERE table_schema = DATABASE()
          AND table_name = :table_name
        """,
        {"table_name": table_name},
    )
    return int(n or 0) > 0


def _proc_exists(engine: Engine, proc_name: str) -> bool:
    n = _scalar(
        engine,
        """
        SELECT COUNT(*)
        FROM information_schema.routines
        WHERE routine_schema = DATABASE()
          AND routine_type = 'PROCEDURE'
          AND routine_name = :proc_name
        """,
        {"proc_name": proc_name},
    )
    return int(n or 0) > 0


def _load_price_trade_dates(engine: Engine, start: date, end: date) -> list[date]:
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT DISTINCT trade_date
                FROM cn_stock_daily_price
                WHERE trade_date BETWEEN :start AND :end
                ORDER BY trade_date
                """
            ),
            {"start": start, "end": end},
        ).fetchall()
    return [_ensure_date(r[0]) for r in rows]


def _find_missing_dates(engine: Engine, start: date, end: date) -> list[date]:
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT p.trade_date
                FROM (
                    SELECT DISTINCT trade_date
                    FROM cn_stock_daily_price
                    WHERE trade_date BETWEEN :start AND :end
                ) p
                WHERE NOT EXISTS (
                    SELECT 1
                    FROM cn_board_member_map_d m
                    WHERE m.trade_date = p.trade_date
                    LIMIT 1
                )
                ORDER BY p.trade_date
                """
            ),
            {"start": start, "end": end},
        ).fetchall()
    return [_ensure_date(r[0]) for r in rows]


def _delete_range(engine: Engine, start: date, end: date) -> int:
    with engine.begin() as conn:
        result = conn.execute(
            text(
                """
                DELETE FROM cn_board_member_map_d
                WHERE trade_date BETWEEN :start AND :end
                """
            ),
            {"start": start, "end": end},
        )
        return int(result.rowcount or 0)


def _is_retryable_mysql_error(exc: Exception) -> bool:
    message = str(exc).lower()
    # 1205 lock wait timeout, 1213 deadlock.
    retry_tokens = [
        "lock wait timeout",
        "deadlock found",
        "(1205",
        "(1213",
        "try restarting transaction",
    ]
    return any(token in message for token in retry_tokens)


def _call_sp_for_date(
    engine: Engine,
    trade_date: date,
    max_retries: int,
    retry_sleep: float,
) -> None:
    attempt = 0
    while True:
        try:
            # One connection/transaction per date: keeps locks short and makes retry safe.
            with engine.begin() as conn:
                conn.execute(
                    text("CALL sp_build_board_member_map(:d1, :d2)"),
                    {"d1": trade_date, "d2": trade_date},
                )
            return
        except OperationalError as exc:
            attempt += 1
            if attempt > max_retries or not _is_retryable_mysql_error(exc):
                raise
            sleep_seconds = retry_sleep * attempt
            print(
                f"[{_ts()}] WARN lock/deadlock on {trade_date}; "
                f"retry {attempt}/{max_retries} after {sleep_seconds:.1f}s"
            )
            time.sleep(sleep_seconds)


def _count_rows(engine: Engine, start: date, end: date) -> int:
    value = _scalar(
        engine,
        """
        SELECT COUNT(*)
        FROM cn_board_member_map_d
        WHERE trade_date BETWEEN :start AND :end
        """,
        {"start": start, "end": end},
    )
    return int(value or 0)


def main() -> None:
    args = build_parser().parse_args()

    start = _parse_date(args.start)
    end = _parse_date(args.end)
    if start > end:
        print(f"ERROR: start {start} > end {end}", file=sys.stderr)
        sys.exit(2)
    if args.resume and args.replace:
        print("ERROR: --resume and --replace cannot be used together.", file=sys.stderr)
        sys.exit(2)

    engine = build_engine(args)

    print("=" * 72)
    print(" GrowthAlpha V8 — Build Board Member Map Full")
    print("=" * 72)
    print(f" DB:        {args.db_name}@{args.db_host}:{args.db_port}")
    print(f" Range:     {start} ~ {end}")
    print(f" Mode:      {'replace' if args.replace else 'resume' if args.resume else 'all'}")
    print(f" Dry-run:   {args.dry_run}")
    print()

    required_tables = ["cn_stock_daily_price", "cn_board_member_map_d"]
    for table_name in required_tables:
        if not _table_exists(engine, table_name):
            print(f"ERROR: required table missing: {table_name}", file=sys.stderr)
            sys.exit(1)

    if not _proc_exists(engine, "sp_build_board_member_map"):
        print("ERROR: required procedure missing: sp_build_board_member_map", file=sys.stderr)
        sys.exit(1)

    price_dates = _load_price_trade_dates(engine, start, end)
    if not price_dates:
        print(f"ERROR: no cn_stock_daily_price trade dates found in {start} ~ {end}", file=sys.stderr)
        sys.exit(1)

    if args.replace:
        target_dates = price_dates
        print(f"[{_ts()}] Target price trade dates: {len(target_dates):,}")
        if args.dry_run:
            print(f"[{_ts()}] [dry-run] Would delete cn_board_member_map_d rows in {start} ~ {end}")
        else:
            deleted = _delete_range(engine, start, end)
            print(f"[{_ts()}] Deleted existing cn_board_member_map_d rows: {deleted:,}")
    elif args.resume:
        target_dates = _find_missing_dates(engine, start, end)
        print(f"[{_ts()}] Missing board-member map trade dates: {len(target_dates):,}/{len(price_dates):,}")
    else:
        target_dates = price_dates
        print(f"[{_ts()}] Target price trade dates: {len(target_dates):,}")

    if not target_dates:
        print(f"[{_ts()}] Nothing to build. Existing rows in range: {_count_rows(engine, start, end):,}")
        print("DONE")
        return

    print(f"[{_ts()}] Build range: {target_dates[0]} ~ {target_dates[-1]}")
    if args.dry_run:
        sample = ", ".join(str(d) for d in target_dates[:10])
        print(f"[{_ts()}] [dry-run] First target dates: {sample}")
        print("DONE")
        return

    failed: list[tuple[date, str]] = []
    t0 = time.time()
    progress_every = max(1, int(args.progress_every or 50))

    for idx, trade_date in enumerate(target_dates, start=1):
        try:
            _call_sp_for_date(
                engine=engine,
                trade_date=trade_date,
                max_retries=max(0, int(args.max_retries)),
                retry_sleep=max(0.0, float(args.retry_sleep)),
            )
        except Exception as exc:
            failed.append((trade_date, str(exc)))
            print(f"[{_ts()}] ERROR failed date={trade_date}: {exc}")
            if args.fail_fast:
                break

        if idx % progress_every == 0 or idx == len(target_dates):
            elapsed = time.time() - t0
            rate = idx / elapsed if elapsed > 0 else 0.0
            remaining = (len(target_dates) - idx) / rate if rate > 0 else 0.0
            print(
                f"[{_ts()}] Progress {idx:,}/{len(target_dates):,} "
                f"({idx * 100.0 / len(target_dates):.1f}%) "
                f"latest={trade_date} elapsed={elapsed/60:.1f}m eta={remaining/60:.1f}m "
                f"failed={len(failed):,}"
            )

    remaining_missing = _find_missing_dates(engine, start, end)
    total_rows = _count_rows(engine, start, end)

    print()
    print("=" * 72)
    print(" Build Summary")
    print("=" * 72)
    print(f" Target dates:       {len(target_dates):,}")
    print(f" Failed calls:       {len(failed):,}")
    print(f" Remaining missing:  {len(remaining_missing):,}")
    print(f" Rows in range:      {total_rows:,}")

    if failed:
        print()
        print("Failed date sample:")
        for d, err in failed[:20]:
            print(f"  - {d}: {err[:300]}")

    if remaining_missing:
        print()
        print("Remaining missing date sample:")
        for d in remaining_missing[:30]:
            print(f"  - {d}")

    if failed or remaining_missing:
        print("FAILED")
        sys.exit(1)

    print("DONE")


if __name__ == "__main__":
    main()
