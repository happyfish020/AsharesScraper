from __future__ import annotations

import argparse
import calendar
import subprocess
import sys
from pathlib import Path
from datetime import date, datetime


def _parse_ym(s: str) -> date:
    s = (s or "").strip()
    return datetime.strptime(s, "%Y-%m").date().replace(day=1)


def _next_month(d: date) -> date:
    if d.month == 12:
        return date(d.year + 1, 1, 1)
    return date(d.year, d.month + 1, 1)


def _month_end(d: date) -> date:
    return date(d.year, d.month, calendar.monthrange(d.year, d.month)[1])


def main() -> int:
    p = argparse.ArgumentParser(
        description="Run rotation rebuild month-by-month (sequential commands)."
    )
    p.add_argument("--start-ym", default="2000-01", help="inclusive YYYY-MM")
    p.add_argument("--end-ym", default="2026-02", help="inclusive YYYY-MM")
    p.add_argument("--top-pct", type=float, default=0.30)
    p.add_argument("--breadth-min", type=float, default=0.60)
    p.add_argument(
        "--rank-signal-mode",
        default="hybrid_base",
        choices=["bulk_sql", "sp_daily", "hybrid_temp", "hybrid_base"],
    )
    p.add_argument("--months-per-chunk", type=int, default=1)
    p.add_argument("--clear-first", type=int, default=1)
    p.add_argument("--retries", type=int, default=8)
    p.add_argument("--retry-sleep-sec", type=float, default=3.0)
    p.add_argument("--continue-on-error", action="store_true")
    p.add_argument("--log-file", default="", help="optional log file path")
    args = p.parse_args()

    start = _parse_ym(args.start_ym)
    end = _parse_ym(args.end_ym)
    if start > end:
        raise SystemExit(f"invalid range: {start}>{end}")

    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    if args.log_file:
        log_path = Path(args.log_file)
    else:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_path = logs_dir / f"rotation_monthly_backfill_{ts}.log"
    failed_months = []

    def _log(msg: str) -> None:
        line = f"{datetime.now().isoformat(timespec='seconds')} {msg}"
        print(line)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    _log(
        "START "
        f"start_ym={args.start_ym} end_ym={args.end_ym} "
        f"mode={args.rank_signal_mode} clear_first={args.clear_first}"
    )

    cur = start
    i = 0
    while cur <= end:
        i += 1
        d1 = cur
        d2 = _month_end(cur)
        month_tag = d1.strftime("%Y-%m")
        cmd = [
            sys.executable,
            "-m",
            "app.tools.rebuild_rotation_three_tables_from_map",
            "--start",
            d1.strftime("%Y-%m-%d"),
            "--end",
            d2.strftime("%Y-%m-%d"),
            "--top-pct",
            str(args.top_pct),
            "--breadth-min",
            str(args.breadth_min),
            "--rank-signal-mode",
            args.rank_signal_mode,
            "--months-per-chunk",
            str(args.months_per_chunk),
            "--clear-first",
            str(args.clear_first),
            "--retries",
            str(args.retries),
            "--retry-sleep-sec",
            str(args.retry_sleep_sec),
        ]
        _log(f"[{i}] month={month_tag} cmd={' '.join(cmd)}")
        t0 = datetime.now()
        rc = subprocess.call(cmd)
        sec = (datetime.now() - t0).total_seconds()
        if rc != 0:
            failed_months.append(month_tag)
            _log(f"[{i}] FAILED month={month_tag} rc={rc} elapsed_sec={sec:.1f}")
            if not args.continue_on_error:
                _log(
                    f"STOP_ON_ERROR failed_months={','.join(failed_months)} "
                    f"log_file={log_path}"
                )
                return rc
        else:
            _log(f"[{i}] OK month={month_tag} elapsed_sec={sec:.1f}")
        cur = _next_month(cur)

    if failed_months:
        _log(f"DONE_WITH_FAILURE failed_months={','.join(failed_months)}")
    else:
        _log("DONE_ALL_OK")
    _log(f"log_file={log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
