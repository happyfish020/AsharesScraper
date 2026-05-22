from __future__ import annotations

import argparse
import builtins
from datetime import date, datetime
from pathlib import Path
from typing import Iterable, List

import akshare as ak
import pandas as pd
import tushare as ts
from sqlalchemy import text
from concurrent.futures import ThreadPoolExecutor, as_completed

from app.settings import build_engine, load_sql_for_current_db
from app.utils.progress import ProgressLogger
from app.tools.sync_cn_stock_daily_price_from_tushare import (
    _parse_ymd,
    patch_pandas_fillna_method_compat,
    resolve_tushare_token,
)


def print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    return builtins.print(*args, **kwargs)


UPSERT_COLS = [
    "symbol",
    "trade_date",
    "total_share",
    "float_share",
    "free_share",
    "total_mv",
    "circ_mv",
    "pe",
    "pe_ttm",
    "pb",
    "ps",
    "ps_ttm",
    "dv_ratio",
    "dv_ttm",
    "turnover_rate_f",
    "volume_ratio",
    "source",
]


def ensure_table(engine) -> None:
    ddl = load_sql_for_current_db("docs/DDL/cn_market.cn_stock_daily_basic.sql")
    with engine.begin() as conn:
        conn.execute(text(ddl))


def normalize_daily_basic(raw: pd.DataFrame, source_label: str) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame(columns=UPSERT_COLS)
    out = raw.copy()
    out["symbol"] = out["ts_code"].astype(str).str.split(".").str[0]
    out["trade_date"] = pd.to_datetime(out["trade_date"], format="%Y%m%d", errors="coerce").dt.date
    numeric_cols = [c for c in UPSERT_COLS if c not in {"symbol", "trade_date", "source"}]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
        else:
            out[col] = None
    out["source"] = source_label
    out = out.dropna(subset=["symbol", "trade_date"])
    return out[UPSERT_COLS].copy()


def iter_trade_dates(engine, start_date: date, end_date: date, calendar_source: str, descending: bool = False) -> List[date]:
    if calendar_source == "board-map":
        sql = """
            SELECT DISTINCT trade_date
            FROM cn_board_member_map_d
            WHERE sector_type='INDUSTRY'
              AND trade_date >= :start_date
              AND trade_date <= :end_date
            ORDER BY trade_date
        """
    elif calendar_source == "price":
        sql = """
            SELECT DISTINCT trade_date
            FROM cn_stock_daily_price
            WHERE trade_date >= :start_date
              AND trade_date <= :end_date
            ORDER BY trade_date
        """
    else:
        raise ValueError(f"unsupported calendar_source: {calendar_source}")
    with engine.connect() as conn:
        rows = conn.execute(text(sql), {"start_date": start_date, "end_date": end_date}).fetchall()
    dates: List[date] = []
    for row in rows:
        dt = pd.to_datetime(row[0], errors="coerce")
        if pd.notna(dt):
            dates.append(dt.date())
    if descending:
        dates.reverse()
    return dates


def chunked(records: List[dict], chunk_size: int) -> Iterable[List[dict]]:
    for i in range(0, len(records), chunk_size):
        yield records[i : i + chunk_size]


def chunked_dates(trade_dates: List[date], batch_size: int) -> Iterable[List[date]]:
    if batch_size <= 0:
        yield trade_dates
        return
    for i in range(0, len(trade_dates), batch_size):
        yield trade_dates[i : i + batch_size]


def upsert_dataframe(engine, df: pd.DataFrame, chunk_size: int = 4000) -> int:
    if df is None or df.empty:
        return 0
    work = df.copy()
    work = work.astype(object)
    work = work.where(pd.notna(work), None)
    insert_sql = """
        INSERT INTO cn_stock_daily_basic (
            symbol, trade_date, total_share, float_share, free_share,
            total_mv, circ_mv, pe, pe_ttm, pb, ps, ps_ttm,
            dv_ratio, dv_ttm, turnover_rate_f, volume_ratio, source
        ) VALUES (
            :symbol, :trade_date, :total_share, :float_share, :free_share,
            :total_mv, :circ_mv, :pe, :pe_ttm, :pb, :ps, :ps_ttm,
            :dv_ratio, :dv_ttm, :turnover_rate_f, :volume_ratio, :source
        )
        ON DUPLICATE KEY UPDATE
            total_share = VALUES(total_share),
            float_share = VALUES(float_share),
            free_share = VALUES(free_share),
            total_mv = VALUES(total_mv),
            circ_mv = VALUES(circ_mv),
            pe = VALUES(pe),
            pe_ttm = VALUES(pe_ttm),
            pb = VALUES(pb),
            ps = VALUES(ps),
            ps_ttm = VALUES(ps_ttm),
            dv_ratio = VALUES(dv_ratio),
            dv_ttm = VALUES(dv_ttm),
            turnover_rate_f = VALUES(turnover_rate_f),
            volume_ratio = VALUES(volume_ratio),
            source = VALUES(source)
    """
    affected = 0
    with engine.begin() as conn:
        for batch in chunked(work[UPSERT_COLS].to_dict(orient="records"), chunk_size):
            ret = conn.execute(text(insert_sql), batch)
            affected += int(ret.rowcount or 0)
    return affected


def fetch_daily_basic(pro, trade_date: date) -> pd.DataFrame:
    return pro.daily_basic(
        trade_date=trade_date.strftime("%Y%m%d"),
        fields=(
            "ts_code,trade_date,total_share,float_share,free_share,"
            "total_mv,circ_mv,pe,pe_ttm,pb,ps,ps_ttm,dv_ratio,dv_ttm,"
            "turnover_rate_f,volume_ratio"
        ),
    )


def _parse_akshare_individual_info(symbol: str, raw: pd.DataFrame, trade_date: date, source_label: str) -> dict | None:
    if raw is None or raw.empty:
        return None
    pairs = {}
    for row in raw.itertuples(index=False):
        item = str(getattr(row, "item", "") or "").strip()
        value = getattr(row, "value", None)
        if item:
            pairs[item] = value

    def _num(name: str):
        return pd.to_numeric(pairs.get(name), errors="coerce")

    rec = {
        "symbol": str(symbol).strip(),
        "trade_date": trade_date,
        "total_share": _num("总股本"),
        "float_share": _num("流通股"),
        "free_share": None,
        "total_mv": _num("总市值"),
        "circ_mv": _num("流通市值"),
        "pe": None,
        "pe_ttm": None,
        "pb": None,
        "ps": None,
        "ps_ttm": None,
        "dv_ratio": None,
        "dv_ttm": None,
        "turnover_rate_f": None,
        "volume_ratio": None,
        "source": source_label,
    }
    if pd.isna(rec["total_mv"]) and pd.isna(rec["circ_mv"]):
        return None
    return rec


def fetch_akshare_spot_universe() -> List[str]:
    spot = ak.stock_zh_a_spot()
    if spot is None or spot.empty:
        return []
    code_col = "代码"
    if code_col not in spot.columns:
        return []
    codes = (
        spot[code_col]
        .astype(str)
        .str.strip()
        .str[-6:]
        .loc[lambda s: s.str.fullmatch(r"\d{6}")]
        .drop_duplicates()
        .tolist()
    )
    return list(codes)


def fetch_daily_basic_akshare_snapshot(
    trade_date: date,
    symbols: List[str] | None = None,
    source_label: str = "akshare_individual_info_em",
    max_workers: int = 12,
    timeout: float = 15.0,
    log=None,
) -> tuple[pd.DataFrame, int]:
    universe = list(symbols or [])
    if not universe:
        universe = fetch_akshare_spot_universe()
    if not universe:
        return pd.DataFrame(columns=UPSERT_COLS), 0

    rows: List[dict] = []
    failures = 0
    progress = ProgressLogger(name="stock_basic.akshare_snapshot", total=len(universe), unit="symbols", log=log, every=25, min_interval_seconds=8.0)

    def _worker(symbol: str):
        raw = ak.stock_individual_info_em(symbol=symbol, timeout=timeout)
        return _parse_akshare_individual_info(symbol=symbol, raw=raw, trade_date=trade_date, source_label=source_label)

    with ThreadPoolExecutor(max_workers=max(1, int(max_workers))) as pool:
        futures = {pool.submit(_worker, sym): sym for sym in universe}
        for fut in as_completed(futures):
            sym = futures[fut]
            try:
                rec = fut.result()
                if rec is not None:
                    rows.append(rec)
                progress.update(current_item=sym, rows=1 if rec is not None else 0)
            except Exception:
                failures += 1
                progress.update(current_item=sym, extra="failed=1")
    progress.finish(extra=f"failures={failures}")

    if not rows:
        return pd.DataFrame(columns=UPSERT_COLS), failures
    out = pd.DataFrame(rows)
    for col in UPSERT_COLS:
        if col not in out.columns:
            out[col] = None
    out = out[UPSERT_COLS].copy()
    out = out.astype(object).where(pd.notna(out), None)
    return out, failures


def load_daily_basic_tushare(
    engine,
    start_date: date,
    end_date: date,
    calendar_source: str,
    source_label: str,
    token: str,
    descending: bool = False,
    batch_size: int = 0,
    log=None,
) -> tuple[int, int, List[date], str]:
    trade_dates = iter_trade_dates(engine, start_date, end_date, calendar_source, descending=descending)
    if not trade_dates:
        return 0, 0, [], "tushare"

    pro = ts.pro_api(token)
    total_rows = 0
    total_affected = 0
    batch_count = max(1, (len(trade_dates) + max(1, batch_size) - 1) // max(1, batch_size)) if batch_size > 0 else 1
    print(
        f"[stock_basic.tushare] start trade_dates={len(trade_dates)} "
        f"range={trade_dates[0]}..{trade_dates[-1]} calendar_source={calendar_source} "
        f"descending={descending} batch_size={batch_size or 'continuous'} batches={batch_count}"
    )
    progress = ProgressLogger(name="stock_basic.tushare", total=len(trade_dates), unit="trade_dates", log=log, every=1, min_interval_seconds=5.0)
    for batch_index, batch_dates in enumerate(chunked_dates(trade_dates, batch_size), start=1):
        if batch_count > 1:
            print(f"[stock_basic.tushare] batch {batch_index}/{batch_count} {batch_dates[0]}..{batch_dates[-1]}")
        for trade_dt in batch_dates:
            progress.note(f"[stock_basic.tushare] fetching trade_date={trade_dt}")
            raw = fetch_daily_basic(pro, trade_dt)
            df = normalize_daily_basic(raw, source_label)
            affected = upsert_dataframe(engine, df)
            total_rows += int(len(df))
            total_affected += affected
            progress.update(current_item=str(trade_dt), rows=int(len(df)), affected=affected)
        if batch_count > 1:
            print(
                f"[stock_basic.tushare] batch_done {batch_index}/{batch_count} "
                f"cumulative_rows={total_rows} cumulative_affected={total_affected}"
            )
    progress.finish()
    return total_rows, total_affected, trade_dates, "tushare"


def load_daily_basic_akshare(
    engine,
    start_date: date,
    end_date: date,
    source_label: str,
    max_workers: int = 12,
    timeout: float = 15.0,
    log=None,
) -> tuple[int, int, List[date], str, int]:
    # Free fallback is snapshot-only: load the latest requested date.
    trade_dt = end_date
    df, failures = fetch_daily_basic_akshare_snapshot(
        trade_date=trade_dt,
        source_label=source_label,
        max_workers=max_workers,
        timeout=timeout,
        log=log,
    )
    affected = upsert_dataframe(engine, df)
    return int(len(df)), int(affected), [trade_dt], "akshare", int(failures)


def apply_view(engine, ddl_path: str) -> None:
    sql = load_sql_for_current_db(ddl_path)
    with engine.begin() as conn:
        conn.execute(text(sql))


def main() -> None:
    parser = argparse.ArgumentParser(description="Load Tushare daily_basic into cn_stock_daily_basic and build leader score views.")
    parser.add_argument("--token", default="", help="Tushare token")
    parser.add_argument("--config", default="", help="Config file path for Tushare token")
    parser.add_argument("--start", default="", help="Start date YYYYMMDD or YYYY-MM-DD")
    parser.add_argument("--end", default="", help="End date YYYYMMDD or YYYY-MM-DD")
    parser.add_argument(
        "--provider",
        choices=["auto", "tushare", "akshare"],
        default="auto",
        help="Data provider: auto tries tushare first and falls back to akshare snapshot",
    )
    parser.add_argument(
        "--calendar-source",
        choices=["board-map", "price"],
        default="price",
        help="Trade-date source used to select backfill dates; board-map now uses all INDUSTRY rows, not BK-only rows",
    )
    parser.add_argument(
        "--date-order",
        choices=["asc", "desc"],
        default="asc",
        help="Trade-date traversal order; use desc for near-to-far historical backfill",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="Optional number of trade dates per batch; 0 means one continuous run",
    )
    parser.add_argument("--source-label", default="tushare_daily_basic", help="Source label written into cn_stock_daily_basic")
    parser.add_argument("--akshare-workers", type=int, default=12, help="Thread workers when using akshare fallback")
    parser.add_argument("--akshare-timeout", type=float, default=15.0, help="Per-symbol timeout seconds for akshare fallback")
    parser.add_argument("--skip-views", action="store_true", help="Skip applying leader score views after load")
    args = parser.parse_args()

    patch_pandas_fillna_method_compat()
    token = ""
    tried_files = []
    if args.provider in {"auto", "tushare"}:
        token, tried_files = resolve_tushare_token(args.token, args.config)
        if args.provider == "tushare" and not token:
            msg = "Tushare token is required. Use --token, env TUSHARE_TOKEN/TS_TOKEN, or --config."
            if tried_files:
                msg += f" Tried files: {', '.join(str(p) for p in tried_files)}"
            raise SystemExit(msg)

    end_date = _parse_ymd(args.end) if str(args.end).strip() else date.today()
    start_date = _parse_ymd(args.start) if str(args.start).strip() else end_date
    if start_date > end_date:
        raise SystemExit(f"invalid date range: {start_date} > {end_date}")

    engine = build_engine()
    ensure_table(engine)
    print(
        f"[stock_basic] launch provider={args.provider} start={start_date} end={end_date} "
        f"calendar_source={args.calendar_source} date_order={args.date_order} batch_size={args.batch_size}"
    )
    total_rows = 0
    total_affected = 0
    used_provider = args.provider
    trade_dates: List[date] = []
    ak_failures = 0

    try:
        if args.provider == "akshare":
            total_rows, total_affected, trade_dates, used_provider, ak_failures = load_daily_basic_akshare(
                engine=engine,
                start_date=start_date,
                end_date=end_date,
                source_label="akshare_individual_info_em",
                max_workers=args.akshare_workers,
                timeout=args.akshare_timeout,
            )
        else:
            if not token:
                raise RuntimeError("tushare token missing")
            total_rows, total_affected, trade_dates, used_provider = load_daily_basic_tushare(
                engine=engine,
                start_date=start_date,
                end_date=end_date,
                calendar_source=args.calendar_source,
                source_label=args.source_label,
                token=token,
                descending=args.date_order == "desc",
                batch_size=max(0, int(args.batch_size)),
            )
    except Exception as e:
        if args.provider != "auto":
            raise
        print(f"provider_fallback triggered=tushare_to_akshare err={e}")
        total_rows, total_affected, trade_dates, used_provider, ak_failures = load_daily_basic_akshare(
            engine=engine,
            start_date=start_date,
            end_date=end_date,
            source_label="akshare_individual_info_em",
            max_workers=args.akshare_workers,
            timeout=args.akshare_timeout,
        )

    if not trade_dates:
        print("no trade dates matched the requested range")
        return
    print(f"daily_basic_provider={used_provider} dates={len(trade_dates)} range={trade_dates[0]}..{trade_dates[-1]}")
    if used_provider == "akshare":
        print(f"akshare_snapshot_only=1 failures={ak_failures}")

    # NOTE: v1/v2 views are no longer applied here — they have been replaced by
    # sp_materialize_leader_score (stored procedure with temp tables).
    # The SP is called from stock_basic_weekly_task.py and daily_materialize_leader_score.py.

    print(
        f"daily_basic_done dates={len(trade_dates)} rows={total_rows} "
        f"affected={total_affected} start={trade_dates[0]} end={trade_dates[-1]}"
    )


if __name__ == "__main__":
    main()
