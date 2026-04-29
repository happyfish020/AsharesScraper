from __future__ import annotations

import argparse
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd
from sqlalchemy import text

from app.tools.sync_cn_stock_daily_price_from_tushare import (
    TushareStockDailySync,
    _parse_ymd,
    resolve_tushare_token,
)


ALLOWED_REQUIRED_COLUMNS = [
    "open",
    "close",
    "pre_close",
    "high",
    "low",
    "volume",
    "amount",
    "amplitude",
    "chg_pct",
    "change",
    "turnover_rate",
]


def normalize_symbol(value: object) -> str:
    s = str(value or "").strip()
    if s.isdigit() and len(s) <= 6:
        return s.zfill(6)
    return s


def infer_ts_code(symbol: str) -> str | None:
    s = normalize_symbol(symbol)
    if len(s) != 6 or not s.isdigit():
        return None
    if s.startswith(("000", "001", "002", "003", "300")):
        return f"{s}.SZ"
    if s.startswith(("600", "601", "603", "605", "688", "689", "900")):
        return f"{s}.SH"
    if s.startswith(("430", "831", "832", "833", "834", "835", "836", "837", "838", "839")):
        return f"{s}.BJ"
    return None


def infer_exchange_from_symbol(symbol: str) -> str | None:
    s = normalize_symbol(symbol)
    if len(s) != 6 or not s.isdigit():
        return None
    if s.startswith(("000", "001", "002", "003", "300")):
        return "SZSE"
    if s.startswith(("600", "601", "603", "605", "688", "689", "900")):
        return "SSE"
    if s.startswith(("430", "831", "832", "833", "834", "835", "836", "837", "838", "839")):
        return "BJSE"
    return None


def _normalize_trade_date_col(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    out["trade_date"] = pd.to_datetime(out["trade_date"], format="%Y%m%d", errors="coerce").dt.date
    out = out.dropna(subset=["trade_date"])
    return out


def iter_date_chunks(start: date, end: date, years_per_chunk: int = 5) -> List[tuple[date, date]]:
    chunks: List[tuple[date, date]] = []
    cur = start
    span = max(1, int(years_per_chunk))
    while cur <= end:
        chunk_end_year = min(cur.year + span - 1, end.year)
        chunk_end = date(chunk_end_year, 12, 31)
        if chunk_end > end:
            chunk_end = end
        chunks.append((cur, chunk_end))
        cur = date(chunk_end.year + 1, 1, 1)
    return chunks


def parse_required_columns(raw: str) -> List[str]:
    cols = [str(x).strip().lower() for x in str(raw or "").split(",") if str(x).strip()]
    if not cols:
        raise SystemExit("required columns list cannot be empty")
    invalid = [c for c in cols if c not in ALLOWED_REQUIRED_COLUMNS]
    if invalid:
        raise SystemExit(
            f"unsupported required columns: {', '.join(invalid)}; "
            f"allowed={', '.join(ALLOWED_REQUIRED_COLUMNS)}"
        )
    deduped: List[str] = []
    seen = set()
    for c in cols:
        if c not in seen:
            deduped.append(c)
            seen.add(c)
    return deduped


def build_workload(
    syncer: TushareStockDailySync,
    workload_mode: str,
    target_source: str,
    required_columns: List[str],
    start_date: date | None = None,
    end_date: date | None = None,
    symbol: str = "",
) -> pd.DataFrame:
    sql = """
        SELECT
            p.symbol,
            MIN(p.trade_date) AS gap_start,
            MAX(p.trade_date) AS gap_end,
            COUNT(*) AS target_rows
        FROM cn_stock_daily_price p
    """
    params: Dict[str, object] = {}
    where_parts: List[str] = ["1=1"]

    if workload_mode == "incomplete-only":
        null_predicate = " OR ".join([f"p.`{col}` IS NULL" for col in required_columns])
        where_parts.append(f"({null_predicate})")
    elif workload_mode != "all-existing":
        raise SystemExit(f"unsupported workload mode: {workload_mode}")

    if target_source.strip():
        where_parts.append("p.source = :target_source")
        params["target_source"] = target_source.strip()
    if start_date is not None:
        where_parts.append("p.trade_date >= :start_date")
        params["start_date"] = start_date
    if end_date is not None:
        where_parts.append("p.trade_date <= :end_date")
        params["end_date"] = end_date
    if symbol.strip():
        where_parts.append("p.symbol = :symbol")
        params["symbol"] = normalize_symbol(symbol)
    sql += " WHERE " + " AND ".join(where_parts)
    sql += " GROUP BY p.symbol ORDER BY gap_start, p.symbol"

    with syncer.engine.connect() as conn:
        df = pd.read_sql(text(sql), conn, params=params)
    if df.empty:
        return pd.DataFrame(columns=["symbol", "gap_start", "gap_end", "target_rows"])

    df["symbol"] = df["symbol"].map(normalize_symbol)
    df["gap_start"] = pd.to_datetime(df["gap_start"], errors="coerce").dt.date
    df["gap_end"] = pd.to_datetime(df["gap_end"], errors="coerce").dt.date
    df["target_rows"] = pd.to_numeric(df["target_rows"], errors="coerce").fillna(0).astype(int)
    return df


def build_universe_workload(
    universe: pd.DataFrame,
    start_date: date,
    end_date: date,
    symbol: str = "",
) -> pd.DataFrame:
    if universe is None or universe.empty:
        return pd.DataFrame(columns=["symbol", "gap_start", "gap_end", "target_rows"])
    work = universe[["symbol"]].copy()
    work["symbol"] = work["symbol"].map(normalize_symbol)
    if symbol.strip():
        work = work[work["symbol"] == normalize_symbol(symbol)].copy()
    work = work.drop_duplicates(subset=["symbol"]).sort_values("symbol").reset_index(drop=True)
    if work.empty:
        return pd.DataFrame(columns=["symbol", "gap_start", "gap_end", "target_rows"])
    work["gap_start"] = start_date
    work["gap_end"] = end_date
    work["target_rows"] = 0
    return work


def list_target_dates(
    syncer: TushareStockDailySync,
    symbol: str,
    target_source: str,
    required_columns: List[str],
    start_date: date,
    end_date: date,
) -> set[date]:
    null_predicate = " OR ".join([f"`{col}` IS NULL" for col in required_columns])
    sql = f"""
        SELECT trade_date
        FROM cn_stock_daily_price
        WHERE symbol = :symbol
          AND trade_date >= :start_date
          AND trade_date <= :end_date
          AND ({null_predicate})
    """
    params: Dict[str, object] = {
        "symbol": normalize_symbol(symbol),
        "start_date": start_date,
        "end_date": end_date,
    }
    if target_source.strip():
        sql += " AND source = :target_source"
        params["target_source"] = target_source.strip()
    sql += " ORDER BY trade_date"
    with syncer.engine.connect() as conn:
        rows = conn.execute(
            text(sql),
            params,
        ).fetchall()
    out: set[date] = set()
    for row in rows:
        dt = pd.to_datetime(row[0], errors="coerce")
        if pd.notna(dt):
            out.add(dt.date())
    return out


def prepare_upsert_frame(
    syncer: TushareStockDailySync,
    raw: pd.DataFrame,
    name: str | None,
    source_label: str,
    window_start: date,
) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame(columns=syncer._upsert_cols)
    df = raw.copy()
    if "symbol" in df.columns:
        df["symbol"] = df["symbol"].map(normalize_symbol)
    df["source"] = source_label
    df["window_start"] = window_start
    df["name"] = name
    for c in syncer._upsert_cols:
        if c not in df.columns:
            df[c] = None
    return df[syncer._upsert_cols].copy()


def fetch_daily_unadjusted(syncer: TushareStockDailySync, ts_code: str, start: date, end: date) -> pd.DataFrame:
    pro = syncer._get_thread_pro()
    last_err: Exception | None = None
    for i in range(max(1, int(syncer.fetch_retries))):
        try:
            with syncer._api_sem:
                df = pro.daily(
                    ts_code=ts_code,
                    start_date=start.strftime("%Y%m%d"),
                    end_date=end.strftime("%Y%m%d"),
                )
            break
        except Exception as e:
            last_err = e
            syncer._tls.pro = None
            import time

            time.sleep(min(8.0, 0.8 * (2**i)))
    else:
        print(f"[WARN] daily fallback fetch failed ts_code={ts_code} start={start} end={end} err={last_err}")
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()
    out = _normalize_trade_date_col(df)
    if out.empty:
        return pd.DataFrame()
    out["symbol"] = out["ts_code"].astype(str).str.split(".").str[0].map(normalize_symbol)
    out["exchange"] = out["ts_code"].astype(str).str.split(".").str[1].str.upper().map(
        {"SH": "SSE", "SZ": "SZSE", "BJ": "BJSE"}
    )
    rename_map = {"pct_chg": "chg_pct", "vol": "volume"}
    out = out.rename(columns=rename_map)
    for c in ("open", "close", "pre_close", "high", "low", "volume", "amount", "change", "chg_pct"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    if "amount" in out.columns:
        out["amount"] = pd.to_numeric(out["amount"], errors="coerce")
    out["turnover_rate"] = None
    out["amplitude"] = ((out["high"] - out["low"]) / out["pre_close"] * 100.0).where(out["pre_close"] > 0)
    return out


def fetch_qfq_daily_chunked(
    syncer: TushareStockDailySync,
    ts_code: str,
    start: date,
    end: date,
    years_per_chunk: int = 5,
) -> pd.DataFrame:
    parts: List[pd.DataFrame] = []
    for chunk_start, chunk_end in iter_date_chunks(start, end, years_per_chunk=years_per_chunk):
        df = syncer.fetch_qfq_daily(ts_code=ts_code, start=chunk_start, end=chunk_end)
        if df is not None and not df.empty:
            parts.append(df)
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    out = out.drop_duplicates(subset=["trade_date"], keep="last").sort_values("trade_date").reset_index(drop=True)
    return out


def fetch_daily_unadjusted_chunked(
    syncer: TushareStockDailySync,
    ts_code: str,
    start: date,
    end: date,
    years_per_chunk: int = 5,
) -> pd.DataFrame:
    parts: List[pd.DataFrame] = []
    for chunk_start, chunk_end in iter_date_chunks(start, end, years_per_chunk=years_per_chunk):
        df = fetch_daily_unadjusted(syncer=syncer, ts_code=ts_code, start=chunk_start, end=chunk_end)
        if df is not None and not df.empty:
            parts.append(df)
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    out = out.drop_duplicates(subset=["trade_date"], keep="last").sort_values("trade_date").reset_index(drop=True)
    return out


def merge_qfq_with_daily_fallback(
    syncer: TushareStockDailySync,
    qfq_df: pd.DataFrame,
    ts_code: str,
    start: date,
    end: date,
) -> tuple[pd.DataFrame, int]:
    if qfq_df is None or qfq_df.empty:
        return qfq_df if qfq_df is not None else pd.DataFrame(), 0

    work = qfq_df.copy()
    missing_mask = (
        work["open"].isna()
        | work["close"].isna()
        | work["pre_close"].isna()
        | work["high"].isna()
        | work["low"].isna()
    )
    if not bool(missing_mask.any()):
        return work, 0

    daily_df = fetch_daily_unadjusted_chunked(syncer=syncer, ts_code=ts_code, start=start, end=end)
    if daily_df.empty:
        return work, 0

    daily_cols = ["trade_date", "open", "close", "pre_close", "high", "low", "change", "chg_pct", "amplitude"]
    daily_df = daily_df[[c for c in daily_cols if c in daily_df.columns]].copy()
    merged = work.merge(daily_df, on="trade_date", how="left", suffixes=("", "_daily"))
    fallback_hits = 0
    for col in ("open", "close", "pre_close", "high", "low", "change", "chg_pct", "amplitude"):
        daily_col = f"{col}_daily"
        if daily_col not in merged.columns:
            continue
        fill_mask = merged[col].isna() & merged[daily_col].notna()
        fallback_hits += int(fill_mask.sum()) if col == "open" else 0
        merged.loc[fill_mask, col] = merged.loc[fill_mask, daily_col]
        merged = merged.drop(columns=[daily_col])
    return merged, fallback_hits


def delete_unmatched_placeholder_rows(
    syncer: TushareStockDailySync,
    symbol: str,
    matched_trade_dates: set[date],
    placeholder_source: str,
) -> int:
    if not matched_trade_dates:
        return 0
    cleanup_start = min(matched_trade_dates)
    cleanup_end = max(matched_trade_dates)
    sql = """
        DELETE FROM cn_stock_daily_price
        WHERE symbol = :symbol
          AND trade_date >= :cleanup_start
          AND trade_date <= :cleanup_end
          AND source = :placeholder_source
    """
    params: Dict[str, object] = {
        "symbol": symbol,
        "cleanup_start": cleanup_start,
        "cleanup_end": cleanup_end,
        "placeholder_source": placeholder_source,
    }
    placeholders = []
    for i, dt in enumerate(sorted(matched_trade_dates)):
        key = f"dt_{i}"
        placeholders.append(f":{key}")
        params[key] = dt
    sql += f" AND trade_date NOT IN ({', '.join(placeholders)})"
    with syncer.engine.begin() as conn:
        ret = conn.execute(text(sql), params)
        return int(ret.rowcount or 0)


def delete_symbol_rows(syncer: TushareStockDailySync, symbol: str, start_date: date, end_date: date) -> int:
    with syncer.engine.begin() as conn:
        ret = conn.execute(
            text(
                """
                DELETE FROM cn_stock_daily_price
                WHERE symbol = :symbol
                  AND trade_date >= :start_date
                  AND trade_date <= :end_date
                """
            ),
            {
                "symbol": normalize_symbol(symbol),
                "start_date": start_date,
                "end_date": end_date,
            },
        )
        return int(ret.rowcount or 0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Backfill cn_stock_daily_price from Tushare by symbol windows. "
            "Default mode rewrites the full existing date window for every matching symbol."
        )
    )
    parser.add_argument("--token", default="", help="Tushare token")
    parser.add_argument("--config", default="", help="Config file path (.env/.ini/.cfg/.json) to read token")
    parser.add_argument("--sleep-ms", type=int, default=100, help="Sleep milliseconds between symbols")
    parser.add_argument("--fetch-retries", type=int, default=5, help="Retry times for per-symbol Tushare fetch")
    parser.add_argument("--api-concurrency", type=int, default=1, help="Global concurrent Tushare API calls")
    parser.add_argument("--source-label", default="tushare_qfq", help="source value written back to cn_stock_daily_price")
    parser.add_argument(
        "--workload-mode",
        choices=["all-existing", "incomplete-only", "all-universe"],
        default="all-existing",
        help="all-existing rewrites each symbol's DB date window; incomplete-only scans rows with required columns missing; all-universe ignores DB windows and fetches the requested full date range from Tushare",
    )
    parser.add_argument("--target-source", default="", help="Optional source filter when selecting rows to build symbol windows")
    parser.add_argument(
        "--required-columns",
        default="open,close,pre_close,high,low,volume,amount,chg_pct",
        help="Comma-separated columns considered required; NULL in any of them marks a row as incomplete",
    )
    parser.add_argument("--start", default="", help="Optional lower bound date YYYYMMDD or YYYY-MM-DD")
    parser.add_argument("--end", default="", help="Optional upper bound date YYYYMMDD or YYYY-MM-DD")
    parser.add_argument("--symbol", default="", help="Only process one symbol")
    parser.add_argument("--symbol-limit", type=int, default=0, help="For trial run: only process first N symbols")
    parser.add_argument("--batch-size", type=int, default=30, help="Checkpoint flush interval by symbols")
    parser.add_argument(
        "--overwrite-mode",
        choices=["full-window", "target-dates"],
        default="full-window",
        help="full-window overwrites the whole per-symbol gap window; target-dates only overwrites rows currently matching the incomplete filter",
    )
    parser.add_argument(
        "--delete-unmatched-placeholders",
        action="store_true",
        help="Delete placeholder rows in the same window when Tushare did not return those trade dates",
    )
    parser.add_argument(
        "--placeholder-source",
        default="legacy_hist",
        help="Source value treated as placeholder when using --delete-unmatched-placeholders",
    )
    parser.add_argument(
        "--replace-existing-symbol",
        action="store_true",
        help="Delete existing cn_stock_daily_price rows for each symbol/date window before inserting freshly fetched Tushare rows",
    )
    parser.add_argument(
        "--state-file",
        default="state/tushare_backfill_all_existing_state.json",
        help="Resume state JSON file",
    )
    parser.add_argument(
        "--csv-dir",
        default="data/tushare_backfill_all_existing",
        help="Directory to save per-symbol CSV files",
    )
    parser.add_argument("--reuse-csv", dest="reuse_csv", action="store_true", help="Reuse existing per-symbol CSV (default)")
    parser.add_argument("--no-reuse-csv", dest="reuse_csv", action="store_false", help="Ignore existing CSV and fetch again")
    parser.add_argument(
        "--fresh-run",
        action="store_true",
        help="Ignore checkpoint state and refetch per-symbol CSV from Tushare for a clean rerun",
    )
    parser.set_defaults(reuse_csv=True)
    args = parser.parse_args()

    start_date = _parse_ymd(args.start) if str(args.start).strip() else None
    end_date = _parse_ymd(args.end) if str(args.end).strip() else None
    if start_date is not None and end_date is not None and start_date > end_date:
        raise SystemExit(f"invalid date range: {start_date} > {end_date}")

    required_columns = parse_required_columns(args.required_columns)
    token, tried_files = resolve_tushare_token(args.token, args.config)
    if not token:
        msg = "Tushare token is required. Use --token, env TUSHARE_TOKEN/TS_TOKEN, or --config."
        if tried_files:
            msg += f" Tried files: {', '.join(str(p) for p in tried_files)}"
        raise SystemExit(msg)

    syncer = TushareStockDailySync(
        token=token,
        sleep_ms=max(0, int(args.sleep_ms)),
        source_label=(args.source_label or "tushare_qfq").strip(),
        fetch_retries=max(1, int(args.fetch_retries)),
        api_concurrency=max(1, int(args.api_concurrency)),
    )

    universe = syncer.list_a_share_universe()
    if universe.empty:
        raise SystemExit("No A-share symbols from Tushare stock_basic.")
    universe = universe[["symbol", "ts_code", "name", "exchange"]].copy()
    universe["symbol"] = universe["symbol"].map(normalize_symbol)
    if args.workload_mode == "all-universe":
        workload = build_universe_workload(
            universe=universe,
            start_date=start_date or date(2000, 1, 4),
            end_date=end_date or datetime.now().date(),
            symbol=(args.symbol or "").strip(),
        )
    else:
        workload = build_workload(
            syncer=syncer,
            workload_mode=(args.workload_mode or "all-existing").strip(),
            target_source=(args.target_source or "").strip(),
            required_columns=required_columns,
            start_date=start_date,
            end_date=end_date,
            symbol=(args.symbol or "").strip(),
        )
    if workload.empty:
        print("no rows matched the backfill criteria")
        return
    work = workload.merge(universe, on="symbol", how="left")
    requested_symbol = normalize_symbol((args.symbol or "").strip())
    if requested_symbol:
        missing_requested = work["symbol"].astype(str).eq(requested_symbol) & work["ts_code"].isna()
        if missing_requested.any():
            inferred_ts_code = infer_ts_code(requested_symbol)
            inferred_exchange = infer_exchange_from_symbol(requested_symbol)
            if inferred_ts_code:
                work.loc[missing_requested, "ts_code"] = inferred_ts_code
                if "exchange" in work.columns:
                    work.loc[missing_requested, "exchange"] = inferred_exchange
                if "name" in work.columns:
                    work.loc[missing_requested, "name"] = work.loc[missing_requested, "name"].fillna("")
    missing_meta = work["ts_code"].isna()
    missing_meta_symbols: List[str] = []
    if missing_meta.any():
        missing_meta_symbols = work.loc[missing_meta, "symbol"].astype(str).drop_duplicates().tolist()
        preview = ", ".join(missing_meta_symbols[:20])
        suffix = "" if len(missing_meta_symbols) <= 20 else f" ... total={len(missing_meta_symbols)}"
        print(f"[WARN] Tushare metadata missing for symbols: {preview}{suffix}")
    work = work.loc[~missing_meta].copy()
    work = work.sort_values(["gap_start", "symbol"]).reset_index(drop=True)
    if int(args.symbol_limit) > 0:
        work = work.head(int(args.symbol_limit)).copy()

    state_file = Path(args.state_file)
    csv_dir = Path(args.csv_dir)
    csv_dir.mkdir(parents=True, exist_ok=True)
    if args.fresh_run:
        completed_symbols = set()
        failed_symbols = {}
    else:
        state = syncer._load_state(state_file)
        completed_symbols = set(str(x) for x in state.get("completed_symbols", []))
        failed_symbols = dict(state.get("failed_symbols", {}))
    for sym in missing_meta_symbols:
        failed_symbols.setdefault(sym, "Tushare metadata missing")

    total = len(work)
    done_this_run = 0
    fetched_rows = 0
    matched_rows = 0
    replaced_rows = 0
    db_rowcount = 0
    deleted_rows = 0
    fallback_filled_rows = 0
    processed_since_flush = 0

    print(
        f"backfill_candidates={total} workload_mode={args.workload_mode} target_source={args.target_source or '<all>'} "
        f"required_columns={','.join(required_columns)} source_label={syncer.source_label} "
        f"overwrite_mode={args.overwrite_mode} missing_meta={len(missing_meta_symbols)} "
        f"delete_unmatched_placeholders={bool(args.delete_unmatched_placeholders)} "
        f"replace_existing_symbol={bool(args.replace_existing_symbol)}"
    )

    for i, rec in enumerate(work.itertuples(index=False), start=1):
        symbol = str(rec.symbol)
        symbol = normalize_symbol(symbol)
        ts_code = str(rec.ts_code)
        name = str(rec.name) if rec.name is not None else None
        gap_start = rec.gap_start
        gap_end = rec.gap_end
        target_rows = int(rec.target_rows)

        if symbol in completed_symbols:
            if i % 200 == 0 or i == total:
                print(f"[resume] progress {i}/{total} skip_completed={len(completed_symbols)}")
            continue

        csv_path = csv_dir / f"{symbol}.csv"
        try:
            use_existing = (not args.fresh_run) and bool(args.reuse_csv) and csv_path.exists()
            if use_existing:
                raw = pd.read_csv(csv_path, dtype={"symbol": "string", "ts_code": "string"})
                if "trade_date" in raw.columns:
                    raw["trade_date"] = pd.to_datetime(raw["trade_date"], errors="coerce").dt.date
                if "symbol" in raw.columns:
                    raw["symbol"] = raw["symbol"].map(normalize_symbol)
            else:
                raw = fetch_qfq_daily_chunked(syncer=syncer, ts_code=ts_code, start=gap_start, end=gap_end)
                raw, fallback_hits = merge_qfq_with_daily_fallback(
                    syncer=syncer,
                    qfq_df=raw,
                    ts_code=ts_code,
                    start=gap_start,
                    end=gap_end,
                )
                raw.to_csv(csv_path, index=False, encoding="utf-8-sig")
            if use_existing:
                fallback_hits = 0

            fetched = 0 if raw is None else int(len(raw))
            df = prepare_upsert_frame(
                syncer=syncer,
                raw=raw,
                name=name,
                source_label=syncer.source_label,
                window_start=gap_start,
            )
            if not df.empty and args.overwrite_mode == "target-dates":
                target_dates = list_target_dates(
                    syncer=syncer,
                    symbol=symbol,
                    target_source=(args.target_source or "").strip(),
                    required_columns=required_columns,
                    start_date=gap_start,
                    end_date=gap_end,
                )
                df = df[df["trade_date"].isin(target_dates)].copy()
            replaced = 0
            if args.replace_existing_symbol:
                replaced = delete_symbol_rows(
                    syncer=syncer,
                    symbol=symbol,
                    start_date=gap_start,
                    end_date=gap_end,
                )
            affected = syncer._upsert_dataframe(df)
            matched_trade_dates = set(pd.to_datetime(df["trade_date"], errors="coerce").dt.date.dropna().tolist()) if not df.empty else set()
            deleted = 0
            if args.delete_unmatched_placeholders:
                deleted = delete_unmatched_placeholder_rows(
                    syncer=syncer,
                    symbol=symbol,
                    matched_trade_dates=matched_trade_dates,
                    placeholder_source=(args.placeholder_source or "legacy_hist").strip(),
                )

            completed_symbols.add(symbol)
            failed_symbols.pop(symbol, None)
            done_this_run += 1
            fetched_rows += fetched
            matched_rows += int(len(df))
            replaced_rows += int(replaced)
            db_rowcount += int(affected)
            deleted_rows += int(deleted)
            fallback_filled_rows += int(fallback_hits)
            processed_since_flush += 1
            print(
                f"[{i}/{total}] {symbol} ok gap={gap_start}..{gap_end} "
                f"target_rows={target_rows} fetched={fetched} matched={len(df)} "
                f"replaced={replaced} db_rowcount={affected} deleted={deleted} "
                f"fallback_filled={fallback_hits} reuse_csv={use_existing}"
            )
        except Exception as e:
            failed_symbols[symbol] = str(e)
            processed_since_flush += 1
            print(f"[{i}/{total}] {symbol} failed err={e}")

        if syncer.sleep_ms > 0:
            import time

            time.sleep(syncer.sleep_ms / 1000.0)

        if processed_since_flush >= max(1, int(args.batch_size)):
            syncer._save_state(state_file, completed_symbols, failed_symbols)
            print(
                f"[checkpoint] batch={processed_since_flush} completed={len(completed_symbols)} "
                f"failed={len(failed_symbols)} state={state_file}"
            )
            processed_since_flush = 0

    syncer._save_state(state_file, completed_symbols, failed_symbols)
    print(
        f"backfill_done total_symbols={total} done_this_run={done_this_run} "
        f"fetched_rows={fetched_rows} matched_rows={matched_rows} replaced_rows={replaced_rows} "
        f"db_rowcount={db_rowcount} deleted_rows={deleted_rows} "
        f"fallback_filled_rows={fallback_filled_rows} "
        f"completed_total={len(completed_symbols)} failed_total={len(failed_symbols)} "
        f"state_file={state_file} csv_dir={csv_dir}"
    )


if __name__ == "__main__":
    main()
