from __future__ import annotations

import argparse
import builtins
import time
from datetime import date, datetime
from pathlib import Path
from typing import Iterable, List

import pandas as pd
import tushare as ts
from sqlalchemy import text

from app.settings import build_engine, load_sql_for_current_db
from app.tools.sync_cn_stock_daily_price_from_tushare import (
    patch_pandas_fillna_method_compat,
    resolve_tushare_token,
)
from app.utils.progress import ProgressLogger


def print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    return builtins.print(*args, **kwargs)


UPSERT_COLS = [
    "ts_code",
    "trade_date",
    "name",
    "open",
    "high",
    "low",
    "close",
    "change",
    "pct_change",
    "vol",
    "amount",
    "pe",
    "pb",
    "float_mv",
    "source",
]

SW_DAILY_MIN_INTERVAL_SECONDS = 8.5
SW_DAILY_RETRY_SLEEP_SECONDS = 65
_SW_DAILY_LAST_CALL_AT = 0.0


def _parse_ymd(s: str) -> date:
    v = (s or "").strip()
    if len(v) == 10 and v[4] == "-" and v[7] == "-":
        return datetime.strptime(v, "%Y-%m-%d").date()
    return datetime.strptime(v, "%Y%m%d").date()


def _to_ymd(d: date) -> str:
    return d.strftime("%Y%m%d")


def ensure_table(engine) -> None:
    ddl = load_sql_for_current_db("docs/DDL/cn_market.cn_sw_industry_daily.sql")
    with engine.begin() as conn:
        conn.execute(text(ddl))


def chunked(records: List[dict], chunk_size: int) -> Iterable[List[dict]]:
    for i in range(0, len(records), chunk_size):
        yield records[i : i + chunk_size]


def get_sw_l1_codes(engine, src: str) -> List[str]:
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT DISTINCT BOARD_ID
                FROM cn_board_industry_master
                WHERE SOURCE = :src
                ORDER BY BOARD_ID
                """
            ),
            {"src": src},
        ).fetchall()
    return [str(r[0]).strip() for r in rows if str(r[0]).strip()]


def fetch_sw_l1_codes_from_tushare(pro, *, src: str) -> List[str]:
    df = pro.index_classify(src=src, level="L1", fields="index_code,industry_name,level,src")
    if df is None or df.empty:
        return []
    return sorted({str(v).strip() for v in df["index_code"].tolist() if str(v).strip()})


def _respect_sw_daily_spacing() -> None:
    global _SW_DAILY_LAST_CALL_AT
    now = time.monotonic()
    if _SW_DAILY_LAST_CALL_AT > 0:
        elapsed = now - _SW_DAILY_LAST_CALL_AT
        remaining = SW_DAILY_MIN_INTERVAL_SECONDS - elapsed
        if remaining > 0:
            print(f"[SW_DAILY] throttle sleep {remaining:.1f}s before request")
            time.sleep(remaining)
    _SW_DAILY_LAST_CALL_AT = time.monotonic()


def _is_sw_daily_rate_limit(msg: str) -> bool:
    return (
        "棰戠巼瓒呴檺" in msg
        or "姣忓垎閽熸渶澶氳闂" in msg
        or ("sw_daily" in msg and "瓒呴檺" in msg)
    )


def fetch_sw_daily_with_retry(
    pro,
    *,
    ts_code: str,
    start_date: date,
    end_date: date,
    fields: str,
    max_retries: int = 6,
) -> pd.DataFrame:
    for attempt in range(max_retries):
        try:
            _respect_sw_daily_spacing()
            return pro.sw_daily(
                ts_code=ts_code,
                start_date=_to_ymd(start_date),
                end_date=_to_ymd(end_date),
                fields=fields,
            )
        except Exception as exc:
            msg = str(exc)
            if not _is_sw_daily_rate_limit(msg) or attempt + 1 == max_retries:
                raise
            print(
                f"[SW_DAILY] {ts_code} hit rate limit; "
                f"sleep {SW_DAILY_RETRY_SLEEP_SECONDS}s then retry {attempt + 2}/{max_retries}"
            )
            time.sleep(SW_DAILY_RETRY_SLEEP_SECONDS)
    return pd.DataFrame()


def normalize_sw_daily(raw: pd.DataFrame, *, source_label: str) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame(columns=UPSERT_COLS)

    out = raw.copy()
    out["ts_code"] = out["ts_code"].astype(str).str.strip()
    out["trade_date"] = pd.to_datetime(out["trade_date"], format="%Y%m%d", errors="coerce").dt.date
    out = out.rename(columns={"pct_chg": "pct_change"})

    numeric_cols = [
        "open",
        "high",
        "low",
        "close",
        "change",
        "pct_change",
        "vol",
        "amount",
        "pe",
        "pb",
        "float_mv",
    ]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
        else:
            out[col] = None

    if "name" not in out.columns:
        out["name"] = None
    out["source"] = source_label
    out = out.dropna(subset=["ts_code", "trade_date"])
    return out[UPSERT_COLS].copy()


def upsert_dataframe(engine, df: pd.DataFrame, *, chunk_size: int = 2000) -> int:
    if df is None or df.empty:
        return 0

    work = df.copy().astype(object)
    work = work.where(pd.notna(work), None)
    insert_sql = """
        INSERT INTO cn_sw_industry_daily (
            ts_code, trade_date, name, open, high, low, close,
            `change`, pct_change, vol, amount, pe, pb, float_mv, source
        ) VALUES (
            :ts_code, :trade_date, :name, :open, :high, :low, :close,
            :change, :pct_change, :vol, :amount, :pe, :pb, :float_mv, :source
        )
        ON DUPLICATE KEY UPDATE
            name = VALUES(name),
            open = VALUES(open),
            high = VALUES(high),
            low = VALUES(low),
            close = VALUES(close),
            `change` = VALUES(`change`),
            pct_change = VALUES(pct_change),
            vol = VALUES(vol),
            amount = VALUES(amount),
            pe = VALUES(pe),
            pb = VALUES(pb),
            float_mv = VALUES(float_mv),
            source = VALUES(source)
    """
    affected = 0
    with engine.begin() as conn:
        for batch in chunked(work[UPSERT_COLS].to_dict(orient="records"), chunk_size):
            ret = conn.execute(text(insert_sql), batch)
            affected += int(ret.rowcount or 0)
    return affected


def resolve_start_dates(engine, ts_codes: List[str], default_start: date) -> dict[str, date]:
    if not ts_codes:
        return {}
    placeholders = ", ".join(f":p{i}" for i in range(len(ts_codes)))
    params = {f"p{i}": code for i, code in enumerate(ts_codes)}
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                f"""
                SELECT ts_code, MAX(trade_date) AS max_trade_date
                FROM cn_sw_industry_daily
                WHERE ts_code IN ({placeholders})
                GROUP BY ts_code
                """
            ),
            params,
        ).fetchall()
    out = {code: default_start for code in ts_codes}
    for ts_code, max_trade_date in rows:
        if max_trade_date is None:
            continue
        out[str(ts_code)] = pd.to_datetime(max_trade_date).date()
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="Sync Shenwan industry daily from Tushare Pro sw_daily.")
    p.add_argument("--start", default="2000-01-01", help="YYYY-MM-DD or YYYYMMDD")
    p.add_argument("--end", default=date.today().strftime("%Y-%m-%d"), help="YYYY-MM-DD or YYYYMMDD")
    p.add_argument("--src", default="SW2021", help="Tushare Shenwan classification source")
    p.add_argument("--master-source", default="TUSHARE_SW2021_L1", help="master source filter in cn_board_industry_master")
    p.add_argument("--codes", default="", help="comma-separated ts_code list; default reads from cn_board_industry_master")
    p.add_argument("--token", default="", help="Tushare token; default uses project token resolver")
    p.add_argument("--config", default="", help="Optional config path for token resolver")
    p.add_argument("--full", action="store_true", help="full reload by requested range instead of incremental from max(trade_date)")
    args = p.parse_args()

    start = _parse_ymd(args.start)
    end = _parse_ymd(args.end)
    if start > end:
        raise SystemExit("start must be <= end")

    patch_pandas_fillna_method_compat()
    token, tried_files = resolve_tushare_token(args.token, args.config)
    if not token:
        msg = "Tushare token required."
        if tried_files:
            msg += f" tried_files={', '.join(str(p) for p in tried_files)}"
        raise SystemExit(msg)
    pro = ts.pro_api(token)

    engine = build_engine()
    ensure_table(engine)

    if args.codes.strip():
        codes = [s.strip() for s in args.codes.replace(";", ",").split(",") if s.strip()]
    else:
        codes = get_sw_l1_codes(engine, args.master_source)
        if not codes:
            codes = fetch_sw_l1_codes_from_tushare(pro, src=args.src)

    if not codes:
        raise SystemExit("No SW L1 ts_code resolved.")

    start_dates = {code: start for code in codes}
    if not args.full:
        existing = resolve_start_dates(engine, codes, start)
        start_dates = {code: max(start, existing.get(code, start)) for code in codes}

    fields = "ts_code,trade_date,name,open,high,low,close,change,pct_change,vol,amount,pe,pb,float_mv"
    total_rows = 0
    total_affected = 0
    touched = 0
    print(
        f"[SW_DAILY] launch codes={len(codes)} start={start} end={end} src={args.src} "
        f"master_source={args.master_source} full={args.full} throttle={SW_DAILY_MIN_INTERVAL_SECONDS}s"
    )
    progress = ProgressLogger(name="sw_daily", total=len(codes), unit="codes", every=1, min_interval_seconds=5.0)

    for i, code in enumerate(codes, start=1):
        code_start = start_dates[code]
        if not args.full and code_start > start:
            code_start = pd.Timestamp(code_start).date()
        if code_start > end:
            print(f"[SW_DAILY] {i}/{len(codes)} {code} skip already up-to-date")
            progress.update(current_item=code, extra="skip=up_to_date")
            continue
        print(f"[SW_DAILY] {i}/{len(codes)} {code} {code_start}..{end}")
        progress.note(f"[SW_DAILY] fetching ts_code={code} range={code_start}..{end}")
        raw = fetch_sw_daily_with_retry(
            pro,
            ts_code=code,
            start_date=code_start,
            end_date=end,
            fields=fields,
        )
        df = normalize_sw_daily(raw, source_label="tushare_sw_daily")
        if df.empty:
            progress.update(current_item=code, extra="rows=0")
            continue
        rows = len(df)
        affected = upsert_dataframe(engine, df)
        total_rows += rows
        total_affected += affected
        touched += 1
        print(f"[SW_DAILY] {code} rows={rows} affected={affected}")
        progress.update(current_item=code, rows=rows, affected=affected)
    progress.finish(extra=f"touched={touched}")

    with engine.connect() as conn:
        summary = conn.execute(
            text(
                """
                SELECT COUNT(*) AS row_cnt,
                       COUNT(DISTINCT ts_code) AS code_cnt,
                       MIN(trade_date) AS min_trade_date,
                       MAX(trade_date) AS max_trade_date
                FROM cn_sw_industry_daily
                """
            )
        ).one()
    print(
        f"done touched={touched} fetched_rows={total_rows} affected={total_affected} "
        f"table_rows={summary[0]} code_cnt={summary[1]} min_trade_date={summary[2]} max_trade_date={summary[3]}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
