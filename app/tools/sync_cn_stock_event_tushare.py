from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timedelta
import time
from pathlib import Path
from typing import Iterable, List, Dict, Tuple

import pandas as pd
import tushare as ts
from sqlalchemy import text

from app.settings import build_engine
from app.utils.progress import ProgressLogger
from app.tools.sync_cn_stock_daily_price_from_tushare import (
    _parse_ymd,
    patch_pandas_fillna_method_compat,
    resolve_tushare_token,
)


EVENT_TABLES = {
    "forecast": "cn_event_earnings_forecast",
    "express": "cn_event_earnings_express",
    "fina_indicator": "cn_event_fina_indicator",
    "disclosure_date": "cn_event_disclosure_date",
    "dividend": "cn_event_dividend",
    "anns_d": "cn_event_announcement_meta",
    "signal_daily": "cn_event_signal_daily",
}


def _symbol_to_ts_code(symbol: str) -> str:
    code = str(symbol).strip()[-6:]
    if code.startswith(("6", "9")):
        return f"{code}.SH"
    if code.startswith(("0", "3")):
        return f"{code}.SZ"
    if code.startswith("8"):
        return f"{code}.BJ"
    return code


def _get_symbol_universe(engine) -> List[str]:
    with engine.connect() as conn:
        rows = conn.execute(text("SELECT DISTINCT symbol FROM cn_stock_daily_price WHERE symbol IS NOT NULL AND symbol <> '' ORDER BY symbol")).fetchall()
    return [str(row[0]).strip() for row in rows if str(row[0]).strip()]


def apply_ddl(engine, ddl_path: str) -> None:
    sql = Path(ddl_path).read_text(encoding="utf-8")
    with engine.begin() as conn:
        statements = [part.strip() for part in sql.split(";") if part.strip()]
        for stmt in statements:
            conn.execute(text(stmt))


def ensure_tables(engine) -> None:
    ddl_files = [
        "docs/DDL/cn_market.cn_event_earnings_forecast.sql",
        "docs/DDL/cn_market.cn_event_earnings_express.sql",
        "docs/DDL/cn_market.cn_event_fina_indicator.sql",
        "docs/DDL/cn_market.cn_event_disclosure_date.sql",
        "docs/DDL/cn_market.cn_event_dividend.sql",
        "docs/DDL/cn_market.cn_event_announcement_meta.sql",
        "docs/DDL/cn_market.cn_event_signal_daily.sql",
    ]
    for ddl in ddl_files:
        apply_ddl(engine, ddl)
    with engine.begin() as conn:
        for stmt in [
            "ALTER TABLE cn_event_fina_indicator ADD COLUMN netprofit_yoy DECIMAL(18,6) NULL AFTER debt_to_eqt",
            "ALTER TABLE cn_event_fina_indicator ADD COLUMN q_profit_yoy DECIMAL(18,6) NULL AFTER tr_yoy",
        ]:
            try:
                conn.execute(text(stmt))
            except Exception:
                pass
        try:
            conn.execute(text("ALTER TABLE cn_event_dividend MODIFY record_date DATE NULL"))
        except Exception:
            pass
        try:
            conn.execute(text("ALTER TABLE cn_event_dividend MODIFY ex_date DATE NULL"))
        except Exception:
            pass
        try:
            conn.execute(text("ALTER TABLE cn_event_dividend DROP PRIMARY KEY"))
        except Exception:
            pass
        try:
            conn.execute(
                text(
                    "ALTER TABLE cn_event_dividend ADD PRIMARY KEY (symbol, ann_date, end_date, div_proc)"
                )
            )
        except Exception:
            pass


def _normalize_symbol(ts_code: str) -> str:
    code = str(ts_code or "").strip()
    if not code:
        return ""
    return code.split(".")[0]


def _coerce_date(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    dt = pd.to_datetime(v, errors="coerce")
    if pd.isna(dt):
        return None
    return dt.date()


def _frame_row_payloads(raw: pd.DataFrame) -> pd.Series:
    if raw is None or raw.empty:
        return pd.Series(dtype="object")
    work = raw.copy().astype(object)
    work = work.where(pd.notna(work), None)
    payloads = []
    for record in work.to_dict(orient="records"):
        payloads.append(json.dumps(record, ensure_ascii=False, default=str))
    return pd.Series(payloads, index=raw.index, dtype="object")


def chunked(records: List[dict], chunk_size: int) -> Iterable[List[dict]]:
    for i in range(0, len(records), chunk_size):
        yield records[i : i + chunk_size]


def _iter_month_ranges(start_date: date, end_date: date) -> Iterable[Tuple[date, date]]:
    cursor = date(start_date.year, start_date.month, 1)
    while cursor <= end_date:
        next_month = date(cursor.year + (1 if cursor.month == 12 else 0), 1 if cursor.month == 12 else cursor.month + 1, 1)
        month_end = next_month - timedelta(days=1)
        yield (max(start_date, cursor), min(end_date, month_end))
        cursor = next_month


def _fetch_by_month(pro, api_name: str, fields: str, start_date: date, end_date: date, log=None) -> pd.DataFrame:
    frames = []
    api = getattr(pro, api_name)
    month_ranges = list(_iter_month_ranges(start_date, end_date))
    progress = ProgressLogger(name=f"event.{api_name}", total=len(month_ranges), unit="months", log=log, every=3, min_interval_seconds=15.0)
    for s, e in month_ranges:
        raw = api(
            start_date=s.strftime("%Y%m%d"),
            end_date=e.strftime("%Y%m%d"),
            fields=fields,
        )
        if raw is not None and not raw.empty:
            frames.append(raw)
            progress.update(current_item=f"{s}..{e}", rows=int(len(raw)))
        else:
            progress.update(current_item=f"{s}..{e}")
    progress.finish()
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _fetch_by_ann_date(pro, api_name: str, fields: str, start_date: date, end_date: date, log=None) -> pd.DataFrame:
    frames = []
    api = getattr(pro, api_name)
    cur = start_date
    total_days = (end_date - start_date).days + 1
    progress = ProgressLogger(name=f"event.{api_name}", total=total_days, unit="days", log=log, every=10, min_interval_seconds=15.0)
    while cur <= end_date:
        for attempt in range(1, 4):
            try:
                raw = api(ann_date=cur.strftime("%Y%m%d"), fields=fields)
                if raw is not None and not raw.empty:
                    frames.append(raw)
                    progress.update(current_item=str(cur), rows=int(len(raw)))
                else:
                    progress.update(current_item=str(cur))
                break
            except Exception as e:
                if attempt == 3:
                    print(f"{api_name} ann_date={cur} failed: {e}")
                    progress.update(current_item=str(cur), extra=f"failed={e}")
                else:
                    time.sleep(1.5 * attempt)
        cur += timedelta(days=1)
    progress.finish()
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)




def _fetch_dividend_by_ann_date(pro, fields: str, start_date: date, end_date: date, log=None) -> pd.DataFrame:
    frames = []
    cur = start_date
    total_days = (end_date - start_date).days + 1
    progress = ProgressLogger(name="event.dividend", total=total_days, unit="days", log=log, every=10, min_interval_seconds=15.0)
    while cur <= end_date:
        ymd = cur.strftime("%Y%m%d")
        for attempt in range(1, 4):
            try:
                raw = pro.dividend(ann_date=ymd, fields=fields)
                if raw is not None and not raw.empty:
                    frames.append(raw)
                    progress.update(current_item=ymd, rows=int(len(raw)))
                else:
                    progress.update(current_item=ymd)
                break
            except Exception as e:
                msg = str(e)
                if "200次" in msg or "每分钟" in msg or "200娆" in msg or "姣忓垎閽" in msg:
                    time.sleep(60)
                    continue
                if attempt == 3:
                    print(f"dividend ann_date={ymd} failed: {e}")
                    progress.update(current_item=ymd, extra=f"failed={e}")
                else:
                    time.sleep(1.5 * attempt)
        cur += timedelta(days=1)
    progress.finish()
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)

def _fetch_symbol_history(
    pro,
    api_name: str,
    fields: str,
    symbols: List[str],
    start_date: date,
    end_date: date,
    *,
    sleep_seconds: float = 0.2,
    log=None,
) -> pd.DataFrame:
    frames = []
    api = getattr(pro, api_name)
    progress = ProgressLogger(name=f"event.{api_name}", total=len(symbols), unit="symbols", log=log, every=50, min_interval_seconds=20.0)
    for sym in symbols:
        raw = None
        for attempt in range(1, 4):
            try:
                raw = api(
                    ts_code=_symbol_to_ts_code(sym),
                    start_date=start_date.strftime("%Y%m%d"),
                    end_date=end_date.strftime("%Y%m%d"),
                    fields=fields,
                )
                break
            except Exception as e:
                msg = str(e)
                if "200娆" in msg or "姣忓垎閽" in msg:
                    time.sleep(60)
                    continue
                if attempt == 3:
                    print(f"{api_name} {sym} failed: {e}")
                else:
                    time.sleep(1.5 * attempt)
        if raw is not None and not raw.empty:
            frames.append(raw)
            progress.update(current_item=sym, rows=int(len(raw)))
        else:
            progress.update(current_item=sym)
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
    progress.finish()
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _fetch_dividend_by_symbol(
    pro,
    fields: str,
    symbols: List[str],
    start_date: date,
    end_date: date,
    *,
    sleep_seconds: float = 0.2,
    log=None,
) -> pd.DataFrame:
    frames = []
    progress = ProgressLogger(name="event.dividend", total=len(symbols), unit="symbols", log=log, every=50, min_interval_seconds=20.0)
    for sym in symbols:
        raw = None
        for attempt in range(1, 4):
            try:
                raw = pro.dividend(
                    ts_code=_symbol_to_ts_code(sym),
                    fields=fields,
                )
                break
            except Exception as e:
                msg = str(e)
                if "200娆" in msg or "姣忓垎閽" in msg:
                    time.sleep(60)
                    continue
                if attempt == 3:
                    print(f"dividend {sym} failed: {e}")
                else:
                    time.sleep(1.5 * attempt)
        if raw is not None and not raw.empty:
            frames.append(raw)
            progress.update(current_item=sym, rows=int(len(raw)))
        else:
            progress.update(current_item=sym)
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
    progress.finish()
    if not frames:
        return pd.DataFrame()
    merged = pd.concat(frames, ignore_index=True)
    if merged.empty:
        return merged
    for col in ["ann_date", "end_date", "record_date", "ex_date", "pay_date"]:
        if col in merged.columns:
            dt = pd.to_datetime(merged[col], format="%Y%m%d", errors="coerce").dt.date
            mask = (dt >= start_date) & (dt <= end_date)
            if mask.any():
                return merged.loc[mask].copy()
    return merged


def _ensure_date_cols(df: pd.DataFrame, cols: List[str]) -> None:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], format="%Y%m%d", errors="coerce").dt.date


def _normalize_forecast(raw: pd.DataFrame, source_label: str) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()
    out = raw.copy()
    out["symbol"] = out.get("ts_code").astype(str).str.split(".").str[0]
    _ensure_date_cols(out, ["ann_date", "end_date"])
    rename = {
        "type": "forecast_type",
        "p_change_min": "p_change_min",
        "p_change_max": "p_change_max",
        "net_profit_min": "net_profit_min",
        "net_profit_max": "net_profit_max",
        "summary": "summary",
    }
    out = out.rename(columns=rename)
    numeric_cols = ["p_change_min", "p_change_max", "net_profit_min", "net_profit_max"]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out["report_type"] = out.get("report_type", "NA")
    out["report_type"] = out["report_type"].fillna("NA")
    out["source"] = source_label
    out["raw_payload"] = _frame_row_payloads(raw)
    keep = [
        "symbol",
        "ann_date",
        "end_date",
        "report_type",
        "forecast_type",
        "p_change_min",
        "p_change_max",
        "net_profit_min",
        "net_profit_max",
        "summary",
        "source",
        "raw_payload",
    ]
    out = out[keep].dropna(subset=["symbol", "ann_date", "end_date"])
    return out


def _normalize_express(raw: pd.DataFrame, source_label: str) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()
    out = raw.copy()
    out["symbol"] = out.get("ts_code").astype(str).str.split(".").str[0]
    _ensure_date_cols(out, ["ann_date", "end_date"])
    numeric_cols = [
        "revenue", "operate_profit", "total_profit", "n_income", "diluted_eps", "diluted_roe", "bps",
        "yoy_sales", "yoy_op", "yoy_tp", "yoy_dedu_np", "yoy_eps", "yoy_roe", "growth_assets",
    ]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out["source"] = source_label
    out["raw_payload"] = _frame_row_payloads(raw)
    keep = [
        "symbol", "ann_date", "end_date", "revenue", "operate_profit", "total_profit", "n_income",
        "diluted_eps", "diluted_roe", "bps", "yoy_sales", "yoy_op", "yoy_tp", "yoy_dedu_np",
        "yoy_eps", "yoy_roe", "growth_assets", "source", "raw_payload",
    ]
    out = out[keep].dropna(subset=["symbol", "ann_date", "end_date"])
    return out


def _normalize_fina_indicator(raw: pd.DataFrame, source_label: str) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()
    out = raw.copy()
    out["symbol"] = out.get("ts_code").astype(str).str.split(".").str[0]
    _ensure_date_cols(out, ["ann_date", "end_date"])
    numeric_cols = [
        "eps", "dt_eps", "roe", "roe_dt", "roa", "grossprofit_margin", "netprofit_margin",
        "debt_to_eqt", "netprofit_yoy", "q_sales_yoy", "or_yoy", "tr_yoy", "q_profit_yoy",
    ]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out["report_type"] = out.get("report_type", "NA")
    out["report_type"] = out["report_type"].fillna("NA")
    out["source"] = source_label
    out["raw_payload"] = _frame_row_payloads(raw)
    keep = [
        "symbol", "ann_date", "end_date", "report_type", "eps", "dt_eps", "roe", "roe_dt", "roa",
        "grossprofit_margin", "netprofit_margin", "debt_to_eqt", "netprofit_yoy",
        "q_sales_yoy", "or_yoy", "tr_yoy", "q_profit_yoy",
        "source", "raw_payload",
    ]
    out = out[keep].dropna(subset=["symbol", "ann_date", "end_date"])
    return out


def _normalize_disclosure_date(raw: pd.DataFrame, source_label: str) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()
    out = raw.copy()
    out["symbol"] = out.get("ts_code").astype(str).str.split(".").str[0]
    _ensure_date_cols(out, ["end_date", "pre_date", "actual_date", "modify_date"])
    out["source"] = source_label
    out["raw_payload"] = _frame_row_payloads(raw)
    keep = ["symbol", "end_date", "pre_date", "actual_date", "modify_date", "source", "raw_payload"]
    out = out[keep].dropna(subset=["symbol", "end_date"])
    return out


def _normalize_dividend(raw: pd.DataFrame, source_label: str) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()
    out = raw.copy()
    out["symbol"] = out.get("ts_code").astype(str).str.split(".").str[0]
    _ensure_date_cols(out, ["ann_date", "end_date", "record_date", "ex_date", "pay_date"])
    numeric_cols = ["stk_div", "cash_div"]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out["div_proc"] = out.get("div_proc", None)
    out["source"] = source_label
    out["raw_payload"] = _frame_row_payloads(raw)
    keep = [
        "symbol", "ann_date", "end_date", "record_date", "ex_date", "pay_date",
        "stk_div", "cash_div", "div_proc", "source", "raw_payload",
    ]
    out = out[keep].dropna(subset=["symbol", "ann_date", "end_date"])
    return out


def _normalize_anns_d(raw: pd.DataFrame, source_label: str) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()
    out = raw.copy()
    out["symbol"] = out.get("ts_code").astype(str).str.split(".").str[0]
    _ensure_date_cols(out, ["ann_date"])
    out["title"] = out.get("title", None)
    out["url"] = out.get("url", None)
    ann_type = out.get("ann_type", None)
    if ann_type is None and "type" in out.columns:
        ann_type = out["type"]
    out["type"] = ann_type
    out["source"] = source_label
    out["raw_payload"] = _frame_row_payloads(raw)
    keep = ["symbol", "ann_date", "title", "url", "type", "source", "raw_payload"]
    out = out[keep].dropna(subset=["symbol", "ann_date"])
    return out


def _upsert(engine, table: str, df: pd.DataFrame, key_cols: List[str], chunk_size: int = 2000) -> int:
    if df is None or df.empty:
        return 0
    work = df.copy().astype(object).where(pd.notna(df), None)
    cols = list(work.columns)
    insert_cols = ", ".join(cols)
    params = ", ".join([f":{c}" for c in cols])
    update_cols = [c for c in cols if c not in key_cols]
    update_clause = ", ".join([f"{c}=VALUES({c})" for c in update_cols])
    sql = f"INSERT INTO {table} ({insert_cols}) VALUES ({params}) ON DUPLICATE KEY UPDATE {update_clause}"
    affected = 0
    with engine.begin() as conn:
        for batch in chunked(work.to_dict(orient="records"), chunk_size):
            ret = conn.execute(text(sql), batch)
            affected += int(ret.rowcount or 0)
    return affected


def _max_date(engine, table: str, col: str) -> date | None:
    with engine.connect() as conn:
        value = conn.execute(text(f"SELECT MAX({col}) FROM {table}")).scalar()
    if value is None:
        return None
    if isinstance(value, date):
        return value
    return _coerce_date(value)


def load_forecast(engine, pro, start_date: date, end_date: date, source_label: str, log=None) -> Tuple[int, int]:
    fields = (
        "ts_code,ann_date,end_date,report_type,type,p_change_min,p_change_max,"
        "net_profit_min,net_profit_max,summary"
    )
    raw = _fetch_by_ann_date(pro, "forecast", fields, start_date, end_date, log=log)
    df = _normalize_forecast(raw, source_label)
    key_cols = ["symbol", "ann_date", "end_date", "forecast_type", "report_type"]
    affected = _upsert(engine, EVENT_TABLES["forecast"], df, key_cols)
    return int(len(df)), int(affected)


def load_express(engine, pro, start_date: date, end_date: date, source_label: str, log=None) -> Tuple[int, int]:
    fields = (
        "ts_code,ann_date,end_date,revenue,operate_profit,total_profit,n_income,diluted_eps,"
        "diluted_roe,bps,yoy_sales,yoy_op,yoy_tp,yoy_dedu_np,yoy_eps,yoy_roe,growth_assets"
    )
    raw = _fetch_by_ann_date(pro, "express", fields, start_date, end_date, log=log)
    df = _normalize_express(raw, source_label)
    key_cols = ["symbol", "ann_date", "end_date"]
    affected = _upsert(engine, EVENT_TABLES["express"], df, key_cols)
    return int(len(df)), int(affected)


def load_fina_indicator(engine, pro, start_date: date, end_date: date, source_label: str, symbols: List[str] | None = None, log=None) -> Tuple[int, int]:
    fields = (
        "ts_code,ann_date,end_date,report_type,eps,dt_eps,roe,roe_dt,roa,grossprofit_margin,"
        "netprofit_margin,debt_to_eqt,netprofit_yoy,q_sales_yoy,or_yoy,tr_yoy,q_profit_yoy"
    )
    raw = _fetch_by_ann_date(pro, "fina_indicator", fields, start_date, end_date, log=log)
    df = _normalize_fina_indicator(raw, source_label)
    key_cols = ["symbol", "ann_date", "end_date", "report_type"]
    affected = _upsert(engine, EVENT_TABLES["fina_indicator"], df, key_cols)
    return int(len(df)), int(affected)


def load_disclosure_date(engine, pro, start_date: date, end_date: date, source_label: str, log=None) -> Tuple[int, int]:
    fields = "ts_code,end_date,pre_date,actual_date,modify_date"
    raw = _fetch_by_month(pro, "disclosure_date", fields, start_date, end_date, log=log)
    df = _normalize_disclosure_date(raw, source_label)
    key_cols = ["symbol", "end_date"]
    affected = _upsert(engine, EVENT_TABLES["disclosure_date"], df, key_cols)
    return int(len(df)), int(affected)


def load_dividend(engine, pro, start_date: date, end_date: date, source_label: str, log=None) -> Tuple[int, int]:
    fields = "ts_code,ann_date,end_date,record_date,ex_date,pay_date,stk_div,cash_div,div_proc"
    raw = _fetch_dividend_by_ann_date(pro, fields, start_date, end_date, log=log)
    df = _normalize_dividend(raw, source_label)
    key_cols = ["symbol", "ann_date", "end_date", "record_date", "ex_date"]
    affected = _upsert(engine, EVENT_TABLES["dividend"], df, key_cols)
    return int(len(df)), int(affected)


def load_anns_d(engine, pro, start_date: date, end_date: date, source_label: str, log=None) -> Tuple[int, int]:
    fields = "ts_code,ann_date,title,url,ann_type"
    raw = _fetch_by_month(pro, "anns_d", fields, start_date, end_date, log=log)
    df = _normalize_anns_d(raw, source_label)
    key_cols = ["symbol", "ann_date", "title"]
    affected = _upsert(engine, EVENT_TABLES["anns_d"], df, key_cols)
    return int(len(df)), int(affected)


def _forecast_to_signal(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = pd.DataFrame()
    out["trade_date"] = df["ann_date"]
    out["symbol"] = df["symbol"]
    out["event_type"] = "FORECAST"
    out["event_subtype"] = df["forecast_type"].fillna("FORECAST_OTHER")
    out["event_direction"] = df["forecast_type"].astype(str).str.contains("增|预增|扭亏|略增", regex=True).map({True: 1, False: -1})
    out["event_score"] = out["event_direction"].map({1: 4, -1: 2}).fillna(2)
    out["anchor_date"] = df["ann_date"]
    out["raw_source_table"] = EVENT_TABLES["forecast"]
    out["raw_event_id"] = (
        df["symbol"].astype(str) + "|" + df["ann_date"].astype(str) + "|" + df["end_date"].astype(str)
    )
    out["version"] = "v1"
    return out


def _express_to_signal(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = pd.DataFrame()
    out["trade_date"] = df["ann_date"]
    out["symbol"] = df["symbol"]
    out["event_type"] = "EXPRESS"
    growth = pd.to_numeric(df.get("yoy_eps"), errors="coerce")
    out["event_subtype"] = growth.apply(lambda x: "EXPRESS_POSITIVE" if x is not None and x >= 0 else "EXPRESS_NEGATIVE")
    out["event_direction"] = growth.apply(lambda x: 1 if x is not None and x >= 0 else -1)
    out["event_score"] = out["event_direction"].map({1: 4, -1: 2}).fillna(2)
    out["anchor_date"] = df["ann_date"]
    out["raw_source_table"] = EVENT_TABLES["express"]
    out["raw_event_id"] = (
        df["symbol"].astype(str) + "|" + df["ann_date"].astype(str) + "|" + df["end_date"].astype(str)
    )
    out["version"] = "v1"
    return out


def _fina_to_signal(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = pd.DataFrame()
    out["trade_date"] = df["ann_date"]
    out["symbol"] = df["symbol"]
    out["event_type"] = "FUNDAMENTAL"
    roe = pd.to_numeric(df.get("roe"), errors="coerce")
    margin = pd.to_numeric(df.get("grossprofit_margin"), errors="coerce")
    subtype = []
    score = []
    direction = []
    for r, m in zip(roe, margin):
        if r is not None and r >= 10:
            subtype.append("FUNDAMENTAL_ROE_IMPROVE")
            score.append(4)
            direction.append(1)
        elif m is not None and m >= 30:
            subtype.append("FUNDAMENTAL_MARGIN_IMPROVE")
            score.append(4)
            direction.append(1)
        else:
            subtype.append("FUNDAMENTAL_NEUTRAL")
            score.append(2)
            direction.append(0)
    out["event_subtype"] = subtype
    out["event_score"] = score
    out["event_direction"] = direction
    out["anchor_date"] = df["ann_date"]
    out["raw_source_table"] = EVENT_TABLES["fina_indicator"]
    out["raw_event_id"] = (
        df["symbol"].astype(str) + "|" + df["ann_date"].astype(str) + "|" + df["end_date"].astype(str)
    )
    out["version"] = "v1"
    return out


def _dividend_to_signal(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = pd.DataFrame()
    out["trade_date"] = df["ann_date"]
    out["symbol"] = df["symbol"]
    out["event_type"] = "DIVIDEND"
    out["event_subtype"] = "DIVIDEND_APPROVED"
    out["event_score"] = 3
    out["event_direction"] = 1
    out["anchor_date"] = df["ann_date"]
    out["raw_source_table"] = EVENT_TABLES["dividend"]
    out["raw_event_id"] = (
        df["symbol"].astype(str) + "|" + df["ann_date"].astype(str) + "|" + df["end_date"].astype(str)
    )
    out["version"] = "v1"
    return out


def rebuild_event_signal_daily(engine, start_date: date, end_date: date, include_sources: set[str] | None = None) -> int:
    sources = {str(x).strip().lower() for x in include_sources} if include_sources else {"forecast", "express", "fina_indicator", "dividend"}
    with engine.connect() as conn:
        forecast = pd.DataFrame()
        express = pd.DataFrame()
        fina = pd.DataFrame()
        dividend = pd.DataFrame()
        if "forecast" in sources:
            forecast = pd.read_sql(
                text("SELECT * FROM cn_event_earnings_forecast WHERE ann_date BETWEEN :s AND :e"),
                conn,
                params={"s": start_date, "e": end_date},
            )
        if "express" in sources:
            express = pd.read_sql(
                text("SELECT * FROM cn_event_earnings_express WHERE ann_date BETWEEN :s AND :e"),
                conn,
                params={"s": start_date, "e": end_date},
            )
        if "fina_indicator" in sources:
            fina = pd.read_sql(
                text("SELECT * FROM cn_event_fina_indicator WHERE ann_date BETWEEN :s AND :e"),
                conn,
                params={"s": start_date, "e": end_date},
            )
        if "dividend" in sources:
            dividend = pd.read_sql(
                text("SELECT * FROM cn_event_dividend WHERE ann_date BETWEEN :s AND :e"),
                conn,
                params={"s": start_date, "e": end_date},
            )

    frames = [
        _forecast_to_signal(forecast),
        _express_to_signal(express),
        _fina_to_signal(fina),
        _dividend_to_signal(dividend),
    ]
    merged = pd.concat([f for f in frames if f is not None and not f.empty], ignore_index=True)
    if merged.empty:
        return 0
    merged = merged.dropna(subset=["trade_date", "symbol", "event_type"])
    key_cols = ["trade_date", "symbol", "event_type", "event_subtype", "raw_event_id", "version"]
    return _upsert(engine, EVENT_TABLES["signal_daily"], merged, key_cols)
def _quality_snapshot(engine, table: str, key_cols: List[str], date_cols: List[str], updated_col: str = "updated_at") -> Dict[str, object]:
    cols = ", ".join(key_cols)
    date_min = date_max = None
    if date_cols:
        col = date_cols[0]
        with engine.connect() as conn:
            date_min = conn.execute(text(f"SELECT MIN({col}) FROM {table}")).scalar()
            date_max = conn.execute(text(f"SELECT MAX({col}) FROM {table}")).scalar()
    with engine.connect() as conn:
        total = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar() or 0
        symbol_cnt = conn.execute(text(f"SELECT COUNT(DISTINCT symbol) FROM {table}")).scalar() or 0
        nulls = {}
        for c in key_cols:
            nulls[c] = conn.execute(text(f"SELECT COUNT(*) FROM {table} WHERE {c} IS NULL")).scalar() or 0
        dup_sql = f"""
            SELECT COUNT(*) - COUNT(DISTINCT {cols})
            FROM {table}
        """
        dup = conn.execute(text(dup_sql)).scalar() or 0
        updated_at = conn.execute(text(f"SELECT MAX({updated_col}) FROM {table}")).scalar()
    return {
        "table": table,
        "total_rows": int(total),
        "distinct_symbols": int(symbol_cnt),
        "date_min": str(date_min) if date_min else None,
        "date_max": str(date_max) if date_max else None,
        "dup_key": int(dup),
        "null_key_counts": nulls,
        "latest_updated_at": str(updated_at) if updated_at else None,
    }


def build_quality_report(engine) -> Dict[str, object]:
    meta = [
        (EVENT_TABLES["forecast"], ["symbol", "ann_date", "end_date", "forecast_type", "report_type"], ["ann_date"]),
        (EVENT_TABLES["express"], ["symbol", "ann_date", "end_date"], ["ann_date"]),
        (EVENT_TABLES["fina_indicator"], ["symbol", "ann_date", "end_date", "report_type"], ["ann_date"]),
        (EVENT_TABLES["disclosure_date"], ["symbol", "end_date"], ["end_date"]),
        (EVENT_TABLES["dividend"], ["symbol", "ann_date", "end_date", "record_date", "ex_date"], ["ann_date"]),
        (EVENT_TABLES["anns_d"], ["symbol", "ann_date", "title"], ["ann_date"]),
        (EVENT_TABLES["signal_daily"], ["trade_date", "symbol", "event_type", "event_subtype", "raw_event_id", "version"], ["trade_date"], "created_at"),
    ]
    out = {"generated_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"), "tables": []}
    for entry in meta:
        table, key_cols, date_cols = entry[0], entry[1], entry[2]
        updated_col = entry[3] if len(entry) > 3 else "updated_at"
        try:
            out["tables"].append(_quality_snapshot(engine, table, key_cols, date_cols, updated_col=updated_col))
        except Exception as e:
            out["tables"].append({"table": table, "error": str(e)})
    return out


def save_quality_report(engine, output_dir: str) -> Path:
    report = build_quality_report(engine)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path = Path(output_dir) / f"event_quality_{ts}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync CN event-driven datasets from Tushare Pro.")
    parser.add_argument("--start", default="2008-01-01", help="Start date YYYYMMDD or YYYY-MM-DD")
    parser.add_argument("--end", default="", help="End date YYYYMMDD or YYYY-MM-DD (default today)")
    parser.add_argument("--token", default="", help="Tushare token")
    parser.add_argument("--config", default="", help="Config file path for Tushare token")
    parser.add_argument("--with-anns", action="store_true", help="Try anns_d if permission allows")
    parser.add_argument("--with-signal", action="store_true", help="Build cn_event_signal_daily")
    parser.add_argument("--skip-fina", action="store_true", help="Skip fina_indicator (slow)")
    parser.add_argument("--max-symbols", type=int, default=0, help="Limit fina_indicator symbols for batch run")
    parser.add_argument("--symbols", default="", help="Comma/space-separated symbols for fina_indicator")
    parser.add_argument("--audit-dir", default="audit_reports", help="Directory to save quality report")
    args = parser.parse_args()

    patch_pandas_fillna_method_compat()
    end_date = _parse_ymd(args.end) if str(args.end).strip() else date.today()
    start_date = _parse_ymd(args.start)
    if start_date > end_date:
        raise SystemExit(f"invalid date range: {start_date} > {end_date}")

    token, tried_files = resolve_tushare_token(args.token, args.config)
    if not token:
        msg = "Tushare token is required. Use --token, env TUSHARE_TOKEN/TS_TOKEN, or --config."
        if tried_files:
            msg += f" Tried files: {', '.join(str(p) for p in tried_files)}"
        raise SystemExit(msg)

    engine = build_engine()
    ensure_tables(engine)
    pro = ts.pro_api(token)

    rows_forecast, aff_forecast = load_forecast(engine, pro, start_date, end_date, "tushare_forecast")
    rows_express, aff_express = load_express(engine, pro, start_date, end_date, "tushare_express")
    symbols = []
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.replace(",", " ").split() if s.strip()]
    elif args.max_symbols and args.max_symbols > 0:
        all_symbols = _get_symbol_universe(engine)
        symbols = all_symbols[: int(args.max_symbols)]

    rows_fina = aff_fina = 0
    if not args.skip_fina:
        rows_fina, aff_fina = load_fina_indicator(
            engine,
            pro,
            start_date,
            end_date,
            "tushare_fina_indicator",
            symbols=symbols if symbols else None,
        )
    rows_disclosure, aff_disclosure = load_disclosure_date(engine, pro, start_date, end_date, "tushare_disclosure_date")
    rows_dividend, aff_dividend = load_dividend(engine, pro, start_date, end_date, "tushare_dividend")

    rows_anns = aff_anns = 0
    if args.with_anns:
        try:
            rows_anns, aff_anns = load_anns_d(engine, pro, start_date, end_date, "tushare_anns_d")
        except Exception as e:
            print(f"anns_d skipped: {e}")

    if args.with_signal:
        rebuild_event_signal_daily(engine, start_date, end_date)

    report_path = save_quality_report(engine, args.audit_dir)
    print(
        "event_sync "
        f"forecast={rows_forecast}/{aff_forecast} "
        f"express={rows_express}/{aff_express} "
        f"fina={rows_fina}/{aff_fina} "
        f"disclosure={rows_disclosure}/{aff_disclosure} "
        f"dividend={rows_dividend}/{aff_dividend} "
        f"anns_d={rows_anns}/{aff_anns} "
        f"report={report_path}"
    )


if __name__ == "__main__":
    main()
