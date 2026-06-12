from __future__ import annotations

import argparse
import builtins
import html
import json
import math
import os
import re
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from io import StringIO
from typing import Callable, Iterable, List, Sequence

import pandas as pd
import requests
from sqlalchemy import text

try:  # Tushare is optional in V2. EastMoney is the primary source.
    import tushare as ts  # type: ignore
except Exception:  # pragma: no cover
    ts = None  # type: ignore

from app.settings import build_engine, load_sql_for_current_db
from app.tools.sync_cn_stock_daily_price_from_tushare import (
    patch_pandas_fillna_method_compat,
    resolve_tushare_token,
)


def print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    return builtins.print(*args, **kwargs)


DDL_PATH = "docs/DDL/cn_market.cn_gcrl_public_fund.sql"
SOURCE_FUND_BASIC_TUSHARE = "tushare_fund_basic_optional"
SOURCE_FUND_BASIC_EASTMONEY = "eastmoney_fund_registry"
SOURCE_EASTMONEY_HOLDING = "eastmoney_fund_archives_jjcc"

DEFAULT_MAX_FUNDS = int(os.getenv("GCRL_CN_DEFAULT_MAX_FUNDS", "500"))
DEFAULT_START_YEAR = int(os.getenv("GCRL_CN_START_YEAR", "2020"))

DEFAULT_PUBLIC_FUND_INSTITUTIONS = [
    ("CN_PUBFUND_EFUND", "易方达基金管理有限公司", "易方达", 1),
    ("CN_PUBFUND_CHINAAMC", "华夏基金管理有限公司", "华夏基金", 1),
    ("CN_PUBFUND_HARVEST", "嘉实基金管理有限公司", "嘉实基金", 1),
    ("CN_PUBFUND_FULLGOAL", "富国基金管理有限公司", "富国基金", 1),
    ("CN_PUBFUND_SOUTHERN", "南方基金管理股份有限公司", "南方基金", 1),
    ("CN_PUBFUND_99FUND", "汇添富基金管理股份有限公司", "汇添富", 1),
    ("CN_PUBFUND_IGW", "景顺长城基金管理有限公司", "景顺长城", 1),
    ("CN_PUBFUND_GFFUNDS", "广发基金管理有限公司", "广发基金", 1),
    ("CN_PUBFUND_CMF", "招商基金管理有限公司", "招商基金", 1),
    ("CN_PUBFUND_BOSERA", "博时基金管理有限公司", "博时基金", 1),
    ("CN_PUBFUND_CGF", "中欧基金管理有限公司", "中欧基金", 1),
    ("CN_PUBFUND_XQGLOBAL", "兴证全球基金管理有限公司", "兴证全球", 1),
    ("CN_PUBFUND_YH", "银华基金管理股份有限公司", "银华基金", 1),
    ("CN_PUBFUND_ICBCCS", "工银瑞信基金管理有限公司", "工银瑞信", 1),
    ("CN_PUBFUND_BOCOM", "交银施罗德基金管理有限公司", "交银施罗德", 1),
]

FUND_BASIC_FIELDS = (
    "ts_code,name,management,custodian,fund_type,found_date,due_date,list_date,"
    "issue_date,delist_date,issue_amount,m_fee,c_fee,duration_year,p_value,min_amount,"
    "exp_return,benchmark,status,invest_type,type,trustee,purc_startdate,redm_startdate,market"
)


@dataclass(frozen=True)
class PeriodPlan:
    periods: list[date]
    label: str


class GcrlDataUnavailable(RuntimeError):
    pass


def _progress(msg: str, log=None) -> None:
    if log is not None:
        try:
            log.info(msg)
            return
        except Exception:
            pass
    print(msg)


def _parse_ymd(raw: str) -> date:
    s = str(raw or "").strip()
    if len(s) == 10 and s[4] == "-" and s[7] == "-":
        return datetime.strptime(s, "%Y-%m-%d").date()
    return datetime.strptime(s, "%Y%m%d").date()


def _to_date_series(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, format="%Y%m%d", errors="coerce").dt.date


def _quarter_ends(year: int) -> list[date]:
    return [date(year, 3, 31), date(year, 6, 30), date(year, 9, 30), date(year, 12, 31)]


def _prev_quarter_end(d: date) -> date:
    month = ((d.month - 1) // 3) * 3 + 1
    return date(d.year, month, 1) - timedelta(days=1)


def _latest_completed_report_period(today: date | None = None) -> date:
    # Fund quarterly disclosure is lagged. Use conservative public-report availability windows.
    t = today or date.today()
    availability = [
        (date(t.year, 4, 30), date(t.year, 3, 31)),
        (date(t.year, 8, 31), date(t.year, 6, 30)),
        (date(t.year, 10, 31), date(t.year, 9, 30)),
        (date(t.year + 1, 3, 31), date(t.year, 12, 31)),
    ]
    for available_on, period in availability:
        if t >= available_on:
            latest = period
        else:
            break
    else:
        return latest
    if t < date(t.year, 4, 30):
        return date(t.year - 1, 12, 31)
    return latest


def _periods_between(start: date, end: date) -> list[date]:
    periods: list[date] = []
    for y in range(start.year, end.year + 1):
        for p in _quarter_ends(y):
            if start <= p <= end:
                periods.append(p)
    return periods


def resolve_period_plan(args) -> PeriodPlan:
    chosen = [bool(args.report_period), bool(args.year), bool(args.start_period or args.end_period), bool(args.backfill_all)]
    if sum(chosen) > 1:
        raise ValueError("Only one of --report-period / --year / --start-period+--end-period / --backfill-all can be used.")
    if args.report_period:
        p = _parse_ymd(args.report_period)
        return PeriodPlan([p], p.strftime("%Y%m%d"))
    if args.year:
        y = int(args.year)
        return PeriodPlan(_quarter_ends(y), str(y))
    if args.start_period or args.end_period:
        if not args.start_period or not args.end_period:
            raise ValueError("--start-period and --end-period must be used together.")
        s = _parse_ymd(args.start_period)
        e = _parse_ymd(args.end_period)
        return PeriodPlan(_periods_between(s, e), f"{s:%Y%m%d}-{e:%Y%m%d}")
    if args.backfill_all:
        end = _latest_completed_report_period()
        return PeriodPlan(_periods_between(date(DEFAULT_START_YEAR, 3, 31), end), f"{DEFAULT_START_YEAR}-{end:%Y%m%d}")
    p = _latest_completed_report_period()
    return PeriodPlan([p], f"default_latest_{p:%Y%m%d}")


def _chunked(records: Sequence[dict], chunk_size: int) -> Iterable[Sequence[dict]]:
    for i in range(0, len(records), chunk_size):
        yield records[i : i + chunk_size]


def _safe_scalar(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def _df_records(df: pd.DataFrame) -> list[dict]:
    if df is None or df.empty:
        return []
    clean = df.replace([float("inf"), float("-inf")], pd.NA)
    records = clean.astype(object).where(pd.notna(clean), None).to_dict(orient="records")
    return [{k: _safe_scalar(v) for k, v in row.items()} for row in records]


def _fund_plain_code(fund_code: object) -> str:
    s = str(fund_code or "").strip()
    return re.sub(r"\D", "", s.split(".")[0]).zfill(6) if s else ""


def _fund_code_with_exchange(raw: object) -> str:
    s = _fund_plain_code(raw)
    if not s:
        return ""
    # EastMoney fund pages work with plain code; registry PK keeps exchange suffix when known, OF otherwise.
    if str(raw).endswith((".SH", ".SZ", ".OF")):
        return str(raw)
    if s.startswith(("5", "6")):
        return f"{s}.SH"
    if s.startswith(("1", "0")):
        return f"{s}.SZ"
    return f"{s}.OF"


def _stock_code_with_exchange(symbol: object) -> str:
    s = re.sub(r"\D", "", str(symbol or "").strip())
    if len(s) != 6:
        return s
    if s.startswith(("6", "9")):
        return f"{s}.SH"
    return f"{s}.SZ"


def _parse_cn_number(raw) -> float | None:
    if raw is None:
        return None
    s0 = str(raw).strip().replace(",", "")
    if s0 in {"", "--", "-", "nan", "None"}:
        return None
    s = s0.replace("%", "")
    multiplier = 1.0
    if "亿" in s:
        multiplier = 100000000.0
        s = s.replace("亿", "")
    elif "万" in s:
        multiplier = 10000.0
        s = s.replace("万", "")
    s = re.sub(r"[^0-9.\-]", "", s)
    if not s or s in {".", "-"}:
        return None
    try:
        return float(s) * multiplier
    except Exception:
        return None


def apply_ddl(engine) -> None:
    sql = load_sql_for_current_db(DDL_PATH)
    with engine.begin() as conn:
        for stmt in [part.strip() for part in sql.split(";") if part.strip()]:
            conn.execute(text(stmt))


def ensure_tables(engine) -> None:
    apply_ddl(engine)


def seed_institution_registry(engine) -> int:
    records = [
        {
            "institution_id": x[0],
            "institution_name": x[1],
            "institution_short_name": x[2],
            "tier": int(x[3]),
        }
        for x in DEFAULT_PUBLIC_FUND_INSTITUTIONS
    ]
    sql = text(
        """
        INSERT INTO cn_gcrl_institution_registry
            (institution_id, institution_name, institution_short_name, country, institution_type, tier, source, is_active)
        VALUES
            (:institution_id, :institution_name, :institution_short_name, 'CN', 'PUBLIC_FUND', :tier, 'seed_gcrl_cn_public_fund_v2', 1)
        ON DUPLICATE KEY UPDATE
            institution_name = VALUES(institution_name),
            institution_short_name = VALUES(institution_short_name),
            tier = VALUES(tier),
            source = VALUES(source),
            is_active = VALUES(is_active)
        """
    )
    with engine.begin() as conn:
        conn.execute(sql, records)
    return len(records)


def load_institution_alias_map(engine) -> dict[str, str]:
    sql = text(
        """
        SELECT institution_id, institution_name, institution_short_name
        FROM cn_gcrl_institution_registry
        WHERE country = 'CN'
          AND institution_type = 'PUBLIC_FUND'
          AND is_active = 1
        """
    )
    mapping: dict[str, str] = {}
    with engine.connect() as conn:
        for inst_id, name, short in conn.execute(sql).fetchall():
            for raw in [name, short, str(name or "").replace("管理有限公司", "").replace("管理股份有限公司", "")]:
                alias = str(raw or "").strip()
                if alias:
                    mapping[alias] = str(inst_id)
    return mapping


def _match_institution_id(text_value: object, alias_map: dict[str, str]) -> str | None:
    s = str(text_value or "").strip()
    if not s:
        return None
    if s in alias_map:
        return alias_map[s]
    for alias, inst_id in alias_map.items():
        if alias and (alias in s or s in alias):
            return inst_id
    return None


def fetch_tushare_fund_basic_optional(token: str, log=None) -> pd.DataFrame:
    if not token or ts is None:
        return pd.DataFrame()
    try:
        patch_pandas_fillna_method_compat()
        pro = ts.pro_api(token)
        frames: list[pd.DataFrame] = []
        for market in ["E", "O"]:
            df = pro.fund_basic(market=market, fields=FUND_BASIC_FIELDS)
            if df is not None and not df.empty:
                frames.append(df)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True).drop_duplicates(subset=["ts_code"], keep="first")
    except Exception as exc:
        _progress(f"[GCRL-CN] optional Tushare fund_basic skipped: {exc}", log)
        return pd.DataFrame()


def fetch_eastmoney_fund_registry(alias_map: dict[str, str], log=None) -> pd.DataFrame:
    # Public JS contains fund code/name/type. Company is not always explicit, so this source is best-effort.
    # If the DB already has Tushare-backed fund registry, that will be preferred by load_existing_fund_registry().
    url = "https://fund.eastmoney.com/js/fundcode_search.js"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        resp = requests.get(url, headers=headers, timeout=20)
        resp.encoding = "utf-8"
        resp.raise_for_status()
        m = re.search(r"\[(.*)\]", resp.text, flags=re.S)
        if not m:
            return pd.DataFrame()
        data = json.loads("[" + m.group(1) + "]")
    except Exception as exc:
        _progress(f"[GCRL-CN] EastMoney fund registry fetch failed: {exc}", log)
        return pd.DataFrame()
    rows = []
    for item in data:
        if not isinstance(item, list) or len(item) < 4:
            continue
        code = _fund_code_with_exchange(item[0])
        name = str(item[2] or "").strip()
        ftype = str(item[3] or "").strip()
        inst_id = _match_institution_id(name, alias_map)
        if not inst_id:
            continue
        rows.append(
            {
                "fund_code": code,
                "fund_name": name,
                "management": None,
                "custodian": None,
                "fund_type": ftype,
                "found_date": None,
                "due_date": None,
                "list_date": None,
                "issue_date": None,
                "delist_date": None,
                "issue_amount": None,
                "m_fee": None,
                "c_fee": None,
                "duration_year": None,
                "p_value": None,
                "min_amount": None,
                "exp_return": None,
                "benchmark": None,
                "status": "L",
                "invest_type": None,
                "type": None,
                "trustee": None,
                "purc_startdate": None,
                "redm_startdate": None,
                "market": code.split(".")[-1] if "." in code else None,
                "institution_id": inst_id,
            }
        )
    return pd.DataFrame(rows).drop_duplicates(subset=["fund_code"], keep="first") if rows else pd.DataFrame()


def normalize_tushare_fund_basic(raw: pd.DataFrame, alias_map: dict[str, str]) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()
    out = raw.copy()
    for c in ["found_date", "due_date", "list_date", "issue_date", "delist_date", "purc_startdate", "redm_startdate"]:
        if c in out.columns:
            out[c] = _to_date_series(out[c])
    for c in ["issue_amount", "m_fee", "c_fee", "duration_year", "p_value", "min_amount", "exp_return"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    out["institution_id"] = out["management"].apply(lambda x: _match_institution_id(x, alias_map))
    out = out[out["institution_id"].notna()].copy()
    out = out.rename(columns={"ts_code": "fund_code", "name": "fund_name"})
    cols = _fund_registry_cols()
    for c in cols:
        if c not in out.columns:
            out[c] = None
    return out[cols].drop_duplicates(subset=["fund_code"], keep="first")


def _fund_registry_cols() -> list[str]:
    return [
        "fund_code", "fund_name", "management", "custodian", "fund_type", "found_date", "due_date", "list_date",
        "issue_date", "delist_date", "issue_amount", "m_fee", "c_fee", "duration_year", "p_value", "min_amount",
        "exp_return", "benchmark", "status", "invest_type", "type", "trustee", "purc_startdate", "redm_startdate",
        "market", "institution_id",
    ]


def load_existing_fund_registry(engine) -> pd.DataFrame:
    try:
        with engine.connect() as conn:
            rows = conn.execute(text("SELECT * FROM cn_gcrl_fund_registry WHERE institution_id IS NOT NULL")).mappings().all()
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()


def replace_fund_registry(engine, fund_df: pd.DataFrame, *, source: str) -> int:
    payload = _df_records(fund_df)
    if not payload:
        return 0
    cols = _fund_registry_cols()
    sql = text(
        f"""
        INSERT INTO cn_gcrl_fund_registry ({', '.join(cols)}, source)
        VALUES ({', '.join(':' + c for c in cols)}, :source)
        ON DUPLICATE KEY UPDATE
            {', '.join(f'{c}=VALUES({c})' for c in cols if c != 'fund_code')},
            source=VALUES(source)
        """
    )
    for r in payload:
        r["source"] = source
    with engine.begin() as conn:
        for chunk in _chunked(payload, 1000):
            conn.execute(sql, list(chunk))
    return len(payload)


def prepare_fund_registry(engine, alias_map: dict[str, str], token: str, log=None) -> tuple[pd.DataFrame, str, int, int]:
    raw_tushare = fetch_tushare_fund_basic_optional(token, log=log)
    tushare_df = normalize_tushare_fund_basic(raw_tushare, alias_map)
    if not tushare_df.empty:
        rows = replace_fund_registry(engine, tushare_df, source=SOURCE_FUND_BASIC_TUSHARE)
        return tushare_df, SOURCE_FUND_BASIC_TUSHARE, int(len(raw_tushare)), rows

    existing = load_existing_fund_registry(engine)
    if existing is not None and not existing.empty:
        cols = _fund_registry_cols()
        for c in cols:
            if c not in existing.columns:
                existing[c] = None
        return existing[cols], "existing_cn_gcrl_fund_registry", int(len(existing)), int(len(existing))

    em_df = fetch_eastmoney_fund_registry(alias_map, log=log)
    if not em_df.empty:
        rows = replace_fund_registry(engine, em_df, source=SOURCE_FUND_BASIC_EASTMONEY)
        return em_df, SOURCE_FUND_BASIC_EASTMONEY, int(len(em_df)), rows

    return pd.DataFrame(columns=_fund_registry_cols()), "none", 0, 0


def _eastmoney_holding_urls(fund_plain_code: str, report_period: date) -> list[str]:
    rt = int(time.time() * 1000)
    q_month = {3: 3, 6: 6, 9: 9, 12: 12}.get(report_period.month, report_period.month)
    base = "https://fundf10.eastmoney.com/FundArchivesDatas.aspx"
    return [
        f"{base}?type=jjcc&code={fund_plain_code}&topline=10&year={report_period.year}&month={q_month}&rt={rt}",
        f"{base}?type=jjcc&code={fund_plain_code}&topline=10&year={report_period.year}&rt={rt}",
        f"{base}?type=jjcc&code={fund_plain_code}&year={report_period.year}&month={q_month}&rt={rt}",
    ]


def _extract_js_content(html_text: str) -> str:
    # EastMoney returns JS object: var apidata={ content:"...",arryear:[...] }.
    patterns = [
        r'content\s*:\s*"(.*?)"\s*,\s*arryear',
        r"content\s*:\s*'(.*?)'\s*,\s*arryear",
        r'"content"\s*:\s*"(.*?)"\s*,\s*"arryear"',
    ]
    for pat in patterns:
        m = re.search(pat, html_text, flags=re.S)
        if m:
            content = m.group(1)
            content = content.replace(r'\"', '"').replace(r"\/", "/")
            content = content.replace(r"\n", "").replace(r"\t", "").replace(r"\r", "")
            return html.unescape(content)
    # Some mirrors return raw HTML table directly.
    if "<table" in html_text and "股票代码" in html_text:
        return html_text
    return ""


def _normalize_eastmoney_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).replace("\n", "").replace(" ", "") for c in out.columns]
    return out


def _pick_col(cols: Sequence[str], candidates: Sequence[str]) -> str | None:
    for cand in candidates:
        for c in cols:
            if cand in str(c):
                return str(c)
    return None


def fetch_eastmoney_fund_holding_one(
    fund_code: str,
    report_period: date,
    *,
    session: requests.Session | None = None,
    timeout: float = 12.0,
) -> pd.DataFrame:
    plain = _fund_plain_code(fund_code)
    if not plain:
        return pd.DataFrame()
    sess = session or requests.Session()
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120 Safari/537.36",
        "Referer": f"https://fundf10.eastmoney.com/ccmx_{plain}.html",
        "Accept": "*/*",
    }
    last_text = ""
    for url in _eastmoney_holding_urls(plain, report_period):
        resp = sess.get(url, headers=headers, timeout=timeout)
        resp.encoding = resp.apparent_encoding or "utf-8"
        resp.raise_for_status()
        last_text = resp.text
        content = _extract_js_content(resp.text)
        if content and "暂无" not in content and "<table" in content:
            break
    else:
        content = _extract_js_content(last_text)
    if not content or "暂无" in content or "<table" not in content:
        return pd.DataFrame()
    try:
        tables = pd.read_html(StringIO(content))
    except Exception:
        return pd.DataFrame()
    if not tables:
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    for table in tables:
        table = _normalize_eastmoney_columns(table)
        cols = list(table.columns)
        code_col = _pick_col(cols, ["股票代码", "代码"])
        name_col = _pick_col(cols, ["股票名称", "名称"])
        ratio_col = _pick_col(cols, ["占净值比例", "占基金净值比例", "比例"])
        shares_col = _pick_col(cols, ["持股数", "持有股数"])
        value_col = _pick_col(cols, ["持仓市值", "市值"])
        if not code_col:
            continue
        rows = []
        for _, row in table.iterrows():
            symbol = re.sub(r"\D", "", str(row.get(code_col, "")))
            if len(symbol) != 6:
                continue
            shares = _parse_cn_number(row.get(shares_col)) if shares_col else None
            market_value = _parse_cn_number(row.get(value_col)) if value_col else None
            ratio = _parse_cn_number(row.get(ratio_col)) if ratio_col else None
            if shares is not None and shares_col and "万" in shares_col and "万" not in str(row.get(shares_col, "")):
                shares *= 10000.0
            if market_value is not None and value_col and "万" in value_col and "万" not in str(row.get(value_col, "")):
                market_value *= 10000.0
            rows.append(
                {
                    "report_period": report_period,
                    "fund_code": fund_code,
                    "stock_code": _stock_code_with_exchange(symbol),
                    "symbol": symbol,
                    "stock_name": str(row.get(name_col, "")).strip() if name_col else None,
                    "ann_date": None,
                    "shares": shares,
                    "market_value": market_value,
                    "stock_mkv_ratio": ratio,
                    "stock_float_ratio": None,
                }
            )
        if rows:
            frames.append(pd.DataFrame(rows))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).drop_duplicates(subset=["fund_code", "stock_code"], keep="first")


def _select_collection_funds(fund_registry: pd.DataFrame, max_funds: int) -> pd.DataFrame:
    if fund_registry is None or fund_registry.empty:
        return pd.DataFrame()
    df = fund_registry.copy()
    if "fund_type" in df.columns:
        mask = df["fund_type"].astype(str).str.contains("股票|混合|ETF|LOF|指数", na=False)
        df = df[mask].copy()
    # deterministic tier/company order, then fund_code; no stock/theme analysis here.
    inst_order = {x[0]: i for i, x in enumerate(DEFAULT_PUBLIC_FUND_INSTITUTIONS)}
    df["_inst_order"] = df["institution_id"].map(inst_order).fillna(9999)
    df["_is_index"] = df.get("invest_type", pd.Series([None] * len(df))).astype(str).str.contains("指数|被动", na=False).astype(int)
    df = df.sort_values(["_inst_order", "_is_index", "fund_code"], ascending=[True, True, True])
    df = df.drop_duplicates(subset=["fund_code"], keep="first")
    if max_funds and max_funds > 0:
        df = df.head(int(max_funds)).copy()
    return df.drop(columns=[c for c in ["_inst_order", "_is_index"] if c in df.columns])


def fetch_eastmoney_fund_holdings(
    fund_registry: pd.DataFrame,
    report_period: date,
    *,
    max_funds: int = DEFAULT_MAX_FUNDS,
    sleep_seconds: float = 0.08,
    log=None,
) -> tuple[pd.DataFrame, dict]:
    df = _select_collection_funds(fund_registry, max_funds=max_funds)
    if df.empty:
        return pd.DataFrame(), {"attempted_funds": 0, "success_funds": 0, "failed_funds": 0, "empty_funds": 0}

    sess = requests.Session()
    frames: list[pd.DataFrame] = []
    attempted = success = failed = empty = 0
    inst_map = dict(zip(df["fund_code"], df["institution_id"]))
    total = len(df["fund_code"].dropna().astype(str).drop_duplicates())
    next_mark = 0
    for fund_code in df["fund_code"].dropna().astype(str).drop_duplicates():
        attempted += 1
        try:
            one = fetch_eastmoney_fund_holding_one(fund_code, report_period, session=sess)
            if one is None or one.empty:
                empty += 1
            else:
                one["institution_id"] = inst_map.get(fund_code)
                frames.append(one)
                success += 1
        except Exception as exc:
            failed += 1
            if failed <= 10:
                _progress(f"[GCRL-CN] holding failed fund={fund_code} err={exc}", log)
        if attempted == 1 or attempted >= next_mark or attempted == total:
            _progress(f"  Holdings progress {attempted}/{total} success={success} empty={empty} failed={failed}", log)
            next_mark = max(next_mark + 25, attempted + 25)
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
    if not frames:
        return pd.DataFrame(), {"attempted_funds": attempted, "success_funds": success, "failed_funds": failed, "empty_funds": empty}
    out = pd.concat(frames, ignore_index=True)
    cols = [
        "report_period", "fund_code", "institution_id", "stock_code", "symbol", "stock_name", "ann_date",
        "shares", "market_value", "stock_mkv_ratio", "stock_float_ratio",
    ]
    for c in cols:
        if c not in out.columns:
            out[c] = None
    out = out[out["institution_id"].notna()].copy()
    out = out[cols].drop_duplicates(subset=["report_period", "fund_code", "stock_code"], keep="first")
    return out, {"attempted_funds": attempted, "success_funds": success, "failed_funds": failed, "empty_funds": empty}


def replace_position_snapshot(engine, report_period: date, df: pd.DataFrame, *, source: str) -> int:
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM cn_gcrl_position_snapshot WHERE report_period = :p"), {"p": report_period})
        payload = _df_records(df)
        if not payload:
            return 0
        cols = list(df.columns)
        sql = text(
            f"""
            INSERT INTO cn_gcrl_position_snapshot ({', '.join(cols)}, source)
            VALUES ({', '.join(':' + c for c in cols)}, :source)
            """
        )
        for r in payload:
            r["source"] = source
        for chunk in _chunked(payload, 1000):
            conn.execute(sql, list(chunk))
    return len(payload)


def build_and_replace_position_change(engine, report_period: date) -> int:
    prev_period = _prev_quarter_end(report_period)
    sql = text(
        """
        SELECT
            COALESCE(cur.report_period, :report_period) AS report_period,
            :prev_period AS prev_report_period,
            COALESCE(cur.fund_code, prev.fund_code) AS fund_code,
            COALESCE(cur.institution_id, prev.institution_id) AS institution_id,
            COALESCE(cur.stock_code, prev.stock_code) AS stock_code,
            COALESCE(cur.symbol, prev.symbol) AS symbol,
            COALESCE(cur.stock_name, prev.stock_name) AS stock_name,
            prev.shares AS prev_shares,
            cur.shares AS current_shares,
            prev.market_value AS prev_market_value,
            cur.market_value AS current_market_value
        FROM (SELECT * FROM cn_gcrl_position_snapshot WHERE report_period = :report_period) cur
        LEFT JOIN (SELECT * FROM cn_gcrl_position_snapshot WHERE report_period = :prev_period) prev
          ON cur.fund_code = prev.fund_code AND cur.stock_code = prev.stock_code
        UNION ALL
        SELECT
            :report_period AS report_period,
            :prev_period AS prev_report_period,
            prev.fund_code,
            prev.institution_id,
            prev.stock_code,
            prev.symbol,
            prev.stock_name,
            prev.shares AS prev_shares,
            NULL AS current_shares,
            prev.market_value AS prev_market_value,
            NULL AS current_market_value
        FROM (SELECT * FROM cn_gcrl_position_snapshot WHERE report_period = :prev_period) prev
        LEFT JOIN (SELECT * FROM cn_gcrl_position_snapshot WHERE report_period = :report_period) cur
          ON cur.fund_code = prev.fund_code AND cur.stock_code = prev.stock_code
        WHERE cur.fund_code IS NULL
        """
    )
    with engine.connect() as conn:
        rows = conn.execute(sql, {"report_period": report_period, "prev_period": prev_period}).fetchall()
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM cn_gcrl_position_change WHERE report_period = :p"), {"p": report_period})
    if not rows:
        return 0
    df = pd.DataFrame(rows, columns=list(rows[0]._mapping.keys()))
    for c in ["prev_shares", "current_shares", "prev_market_value", "current_market_value"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["change_shares"] = df["current_shares"].fillna(0) - df["prev_shares"].fillna(0)
    df["market_value_change"] = df["current_market_value"].fillna(0) - df["prev_market_value"].fillna(0)

    def classify(row) -> str:
        prev = row["prev_shares"]
        cur = row["current_shares"]
        if pd.isna(prev) and not pd.isna(cur):
            return "NEW_POSITION"
        if not pd.isna(prev) and pd.isna(cur):
            return "EXIT_POSITION"
        if row["change_shares"] > 0:
            return "ADD_POSITION"
        if row["change_shares"] < 0:
            return "REDUCE_POSITION"
        return "UNCHANGED"

    df["change_type"] = df.apply(classify, axis=1)
    df["change_ratio_pct"] = None
    mask = df["prev_shares"].notna() & (df["prev_shares"] != 0)
    df.loc[mask, "change_ratio_pct"] = df.loc[mask, "change_shares"] / df.loc[mask, "prev_shares"] * 100.0
    cols = [
        "report_period", "prev_report_period", "fund_code", "institution_id", "stock_code", "symbol", "stock_name", "change_type",
        "prev_shares", "current_shares", "change_shares", "change_ratio_pct", "prev_market_value", "current_market_value", "market_value_change",
    ]
    payload = _df_records(df[cols])
    for r in payload:
        r["source"] = "derived_from_cn_gcrl_position_snapshot"
    ins = text(
        f"""
        INSERT INTO cn_gcrl_position_change ({', '.join(cols)}, source)
        VALUES ({', '.join(':' + c for c in cols)}, :source)
        """
    )
    with engine.begin() as conn:
        for chunk in _chunked(payload, 1000):
            conn.execute(ins, list(chunk))
    return len(payload)


def upsert_freshness(engine, dataset_name: str, report_period: date, source: str, status: str, row_count: int, message: str = "") -> None:
    sql = text(
        """
        INSERT INTO cn_gcrl_data_freshness (dataset_name, report_period, source, status, row_count, message)
        VALUES (:dataset_name, :report_period, :source, :status, :row_count, :message)
        ON DUPLICATE KEY UPDATE
            status = VALUES(status),
            row_count = VALUES(row_count),
            refreshed_at = CURRENT_TIMESTAMP,
            message = VALUES(message)
        """
    )
    with engine.begin() as conn:
        conn.execute(sql, {"dataset_name": dataset_name, "report_period": report_period, "source": source, "status": status, "row_count": int(row_count), "message": str(message or "")[:512]})


def upsert_source_status(engine, source_name: str, source_type: str, status: str, message: str = "", row_count: int = 0) -> None:
    sql = text(
        """
        INSERT INTO cn_gcrl_data_source_status (source_name, source_type, status, message, row_count, last_success_time)
        VALUES (:source_name, :source_type, :status, :message, :row_count, IF(:status IN ('OK','SUCCESS','PARTIAL_OK'), CURRENT_TIMESTAMP, NULL))
        ON DUPLICATE KEY UPDATE
            source_type = VALUES(source_type),
            status = VALUES(status),
            message = VALUES(message),
            row_count = VALUES(row_count),
            last_check_time = CURRENT_TIMESTAMP,
            last_success_time = IF(:status IN ('OK','SUCCESS','PARTIAL_OK'), CURRENT_TIMESTAMP, last_success_time)
        """
    )
    with engine.begin() as conn:
        conn.execute(sql, {"source_name": source_name, "source_type": source_type, "status": status, "message": str(message or "")[:512], "row_count": int(row_count)})


def run_one_period(
    *,
    engine,
    report_period: date,
    token: str = "",
    max_funds: int = DEFAULT_MAX_FUNDS,
    eastmoney_sleep_seconds: float = 0.08,
    dry_run: bool = False,
    allow_empty: bool = False,
    log=None,
) -> dict:
    _progress("================================================", log)
    _progress(f"GCRL_CN_PUBLIC_FUND_V2 period={report_period:%Y%m%d}", log)
    _progress("================================================", log)

    ensure_tables(engine)
    _progress("Step 1/5 Institution Registry", log)
    seeded = seed_institution_registry(engine)
    alias_map = load_institution_alias_map(engine)
    _progress(f"  institutions={seeded} status=OK", log)

    _progress("Step 2/5 Fund Registry", log)
    fund_df, fund_source, raw_fund_rows, fund_rows = prepare_fund_registry(engine, alias_map, token, log=log)
    _progress(f"  source={fund_source} raw_rows={raw_fund_rows} matched_rows={len(fund_df)} saved_rows={fund_rows}", log)
    if dry_run:
        return {"status": "DRY_RUN", "report_period": report_period.isoformat(), "fund_rows": int(fund_rows)}
    upsert_freshness(engine, "cn_gcrl_fund_registry", report_period, fund_source, "OK" if fund_rows > 0 else "EMPTY", fund_rows)
    upsert_source_status(engine, fund_source.upper(), "FUND_REGISTRY", "OK" if fund_rows > 0 else "EMPTY", row_count=fund_rows)

    _progress("Step 3/5 Holdings", log)
    pos_df, stats = fetch_eastmoney_fund_holdings(fund_df, report_period, max_funds=max_funds, sleep_seconds=eastmoney_sleep_seconds, log=log)
    snapshot_rows = replace_position_snapshot(engine, report_period, pos_df, source=SOURCE_EASTMONEY_HOLDING)
    holding_status = "OK" if snapshot_rows > 0 else "EMPTY"
    _progress(f"  holdings_rows={snapshot_rows} stats={stats}", log)
    upsert_freshness(engine, "cn_gcrl_position_snapshot", report_period, SOURCE_EASTMONEY_HOLDING, holding_status, snapshot_rows, str(stats))
    upsert_source_status(engine, SOURCE_EASTMONEY_HOLDING.upper(), "HOLDING_SOURCE", holding_status, message=str(stats), row_count=snapshot_rows)

    if snapshot_rows <= 0 and not allow_empty:
        raise GcrlDataUnavailable(
            "GCRL CN holding collection returned 0 rows. This is a failed data run; use --allow-empty only for source debugging. "
            f"stats={stats}"
        )

    _progress("Step 4/5 Position Change", log)
    change_rows = build_and_replace_position_change(engine, report_period) if snapshot_rows > 0 else 0
    _progress(f"  change_rows={change_rows}", log)
    upsert_freshness(engine, "cn_gcrl_position_change", report_period, "derived_from_cn_gcrl_position_snapshot", "OK" if change_rows > 0 else "EMPTY", change_rows)

    _progress("Step 5/5 Freshness", log)
    final_status = "OK" if snapshot_rows > 0 else "EMPTY_ALLOWED"
    _progress(f"  status={final_status}", log)
    _progress("COMPLETE", log)
    return {
        "status": final_status,
        "report_period": report_period.isoformat(),
        "seeded_institutions": seeded,
        "fund_source": fund_source,
        "raw_fund_rows": int(raw_fund_rows),
        "fund_registry_rows": int(fund_rows),
        "holding_source_used": SOURCE_EASTMONEY_HOLDING,
        "holding_stats": stats,
        "snapshot_rows": int(snapshot_rows),
        "change_rows": int(change_rows),
    }


def run_gcrl_cn_public_fund_sync(
    *,
    engine,
    report_period: date,
    token: str = "",
    max_funds: int = DEFAULT_MAX_FUNDS,
    eastmoney_sleep_seconds: float = 0.08,
    dry_run: bool = False,
    allow_empty: bool = False,
    log=None,
    **_ignored,
) -> dict:
    return run_one_period(
        engine=engine,
        report_period=report_period,
        token=token,
        max_funds=max_funds,
        eastmoney_sleep_seconds=eastmoney_sleep_seconds,
        dry_run=dry_run,
        allow_empty=allow_empty,
        log=log,
    )


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="GCRL CN public fund collector V2. EastMoney holdings primary; Tushare optional only for registry enrichment.")
    parser.add_argument("--report-period", default="", help="Quarter end YYYYMMDD, e.g. 20250331. Default: latest completed reporting period.")
    parser.add_argument("--year", type=int, default=0, help="Backfill one year: YYYY.")
    parser.add_argument("--start-period", default="", help="Range backfill start quarter end YYYYMMDD.")
    parser.add_argument("--end-period", default="", help="Range backfill end quarter end YYYYMMDD.")
    parser.add_argument("--backfill-all", action="store_true", help=f"Backfill all quarters from {DEFAULT_START_YEAR} to latest completed period.")
    parser.add_argument("--token", default="", help="Optional Tushare token for fund_basic enrichment. Not required.")
    parser.add_argument("--config", default="", help="Optional config file path for token discovery.")
    parser.add_argument("--max-funds", type=int, default=DEFAULT_MAX_FUNDS, help=f"Max funds to crawl per period. Default {DEFAULT_MAX_FUNDS}; use 0 for all selected funds.")
    parser.add_argument("--eastmoney-sleep-seconds", type=float, default=0.08)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-empty", action="store_true", help="Debug only: do not fail when holdings are empty.")
    args = parser.parse_args(argv)

    plan = resolve_period_plan(args)
    token, _ = resolve_tushare_token(args.token, args.config)
    engine = build_engine()
    _progress(f"[GCRL-CN] plan={plan.label} periods={','.join(p.strftime('%Y%m%d') for p in plan.periods)}")
    results = []
    for p in plan.periods:
        result = run_one_period(
            engine=engine,
            report_period=p,
            token=token or "",
            max_funds=max(0, int(args.max_funds)),
            eastmoney_sleep_seconds=max(0.0, float(args.eastmoney_sleep_seconds)),
            dry_run=bool(args.dry_run),
            allow_empty=bool(args.allow_empty),
        )
        results.append(result)
    _progress("================================================")
    _progress("GCRL_CN_PUBLIC_FUND_V2 SUMMARY")
    for r in results:
        _progress(str(r))
    _progress("================================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
