from __future__ import annotations

import json
import os
import random
import re
import time
from dataclasses import dataclass
from datetime import date, datetime, time as dt_time, timedelta
from pathlib import Path
from typing import Dict, List, Set, Tuple

import akshare as ak
import pandas as pd
import pytz
import requests
import tushare as ts
from sqlalchemy import text

from app.tasks.db_accessor import DbAccessor, infer_exchange_from_code6
from app.tasks.state_store import StateStore
from app.tools.sync_cn_stock_daily_price_from_tushare import resolve_tushare_token
from app.utils.utils_tools import normalize_stock_code
from app.utils.wireguard_helper import activate_tunnel


SPOT_RENAME_MAP = {
    "名称": "name",
    "最新价": "close",
    "今开": "open",
    "最高": "high",
    "最低": "low",
    "成交量": "volume",
    "成交额": "amount",
    "涨跌幅": "chg_pct",
    "涨跌额": "change",
    "换手率": "turnover_rate",
    "振幅": "amplitude",
    "昨收": "pre_close",
}


def infer_exchange_from_prefixed_code(code: str) -> str | None:
    if not isinstance(code, str):
        return None
    c = code.strip().lower()
    if c.startswith("sh"):
        return "SSE"
    if c.startswith("sz"):
        return "SZSE"
    if c.startswith("bj"):
        return "BJSE"
    return None


@dataclass
class StockLoaderTask:
    name: str = "StockLoader"

    def __init__(self):
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        self.data_dir = os.path.join(root_dir, "data")
        self.data_source_flag = "tu"
        self._ts_pro = None
        self._ts_code_by_symbol: dict[str, str] = {}

    def run(self, ctx) -> None:
        cfg = ctx.config
        self.log = ctx.log
        self.engine = ctx.engine
        self.data_source_flag = str(getattr(cfg, "data_source_flag", "tu") or "tu").strip().lower()
        db = DbAccessor(ctx.engine, self.log)
        state = StateStore(cfg.scanned_file, cfg.failed_file, self.log)

        if self.data_source_flag == "tu":
            self._init_tushare_universe()

        self.log.info(f"[START][STOCK] window={cfg.start_date}~{cfg.end_date} source={self.data_source_flag}")
        if cfg.manual_stock_symbols:
            work_symbols = [(s, s) for s in sorted(set(cfg.manual_stock_symbols))]
            self.log.info(f"[CONFIG][STOCK] manual symbols={len(work_symbols)}")
        else:
            work_symbols = self._get_all_symbols_from_spot(use_cache=True)

        stock_daily_mode = str(os.getenv("V8_STOCK_DAILY_MODE", "") or "").strip().lower()
        if stock_daily_mode in ("spot", "spot_repair") and self.data_source_flag == "tu":
            repair_lookback_days = int(os.getenv("V8_DAILY_REPAIR_LOOKBACK_DAYS", "15") or "15")
            self.log.info(
                f"[STOCK][TU] mode={stock_daily_mode}; using latest-day smart bulk loader "
                f"(repair_lookback_days={repair_lookback_days})"
            )
            ok = self._load_tushare_latest_day_smart(
                db=db,
                work_symbols=work_symbols,
                trade_date=cfg.end_date,
            )
            if ok:
                self._refresh_stock_active_universe_status()
                return
            self.log.info("[STOCK][TU] spot mode bulk load failed, fallback to standard flow")

        if cfg.look_back_days == 1 and self.data_source_flag == "tu":
            ok = self._load_tushare_latest_day_smart(
                db=db,
                work_symbols=work_symbols,
                trade_date=cfg.end_date,
            )
            if ok:
                self._refresh_stock_active_universe_status()
                return
            self.log.info("[STOCK][TU] smart latest-day load failed, fallback to per-symbol fetch")

        if cfg.look_back_days == 1 and self.data_source_flag == "ak":
            is_intraday, latest_trading_date = self._get_intraday_status_and_last_trade_date()
            if not is_intraday:
                self.log.info(f"[STOCK] use AkShare spot bulk insert for latest trading day {latest_trading_date}")
                ok = self._bulk_insert_latest_day_with_spot(latest_trading_date)
                if ok:
                    self.log.info(f"[STOCK] latest trading day {latest_trading_date} bulk inserted, skip per-symbol fetch")
                    self._refresh_stock_active_universe_status()
                    return
                self.log.info("[STOCK] bulk spot insert failed, fallback to per-symbol history fetch")

        self._load_symbols_days(
            ctx=ctx,
            db=db,
            state=state,
            work_symbols=work_symbols,
            start_date=cfg.start_date,
            end_date=cfg.end_date,
            is_continue_load=True,
            state_flush_every=cfg.state_flush_every,
        )
        self._refresh_stock_active_universe_status()

    def _init_tushare_universe(self) -> None:
        token, tried_files = resolve_tushare_token("", "")
        if not token:
            msg = "[STOCK] Tushare token missing"
            if tried_files:
                msg += f"; tried_files={', '.join(str(p) for p in tried_files)}"
            raise RuntimeError(msg)
        self._ts_pro = ts.pro_api(token)
        frames = []
        for st in ("L", "D", "P"):
            df = self._ts_pro.stock_basic(
                exchange="",
                list_status=st,
                fields="ts_code,symbol,name,exchange,list_status",
            )
            if df is not None and not df.empty:
                frames.append(df)
        if frames:
            uni = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["symbol"])
            self._ts_code_by_symbol = {
                str(row["symbol"]).strip(): str(row["ts_code"]).strip()
                for _, row in uni.iterrows()
                if str(row.get("symbol", "")).strip() and str(row.get("ts_code", "")).strip()
            }
        self.log.info(f"[STOCK] tushare universe ready symbols={len(self._ts_code_by_symbol)}")

    def _refresh_stock_active_universe_status(self) -> None:
        enabled = str(os.getenv("STOCK_AUTO_REFRESH_ACTIVE_UNIVERSE", "1")).strip().lower()
        if enabled in {"0", "false", "no", "off"}:
            self.log.info("[STOCK] skip active-universe refresh by env STOCK_AUTO_REFRESH_ACTIVE_UNIVERSE")
            return

        recent_days = int(os.getenv("STOCK_ACTIVE_RECENT_DAYS", "30"))
        min_trade_days = int(os.getenv("STOCK_ACTIVE_MIN_TRADE_DAYS", "1"))
        try:
            with self.engine.begin() as conn:
                proc_exists = conn.execute(
                    text(
                        """
                        SELECT COUNT(*)
                        FROM information_schema.routines
                        WHERE routine_schema = DATABASE()
                          AND routine_type = 'PROCEDURE'
                          AND routine_name = 'sp_refresh_stock_universe_status'
                        """
                    )
                ).scalar()
                if int(proc_exists or 0) <= 0:
                    self.log.warning("[STOCK] sp_refresh_stock_universe_status not found, skip active-universe refresh")
                    return

                conn.execute(
                    text(
                        """
                        CALL sp_refresh_stock_universe_status(
                            NULL,
                            :recent_days,
                            :min_trade_days
                        )
                        """
                    ),
                    {"recent_days": recent_days, "min_trade_days": min_trade_days},
                )
                conn.execute(
                    text(
                        """
                        CREATE OR REPLACE
                        ALGORITHM = UNDEFINED VIEW cn_stock_non_active_universe_v AS
                        SELECT
                            s.symbol,
                            s.is_active,
                            s.inactive_reason,
                            s.first_trade_date,
                            s.last_trade_date,
                            s.recent_trade_days,
                            s.updated_at
                        FROM cn_stock_universe_status_t s
                        WHERE IFNULL(s.is_active, 1) = 0
                        """
                    )
                )
            self.log.info(
                f"[STOCK] active-universe refreshed: recent_days={recent_days}, min_trade_days={min_trade_days}"
            )
        except Exception as e:
            self.log.warning(f"[STOCK] active-universe refresh failed: {e}")

    def get_state_files(self, cfg):
        return [getattr(cfg, "scanned_file", None), getattr(cfg, "failed_file", None)]

    def _get_all_symbols_from_spot(self, use_cache: bool = True) -> List[Tuple[str, str]]:
        cache_path = Path(self.data_dir) / "all_symbols_spot.json"
        if use_cache and cache_path.exists():
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list) and data and isinstance(data[0], list):
                    return [(n, s) for n, s in data]
            except Exception:
                pass

        spot_df = self._fetch_spot_df_with_diagnostics()
        if spot_df is None or spot_df.empty:
            raise RuntimeError("ak.stock_zh_a_spot() empty; cannot build universe")
        code_col = "代码" if "代码" in spot_df.columns else "浠ｇ爜"
        name_col = "名称" if "名称" in spot_df.columns else "鍚嶇О"
        spot_df = spot_df[~spot_df[code_col].astype(str).str.startswith("bj")].copy()
        spot_df["symbol"] = spot_df[code_col].astype(str).str.slice(start=2)
        spot_df["name"] = spot_df[name_col]
        out = sorted(
            [(r["name"], r["symbol"]) for _, r in spot_df[["name", "symbol"]].dropna().iterrows()],
            key=lambda x: x[1],
        )

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.log.warning(f"write universe cache failed: {e}")
        return out

    def _probe_sina_spot_response_head(self) -> str:
        try:
            from akshare.stock.cons import zh_sina_a_stock_payload, zh_sina_a_stock_url
        except Exception as e:
            return f"probe_import_failed={e}"

        try:
            payload = zh_sina_a_stock_payload.copy()
            payload.update({"page": "1"})
            resp = requests.get(zh_sina_a_stock_url, params=payload, timeout=15)
            head = (resp.text or "")[:200].encode("unicode_escape", "ignore").decode("ascii", "ignore")
            return (
                f"probe_status={resp.status_code}, "
                f"probe_content_type={resp.headers.get('Content-Type')}, "
                f"probe_body_head={head}"
            )
        except Exception as e:
            return f"probe_request_failed={e}"

    def _fetch_spot_df_with_diagnostics(self) -> pd.DataFrame | None:
        try:
            return ak.stock_zh_a_spot()
        except Exception as e:
            msg = str(e)
            if "<" in msg or "decode value" in msg.lower():
                diag = self._probe_sina_spot_response_head()
                raise RuntimeError(f"ak.stock_zh_a_spot decode failed: {msg}; {diag}") from e
            raise

    def _sanitize_spot_rows_before_insert(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return df
        out = df.copy()
        html_like = re.compile(r"<!doctype|<html|<[^>]+>", flags=re.IGNORECASE)
        for col in out.columns:
            if out[col].dtype == object:
                col_str = out[col].astype(str)
                bad_mask = col_str.str.contains(html_like, na=False)
                if int(bad_mask.sum()) > 0:
                    out.loc[bad_mask, col] = None
        if "symbol" in out.columns:
            sym = out["symbol"].astype(str)
            out = out.loc[sym.str.fullmatch(r"\d{6}")].copy()
        return out

    @staticmethod
    def _yyyymmdd_to_date(value: str) -> date:
        return pd.to_datetime(str(value).replace("-", ""), format="%Y%m%d", errors="raise").date()

    @staticmethod
    def _date_to_yyyymmdd(value) -> str:
        return pd.to_datetime(value).strftime("%Y%m%d")

    def _get_tushare_trade_dates(self, end_date: str, lookback_calendar_days: int = 90) -> List[date]:
        if self._ts_pro is None:
            self._init_tushare_universe()
        end_dt = self._yyyymmdd_to_date(end_date)
        start_dt = end_dt - timedelta(days=lookback_calendar_days)
        cal = self._ts_pro.trade_cal(
            exchange="SSE",
            start_date=start_dt.strftime("%Y%m%d"),
            end_date=end_dt.strftime("%Y%m%d"),
            fields="cal_date,is_open",
        )
        if cal is None or cal.empty:
            raise RuntimeError("tushare trade_cal returned empty")
        cal = cal[pd.to_numeric(cal["is_open"], errors="coerce") == 1].copy()
        cal["trade_date"] = pd.to_datetime(cal["cal_date"], format="%Y%m%d", errors="coerce")
        cal = cal.dropna(subset=["trade_date"])
        return sorted(cal["trade_date"].dt.date.unique().tolist())

    def _fetch_tushare_daily_by_trade_date(self, trade_date: str) -> pd.DataFrame | None:
        if self._ts_pro is None:
            self._init_tushare_universe()
        trade_date8 = pd.to_datetime(str(trade_date).replace("-", ""), format="%Y%m%d", errors="raise").strftime("%Y%m%d")
        df = self._ts_pro.daily(trade_date=trade_date8)
        if df is None or df.empty:
            return None
        out = df.copy()
        out["symbol"] = out["ts_code"].astype(str).str.split(".").str[0].apply(normalize_stock_code)
        out["trade_date"] = pd.to_datetime(out["trade_date"], format="%Y%m%d", errors="coerce")
        out = out.dropna(subset=["trade_date", "symbol"])
        out["open"] = pd.to_numeric(out.get("open"), errors="coerce")
        out["close"] = pd.to_numeric(out.get("close"), errors="coerce")
        out["pre_close"] = pd.to_numeric(out.get("pre_close"), errors="coerce")
        out["high"] = pd.to_numeric(out.get("high"), errors="coerce")
        out["low"] = pd.to_numeric(out.get("low"), errors="coerce")
        out["volume"] = pd.to_numeric(out.get("vol"), errors="coerce")
        out["amount"] = pd.to_numeric(out.get("amount"), errors="coerce")
        out["chg_pct"] = pd.to_numeric(out.get("pct_chg"), errors="coerce")
        out["change"] = pd.to_numeric(out.get("change"), errors="coerce")
        out["turnover_rate"] = None
        out["amplitude"] = ((out["high"] - out["low"]) / out["pre_close"] * 100.0).where(out["pre_close"] > 0)
        out["source"] = "tushare_daily"
        return out

    def _get_latest_existing_stock_dates(self, symbols: Set[str]) -> Dict[str, date | None]:
        symbols = {normalize_stock_code(s) for s in symbols if str(s).strip()}
        result: Dict[str, date | None] = {s: None for s in symbols}
        if not symbols:
            return result
        from sqlalchemy import bindparam
        sql = text("""
            SELECT symbol, MAX(trade_date) AS max_trade_date
            FROM cn_stock_daily_price
            WHERE symbol IN :symbols
            GROUP BY symbol
        """).bindparams(bindparam("symbols", expanding=True))
        with self.engine.begin() as conn:
            rows = conn.execute(sql, {"symbols": sorted(symbols)}).fetchall()
        for sym, max_dt in rows:
            if max_dt is not None:
                result[normalize_stock_code(sym)] = max_dt.date() if hasattr(max_dt, "date") else pd.to_datetime(max_dt).date()
        return result

    def _prepare_tushare_daily_insert_df(
        self,
        daily_df: pd.DataFrame,
        trade_date: str,
        name_by_symbol: Dict[str, str],
        symbols: Set[str] | None = None,
    ) -> pd.DataFrame:
        if daily_df is None or daily_df.empty:
            return pd.DataFrame()
        insert_df = daily_df.copy()
        if symbols is not None:
            symbols = {normalize_stock_code(s) for s in symbols if str(s).strip()}
            insert_df = insert_df[insert_df["symbol"].isin(symbols)].copy()
        if insert_df.empty:
            return pd.DataFrame()

        insert_df["name"] = insert_df["symbol"].map(name_by_symbol).fillna(insert_df["symbol"])
        insert_df["window_start"] = pd.to_datetime(str(trade_date).replace("-", ""), format="%Y%m%d", errors="coerce")
        insert_df["exchange"] = insert_df["symbol"].apply(infer_exchange_from_code6)
        table_cols = [
            "symbol", "trade_date", "open", "close", "pre_close", "high", "low", "volume", "amount",
            "amplitude", "chg_pct", "change", "turnover_rate", "source", "window_start", "exchange", "name",
        ]
        for col in table_cols:
            if col not in insert_df.columns:
                insert_df[col] = None
        insert_df = insert_df[table_cols]

        numeric_cols = ["open", "close", "pre_close", "high", "low", "volume", "amount", "amplitude", "chg_pct", "change", "turnover_rate"]
        for col in numeric_cols:
            if col in insert_df.columns:
                insert_df[col] = pd.to_numeric(insert_df[col], errors="coerce")

        insert_df = DbAccessor._drop_fully_empty_quote_rows(insert_df, id_col="symbol", label="stock")
        return insert_df

    def _insert_tushare_daily_rows(self, daily_df: pd.DataFrame, symbols: Set[str], trade_date: str, name_by_symbol: Dict[str, str]) -> int:
        insert_df = self._prepare_tushare_daily_insert_df(daily_df, trade_date, name_by_symbol, symbols=symbols)
        if insert_df.empty:
            return 0
        insert_df.to_sql("cn_stock_daily_price", self.engine, if_exists="append", index=False, chunksize=500)
        return len(insert_df)

    def _overwrite_tushare_daily_rows(self, daily_df: pd.DataFrame, trade_date: str, name_by_symbol: Dict[str, str]) -> int:
        insert_df = self._prepare_tushare_daily_insert_df(daily_df, trade_date, name_by_symbol, symbols=None)
        if insert_df.empty:
            return 0
        target = self._yyyymmdd_to_date(trade_date)
        with self.engine.begin() as conn:
            conn.execute(text("DELETE FROM cn_stock_daily_price WHERE trade_date = :dt"), {"dt": target})
        insert_df.to_sql("cn_stock_daily_price", self.engine, if_exists="append", index=False, chunksize=1000)
        return len(insert_df)

    def _build_missing_tushare_trade_date_map(
        self,
        missing_symbols: Set[str],
        latest_existing: Dict[str, date | None],
        trade_dates: List[date],
        target_date: date,
    ) -> Dict[date, Set[str]]:
        trade_idx = {d: i for i, d in enumerate(trade_dates)}
        if target_date not in trade_idx:
            return {}
        target_idx = trade_idx[target_date]
        missing_by_date: Dict[date, Set[str]] = {}
        for symbol in missing_symbols:
            sym = normalize_stock_code(symbol)
            last_dt = latest_existing.get(sym)
            if last_dt is None or last_dt not in trade_idx:
                start_idx = target_idx
            else:
                start_idx = min(trade_idx[last_dt] + 1, target_idx)
            for d in trade_dates[start_idx : target_idx + 1]:
                missing_by_date.setdefault(d, set()).add(sym)
        return missing_by_date

    def _load_missing_tushare_trade_dates_bulk(
        self,
        missing_by_date: Dict[date, Set[str]],
        name_by_symbol: Dict[str, str],
    ) -> None:
        if not missing_by_date:
            return
        total = len(missing_by_date)
        for i, dt_value in enumerate(sorted(missing_by_date), start=1):
            trade_date8 = dt_value.strftime("%Y%m%d")
            symbols = missing_by_date[dt_value]
            self.log.info(
                f"[STOCK-TU-DATE] ({i}/{total}) FETCH trade_date={trade_date8} "
                f"reason_missing_symbols={len(symbols)} mode=full_market_overwrite"
            )
            daily_df = self._fetch_tushare_daily_by_trade_date(trade_date8)
            if daily_df is None or daily_df.empty:
                raise RuntimeError(f"[STOCK][TU] tushare daily empty for missing trade_date={trade_date8}")
            inserted = self._overwrite_tushare_daily_rows(daily_df, trade_date8, name_by_symbol)
            self.log.info(f"[STOCK-TU-DATE] ({i}/{total}) FIXED trade_date={trade_date8} overwritten_rows={inserted}")
            time.sleep(random.uniform(0.15, 0.35))

    def _verify_tushare_trade_dates_complete(self, trade_dates: List[date]) -> None:
        for dt_value in sorted(set(trade_dates)):
            trade_date8 = dt_value.strftime("%Y%m%d")
            daily_df = self._fetch_tushare_daily_by_trade_date(trade_date8)
            if daily_df is None or daily_df.empty:
                raise RuntimeError(f"[STOCK][TU] final verify source empty for trade_date={trade_date8}")
            self._verify_tushare_latest_day_complete(daily_df, trade_date8)

    def _verify_tushare_latest_day_complete(self, daily_df: pd.DataFrame, trade_date: str) -> None:
        expected_symbols = set(daily_df["symbol"].dropna().astype(str).map(normalize_stock_code).tolist())
        target = self._yyyymmdd_to_date(trade_date)
        existing_df = pd.read_sql(
            text("SELECT symbol FROM cn_stock_daily_price WHERE trade_date = :dt"),
            self.engine,
            params={"dt": target},
        )
        existing_symbols = set(existing_df["symbol"].astype(str).map(normalize_stock_code).tolist()) if not existing_df.empty else set()
        missing_latest = expected_symbols - existing_symbols
        if missing_latest:
            sample = sorted(missing_latest)[:20]
            raise RuntimeError(f"[STOCK][TU] final verify failed: latest trade_date missing symbols={len(missing_latest)} sample={sample}")
        self.log.info(f"[STOCK][TU] final verify PASS: latest trade_date={target} symbols={len(expected_symbols)}")

    def _load_tushare_latest_day_smart(self, db: DbAccessor, work_symbols: List[Tuple[str, str]], trade_date: str) -> bool:
        try:
            target_date = self._yyyymmdd_to_date(trade_date)
            trade_dates = self._get_tushare_trade_dates(trade_date)
            if not trade_dates or target_date not in trade_dates:
                self.log.info(f"[STOCK][TU] {trade_date} is not an open trading day, skip smart latest-day load")
                return True

            daily_df = self._fetch_tushare_daily_by_trade_date(trade_date)
            if daily_df is None or daily_df.empty:
                raise RuntimeError(f"tushare daily returned empty for trade_date={trade_date}")

            name_by_symbol = {normalize_stock_code(symbol): str(name) for name, symbol in work_symbols}
            expected_symbols = set(daily_df["symbol"].dropna().astype(str).map(normalize_stock_code).tolist())
            existing_latest_df = pd.read_sql(
                text("SELECT symbol FROM cn_stock_daily_price WHERE trade_date = :dt"),
                self.engine,
                params={"dt": target_date},
            )
            existing_latest = set(existing_latest_df["symbol"].astype(str).map(normalize_stock_code).tolist()) if not existing_latest_df.empty else set()
            missing_latest = expected_symbols - existing_latest
            if not missing_latest:
                self.log.info(f"[STOCK][TU] latest trade_date={target_date} already complete; verify only")
                self._verify_tushare_latest_day_complete(daily_df, trade_date)
                return True

            latest_existing = self._get_latest_existing_stock_dates(missing_latest)
            trade_idx = {d: i for i, d in enumerate(trade_dates)}
            target_idx = trade_idx[target_date]
            lagging_symbols: Set[str] = set()
            latest_only_symbols: Set[str] = set()
            for symbol in missing_latest:
                last_dt = latest_existing.get(symbol)
                if last_dt is None or last_dt not in trade_idx or (target_idx - trade_idx[last_dt]) > 1:
                    lagging_symbols.add(symbol)
                else:
                    latest_only_symbols.add(symbol)

            self.log.info(
                f"[STOCK][TU] audit trade_date={target_date}: expected={len(expected_symbols)} "
                f"existing_latest={len(existing_latest)} missing_latest={len(missing_latest)} "
                f"lagging_gt_1_day={len(lagging_symbols)} latest_only={len(latest_only_symbols)}"
            )

            missing_by_date = self._build_missing_tushare_trade_date_map(
                missing_symbols=missing_latest,
                latest_existing=latest_existing,
                trade_dates=trade_dates,
                target_date=target_date,
            )
            if missing_by_date:
                summary = ", ".join(
                    f"{d.strftime('%Y%m%d')}:{len(v)}" for d, v in sorted(missing_by_date.items())
                )
                self.log.info(f"[STOCK][TU] date-first fill missing trade_dates={summary}")
                self._load_missing_tushare_trade_dates_bulk(
                    missing_by_date=missing_by_date,
                    name_by_symbol=name_by_symbol,
                )

            self._verify_tushare_trade_dates_complete(list(missing_by_date.keys()) or [target_date])
            self.log.info("[DONE][STOCK][TU] smart latest-day loader finished")
            return True
        except Exception as e:
            self.log.info(f"[STOCK][TU] smart latest-day load failed: {e}")
            return False

    def _load_symbols_days(
        self,
        ctx,
        db: DbAccessor,
        state: StateStore,
        work_symbols: List[Tuple[str, str]],
        start_date: str,
        end_date: str,
        is_continue_load: bool,
        state_flush_every: int,
    ) -> None:
        activate_tunnel("cn")
        scanned: Set[str] = state.load_scanned() if is_continue_load else set()
        failed: Set[str] = state.load_failed() if is_continue_load else set()
        total = len(work_symbols)
        self.log.info(
            f"[START][STOCK] window={start_date}~{end_date} | symbols={total} | continue={is_continue_load} | source={self.data_source_flag}"
        )

        for i, (name, symbol) in enumerate(work_symbols, start=1):
            if state_flush_every > 0 and i % state_flush_every == 0:
                state.save_scanned(scanned)
                state.save_failed(failed)
                self.log.info(f"[STATE] flushed scanned/failed at {i}")

            symbol6 = normalize_stock_code(symbol)
            if symbol in scanned:
                self._log_progress("STOCK", i, total, symbol, "SKIP scanned")
                continue
            if symbol in failed:
                self._log_progress("STOCK", i, total, symbol, "SKIP failed")
                continue

            self._log_progress("STOCK", i, total, symbol, "CHECK DB coverage")
            try:
                existing_days = db.get_existing_stock_days(symbol6, start_date, end_date)
                df = self._load_stock_price_by_flag(
                    stock_code=symbol6,
                    start_date=start_date,
                    end_date=end_date,
                    name=name,
                    adjust="qfq",
                )
                if df is None or df.empty:
                    raise RuntimeError(f"empty data from source={self.data_source_flag}")

                stock_trade_days = set(df["trade_date"].dt.date.unique())
                missing = stock_trade_days - existing_days
                if not missing:
                    scanned.add(symbol)
                    self._log_progress("STOCK", i, total, symbol, "OK already complete")
                    continue

                self._log_progress("STOCK", i, total, symbol, f"INSERT missing_days={len(missing)}")
                df["name"] = name
                inserted = db.insert_missing_stock_days(symbol6, df, missing, start_date=start_date)
                remaining = stock_trade_days - db.get_existing_stock_days(symbol6, start_date, end_date)
                if remaining:
                    raise RuntimeError(f"still missing {len(remaining)} trade days")

                scanned.add(symbol)
                self._log_progress("STOCK", i, total, symbol, f"FIXED inserted={inserted}")
            except Exception as e:
                failed.add(symbol)
                self.log.info(f"FAILED {symbol}: {e}")
                self._log_progress("STOCK", i, total, symbol, f"FAILED {e}")

            if i % 20 == 0:
                time.sleep(1.0)
            else:
                time.sleep(random.uniform(0.3, 0.8))

        state.save_scanned(scanned)
        state.save_failed(failed)
        self.log.info("[DONE][STOCK] loader finished")

    def _load_stock_price_by_flag(
        self,
        stock_code: str,
        start_date: str,
        end_date: str,
        name: str,
        adjust: str = "qfq",
    ) -> pd.DataFrame | None:
        if self.data_source_flag == "tu":
            return self._load_stock_price_tushare(stock_code, start_date, end_date, name=name)
        return self.load_stock_price_eastmoney(stock_code=stock_code, start_date=start_date, end_date=end_date, name=name, adjust=adjust)

    def _load_stock_price_tushare(
        self,
        stock_code: str,
        start_date: str,
        end_date: str,
        name: str,
    ) -> pd.DataFrame | None:
        ts_code = self._ts_code_by_symbol.get(str(stock_code).strip())
        if not ts_code:
            self.log.warning(f"[STOCK] tushare ts_code not found for {stock_code}")
            return None

        df = ts.pro_bar(
            ts_code=ts_code,
            adj="qfq",
            start_date=pd.to_datetime(start_date).strftime("%Y%m%d"),
            end_date=pd.to_datetime(end_date).strftime("%Y%m%d"),
            factors=["tor"],
            asset="E",
            freq="D",
        )
        if df is None or df.empty:
            return None

        out = df.copy()
        out["trade_date"] = pd.to_datetime(out["trade_date"], format="%Y%m%d", errors="coerce")
        out = out.dropna(subset=["trade_date"]).sort_values("trade_date").reset_index(drop=True)
        out["open"] = pd.to_numeric(out.get("open"), errors="coerce")
        out["close"] = pd.to_numeric(out.get("close"), errors="coerce")
        out["pre_close"] = pd.to_numeric(out.get("pre_close"), errors="coerce")
        out["high"] = pd.to_numeric(out.get("high"), errors="coerce")
        out["low"] = pd.to_numeric(out.get("low"), errors="coerce")
        out["volume"] = pd.to_numeric(out.get("vol"), errors="coerce")
        out["amount"] = pd.to_numeric(out.get("amount"), errors="coerce")
        out["chg_pct"] = pd.to_numeric(out.get("pct_chg"), errors="coerce")
        out["change"] = pd.to_numeric(out.get("change"), errors="coerce")
        out["turnover_rate"] = pd.to_numeric(out.get("turnover_rate", out.get("tor")), errors="coerce")
        out["amplitude"] = ((out["high"] - out["low"]) / out["pre_close"] * 100.0).where(out["pre_close"] > 0)
        out["name"] = name
        out["source"] = "tushare_qfq"

        keep_cols = [
            "trade_date",
            "name",
            "open",
            "close",
            "high",
            "low",
            "volume",
            "amount",
            "amplitude",
            "chg_pct",
            "change",
            "turnover_rate",
            "pre_close",
            "source",
        ]
        for c in keep_cols:
            if c not in out.columns:
                out[c] = None
        return out[keep_cols]

    def _get_intraday_status_and_last_trade_date(self) -> tuple[bool, str]:
        tz = pytz.timezone("Asia/Shanghai")
        now = datetime.now(tz)
        reference_date = (now - timedelta(days=1)).date() if now.hour < 8 else now.date()
        try:
            if self.data_source_flag == "tu":
                if self._ts_pro is None:
                    self._init_tushare_universe()
                trade_df = self._ts_pro.trade_cal(
                    exchange="SSE",
                    start_date=(reference_date - timedelta(days=60)).strftime("%Y%m%d"),
                    end_date=reference_date.strftime("%Y%m%d"),
                    fields="cal_date,is_open",
                )
                if trade_df is None or trade_df.empty:
                    raise RuntimeError("trade_cal returned empty")
                trade_df["calendar_date"] = pd.to_datetime(trade_df["cal_date"], format="%Y%m%d", errors="coerce")
                trade_df = trade_df[pd.to_numeric(trade_df["is_open"], errors="coerce") == 1]
                trading_days = trade_df["calendar_date"].dt.date.values
            else:
                activate_tunnel("cn")
                trade_df = ak.tool_trade_date_hist_sina()
                if trade_df is None or trade_df.empty:
                    raise RuntimeError("tool_trade_date_hist_sina returned empty")
                cal_col = trade_df.columns[0]
                trade_df = trade_df.rename(columns={cal_col: "calendar_date"})
                trade_df["calendar_date"] = pd.to_datetime(trade_df["calendar_date"], errors="coerce")
                trade_df = trade_df.dropna(subset=["calendar_date"])
                window_start = reference_date - timedelta(days=60)
                trading_days = trade_df[
                    (trade_df["calendar_date"].dt.date >= window_start)
                    & (trade_df["calendar_date"].dt.date <= reference_date)
                ]["calendar_date"].dt.date.values

            if len(trading_days) == 0:
                raise RuntimeError("No trading days in last 60 days")
            last_trade_date_obj = max(d for d in trading_days if d <= reference_date)
            last_trade_date_str = last_trade_date_obj.strftime("%Y-%m-%d")
            is_today_trading_day = last_trade_date_obj == now.date()
            in_trading_hours = dt_time(9, 30) <= now.time() < dt_time(15, 0)
            return is_today_trading_day and in_trading_hours, last_trade_date_str
        except Exception as e:
            self.log.info(f"Error in get_intraday_status_and_last_trade_date: {e}")
            is_weekend = reference_date.weekday() >= 5
            in_session = dt_time(9, 30) <= now.time() < dt_time(15, 0)
            fallback_last = reference_date - timedelta(days=(reference_date.weekday() - 4) % 7)
            return (not is_weekend and in_session, fallback_last.strftime("%Y-%m-%d"))

    def _bulk_insert_latest_day_with_spot(self, latest_trading_date: str) -> bool:
        try:
            spot_df = self._fetch_spot_df_with_diagnostics()
            if spot_df is None or spot_df.empty:
                return False

            code_col = "代码" if "代码" in spot_df.columns else "浠ｇ爜"
            spot_df = spot_df[~spot_df[code_col].astype(str).str.startswith("bj")].copy()
            spot_df["symbol"] = spot_df[code_col].astype(str).str.slice(start=2)
            if spot_df.empty:
                return False

            symbols = sorted(spot_df["symbol"].unique().tolist())
            symbols_path = Path("data/symbols.json")
            symbols_path.parent.mkdir(parents=True, exist_ok=True)
            with open(symbols_path, "w", encoding="utf-8") as f:
                json.dump(symbols, f, ensure_ascii=False, indent=2)

            existing_query = text("SELECT symbol FROM cn_stock_daily_price WHERE trade_date = :dt")
            existing_df = pd.read_sql(existing_query, self.engine, params={"dt": pd.to_datetime(latest_trading_date).date()})
            existing_symbols = set(existing_df["symbol"]) if not existing_df.empty else set()
            missing_symbols = set(symbols) - existing_symbols
            if not missing_symbols:
                return True

            missing_df = spot_df[spot_df["symbol"].isin(missing_symbols)].copy()
            missing_df["trade_date"] = pd.to_datetime(latest_trading_date)
            missing_df["exchange"] = missing_df[code_col].apply(infer_exchange_from_prefixed_code)
            missing_df.rename(columns=SPOT_RENAME_MAP, inplace=True)
            for col in [
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
                "name",
                "exchange",
            ]:
                if col not in missing_df.columns:
                    missing_df[col] = None

            missing_df["symbol"] = missing_df["symbol"].apply(normalize_stock_code)
            missing_df["window_start"] = pd.to_datetime(latest_trading_date)
            missing_df["source"] = "ak_spot"
            missing_df["exchange"] = missing_df["exchange"].fillna(missing_df["symbol"].apply(infer_exchange_from_code6))
            table_cols = [
                "symbol",
                "trade_date",
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
                "source",
                "window_start",
                "exchange",
                "name",
            ]
            missing_df = self._sanitize_spot_rows_before_insert(missing_df[table_cols])
            if missing_df.empty:
                return False

            missing_df.to_sql("cn_stock_daily_price", self.engine, if_exists="append", index=False, chunksize=500)
            return True
        except Exception as e:
            self.log.info(f"bulk_insert_latest_day_with_spot failed: {e}")
            return False

    def _log_progress(self, stage: str, cur: int, total: int, symbol: str, msg: str) -> None:
        pct = (cur / total) * 100 if total else 0
        self.log.info(f"[{stage}] ({cur}/{total} | {pct:6.2f}%) {symbol} {msg}")

    def load_stock_price_eastmoney(
        self,
        stock_code: str,
        start_date: str,
        end_date: str,
        name: str,
        adjust: str = "qfq",
    ) -> pd.DataFrame | None:
        df = ak.stock_zh_a_hist(
            symbol=stock_code,
            period="daily",
            start_date=start_date,
            end_date=end_date,
            adjust=adjust,
        )
        if df is None or df.empty:
            return None

        rename_map = {
            "日期": "trade_date",
            "开盘": "open",
            "收盘": "close",
            "最高": "high",
            "最低": "low",
            "成交量": "volume",
            "成交额": "amount",
            "振幅": "amplitude",
            "涨跌幅": "chg_pct",
            "涨跌额": "change",
            "换手率": "turnover_rate",
        }
        legacy_rename_map = {
            "鏃ユ湡": "trade_date",
            "寮€鐩": "open",
            "鏀剁洏": "close",
            "鏈€楂": "high",
            "鏈€浣": "low",
            "鎴愪氦閲": "volume",
            "鎴愪氦棰": "amount",
            "鎸箙": "amplitude",
            "娑ㄨ穼骞": "chg_pct",
            "娑ㄨ穼棰": "change",
            "鎹㈡墜鐜": "turnover_rate",
        }
        df = df.rename(columns=rename_map).rename(columns=legacy_rename_map)
        df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
        for col in ["open", "close", "high", "low", "volume", "amount", "amplitude", "chg_pct", "change", "turnover_rate"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.sort_values("trade_date").reset_index(drop=True)
        df["pre_close"] = df["close"].shift(1)
        df["chg_pct"] = ((df["close"] - df["pre_close"]) / df["pre_close"] * 100).round(4)
        if not df.empty:
            df.loc[0, ["pre_close", "chg_pct"]] = None, None
        df["name"] = name

        keep_cols = [
            "trade_date",
            "name",
            "open",
            "close",
            "high",
            "low",
            "volume",
            "amount",
            "amplitude",
            "chg_pct",
            "change",
            "turnover_rate",
            "pre_close",
        ]
        for c in keep_cols:
            if c not in df.columns:
                df[c] = None
        return df.dropna(subset=["trade_date", "close"])[keep_cols].copy()
