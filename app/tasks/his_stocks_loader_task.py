from __future__ import annotations

import random
import time
from dataclasses import dataclass
from datetime import date, datetime
from typing import Iterable, List, Set, Tuple

import akshare as ak
import pandas as pd
import tushare as ts

from app.tasks.db_accessor import DbAccessor
from app.tasks.state_store import StateStore
from app.tools.sync_cn_stock_daily_price_from_tushare import resolve_tushare_token
from app.utils.utils_tools import normalize_stock_code


@dataclass
class HisStocksLoaderTask:
    name: str = "HisStocksLoader"

    def run(self, ctx) -> None:
        cfg = ctx.config
        self.log = ctx.log
        db = DbAccessor(ctx.engine, self.log)
        state = StateStore(cfg.his_stock_scanned_file, cfg.his_stock_failed_file, self.log)
        self.data_source_flag = str(getattr(cfg, "data_source_flag", "tu") or "tu").strip().lower()
        self.source_order = ["tu"] if self.data_source_flag == "tu" else ["ak"]
        self.max_symbols = max(0, int(getattr(cfg, "his_max_symbols", 0) or 0))
        self.manual_symbols = []
        for s in (getattr(cfg, "his_symbols", []) or []):
            ss = str(s).strip()
            if ss.isdigit() and len(ss) < 6:
                ss = ss.zfill(6)
            self.manual_symbols.append(normalize_stock_code(ss))
        self.source_disabled: Set[str] = set()
        self.source_fail_streak = {"ak": 0, "tu": 0}
        self.source_disable_threshold = 10
        self.ignore_state = bool(getattr(cfg, "his_ignore_state", False))
        self.universe_frequency = str(getattr(cfg, "his_universe_frequency", "monthly") or "monthly").lower()
        self._ts_pro = None
        self._ts_code_by_symbol = {}
        if self.data_source_flag == "tu":
            self._init_tushare_universe()

        start_date = cfg.his_start_date or cfg.start_date
        end_date = cfg.his_end_date or cfg.end_date
        if not start_date or not end_date:
            raise RuntimeError("his_stocks requires start/end date in YYYYMMDD.")
        if start_date > end_date:
            raise RuntimeError(f"invalid history window: start_date={start_date} > end_date={end_date}")

        self.log.info(f"[START][HIS_STOCKS] window={start_date}~{end_date} source={self.data_source_flag}")
        if self.manual_symbols:
            work_symbols = [("HIS", s) for s in self._dedup_preserve_order(self.manual_symbols)]
            if self.max_symbols > 0:
                work_symbols = work_symbols[: self.max_symbols]
            self._load_historical_prices(
                db=db,
                state=state,
                work_symbols=work_symbols,
                start_date=start_date,
                end_date=end_date,
                state_flush_every=cfg.state_flush_every,
            )
        else:
            self._load_historical_prices_by_anchor_discovery(
                db=db,
                state=state,
                global_start_date=start_date,
                global_end_date=end_date,
                state_flush_every=cfg.state_flush_every,
            )
        self.log.info("[DONE][HIS_STOCKS] run finished")

    def get_state_files(self, cfg):
        return [getattr(cfg, "his_stock_scanned_file", None), getattr(cfg, "his_stock_failed_file", None)]

    @staticmethod
    def _dedup_preserve_order(items: List[str]) -> List[str]:
        out = []
        seen = set()
        for x in items:
            if x in seen:
                continue
            seen.add(x)
            out.append(x)
        return out

    def _init_tushare_universe(self) -> None:
        token, tried_files = resolve_tushare_token("", "")
        if not token:
            msg = "[HIS_STOCKS] Tushare token missing"
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
        self.log.info(f"[HIS_STOCKS] tushare universe ready symbols={len(self._ts_code_by_symbol)}")

    def _query_symbols_on_trade_date(self, trade_date_ymd: str) -> Set[str]:
        try:
            df = ak.stock_zh_a_spot_em()
            if df is None or df.empty:
                return set()
            code_col = "代码" if "代码" in df.columns else "浠ｇ爜"
            out: Set[str] = set()
            for raw_code in df[code_col].dropna().astype(str):
                code = raw_code.strip().lower()
                if code.startswith("bj"):
                    continue
                six = normalize_stock_code(code[2:] if (code.startswith("sh") or code.startswith("sz")) else code)
                if six.startswith(("6", "0", "3")):
                    out.add(six)
            return out
        except Exception as e:
            self.log.warning(f"stock_zh_a_spot_em failed: {e}")
            return set()

    def _get_anchor_trade_dates(self, start_date: str, end_date: str, frequency: str, anchor_pos: str = "end") -> List[str]:
        start_dt = datetime.strptime(start_date, "%Y%m%d").date()
        end_dt = datetime.strptime(end_date, "%Y%m%d").date()
        td_df = ak.tool_trade_date_hist_sina()
        if td_df is None or td_df.empty:
            raise RuntimeError("tool_trade_date_hist_sina returned empty")
        col = td_df.columns[0]

        def to_date(v):
            if isinstance(v, date):
                return v
            if hasattr(v, "date"):
                return v.date()
            s = str(v).strip()
            if len(s) == 10:
                return date.fromisoformat(s)
            return date(int(s[:4]), int(s[4:6]), int(s[6:8]))

        all_dates = sorted([to_date(v) for v in td_df[col].tolist()])
        filtered = [d for d in all_dates if start_dt <= d <= end_dt]
        if not filtered:
            return []

        df = pd.DataFrame({"calendar_date": pd.to_datetime(filtered)})
        if frequency == "weekly":
            iso = df["calendar_date"].dt.isocalendar()
            df["yw"] = iso["year"].astype(str) + "-" + iso["week"].astype(str).str.zfill(2)
            agg = "min" if anchor_pos == "start" else "max"
            anchors = df.groupby("yw", as_index=False)["calendar_date"].agg(agg)
            return anchors["calendar_date"].dt.strftime("%Y-%m-%d").tolist()

        df["ym"] = df["calendar_date"].dt.strftime("%Y-%m")
        agg = "min" if anchor_pos == "start" else "max"
        anchors = df.groupby("ym", as_index=False)["calendar_date"].agg(agg)
        return anchors["calendar_date"].dt.strftime("%Y-%m-%d").tolist()

    def _load_historical_prices(
        self,
        db: DbAccessor,
        state: StateStore,
        work_symbols: Iterable[Tuple[str, str]],
        start_date: str,
        end_date: str,
        state_flush_every: int,
    ) -> None:
        scanned: Set[str] = state.load_scanned()
        failed: Set[str] = state.load_failed()
        if self.ignore_state:
            scanned = set()
            failed = set()

        symbols = list(work_symbols)
        total = len(symbols)
        for i, (_, symbol) in enumerate(symbols, start=1):
            symbol = self._normalize_symbol_safe(symbol)
            if not symbol:
                self._log_progress(i, total, symbol, "SKIP invalid symbol")
                continue
            if symbol in scanned:
                self._log_progress(i, total, symbol, "SKIP scanned")
                continue

            try:
                existing_days = db.get_existing_stock_days(symbol, start_date, end_date)
                df, trace = self._load_with_flag(symbol, start_date, end_date)
                if df is None or df.empty:
                    failed.add(symbol)
                    self._log_progress(i, total, symbol, f"NO DATA | {trace}")
                    continue

                trade_days = set(df["trade_date"].dt.date.unique())
                missing = trade_days - existing_days
                if not missing:
                    failed.discard(symbol)
                    scanned.add(symbol)
                    self._log_progress(i, total, symbol, "OK already complete")
                    continue

                inserted = db.insert_missing_stock_days(symbol, df, missing, start_date=start_date)
                failed.discard(symbol)
                scanned.add(symbol)
                self._log_progress(i, total, symbol, f"FIXED inserted={inserted}")
            except Exception as e:
                failed.add(symbol)
                self._log_progress(i, total, symbol, f"FAILED {e}")

            if state_flush_every > 0 and i % state_flush_every == 0:
                state.save_scanned(scanned)
                state.save_failed(failed)
            time.sleep(self._dynamic_sleep_seconds())

        state.save_scanned(scanned)
        state.save_failed(failed)

    def _load_historical_prices_by_anchor_discovery(
        self,
        db: DbAccessor,
        state: StateStore,
        global_start_date: str,
        global_end_date: str,
        state_flush_every: int,
    ) -> None:
        scanned: Set[str] = state.load_scanned()
        failed: Set[str] = state.load_failed()
        anchors = self._get_anchor_trade_dates(global_start_date, global_end_date, self.universe_frequency, anchor_pos="start")
        if not anchors:
            raise RuntimeError("No trading dates resolved for historical universe anchors.")

        total_new = 0
        for anchor_ymd in anchors:
            anchor_symbols = self._query_symbols_on_trade_date(anchor_ymd)
            pending = sorted(anchor_symbols - scanned - failed)
            if not pending:
                continue
            if self.max_symbols > 0:
                remain = self.max_symbols - total_new
                if remain <= 0:
                    break
                pending = pending[:remain]
            total_new += len(pending)
            anchor_start = datetime.strptime(anchor_ymd, "%Y-%m-%d").strftime("%Y%m%d")
            self._load_historical_prices(
                db=db,
                state=state,
                work_symbols=[("HIS", s) for s in pending],
                start_date=anchor_start,
                end_date=global_end_date,
                state_flush_every=state_flush_every,
            )

    def _load_with_flag(self, stock_code: str, start_date: str, end_date: str) -> tuple[pd.DataFrame | None, str]:
        try:
            if self.data_source_flag == "tu":
                df = self._load_from_tushare(stock_code, start_date, end_date)
                return df, f"tu:{'ok' if df is not None and not df.empty else 'empty'}"
            df = self._load_from_ak(stock_code, start_date, end_date)
            return df, f"ak:{'ok' if df is not None and not df.empty else 'empty'}"
        except Exception as e:
            return None, f"{self.data_source_flag}:err({type(e).__name__})"

    @staticmethod
    def _normalize_symbol_safe(symbol: object) -> str:
        s = str(symbol).strip() if symbol is not None else ""
        if not s:
            return ""
        if s.isdigit() and len(s) < 6:
            s = s.zfill(6)
        return s if len(s) == 6 and s.isdigit() else ""

    def _dynamic_sleep_seconds(self) -> float:
        return random.uniform(1.0, 2.0)

    def _load_from_ak(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame | None:
        df = ak.stock_zh_a_hist(
            symbol=stock_code,
            period="daily",
            start_date=start_date,
            end_date=end_date,
            adjust="qfq",
        )
        if df is None or df.empty:
            return None
        df = df.rename(
            columns={
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
        )
        df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
        df = df.dropna(subset=["trade_date"]).sort_values("trade_date").reset_index(drop=True)
        for col in ["open", "high", "low", "close", "volume", "amount", "amplitude", "chg_pct", "change", "turnover_rate"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df["pre_close"] = df["close"].shift(1)
        df["source"] = "ak_hist"
        keep = [
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
        ]
        for c in keep:
            if c not in df.columns:
                df[c] = None
        return df[keep]

    def _load_from_tushare(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame | None:
        ts_code = self._ts_code_by_symbol.get(str(stock_code).strip())
        if not ts_code:
            return None

        df = ts.pro_bar(
            ts_code=ts_code,
            adj="qfq",
            start_date=datetime.strptime(start_date, "%Y%m%d").strftime("%Y%m%d"),
            end_date=datetime.strptime(end_date, "%Y%m%d").strftime("%Y%m%d"),
            factors=["tor"],
            asset="E",
            freq="D",
        )
        if df is None or df.empty:
            return None

        df = df.copy()
        df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d", errors="coerce")
        df["open"] = pd.to_numeric(df.get("open"), errors="coerce")
        df["high"] = pd.to_numeric(df.get("high"), errors="coerce")
        df["low"] = pd.to_numeric(df.get("low"), errors="coerce")
        df["close"] = pd.to_numeric(df.get("close"), errors="coerce")
        df["pre_close"] = pd.to_numeric(df.get("pre_close"), errors="coerce")
        df["volume"] = pd.to_numeric(df.get("vol"), errors="coerce")
        df["amount"] = pd.to_numeric(df.get("amount"), errors="coerce")
        df["chg_pct"] = pd.to_numeric(df.get("pct_chg"), errors="coerce")
        df["change"] = pd.to_numeric(df.get("change"), errors="coerce")
        df["amplitude"] = (df["high"] - df["low"]) / df["pre_close"] * 100.0
        df["turnover_rate"] = pd.to_numeric(df.get("turnover_rate", df.get("tor")), errors="coerce")
        df["source"] = "tushare_qfq"
        df = df.dropna(subset=["trade_date"]).sort_values("trade_date").reset_index(drop=True)
        keep = [
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
        ]
        for c in keep:
            if c not in df.columns:
                df[c] = None
        return df[keep]

    def _log_progress(self, cur: int, total: int, symbol: str, msg: str) -> None:
        pct = (cur / total) * 100 if total else 0.0
        sym = symbol if symbol else "<EMPTY_SYMBOL>"
        self.log.info(f"[HIS_STOCKS] ({cur}/{total} | {pct:6.2f}%) {sym} {msg}")
