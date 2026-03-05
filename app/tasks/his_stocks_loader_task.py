from __future__ import annotations

import random
import time
from dataclasses import dataclass
from datetime import datetime
from typing import List, Set, Tuple

import akshare as ak
import baostock as bs
import pandas as pd

from app.tasks.db_accessor import DbAccessor
from app.tasks.state_store import StateStore
from app.utils.utils_tools import normalize_stock_code


@dataclass
class HisStocksLoaderTask:
    name: str = "HisStocksLoader"

    def run(self, ctx) -> None:
        cfg = ctx.config
        self.log = ctx.log
        db = DbAccessor(ctx.engine, self.log)
        state = StateStore(cfg.his_stock_scanned_file, cfg.his_stock_failed_file, self.log)
        self.source_order = self._parse_source_order(getattr(cfg, "his_source_order", "baostock,ak"))
        # Force-disable AK in current environment due to persistent connectivity issues.
        self.source_order = [s for s in self.source_order if s != "ak"]
        if not self.source_order:
            self.source_order = ["baostock"]
        self.max_symbols = max(0, int(getattr(cfg, "his_max_symbols", 0) or 0))
        self.manual_symbols = []
        for s in (getattr(cfg, "his_symbols", []) or []):
            ss = str(s).strip()
            if ss.isdigit() and len(ss) < 6:
                ss = ss.zfill(6)
            self.manual_symbols.append(normalize_stock_code(ss))
        self.source_disabled: Set[str] = set()
        self.source_disabled.add("ak")
        self.source_fail_streak = {"ak": 0, "yf": 0, "baostock": 0}
        self.source_disable_threshold = 10
        self.ignore_state = bool(getattr(cfg, "his_ignore_state", False))
        self.alternate_bs_ak = bool(getattr(cfg, "his_alternate_bs_ak", False))
        self.universe_frequency = str(getattr(cfg, "his_universe_frequency", "monthly") or "monthly").lower()

        start_date = cfg.his_start_date or cfg.start_date
        end_date = cfg.his_end_date or cfg.end_date
        if not start_date or not end_date:
            raise RuntimeError("his_stocks requires start/end date in YYYYMMDD.")
        if start_date > end_date:
            raise RuntimeError(f"invalid history window: start_date={start_date} > end_date={end_date}")

        self.log.info(f"[START][HIS_STOCKS] window={start_date}~{end_date}")
        self.log.info(f"[SOURCE][HIS_STOCKS] order={','.join(self.source_order)}")
        if self.alternate_bs_ak:
            self.log.info("[SOURCE][HIS_STOCKS] alternate_bs_ak enabled (odd->baostock first, even->ak first)")

        lg = bs.login()
        if lg.error_code != "0":
            raise RuntimeError(f"baostock login failed: {lg.error_msg}")
        self.log.info("baostock login success (historical)")

        try:
            if self.manual_symbols:
                work_symbols = [("HIS", s) for s in self._dedup_preserve_order(self.manual_symbols)]
                self.log.info("[UNIVERSE][HIS_STOCKS] using manual symbols, skip universe scan")
                if self.max_symbols > 0:
                    work_symbols = work_symbols[: self.max_symbols]
                    self.log.info(f"[UNIVERSE][HIS_STOCKS] max_symbols={self.max_symbols} enabled")
                self.log.info(f"[UNIVERSE][HIS_STOCKS] symbols={len(work_symbols)}")
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
        finally:
            try:
                bs.logout()
            except Exception:
                pass
            self.log.info("baostock logout (historical)")

    def get_state_files(self, cfg):
        return [getattr(cfg, "his_stock_scanned_file", None), getattr(cfg, "his_stock_failed_file", None)]

    @staticmethod
    def _parse_source_order(source_order: str) -> List[str]:
        allowed = {"ak", "yf", "baostock"}
        result = []
        for part in (source_order or "").split(","):
            src = part.strip().lower()
            if src in allowed and src not in result:
                result.append(src)
        if not result:
            result = ["ak", "yf", "baostock"]
        return result

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

    def _query_symbols_on_trade_date(self, trade_date_ymd: str) -> Set[str]:
        rs = bs.query_all_stock(trade_date_ymd)
        if rs.error_code != "0":
            self.log.warning(f"query_all_stock failed at {trade_date_ymd}: {rs.error_msg}")
            return set()

        rows = []
        while rs.next():
            rows.append(rs.get_row_data())
        if not rows:
            return set()

        df = pd.DataFrame(rows, columns=rs.fields)
        if "code" not in df.columns:
            return set()

        out: Set[str] = set()
        for raw_code in df["code"].dropna().astype(str):
            code = raw_code.strip().lower()
            if not code:
                continue
            if code.startswith("bj."):
                continue
            if not (code.startswith("sh.") or code.startswith("sz.")):
                continue
            six = normalize_stock_code(code.split(".")[-1])
            if six.startswith(("6", "0", "3")):
                out.add(six)
        return out

    def _get_anchor_trade_dates(self, start_date: str, end_date: str, frequency: str, anchor_pos: str = "end") -> List[str]:
        rs = bs.query_trade_dates(
            start_date=datetime.strptime(start_date, "%Y%m%d").strftime("%Y-%m-%d"),
            end_date=datetime.strptime(end_date, "%Y%m%d").strftime("%Y-%m-%d"),
        )
        if rs.error_code != "0":
            raise RuntimeError(f"query_trade_dates failed: {rs.error_msg}")

        rows = []
        while rs.next():
            rows.append(rs.get_row_data())
        if not rows:
            return []

        df = pd.DataFrame(rows, columns=rs.fields)
        df = df[df["is_trading_day"] == "1"].copy()
        if df.empty:
            return []

        df["calendar_date"] = pd.to_datetime(df["calendar_date"], errors="coerce")
        df = df.dropna(subset=["calendar_date"])
        if frequency == "weekly":
            # Weekly anchor: min/max trading day of each ISO week.
            iso = df["calendar_date"].dt.isocalendar()
            df["yw"] = iso["year"].astype(str) + "-" + iso["week"].astype(str).str.zfill(2)
            agg = "min" if anchor_pos == "start" else "max"
            anchors = df.groupby("yw", as_index=False)["calendar_date"].agg(agg)
            return anchors["calendar_date"].dt.strftime("%Y-%m-%d").tolist()

        # Default monthly anchor: min/max trading day of each month.
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
            self.log.info("[STATE][HIS_STOCKS] ignore_state enabled; scanned/failed bypassed")

        symbols = list(work_symbols)
        total = len(symbols)
        self.log.info(f"[LOAD][HIS_STOCKS] symbols={total} | continue=True")

        for i, (_, symbol) in enumerate(symbols, start=1):
            symbol = self._normalize_symbol_safe(symbol)
            if not symbol:
                self._log_progress(i, total, symbol, "SKIP invalid symbol")
                continue
            if symbol in scanned:
                self._log_progress(i, total, symbol, "SKIP scanned")
                continue
            if symbol in failed:
                self._log_progress(i, total, symbol, "RETRY previous failed")

            try:
                existing_days = db.get_existing_stock_days(symbol, start_date, end_date)
                df, trace = self._load_with_fallback(symbol, start_date, end_date)
                if df is None or df.empty:
                    if self._trace_needs_retry(trace):
                        failed.add(symbol)
                        self._log_progress(i, total, symbol, f"NO DATA(retry) | {trace}")
                    else:
                        failed.discard(symbol)
                        scanned.add(symbol)
                        self._log_progress(i, total, symbol, f"NO DATA(stable) | {trace}")
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
                self.log.info(f"[STATE][HIS_STOCKS] flushed at {i}")

            time.sleep(self._dynamic_sleep_seconds())

        state.save_scanned(scanned)
        state.save_failed(failed)
        self.log.info("[DONE][HIS_STOCKS] loader finished")

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
        if self.ignore_state:
            scanned = set()
            failed = set()
            self.log.info("[STATE][HIS_STOCKS] ignore_state enabled; scanned/failed bypassed")

        anchors = self._get_anchor_trade_dates(
            global_start_date, global_end_date, self.universe_frequency, anchor_pos="start"
        )
        if not anchors:
            raise RuntimeError("No trading dates resolved for historical universe anchors.")

        total_new = 0
        total_anchors = len(anchors)
        for i, anchor_ymd in enumerate(anchors, start=1):
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
            self.log.info(
                f"[UNIVERSE][HIS_STOCKS] anchor={anchor_ymd} ({i}/{total_anchors}) new_symbols={len(pending)} "
                f"load_window={anchor_start}~{global_end_date}"
            )

            self._load_historical_prices(
                db=db,
                state=state,
                work_symbols=[("HIS", s) for s in pending],
                start_date=anchor_start,
                end_date=global_end_date,
                state_flush_every=state_flush_every,
            )

            # Refresh state after each anchor batch to support resume.
            scanned = state.load_scanned()
            failed = state.load_failed()
            if self.ignore_state:
                # keep in-memory strict run semantics while still persisting progress on disk
                scanned = scanned
                failed = failed

        self.log.info(f"[DONE][HIS_STOCKS] anchor-discovery finished | total_new_symbols={total_new}")

    def _load_with_fallback(self, stock_code: str, start_date: str, end_date: str) -> tuple[pd.DataFrame | None, str]:
        ordered_sources = self._ordered_sources_for_symbol(stock_code)
        attempts: List[str] = []
        for src in ordered_sources:
            if src in self.source_disabled:
                attempts.append(f"{src}:disabled")
                continue
            try:
                if src == "ak":
                    df = self._load_from_ak(stock_code, start_date, end_date)
                elif src == "yf":
                    df = self._load_from_yf(stock_code, start_date, end_date)
                else:
                    df = self._load_from_baostock(stock_code, start_date, end_date)
                if df is not None and not df.empty:
                    self.source_fail_streak[src] = 0
                    attempts.append(f"{src}:ok({len(df)})")
                    return df, "; ".join(attempts)
                attempts.append(f"{src}:empty")
                if src == "baostock":
                    self.log.warning(f"[HIS_STOCKS][baostock] {stock_code} empty for {start_date}~{end_date}")
                # Empty response is often legitimate (e.g. not listed yet / non-stock code).
                # Do not count it toward source circuit-breaker.
            except Exception as e:
                self.source_fail_streak[src] += 1
                attempts.append(f"{src}:err({type(e).__name__})")
                if src == "ak" and self._is_ak_hard_failure(e):
                    self.source_disabled.add("ak")
                    self.log.warning("[HIS_STOCKS][ak] disabled for current run due to network/connectivity error")
                self._maybe_disable_source(src)
                self.log.warning(f"[HIS_STOCKS][{src}] {stock_code} failed: {e}")
        return None, "; ".join(attempts) if attempts else "no-source-attempt"

    def _ordered_sources_for_symbol(self, stock_code: str) -> List[str]:
        if not self.alternate_bs_ak:
            return list(self.source_order)

        if "baostock" not in self.source_order or "ak" not in self.source_order:
            return list(self.source_order)

        try:
            is_odd = int(stock_code) % 2 == 1
        except Exception:
            return list(self.source_order)

        first = "baostock" if is_odd else "ak"
        second = "ak" if is_odd else "baostock"
        out = [first, second]
        for src in self.source_order:
            if src not in out:
                out.append(src)
        return out

    def _maybe_disable_source(self, src: str) -> None:
        if src in self.source_disabled:
            return
        if self.source_fail_streak.get(src, 0) >= self.source_disable_threshold:
            self.source_disabled.add(src)
            self.log.warning(
                f"[HIS_STOCKS][{src}] disabled for current run after {self.source_fail_streak[src]} consecutive failures"
            )

    @staticmethod
    def _is_proxy_failure(exc: Exception) -> bool:
        msg = str(exc).lower()
        return ("proxyerror" in msg) or ("127.0.0.1" in msg) or ("port=9" in msg)

    @staticmethod
    def _is_ak_hard_failure(exc: Exception) -> bool:
        msg = str(exc).lower()
        return (
            HisStocksLoaderTask._is_proxy_failure(exc)
            or ("remotedisconnected" in msg)
            or ("connection aborted" in msg)
        )

    @staticmethod
    def _trace_needs_retry(trace: str) -> bool:
        t = (trace or "").lower()
        return ("err(" in t) or ("disabled" in t) or ("no-source-attempt" in t)

    @staticmethod
    def _normalize_symbol_safe(symbol: object) -> str:
        s = str(symbol).strip() if symbol is not None else ""
        if not s:
            return ""
        if s.isdigit() and len(s) < 6:
            s = s.zfill(6)
        if len(s) == 6 and s.isdigit():
            return s
        return ""

    def _dynamic_sleep_seconds(self) -> float:
        # Use a conservative global throttle to reduce remote disconnect/rate-limit.
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
        df = df.rename(columns=rename_map)
        if "trade_date" not in df.columns:
            return None

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

    def _load_from_yf(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame | None:
        try:
            import yfinance as yf
        except Exception:
            return None

        if stock_code.startswith(("6", "9")):
            ticker = f"{stock_code}.SS"
        elif stock_code.startswith(("0", "3")):
            ticker = f"{stock_code}.SZ"
        else:
            return None

        start = datetime.strptime(start_date, "%Y%m%d")
        end_exclusive = datetime.strptime(end_date, "%Y%m%d") + pd.Timedelta(days=1)
        hist = yf.download(ticker, start=start.strftime("%Y-%m-%d"), end=end_exclusive.strftime("%Y-%m-%d"), interval="1d", auto_adjust=False, progress=False)
        if hist is None or hist.empty:
            return None

        hist = hist.reset_index()
        date_col = "Date" if "Date" in hist.columns else hist.columns[0]
        df = pd.DataFrame()
        df["trade_date"] = pd.to_datetime(hist[date_col], errors="coerce")
        df["open"] = pd.to_numeric(hist.get("Open"), errors="coerce")
        df["high"] = pd.to_numeric(hist.get("High"), errors="coerce")
        df["low"] = pd.to_numeric(hist.get("Low"), errors="coerce")
        df["close"] = pd.to_numeric(hist.get("Close"), errors="coerce")
        df["volume"] = pd.to_numeric(hist.get("Volume"), errors="coerce")
        df["amount"] = None
        df = df.dropna(subset=["trade_date"]).sort_values("trade_date").reset_index(drop=True)
        df["pre_close"] = df["close"].shift(1)
        df["chg_pct"] = (df["close"] / df["pre_close"] - 1.0) * 100.0
        df["change"] = df["close"] - df["pre_close"]
        df["amplitude"] = (df["high"] - df["low"]) / df["pre_close"] * 100.0
        df["turnover_rate"] = None
        df["source"] = "yf_hist"

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
        return df[keep]

    def _load_from_baostock(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame | None:
        symbol = f"sh.{stock_code}" if stock_code.startswith(("6", "9")) else f"sz.{stock_code}"
        fields = "date,open,high,low,close,preclose,volume,amount,turn"
        rs = bs.query_history_k_data_plus(
            symbol,
            fields=fields,
            start_date=datetime.strptime(start_date, "%Y%m%d").strftime("%Y-%m-%d"),
            end_date=datetime.strptime(end_date, "%Y%m%d").strftime("%Y-%m-%d"),
            frequency="d",
            adjustflag="2",
        )
        if rs.error_code != "0":
            return None

        rows = []
        while rs.next():
            rows.append(rs.get_row_data())
        if not rows:
            return None

        df = pd.DataFrame(rows, columns=fields.split(","))
        df["trade_date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["trade_date"]).copy()
        df.rename(columns={"preclose": "pre_close", "turn": "turnover_rate"}, inplace=True)
        for col in ["open", "high", "low", "close", "pre_close", "volume", "amount", "turnover_rate"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df["chg_pct"] = (df["close"] / df["pre_close"] - 1.0) * 100.0
        df["change"] = df["close"] - df["pre_close"]
        df["amplitude"] = (df["high"] - df["low"]) / df["pre_close"] * 100.0
        df["source"] = "baostock_hist"

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
        return df[keep]

    def _log_progress(self, cur: int, total: int, symbol: str, msg: str) -> None:
        pct = (cur / total) * 100 if total else 0.0
        sym = symbol if symbol else "<EMPTY_SYMBOL>"
        self.log.info(f"[HIS_STOCKS] ({cur}/{total} | {pct:6.2f}%) {sym} {msg}")
