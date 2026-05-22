from __future__ import annotations

 
import os
import random
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Set
from datetime import date
from urllib import request as urlrequest

import pandas as pd

#from price_loader import load_index_price_em
from app.tasks.db_accessor import DbAccessor
from app.tasks.state_store import StateStore
from app.utils.wireguard_helper import activate_tunnel, deactivate_tunnel, switch_wire_guard
from app.utils.tushare_pro_client import build_tushare_pro_client
import akshare as ak 
from app.tools.sync_cn_stock_daily_price_from_tushare import resolve_tushare_token, patch_pandas_fillna_method_compat

@dataclass
class IndexLoaderTask:
    name: str = "IndexLoader"

    def _tushare_min_interval_seconds(self) -> float:
        return max(0.0, float(os.getenv("INDEX_TUSHARE_MIN_INTERVAL_SECONDS", "0.8")))

    def _respect_tushare_spacing(self) -> None:
        min_interval = self._tushare_min_interval_seconds()
        now = time.monotonic()
        last_call_at = getattr(self, "_ts_last_call_at", None)
        if last_call_at is not None:
            elapsed = now - last_call_at
            remaining = min_interval - elapsed
            if remaining > 0:
                time.sleep(remaining)
        self._ts_last_call_at = time.monotonic()

    def _init_tushare_if_needed(self, *, force: bool = False) -> None:
        if self.data_source_flag != "tu" and not force:
            return
        if self._ts_token and self._ts_pro is not None:
            return
        patch_pandas_fillna_method_compat()
        token, tried_files = resolve_tushare_token("", "")
        self._ts_token = token.strip() if token else ""
        if not self._ts_token:
            msg = "[INDEX] Tushare token not found"
            if tried_files:
                msg += f"; tried_files={', '.join(str(p) for p in tried_files)}"
            raise RuntimeError(msg)
        timeout_seconds = max(5.0, float(os.getenv("INDEX_TUSHARE_TIMEOUT_SECONDS", os.getenv("TUSHARE_PRO_TIMEOUT_SECONDS", "30"))))
        self._ts_pro = build_tushare_pro_client(self._ts_token, timeout=timeout_seconds)
        self._ts_last_call_at = None

    def run(self, ctx) -> None:
        cfg = ctx.config
        self.log = ctx.log
        self.engine=ctx.engine
        self.data_source_flag = str(getattr(cfg, "data_source_flag", "tu") or "tu").strip().lower()
        self._ts_pro = None
        self._ts_token = None

        db = DbAccessor(ctx.engine, self.log)
        #state = StateStore(cfg.state_dir / "index_scanned.json", cfg.state_dir / "index_failed.json", self.log)

        indices = cfg.index_symbols or []
        if not indices:
            self.log.warning("[INDEX] index_symbols 为空，跳过指数加载")
            return

        self.log.info(f"[START][INDEX] window={cfg.start_date}~{cfg.end_date} | indices={len(indices)} | source={self.data_source_flag}")
        if self.data_source_flag == "tu":
            try:
                self._init_tushare_if_needed()
            except Exception as e:
                raise RuntimeError(f"[INDEX] initialize tushare failed: {e}") from e
        activate_tunnel("cn")

        #scanned = state.load_scanned()
        #failed = state.load_failed()
        tmp_csv_dir = cfg.state_dir / "tmp_csv"

        for i, index_code in enumerate(indices, start=1):
            #if index_code in scanned:
            #    self._log_progress( "INDEX", i, len(indices), index_code, "SKIP scanned")
            #    continue
            #if index_code in failed:
            #    self._log_progress( "INDEX", i, len(indices), index_code, "SKIP failed")
            #    continue

            try:
                existing = db.get_existing_index_days(index_code, cfg.start_date, cfg.end_date)
                df = self._load_index_price_best_effort(index_code, cfg.start_date, cfg.end_date, tmp_csv_dir)
                if df is None or df.empty:
                    raise RuntimeError("empty index data")

                trade_days = set(df["trade_date"].dt.date.unique())
                missing = trade_days - existing
                if not missing:
                    #scanned.add(index_code)
                    self._log_progress( "INDEX", i, len(indices), index_code, "OK already complete")
                    continue

                self._log_progress( "INDEX", i, len(indices), index_code, f"INSERT missing_days={len(missing)}")
                inserted = db.insert_missing_index_days(index_code, df, missing)

                remaining = trade_days - db.get_existing_index_days(index_code, cfg.start_date, cfg.end_date)
                if remaining:
                    raise RuntimeError(f"still missing {len(remaining)} index days")

                #scanned.add(index_code)
                self._log_progress( "INDEX", i, len(indices), index_code, f"FIXED inserted={inserted}")

            except Exception as e:
                #failed.add(index_code)
                self.log.info(f"FAILED index {index_code}: {e}")
                #sys.rasie(e)
                #try:
                #    switch_wire_guard("cn")
                #except Exception:
                #    pass
                #self._log_progress(  "INDEX", i, len(indices), index_code, f"FAILED {e}")

            time.sleep(random.uniform(0.3, 0.8))

        #state.save_scanned(scanned)
        #state.save_failed(failed)
        self.log.info("[DONE][INDEX] loader finished")
        if hasattr(self._ts_pro, "close"):
            self._ts_pro.close()

    def repair_specific_dates(self, ctx, index_code: str, dates: list[date]) -> int:
        if not dates:
            return 0
        self.log = ctx.log
        self.engine = ctx.engine
        self.data_source_flag = str(getattr(ctx.config, "data_source_flag", "tu") or "tu").strip().lower()
        self._ts_pro = None
        self._ts_token = None
        # Targeted index repair must use Tushare index_daily as the canonical source.
        # This avoids the deprecated Sohu path and keeps local index symbols aligned
        # with cn_index_daily_price.index_code while querying Tushare ts_code.
        self._init_tushare_if_needed(force=True)
        db = DbAccessor(ctx.engine, self.log)
        tmp_csv_dir = ctx.config.state_dir / "tmp_csv"
        local_index_code, ts_code = self._normalize_index_symbol_pair(index_code)
        self.log.info("[INDEX][REPAIR][TU] symbol align requested=%s local=%s tushare=%s", index_code, local_index_code, ts_code)
        inserted_total = 0
        for dt_value in sorted(set(dates)):
            date8 = pd.to_datetime(dt_value).strftime("%Y%m%d")
            existing = db.get_existing_index_days(local_index_code, date8, date8)
            if dt_value in existing:
                continue
            df = self._load_index_price_best_effort(local_index_code, date8, date8, tmp_csv_dir)
            if df is None or df.empty:
                self.log.warning("[INDEX][REPAIR] %s %s no data returned from Tushare index_daily", local_index_code, date8)
                continue
            inserted_total += db.insert_missing_index_days(local_index_code, df, {dt_value})
        if hasattr(self._ts_pro, "close"):
            self._ts_pro.close()
        return inserted_total

    # ------------------------
    def _load_index_price_with_failover(self,   index_code: str, start_date: str, end_date: str) -> pd.DataFrame | None:
        if self.data_source_flag == "tu":
            return self._load_index_price_tushare(index_code, start_date, end_date)
        return self.load_index_price_em(index_code=index_code, start_date=start_date, end_date=end_date)

    def _load_index_price_best_effort(self, index_code: str, start_date: str, end_date: str, tmp_csv_dir: Path) -> pd.DataFrame | None:
        """Load index daily price using Tushare index_daily as the canonical source.

        Local DB symbols stay unchanged, e.g. sh000001 / sz399001 / sh000688.
        Only the outbound query symbol is converted to Tushare ts_code, e.g.
        000001.SH / 399001.SZ / 000688.SH. Sohu is intentionally not used.
        """
        primary = None
        secondary = None

        local_code, ts_code = self._normalize_index_symbol_pair(index_code)
        self.log.info(
            "[INDEX][TU] load index_code=%s normalized_local=%s ts_code=%s start=%s end=%s",
            index_code,
            local_code,
            ts_code,
            start_date,
            end_date,
        )

        try:
            # Always prefer Tushare for index load / targeted repair.
            primary = self._load_index_price_tushare(index_code, start_date, end_date)
        except Exception as exc:
            self.log.warning("[INDEX][TU] load failed for %s: %s", index_code, exc)

        # Optional fallback kept disabled by default to keep index repair deterministic.
        # Enable only if a temporary Tushare outage blocks urgent backfill.
        allow_em_fallback = str(os.getenv("INDEX_ALLOW_EM_FALLBACK", "0")).strip().lower() in {"1", "true", "yes", "y"}
        if allow_em_fallback:
            try:
                secondary = self.load_index_price_em(index_code=index_code, start_date=start_date, end_date=end_date)
            except Exception as exc:
                self.log.warning("[INDEX] EM fallback load failed for %s: %s", index_code, exc)

        merged = self._merge_index_frames(primary, secondary, prefer="primary")
        if merged is None or merged.empty:
            return None
        return merged

    @staticmethod
    def _normalize_index_symbol_pair(index_code: str) -> tuple[str, str]:
        """Return (local_index_code, tushare_ts_code).

        Local DB code examples:
            sh000001 -> 000001.SH
            sz399001 -> 399001.SZ
            sh000688 -> 000688.SH

        If a caller already passes Tushare style, keep the Tushare code and infer
        the local code used by cn_index_daily_price.
        """
        raw = str(index_code or "").strip()
        if not raw:
            raise ValueError("index_code must be non-empty")

        upper = raw.upper()
        lower = raw.lower()

        if "." in upper:
            code6, suffix = upper.split(".", 1)
            suffix = suffix[:2]
            if suffix not in {"SH", "SZ"}:
                raise ValueError(f"unsupported Tushare index suffix: {raw}")
            local = ("sh" if suffix == "SH" else "sz") + code6.zfill(6)
            return local, f"{code6.zfill(6)}.{suffix}"

        if lower.startswith(("sh", "sz")) and len(lower) >= 8:
            prefix = lower[:2]
            code6 = lower[2:8]
            suffix = "SH" if prefix == "sh" else "SZ"
            return f"{prefix}{code6}", f"{code6}.{suffix}"

        code6 = lower[-6:].zfill(6)
        # A-share index convention: 399xxx is SZ; most 000xxx/000688/000300/000905/000852 are SH.
        suffix = "SZ" if code6.startswith("399") else "SH"
        local = ("sh" if suffix == "SH" else "sz") + code6
        return local, f"{code6}.{suffix}"

    def _merge_index_frames(self, primary: pd.DataFrame | None, secondary: pd.DataFrame | None, prefer: str = "primary") -> pd.DataFrame | None:
        frames: list[pd.DataFrame] = []
        for frame in [primary, secondary]:
            if frame is None or frame.empty:
                continue
            work = frame.copy()
            work["trade_date"] = pd.to_datetime(work["trade_date"], errors="coerce")
            frames.append(work)
        if not frames:
            return None
        if len(frames) == 1:
            return frames[0].sort_values("trade_date").reset_index(drop=True)

        keep_cols = ["trade_date", "open", "close", "high", "low", "volume", "amount", "source", "pre_close", "chg_pct"]
        priority = [0, 1] if prefer == "primary" else [1, 0]
        combined = pd.concat([frames[i] for i in priority], ignore_index=True)
        combined = combined.sort_values("trade_date").drop_duplicates(subset=["trade_date"], keep="first").reset_index(drop=True)
        for col in keep_cols:
            if col not in combined.columns:
                combined[col] = None
        return combined[keep_cols].dropna(subset=["trade_date", "close"]).copy()

    def _load_index_price_tushare(self, index_code: str, start_date: str, end_date: str) -> pd.DataFrame | None:
        if not self._ts_token or self._ts_pro is None:
            self._init_tushare_if_needed(force=True)

        local_index_code, ts_code = self._normalize_index_symbol_pair(index_code)
        self.log.info("[INDEX][TU] symbol align local=%s tushare=%s", local_index_code, ts_code)
        max_attempts = max(1, int(os.getenv("INDEX_TUSHARE_MAX_ATTEMPTS", "3")))
        retry_sleep_seconds = max(1.0, float(os.getenv("INDEX_TUSHARE_RETRY_SLEEP_SECONDS", "3")))
        df = None
        last_error = None
        for attempt in range(1, max_attempts + 1):
            try:
                self._respect_tushare_spacing()
                df = self._ts_pro.index_daily(
                    ts_code=ts_code,
                    start_date=pd.to_datetime(start_date).strftime("%Y%m%d"),
                    end_date=pd.to_datetime(end_date).strftime("%Y%m%d"),
                )
                break
            except Exception as exc:
                last_error = exc
                if attempt >= max_attempts:
                    raise
                self.log.warning(
                    "[INDEX][TU] %s attempt %s/%s failed: %s; retry in %.1fs",
                    index_code,
                    attempt,
                    max_attempts,
                    exc,
                    retry_sleep_seconds,
                )
                time.sleep(retry_sleep_seconds)
        if df is None or df.empty:
            if last_error is not None:
                self.log.info(last_error)
            return None

        df = df.rename(
            columns={
                "trade_date": "trade_date",
                "pre_close": "pre_close",
                "pct_chg": "chg_pct",
                "vol": "volume",
            }
        )
        df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d", errors="coerce")
        for col in ["open", "close", "high", "low", "pre_close", "volume", "amount", "chg_pct"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.sort_values("trade_date").reset_index(drop=True)
        df["source"] = "tushare:index_daily"

        keep = ["trade_date","open","close","high","low","volume","amount","source","pre_close","chg_pct"]
        for c in keep:
            if c not in df.columns:
                df[c] = None
        out = df[keep].dropna(subset=["trade_date"]).copy()
        quote_cols = ["open", "close", "high", "low", "pre_close", "volume", "amount", "chg_pct"]
        out = out.loc[out[quote_cols].notna().any(axis=1)].copy()
        if out.empty:
            return None
        return out

    def _log_progress(self,  stage: str, cur: int, total: int, symbol: str, msg: str) -> None:
        pct = (cur / total) * 100 if total else 0
        self.log.info(f"[{stage}] ({cur}/{total} | {pct:6.2f}%) {symbol} {msg}")


    @staticmethod
    def _safe_float(v):
        if v is None:
            return None
        s = str(v).strip().replace(",", "")
        if not s or s == "--":
            return None
        s = s.replace("%", "")
        try:
            return float(s)
        except Exception:
            return None

    @staticmethod
    def _extract_jsonp_payload(text: str):
        raw = (text or "").strip()
        if not raw:
            return None
        if raw.startswith("[") or raw.startswith("{"):
            return json.loads(raw)
        m = re.search(r"^[^(]*\((.*)\)\s*;?\s*$", raw, flags=re.S)
        if not m:
            return None
        return json.loads(m.group(1))

    def _load_sh000688_from_sohu_csv(
        self,
        index_code: str,
        start_date: str,
        end_date: str,
        tmp_csv_dir: Path,
    ) -> pd.DataFrame | None:
        code6 = str(index_code).strip().lower()
        if code6.startswith(("sh", "sz")):
            code6 = code6[2:]
        start_fmt = pd.to_datetime(start_date).strftime("%Y%m%d")
        end_fmt = pd.to_datetime(end_date).strftime("%Y%m%d")

        page_url = f"https://q.stock.sohu.com/zs/{code6}/lshq.shtml"
        api_url = (
            "https://q.stock.sohu.com/hisHq"
            f"?code=zs_{code6}&start={start_fmt}&end={end_fmt}"
            "&stat=1&order=D&period=d&callback=historySearchHandler&rt=jsonp"
        )
        mode = "single-day" if start_fmt == end_fmt else "multi-day"
        self.log.info(f"[INDEX][SOHU] {index_code} {mode} fetch from {page_url}")

        req = urlrequest.Request(
            api_url,
            headers={
                "User-Agent": "Mozilla/5.0",
                "Referer": page_url,
            },
        )
        with urlrequest.urlopen(req, timeout=20) as resp:
            body = resp.read().decode("utf-8", errors="ignore")
        payload = self._extract_jsonp_payload(body)
        if not payload:
            return None
        data_obj = payload[0] if isinstance(payload, list) and payload else payload
        hq_rows = data_obj.get("hq", []) if isinstance(data_obj, dict) else []
        if not hq_rows:
            return None

        rows = []
        for r in hq_rows:
            if not isinstance(r, list) or len(r) < 3:
                continue
            trade_date = pd.to_datetime(str(r[0]), errors="coerce")
            open_v = self._safe_float(r[1] if len(r) > 1 else None)
            close_v = self._safe_float(r[2] if len(r) > 2 else None)
            chg_pct_v = self._safe_float(r[4] if len(r) > 4 else None)
            low_v = self._safe_float(r[5] if len(r) > 5 else None)
            high_v = self._safe_float(r[6] if len(r) > 6 else None)
            volume_v = self._safe_float(r[7] if len(r) > 7 else None)
            amount_v = self._safe_float(r[8] if len(r) > 8 else None)
            rows.append(
                {
                    "trade_date": trade_date,
                    "open": open_v,
                    "close": close_v,
                    "high": high_v,
                    "low": low_v,
                    "volume": volume_v,
                    "amount": amount_v,
                    "chg_pct": chg_pct_v,
                    "source": "sohu",
                }
            )
        if not rows:
            return None

        df = pd.DataFrame(rows).dropna(subset=["trade_date", "close"]).copy()
        if df.empty:
            return None
        df = df.sort_values("trade_date").reset_index(drop=True)
        df["pre_close"] = df["close"].shift(1)
        calc_chg_pct = ((df["close"] - df["pre_close"]) / df["pre_close"] * 100).round(4)
        df["chg_pct"] = df["chg_pct"].where(df["chg_pct"].notna(), calc_chg_pct)
        if not df.empty:
            df.loc[0, "pre_close"] = None

        tmp_csv_dir.mkdir(parents=True, exist_ok=True)
        csv_path = tmp_csv_dir / f"{index_code}_{start_fmt}_{end_fmt}_sohu.csv"
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        self.log.info(f"[INDEX][SOHU] temp csv saved: {csv_path}")

        csv_df = pd.read_csv(csv_path)
        csv_df["trade_date"] = pd.to_datetime(csv_df["trade_date"], errors="coerce")
        for col in ["open", "close", "high", "low", "volume", "amount", "pre_close", "chg_pct"]:
            if col in csv_df.columns:
                csv_df[col] = pd.to_numeric(csv_df[col], errors="coerce")

        keep = ["trade_date", "open", "close", "high", "low", "volume", "amount", "source", "pre_close", "chg_pct"]
        for c in keep:
            if c not in csv_df.columns:
                csv_df[c] = None
        return csv_df[keep].dropna(subset=["trade_date", "close"]).copy()




    def load_index_price_em( self,
        index_code: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame | None:
        """
        指数行情（日线，东财；AKShare: stock_zh_index_daily_em）
    
        AK 返回列（英文）:
            date, open, close, high, low, volume, amount
    
        DB DDL 对齐（并保留兼容列）:
            trade_date, open, close, high, low, volume, amount, pre_close, chg_pct
    
        新增：3 次重试机制
        """
        if not isinstance(index_code, str) or not index_code.strip():
            raise ValueError("index_code must be non-empty str")
    
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            deactivate_tunnel("cn")
            #LOG.info("deactivate_tunnel - in load_index_price_em")
             
            #activate_tunnel("cn")
            try:
                df = ak.stock_zh_index_daily_em(
                    symbol=index_code,
                    start_date=start_date,
                    end_date=end_date,
                )
    
                if df is None or df.empty:
                    return None
    
                df = df.rename(columns={"date": "trade_date"})
    
                df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
                for col in ["open", "close", "high", "low", "volume", "amount"]:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors="coerce")
    
                df = df.dropna(subset=["trade_date", "close"]).copy()
                df = df.sort_values("trade_date").reset_index(drop=True)
    
                df["pre_close"] = df["close"].shift(1)
                df["chg_pct"] = ((df["close"] - df["pre_close"]) / df["pre_close"] * 100).round(4)
                df.loc[0, ["pre_close", "chg_pct"]] = None, None
    
                keep_cols = [
                    "trade_date",
                    "open",
                    "close",
                    "high",
                    "low",
                    "volume",
                    "amount",
                    "pre_close",
                    "chg_pct",
                ]
                for c in keep_cols:
                    if c not in df.columns:
                        df[c] = None
                out = df[keep_cols].dropna(subset=["trade_date"]).copy()
                quote_cols = ["open", "close", "high", "low", "pre_close", "volume", "amount", "chg_pct"]
                out = out.loc[out[quote_cols].notna().any(axis=1)].copy()
                if out.empty:
                    if attempt == max_retries:
                        return None
                    time.sleep(1.0)
                    continue
                return out
    
            except Exception as e:
                if attempt == max_retries:
                    self.log.info(e)
                    raise RuntimeError(
                        f"获取指数 {index_code} 日线数据失败（已重试 {max_retries} 次）：{str(e)}"
                    ) from e
                #switch_wire_guard("cn")
                #LOG.info("deactivate_tunnel - in load_index_price_em")
                #time.sleep(60)
                #activate_tunnel("cn")
                #LOG.info("activate_tunnel - in load_index_price_em")
        
        #   
                
    
        return None


    
