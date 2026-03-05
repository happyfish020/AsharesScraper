from __future__ import annotations

 
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
import baostock as bs

#from price_loader import load_index_price_em
from app.tasks.db_accessor import DbAccessor
from app.tasks.state_store import StateStore
from app.utils.wireguard_helper import activate_tunnel, deactivate_tunnel, switch_wire_guard
import akshare as ak 

@dataclass
class IndexLoaderTask:
    name: str = "IndexLoader"

    def run(self, ctx) -> None:
        cfg = ctx.config
        self.log = ctx.log
        self.engine=ctx.engine

        db = DbAccessor(ctx.engine, self.log)
        #state = StateStore(cfg.state_dir / "index_scanned.json", cfg.state_dir / "index_failed.json", self.log)

        indices = cfg.index_symbols or []
        if not indices:
            self.log.warning("[INDEX] index_symbols 为空，跳过指数加载")
            return

        self.log.info(f"[START][INDEX] window={cfg.start_date}~{cfg.end_date} | indices={len(indices)}")
        activate_tunnel("cn")
        lg = bs.login()
        if lg.error_code != '0':
            self.log.error(f"baostock login failed: {lg.error_msg}")
        else:
            self.log.info("baostock login success (global for index batch)")

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
                 
                if index_code =="sh000688":
                    df = self._load_sh000688_from_sohu_csv(
                        index_code=index_code,
                        start_date=cfg.start_date,
                        end_date=cfg.end_date,
                        tmp_csv_dir=tmp_csv_dir,
                    )
                else:
                    df = self._load_index_price_with_failover( index_code, cfg.start_date, cfg.end_date)
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

        try:
            bs.logout()
        except Exception:
            pass
        self.log.info("baostock logout (index)")

        #state.save_scanned(scanned)
        #state.save_failed(failed)
        self.log.info("[DONE][INDEX] loader finished")

    # ------------------------
    def _load_index_price_with_failover(self,   index_code: str, start_date: str, end_date: str) -> pd.DataFrame | None:
        df = self._load_index_price_baostock_internal( index_code, start_date, end_date)
        if df is not None and not df.empty:
            return df
        self.log.warning(f"baostock failed for index {index_code}, switching to eastmoney")
        return self.load_index_price_em(index_code=index_code, start_date=start_date, end_date=end_date)

    def _load_index_price_baostock_internal(self,   index_code: str, start_date: str, end_date: str) -> pd.DataFrame | None:
        code_clean = str(index_code).strip().lower()
        # accept sh000300/sz399001 OR 000300/399001
        if code_clean.startswith("sh") or code_clean.startswith("sz"):
            code6 = code_clean[2:]
            prefix = code_clean[:2]
        else:
            code6 = code_clean
            # crude heuristic
            prefix = "sz" if code6.startswith(("0","3")) else "sh"
        symbol = f"{prefix}.{code6}"

        fields = "date,open,high,low,close,preclose,volume,amount,pctChg"
        rs = bs.query_history_k_data_plus(
            symbol,
            fields=fields,
            start_date=pd.to_datetime(start_date).strftime("%Y-%m-%d"),
            end_date=pd.to_datetime(end_date).strftime("%Y-%m-%d"),
            frequency="d",
            adjustflag="1",
        )
        if rs.error_code != '0':
            self.log.warning(f"baostock query failed {symbol}: {rs.error_msg}")
            return None

        data = []
        while rs.next():
            data.append(rs.get_row_data())
        if not data:
            return None

        df = pd.DataFrame(data, columns=fields.split(","))
        df["trade_date"] = pd.to_datetime(df["date"])
        df.rename(columns={
            "preclose": "pre_close",
            "pctChg": "chg_pct",
        }, inplace=True)
        for col in ["open","high","low","close","pre_close","volume","amount","chg_pct"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df["source"] = "baostock"
        keep = ["trade_date","open","close","high","low","volume","amount","source","pre_close","chg_pct"]
        return df[keep]

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
    
        max_retries = 0
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
                return df[keep_cols]
    
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


    
