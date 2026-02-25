from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, date, time as dt_time
from pathlib import Path
from typing import List, Tuple, Set

import pandas as pd
import pytz
import baostock as bs
import akshare as ak

#from price_loader import load_stock_price_eastmoney, normalize_stock_code
from app.tasks.db_accessor import DbAccessor, infer_exchange_from_code6
from app.tasks.state_store import StateStore
from app.utils.utils_tools import normalize_stock_code
from app.utils.wireguard_helper import activate_tunnel, switch_wire_guard
import os 

# NOTE: keep this mapping stable with original runner.py
SPOT_RENAME_MAP = {
    '名称': 'name',
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
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".." ,".."))
        self.data_dir = os.path.join(root_dir, "data")


    def run(self, ctx) -> None:
        cfg = ctx.config
        self.log = ctx.log
        self.engine=ctx.engine
        db = DbAccessor(ctx.engine, self.log)
        state = StateStore(cfg.scanned_file, cfg.failed_file, self.log)

        if cfg.look_back_days == 1:
            is_intraday, latest_trading_date = self._get_intraday_status_and_last_trade_date( )
            if not is_intraday:
                self.log.info(f"检测到盘后时间，使用 ak.stock_zh_a_spot() 批量补齐最新交易日 {latest_trading_date}")
                ok = self._bulk_insert_latest_day_with_spot(latest_trading_date )
                if ok:
                    self.log.info(f"最新交易日 {latest_trading_date} 已批量补齐，跳过个股逐只拉取")
                    return
                self.log.info("全行情快照批量补齐失败，回退到原有逐只拉取逻辑")
            else:
                self.log.info("当前为盘中时间，无法使用收盘快照，执行原有逐只拉取逻辑")

        
        else:
            # >=1 
            self.log.info(f"[START][STOCK] window={cfg.start_date}~{cfg.end_date}")

        

        # universe
            if cfg.manual_stock_symbols:
                work_symbols = [(s, s) for s in sorted(set(cfg.manual_stock_symbols))]
                self.log.info(f"[CONFIG][STOCK] manual symbols = {len(work_symbols)}")
            else:
                work_symbols = self._get_all_symbols_from_spot( use_cache=True)
    
            # load
            
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
        # if all ok then
        # call sp 


    # ------------------------
    # universe
    def _get_all_symbols_from_spot(self,  use_cache: bool = True) -> List[Tuple[str, str]]:
        #cache_path = Path("data/all_symbols_spot.json")

        cache_path = os.path.join(self.data_dir, "all_symbols_spot.csv")
        if use_cache and cache_path.exists():
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list) and data and isinstance(data[0], list):
                    return [(n, s) for n, s in data]
            except Exception:
                pass

        spot_df = ak.stock_zh_a_spot()
        if spot_df is None or spot_df.empty:
            raise RuntimeError("ak.stock_zh_a_spot() empty; cannot build universe")
        spot_df = spot_df[~spot_df['代码'].str.startswith('bj')].copy()
        spot_df["symbol"] = spot_df["代码"].str.slice(start=2)
        spot_df["name"] = spot_df["名称"]
        out = sorted([(r["name"], r["symbol"]) for _, r in spot_df[["name","symbol"]].dropna().iterrows()], key=lambda x: x[1])

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.log.warning(f"写入 universe cache 失败: {e}")

        return out

    # ------------------------
    # main loop (historical per symbol)
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

        lg = bs.login()
        if lg.error_code != '0':
            self.log.error(f"baostock login failed: {lg.error_msg}")
        else:
            self.log.info("baostock login success (global for this batch)")

        scanned: Set[str] = state.load_scanned() if is_continue_load else set()
        failed: Set[str] = state.load_failed() if is_continue_load else set()

        total = len(work_symbols)
        self.log.info(f"[START][STOCK] window={start_date}~{end_date} | symbols={total} | continue={is_continue_load}")

        processed = 0
        for i, (name, symbol) in enumerate(work_symbols, start=1):
            processed += 1
            if state_flush_every > 0 and processed % state_flush_every == 0:
                state.save_scanned(scanned)
                state.save_failed(failed)
                self.log.info(f"[STATE] flushed scanned/failed at {processed}")

            symbol6 = normalize_stock_code(symbol)

            if symbol in scanned:
                self._log_progress( "STOCK", i, total, symbol, "SKIP scanned")
                continue
            if symbol in failed:
                self._log_progress( "STOCK", i, total, symbol, "SKIP failed")
                continue

            self._log_progress(  "STOCK", i, total, symbol, "CHECK DB coverage")

            try:
                existing_days = db.get_existing_stock_days(symbol6, start_date, end_date)

                df = self._load_stock_price_with_failover(
                    
                    stock_code=symbol6,
                    start_date=start_date,
                    end_date=end_date,
                    name=name,
                    adjust="qfq",
                )
                if df is None or df.empty:
                    raise RuntimeError("empty data from both baostock and eastmoney")

                stock_trade_days = set(df["trade_date"].dt.date.unique())
                missing = stock_trade_days - existing_days

                if not missing:
                    scanned.add(symbol)
                    self._log_progress( "STOCK", i, total, symbol, "OK already complete")
                    continue

                self._log_progress( "STOCK", i, total, symbol, f"INSERT missing_days={len(missing)}")
                df["name"] = name
                inserted = db.insert_missing_stock_days(symbol6, df, missing, start_date=start_date)

                remaining = stock_trade_days - db.get_existing_stock_days(symbol6, start_date, end_date)
                if remaining:
                    raise RuntimeError(f"still missing {len(remaining)} trade days")

                scanned.add(symbol)
                self._log_progress( "STOCK", i, total, symbol, f"FIXED inserted={inserted}")

            except Exception as e:
                failed.add(symbol)
                self.log.info(f"FAILED {symbol}: {e}")
                #try:
                #    switch_wire_guard("cn")
                #except Exception:
                #    pass
                self._log_progress("STOCK", i, total, symbol, f"FAILED {e}")

            # rate limiting
            if processed % 20 == 0:
                self.log.info("Processed 20 stocks, sleeping 1s to respect rate limit...")
                time.sleep(1.0)
            else:
                time.sleep(random.uniform(0.3, 0.8))

        try:
            bs.logout()
        except Exception:
            pass
        self.log.info("baostock logout")

        state.save_scanned(scanned)
        state.save_failed(failed)
        self.log.info("[DONE][STOCK] loader finished")

    # ------------------------
    # failover fetch
    def _load_stock_price_with_failover(
        self,
        
        stock_code: str,
        start_date: str,
        end_date: str,
        name: str,
        adjust: str = "qfq",
    ) -> pd.DataFrame | None:
        df = self._load_stock_price_baostock_internal( stock_code, start_date, end_date, adjust=adjust)
        if df is not None and not df.empty:
            return df
        self.log.warning(f"baostock failed for {stock_code}, switching to eastmoney")
        return self.load_stock_price_eastmoney(stock_code=stock_code, start_date=start_date, end_date=end_date, name=name)

    def _load_stock_price_baostock_internal(
        self,  stock_code: str, start_date: str, end_date: str, adjust: str = "qfq"
    ) -> pd.DataFrame | None:
        code = str(stock_code).strip()
        # baostock expects: sh.600000 / sz.000001
        if code.startswith(("6", "9")):
            symbol = f"sh.{code}"
        elif code.startswith(("0", "3")):
            symbol = f"sz.{code}"
        else:
            return None

        fields = "date,open,high,low,close,preclose,volume,amount,turn"
        rs = bs.query_history_k_data_plus(
            symbol,
            fields=fields,
            start_date=pd.to_datetime(start_date).strftime("%Y-%m-%d"),
            end_date=pd.to_datetime(end_date).strftime("%Y-%m-%d"),
            frequency="d",
            adjustflag="2" if adjust == "qfq" else "3" if adjust == "hfq" else "1",
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
            "turn": "turnover_rate",
        }, inplace=True)
        for col in ["open","high","low","close","pre_close","volume","amount","turnover_rate"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        # compute derived cols to align with eastmoney schema if needed
        if "chg_pct" not in df.columns:
            df["chg_pct"] = (df["close"] / df["pre_close"] - 1.0) * 100.0
        if "change" not in df.columns:
            df["change"] = df["close"] - df["pre_close"]
        if "amplitude" not in df.columns:
            df["amplitude"] = (df["high"] - df["low"]) / df["pre_close"] * 100.0

        keep = ["trade_date","open","close","pre_close","high","low","volume","amount","amplitude","chg_pct","change","turnover_rate"]
        df = df[keep]
        df["source"] = "baostock"
        return df

    # ------------------------
    # intraday helper
    def _get_intraday_status_and_last_trade_date(self) -> tuple[bool, str]:
        tz = pytz.timezone('Asia/Shanghai')
        now = datetime.now(tz)

        reference_date = (now - timedelta(days=1)).date() if now.hour < 8 else now.date()
        activate_tunnel("cn")
        lg = bs.login()
        if lg.error_code != '0':
            self.log.info(f"baostock login failed: {lg.error_msg}")
            is_weekend = reference_date.weekday() >= 5
            in_session = dt_time(9, 30) <= now.time() < dt_time(15, 0)
            fallback_last_date = reference_date - timedelta(days=(reference_date.weekday() + 2) % 7 if is_weekend else 0)
            return (not is_weekend and in_session, fallback_last_date.strftime('%Y-%m-%d'))

        try:
            start = (reference_date - timedelta(days=60)).strftime('%Y-%m-%d')
            end = reference_date.strftime('%Y-%m-%d')
            rs = bs.query_trade_dates(start_date=start, end_date=end)
            if rs.error_code != '0':
                raise RuntimeError(rs.error_msg)

            data_list = []
            while rs.next():
                data_list.append(rs.get_row_data())
            trade_df = pd.DataFrame(data_list, columns=rs.fields)
            trade_df['calendar_date'] = pd.to_datetime(trade_df['calendar_date'])
            trade_df['is_trading_day'] = trade_df['is_trading_day'].astype(int)
            trading_days = trade_df[trade_df['is_trading_day'] == 1]['calendar_date'].dt.date.values
            if len(trading_days) == 0:
                raise RuntimeError("No trading days in last 60 days")
            last_trade_date_obj = max(d for d in trading_days if d <= reference_date)
            last_trade_date_str = last_trade_date_obj.strftime('%Y-%m-%d')

            is_today_trading_day = last_trade_date_obj == now.date()
            in_trading_hours = dt_time(9, 30) <= now.time() < dt_time(15, 0)
            is_intraday = is_today_trading_day and in_trading_hours
            return is_intraday, last_trade_date_str
        except Exception as e:
            self.log.info(f"Error in get_intraday_status_and_last_trade_date: {e}")
            is_weekend = reference_date.weekday() >= 5
            in_session = dt_time(9, 30) <= now.time() < dt_time(15, 0)
            fallback_last = reference_date - timedelta(days=(reference_date.weekday() - 4) % 7)
            return (not is_weekend and in_session, fallback_last.strftime('%Y-%m-%d'))
        finally:
            try:
                bs.logout()
            except Exception:
                pass

    def _bulk_insert_latest_day_with_spot(self, latest_trading_date: str, ) -> bool:
        try:
            spot_df = ak.stock_zh_a_spot()
            if spot_df is None or spot_df.empty:
                self.log.info("ak.stock_zh_a_spot() 返回空数据，批量补齐失败")
                return False

            spot_df = spot_df[~spot_df['代码'].str.startswith('bj')].copy()
            spot_df['symbol'] = spot_df['代码'].str.slice(start=2)
            if spot_df.empty:
                self.log.info("过滤北交所后无有效股票数据")
                return False

            symbols = sorted(spot_df['symbol'].unique().tolist())
            symbols_path = Path("data/symbols.json")
            symbols_path.parent.mkdir(parents=True, exist_ok=True)
            with open(symbols_path, 'w', encoding='utf-8') as f:
                json.dump(symbols, f, ensure_ascii=False, indent=2)
            self.log.info(f"已更新全市场股票列表 data/symbols.json，共 {len(symbols)} 只股票")

            all_symbols = set(symbols)
            existing_query = f"SELECT symbol FROM CN_STOCK_DAILY_PRICE WHERE trade_date = TO_DATE('{latest_trading_date}', 'YYYY-MM-DD')"
            existing_df = pd.read_sql(existing_query, self.engine)
            existing_symbols = set(existing_df['symbol']) if not existing_df.empty else set()

            missing_symbols = all_symbols - existing_symbols
            if not missing_symbols:
                self.log.info(f"最新交易日 {latest_trading_date} 已全部存在（基于全行情快照），无需补齐")
                return True

            missing_df = spot_df[spot_df['symbol'].isin(missing_symbols)].copy()
            self.log.info(f"发现 {len(missing_symbols)} 只股票在 {latest_trading_date} 缺失，将补齐（含新股）")

            missing_df["trade_date"] = pd.to_datetime(latest_trading_date)

            if "代码" in missing_df.columns:
                missing_df["exchange"] = missing_df["代码"].apply(infer_exchange_from_prefixed_code)
            else:
                missing_df["exchange"] = None

            missing_df.rename(columns=SPOT_RENAME_MAP, inplace=True)

            # align schema columns
            for col in ["open","close","pre_close","high","low","volume","amount","amplitude","chg_pct","change","turnover_rate","name","exchange"]:
                if col not in missing_df.columns:
                    missing_df[col] = None

            missing_df["symbol"] = missing_df["symbol"].apply(normalize_stock_code)
            missing_df["window_start"] = pd.to_datetime(latest_trading_date)
            missing_df["source"] = "ak_spot"

            # ensure exchange if still missing
            missing_df["exchange"] = missing_df["exchange"].fillna(missing_df["symbol"].apply(infer_exchange_from_code6))

            table_cols = [
                "symbol","trade_date","open","close","pre_close","high","low","volume","amount",
                "amplitude","chg_pct","change","turnover_rate","source","window_start","exchange","name",
            ]
            missing_df = missing_df[table_cols]

            missing_df.to_sql("cn_stock_daily_price", self.engine, if_exists="append", index=False, chunksize=500)
            self.log.info(f"批量补齐完成：插入 {len(missing_df)} 条记录")
            return True
        except Exception as e:
            self.log.info(f"bulk_insert_latest_day_with_spot failed: {e}")
            return False

    def _log_progress(self,  stage: str, cur: int, total: int, symbol: str, msg: str) -> None:
        pct = (cur / total) * 100 if total else 0
        self.log.info(f"[{stage}] ({cur}/{total} | {pct:6.2f}%) {symbol} {msg}")

    
    # =====================================================
    # 股票（日线）- 东财
    # =====================================================
    def load_stock_price_eastmoney(self,
        stock_code: str,
        start_date: str,
        end_date: str,
        name: str,
        adjust: str = "qfq",
        
    ) -> pd.DataFrame | None:
        """
        使用东财接口拉取 A 股股票日线行情（AKShare: stock_zh_a_hist）
    
        Returns
        -------
        DataFrame columns（英文列名对齐 DB DDL / AK 返回语义）:
            trade_date, open, close, high, low,
            volume, amount, amplitude, chg_pct, change, turnover_rate, pre_close
        """
        df = ak.stock_zh_a_hist(
            symbol=stock_code,
            period="daily",
            start_date=start_date,
            end_date=end_date,
            adjust=adjust,
        )
    
        if df is None or df.empty:
            return None
    
        # 重命名列
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
    
        # 日期转换
        df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
    
        # 数值列转换
        numeric_cols = [
            "open", "close", "high", "low", "volume", "amount",
            "amplitude", "chg_pct", "change", "turnover_rate"
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
    
        # 关键修改：计算 pre_close 和 chg_pct（参考 load_index_price_em）
        df = df.sort_values("trade_date").reset_index(drop=True)
        df["pre_close"] = df["close"].shift(1)
        df["chg_pct"] = ((df["close"] - df["pre_close"]) / df["pre_close"] * 100).round(4)
    
        # 第一行没有前一日数据，置为空
        df.loc[0, ["pre_close", "chg_pct"]] = None, None
        df["name"] = name
        # 保留所有需要的列，如果缺少则补 None
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
            "pre_close",           # 新增
        ]
        for c in keep_cols:
            if c not in df.columns:
                df[c] = None
    
        df = df.dropna(subset=["trade_date", "close"]).copy()
    
        return df[keep_cols]
    
