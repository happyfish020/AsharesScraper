"""
AsharesScraper
=================================================

职责：
- 采集 A 股 / 指数 日级历史数据（东财）
- 只补缺失交易日（DB 对照）
- 断点再续（scanned / failed）
- 数据覆盖率与交易日日历审计
- 为 UnifiedRisk Phase-2 提供可靠数据事实

注意：
- 本项目不是 UnifiedRisk
- 不计算因子 / 不做回测
"""
import subprocess
import sys
import ctypes
 
import random
import re
import os, sys
import json
import time 
from pathlib import Path
from datetime import  datetime, timedelta, date, time as dt_time
from sqlalchemy import create_engine, text, inspect

import numpy as np

 


from typing import Set
import argparse
import pytz
import baostock as bs
import pandas as pd
import akshare as ak
from logger import setup_logging, get_logger

from ak_fut_index_daily import load_index_futures_daily
from ak_option_sse_daily_sina import load_option_sse_daily
from price_loader import (
    load_stock_price_eastmoney,
    load_index_price_em,
    normalize_stock_code,
)
#from universe_loader import load_universe
from oracle_utils import create_tables_if_not_exists
from coverage_audit import run_full_coverage_audit
from ak_fund_etf_spot_em import load_fund_etf_hist_baostock, load_fund_etf_hist_em,  load_spot_as_hist_today
from wireguard_helper import activate_tunnel, deactivate_tunnel, switch_wire_guard, toggle_vpn

# =====================================================
# ====================== CONFIG =======================
# =====================================================

# -------- 数据窗口 --------
look_back_days = 20

# -------- 股票来源（手工优先） --------
MANUAL_STOCK_SYMBOLS = []   # 非空则只跑这些，否则跑 universe

# -------- 指数 --------
INDEX_SYMBOLS = [
    "sz399001",
    "sh000001",
    "sz399006",
    #"sh000688", # kc500 000688
    "sh000300", #沪深 300 指数 #hs300
    "sh000905",  # 中证500 , 中证 500（CSI 500） zz500
    "sh000016", # 上证 50（SSE 50）
    "sh000852", # 中证 1000（CSI 1000）
]
BASE_INDEX = "sh000300"

# -------- 断点再续 --------
STATE_DIR = "state"
SCANNED_FILE = os.path.join(STATE_DIR, "scanned.json")
FAILED_FILE = os.path.join(STATE_DIR, "failed.json")
STATE_FLUSH_EVERY = 3

# -------- 审计输出 --------
AUDIT_OUTPUT_DIR = "audit_reports"

# =====================================================
# ====================== DB ===========================
# =====================================================

DB_USER = "secopr"
DB_PASSWORD = "secopr"
DB_HOST = "localhost"
DB_PORT = "1521"
DB_SERVICE = "xe"
SCHEMA_NAME = "SECOPR"



DSN = f"{DB_HOST}:{DB_PORT}/{DB_SERVICE}"
CONNECTION_STRING = f"oracle+oracledb://{DB_USER}:{DB_PASSWORD}@{DSN}"

engine = create_engine(
    CONNECTION_STRING,
    pool_pre_ping=True,
    future=True,
)


# 状态文件路径
SCANNED_STATE = "state/scanned.json"
FAILED_STATE = "state/failed.json"

# =====================================================
# ==================== WINDOW =========================
# =====================================================

# =====================================================
# ==================== UTIL ===========================
# =====================================================

 
def log_progress(stage: str, cur: int, total: int, symbol: str, msg: str):
    pct = (cur / total) * 100 if total else 0
    LOG.info(f"[{stage}] ({cur}/{total} | {pct:6.2f}%) {symbol} {msg}")

def infer_exchange_from_prefixed_code(code: str) -> str | None:
    """code: 'sh600000'/'sz000001'/'bj830001' -> SSE/SZSE/BJSE"""
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


def infer_exchange_from_code6(code6: str) -> str | None:
    """6 位纯数字 -> SSE/SZSE/BJSE（仅用于补全 exchange 字段）"""
    if not isinstance(code6, str) or len(code6) != 6 or not code6.isdigit():
        return None
    if code6.startswith(("6", "9")):
        return "SSE"
    if code6.startswith(("0", "3")):
        return "SZSE"
    if code6.startswith("8"):
        return "BJSE"
    return None








def clear_state_files():
    """
    清除断点状态文件（refresh 模式）
    """
    for path in [SCANNED_FILE, FAILED_FILE]:
        if os.path.exists(path):
            os.remove(path)
            LOG.info(f"[REFRESH] removed {path}")
        else:
            LOG.info(f"[REFRESH] not found, skip {path}")

# ==================== 日志函数 ====================
 
def parse_args():
    parser = argparse.ArgumentParser(description="AsharesScraper - A股数据加载器")
    parser.add_argument("--refresh", action="store_true", help="清空状态文件，从头开始抓取")
    parser.add_argument("--days", type=int, default=1, help="回溯天数（默认1，即只补最新交易日数据）")
    return parser.parse_args()

def load_state(path: str) -> Set[str]:
    if not os.path.exists(path):
        return set()
    with open(path, "r", encoding="utf-8") as f:
        return set(json.load(f))


def save_state(path: str, state: Set[str]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sorted(list(state)), f, indent=2, ensure_ascii=False)

##########
# ==================== 交易日与盘中判断 ====================
def get_intraday_status_and_last_trade_date() -> tuple[bool, str]:
    """
    判断当前是否为交易日盘中时间，并返回最近一个交易日
    Returns:
        (is_intraday: bool, last_trade_date: str)
    """
    tz = pytz.timezone('Asia/Shanghai')
    now =  datetime.now(tz)

    if now.hour < 8:
        reference_date = (now - timedelta(days=1)).date()
    else:
        reference_date = now.date()

    lg = bs.login()
    if lg.error_code != '0':
        LOG.info(f"baostock login failed: {lg.error_msg}")
        is_weekend = reference_date.weekday() >= 5
        in_session = dt_time(9, 30) <= now.time() < dt_time(15, 0)
        fallback_last_date = reference_date - timedelta(days=(reference_date.weekday() + 2) % 7 if is_weekend else 0)
        return (not is_weekend and in_session, fallback_last_date.strftime('%Y-%m-%d'))

    try:
        start_date = (reference_date - timedelta(days=60)).strftime('%Y-%m-%d')
        end_date = reference_date.strftime('%Y-%m-%d')

        rs = bs.query_trade_dates(start_date=start_date, end_date=end_date)
        if rs.error_code != '0':
            raise Exception(f"query_trade_dates failed: {rs.error_msg}")

        data_list = []
        while rs.next():
            data_list.append(rs.get_row_data())

        trade_df = pd.DataFrame(data_list, columns=rs.fields)
        trade_df['calendar_date'] = pd.to_datetime(trade_df['calendar_date'])
        trade_df['is_trading_day'] = trade_df['is_trading_day'].astype(int)

        trading_days = trade_df[trade_df['is_trading_day'] == 1]['calendar_date'].dt.date.values
        if len(trading_days) == 0:
            raise Exception("No trading days found in the past 60 days")

        last_trade_date_obj = max(d for d in trading_days if d <= reference_date)
        last_trade_date_str = last_trade_date_obj.strftime('%Y-%m-%d')

        is_today_trading_day = last_trade_date_obj == now.date()
        in_trading_hours = dt_time(9, 30) <= now.time() < dt_time(15, 0)

        # todo  if T h < 9:30 how spot data looks like ??
        is_intraday = is_today_trading_day and in_trading_hours

        return is_intraday, last_trade_date_str

    except Exception as e:
        LOG.info(f"Error in get_intraday_status_and_last_trade_date: {e}")
        is_weekend = reference_date.weekday() >= 5
        
        in_session = dt_time(9, 30) <= now.time() < dt_time(15, 0)
        fallback_last = reference_date - timedelta(days=(reference_date.weekday() - 4) % 7)
        return (not is_weekend and in_session, fallback_last.strftime('%Y-%m-%d'))

    finally:
        bs.logout()

# ==================== 批量补最新交易日（spot） ====================
def bulk_insert_latest_day_with_spot(latest_trading_date: str, engine) -> bool:
    """使用 ak.stock_zh_a_spot() 批量插入最新交易日缺失记录，并更新全市场符号列表"""
    try:
 
        spot_df = ak.stock_zh_a_spot()
        if spot_df.empty:
            LOG.info("ak.stock_zh_a_spot() 返回空数据，批量补齐失败")
            return False

        # 过滤北交所 + 统一代码
        spot_df = spot_df[~spot_df['代码'].str.startswith('bj')].copy()
        spot_df['symbol'] = spot_df['代码'].str.slice(start=2)

        if spot_df.empty:
            LOG.info("过滤北交所后无有效股票数据")
            return False

        # === 新增：保存所有 symbol 到 data/symbols.json ===
        symbols = sorted(spot_df['symbol'].unique().tolist())  # 去重并排序，便于对比
        symbols_path = Path("data/symbols.json")
        symbols_path.parent.mkdir(parents=True, exist_ok=True)  # 自动创建 data 目录
        with open(symbols_path, 'w', encoding='utf-8') as f:
            json.dump(symbols, f, ensure_ascii=False, indent=2)
        LOG.info(f"已更新全市场股票列表 data/symbols.json，共 {len(symbols)} 只股票")

        # 以 spot 返回的 symbol 作为全量目标集合
        all_symbols = set(symbols)

        # 查询数据库中该交易日已存在的 symbol
        existing_query = f"SELECT symbol FROM CN_STOCK_DAILY_PRICE WHERE trade_date = TO_DATE('{latest_trading_date}', 'YYYY-MM-DD')"
        existing_df = pd.read_sql(existing_query, engine)
        existing_symbols = set(existing_df['symbol'])

        # 计算缺失
        missing_symbols = all_symbols - existing_symbols
        if not missing_symbols:
            LOG.info(f"最新交易日 {latest_trading_date} 已全部存在（基于全行情快照），无需补齐")
            return True

        # 提取缺失股票数据
        missing_df = spot_df[spot_df['symbol'].isin(missing_symbols)].copy()
        LOG.info(f"发现 {len(missing_symbols)} 只股票在 {latest_trading_date} 缺失，将补齐（含新股）")

                # 字段映射（对齐新版 DDL：amount / turnover_rate 等；尽力从 spot 字段补全）
        missing_df["trade_date"] = pd.to_datetime(latest_trading_date)

        # 先补 exchange（基于原始带前缀代码）
        if "代码" in missing_df.columns:
            missing_df["exchange"] = missing_df["代码"].apply(infer_exchange_from_prefixed_code)
        else:
            missing_df["exchange"] = None

        rename_map = {
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
            "昨收": "pre_close",  # 前收：入库 + 可用于兜底计算
        }
        missing_df.rename(columns=rename_map, inplace=True)

        # 兜底：如没有 pre_close，但有 close/change，可推算
        if "pre_close" not in missing_df.columns and "close" in missing_df.columns and "change" in missing_df.columns:
            missing_df["pre_close"] = pd.to_numeric(missing_df["close"], errors="coerce") - pd.to_numeric(missing_df["change"], errors="coerce")

        # 兜底：如没有 amplitude，但有 high/low/pre_close，则计算（百分比）
        if "amplitude" not in missing_df.columns and all(c in missing_df.columns for c in ["high", "low", "pre_close"]):
            high = pd.to_numeric(missing_df["high"], errors="coerce")
            low = pd.to_numeric(missing_df["low"], errors="coerce")
            pre_close = pd.to_numeric(missing_df["pre_close"], errors="coerce")
            missing_df["amplitude"] = ((high - low) / pre_close * 100).round(4)

        # 数值列转换（注意：按 AKShare 语义保留“百分比”单位，不做 /100）
        for col in ["open", "close", "high", "low", "volume", "amount", "amplitude", "chg_pct", "change", "turnover_rate"]:
            if col in missing_df.columns:
                missing_df[col] = pd.to_numeric(missing_df[col], errors="coerce")

        missing_df["source"] = "ak_spot"
        missing_df["window_start"] = pd.to_datetime(latest_trading_date)

        table_columns = [
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
            "name"
        ]
        for c in table_columns:
            if c not in missing_df.columns:
                missing_df[c] = None

        insert_df = missing_df[table_columns]

        inserted_count = len(insert_df)
        #insert_df.to_sql('cn_stock_daily_price', engine, if_exists='append', index=False, method='multi')
        insert_df.to_sql(
            name="cn_stock_daily_price",
            con=engine,
            if_exists="append",
            index=False,
            chunksize=200,        # 与原代码一致，分批安全插入
        )
        
        LOG.info(f"成功批量插入最新交易日 {latest_trading_date} 缺失记录 {inserted_count} 条（source='ak_spot'）")
        return True

    except Exception as e:
        LOG.info(f"批量补齐最新交易日失败: {str(e)}")
        return False

# =====================================================
# ==================== DB READ ========================
# =====================================================

def get_expected_trade_days() -> Set[date]:
    sql = """
        SELECT trade_date
        FROM cn_index_daily_price
        WHERE index_code = :idx
          AND trade_date BETWEEN TO_DATE(:s,'YYYYMMDD')
                              AND TO_DATE(:e,'YYYYMMDD')
        ORDER BY trade_date
    """
    with engine.begin() as conn:
        rows = conn.execute(
            text(sql),
            {"idx": BASE_INDEX, "s": start_date, "e": end_date},
        ).fetchall()
    return {r[0].date() for r in rows}


def get_existing_stock_days(symbol: str) -> Set[date]:
    sql = """
        SELECT trade_date
        FROM cn_stock_daily_price
        WHERE symbol = :sym
          AND trade_date BETWEEN TO_DATE(:s,'YYYYMMDD')
                              AND TO_DATE(:e,'YYYYMMDD')
    """
    try: 
        with engine.begin() as conn:
            rows = conn.execute(
                text(sql),
                {"sym": symbol, "s": start_date, "e": end_date},
            ).fetchall()
        return {r[0].date() for r in rows}
    except Exception as e: 
        RuntimeError(e)

def get_existing_index_days(index_code: str) -> Set[date]:
    sql = """
        SELECT trade_date
        FROM cn_index_daily_price
        WHERE index_code = :idx
          AND trade_date BETWEEN TO_DATE(:s,'YYYYMMDD')
                              AND TO_DATE(:e,'YYYYMMDD')
    """
    with engine.begin() as conn:
        rows = conn.execute(
            text(sql),
            {"idx": index_code, "s": start_date, "e": end_date},
        ).fetchall()
    return {r[0].date() for r in rows}

# =====================================================

# ==================== DB WRITE =======================
# =====================================================

def insert_missing_stock_days(symbol: str, df: pd.DataFrame, missing: Set[date]) -> int:
    df = df[df["trade_date"].dt.date.isin(missing)].copy()
    if df.empty:
        return 0
    symbol6 = normalize_stock_code(symbol)

    df["symbol"] = symbol6
    #df["source"] = "eastmoney"
    df["window_start"] = pd.to_datetime(start_date, format="%Y%m%d", errors="coerce") if isinstance(start_date, str) else None
    df["exchange"] = infer_exchange_from_code6(symbol6)

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
    for c in table_cols:
        if c not in df.columns:
            df[c] = None

    df = df[table_cols]


    numeric_cols = [
        'open', 'close', 'pre_close', 'high', 'low',
        'volume', 'amount',
        'amplitude', 'chg_pct', 'change', 'turnover_rate'
    ]
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: None if pd.isna(x) else f"{float(x):.10g}"   # .10g 通用格式，避免多余0
            ) 
    try:
        df.to_sql(
            "cn_stock_daily_price",
            engine,
            if_exists="append",
            index=False,
            chunksize=200,
        )
        return len(df)
    except Exception as e:
        LOG.info(f"Symbol: {symbol6} to DB failed - {e}")
        raise RuntimeError(e)

def insert_missing_index_days(index_code: str, df: pd.DataFrame, missing: Set[date]) -> int:
    df = df[df["trade_date"].dt.date.isin(missing)].copy()
    if df.empty:
        return 0

    df["index_code"] = index_code
    df["source"] = "eastmoney"

    table_cols = [
        "index_code",
        "trade_date",
        "open",
        "close",
        "high",
        "low",
        "volume",
        "amount",
        "source",
        "pre_close",
        "chg_pct",
    ]
    for c in table_cols:
        if c not in df.columns:
            df[c] = None

    df = df[table_cols]

    try:
        df.to_sql(
            "cn_index_daily_price",
            engine,
            if_exists="append",
            index=False,
            chunksize=200,
        )
        return len(df)
    except Exception as e:
        LOG.info(f"idx: {index_code} to DB failed - {e}")
        raise RuntimeError(e)

def load_symbols_from_json():
    """
    原函数：从 JSON 文件加载股票列表
    新实现：从数据库 CN_STOCK_DAILY_PRICE 表查询每个 symbol 的最后一条交易日记录
             返回 set[(stock_name, symbol)] 
             - 如果能关联到名称（可扩展），否则 name = symbol
             - 当前因无 name 字段，暂时使用 symbol 作为 name（后续可 join 其他表）
    """
    LOG.info(f"[{datetime.now()}] 从数据库加载股票代码列表（基于最新交易日）...")

    try:
        from oracle_utils import get_engine
        
        engine = get_engine()
        sql = """
        SELECT DISTINCT SYMBOL, NAME,
               FIRST_VALUE(SYMBOL) OVER (PARTITION BY SYMBOL ORDER BY TRADE_DATE DESC) AS STOCK_NAME_TEMP
        FROM (
            SELECT SYMBOL, TRADE_DATE,NAME
            FROM cn_stock_daily_price
            WHERE TRADE_DATE = (
                SELECT MAX(TRADE_DATE) 
                FROM cn_stock_daily_price sd 
                WHERE sd.SYMBOL = cn_stock_daily_price.SYMBOL
            )
        )
        """

        with engine.connect() as conn:
            result = conn.execute(text(sql))
            symbols_set = set()
            
            for row in result:
                symbol = row[0].strip() if row[0] else None
                
                name = row[1].strip() if row[1] else None  
                
                if symbol:
                    symbols_set.add((name, symbol))
            
            LOG.info(f"[{datetime.now()}] 从数据库加载到 {len(symbols_set)} 只股票")
            return symbols_set

    except Exception as e:
        LOG.info(f"错误: 从数据库加载股票列表失败: {e}")
        LOG.info(" fallback: 返回空集合")
        return set()
    

    
def load_symbols_from_json_1() -> set:
    """从 data/symbols.json 加载股票列表"""
    symbols_path = Path("data/symbols.json")
    if not symbols_path.exists():
        LOG.info("警告: data/symbols.json 不存在，将尝试通过 spot 接口生成（仅当 days=1 时有效）")
        return set()  # 空集，后续会走失败回退逻辑
    with open(symbols_path, 'r', encoding='utf-8') as f:
        symbols = json.load(f)
    LOG.info(f"从 data/symbols.json 加载股票列表，共 {len(symbols)} 只")
    return set(symbols)


# ================= 1️⃣ STOCK LOADER ==================
# =====================================================

 
def run_stock_loader():
    global look_back_days

 
    if look_back_days ==1:
        is_intraday, latest_trading_date = get_intraday_status_and_last_trade_date()
        if not is_intraday:
            LOG.info(f"检测到盘后时间，使用 ak.stock_zh_a_spot() 批量补齐最新交易日 {latest_trading_date}")
            if bulk_insert_latest_day_with_spot(latest_trading_date, engine):
                LOG.info(f"最新交易日 {latest_trading_date} 已批量补齐，跳过个股逐只拉取")
                return
            else:
                LOG.info("全行情快照批量补齐失败，回退到原有逐只拉取逻辑")
        else:
            LOG.info("当前为盘中时间，无法使用收盘快照，执行原有逐只拉取逻辑")
        
        LOG.info("[DONE][STOCK] 全行情快照批量 loader finished")
        return 

    #look_back_days = 7 
    ############
    else:
        LOG.info(f"[START][STOCK] window={start_date}~{end_date}")
    
        is_continue_load = True
        if MANUAL_STOCK_SYMBOLS:
            work_symbols = set(MANUAL_STOCK_SYMBOLS)
            LOG.info(f"[CONFIG][STOCK] manual symbols = {len(work_symbols)}")
        else:
            
            #work_symbols_set = load_symbols_from_json()
            #work_symbols_list = list(work_symbols_set)
    
            #if not work_symbols_list:
            #    LOG.info("无可用股票列表，可能 data/symbols.json 未生成，退化为原有逻辑或跳过")
                  ### 可选择 return 或继续原有 universe_loader 逻辑作为兜底
                #raise Exception("read CN_SECURITY_MASTER for all symbols")
                #work_symbols = set(load_universe().keys())
                #LOG.info(f"[CONFIG][STOCK] universe symbols = {len(work_symbols)}")
                # read CN_SECURITY_MASTER for all symbols
            work_symbols_list=  get_all_symbols_from_spot()
        load_symbols_days(work_symbols =work_symbols_list , start_date=start_date , end_date=end_date, is_continue_load=is_continue_load)
    
        #end def 

def load_symbols_days_no(work_symbols: list, start_date:str, end_date: str, is_continue_load:bool=False):
    
    if work_symbols and isinstance(work_symbols[0], tuple):
        # 有 name 的情况
        symbol_list = [sym for name, sym in work_symbols]
        name_map = {sym: name for name, sym in work_symbols}
    else:
        # 只有 symbol 的情况
        symbol_list = work_symbols
        name_map = {sym: sym for sym in symbol_list}  # name fallback to symbol
    
    total = len(work_symbols)
    processed = 0

    
    scanned = {}
    failed = {}


    if is_continue_load:
        scanned = load_state(SCANNED_FILE)
        failed = load_state(FAILED_FILE)


    LOG.info(f"[START][STOCK] window={start_date}~{end_date} ,检查 窗口内是否缺失 ，若缺失再拉")
    ##
    #for i, symbol in enumerate(sorted(work_symbols), start=1):
    processed = 0
    for i, (name, symbol) in enumerate(sorted(work_symbols), start=1):
        processed += 1
        if processed % STATE_FLUSH_EVERY == 0:
            save_state(SCANNED_FILE, scanned)
            save_state(FAILED_FILE, failed)
            LOG.info(f"[STATE] flushed scanned/failed at {processed}")
        
        
        # ===== 0️⃣ 统一 symbol → 6 位 DB 主键 =====
        symbol6 = normalize_stock_code(symbol)
        LOG.info(f"正在处理第 {processed } 只：{symbol6} - {name}")   # ← 可以直接用
        # ===== 1️⃣ 断点再续检查（注意：scanned / failed 用原始 symbol）=====
        if symbol in scanned:
            log_progress("STOCK", i, total, symbol, "SKIP scanned")
            continue
    
        if symbol in failed:
            log_progress("STOCK", i, total, symbol, "SKIP failed")
            continue
    
        log_progress("STOCK", i, total, symbol, "CHECK DB coverage")
    
        try:
            # ===== 2️⃣ 先取 DB 中已有的交易日 =====
            existing_days = get_existing_stock_days(symbol6)
    
            # ===== 3️⃣ 拉东财数据（作为“该股可交易日事实源”）=====
            #time.sleep(1)
            time.sleep(random.uniform(0.5, 1))
             
            df = load_stock_price_eastmoney(
                stock_code=symbol6,
                start_date=start_date,
                end_date=end_date,
                name = name, # <================ todo
            )
            
            if df is None or df.empty:
                raise RuntimeError("empty eastmoney data")
    
            # ===== 4️⃣ 该股票在窗口内“真实可交易日” =====
            stock_trade_days = set(df["trade_date"].dt.date.tolist())
    
            # ===== 5️⃣ 真正需要补的 = 可交易日 - DB 已有 =====
            missing = stock_trade_days - existing_days
    
            if not missing:
                scanned.add(symbol)
                log_progress("STOCK", i, total, symbol, "OK already complete")
                continue
    
            log_progress(
                "STOCK", i, total, symbol,
                f"INSERT missing_days={len(missing)}"
            )
            df['name'] = name # add name  
            inserted = insert_missing_stock_days(symbol6, df, missing)
    
            # ===== 6️⃣ 再校验（⚠️ 用 stock_trade_days，不再用指数交易日）=====
            remaining = stock_trade_days - get_existing_stock_days(symbol6)
            if remaining:
                raise RuntimeError(
                        f"still missing {len(remaining)} trade days (after insert)"
                )
    
            scanned.add(symbol)
    
            # ===== 7️⃣ 规范化提示（只做日志，不影响逻辑）=====
            if symbol != symbol6:
                log_progress(
                    "STOCK", i, total, symbol,
                    f"NORMALIZE -> {symbol6}"
                )
    
            log_progress(
                "STOCK", i, total, symbol,
                f"FIXED inserted={inserted}"
            )
    
        except Exception as e:
            failed.add(symbol)
            
            LOG.info(e)
            
            
            switch_wire_guard("cn")
            log_progress("STOCK", i, total, symbol, f"FAILED {e}")
    
        # ===== 8️⃣ 状态文件 checkpoint（每 N 个）=====
        
        
    save_state(SCANNED_FILE, scanned)
    save_state(FAILED_FILE, failed)
    LOG.info("[DONE][STOCK] loader finished")




# ==================== baostock 单只核心函数（无 login/logout） ====================
def _load_stock_price_baostock_internal(
    stock_code: str, start_date: str, end_date: str, name: str, adjust: str = "qfq"
) -> pd.DataFrame | None:
    """纯内部调用，不负责登录登出"""
    code_clean = str(stock_code).strip()
    if len(code_clean) != 6 or not code_clean.isdigit():
        return None

    # 前缀转换
    if code_clean.startswith(('6', '68', '69')):
        symbol = f"sh.{code_clean}"
    elif code_clean.startswith(('0', '3')):
        symbol = f"sz.{code_clean}"
    elif code_clean.startswith(('4', '8')):
        symbol = f"bj.{code_clean}"
    else:
        return None

    adjust_map = {"qfq": "2", "hfq": "3", "": "1", None: "1"}
    adjustflag = adjust_map.get(adjust, "1")

    fields = "date,code,open,high,low,close,preclose,volume,amount,turn,pctChg"

    def to_baostock_date(d: str) -> str:
        d = d.strip().replace("-", "")
        if len(d) != 8 or not d.isdigit():
            raise ValueError(f"无效日期格式: {d}")
        return f"{d[:4]}-{d[4:6]}-{d[6:8]}"

    try:
        
  
        rs = bs.query_history_k_data_plus(
            code=symbol,
            fields=fields,
            start_date=to_baostock_date(start_date),
            end_date=to_baostock_date(end_date),
            frequency="d",
            adjustflag=adjustflag
        )
        if rs.error_code != '0' or rs.data is None:
            return None

        df = rs.get_data()
        if df.empty:
            return None

        # 重命名 + 数值转换（与原 eastmoney 完全对齐）
        rename_map = {
            'date': 'trade_date',
            'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close',
            'preclose': 'pre_close',
            'volume': 'volume', 'amount': 'amount',
            'turn': 'turnover_rate',
            'pctChg': 'chg_pct_raw',
        }

        df = df.rename(columns=rename_map)
        # ─── 关键修复：转换为 datetime ───
        df['trade_date'] = pd.to_datetime(df['trade_date'], errors='coerce')

        numeric_cols = ['open', 'high', 'low', 'close', 'pre_close', 'volume', 'amount', 'turnover_rate', 'chg_pct_raw']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # 衍生字段计算（与原函数完全一致）
        df = df.sort_values("trade_date").reset_index(drop=True)
        df['amplitude'] = ((df['high'] - df['low']) / df['pre_close'] * 100).where(df['pre_close'] != 0, np.nan)
        df['change'] = df['close'] - df['pre_close']
        df['chg_pct'] = ((df['close'] - df['pre_close']) / df['pre_close'] * 100).round(4)
        df.loc[0, ['pre_close', 'change', 'chg_pct', 'amplitude']] = None, None, None, None

        df['name'] = name

        keep_cols = [
            "trade_date", "name", "open", "close", "high", "low",
            "volume", "amount", "amplitude", "chg_pct", "change",
            "turnover_rate", "pre_close"
        ]
        for c in keep_cols:
            if c not in df.columns:
                df[c] = None
        df = df[keep_cols].dropna(subset=["trade_date", "close"]).copy()
        df = df.replace([np.nan, np.inf, -np.inf], None)
        df["source"] = "batstock"
        return df

    except Exception as e:
        LOG.error( f"Exception:{e} - Symbol:{stock_code} , Name:{name}")
        #raise e
        return None


# ==================== 带 failover 的统一入口 ====================
def load_stock_price_with_failover(
    stock_code: str, start_date: str, end_date: str, name: str, adjust: str = "qfq"
) -> pd.DataFrame | None:
    """对外统一接口：baostock 主 → eastmoney failover"""
    # 1. 先试 baostock
    df = _load_stock_price_baostock_internal(stock_code, start_date, end_date, name, adjust)
    if df is not None and not df.empty:
        return df

    # 2. failover 到 akshare
    LOG.warning(f"baostock failed for {stock_code}, switching to eastmoney")
    return load_stock_price_eastmoney(
        stock_code=stock_code,
        start_date=start_date,
        end_date=end_date,
        name=name,
        adjust=adjust
    )

def load_symbols_days(work_symbols: list, start_date:str, end_date: str, is_continue_load:bool=False):
    
    lg = bs.login() 
    if lg.error_code != '0':
        LOG.error(f"baostock login failed: {lg.error_msg}")
        # 可以选择直接 return 或继续用 akshare
    else:
        LOG.info("baostock login success (global for this batch)")

    if work_symbols and isinstance(work_symbols[0], tuple):
        # 有 name 的情况
        symbol_list = [sym for name, sym in work_symbols]
        name_map = {sym: name for name, sym in work_symbols}
    else:
        # 只有 symbol 的情况
        symbol_list = work_symbols
        name_map = {sym: sym for sym in symbol_list}  # name fallback to symbol
    
    total = len(work_symbols)
    processed = 0

    
    scanned = {}
    failed = {}


    if is_continue_load:
        scanned = load_state(SCANNED_FILE)
        failed = load_state(FAILED_FILE)


    LOG.info(f"[START][STOCK] window={start_date}~{end_date} ,检查 窗口内是否缺失 ，若缺失再拉")
    ##
    #for i, symbol in enumerate(sorted(work_symbols), start=1):
    processed = 0
    for i, (name, symbol) in enumerate(sorted(work_symbols), start=1):
        processed += 1
        if processed % STATE_FLUSH_EVERY == 0:
            save_state(SCANNED_FILE, scanned)
            save_state(FAILED_FILE, failed)
            LOG.info(f"[STATE] flushed scanned/failed at {processed}")
        
        
        # ===== 0️⃣ 统一 symbol → 6 位 DB 主键 =====
        symbol6 = normalize_stock_code(symbol)
        LOG.info(f"正在处理第 {processed } 只：{symbol6} - {name}")   # ← 可以直接用
        # ===== 1️⃣ 断点再续检查（注意：scanned / failed 用原始 symbol）=====
        if symbol in scanned:
            log_progress("STOCK", i, total, symbol, "SKIP scanned")
            continue
    
        if symbol in failed:
            log_progress("STOCK", i, total, symbol, "SKIP failed")
            continue
    
        log_progress("STOCK", i, total, symbol, "CHECK DB coverage")
    
        try:
            # ===== 2️⃣ 先取 DB 中已有的交易日 =====
            existing_days = get_existing_stock_days(symbol6)
    
            # ========== 关键替换：使用带 failover 的函数 ==========
            df = load_stock_price_with_failover(
                stock_code=symbol6,
                start_date=start_date,
                end_date=end_date,
                name=name,
                adjust="qfq"   # 你原来用的参数，可改
            ) 
            
            if df is None or df.empty:
                raise RuntimeError("empty data from both baostock and eastmoney")

            #stock_trade_days = set(df["trade_date"].dt.date.tolist())
            stock_trade_days = set(df["trade_date"].dt.date.unique())
            missing = stock_trade_days - existing_days

            if not missing:
                scanned.add(symbol)   # 注意：你原来是 scanned.add，但 scanned 是 dict？建议改成 set
                log_progress("STOCK", i, total, symbol, "OK already complete")
                continue

            log_progress("STOCK", i, total, symbol, f"INSERT missing_days={len(missing)}")
            df['name'] = name
            inserted = insert_missing_stock_days(symbol6, df, missing)

            remaining = stock_trade_days - get_existing_stock_days(symbol6)
            if remaining:
                raise RuntimeError(f"still missing {len(remaining)} trade days")

            scanned.add(symbol)
            log_progress("STOCK", i, total, symbol, f"FIXED inserted={inserted}")

        except Exception as e:
            failed.add(symbol)
            LOG.info(f"FAILED {symbol}: {e}")
            switch_wire_guard("cn")
            log_progress("STOCK", i, total, symbol, f"FAILED {e}")

        # ==================== 每 20 只 sleep 1 秒 ====================
        if processed % 20 == 0:
            LOG.info(f"Processed {processed} stocks, sleeping 1s to respect baostock rate limit...")
            time.sleep(1.0)
        else:
            # 保留你原来的随机小延时（可选）
            time.sleep(random.uniform(0.3, 0.8))

    # ==================== 结束登出 ====================
    bs.logout()
    LOG.info("baostock logout")

    save_state(SCANNED_FILE, scanned)
    save_state(FAILED_FILE, failed)
    LOG.info("[DONE][STOCK] loader finished") 
# =====================================================
def load_index_price_with_failover(
    index_code: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame | None:
    """主函数：baostock → eastmoney failover"""
    # 先试 baostock
    df = _load_index_price_baostock_internal(index_code, start_date, end_date)
    if df is not None and not df.empty:
        return df

    # failover
    LOG.warning(f"baostock failed for index {index_code}, switching to eastmoney")
    return load_index_price_em(  # 原函数保留，作为备用
        index_code=index_code,
        start_date=start_date,
        end_date=end_date,
    )
 

def _load_index_price_baostock_internal(
    index_code: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame | None:
    """baostock 单次查询（不负责 login/logout）"""
    code_clean = str(index_code).strip()
    
    # 自动加前缀（常见指数规则）
    if code_clean.startswith(('0', '3')):
        symbol = f"sz.{code_clean}"
    elif code_clean.startswith(('0', '8', '9')):  # 上证、深证、国债等
        symbol = f"sh.{code_clean}"
    else:
        # 如果已经带前缀，直接用
        if '.' in code_clean and len(code_clean.split('.')[1]) == 6:
            symbol = code_clean
        elif code_clean.startswith(('sz', 'sh')) and len(code_clean ) == 8:
            symbol = code_clean.replace("sz","sz.").replace("sh","sh.")

    # 日期转 YYYY-MM-DD
    def to_bs_date(d: str) -> str:
        d = d.replace("-", "").strip()
        if len(d) == 8 and d.isdigit():
            return f"{d[:4]}-{d[4:6]}-{d[6:8]}"
        return d  # 如果已经是 YYYY-MM-DD，直接返回

    start_dt = to_bs_date(start_date)
    end_dt = to_bs_date(end_date)

    fields = "date,code,open,high,low,close,preclose,volume,amount,turn,pctChg"

    try:
        rs = bs.query_history_k_data_plus(
            code=symbol,
            fields=fields,
            start_date=start_dt,
            end_date=end_dt,
            frequency="d",
            adjustflag="1"  # 指数一般不复权
        )

        if rs.error_code != '0':
            return None

        df = rs.get_data()
        if df.empty:
            return None

        # 重命名 + 类型转换
        rename_map = {
            'date': 'trade_date',
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'preclose': 'pre_close',
            'volume': 'volume',
            'amount': 'amount',
            'turn': 'turnover_rate',     # baostock 有 turn，但原函数无，可选保留
            'pctChg': 'chg_pct_raw',
        }
        df = df.rename(columns=rename_map)

        # 日期转 datetime
        df['trade_date'] = pd.to_datetime(df['trade_date'], errors='coerce')
        df = df.dropna(subset=['trade_date', 'close'])

        # 数值转换
        numeric_cols = ['open', 'close', 'high', 'low', 'pre_close', 'volume', 'amount']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # 计算衍生字段（与原函数一致）
        df = df.sort_values("trade_date").reset_index(drop=True)
        df["pre_close"] = df["close"].shift(1)
        df["chg_pct"] = ((df["close"] - df["pre_close"]) / df["pre_close"] * 100).round(4)
        df.loc[0, ["pre_close", "chg_pct"]] = None, None

        # 保持列一致（原函数无 turnover_rate，但 baostock 有，可选加）
        keep_cols = [
            "trade_date", "open", "close", "high", "low",
            "volume", "amount", "pre_close", "chg_pct"
        ]
        for c in keep_cols:
            if c not in df.columns:
                df[c] = None

        df = df[keep_cols].copy()
        df = df.replace([np.nan, np.inf, -np.inf], None)

        return df

    except Exception as e:
        LOG.warning(f"index_code:{symbol}, e:{e}")
        return None

# ================= 2️⃣ INDEX LOADER ==================
# =====================================================

def run_index_loader():
    LOG.info(f"[START][INDEX] window={start_date}~{end_date}")

    # baostock 全局登录一次
    lg = bs.login()
    if lg.error_code != '0':
        LOG.error(f"baostock login failed: {lg.error_msg}")
        # 可以选择继续用 eastmoney，或直接退出
    else:
        LOG.info("baostock login success (global for index batch)")

    LOG.info(f"[START][INDEX] window={start_date}~{end_date}")

    total = len(INDEX_SYMBOLS)
    processed = 0 
    for i, idx in enumerate(INDEX_SYMBOLS, start=1):
        processed = processed +1
        log_progress("INDEX", i, total, idx, "CHECK DB coverage")
        #symbol6 = normalize_stock_code()
        try:
            existing = get_existing_index_days(idx)
            # 使用带 failover 的函数
            df = load_index_price_with_failover(
                index_code=idx,
                start_date=start_date,
                end_date=end_date,
            )
             
            if df is None or df.empty:
                raise RuntimeError("empty index data")

            expected = set(df["trade_date"].dt.date.tolist())
            missing = expected - existing

            if not missing:
                log_progress("INDEX", i, total, idx, "OK already complete")
                continue

            inserted = insert_missing_index_days(idx, df, missing)

            remaining = expected - get_existing_index_days(idx)
            if remaining:
                raise RuntimeError(f"still missing {len(remaining)} days")

            log_progress("INDEX", i, total, idx, f"FIXED inserted={inserted}")

        except Exception as e:
            log_progress("INDEX", i, total, idx, f"FAILED {e}")
            # 可以选择 raise 或 continue，根据你的容错需求

        # 频率控制（指数数量少，10个一休）
        if processed % 10 == 0:
            LOG.info(f"Processed {processed} indices, sleeping 1s...")
            time.sleep(1.0)
        else:
            time.sleep(0.3)  # 小间隔防限流

    # 结束登出
    bs.logout()
    LOG.info("baostock logout")
    LOG.info("[DONE][INDEX] loader finished")
 
    LOG.info("[DONE][INDEX] loader finished")

# =====================================================
# ================= 3️⃣ STOCK AUDIT ===================
# =====================================================
 
# =====================================================
# ================= 4️⃣ INDEX AUDIT ===================
# =====================================================

 
# =====================================================
# ================= PIPELINE ==========================
# =====================================================

def run_data_pipeline():
    LOG.info("[START] AsharesScraper")

    with engine.begin() as conn:
        create_tables_if_not_exists(conn)
    
    run_stock_loader()   # 1️⃣ 拉股票 
    
    if look_back_days >1:
        load_fund_etf_hist_baostock(start_date=start_date, end_date=end_date)
        LOG.info("[DONE] load_fund_etf_hist_em finished")
    else:
        load_spot_as_hist_today()
        LOG.info("[DONE] load_spot_as_hist_today finished")
    LOG.info("[DONE] run_stock_loader finished")
    
    run_index_loader()   # 2️⃣ 拉指数
    LOG.info("[DONE] run_index_loader finished")
    LOG.info("[DONE] load_spot_as_hist_today finished")

    load_index_futures_daily(start_date=start_date, end_date=end_date)    
    LOG.info("[DONE] load_index_futures_daily finished")
    #load_fund_etf_hist_em(start_date=start_date, end_date=end_date, adjust="qfq")
    

   
    #
    load_option_sse_daily()


    is_missing_list = run_full_coverage_audit(engine=engine, 
                            start_date=start_date, 
                            end_date=end_date,
                            stock_symbols=None,
                            index_codes=INDEX_SYMBOLS
                            )
    #if len(is_missing_list) ==0 :
    #    pass
        #load_symbols_days(work_symbols =is_missing_list , start_date=start_date , end_date=end_date, is_continue_load=False)

      

    LOG.info("[DONE] AsharesScraper finished")


######################################################


# 辅助函数：获取交易所（复用之前的逻辑）
def get_exchange(code: str) -> str:
    code = str(code).strip()
    if code.startswith(('6', '9')):
        return 'SH'
    elif code.startswith(('0', '3')):
        return 'SZ'
    elif code.startswith(('4', '8', '43', '83', '87', '88')) or len(code) > 6:
        return 'BJ'
    return 'UNKNOWN'

####################################################################
# =====================================================

# ==============================
#     表创建函数
# ==============================
def create_table_if_not_exists(table_name: str, create_sql: str, indexes: list = None):
    inspector = inspect(engine)
    
    if not inspector.has_table(table_name, schema=SCHEMA_NAME):
        LOG.info(f"表 {SCHEMA_NAME}.{table_name} 不存在，正在创建...")
        
        with engine.connect() as conn:
            conn.execute(text(create_sql))
            
            # 创建索引（如果有）
            if indexes:
                for idx_sql in indexes:
                    conn.execute(text(idx_sql))
            
            conn.commit()
        LOG.info(f"表 {table_name} 创建完成")
    else:
        LOG.info(f"表 {SCHEMA_NAME}.{table_name} 已存在，跳过创建")


#
def get_hot_rank_em():
    SCHEMA_NAME = "SECOPR"

    # ==============================
    #     新增：股票热度排行表
    # ==============================
    HOT_RANK_TABLE = "CN_STOCK_HOT_RANK_EM"
    today = date.today()  # only for today.
    hot_rank_create_sql = f"""
    CREATE TABLE {SCHEMA_NAME}.{HOT_RANK_TABLE} (
        RANK_CURRENT     NUMBER(5)      NOT NULL,
        SYMBOL           VARCHAR2(10)   NOT NULL,
        EXCHANGE         VARCHAR2(10)   NOT NULL,
        STOCK_NAME       VARCHAR2(60),
        LATEST_PRICE     NUMBER(12,4),
        CHANGE_AMOUNT    NUMBER(12,6),
        CHANGE_PCT       NUMBER(8,4),
        ASOF_DATE        DATE           NOT NULL,
        SOURCE           VARCHAR2(30)   DEFAULT 'AKSHARE_EASTMONEY',
        CREATED_AT       DATE           DEFAULT SYSDATE,

        CONSTRAINT CN_STOCK_HOT_RANK_PK
            PRIMARY KEY (SYMBOL, ASOF_DATE)
    )
    """
    hot_rank_indexes = [
        f"CREATE INDEX IDX_HOT_RANK_DATE ON {SCHEMA_NAME}.{HOT_RANK_TABLE} (ASOF_DATE)",
        f"CREATE INDEX IDX_HOT_RANK_SYMBOL ON {SCHEMA_NAME}.{HOT_RANK_TABLE} (SYMBOL)"
    ]

    # ==============================
    #     创建热度排行表（如果不存在）
    # ==============================
    LOG.info("\n=== 检查并创建股票热度排行表 ===")
    create_table_if_not_exists(HOT_RANK_TABLE, hot_rank_create_sql, hot_rank_indexes)

    # ==============================
    #     采集 & 插入 股票热度排行（人气榜）
    # ==============================
    LOG.info("\n=== 采集东方财富股票热度排行（前100名） ===")

    try:
        df_hot = ak.stock_hot_rank_em()
        LOG.info(f"获取热度排行：{len(df_hot)} 条记录")

        if df_hot is None or df_hot.empty:
            LOG.info("热度排行数据为空，跳过插入")
            return

        # --- 字段映射（保持原逻辑）---
        df_hot = df_hot.rename(columns={
            '当前排名': 'RANK_CURRENT',
            '代码': 'SYMBOL',
            '股票名称': 'STOCK_NAME',
            '最新价': 'LATEST_PRICE',
            '涨跌额': 'CHANGE_AMOUNT',
            '涨跌幅': 'CHANGE_PCT'
        })

        # --- 必要列兜底 ---
        for c in ["RANK_CURRENT", "SYMBOL", "STOCK_NAME", "LATEST_PRICE", "CHANGE_AMOUNT", "CHANGE_PCT"]:
            if c not in df_hot.columns:
                df_hot[c] = None

        # --- 统一代码格式 ---
        df_hot["SYMBOL"] = df_hot["SYMBOL"].astype(str).str.strip()

        # --- 推导交易所：兼容 'SZ002202' / 'SH600879' / 'BJ8xxxxx' / '002202' ---
        def _derive_exchange(sym: str) -> str:
            s = str(sym).strip().upper()
            if s.startswith("SH"):
                return "SH"
            if s.startswith("SZ"):
                return "SZ"
            if s.startswith("BJ"):
                return "BJ"
            # fallback: 尽量抽取 6 位数字再走原 get_exchange
            digits = "".join(ch for ch in s if ch.isdigit())
            return get_exchange(digits if digits else s)

        df_hot["EXCHANGE"] = df_hot["SYMBOL"].apply(_derive_exchange)

        # --- 数值清洗：去掉 %, 逗号, 占位符；并把 NaN/inf 变成 None（否则 Oracle NUMBER 报 DPY-4004）---
        import numpy as np

        def _clean_numeric_series(s: pd.Series) -> pd.Series:
            if s is None:
                return s
            # 统一转字符串再清洗（兼容 '--' / '—' / None / '9.99%' / '1,234.56'）
            ss = s.astype(str).str.strip()
            ss = ss.replace({"None": "", "nan": "", "NaN": "", "--": "", "—": "", "N/A": ""})
            ss = ss.str.replace("%", "", regex=False).str.replace(",", "", regex=False)
            out = pd.to_numeric(ss, errors="coerce")
            out = out.replace([np.inf, -np.inf], np.nan)
            return out

        # rank 必须是整数
        df_hot["RANK_CURRENT"] = _clean_numeric_series(df_hot["RANK_CURRENT"]).astype("Int64")

        # 其它数值列
        for col in ["LATEST_PRICE", "CHANGE_AMOUNT", "CHANGE_PCT"]:
            df_hot[col] = _clean_numeric_series(df_hot[col])

        # --- 补充必要字段（保持原逻辑）---
        df_hot["ASOF_DATE"] = today
        df_hot["SOURCE"] = "AKSHARE_EASTMONEY"
        created_at_dt = datetime.now()
        df_hot["CREATED_AT"] = created_at_dt

        # --- 过滤关键字段缺失（避免 NOT NULL / PK / 非法绑定）---
        df_hot = df_hot[
            df_hot["SYMBOL"].notna() & (df_hot["SYMBOL"].astype(str).str.len() > 0) &
            df_hot["RANK_CURRENT"].notna()
        ].copy()

        # --- 只保留入库列，顺序对齐 SQL ---
        df_hot = df_hot[[
            "RANK_CURRENT", "SYMBOL", "EXCHANGE", "STOCK_NAME",
            "LATEST_PRICE", "CHANGE_AMOUNT", "CHANGE_PCT",
            "ASOF_DATE", "SOURCE", "CREATED_AT"
        ]]

        # --- pandas/np 的 NaN/NA -> None；并确保数值是 Python 原生类型（更稳）---
        def _py_num(v):
            if v is None:
                return None
            try:
                fv = float(v)
                if fv != fv or fv == float("inf") or fv == float("-inf"):
                    return None
                return fv
            except Exception:
                return None

        records_hot = []
        for r in df_hot.to_dict("records"):
            rc = r.get("RANK_CURRENT")
            try:
                rank_val = int(rc) if rc is not None and str(rc) != "<NA>" else None
            except Exception:
                rank_val = None

            rec = {
                "RANK_CURRENT": rank_val,
                "SYMBOL": (r.get("SYMBOL") or None),
                "EXCHANGE": (r.get("EXCHANGE") or None),
                "STOCK_NAME": (r.get("STOCK_NAME") or None),
                "LATEST_PRICE": _py_num(r.get("LATEST_PRICE")),
                "CHANGE_AMOUNT": _py_num(r.get("CHANGE_AMOUNT")),
                "CHANGE_PCT": _py_num(r.get("CHANGE_PCT")),
                "ASOF_DATE": today,
                "SOURCE": (r.get("SOURCE") or "AKSHARE_EASTMONEY"),
                "CREATED_AT": created_at_dt,
            }
            # 最后一层兜底：必须字段缺失就跳过
            if rec["RANK_CURRENT"] is None or rec["SYMBOL"] is None or rec["EXCHANGE"] is None:
                continue
            records_hot.append(rec)

        # 插入（幂等方式）
        with engine.connect() as conn:
            # 先删除当天旧数据
            conn.execute(
                text(f"DELETE FROM {SCHEMA_NAME}.{HOT_RANK_TABLE} WHERE ASOF_DATE = :dt"),
                {"dt": today}
            )

            if records_hot:
                conn.execute(
                    text(f"""
                        INSERT INTO {SCHEMA_NAME}.{HOT_RANK_TABLE}
                        (RANK_CURRENT, SYMBOL, EXCHANGE, STOCK_NAME,
                         LATEST_PRICE, CHANGE_AMOUNT, CHANGE_PCT,
                         ASOF_DATE, SOURCE, CREATED_AT)
                        VALUES
                        (:RANK_CURRENT, :SYMBOL, :EXCHANGE, :STOCK_NAME,
                         :LATEST_PRICE, :CHANGE_AMOUNT, :CHANGE_PCT,
                         :ASOF_DATE, :SOURCE, :CREATED_AT)
                    """),
                    records_hot
                )

            conn.commit()

        LOG.info(f"股票热度排行插入完成：{len(records_hot)} 条（日期：{today}）")

    except Exception as e:
        LOG.info("采集或插入股票热度排行失败:", str(e))

    # 可选：简单验证（保持原逻辑）
    try:
        with engine.connect() as conn:
            count = conn.execute(
                text(f"SELECT COUNT(*) FROM {SCHEMA_NAME}.{HOT_RANK_TABLE} WHERE ASOF_DATE = :dt"),
                {"dt": today}
            ).scalar()
            LOG.info(f"验证：当天热度排行记录数 = {count}")
    except Exception as e:
        LOG.info("验证查询失败:", str(e))

    LOG.info("\n=== 股票热度排行采集任务完成 ===")

def get_all_symbols_from_spot(use_cache: bool = True) -> list[tuple[str, str]]:
    """
    从 ak.stock_zh_a_spot() 获取当前全市场可交易 A 股符号列表
    返回格式：list[(name: str, symbol: str)]，例如 [("平安银行", "000001"), ...]
    
    - 过滤掉北交所（代码以 bj 开头）
    - symbol 统一为 6 位纯数字
    - 按 symbol 排序返回
    - 支持缓存（data/all_symbols_with_name.json）
    """
    cache_file = Path("data/all_symbols_with_name.json")
    
    if use_cache and cache_file.exists():
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # 兼容旧缓存（如果是纯 list[str]，自动转换）
            if isinstance(data, list) and all(isinstance(x, str) for x in data):
                LOG.info("旧缓存格式（纯 symbol），重新拉取")
            else:
                symbols_list = [(item["name"], item["symbol"]) for item in data]
                LOG.info(f"从缓存加载 {len(symbols_list)} 个 (name, symbol) 对")
                return symbols_list
        except Exception as e:
            LOG.info(f"缓存加载失败，将重新拉取: {e}")

    try:
        df_spot = ak.stock_zh_a_spot()
        if df_spot.empty:
            LOG.info("ak.stock_zh_a_spot() 返回空数据，无法获取全市场符号")
            return []

        # 过滤北交所（代码以 bj 开头）
        df_spot = df_spot[~df_spot['代码'].str.startswith('bj', na=False)].copy()

        # 统一提取 6 位数字代码
        def extract_code(code: str) -> str:
            code = str(code).strip()
            m = re.search(r'\d{6}', code)
            return m.group(0) if m else None

        df_spot['symbol'] = df_spot['代码'].apply(extract_code)
        df_spot = df_spot[df_spot['symbol'].notna() & (df_spot['symbol'].str.len() == 6)]

        # 获取名称列（假设 ak 返回 '名称' 列）
        name_col = '名称' if '名称' in df_spot.columns else 'name'
        if name_col not in df_spot.columns:
            df_spot['name'] = None
        else:
            df_spot = df_spot.rename(columns={name_col: 'name'})

        # 去重 + 按 symbol 排序
        df_spot = df_spot[['name', 'symbol']].drop_duplicates(subset=['symbol'])
        df_spot = df_spot.sort_values('symbol')

        # 转为 list[tuple[name, symbol]]
        symbols_list = [
            (row['name'] if pd.notna(row['name']) else "", row['symbol'])
            for _, row in df_spot.iterrows()
        ]

        LOG.info(f"从 ak.stock_zh_a_spot() 获取到 {len(symbols_list)} 个有效 A 股 (name, symbol) 对（已排除北交所）")

        # 保存缓存（list of dict，便于未来扩展）
        cache_data = [{"name": name, "symbol": symbol} for name, symbol in symbols_list]
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)

        return symbols_list

    except Exception as e:
        LOG.info(f"get_all_symbols_from_spot 失败: {str(e)}")
        return []

# ================= MAIN ==============================
# =====================================================

# ==================== 主入口 ====================

#
#start_date = (date.today() - timedelta(days=look_back_days)).strftime("%Y%m%d")
start_date = None
end_daten  = None 


def main():
    global look_back_days , start_date, end_date
    
    args = parse_args()
    look_back_days = args.days
    start_date  = (date.today() - timedelta(days=look_back_days)).strftime("%Y%m%d")
    end_date = date.today().strftime("%Y%m%d")

    if args.refresh:
        LOG.info("刷新模式：清空状态文件")
        for path in [SCANNED_STATE, FAILED_STATE]:
            if os.path.exists(path):
                os.remove(path)
                pass  
    LOG.info(f"启动 A股数据加载，回溯天数: {look_back_days}")

    run_data_pipeline()

    # 如有需要，可继续调用 run_index_loader()、审计等（保持原逻辑）
    
    LOG.info("本次运行完成")

if __name__ == '__main__':
    #get_hot_rank_em()
    setup_logging(market="cn", mode="scraper")
    LOG = get_logger("Main")
    LOG.info("start..")
    #activate_tunnel("cn")
    toggle_vpn("cn", "start")
    LOG.info("wireguard activated!")
    
    #deactivate_tunnel("cn")

    main()