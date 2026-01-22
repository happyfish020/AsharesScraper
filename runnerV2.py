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

import os, sys
import json
from pathlib import Path
from datetime import datetime, timedelta, time,date
from sqlalchemy import create_engine, text, inspect

from typing import Set
import argparse
import pytz
import baostock as bs
import pandas as pd
import akshare as ak
 
from price_loader import (
    load_stock_price_eastmoney,
    load_index_price_em,
    normalize_stock_code,
)
#from universe_loader import load_universe
from oracle_utils import create_tables_if_not_exists
from coverage_audit import run_full_coverage_audit
from ak_fund_etf_spot_em import load_fund_etf_spot_em

# =====================================================
# ====================== CONFIG =======================
# =====================================================

# -------- 数据窗口 --------
look_back_days = 20

# -------- 股票来源（手工优先） --------
MANUAL_STOCK_SYMBOLS = []   # 非空则只跑这些，否则跑 universe

# -------- 指数 --------
INDEX_SYMBOLS = [
    "sh000300",
    "sh000001",
    "sz399006",
    "sh000688",
    "sh000905",  # 中证500
]
BASE_INDEX = "sh000300"

# -------- 断点再续 --------
STATE_DIR = "state"
SCANNED_FILE = os.path.join(STATE_DIR, "scanned.json")
FAILED_FILE = os.path.join(STATE_DIR, "failed.json")
STATE_FLUSH_EVERY = 30

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

def log(msg: str):
    print(msg, flush=True)


def log_progress(stage: str, cur: int, total: int, symbol: str, msg: str):
    pct = (cur / total) * 100 if total else 0
    log(f"[{stage}] ({cur}/{total} | {pct:6.2f}%) {symbol} {msg}")

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
            print(f"[REFRESH] removed {path}")
        else:
            print(f"[REFRESH] not found, skip {path}")

# ==================== 日志函数 ====================
def log(msg: str):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

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
    now = datetime.now(tz)

    if now.hour < 8:
        reference_date = (now - timedelta(days=1)).date()
    else:
        reference_date = now.date()

    lg = bs.login()
    if lg.error_code != '0':
        log(f"baostock login failed: {lg.error_msg}")
        is_weekend = reference_date.weekday() >= 5
        in_session = time(9, 30) <= now.time() < time(15, 0)
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
        in_trading_hours = time(9, 30) <= now.time() < time(15, 0)

        # todo  if T h < 9:30 how spot data looks like ??
        is_intraday = is_today_trading_day and in_trading_hours

        return is_intraday, last_trade_date_str

    except Exception as e:
        log(f"Error in get_intraday_status_and_last_trade_date: {e}")
        is_weekend = reference_date.weekday() >= 5
        in_session = time(9, 30) <= now.time() < time(15, 0)
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
            log("ak.stock_zh_a_spot() 返回空数据，批量补齐失败")
            return False

        # 过滤北交所 + 统一代码
        spot_df = spot_df[~spot_df['代码'].str.startswith('bj')].copy()
        spot_df['symbol'] = spot_df['代码'].str.slice(start=2)

        if spot_df.empty:
            log("过滤北交所后无有效股票数据")
            return False

        # === 新增：保存所有 symbol 到 data/symbols.json ===
        symbols = sorted(spot_df['symbol'].unique().tolist())  # 去重并排序，便于对比
        symbols_path = Path("data/symbols.json")
        symbols_path.parent.mkdir(parents=True, exist_ok=True)  # 自动创建 data 目录
        with open(symbols_path, 'w', encoding='utf-8') as f:
            json.dump(symbols, f, ensure_ascii=False, indent=2)
        log(f"已更新全市场股票列表 data/symbols.json，共 {len(symbols)} 只股票")

        # 以 spot 返回的 symbol 作为全量目标集合
        all_symbols = set(symbols)

        # 查询数据库中该交易日已存在的 symbol
        existing_query = f"SELECT symbol FROM CN_STOCK_DAILY_PRICE WHERE trade_date = TO_DATE('{latest_trading_date}', 'YYYY-MM-DD')"
        existing_df = pd.read_sql(existing_query, engine)
        existing_symbols = set(existing_df['symbol'])

        # 计算缺失
        missing_symbols = all_symbols - existing_symbols
        if not missing_symbols:
            log(f"最新交易日 {latest_trading_date} 已全部存在（基于全行情快照），无需补齐")
            return True

        # 提取缺失股票数据
        missing_df = spot_df[spot_df['symbol'].isin(missing_symbols)].copy()
        log(f"发现 {len(missing_symbols)} 只股票在 {latest_trading_date} 缺失，将补齐（含新股）")

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
        
        log(f"成功批量插入最新交易日 {latest_trading_date} 缺失记录 {inserted_count} 条（source='ak_spot'）")
        return True

    except Exception as e:
        log(f"批量补齐最新交易日失败: {str(e)}")
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
    df["source"] = "eastmoney"
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
        print(f"Symbol: {symbol6} to DB failed - {e}")
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
        print(f"idx: {index_code} to DB failed - {e}")
        raise RuntimeError(e)

def load_symbols_from_json():
    """
    原函数：从 JSON 文件加载股票列表
    新实现：从数据库 CN_STOCK_DAILY_PRICE 表查询每个 symbol 的最后一条交易日记录
             返回 set[(stock_name, symbol)] 
             - 如果能关联到名称（可扩展），否则 name = symbol
             - 当前因无 name 字段，暂时使用 symbol 作为 name（后续可 join 其他表）
    """
    print(f"[{datetime.now()}] 从数据库加载股票代码列表（基于最新交易日）...")

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
            
            print(f"[{datetime.now()}] 从数据库加载到 {len(symbols_set)} 只股票")
            return symbols_set

    except Exception as e:
        print(f"错误: 从数据库加载股票列表失败: {e}")
        print(" fallback: 返回空集合")
        return set()
    

    
def load_symbols_from_json_1() -> set:
    """从 data/symbols.json 加载股票列表"""
    symbols_path = Path("data/symbols.json")
    if not symbols_path.exists():
        log("警告: data/symbols.json 不存在，将尝试通过 spot 接口生成（仅当 days=1 时有效）")
        return set()  # 空集，后续会走失败回退逻辑
    with open(symbols_path, 'r', encoding='utf-8') as f:
        symbols = json.load(f)
    log(f"从 data/symbols.json 加载股票列表，共 {len(symbols)} 只")
    return set(symbols)


# ================= 1️⃣ STOCK LOADER ==================
# =====================================================

 
def run_stock_loader():
    global look_back_days

 
    if look_back_days == 1:
        is_intraday, latest_trading_date = get_intraday_status_and_last_trade_date()
        if not is_intraday:
            log(f"检测到盘后时间，使用 ak.stock_zh_a_spot() 批量补齐最新交易日 {latest_trading_date}")
            if bulk_insert_latest_day_with_spot(latest_trading_date, engine):
                log(f"最新交易日 {latest_trading_date} 已批量补齐，跳过个股逐只拉取")
                return
            else:
                log("全行情快照批量补齐失败，回退到原有逐只拉取逻辑")
        else:
            log("当前为盘中时间，无法使用收盘快照，执行原有逐只拉取逻辑")
        
        log("[DONE][STOCK] 全行情快照批量 loader finished")
        #return 

    #look_back_days = 7 
    ############

    log(f"[START][STOCK] window={start_date}~{end_date}")

    is_continue_load = True
    if MANUAL_STOCK_SYMBOLS:
        work_symbols = set(MANUAL_STOCK_SYMBOLS)
        log(f"[CONFIG][STOCK] manual symbols = {len(work_symbols)}")
    else:
        
        work_symbols_set = load_symbols_from_json()
        work_symbols_list = list(work_symbols_set)

        if not work_symbols_list:
            log("无可用股票列表，可能 data/symbols.json 未生成，退化为原有逻辑或跳过")
              ### 可选择 return 或继续原有 universe_loader 逻辑作为兜底
            #raise Exception("read CN_SECURITY_MASTER for all symbols")
            #work_symbols = set(load_universe().keys())
            #log(f"[CONFIG][STOCK] universe symbols = {len(work_symbols)}")
            # read CN_SECURITY_MASTER for all symbols
    load_symbols_days(work_symbols =work_symbols_list , start_date=start_date , end_date=end_date, is_continue_load=is_continue_load)

    #end def 

def load_symbols_days(work_symbols: list, start_date:str, end_date: str, is_continue_load:bool=False):
    
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


    log(f"[START][STOCK] window={start_date}~{end_date} ,检查 窗口内是否缺失 ，若缺失再拉")
    ##
    #for i, symbol in enumerate(sorted(work_symbols), start=1):
    for i, (name, symbol) in enumerate(sorted(work_symbols), start=1):

        # ===== 0️⃣ 统一 symbol → 6 位 DB 主键 =====
        symbol6 = normalize_stock_code(symbol)
        print(f"正在处理第 {i} 只：{symbol6} - {name}")   # ← 可以直接用
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
            import time
            #time.sleep(1)
            df = load_stock_price_eastmoney(
                stock_code=symbol6,
                start_date=start_date,
                end_date=end_date,
              #  name = name, <================ todo
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
            log_progress("STOCK", i, total, symbol, f"FAILED {e}")
    
        # ===== 8️⃣ 状态文件 checkpoint（每 N 个）=====
            processed += 1
        if processed % STATE_FLUSH_EVERY == 0:
            save_state(SCANNED_FILE, scanned)
            save_state(FAILED_FILE, failed)
            log(f"[STATE] flushed scanned/failed at {processed}")
    
    ##
    save_state(SCANNED_FILE, scanned)
    save_state(FAILED_FILE, failed)
    log("[DONE][STOCK] loader finished")

# =====================================================
# ================= 2️⃣ INDEX LOADER ==================
# =====================================================

def run_index_loader():
    log(f"[START][INDEX] window={start_date}~{end_date}")

    total = len(INDEX_SYMBOLS)

    for i, idx in enumerate(INDEX_SYMBOLS, start=1):
        log_progress("INDEX", i, total, idx, "CHECK DB coverage")
        #symbol6 = normalize_stock_code()
        try:
            existing = get_existing_index_days(idx)

            df = load_index_price_em(
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
            raise e

    log("[DONE][INDEX] loader finished")

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
    log("[START] AsharesScraper")

    with engine.begin() as conn:
        create_tables_if_not_exists(conn)
    

    load_fund_etf_spot_em()
    sys.exit()
    run_stock_loader()   # 1️⃣ 拉股票
    run_index_loader()   # 2️⃣ 拉指数
    is_missing_list = run_full_coverage_audit(engine=engine, 
                            start_date=start_date, 
                            end_date=end_date,
                            stock_symbols=None,
                            index_codes=INDEX_SYMBOLS
                            )
    if len(is_missing_list):
        pass
        #load_symbols_days(work_symbols =is_missing_list , start_date=start_date , end_date=end_date, is_continue_load=False)

      

    log("[DONE] AsharesScraper finished")


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
        print(f"表 {SCHEMA_NAME}.{table_name} 不存在，正在创建...")
        
        with engine.connect() as conn:
            conn.execute(text(create_sql))
            
            # 创建索引（如果有）
            if indexes:
                for idx_sql in indexes:
                    conn.execute(text(idx_sql))
            
            conn.commit()
        print(f"表 {table_name} 创建完成")
    else:
        print(f"表 {SCHEMA_NAME}.{table_name} 已存在，跳过创建")


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
    print("\n=== 检查并创建股票热度排行表 ===")
    create_table_if_not_exists(HOT_RANK_TABLE, hot_rank_create_sql, hot_rank_indexes)

    # ==============================
    #     采集 & 插入 股票热度排行（人气榜）
    # ==============================
    print("\n=== 采集东方财富股票热度排行（前100名） ===")

    try:
        df_hot = ak.stock_hot_rank_em()
        print(f"获取热度排行：{len(df_hot)} 条记录")

        if df_hot is None or df_hot.empty:
            print("热度排行数据为空，跳过插入")
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

        print(f"股票热度排行插入完成：{len(records_hot)} 条（日期：{today}）")

    except Exception as e:
        print("采集或插入股票热度排行失败:", str(e))

    # 可选：简单验证（保持原逻辑）
    try:
        with engine.connect() as conn:
            count = conn.execute(
                text(f"SELECT COUNT(*) FROM {SCHEMA_NAME}.{HOT_RANK_TABLE} WHERE ASOF_DATE = :dt"),
                {"dt": today}
            ).scalar()
            print(f"验证：当天热度排行记录数 = {count}")
    except Exception as e:
        print("验证查询失败:", str(e))

    print("\n=== 股票热度排行采集任务完成 ===")




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
        log("刷新模式：清空状态文件")
        for path in [SCANNED_STATE, FAILED_STATE]:
            if os.path.exists(path):
                os.remove(path)
                pass  
    log(f"启动 A股数据加载，回溯天数: {look_back_days}")

    run_data_pipeline()

    # 如有需要，可继续调用 run_index_loader()、审计等（保持原逻辑）
  
    log("本次运行完成")

if __name__ == '__main__':
    #get_hot_rank_em()
    main()