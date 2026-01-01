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

from typing import Set
import argparse
import pytz
import baostock as bs
import pandas as pd
import akshare as ak
from sqlalchemy import create_engine, text

from price_loader import (
    load_stock_price_eastmoney,
    load_index_price_em,
    normalize_stock_code,
)
from universe_loader import load_universe
from oracle_utils import create_tables_if_not_exists
from coverage_audit import run_full_coverage_audit

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

        # 字段映射（已按你要求修正）
        missing_df['trade_date'] = pd.to_datetime(latest_trading_date)
        missing_df.rename(columns={
            '最新价': 'close',
            '涨跌幅': 'chg_pct',
            '成交量': 'volume',
            '成交额': 'turnover',   # 成交额 → turnover
        }, inplace=True)

        if 'volume' in missing_df.columns:
            missing_df['volume'] = missing_df['volume'] * 100
        if 'chg_pct' in missing_df.columns:
            missing_df['chg_pct'] = missing_df['chg_pct'] / 100.0

        if '涨跌额' in missing_df.columns and 'close' in missing_df.columns:
            missing_df['pre_close'] = missing_df['close'] - missing_df['涨跌额']

        missing_df['source'] = 'sina'
        missing_df['window_start'] = pd.to_datetime(latest_trading_date)
        missing_df['exchange'] = None

        table_columns = ['symbol', 'trade_date', 'close', 'pre_close', 'chg_pct', 'turnover',
                         'volume', 'source', 'window_start', 'exchange']
        insert_df = missing_df[[col for col in table_columns if col in missing_df.columns]]

        inserted_count = len(insert_df)
        #insert_df.to_sql('cn_stock_daily_price', engine, if_exists='append', index=False, method='multi')
        insert_df.to_sql(
            name="cn_stock_daily_price",
            con=engine,
            if_exists="append",
            index=False,
            chunksize=200,        # 与原代码一致，分批安全插入
        )
        
        log(f"成功批量插入最新交易日 {latest_trading_date} 缺失记录 {inserted_count} 条（source='em'）")
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
    df = df[["symbol", "trade_date", "close", "turnover", "pre_close", "chg_pct", "source"]]
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
    df = df[["index_code", "trade_date", "close", "turnover", "pre_close", "chg_pct", "source"]]
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

# =====================================================




def load_symbols_from_json() -> set:
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

    scanned = load_state(SCANNED_FILE)
    failed = load_state(FAILED_FILE)

    if MANUAL_STOCK_SYMBOLS:
        work_symbols = set(MANUAL_STOCK_SYMBOLS)
        log(f"[CONFIG][STOCK] manual symbols = {len(work_symbols)}")
    else:
        
        work_symbols = load_symbols_from_json()
        if not work_symbols:
            log("无可用股票列表，可能 data/symbols.json 未生成，退化为原有逻辑或跳过")
              ### 可选择 return 或继续原有 universe_loader 逻辑作为兜底
        
            work_symbols = set(load_universe().keys())
            log(f"[CONFIG][STOCK] universe symbols = {len(work_symbols)}")

       
    total = len(work_symbols)
    processed = 0
    log(f"[START][STOCK] window={start_date}~{end_date} ,检查 窗口内是否缺失 ，若缺失再拉")
    ##
    for i, symbol in enumerate(sorted(work_symbols), start=1):
    
        # ===== 0️⃣ 统一 symbol → 6 位 DB 主键 =====
        symbol6 = normalize_stock_code(symbol)
    
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

    #run_stock_loader()   # 1️⃣ 拉股票
    #run_index_loader()   # 2️⃣ 拉指数
    is_missing_list = run_full_coverage_audit(engine=engine, 
                            start_date=start_date, 
                            end_date=end_date,
                            stock_symbols=None,
                            index_codes=INDEX_SYMBOLS
                            )
    if len(is_missing_list):

        pass

      

    log("[DONE] AsharesScraper finished")

# =====================================================
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
    main()