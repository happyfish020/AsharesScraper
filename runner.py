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

import os
import json
from datetime import date, timedelta
from typing import Set
import argparse

import pandas as pd
from sqlalchemy import create_engine, text

from price_loader import (
    load_stock_price_eastmoney,
    load_index_price_em,
    normalize_stock_code,
)
from universe_loader import load_universe
from oracle_utils import create_tables_if_not_exists
from coverage_audit import (
    audit_stock_missing_days,
    audit_index_calendar_health,
)

# =====================================================
# ====================== CONFIG =======================
# =====================================================

# -------- 数据窗口 --------
LOOKBACK_DAYS = 25

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

# =====================================================
# ==================== WINDOW =========================
# =====================================================

END_DATE = date.today().strftime("%Y%m%d")
START_DATE = (date.today() - timedelta(days=LOOKBACK_DAYS)).strftime("%Y%m%d")

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


def parse_args():
    parser = argparse.ArgumentParser(
        description="AsharesScraper - Data Loader"
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Clear scanned/failed state and restart from scratch",
    )
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
            {"idx": BASE_INDEX, "s": START_DATE, "e": END_DATE},
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
                {"sym": symbol, "s": START_DATE, "e": END_DATE},
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
            {"idx": index_code, "s": START_DATE, "e": END_DATE},
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
# ================= 1️⃣ STOCK LOADER ==================
# =====================================================

def run_stock_loader():
    log(f"[START][STOCK] window={START_DATE}~{END_DATE}")

    scanned = load_state(SCANNED_FILE)
    failed = load_state(FAILED_FILE)

    if MANUAL_STOCK_SYMBOLS:
        work_symbols = set(MANUAL_STOCK_SYMBOLS)
        log(f"[CONFIG][STOCK] manual symbols = {len(work_symbols)}")
    else:
        work_symbols = set(load_universe().keys())
        log(f"[CONFIG][STOCK] universe symbols = {len(work_symbols)}")

     
    total = len(work_symbols)
    processed = 0

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
                start_date=START_DATE,
                end_date=END_DATE,
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
    log(f"[START][INDEX] window={START_DATE}~{END_DATE}")

    total = len(INDEX_SYMBOLS)

    for i, idx in enumerate(INDEX_SYMBOLS, start=1):
        log_progress("INDEX", i, total, idx, "CHECK DB coverage")
        #symbol6 = normalize_stock_code()
        try:
            existing = get_existing_index_days(idx)

            df = load_index_price_em(
                index_code=idx,
                start_date=START_DATE,
                end_date=END_DATE,
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

    log("[DONE][INDEX] loader finished")

# =====================================================
# ================= 3️⃣ STOCK AUDIT ===================
# =====================================================

def run_stock_audit():
    log("[START][AUDIT][STOCK]")
    os.makedirs(AUDIT_OUTPUT_DIR, exist_ok=True)

    df = audit_stock_missing_days(
        engine=engine,
        start_date=START_DATE,
        end_date=END_DATE,
        stock_symbols=None,
    )

    log(f"[AUDIT][STOCK] missing records = {len(df)}")
    log(df.head(20).to_string())

    out = os.path.join(AUDIT_OUTPUT_DIR, "stock_missing_days.csv")
    df.to_csv(out, index=False)
    log(f"[AUDIT][STOCK] report saved: {out}")

# =====================================================
# ================= 4️⃣ INDEX AUDIT ===================
# =====================================================

def run_index_audit():
    log("[START][AUDIT][INDEX]")
    os.makedirs(AUDIT_OUTPUT_DIR, exist_ok=True)

    df_gap, df_health = audit_index_calendar_health(
        engine=engine,
        index_codes=INDEX_SYMBOLS,
        start_date=START_DATE,
        end_date=END_DATE,
    )

    log(f"[AUDIT][INDEX] gap rows = {len(df_gap)}")
    log(df_gap.head(20).to_string())

    df_gap.to_csv(
        os.path.join(AUDIT_OUTPUT_DIR, "index_calendar_gaps.csv"),
        index=False,
    )
    df_health.to_csv(
        os.path.join(AUDIT_OUTPUT_DIR, "index_calendar_health.csv"),
        index=False,
    )

    log("[AUDIT][INDEX] reports saved")

# =====================================================
# ================= PIPELINE ==========================
# =====================================================

def run_data_pipeline():
    log("[START] AsharesScraper")

    with engine.begin() as conn:
        create_tables_if_not_exists(conn)

    run_stock_loader()   # 1️⃣ 拉股票
    run_index_loader()   # 2️⃣ 拉指数
    run_stock_audit()    # 3️⃣ 审核股票
    run_index_audit()    # 4️⃣ 审核指数

    log("[DONE] AsharesScraper finished")

# =====================================================
# ================= MAIN ==============================
# =====================================================

def main():
    args = parse_args()

    # ===== refresh 模式：清空状态 =====
    if args.refresh:
        print("[MODE] refresh enabled, clearing state files")
        clear_state_files()

    run_data_pipeline()


if __name__ == "__main__":
    main()
