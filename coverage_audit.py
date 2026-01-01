import pandas as pd
from sqlalchemy import text
from typing import List, Tuple, Optional


###############################################

# =====================================================
# ==================== UTIL ===========================
# =====================================================

def log(msg: str):
    print(msg, flush=True)


def log_progress(stage: str, cur: int, total: int, symbol: str, msg: str):
    pct = (cur / total) * 100 if total else 0
    log(f"[{stage}] ({cur}/{total} | {pct:6.2f}%) {symbol} {msg}")




# =====================================================
# CONFIG (冻结)
# =====================================================

BASE_INDEX_SET_DEFAULT = [
    "sh000300",  # 沪深300
    "sh000001",  # 上证
    "sz399006",  # 创业板
    "sh000688",   # 科创50  #"KC50": 
    "sh000905",  # 中证500

]

# =====================================================
# SQL: 股票缺失审计
# =====================================================

SQL_STOCK_WINDOW_START_MISSING = """
WITH base_start AS (
    SELECT MIN(trade_date) AS trade_date
    FROM cn_index_daily_price
    WHERE index_code = :base_index
      AND trade_date BETWEEN TO_DATE(:start_date,'YYYYMMDD')
                          AND TO_DATE(:end_date,'YYYYMMDD')
),
targets AS (
    SELECT DISTINCT symbol, exchange
    FROM cn_stock_daily_price
    {symbol_filter}
)
SELECT
    t.symbol,
    t.exchange,
    b.trade_date AS missing_date,
    'WINDOW_START' AS missing_type
FROM targets t
CROSS JOIN base_start b
LEFT JOIN cn_stock_daily_price p
  ON p.symbol = t.symbol
 AND p.exchange = t.exchange
 AND p.trade_date = b.trade_date
WHERE p.trade_date IS NULL
"""

SQL_STOCK_GAP_MISSING = """
WITH base_calendar AS (
    SELECT trade_date
    FROM cn_index_daily_price
    WHERE index_code = :base_index
      AND trade_date BETWEEN TO_DATE(:start_date,'YYYYMMDD')
                          AND TO_DATE(:end_date,'YYYYMMDD')
),
targets AS (
    SELECT DISTINCT symbol, exchange
    FROM cn_stock_daily_price
    {symbol_filter}
),
expected AS (
    SELECT t.symbol, t.exchange, c.trade_date
    FROM targets t
    CROSS JOIN base_calendar c
),
actual AS (
    SELECT symbol, exchange, trade_date
    FROM cn_stock_daily_price
    WHERE trade_date BETWEEN TO_DATE(:start_date,'YYYYMMDD')
                          AND TO_DATE(:end_date,'YYYYMMDD')
)
SELECT
    e.symbol,
    e.exchange,
    e.trade_date AS missing_date,
    'GAP' AS missing_type
FROM expected e
LEFT JOIN actual a
  ON a.symbol = e.symbol
 AND a.exchange = e.exchange
 AND a.trade_date = e.trade_date
WHERE a.trade_date IS NULL
ORDER BY e.symbol, e.trade_date
"""


# =====================================================
# 核心函数 1：股票缺失审计
# =====================================================
import pandas as pd
import akshare as ak
from sqlalchemy import create_engine
from datetime import datetime
import math

 
import pandas as pd
import math

def audit_stock_missing_days(
    engine,
    base_index_df: pd.DataFrame,
    start_date: str,
    end_date: str,
    stock_symbols: list | None = None,
) -> pd.DataFrame:
    """
    审计股票在指定区间内的数据缺失情况（WINDOW_START 和 GAP 类型）
    
    不内部获取交易日历，直接使用外部传入的基准指数 DataFrame 作为标准
    
    Parameters:
    - engine: SQLAlchemy engine
    - base_index_df: pd.DataFrame - 基准指数日线数据（必须包含 'date' 列）
    - start_date: str - 'YYYY-MM-DD'
    - end_date: str - 'YYYY-MM-DD'
    - stock_symbols: list[str] | None - 指定股票列表；None 表示数据库中所有股票
    
    Returns:
    - pd.DataFrame: 包含 symbol, missing_type, missing_date, missing_count 等
    """
    log(f"开始审计股票缺失数据 [{start_date} ~ {end_date}]，使用外部传入基准交易日历")

    # Step 1: 从传入的 base_index_df 提取预期交易日集合
    try:
        base_dates = set(pd.to_datetime(base_index_df['date']).dt.date)
        sorted_base_dates = sorted(base_dates)
        log(f"基准交易日总数: {len(sorted_base_dates)}")
    except Exception as e:
        log(f"解析基准交易日历失败: {e}")
        raise

    if not base_dates:
        log("基准交易日为空，无法审计")
        return pd.DataFrame(columns=['symbol', 'missing_type', 'missing_date', 'missing_count'])

    # Step 2: 获取待审计股票列表
    if stock_symbols is None:
        symbol_query = "SELECT DISTINCT symbol FROM cn_stock_daily_price"
        all_symbols_df = pd.read_sql(symbol_query, engine)
        stock_symbols = all_symbols_df['symbol'].tolist()
        log(f"自动获取数据库中股票总数: {len(stock_symbols)} 只")
    else:
        log(f"指定审计股票数: {len(stock_symbols)} 只")

    if not stock_symbols:
        log("无股票可审计")
        return pd.DataFrame(columns=['symbol', 'missing_type', 'missing_date', 'missing_count'])

    # Step 3: 分批查询数据库现有记录（解决 ORA-01795）
    batch_size = 900
    batches = math.ceil(len(stock_symbols) / batch_size)
    log(f"分 {batches} 批查询数据库现有交易日记录")

    existing_records = []
    for i in range(0, len(stock_symbols), batch_size):
        batch_symbols = stock_symbols[i:i + batch_size]
        symbols_str = "','".join(batch_symbols)
        query = f"""
            SELECT symbol, TRADE_DATE
            FROM cn_stock_daily_price
            WHERE symbol IN ('{symbols_str}')
              AND TRADE_DATE BETWEEN TO_DATE('{start_date}', 'YYYY-MM-DD')
                                 AND TO_DATE('{end_date}', 'YYYY-MM-DD')
        """
        try:
            batch_df = pd.read_sql(query, engine)
            if not batch_df.empty:
                batch_df['trade_date'] = pd.to_datetime(batch_df['trade_date']).dt.date
                existing_records.append(batch_df)
            log(f"批次 {i//batch_size + 1}/{batches} 查询完成，记录数: {len(batch_df)}")
        except Exception as e:
            log(f"批次查询失败: {e}")
            raise

    # 合并所有批次
    existing_df = pd.concat(existing_records, ignore_index=True) if existing_records else pd.DataFrame(columns=['symbol', 'trade_date'])

    # Step 4: 计算每只股票的缺失
    missing_records = []

    for symbol in stock_symbols:
        stock_dates = set(existing_df[existing_df['symbol'] == symbol]['trade_date'])
        missing_dates = base_dates - stock_dates

        if not missing_dates:
            continue

        sorted_missing = sorted(missing_dates)

        # WINDOW_START 判断
        window_start_date = sorted_base_dates[0]
        continuous_from_start = []
        for d in sorted_base_dates:
            if d in missing_dates:
                continuous_from_start.append(d)
            else:
                break

        if continuous_from_start and continuous_from_start[0] == window_start_date:
            missing_records.append({
                'symbol': symbol,
                'missing_type': 'WINDOW_START',
                'missing_date': window_start_date.strftime('%Y-%m-%d'),
                'missing_count': len(continuous_from_start)
            })

        # GAP 类型
        gap_dates = [d for d in sorted_missing if d not in continuous_from_start]
        for gap_date in gap_dates:
            missing_records.append({
                'symbol': symbol,
                'missing_type': 'GAP',
                'missing_date': gap_date.strftime('%Y-%m-%d'),
                'missing_count': 1
            })

    # Step 5: 输出结果
    result_df = pd.DataFrame(missing_records)
    if not result_df.empty:
        result_df = result_df.sort_values(['missing_type', 'symbol', 'missing_date'])
        window_count = len(result_df[result_df['missing_type'] == 'WINDOW_START'])
        gap_count = len(result_df[result_df['missing_type'] == 'GAP'])
        log(f"股票审计完成，发现缺失 {len(result_df)} 条（WINDOW_START: {window_count}，GAP: {gap_count}）")
    else:
        log("股票审计完成，未发现缺失数据")

    return result_df
# =====================================================
# 核心函数 2：指数 GAP 交叉审计 + Health
# =====================================================
 

def audit_index_missing_days(
    engine,
    base_index_df: pd.DataFrame,
    index_codes: list,
    start_date: str,
    end_date: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    审计多个指数在指定区间的日历健康情况（相对于实时基准交易日历）
    
    参照 audit_stock_missing_days 风格：从数据库读取已有数据，分批查询
    
    Parameters:
    - base_index_df: pd.DataFrame - 实时获取的基准指数日线（包含 'date' 列）
    - index_codes: list[str] - 要审计的指数代码列表
    - start_date: str - 'YYYY-MM-DD'
    - end_date: str - 'YYYY-MM-DD'
    
    Returns:
    - gap_df: pd.DataFrame - GAP 类型缺失
    - window_start_df: pd.DataFrame - WINDOW_START 类型缺失
    """
    log(f"开始审计指数日历健康 [{start_date} ~ {end_date}]，审计指数数: {len(index_codes)}")

    # Step 1: 从传入的 base_index_df 提取标准交易日
    try:
        base_dates = set(pd.to_datetime(base_index_df['date']).dt.date)
        sorted_base_dates = sorted(base_dates)
        log(f"基准交易日总数: {len(sorted_base_dates)}")
    except Exception as e:
        log(f"解析基准交易日历失败: {e}")
        raise

    if not base_dates:
        log("基准交易日为空，无法审计指数")
        empty_df = pd.DataFrame(columns=['index_code', 'missing_type', 'missing_date', 'missing_count'])
        return empty_df.copy(), empty_df.copy()

    if not index_codes:
        log("无指数可审计")
        empty_df = pd.DataFrame(columns=['index_code', 'missing_type', 'missing_date', 'missing_count'])
        return empty_df.copy(), empty_df.copy()

    # Step 2: 分批从数据库查询指数已有交易日（解决 ORA-01795）
    batch_size = 900
    batches = math.ceil(len(index_codes) / batch_size)
    log(f"分 {batches} 批查询数据库中指数现有记录")

    existing_records = []
    for i in range(0, len(index_codes), batch_size):
        batch_codes = index_codes[i:i + batch_size]
        codes_str = "','" .join(batch_codes)
        query = f"""
            SELECT index_code, TRADE_DATE
            FROM cn_index_daily_price
            WHERE index_code IN ('{codes_str}')
              AND TRADE_DATE BETWEEN TO_DATE('{start_date}', 'YYYY-MM-DD')
                                 AND TO_DATE('{end_date}', 'YYYY-MM-DD')
        """
        try:
            batch_df = pd.read_sql(query, engine)
            if not batch_df.empty:
                batch_df['trade_date'] = pd.to_datetime(batch_df['trade_date']).dt.date
                existing_records.append(batch_df)
            log(f"指数批次 {i//batch_size + 1}/{batches} 查询完成，记录数: {len(batch_df)}")
        except Exception as e:
            log(f"指数批次查询失败: {e}")
            raise

    # 合并结果
    existing_df = pd.concat(existing_records, ignore_index=True) if existing_records else pd.DataFrame(columns=['index_code', 'trade_date'])

    # Step 3: 计算每个指数的缺失
    gap_records = []
    window_start_records = []

    window_start_date = sorted_base_dates[0]  # 区间起始日

    for index_code in index_codes:
        index_dates = set(existing_df[existing_df['index_code'] == index_code]['trade_date'])
        missing_dates = base_dates - index_dates

        if not missing_dates:
            continue  # 该指数完整，无缺失

        sorted_missing = sorted(missing_dates)

        # WINDOW_START 判断：从起始日期连续缺失
        continuous_from_start = []
        for d in sorted_base_dates:
            if d in missing_dates:
                continuous_from_start.append(d)
            else:
                break

        if continuous_from_start and continuous_from_start[0] == window_start_date:
            window_start_records.append({
                'index_code': index_code,
                'missing_type': 'WINDOW_START',
                'missing_date': window_start_date.strftime('%Y-%m-%d'),
                'missing_count': len(continuous_from_start)
            })

        # GAP 类型：非起始部分的缺失
        gap_dates = [d for d in sorted_missing if d not in continuous_from_start]
        for gap_date in gap_dates:
            gap_records.append({
                'index_code': index_code,
                'missing_type': 'GAP',
                'missing_date': gap_date.strftime('%Y-%m-%d'),
                'missing_count': 1
            })

    # Step 4: 转为 DataFrame 并排序
    gap_df = pd.DataFrame(gap_records)
    window_start_df = pd.DataFrame(window_start_records)

    if not gap_df.empty:
        gap_df = gap_df.sort_values(['index_code', 'missing_date'])
    if not window_start_df.empty:
        window_start_df = window_start_df.sort_values(['index_code'])

    total_gaps = len(gap_df)
    total_windows = len(window_start_df)
    log(f"指数日历审计完成，GAP 缺失 {total_gaps} 条，WINDOW_START 缺失 {total_windows} 条")

    return gap_df, window_start_df 
 
# =====================================================
# 统一入口（可选）
# =====================================================

def run_full_coverage_audit(
    engine,
    start_date: str,
    end_date: str,
    stock_symbols: Optional[List[Tuple[str, str]]] = None,
    index_codes: Optional[List[str]] = None,
):
    """
    一次性跑完：
      - 股票缺失审计
      - 指数 GAP + health
    """

    base_index = "sh000300"  # 或 "sh000001"

    base_index_df = ak.stock_zh_index_daily_em(
        symbol=base_index,
        start_date=start_date,
        end_date=end_date,
    )
    
    if base_index_df.empty:
        print("获取基准交易日历失败")
    else:
        # Step 2: 股票缺失审计
        gap_df, window_start_df = audit_index_missing_days(
            engine=engine,
            base_index_df=base_index_df,
            index_codes=index_codes,
            start_date=start_date,
            end_date=end_date,
        )
        gap_df.to_csv("audit_reports/audit_index_gap.csv", index=False)
        window_start_df.to_csv("audit_reports/audit_index_window_start.csv", index=False)

        ##############################################################
        stock_missing_df = audit_stock_missing_days(
            engine=engine,
            base_index_df=base_index_df,
            start_date=start_date,
            end_date=end_date,
        )
        stock_missing_df.to_csv("audit_reports/audit_stock_missing.csv", index=False)

        if len(stock_missing_df) >0:
            return stock_missing_df['symbol'].tolist()
              
    
        # Step 3: 指数健康审计
        #index_codes = ["sh000001", "sh000300", "sz399001", "sz399006", "sh000688"]
