import pandas as pd
from sqlalchemy import text
from typing import List, Tuple, Optional


# =====================================================
# CONFIG (冻结)
# =====================================================

BASE_INDEX_SET_DEFAULT = [
    "sh000300",  # 沪深300
    "sh000001",  # 上证
    "sz399006",  # 创业板
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

def audit_stock_missing_days(
    engine,
    start_date: str,
    end_date: str,
    stock_symbols: Optional[List[Tuple[str, str]]] = None,
    base_index: str = "sh000300",
) -> pd.DataFrame:
    """
    只输出 WINDOW_NOT_COVERED 的股票及缺失交易日
    """

    if stock_symbols:
        symbol_filter = "WHERE (symbol, exchange) IN ({})".format(
            ",".join(f"('{s}','{e}')" for s, e in stock_symbols)
        )
    else:
        symbol_filter = ""

    with engine.begin() as conn:
        df_start = pd.read_sql(
            SQL_STOCK_WINDOW_START_MISSING.format(symbol_filter=symbol_filter),
            conn,
            params={
                "start_date": start_date,
                "end_date": end_date,
                "base_index": base_index,
            },
        )

        df_gap = pd.read_sql(
            SQL_STOCK_GAP_MISSING.format(symbol_filter=symbol_filter),
            conn,
            params={
                "start_date": start_date,
                "end_date": end_date,
                "base_index": base_index,
            },
        )

    df = pd.concat([df_start, df_gap], ignore_index=True)

    if df.empty:
        print("[STOCK COVERAGE] No WINDOW_NOT_COVERED stocks.")
        return df

    print("\n[STOCK COVERAGE] Missing trading days:")
    print(df)

    return df


# =====================================================
# 核心函数 2：指数 GAP 交叉审计 + Health
# =====================================================

def audit_index_calendar_health(
    engine,
    index_codes: List[str],
    start_date: str,
    end_date: str,
):
    """
    多指数交易日交叉校验

    返回：
      - df_gap: index_code | missing_date | gap_type
      - df_health: index_code | calendar_health
    """

    if len(index_codes) < 2:
        raise ValueError("index_codes must contain >= 2 items")

    calendars = {}

    with engine.begin() as conn:
        for idx in index_codes:
            df = pd.read_sql(
                """
                SELECT trade_date
                FROM cn_index_daily_price
                WHERE index_code = :idx
                  AND trade_date BETWEEN TO_DATE(:start_date,'YYYYMMDD')
                                      AND TO_DATE(:end_date,'YYYYMMDD')
                ORDER BY trade_date
                """,
                conn,
                params={
                    "idx": idx,
                    "start_date": start_date,
                    "end_date": end_date,
                },
            )
            calendars[idx] = set(df["trade_date"].dt.date.tolist())

    # 交易日交集
    common_days = set.intersection(*calendars.values())

    gap_records = []
    health_records = []

    for idx, days in calendars.items():
        missing = sorted(common_days - days)
        extra = sorted(days - common_days)

        if missing or extra:
            health = "FAIL"
        else:
            health = "PASS"

        health_records.append({
            "index_code": idx,
            "calendar_health": health,
        })

        for d in missing:
            gap_records.append({
                "index_code": idx,
                "missing_date": d,
                "gap_type": "MISSING_IN_INDEX",
            })

        for d in extra:
            gap_records.append({
                "index_code": idx,
                "missing_date": d,
                "gap_type": "EXTRA_IN_INDEX",
            })

    df_gap = pd.DataFrame(gap_records)
    df_health = pd.DataFrame(health_records)

    if df_gap.empty:
        print("[INDEX COVERAGE] All index calendars are consistent.")
    else:
        print("\n[INDEX COVERAGE] Calendar gaps:")
        print(df_gap.sort_values(["index_code", "missing_date"]))

    print("\n[INDEX COVERAGE] Calendar health:")
    print(df_health)

    return df_gap, df_health


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

    if index_codes is None:
        index_codes = BASE_INDEX_SET_DEFAULT

    df_stock = audit_stock_missing_days(
        engine,
        start_date,
        end_date,
        stock_symbols=stock_symbols,
        base_index=index_codes[0],
    )

    df_index_gap, df_index_health = audit_index_calendar_health(
        engine,
        index_codes,
        start_date,
        end_date,
    )

    return df_stock, df_index_gap, df_index_health
