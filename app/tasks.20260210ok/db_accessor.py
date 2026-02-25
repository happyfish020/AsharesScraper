from __future__ import annotations

from dataclasses import dataclass
from datetime import date as dt_date
from typing import Set

import pandas as pd
from sqlalchemy import text

from app.utils.utils_tools import normalize_stock_code

def infer_exchange_from_code6(code6: str) -> str | None:
    if not isinstance(code6, str) or len(code6) != 6 or not code6.isdigit():
        return None
    if code6.startswith(("6", "9")):
        return "SSE"
    if code6.startswith(("0", "3")):
        return "SZSE"
    if code6.startswith("8"):
        return "BJSE"
    return None

@dataclass
class DbAccessor:
    engine: object
    log: object

    def get_existing_stock_days(self, symbol6: str, start_date: str, end_date: str) -> Set[dt_date]:
        sql = """
            SELECT trade_date
            FROM cn_stock_daily_price
            WHERE symbol = :sym
              AND trade_date BETWEEN TO_DATE(:s,'YYYYMMDD')
                                  AND TO_DATE(:e,'YYYYMMDD')
        """
        with self.engine.begin() as conn:
            rows = conn.execute(text(sql), {"sym": symbol6, "s": start_date, "e": end_date}).fetchall()
        return {r[0].date() for r in rows}

    def get_existing_index_days(self, index_code: str, start_date: str, end_date: str) -> Set[dt_date]:
        sql = """
            SELECT trade_date
            FROM cn_index_daily_price
            WHERE index_code = :idx
              AND trade_date BETWEEN TO_DATE(:s,'YYYYMMDD')
                                  AND TO_DATE(:e,'YYYYMMDD')
        """
        with self.engine.begin() as conn:
            rows = conn.execute(text(sql), {"idx": index_code, "s": start_date, "e": end_date}).fetchall()
        return {r[0].date() for r in rows}

    def insert_missing_stock_days(self, symbol: str, df: pd.DataFrame, missing: Set[dt_date], start_date: str) -> int:
        df = df[df["trade_date"].dt.date.isin(missing)].copy()
        if df.empty:
            return 0
        symbol6 = normalize_stock_code(symbol)
        df["symbol"] = symbol6
        df["window_start"] = pd.to_datetime(start_date, format="%Y%m%d", errors="coerce") if isinstance(start_date, str) else None
        df["exchange"] = infer_exchange_from_code6(symbol6)

        table_cols = [
            "symbol","trade_date","open","close","pre_close","high","low","volume","amount",
            "amplitude","chg_pct","change","turnover_rate","source","window_start","exchange","name",
        ]
        for c in table_cols:
            if c not in df.columns:
                df[c] = None
        df = df[table_cols]

        numeric_cols = ['open','close','pre_close','high','low','volume','amount','amplitude','chg_pct','change','turnover_rate']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: None if pd.isna(x) else f"{float(x):.10g}")

        try:
            df.to_sql("cn_stock_daily_price", self.engine, if_exists="append", index=False, chunksize=200)
            return len(df)
        except Exception as e:
            self.log.info(f"Symbol: {symbol6} to DB failed - {e}")
            raise

    def insert_missing_index_days(self, index_code: str, df: pd.DataFrame, missing: Set[dt_date]) -> int:
        df = df[df["trade_date"].dt.date.isin(missing)].copy()
        if df.empty:
            return 0
        df["index_code"] = index_code
        df["source"] = df.get("source", "eastmoney")

        table_cols = ["index_code","trade_date","open","close","high","low","volume","amount","source","pre_close","chg_pct"]
        for c in table_cols:
            if c not in df.columns:
                df[c] = None
        df = df[table_cols]

        try:
            df.to_sql("cn_index_daily_price", self.engine, if_exists="append", index=False, chunksize=200)
            return len(df)
        except Exception as e:
            self.log.info(f"Index: {index_code} to DB failed - {e}")
            raise
