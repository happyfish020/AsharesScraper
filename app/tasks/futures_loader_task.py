from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
from datetime import datetime, timedelta
from sqlalchemy import text

import time
 
 
import numpy as np
  
import akshare as ak

from app.utils.wireguard_helper import activate_tunnel, deactivate_tunnel

 

TABLE_NAME = "CN_FUT_INDEX_HIS"

MYSQL_CREATE_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
    TRADE_DATE          DATE            NOT NULL,
    SYMBOL              VARCHAR(20)     NOT NULL,
    MAIN_CONTRACT       VARCHAR(20)     NOT NULL,
    OPEN_PRICE          DECIMAL(20,6),
    HIGH_PRICE          DECIMAL(20,6),
    LOW_PRICE           DECIMAL(20,6),
    CLOSE_PRICE         DECIMAL(20,6),
    SETTLE_PRICE        DECIMAL(20,6),
    PRE_SETTLE          DECIMAL(20,6),
    VOLUME              DECIMAL(24,6),
    TURNOVER            DECIMAL(24,6),
    OPEN_INTEREST       DECIMAL(24,6),
    VARIETY             VARCHAR(20),
    SOURCE              VARCHAR(30),
    CREATED_AT          DATETIME        DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (TRADE_DATE, SYMBOL),
    KEY IDX_FUT_DATE (TRADE_DATE),
    KEY IDX_FUT_SYMBOL (SYMBOL)
)
"""


@dataclass
class FuturesLoaderTask:
    name: str = "IndexFuturesLoader"

    def _table_ref(self) -> str:
        return TABLE_NAME

    def _ensure_table(self) -> None:
        with self.engine.begin() as conn:
            conn.execute(text(MYSQL_CREATE_TABLE_SQL))
 
        
    def run(self, ctx) -> None:
        cfg = ctx.config
        self.log = ctx.log
        self.engine=ctx.engine
        deactivate_tunnel("cn")
        self.load_index_futures_daily(start_date=cfg.start_date, end_date=cfg.end_date)
        activate_tunnel("cn")  

             
    def load_index_futures_daily(self, start_date: str, end_date: str = None):
         
        self._ensure_table()
        
        # 日期标准化
        def to_date_str(d):
            if isinstance(d, datetime):
                return d.strftime("%Y%m%d")
            return str(d).replace("-", "")
    
        start_dt = datetime.strptime(to_date_str(start_date), "%Y%m%d")
        if end_date:
            end_dt = datetime.strptime(to_date_str(end_date), "%Y%m%d")
        else:
            end_dt = datetime.today()
    
        current_date = start_dt
        total_records = 0
        failed_dates = []
    
        while current_date <= end_dt:
            date_str = current_date.strftime("%Y%m%d")
            self.log.info(f"Processing date: {date_str}")
    
            try:
                # 单日拉取
                df = ak.get_cffex_daily(date = date_str)
    
                if df.empty:
                    self.log.warning(f"No data for {date_str}")
                    current_date += timedelta(days=1)
                    continue
    
                # 过滤股指期货
                target_prefix = ('IF', 'IH', 'IC', 'IM')
                df = df[df['symbol'].str.startswith(target_prefix)].copy()
    
                if df.empty:
                    self.log.info(f"No IF/IH/IC/IM data for {date_str}")
                    current_date += timedelta(days=1)
                    continue
    
                # 按 symbol 分组，取成交量最大的作为主力
                df = df.sort_values(['symbol', 'volume'], ascending=[True, False])
                df_main = df.drop_duplicates('symbol', keep='first')
    
                # 重命名映射（只用实际存在的列）
                rename_map = {
                    'date': 'TRADE_DATE',
                    'symbol': 'SYMBOL',
                    'open': 'OPEN_PRICE',
                    'high': 'HIGH_PRICE',
                    'low': 'LOW_PRICE',
                    'close': 'CLOSE_PRICE',
                    'settle': 'SETTLE_PRICE',
                    'pre_settle': 'PRE_SETTLE',
                    'volume': 'VOLUME',
                    'turnover': 'TURNOVER',
                    'open_interest': 'OPEN_INTEREST',
                    'variety': 'VARIETY',
                }
                df_main = df_main.rename(columns={k: v for k, v in rename_map.items() if k in df_main.columns})
    
                # 主力合约代码就是 symbol 本身（已筛选最大 volume）
                df_main['MAIN_CONTRACT'] = df_main['SYMBOL']
    
                # 类型转换
                numeric_cols = ['OPEN_PRICE', 'HIGH_PRICE', 'LOW_PRICE', 'CLOSE_PRICE', 'SETTLE_PRICE',
                                'PRE_SETTLE', 'VOLUME', 'TURNOVER', 'OPEN_INTEREST']
                for col in numeric_cols:
                    if col in df_main.columns:
                        df_main[col] = pd.to_numeric(df_main[col], errors='coerce')
    
                df_main['TRADE_DATE'] = pd.to_datetime(df_main['TRADE_DATE'], errors='coerce').dt.date
                df_main = df_main.replace([pd.NA, pd.NaT, np.nan, np.inf, -np.inf], None)
    
                df_main['SOURCE'] = 'AKSHARE_FUTURES_DAILY'
                df_main['CREATED_AT'] = pd.Timestamp.now()
    
                records = df_main.to_dict('records')
                if not records:
                    current_date += timedelta(days=1)
                    continue
    
                # MERGE 插入（只绑定存在的字段）
                with self.engine.connect() as conn:
                    merge_sql = f"""
                    INSERT INTO {self._table_ref()} (
                        TRADE_DATE, SYMBOL, MAIN_CONTRACT, OPEN_PRICE, HIGH_PRICE, LOW_PRICE,
                        CLOSE_PRICE, SETTLE_PRICE, PRE_SETTLE, VOLUME, TURNOVER, OPEN_INTEREST,
                        VARIETY, SOURCE, CREATED_AT
                    ) VALUES (
                        :TRADE_DATE, :SYMBOL, :MAIN_CONTRACT, :OPEN_PRICE, :HIGH_PRICE, :LOW_PRICE,
                        :CLOSE_PRICE, :SETTLE_PRICE, :PRE_SETTLE, :VOLUME, :TURNOVER, :OPEN_INTEREST,
                        :VARIETY, :SOURCE, :CREATED_AT
                    )
                    ON DUPLICATE KEY UPDATE
                        MAIN_CONTRACT = VALUES(MAIN_CONTRACT),
                        OPEN_PRICE = VALUES(OPEN_PRICE),
                        HIGH_PRICE = VALUES(HIGH_PRICE),
                        LOW_PRICE = VALUES(LOW_PRICE),
                        CLOSE_PRICE = VALUES(CLOSE_PRICE),
                        SETTLE_PRICE = VALUES(SETTLE_PRICE),
                        PRE_SETTLE = VALUES(PRE_SETTLE),
                        VOLUME = VALUES(VOLUME),
                        TURNOVER = VALUES(TURNOVER),
                        OPEN_INTEREST = VALUES(OPEN_INTEREST),
                        VARIETY = VALUES(VARIETY),
                        SOURCE = VALUES(SOURCE),
                        CREATED_AT = VALUES(CREATED_AT)
                    """
    
                    conn.execute(text(merge_sql), records)
                    conn.commit()
    
                inserted = len(records)
                total_records += inserted
                self.log.info(f"Date {date_str}: upserted {inserted} records")
    
                time.sleep(0.8)  # 防限流
    
            except Exception as e:
                self.log.error(f"Date {date_str} failed: {str(e)}")
                failed_dates.append(date_str)
    
            current_date += timedelta(days=1)
    
        self.log.info(f"Load completed. Total records upserted: {total_records}")
        if failed_dates:
            self.log.warning(f"Failed dates ({len(failed_dates)}): {', '.join(failed_dates[:5])} ...")
    
