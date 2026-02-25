from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
from datetime import datetime, timedelta
from sqlalchemy import text

import time
 
 
import numpy as np
 
import akshare as ak

from app.utils.oracle_utils import create_table_if_not_exists, get_engine
from app.utils.wireguard_helper import deactivate_tunnel

 

SCHEMA_NAME = "SECOPR"
TABLE_NAME = "CN_FUT_INDEX_HIS"

create_table_sql = f"""
CREATE TABLE {SCHEMA_NAME}.{TABLE_NAME} (
    TRADE_DATE          DATE            NOT NULL,
    SYMBOL              VARCHAR2(20)    NOT NULL,
    MAIN_CONTRACT       VARCHAR2(20)    NOT NULL,
    OPEN_PRICE          NUMBER,
    HIGH_PRICE          NUMBER,
    LOW_PRICE           NUMBER,
    CLOSE_PRICE         NUMBER,
    SETTLE_PRICE        NUMBER,
    PRE_SETTLE          NUMBER,
    VOLUME              NUMBER,
    TURNOVER            NUMBER,
    OPEN_INTEREST       NUMBER,
    VARIETY             VARCHAR2(20),
    SOURCE              VARCHAR2(30),
    CREATED_AT          TIMESTAMP       DEFAULT SYSDATE,
    CONSTRAINT CN_FID_PK PRIMARY KEY (TRADE_DATE, SYMBOL)
)
"""

indexes = [
    f"CREATE INDEX IDX_FUT_DATE ON {SCHEMA_NAME}.{TABLE_NAME} (TRADE_DATE)",
    f"CREATE INDEX IDX_FUT_SYMBOL ON {SCHEMA_NAME}.{TABLE_NAME} (SYMBOL)"
]


@dataclass
class FuturesLoaderTask:
    name: str = "IndexFuturesLoader"
 
        
    def run(self, ctx) -> None:
        cfg = ctx.config
        self.log = ctx.log
        self.engine=ctx.engine
    
        self.load_index_futures_daily(start_date=cfg.start_date, end_date=cfg.end_date)


             
    def load_index_futures_daily(self, start_date: str, end_date: str = None):
         
        create_table_if_not_exists(TABLE_NAME, create_table_sql, indexes)
        deactivate_tunnel("cn")
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
                    MERGE INTO {SCHEMA_NAME}.{TABLE_NAME} t
                    USING (SELECT :TRADE_DATE dt, :SYMBOL sym FROM dual) s
                    ON (t.TRADE_DATE = s.dt AND t.SYMBOL = s.sym)
                    WHEN MATCHED THEN
                        UPDATE SET
                            t.MAIN_CONTRACT   = :MAIN_CONTRACT,
                            t.OPEN_PRICE      = :OPEN_PRICE,
                            t.HIGH_PRICE      = :HIGH_PRICE,
                            t.LOW_PRICE       = :LOW_PRICE,
                            t.CLOSE_PRICE     = :CLOSE_PRICE,
                            t.SETTLE_PRICE    = :SETTLE_PRICE,
                            t.PRE_SETTLE      = :PRE_SETTLE,
                            t.VOLUME          = :VOLUME,
                            t.TURNOVER        = :TURNOVER,
                            t.OPEN_INTEREST   = :OPEN_INTEREST,
                            t.VARIETY         = :VARIETY,
                            t.SOURCE          = :SOURCE,
                            t.CREATED_AT      = :CREATED_AT
                    WHEN NOT MATCHED THEN
                        INSERT (
                            TRADE_DATE, SYMBOL, MAIN_CONTRACT, OPEN_PRICE, HIGH_PRICE, LOW_PRICE,
                            CLOSE_PRICE, SETTLE_PRICE, PRE_SETTLE, VOLUME, TURNOVER, OPEN_INTEREST,
                            VARIETY, SOURCE, CREATED_AT
                        ) VALUES (
                            :TRADE_DATE, :SYMBOL, :MAIN_CONTRACT, :OPEN_PRICE, :HIGH_PRICE, :LOW_PRICE,
                            :CLOSE_PRICE, :SETTLE_PRICE, :PRE_SETTLE, :VOLUME, :TURNOVER, :OPEN_INTEREST,
                            :VARIETY, :SOURCE, :CREATED_AT
                        )
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
    