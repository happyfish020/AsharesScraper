from __future__ import annotations
from dataclasses import dataclass
from logging import log
import traceback
import pandas as pd
import numpy as np
from datetime import date, datetime
from sqlalchemy import create_engine, text, inspect
import akshare as ak
import time
import  baostock as bs

from app.utils.oracle_utils import create_table_if_not_exists, get_engine
from app.utils.wireguard_helper import activate_tunnel, toggle_vpn  

#from ak_fund_etf_spot_em import load_fund_etf_hist_baostock, load_spot_as_hist_today

ETF_HIST_TABLE = "CN_FUND_ETF_HIST_EM"


SCHEMA_NAME = "SECOPR"

etf_hist_sql = f"""
CREATE TABLE {SCHEMA_NAME}.{ETF_HIST_TABLE} (
    CODE            VARCHAR2(20)    NOT NULL,
    DATA_DATE       DATE            NOT NULL,
    OPEN_PRICE      NUMBER,
    CLOSE_PRICE     NUMBER,
    HIGH_PRICE      NUMBER,
    LOW_PRICE       NUMBER,
    VOLUME          NUMBER,
    AMOUNT          NUMBER,
    AMPLITUDE       NUMBER,
    CHANGE_PCT      NUMBER,
    CHANGE_AMOUNT   NUMBER,
    TURNOVER_RATE   NUMBER,
    ADJUST_TYPE     VARCHAR2(10)    NOT NULL,
    SOURCE          VARCHAR2(30),
    CREATED_AT      TIMESTAMP       DEFAULT SYSDATE,
    CONSTRAINT CN_FEH_PK PRIMARY KEY (CODE, DATA_DATE, ADJUST_TYPE)
)
"""

etf_hist_indexes = [
    f"CREATE INDEX IDX_FEH_CODE ON {SCHEMA_NAME}.{ETF_HIST_TABLE} (CODE)",
    f"CREATE INDEX IDX_FEH_DATE ON {SCHEMA_NAME}.{ETF_HIST_TABLE} (DATA_DATE)",
    f"CREATE INDEX IDX_FEH_ADJUST ON {SCHEMA_NAME}.{ETF_HIST_TABLE} (ADJUST_TYPE)"
]


@dataclass
class EtfLoaderTask:
    name: str = "ETFLoader"

    def run(self, ctx) -> None:
        cfg = ctx.config
        self.log = ctx.log
        self.engine=ctx.engine
        if cfg.look_back_days >= 1:
            self.load_fund_etf_hist_baostock(start_date=cfg.start_date, end_date=cfg.end_date)
            ctx.self.log.info("[DONE] load_fund_etf_hist_baostock finished")
        else:
            self.load_spot_as_hist_today()
            ctx.self.log.info("[DONE] load_spot_as_hist_today finished")


    
    def load_fund_etf_hist_baostock(self, start_date: str, end_date: str, frequency: str = "d", adjustflag: str = "3"):
        """
        使用 baostock 下载 ETF 历史行情 并存入 CN_FUND_ETF_HIST_EM 风格的表
        
        参数说明：
        - frequency: "d"=日线, "w"=周线, "m"=月线, "5"/"15"/"30"/"60"=分钟线
        - adjustflag: "1"=不复权, "2"=前复权, "3"=后复权（默认后复权，与多数人习惯一致）
        
        注意：ETF代码需带前缀，例如 'sh.510300', 'sz.159941'
              spot表里的 CODE 如果没有前缀，需要提前补上 sh./sz.
        """
        self.log.info(f"=== Historical load (baostock) started: {start_date} → {end_date} | frequency={frequency} | adjustflag={adjustflag} ===")
        create_table_if_not_exists(ETF_HIST_TABLE, etf_hist_sql, etf_hist_indexes)
    
        def normalize_date_ymd(d: str) -> str:
            d = d.replace("-", "").strip()
            if len(d) == 8 and d.isdigit():
                return f"{d[:4]}-{d[4:6]}-{d[6:8]}"
            raise ValueError(f"Date format error: {d} (expect YYYYMMDD or YYYY-MM-DD)")
    
        start_dt = normalize_date_ymd(start_date)
        end_dt   = normalize_date_ymd(end_date)
    
        # 登录 baostock（每次运行登录一次即可）
        activate_tunnel("cn")
        lg = bs.login()
        time.sleep(1)

        if lg.error_code != '0':
            self.log.error(f"baostock login failed: {lg.error_msg}")
            return
        self.log.info("baostock login success")
    
        # 从 spot 表取代码列表（假设 CODE 已带 sh./sz. 前缀）
        with self.engine.connect() as conn:
            df_codes = pd.read_sql(f"SELECT DISTINCT CODE FROM {SCHEMA_NAME}.{ETF_HIST_TABLE}", conn)
        
        if df_codes.empty:
            self.log.info("No ETF codes in spot table.")
            bs.logout()
            return
        df_codes['orig_code'] = df_codes['code']
        # 代码前缀处理函数
        def add_prefix(code: str) -> str:
            code = str(code).strip()
            if code.startswith('5') :
                return f"sh.{code}"
            elif code.startswith('15') or  code.startswith('16'):
                return f"sz.{code}"
            else:
                raise Exception(f"Wrong etf code:{code}")
        codes = df_codes['code'].astype(str).sort_values().unique().tolist()    
        #codes = df_codes['code'].astype(str).apply(add_prefix).sort_values().unique().tolist()
        self.log.info(f"Processing {len(codes)} symbols with baostock (after prefix).")
    
        self.log.info(f"Processing {len(codes)} symbols with baostock.")
    
        total = 0
        failed = []
    
        fields = "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,pctChg"
    
        processed = 0
        
        
        for i, symbol in enumerate(codes, 1):
            processed += 1
            if processed % 20 == 0:
               time.sleep(1)
            self.log.info(f"[{i:4d}/{len(codes)}] {symbol} ...  ")
            max_retries = 5
            for attempt in range(1, max_retries + 1):
                try:
                    rs = bs.query_history_k_data_plus(
                        code=symbol,
                        fields=fields,
                        start_date=start_dt,
                        end_date=end_dt,
                        frequency=frequency,
                        adjustflag=adjustflag
                    )
        
                    if rs.error_code != '0':
                        self.log.info(f"query error: {rs.error_msg}")
                        failed.append(symbol)
                        continue
        
                    df = rs.get_data()
        
                    if df.empty:
                        self.log.info("empty")
                        continue
        
                    # ─────────────── 先重命名 ───────────────
                    rename_map = {
                        'date':       'DATA_DATE',
                        'open':       'OPEN_PRICE',
                        'high':       'HIGH_PRICE',
                        'low':        'LOW_PRICE',
                        'close':      'CLOSE_PRICE',
                        'preclose':   'PRE_CLOSE',          # 改名避免冲突
                        'volume':     'VOLUME',
                        'amount':     'AMOUNT',
                        'turn':       'TURNOVER_RATE',
                        'pctChg':     'CHANGE_PCT',
                    }
                    df = df.rename(columns=rename_map)
        
                    # ─────────────── 尽早转为数值类型 ───────────────
                    price_cols = ['OPEN_PRICE', 'HIGH_PRICE', 'LOW_PRICE', 'CLOSE_PRICE', 'PRE_CLOSE',
                                  'VOLUME', 'AMOUNT', 'TURNOVER_RATE', 'CHANGE_PCT']
                    for col in price_cols:
                        if col in df.columns:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
        
                    # ─────────────── 现在再做计算 ───────────────
                    df['CHANGE_AMOUNT'] = df['CLOSE_PRICE'] - df['PRE_CLOSE']
                    df['AMPLITUDE'] = ((df['HIGH_PRICE'] - df['LOW_PRICE']) / df['PRE_CLOSE'] * 100
                                       ).where(df['PRE_CLOSE'] != 0, np.nan)
        
                    # 可以删掉不再需要的列
                    df = df.drop(columns=['PRE_CLOSE'], errors='ignore')
        
                    # 日期处理
                    df['DATA_DATE'] = pd.to_datetime(df['DATA_DATE']).dt.date
        
                    # 最后的清理（可选，to_numeric 已经做了大部分）
                    df = df.replace([np.nan, np.inf, -np.inf], None)
        
                    # 存原始 code（不带 sh./sz.）
                    df['CODE'] = symbol
                    df['ADJUST_TYPE'] = {"1": "NONE", "2": "PRE", "3": "POST"}.get(adjustflag, adjustflag)
                    df['SOURCE'] = 'BAOSTOCK'
                    df['CREATED_AT'] = pd.Timestamp.now()
        
                    # keep_cols 里把 PRE_CLOSE 去掉
                    keep_cols = [
                        'CODE', 'DATA_DATE', 'OPEN_PRICE', 'CLOSE_PRICE', 'HIGH_PRICE', 'LOW_PRICE',
                        'VOLUME', 'AMOUNT', 'AMPLITUDE', 'CHANGE_PCT', 'CHANGE_AMOUNT', 'TURNOVER_RATE',
                        'ADJUST_TYPE', 'SOURCE', 'CREATED_AT'
                    ]
                    df = df[[c for c in keep_cols if c in df.columns]]
        
         
        
                    records = df.to_dict('records')
        
                    # 使用你原来的 MERGE 逻辑插入/更新
                    merge_sql = f"""
                        MERGE INTO {SCHEMA_NAME}.{ETF_HIST_TABLE} t
                        USING (SELECT :CODE code, :DATA_DATE dt, :ADJUST_TYPE adj FROM dual) s
                        ON (t.CODE = s.code AND t.DATA_DATE = s.dt AND t.ADJUST_TYPE = s.adj)
                        WHEN MATCHED THEN
                            UPDATE SET
                                t.OPEN_PRICE = :OPEN_PRICE,
                                t.CLOSE_PRICE = :CLOSE_PRICE,
                                t.HIGH_PRICE = :HIGH_PRICE,
                                t.LOW_PRICE = :LOW_PRICE,
                                t.VOLUME = :VOLUME,
                                t.AMOUNT = :AMOUNT,
                                t.AMPLITUDE = :AMPLITUDE,
                                t.CHANGE_PCT = :CHANGE_PCT,
                                t.CHANGE_AMOUNT = :CHANGE_AMOUNT,
                                t.TURNOVER_RATE = :TURNOVER_RATE,
                                t.SOURCE = :SOURCE,
                                t.CREATED_AT = :CREATED_AT
                        WHEN NOT MATCHED THEN
                            INSERT (CODE, DATA_DATE, OPEN_PRICE, CLOSE_PRICE, HIGH_PRICE, LOW_PRICE,
                                    VOLUME, AMOUNT, AMPLITUDE, CHANGE_PCT, CHANGE_AMOUNT, TURNOVER_RATE,
                                    ADJUST_TYPE, SOURCE, CREATED_AT)
                            VALUES (:CODE, :DATA_DATE, :OPEN_PRICE, :CLOSE_PRICE, :HIGH_PRICE, :LOW_PRICE,
                                    :VOLUME, :AMOUNT, :AMPLITUDE, :CHANGE_PCT, :CHANGE_AMOUNT, :TURNOVER_RATE,
                                    :ADJUST_TYPE, :SOURCE, :CREATED_AT)
                        """
                    with self.engine.connect() as conn:
                        conn.execute(text(merge_sql), records)
                        conn.commit()
        
                    count = len(records)
                    total += count
                    self.log.info(f"ok ({count})")
        
                    time.sleep(0.4)  # baostock 免费版有频率限制，建议稍作间隔
        
                except Exception as e:
                    if attempt == max_retries:
                        raise RuntimeError(f"获取etf 数据失败（已重试 {max_retries} 次）：{str(e)}" ) from e
                    self.log.info(f"error: {str(e)}")
                    failed.append(symbol)
                
                #end tried
            # end for 
    
        bs.logout()
        self.log.info(f"\nHistorical load (baostock) finished. Total records: {total}")
        if failed:
            self.log.info(f"Failed symbols: {len(failed)} | {', '.join(failed[:8])}{'...' if len(failed)>8 else ''}")
     
     
     # ==============================
    #

    def load_spot_as_hist_today(self):
        """
        盘后运行：用 ak.fund_etf_spot_em() 的当天数据作为历史日K补入 CN_FUND_ETF_HIST_EM
        只插入当天缺失的记录，ADJUST_TYPE='SPOT_TODAY'
        """
        today_str = date.today().strftime("%Y-%m-%d")
        self.log.info(f"=== 盘后补当天历史记录 ({today_str}) ===")
    
        create_table_if_not_exists(ETF_HIST_TABLE, etf_hist_sql, etf_hist_indexes)
            
        #switch_wire_guard("cn")
        activate_tunnel("cn")
        try:
            df = ak.fund_etf_spot_em()
            if df.empty:
                self.log.info("ak.fund_etf_spot_em() 返回空数据")
                return
    
            self.log.info(f"获取到 {len(df)} 条当天 spot 数据")
    
            # 列重命名 & 映射到 hist 表字段
            rename_map = {
                '代码': 'CODE',
                '最新价': 'CLOSE_PRICE',       # 用最新价近似收盘
                '开盘价': 'OPEN_PRICE',
                '最高价': 'HIGH_PRICE',
                '最低价': 'LOW_PRICE',
                '成交量': 'VOLUME',
                '成交额': 'AMOUNT',
                '涨跌幅': 'CHANGE_PCT',
                '涨跌额': 'CHANGE_AMOUNT',
                '换手率': 'TURNOVER_RATE',
                '数据日期': 'DATA_DATE',       # 尽量用这个，如果没有则用 today
            }
    
            df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    
            # 类型转换
            numeric_cols = ['OPEN_PRICE', 'CLOSE_PRICE', 'HIGH_PRICE', 'LOW_PRICE',
                            'VOLUME', 'AMOUNT', 'CHANGE_PCT', 'CHANGE_AMOUNT', 'TURNOVER_RATE']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col].astype(str).str.replace('%', ''), errors='coerce')
    
            # 日期处理（优先用接口返回的 '数据日期'，否则用系统当天）
            if 'DATA_DATE' in df.columns:
                df['DATA_DATE'] = pd.to_datetime(df['DATA_DATE'], errors='coerce').dt.date
            else:
                df['DATA_DATE'] = date.today()
    
            # 可选：计算振幅（如果需要）
            if all(col in df.columns for col in ['HIGH_PRICE', 'LOW_PRICE', 'OPEN_PRICE']):
                df['AMPLITUDE'] = ((df['HIGH_PRICE'] - df['LOW_PRICE']) / df['OPEN_PRICE']) * 100
            else:
                df['AMPLITUDE'] = None
    
            df = df.replace([np.nan, np.inf, -np.inf], None)
    
            # 固定字段
            df['ADJUST_TYPE'] = 'qfq'   # 区分这是从 spot 补的当天记录
            df['SOURCE'] = 'AKSHARE_SPOT_AS_HIST'
            df['CREATED_AT'] = pd.Timestamp.now()
    
            # 只保留 hist 表需要的列
            keep_cols = ['CODE', 'DATA_DATE', 'OPEN_PRICE', 'CLOSE_PRICE', 'HIGH_PRICE', 'LOW_PRICE',
                         'VOLUME', 'AMOUNT', 'AMPLITUDE', 'CHANGE_PCT', 'CHANGE_AMOUNT', 'TURNOVER_RATE',
                         'ADJUST_TYPE', 'SOURCE', 'CREATED_AT']
            df = df[[c for c in keep_cols if c in df.columns or c in ['CODE', 'DATA_DATE', 'ADJUST_TYPE']]]
    
            records = df.to_dict('records')
            if not records:
                self.log.info("无有效记录可插入")
                return
    
            with self.engine.connect() as conn:
                # 检查当天 + ADJUST_TYPE 已存在的记录
                existing_sql = text(f"""
                    SELECT CODE  , DATA_DATE 
                    FROM {SCHEMA_NAME}.{ETF_HIST_TABLE}
                    WHERE DATA_DATE = :dt AND ADJUST_TYPE = 'qfq'
                """)
                existing = pd.read_sql(existing_sql, conn, params={"dt": df['DATA_DATE'].iloc[0], "adj": 'SPOT_TODAY'})
                existing.columns = [col.upper() for col in existing.columns]
                existing_set = {(row['CODE'], row['DATA_DATE']) for _, row in existing.iterrows()}
    
                # 过滤出真正缺失的
                missing_records = [
                    rec for rec in records
                    if (rec['CODE'], rec['DATA_DATE']) not in existing_set
                ]
    
                if not missing_records:
                    self.log.info("当天所有记录已存在，无需插入")
                    return
    
                # 批量插入
                conn.execute(text(f"""
                    INSERT INTO {SCHEMA_NAME}.{ETF_HIST_TABLE}
                    (CODE, DATA_DATE, OPEN_PRICE, CLOSE_PRICE, HIGH_PRICE, LOW_PRICE,
                     VOLUME, AMOUNT, AMPLITUDE, CHANGE_PCT, CHANGE_AMOUNT, TURNOVER_RATE,
                     ADJUST_TYPE, SOURCE, CREATED_AT)
                    VALUES
                    (:CODE, :DATA_DATE, :OPEN_PRICE, :CLOSE_PRICE, :HIGH_PRICE, :LOW_PRICE,
                     :VOLUME, :AMOUNT, :AMPLITUDE, :CHANGE_PCT, :CHANGE_AMOUNT, :TURNOVER_RATE,
                     :ADJUST_TYPE, :SOURCE, :CREATED_AT)
                """), missing_records)
    
                conn.commit()
    
            self.log.info(f"成功插入 {len(missing_records)} 条当天 spot 作为历史记录（{today_str}）")
    
        except Exception as e:
            self.log.info(f"盘后补历史失败: {e}")
            traceback.print_exc()
            #raise e    
    