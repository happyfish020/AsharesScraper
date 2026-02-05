import traceback
import pandas as pd
import numpy as np
from datetime import date, datetime
from sqlalchemy import create_engine, text, inspect
import akshare as ak
import time
import  baostock as bs  

from wireguard_helper import activate_tunnel, deactivate_tunnel, switch_wire_guard, toggle_vpn

from logger import setup_logging, get_logger
LOG = get_logger("Main")
# ==============================
#        Database Configuration
# ==============================
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

SCHEMA_NAME = "SECOPR"

# ==============================
#     Table Creation Helper
# ==============================
def create_table_if_not_exists(table_name: str, create_sql: str, indexes: list = None):
    inspector = inspect(engine)
    
    if not inspector.has_table(table_name, schema=SCHEMA_NAME):
        LOG.info(f"Creating table {SCHEMA_NAME}.{table_name}...")
        
        with engine.connect() as conn:
            conn.execute(text(create_sql))
            
            if indexes:
                for idx_sql in indexes:
                    conn.execute(text(idx_sql))
            
            conn.commit()
        LOG.info(f"Table {table_name} created.")
    else:
        LOG.info(f"Table {SCHEMA_NAME}.{table_name} already exists.")

# ==============================
#     实时 Spot 表
# ==============================
ETF_SPOT_TABLE = "CN_FUND_ETF_SPOT_EM"

etf_spot_sql = f"""
CREATE TABLE {SCHEMA_NAME}.{ETF_SPOT_TABLE} (
    CODE                  VARCHAR2(20)    NOT NULL,
    NAME                  VARCHAR2(100),
    LATEST_PRICE          NUMBER,
    IOPV                  NUMBER,
    DISCOUNT_RATE         NUMBER,
    CHANGE_AMOUNT         NUMBER,
    CHANGE_PCT            NUMBER,
    VOLUME                NUMBER,
    AMOUNT                NUMBER,
    OPEN_PRICE            NUMBER,
    HIGH_PRICE            NUMBER,
    LOW_PRICE             NUMBER,
    PRE_CLOSE             NUMBER,
    TURNOVER_RATE         NUMBER,
    VOLUME_RATIO          NUMBER,
    COMMISSION_RATIO      NUMBER,
    OUTER_DISC            NUMBER,
    INNER_DISC            NUMBER,
    MAIN_INFLOW_NET       NUMBER,
    MAIN_INFLOW_PCT       NUMBER,
    SUPER_INFLOW_NET      NUMBER,
    SUPER_INFLOW_PCT      NUMBER,
    LARGE_INFLOW_NET      NUMBER,
    LARGE_INFLOW_PCT      NUMBER,
    MEDIUM_INFLOW_NET     NUMBER,
    MEDIUM_INFLOW_PCT     NUMBER,
    SMALL_INFLOW_NET      NUMBER,
    SMALL_INFLOW_PCT      NUMBER,
    CURRENT_HAND          NUMBER,
    BID1                  NUMBER,
    ASK1                  NUMBER,
    LATEST_SHARES         NUMBER,
    CIRC_MARKET_CAP       NUMBER,
    TOTAL_MARKET_CAP      NUMBER,
    DATA_DATE             DATE            NOT NULL,
    UPDATE_TIME           TIMESTAMP,
    SOURCE                VARCHAR2(30),
    CREATED_AT            TIMESTAMP       DEFAULT SYSDATE,
    CONSTRAINT CN_FES_PK PRIMARY KEY (CODE, DATA_DATE)
)
"""

etf_spot_indexes = [
    f"CREATE INDEX IDX_ETF_CODE ON {SCHEMA_NAME}.{ETF_SPOT_TABLE} (CODE)",
    f"CREATE INDEX IDX_ETF_DATA_DATE ON {SCHEMA_NAME}.{ETF_SPOT_TABLE} (DATA_DATE)"
]

# ==============================
#     历史行情独立表
# ==============================
ETF_HIST_TABLE = "CN_FUND_ETF_HIST_EM"

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

# ==============================
#     实时行情加载（原有功能）
# ==============================
def load_fund_etf_spot_em_intraday():
    today = date.today()
    LOG.info("=== Checking/Creating spot table ===")
    create_table_if_not_exists(ETF_SPOT_TABLE, etf_spot_sql, etf_spot_indexes)

    LOG.info("\n=== Loading real-time ETF spot data ===")
    try:
        df = ak.fund_etf_spot_em()
        if df.empty:
            LOG.info("No data from ak.fund_etf_spot_em()")
            return

        LOG.info(f"Loaded {len(df)} records.")

        rename_map = {
            '代码': 'CODE',
            '名称': 'NAME',
            '最新价': 'LATEST_PRICE',
            'IOPV实时估值': 'IOPV',
            '基金折价率': 'DISCOUNT_RATE',
            '涨跌额': 'CHANGE_AMOUNT',
            '涨跌幅': 'CHANGE_PCT',
            '成交量': 'VOLUME',
            '成交额': 'AMOUNT',
            '开盘价': 'OPEN_PRICE',
            '最高价': 'HIGH_PRICE',
            '最低价': 'LOW_PRICE',
            '昨收': 'PRE_CLOSE',
            '换手率': 'TURNOVER_RATE',
            '量比': 'VOLUME_RATIO',
            '委比': 'COMMISSION_RATIO',
            '外盘': 'OUTER_DISC',
            '内盘': 'INNER_DISC',
            '主力净流入-净额': 'MAIN_INFLOW_NET',
            '主力净流入-净占比': 'MAIN_INFLOW_PCT',
            '超大单净流入-净额': 'SUPER_INFLOW_NET',
            '超大单净流入-净占比': 'SUPER_INFLOW_PCT',
            '大单净流入-净额': 'LARGE_INFLOW_NET',
            '大单净流入-净占比': 'LARGE_INFLOW_PCT',
            '中单净流入-净额': 'MEDIUM_INFLOW_NET',
            '中单净流入-净占比': 'MEDIUM_INFLOW_PCT',
            '小单净流入-净额': 'SMALL_INFLOW_NET',
            '小单净流入-净占比': 'SMALL_INFLOW_PCT',
            '现手': 'CURRENT_HAND',
            '买一': 'BID1',
            '卖一': 'ASK1',
            '最新份额': 'LATEST_SHARES',
            '流通市值': 'CIRC_MARKET_CAP',
            '总市值': 'TOTAL_MARKET_CAP',
            '数据日期': 'DATA_DATE',
            '更新时间': 'UPDATE_TIME'
        }
        df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)

        pct_cols = ['DISCOUNT_RATE', 'CHANGE_PCT', 'TURNOVER_RATE', 'VOLUME_RATIO',
                    'COMMISSION_RATIO', 'MAIN_INFLOW_PCT', 'SUPER_INFLOW_PCT',
                    'LARGE_INFLOW_PCT', 'MEDIUM_INFLOW_PCT', 'SMALL_INFLOW_PCT']
        for col in pct_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace('%', '', regex=False).str.strip()
                df[col] = pd.to_numeric(df[col], errors='coerce')

        numeric_cols = ['LATEST_PRICE', 'IOPV', 'CHANGE_AMOUNT', 'VOLUME', 'AMOUNT',
                        'OPEN_PRICE', 'HIGH_PRICE', 'LOW_PRICE', 'PRE_CLOSE',
                        'OUTER_DISC', 'INNER_DISC', 'MAIN_INFLOW_NET', 'SUPER_INFLOW_NET',
                        'LARGE_INFLOW_NET', 'MEDIUM_INFLOW_NET', 'SMALL_INFLOW_NET',
                        'CURRENT_HAND', 'BID1', 'ASK1', 'LATEST_SHARES',
                        'CIRC_MARKET_CAP', 'TOTAL_MARKET_CAP']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        if 'DATA_DATE' in df.columns:
            df['DATA_DATE'] = pd.to_datetime(df['DATA_DATE'], errors='coerce').dt.date
        if 'UPDATE_TIME' in df.columns:
            df['UPDATE_TIME'] = pd.to_datetime(df['UPDATE_TIME'], errors='coerce')

        df = df.replace([np.nan, np.inf, -np.inf], None)

        df['SOURCE'] = 'AKSHARE_EASTMONEY'
        df['CREATED_AT'] = pd.Timestamp.now()

        records = df.to_dict('records')

        with engine.connect() as conn:
            conn.execute(text(f"DELETE FROM {SCHEMA_NAME}.{ETF_SPOT_TABLE} WHERE DATA_DATE = :dt"), {"dt": today})

            if records:
                conn.execute(text(f"""
                    INSERT INTO {SCHEMA_NAME}.{ETF_SPOT_TABLE} (
                        CODE, NAME, LATEST_PRICE, IOPV, DISCOUNT_RATE, CHANGE_AMOUNT, CHANGE_PCT,
                        VOLUME, AMOUNT, OPEN_PRICE, HIGH_PRICE, LOW_PRICE, PRE_CLOSE,
                        TURNOVER_RATE, VOLUME_RATIO, COMMISSION_RATIO, OUTER_DISC, INNER_DISC,
                        MAIN_INFLOW_NET, MAIN_INFLOW_PCT, SUPER_INFLOW_NET, SUPER_INFLOW_PCT,
                        LARGE_INFLOW_NET, LARGE_INFLOW_PCT, MEDIUM_INFLOW_NET, MEDIUM_INFLOW_PCT,
                        SMALL_INFLOW_NET, SMALL_INFLOW_PCT, CURRENT_HAND, BID1, ASK1,
                        LATEST_SHARES, CIRC_MARKET_CAP, TOTAL_MARKET_CAP,
                        DATA_DATE, UPDATE_TIME, SOURCE, CREATED_AT
                    ) VALUES (
                        :CODE, :NAME, :LATEST_PRICE, :IOPV, :DISCOUNT_RATE, :CHANGE_AMOUNT, :CHANGE_PCT,
                        :VOLUME, :AMOUNT, :OPEN_PRICE, :HIGH_PRICE, :LOW_PRICE, :PRE_CLOSE,
                        :TURNOVER_RATE, :VOLUME_RATIO, :COMMISSION_RATIO, :OUTER_DISC, :INNER_DISC,
                        :MAIN_INFLOW_NET, :MAIN_INFLOW_PCT, :SUPER_INFLOW_NET, :SUPER_INFLOW_PCT,
                        :LARGE_INFLOW_NET, :LARGE_INFLOW_PCT, :MEDIUM_INFLOW_NET, :MEDIUM_INFLOW_PCT,
                        :SMALL_INFLOW_NET, :SMALL_INFLOW_PCT, :CURRENT_HAND, :BID1, :ASK1,
                        :LATEST_SHARES, :CIRC_MARKET_CAP, :TOTAL_MARKET_CAP,
                        :DATA_DATE, :UPDATE_TIME, :SOURCE, :CREATED_AT
                    )
                """), records)
            conn.commit()

        LOG.info(f"Inserted/updated {len(records)} spot records for {today}.")

    except Exception as e:
        LOG.info(f"Spot load failed: {e}")

# ==============================
#     历史行情加载（独立表）
# ==============================
def load_fund_etf_hist_em(start_date: str, end_date: str, period: str = "daily", adjust: str = ""):
    """
    加载 ETF 历史行情到独立表 CN_FUND_ETF_HIST_EM
    """
    LOG.info(f"=== Historical load started: {start_date} → {end_date} | period={period} | adjust='{adjust}' ===")
    create_table_if_not_exists(ETF_HIST_TABLE, etf_hist_sql, etf_hist_indexes)

    def normalize_date(d: str) -> str:
        d = d.replace("-", "").strip()
        if len(d) == 8 and d.isdigit():
            return d
        raise ValueError(f"Date format error: {d} (expect YYYYMMDD or YYYY-MM-DD)")

    start_date = normalize_date(start_date)
    end_date = normalize_date(end_date)

    # 从 spot 表取代码列表
    with engine.connect() as conn:
        df_codes = pd.read_sql(f"SELECT DISTINCT CODE FROM {SCHEMA_NAME}.{ETF_SPOT_TABLE}", conn)
    
    if df_codes.empty:
        LOG.info("No ETF codes in spot table. Run load_fund_etf_spot_em() first.")
        return

    codes = df_codes['code'].sort_values().tolist()
    LOG.info(f"Processing {len(codes)} symbols.")

    total = 0
    failed = []

    for i, symbol in enumerate(codes, 1):
        LOG.info(f"[{i:4d}/{len(codes)}] {symbol} ... ")
        try:
            df = ak.fund_etf_hist_em(
                symbol=symbol,
                period=period,
                start_date=start_date,
                end_date=end_date,
                adjust=adjust
            )

            if df.empty:
                LOG.info("empty")
                continue

            rename_map = {
                '日期': 'DATA_DATE',
                '开盘': 'OPEN_PRICE',
                '收盘': 'CLOSE_PRICE',
                '最高': 'HIGH_PRICE',
                '最低': 'LOW_PRICE',
                '成交量': 'VOLUME',
                '成交额': 'AMOUNT',
                '振幅': 'AMPLITUDE',
                '涨跌幅': 'CHANGE_PCT',
                '涨跌额': 'CHANGE_AMOUNT',
                '换手率': 'TURNOVER_RATE'
            }
            df.rename(columns=rename_map, inplace=True)

            for col in ['OPEN_PRICE', 'CLOSE_PRICE', 'HIGH_PRICE', 'LOW_PRICE', 'VOLUME',
                        'AMOUNT', 'AMPLITUDE', 'CHANGE_PCT', 'CHANGE_AMOUNT', 'TURNOVER_RATE']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            df['DATA_DATE'] = pd.to_datetime(df['DATA_DATE'], errors='coerce').dt.date
            df = df.replace([np.nan, np.inf, -np.inf], None)

            df['CODE'] = symbol
            df['ADJUST_TYPE'] = adjust if adjust else 'NONE'
            df['SOURCE'] = 'AKSHARE'
            df['CREATED_AT'] = pd.Timestamp.now()

            records = df.to_dict('records')

            with engine.connect() as conn:
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
                conn.execute(text(merge_sql), records)
                conn.commit()

            count = len(records)
            total += count
            LOG.info(f"ok ({count})")

            time.sleep(0.5)  # 防限流

        except Exception as e:
            LOG.info(f"error: {str(e)}")
            failed.append(symbol)

    LOG.info(f"\nHistorical load finished. Total records: {total}")
    if failed:
        LOG.info(f"Failed symbols: {len(failed)} | {', '.join(failed[:8])}{'...' if len(failed)>8 else ''}")



def load_fund_etf_hist_baostock(start_date: str, end_date: str, frequency: str = "d", adjustflag: str = "3"):
    """
    使用 baostock 下载 ETF 历史行情 并存入 CN_FUND_ETF_HIST_EM 风格的表
    
    参数说明：
    - frequency: "d"=日线, "w"=周线, "m"=月线, "5"/"15"/"30"/"60"=分钟线
    - adjustflag: "1"=不复权, "2"=前复权, "3"=后复权（默认后复权，与多数人习惯一致）
    
    注意：ETF代码需带前缀，例如 'sh.510300', 'sz.159941'
          spot表里的 CODE 如果没有前缀，需要提前补上 sh./sz.
    """
    LOG.info(f"=== Historical load (baostock) started: {start_date} → {end_date} | frequency={frequency} | adjustflag={adjustflag} ===")
    create_table_if_not_exists(ETF_HIST_TABLE, etf_hist_sql, etf_hist_indexes)

    def normalize_date_ymd(d: str) -> str:
        d = d.replace("-", "").strip()
        if len(d) == 8 and d.isdigit():
            return f"{d[:4]}-{d[4:6]}-{d[6:8]}"
        raise ValueError(f"Date format error: {d} (expect YYYYMMDD or YYYY-MM-DD)")

    start_dt = normalize_date_ymd(start_date)
    end_dt   = normalize_date_ymd(end_date)

    # 登录 baostock（每次运行登录一次即可）
    lg = bs.login()
    if lg.error_code != '0':
        LOG.error(f"baostock login failed: {lg.error_msg}")
        return
    LOG.info("baostock login success")

    # 从 spot 表取代码列表（假设 CODE 已带 sh./sz. 前缀）
    with engine.connect() as conn:
        df_codes = pd.read_sql(f"SELECT DISTINCT CODE FROM {SCHEMA_NAME}.{ETF_HIST_TABLE}", conn)
    
    if df_codes.empty:
        LOG.info("No ETF codes in spot table.")
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
    LOG.info(f"Processing {len(codes)} symbols with baostock (after prefix).")

    LOG.info(f"Processing {len(codes)} symbols with baostock.")

    total = 0
    failed = []

    fields = "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,pctChg"

    processed = 0
    
    
    for i, symbol in enumerate(codes, 1):
        processed += 1
        if processed % 20 == 0:
           time.sleep(1)
        LOG.info(f"[{i:4d}/{len(codes)}] {symbol} ...  ")

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
                LOG.info(f"query error: {rs.error_msg}")
                failed.append(symbol)
                continue

            df = rs.get_data()

            if df.empty:
                LOG.info("empty")
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
            with engine.connect() as conn:
                conn.execute(text(merge_sql), records)
                conn.commit()

            count = len(records)
            total += count
            LOG.info(f"ok ({count})")

            time.sleep(0.4)  # baostock 免费版有频率限制，建议稍作间隔

        except Exception as e:
            LOG.info(f"error: {str(e)}")
            failed.append(symbol)

    bs.logout()
    LOG.info(f"\nHistorical load (baostock) finished. Total records: {total}")
    if failed:
        LOG.info(f"Failed symbols: {len(failed)} | {', '.join(failed[:8])}{'...' if len(failed)>8 else ''}")
 
 
 # ==============================
#     盘后：用 spot 数据补当天 hist 记录（批量、无需逐 code 循环）
# ==============================
def load_spot_as_hist_today():
    """
    盘后运行：用 ak.fund_etf_spot_em() 的当天数据作为历史日K补入 CN_FUND_ETF_HIST_EM
    只插入当天缺失的记录，ADJUST_TYPE='SPOT_TODAY'
    """
    today_str = date.today().strftime("%Y-%m-%d")
    LOG.info(f"=== 盘后补当天历史记录 ({today_str}) ===")

    create_table_if_not_exists(ETF_HIST_TABLE, etf_hist_sql, etf_hist_indexes)
        
    #switch_wire_guard("cn")
    toggle_vpn("cn", action="start")
    try:
        df = ak.fund_etf_spot_em()
        if df.empty:
            LOG.info("ak.fund_etf_spot_em() 返回空数据")
            return

        LOG.info(f"获取到 {len(df)} 条当天 spot 数据")

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
            LOG.info("无有效记录可插入")
            return

        with engine.connect() as conn:
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
                LOG.info("当天所有记录已存在，无需插入")
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

        LOG.info(f"成功插入 {len(missing_records)} 条当天 spot 作为历史记录（{today_str}）")

    except Exception as e:
        LOG.info(f"盘后补历史失败: {e}")
        traceback.print_exc()
        #raise e    
# ==============================
#     测试入口（可选）
# ==============================
if __name__ == "__main__":
    # load_fund_etf_spot_em()
    # load_fund_etf_hist_em("20240101", "20260119", adjust="")
    pass