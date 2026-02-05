import pandas as pd
import numpy as np
from datetime import date
from sqlalchemy import create_engine, text, inspect
import akshare as ak
import time
import random

from logger import setup_logging, get_logger
LOG = get_logger("Main")


from wireguard_helper import activate_tunnel, deactivate_tunnel, switch_wire_guard

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
#     Option Daily Table
# ==============================
OPTION_DAILY_TABLE = "CN_OPTION_SSE_DAILY"

option_daily_sql = f"""
CREATE TABLE {SCHEMA_NAME}.{OPTION_DAILY_TABLE} (
    CONTRACT_CODE     VARCHAR2(20)    NOT NULL,
    UNDERLYING_CODE   VARCHAR2(20),
    EXPIRY_MONTH      VARCHAR2(6),
    DATA_DATE         DATE            NOT NULL,
    OPEN_PRICE        NUMBER,
    HIGH_PRICE        NUMBER,
    LOW_PRICE         NUMBER,
    CLOSE_PRICE       NUMBER,
    VOLUME            NUMBER,
    SOURCE            VARCHAR2(30),
    CREATED_AT        TIMESTAMP       DEFAULT SYSDATE,
    CONSTRAINT CN_OPTION_DAILY_PK PRIMARY KEY (CONTRACT_CODE, DATA_DATE)
)
"""

option_indexes = [
    f"CREATE INDEX IDX_OPTION_CONTRACT ON {SCHEMA_NAME}.{OPTION_DAILY_TABLE} (CONTRACT_CODE)",
    f"CREATE INDEX IDX_OPTION_DATE ON {SCHEMA_NAME}.{OPTION_DAILY_TABLE} (DATA_DATE)",
    f"CREATE INDEX IDX_OPTION_UNDERLYING ON {SCHEMA_NAME}.{OPTION_DAILY_TABLE} (UNDERLYING_CODE)"
]

# ==============================
#     Load SSE/SZSE ETF Option Daily Data
# ==============================
def load_option_sse_daily():
    today = date.today()
    LOG.info("=== Creating/Checking Option SSE Daily table ===")
    create_table_if_not_exists(OPTION_DAILY_TABLE, option_daily_sql, option_indexes)

    LOG.info(f"\n=== Loading SSE/SZSE ETF Option Daily Data for {today} ===")
    
    etf_codes = [
        "510050", "510300", "510500", "588000", "588080",
        "159919", "159922", "159915", "159901"
    ]

    total_processed = 0
    max_retries = 5
           
    for attempt in range(1, max_retries + 1):
        try:
            for etf_idx, underlying in enumerate(etf_codes, 1):
                LOG.info(f"\n[{etf_idx:2d}/{len(etf_codes)}] Processing underlying: {underlying}")
        
                # 检查该 underlying 当天是否已有数据（若有则跳过整个 underlying）
                with engine.connect() as conn:
                    count_sql = text(f"""
                        SELECT COUNT(*) FROM {SCHEMA_NAME}.{OPTION_DAILY_TABLE}
                        WHERE UNDERLYING_CODE = :underlying AND DATA_DATE = :dt
                    """)
                    count = conn.scalar(count_sql, {"underlying": underlying, "dt": today})
        
                if count > 0:
                    LOG.info(f"    Skip: Already has {count} records for today")
                    continue
        
                records = []  # 每个 underlying 单独收集
        
                try:
                    expire_list = ak.option_sse_list_sina(symbol=underlying)
                    if not expire_list:
                        LOG.info("    No expiry months available")
                        continue
        
                    LOG.info(f"    Found {len(expire_list)} expiry months: {', '.join(expire_list)}")
        
                    for expiry_idx, expiry in enumerate(expire_list, 1):
                        LOG.info(f"  [{expiry_idx:2d}/{len(expire_list)}] Expiry: {expiry}")
        
                        try:
                            df_codes = ak.option_sse_codes_sina(trade_date=expiry, underlying=underlying)
                            if df_codes.empty:
                                LOG.info("      No contracts found")
                                continue
        
                            contracts = df_codes['期权代码'].tolist()
                            LOG.info(f"      Found {len(contracts)} contracts")
        
                            for contract_idx, contract in enumerate(contracts, 1):
                                LOG.info(f"        [{contract_idx:3d}/{len(contracts)}] {contract} ... " )
        
                                time.sleep(random.uniform(1.5, 4.0))
        
                                try:
                                    df_hist = ak.option_sse_daily_sina(symbol=contract)
                                    if df_hist.empty:
                                        LOG.info("empty")
                                        continue
        
                                    df_latest = df_hist.iloc[[0]].copy()
                                    latest_date = pd.to_datetime(df_latest['日期'].iloc[0]).date()
        
                                    df_latest['CONTRACT_CODE'] = contract
                                    df_latest['UNDERLYING_CODE'] = underlying
                                    df_latest['EXPIRY_MONTH'] = expiry
                                    df_latest['SOURCE'] = 'AKSHARE_OPTION_SSE_SINA'
                                    df_latest['CREATED_AT'] = pd.Timestamp.now()
        
                                    records.extend(df_latest.to_dict('records'))
                                    total_processed += 1
                                    LOG.info("ok")
                                    
                                except Exception as inner_e:
                                    LOG.info(f"error: {str(inner_e)}")
        
                        except Exception as expiry_e:
                            LOG.info(f"    Expiry {expiry} failed: {str(expiry_e)}")
        
                except Exception as etf_e:
                    LOG.info(f"  Underlying {underlying} failed: {str(etf_e)}")
        
                # 每个 underlying 处理完后 → 只插入缺失的数据
                if records:
                    df_etf = pd.DataFrame(records)
        
                    # 调试：显示实际列
                    LOG.info(f"    Columns collected for {underlying}: {list(df_etf.columns)}")
        
                    # 重命名 & 类型转换（只处理实际存在的字段）
                    rename_map = {
                        '日期': 'DATA_DATE',
                        '开盘': 'OPEN_PRICE',
                        '最高': 'HIGH_PRICE',
                        '最低': 'LOW_PRICE',
                        '收盘': 'CLOSE_PRICE',
                        '成交量': 'VOLUME',
                    }
                    df_etf.rename(columns=rename_map, inplace=True)
        
                    numeric_cols = ['OPEN_PRICE', 'HIGH_PRICE', 'LOW_PRICE', 'CLOSE_PRICE', 'VOLUME']
                    for col in numeric_cols:
                        if col in df_etf.columns:
                            df_etf[col] = pd.to_numeric(df_etf[col], errors='coerce')
        
                    df_etf['DATA_DATE'] = pd.to_datetime(df_etf['DATA_DATE'], errors='coerce').dt.date
                    df_etf = df_etf.replace([np.nan, np.inf, -np.inf], None)
        
                    # 确保关键字段存在
                    for required in ['CONTRACT_CODE', 'UNDERLYING_CODE', 'EXPIRY_MONTH', 'SOURCE', 'CREATED_AT']:
                        if required not in df_etf.columns:
                            df_etf[required] = None
        
                    etf_records = df_etf.to_dict('records')
                    ###
                    if etf_records:
                        with engine.connect() as conn:
                            # 预过滤缺失记录（可选，但保留以减少 MERGE 负载）
                            existing_sql = text(f"""
                                SELECT CONTRACT_CODE 
                                FROM {SCHEMA_NAME}.{OPTION_DAILY_TABLE}
                                WHERE UNDERLYING_CODE = :underlying AND DATA_DATE = :dt
                            """)
                            existing_df = pd.read_sql(
                                existing_sql,
                                conn,
                                params={"underlying": underlying, "dt": today}
                            )
        
                            if not existing_df.empty:
                                existing_contracts = set(existing_df['CONTRACT_CODE'])
                            else:
                                existing_contracts = set()
        
                            missing_records = [
                                rec for rec in etf_records
                                if rec['CONTRACT_CODE'] not in existing_contracts
                            ]
        
                            if not missing_records:
                                LOG.info(f"    All {len(etf_records)} records already exist for {underlying}, skip")
                                continue
        
                            # 固定字段列表（根据实际数据）
                            core_cols = [
                                'CONTRACT_CODE', 'DATA_DATE', 'UNDERLYING_CODE', 'EXPIRY_MONTH',
                                'OPEN_PRICE', 'HIGH_PRICE', 'LOW_PRICE', 'CLOSE_PRICE', 'VOLUME', 'SOURCE'
                            ]
        
                            # 过滤掉实际不存在的列（防止接口返回少字段）
                            available_cols = [c for c in core_cols if c in missing_records[0]]
        
                            # UPDATE SET：只更新数值字段 + SOURCE
                            updatable_cols = [c for c in available_cols 
                                              if c in ['OPEN_PRICE', 'HIGH_PRICE', 'LOW_PRICE', 'CLOSE_PRICE', 'VOLUME', 'SOURCE']]
                            update_set = ", ".join([f"t.{c} = s.{c}" for c in updatable_cols])
        
                            # INSERT 列：核心字段 + CREATED_AT（不重复）
                            insert_cols = ", ".join(available_cols + ['CREATED_AT'])
                            insert_vals = ", ".join([f":{c}" for c in available_cols] + ["SYSDATE"])
        
                            # 生成 MERGE
                            merge_sql = f"""
                            MERGE INTO {SCHEMA_NAME}.{OPTION_DAILY_TABLE} t
                            USING (SELECT 
                                :CONTRACT_CODE     AS CONTRACT_CODE,
                                :DATA_DATE         AS DATA_DATE,
                                :UNDERLYING_CODE   AS UNDERLYING_CODE,
                                :EXPIRY_MONTH      AS EXPIRY_MONTH,
                                :OPEN_PRICE        AS OPEN_PRICE,
                                :HIGH_PRICE        AS HIGH_PRICE,
                                :LOW_PRICE         AS LOW_PRICE,
                                :CLOSE_PRICE       AS CLOSE_PRICE,
                                :VOLUME            AS VOLUME,
                                :SOURCE            AS SOURCE
                                FROM dual) s
                            ON (t.CONTRACT_CODE = s.CONTRACT_CODE AND t.DATA_DATE = s.DATA_DATE)
                            WHEN MATCHED THEN
                                UPDATE SET
                                    {update_set},
                                    t.CREATED_AT = SYSDATE
                            WHEN NOT MATCHED THEN
                                INSERT ({insert_cols})
                                VALUES ({insert_vals})
                            """
        
                            # 调试输出（运行一次后可注释掉）
                            LOG.info(f"Generated MERGE for {underlying}:\n{merge_sql}")
        
                            conn.execute(text(merge_sql), missing_records)
                            conn.commit()
        
                            LOG.info(f"    Merged {len(missing_records)} records for {underlying} (total collected: {len(etf_records)})")            
        
                    ###
        #try
        except Exception as e:
            if attempt == max_retries:
                raise RuntimeError(
                    f"获取etf 数据失败（已重试 {max_retries} 次）：{str(e)}"
                ) from e
            switch_wire_guard("cn")
            time.sleep(60)
     
    #for 
    LOG.info(f"\n=== 完成 ===\n总处理合约数量: {total_processed}")

if __name__ == "__main__":
    load_option_sse_daily()