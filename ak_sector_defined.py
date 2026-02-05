import pandas as pd
from datetime import date
from sqlalchemy import create_engine, text, inspect
import akshare as ak
import time
import random
from logger import setup_logging, get_logger
LOG = get_logger("Main")


# ==============================
#        数据库连接配置
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
#     表创建函数
# ==============================
def create_table_if_not_exists(table_name: str, create_sql: str, indexes: list = None):
    inspector = inspect(engine)
    
    if not inspector.has_table(table_name, schema=SCHEMA_NAME):
        LOG.info(f"表 {SCHEMA_NAME}.{table_name} 不存在，正在创建...")
        
        with engine.connect() as conn:
            conn.execute(text(create_sql))
            
            # 创建索引（如果有）
            if indexes:
                for idx_sql in indexes:
                    conn.execute(text(idx_sql))
            
            conn.commit()
        LOG.info(f"表 {table_name} 创建完成")
    else:
        LOG.info(f"表 {SCHEMA_NAME}.{table_name} 已存在，跳过创建")

# ==============================
#     1. 行业板块主表
# ==============================
INDUSTRY_MASTER_TABLE = "CN_BOARD_INDUSTRY_MASTER"

industry_master_sql = f"""
CREATE TABLE {SCHEMA_NAME}.{INDUSTRY_MASTER_TABLE} (
    board_id     VARCHAR2(40)   NOT NULL,
    board_name   VARCHAR2(80),
    PROVIDER     VARCHAR2(20)   DEFAULT 'EASTMONEY',
    ASOF_DATE    DATE           NOT NULL,
    SOURCE       VARCHAR2(30),
    CREATED_AT   DATE           DEFAULT SYSDATE,
    RAW_JSON     CLOB,
    CONSTRAINT CN_BIM_PK PRIMARY KEY (board_id, ASOF_DATE)
)
"""

industry_indexes = [
    f"CREATE INDEX IDX_INDUSTRY_NAME ON {SCHEMA_NAME}.{INDUSTRY_MASTER_TABLE} (board_name)",
    f"CREATE INDEX IDX_INDUSTRY_ASOF ON {SCHEMA_NAME}.{INDUSTRY_MASTER_TABLE} (ASOF_DATE)"
]

# ==============================
#     2. 概念板块主表
# ==============================
CONCEPT_MASTER_TABLE = "CN_BOARD_CONCEPT_MASTER"

concept_master_sql = f"""
CREATE TABLE {SCHEMA_NAME}.{CONCEPT_MASTER_TABLE} (
    CONCEPT_ID   VARCHAR2(40)   NOT NULL,
    CONCEPT_NAME VARCHAR2(80),
    PROVIDER     VARCHAR2(20)   DEFAULT 'EASTMONEY',
    ASOF_DATE    DATE           NOT NULL,
    SOURCE       VARCHAR2(30),
    CREATED_AT   DATE           DEFAULT SYSDATE,
    RAW_JSON     CLOB,
    CONSTRAINT CN_BCM_PK PRIMARY KEY (CONCEPT_ID, ASOF_DATE)
)
"""

concept_indexes = [
    f"CREATE INDEX IDX_CONCEPT_NAME ON {SCHEMA_NAME}.{CONCEPT_MASTER_TABLE} (CONCEPT_NAME)",
    f"CREATE INDEX IDX_CONCEPT_ASOF ON {SCHEMA_NAME}.{CONCEPT_MASTER_TABLE} (ASOF_DATE)"
]

# ==============================
#     3. 行业成分表
# ==============================
INDUSTRY_CONS_TABLE = "CN_BOARD_INDUSTRY_CONS"

industry_cons_sql = f"""
CREATE TABLE {SCHEMA_NAME}.{INDUSTRY_CONS_TABLE} (
    board_id     VARCHAR2(40)   NOT NULL,
    SYMBOL       VARCHAR2(10)   NOT NULL,
    EXCHANGE     VARCHAR2(10),
    ASOF_DATE    DATE           NOT NULL,
    SOURCE       VARCHAR2(30),
    CREATED_AT   DATE           DEFAULT SYSDATE,
    CONSTRAINT CN_BIC_PK PRIMARY KEY (board_id, SYMBOL, ASOF_DATE)
)
"""

industry_cons_indexes = [
    f"CREATE INDEX IDX_INDUSTRY_CONS_SYMBOL ON {SCHEMA_NAME}.{INDUSTRY_CONS_TABLE} (SYMBOL)",
    f"CREATE INDEX IDX_INDUSTRY_CONS_BOARD ON {SCHEMA_NAME}.{INDUSTRY_CONS_TABLE} (board_id)"
]

# ==============================
#     4. 概念成分表
# ==============================
CONCEPT_CONS_TABLE = "CN_BOARD_CONCEPT_CONS"

concept_cons_sql = f"""
CREATE TABLE {SCHEMA_NAME}.{CONCEPT_CONS_TABLE} (
    CONCEPT_ID   VARCHAR2(40)   NOT NULL,
    SYMBOL       VARCHAR2(10)   NOT NULL,
    EXCHANGE     VARCHAR2(10),
    ASOF_DATE    DATE           NOT NULL,
    SOURCE       VARCHAR2(30),
    CREATED_AT   DATE           DEFAULT SYSDATE,
    CONSTRAINT CN_BCC_PK PRIMARY KEY (CONCEPT_ID, SYMBOL, ASOF_DATE)
)
"""

concept_cons_indexes = [
    f"CREATE INDEX IDX_CONCEPT_CONS_SYMBOL ON {SCHEMA_NAME}.{CONCEPT_CONS_TABLE} (SYMBOL)",
    f"CREATE INDEX IDX_CONCEPT_CONS_CONCEPT ON {SCHEMA_NAME}.{CONCEPT_CONS_TABLE} (CONCEPT_ID)"
]



    # 1. 创建所有表（如果不存在）

# 辅助函数：获取交易所（复用之前的逻辑）
def get_exchange(code: str) -> str:
    code = str(code).strip()
    if code.startswith(('6', '9')):
        return 'SH'
    elif code.startswith(('0', '3')):
        return 'SZ'
    elif code.startswith(('4', '8', '43', '83', '87', '88')) or len(code) > 6:
        return 'BJ'
    return 'UNKNOWN'

def get_industry_concept_records():
    # ==============================
    # 2. 采集 & 插入 行业板块主表
    # ==============================
    LOG.info("=== 检查并创建表结构 ===")
    create_table_if_not_exists(INDUSTRY_MASTER_TABLE, industry_master_sql, industry_indexes)
    create_table_if_not_exists(CONCEPT_MASTER_TABLE, concept_master_sql, concept_indexes)
    create_table_if_not_exists(INDUSTRY_CONS_TABLE, industry_cons_sql, industry_cons_indexes)
    create_table_if_not_exists(CONCEPT_CONS_TABLE, concept_cons_sql, concept_cons_indexes)

    LOG.info("\n=== 采集行业板块主表 ===")
    try:
        df_industry = ak.stock_board_industry_name_em()
        LOG.info(f"获取行业板块：{len(df_industry)} 条")
    except Exception as e:
        LOG.info("获取行业板块失败:", e)
        df_industry = pd.DataFrame()

    if not df_industry.empty:
        df_industry = df_industry[['板块代码', '板块名称']].drop_duplicates()
        df_industry.columns = ['board_id', 'board_name']
        
        df_industry['PROVIDER'] = 'EASTMONEY'
        df_industry['ASOF_DATE'] = today
        df_industry['SOURCE'] = 'AKSHARE'
        df_industry['CREATED_AT'] = pd.Timestamp.now()
        
        records_ind = df_industry.to_dict('records')
        
        with engine.connect() as conn:
            conn.execute(text(f"DELETE FROM {SCHEMA_NAME}.{INDUSTRY_MASTER_TABLE} WHERE ASOF_DATE = :dt"), {"dt": today})
            if records_ind:
                conn.execute(
                    text(f"""
                        INSERT INTO {SCHEMA_NAME}.{INDUSTRY_MASTER_TABLE}
                        (board_id, board_name, PROVIDER, ASOF_DATE, SOURCE, CREATED_AT)
                        VALUES (:board_id, :board_name, :PROVIDER, :ASOF_DATE, :SOURCE, :CREATED_AT)
                    """),
                    records_ind
                )
            conn.commit()
        LOG.info(f"行业板块主表插入完成：{len(records_ind)} 条")

    # ==============================
    # 3. 采集 & 插入 概念板块主表
    # ==============================
    LOG.info("\n=== 采集概念板块主表 ===")
    try:
        df_concept = ak.stock_board_concept_name_em()
        LOG.info(f"获取概念板块：{len(df_concept)} 条")
    except Exception as e:
        LOG.info("获取概念板块失败:", e)
        df_concept = pd.DataFrame()

    if not df_concept.empty:
        df_concept = df_concept[['板块代码', '板块名称']].drop_duplicates()
        df_concept.columns = ['CONCEPT_ID', 'CONCEPT_NAME']
        
        df_concept['PROVIDER'] = 'EASTMONEY'
        df_concept['ASOF_DATE'] = today
        df_concept['SOURCE'] = 'AKSHARE'
        df_concept['CREATED_AT'] = pd.Timestamp.now()
        
        records_concept = df_concept.to_dict('records')
        
        with engine.connect() as conn:
            conn.execute(text(f"DELETE FROM {SCHEMA_NAME}.{CONCEPT_MASTER_TABLE} WHERE ASOF_DATE = :dt"), {"dt": today})
            if records_concept:
                conn.execute(
                    text(f"""
                        INSERT INTO {SCHEMA_NAME}.{CONCEPT_MASTER_TABLE}
                        (CONCEPT_ID, CONCEPT_NAME, PROVIDER, ASOF_DATE, SOURCE, CREATED_AT)
                        VALUES (:CONCEPT_ID, :CONCEPT_NAME, :PROVIDER, :ASOF_DATE, :SOURCE, :CREATED_AT)
                    """),
                    records_concept
                )
            conn.commit()
        LOG.info(f"概念板块主表插入完成：{len(records_concept)} 条")

BATCH_SIZE = 30          # 每批处理多少个板块
SLEEP_BASE = 6           # 基础延时秒数
SLEEP_RANDOM = 8         # 随机延时范围（0~8秒）


def get_industry_map():
        # ==============================
    # 4. 采集 & 插入 行业板块成分表（逐板块调用）
    # ==============================
    LOG.info("\n=== 开始采集行业板块成分股（逐板块调用，注意限频） ===")
    

    try:
        with engine.connect() as conn:
            # 读取当天最新的行业主表（假设已采集）
            df_industry_boards = pd.read_sql(
                text(f"""
                    SELECT board_id, board_name 
                    FROM {SCHEMA_NAME}.{INDUSTRY_MASTER_TABLE}
                    WHERE ASOF_DATE = :dt
                """),
                conn,
                params={"dt": today}
            )
            
            LOG.info(f"找到 {len(df_industry_boards)} 个行业板块需要采集成分股")
            
            if df_industry_boards.empty:
                LOG.info("当天行业主表无数据，跳过成分采集")
            else:
                # 先清空当天成分数据（幂等）
                conn.execute(
                    text(f"DELETE FROM {SCHEMA_NAME}.{INDUSTRY_CONS_TABLE} WHERE ASOF_DATE = :dt"),
                    {"dt": today}
                )
                conn.commit()

        inserted_total = 0
        for i in range(0, len(df_industry_boards), BATCH_SIZE):
            batch_boards = df_industry_boards.iloc[i:i + BATCH_SIZE]
            
            for _, row in batch_boards.iterrows():
                board_id = row['board_id']
                board_name = row['board_name']
                
                try:
                    
                    time.sleep(SLEEP_BASE + random.random() * SLEEP_RANDOM)  # 防限频
                    
                    LOG.info(f"  正在采集行业：{board_name} ({board_id}) ...")
                    df_cons = ak.stock_board_industry_cons_em(symbol=board_id)
                    
                    if df_cons.empty:
                        LOG.info(f"    → 无成分股或接口返回空")
                        continue
                    
                    # 数据清洗
                    df_cons = df_cons[['代码', '名称']].drop_duplicates()
                    df_cons = df_cons.rename(columns={'代码': 'SYMBOL', '名称': 'STOCK_NAME'})
                    
                    df_cons['board_id'] = board_id
                    df_cons['EXCHANGE'] = df_cons['SYMBOL'].apply(get_exchange)
                    df_cons['ASOF_DATE'] = today
                    df_cons['SOURCE'] = 'AKSHARE_EASTMONEY'
                    df_cons['CREATED_AT'] = pd.Timestamp.now()
                    
                    records_cons = df_cons.to_dict('records')
                    
                    if records_cons:
                        with engine.connect() as conn:
                            conn.execute(
                                text(f"""
                                    INSERT INTO {SCHEMA_NAME}.{INDUSTRY_CONS_TABLE}
                                    (board_id, SYMBOL, EXCHANGE, ASOF_DATE, SOURCE, CREATED_AT)
                                    VALUES 
                                    (:board_id, :SYMBOL, :EXCHANGE, :ASOF_DATE, :SOURCE, :CREATED_AT)
                                """),
                                records_cons
                            )
                            conn.commit()
                        
                        inserted_total += len(records_cons)
                        LOG.info(f"    → 插入 {len(records_cons)} 条成分股")
                    
                except Exception as e:
                    LOG.info(f"    采集 {board_name}({board_id}) 失败: {str(e)}")
                    time.sleep(10)  # 出错多等一会儿
            
            LOG.info(f"本批次完成，已累计插入成分股：{inserted_total} 条")
            time.sleep(30)  # 批次间更长休息

        LOG.info(f"\n行业成分表采集完成，总插入 {inserted_total} 条记录")

    except Exception as e:
        LOG.info("行业成分采集整体异常:", str(e))
   
    # finished  INDUSTRY_CONS_TABLE

def get_concept_map():
    # ==============================
    # 5. 采集 & 插入 概念板块成分表（同上逻辑）
    # ==============================
    LOG.info("\n=== 开始采集概念板块成分股（逐板块调用，注意限频更严格） ===")
    
    try:
        with engine.connect() as conn:
            df_concept_boards = pd.read_sql(
                text(f"""
                    SELECT CONCEPT_ID, CONCEPT_NAME 
                    FROM {SCHEMA_NAME}.{CONCEPT_MASTER_TABLE}
                    WHERE ASOF_DATE = :dt
                """),
                conn,
                params={"dt": today}
            )
            
            LOG.info(f"找到 {len(df_concept_boards)} 个概念板块需要采集成分股")
            
            if df_concept_boards.empty:
                LOG.info("当天概念主表无数据，跳过成分采集")
            else:
                conn.execute(
                    text(f"DELETE FROM {SCHEMA_NAME}.{CONCEPT_CONS_TABLE} WHERE ASOF_DATE = :dt"),
                    {"dt": today}
                )
                conn.commit()
        
        inserted_total_concept = 0
        for i in range(0, len(df_concept_boards), BATCH_SIZE):
            batch_boards = df_concept_boards.iloc[i:i + BATCH_SIZE]
            
            for _, row in batch_boards.iterrows():
                concept_id = row['concept_id']
                concept_name = row['concept_name']
                
                try:
                    time.sleep(SLEEP_BASE + random.random() * SLEEP_RANDOM + 2)  # 概念接口更严格
                    
                    LOG.info(f"  正在采集概念：{concept_name} ({concept_id}) ...")
                    df_cons = ak.stock_board_concept_cons_em(symbol=concept_id)
                    
                    if df_cons.empty:
                        LOG.info(f"    → 无成分股或接口返回空")
                        continue
                    
                    df_cons = df_cons[['代码', '名称']].drop_duplicates()
                    df_cons = df_cons.rename(columns={'代码': 'SYMBOL', '名称': 'STOCK_NAME'})
                    
                    df_cons['CONCEPT_ID'] = concept_id
                    df_cons['EXCHANGE'] = df_cons['SYMBOL'].apply(get_exchange)
                    df_cons['ASOF_DATE'] = today
                    df_cons['SOURCE'] = 'AKSHARE_EASTMONEY'
                    df_cons['CREATED_AT'] = pd.Timestamp.now()
                    
                    records_cons = df_cons.to_dict('records')
                    
                    if records_cons:
                        with engine.connect() as conn:
                            conn.execute(
                                text(f"""
                                    INSERT INTO {SCHEMA_NAME}.{CONCEPT_CONS_TABLE}
                                    (CONCEPT_ID, SYMBOL, EXCHANGE, ASOF_DATE, SOURCE, CREATED_AT)
                                    VALUES 
                                    (:CONCEPT_ID, :SYMBOL, :EXCHANGE, :ASOF_DATE, :SOURCE, :CREATED_AT)
                                """),
                                records_cons
                            )
                            conn.commit()
                        
                        inserted_total_concept += len(records_cons)
                        LOG.info(f"    → 插入 {len(records_cons)} 条成分股")
                    
                except Exception as e:
                    LOG.info(f"    采集 {concept_name}({concept_id}) 失败: {str(e)}")
                    time.sleep(15)
            
            LOG.info(f"本批次完成，已累计插入概念成分股：{inserted_total_concept} 条")
            time.sleep(45)  # 概念批次间更长休息

        LOG.info(f"\n概念成分表采集完成，总插入 {inserted_total_concept} 条记录")

    except Exception as e:
        LOG.info("概念成分采集整体异常:", str(e))

    LOG.info("\n=== 所有采集任务完成 ===")


# ==============================
#           主程序
# ==============================
if __name__ == "__main__":
    today = date.today()   # 2026-01-09 或当前采集日期
    #get_industry_concept_records()

    #get_industry_map()
    get_concept_map()
    LOG.info("脚本执行结束。")