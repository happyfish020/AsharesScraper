import pandas as pd
from datetime import date
from sqlalchemy import create_engine, text, inspect
import akshare as ak

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

# 表相关常量
SCHEMA_NAME = "SECOPR"
TABLE_NAME = "CN_SECURITY_MASTER"

# ==============================
#     检查并创建表（如果不存在）
# ==============================
def create_table_if_not_exists():
    inspector = inspect(engine)
    
    if not inspector.has_table(TABLE_NAME, schema=SCHEMA_NAME):
        print(f"表 {SCHEMA_NAME}.{TABLE_NAME} 不存在，正在创建...")
        
        create_table_sql = f"""
        CREATE TABLE {SCHEMA_NAME}.{TABLE_NAME} (
            SYMBOL      VARCHAR2(10)   NOT NULL,
            EXCHANGE    VARCHAR2(10)   NOT NULL,
            NAME        VARCHAR2(100),
            FULLNAME    VARCHAR2(200),
            STATUS      VARCHAR2(20)   DEFAULT 'ACTIVE',
            LIST_DATE   DATE,
            DELIST_DATE DATE,
            ASOF_DATE   DATE           NOT NULL,
            SOURCE      VARCHAR2(50)   DEFAULT 'AKSHARE',
            CREATED_AT  DATE           DEFAULT SYSDATE,
            UPDATED_AT  DATE           DEFAULT SYSDATE,
            RAW_JSON    CLOB,
            
            CONSTRAINT CN_SECURITY_MASTER_PK 
                PRIMARY KEY (SYMBOL, EXCHANGE, ASOF_DATE)
        )
        """
        
        create_index1 = f"""
        CREATE INDEX IDX_CN_SEC_MASTER_CODE_EXCH 
        ON {SCHEMA_NAME}.{TABLE_NAME} (SYMBOL, EXCHANGE)
        """
        
        create_index2 = f"""
        CREATE INDEX IDX_CN_SEC_MASTER_NAME 
        ON {SCHEMA_NAME}.{TABLE_NAME} (NAME)
        """
        
        create_index3 = f"""
        CREATE INDEX IDX_CN_SEC_MASTER_ASOF 
        ON {SCHEMA_NAME}.{TABLE_NAME} (ASOF_DATE)
        """
        
        with engine.connect() as conn:
            conn.execute(text(create_table_sql))
            conn.execute(text(create_index1))
            conn.execute(text(create_index2))
            conn.execute(text(create_index3))
            conn.commit()
        print("表结构、主键和常用索引已创建完成")
    else:
        print(f"表 {SCHEMA_NAME}.{TABLE_NAME} 已存在，跳过创建")

# ==============================
#           主程序
# ==============================
if __name__ == "__main__":
    # 1. 确保表结构存在
    create_table_if_not_exists()

    # 2. 获取最新 A 股列表
    print("\n正在从 akshare 获取 A 股基础信息...")
    try:
        df = ak.stock_info_a_code_name()
        print(f"成功获取 {len(df)} 条记录")
    except Exception as e:
        print("获取数据失败:", str(e))
        exit(1)

    # 3. 数据清洗与字段映射
    df = df.rename(columns={
        'code': 'SYMBOL',
        'name': 'NAME'
    })

    df['SYMBOL'] = df['SYMBOL'].astype(str).str.strip()

    def get_exchange(code: str) -> str:
        if code.startswith(('6', '9')):
            return 'SH'
        elif code.startswith(('0', '3')):
            return 'SZ'
        elif code.startswith(('4', '8', '43', '83', '87', '88')) or len(code) > 6:
            return 'BJ'
        return 'UNKNOWN'

    df['EXCHANGE'] = df['SYMBOL'].apply(get_exchange)

    # 补充必要字段
    today = date.today()              # 或者固定使用：date(2026, 1, 9)
    df['ASOF_DATE'] = today
    df['STATUS'] = 'ACTIVE'
    df['SOURCE'] = 'AKSHARE'
    df['CREATED_AT'] = pd.Timestamp.now()

    # 只保留要插入的字段
    columns_to_insert = [
        'SYMBOL', 'EXCHANGE', 'NAME', 'STATUS',
        'ASOF_DATE', 'SOURCE', 'CREATED_AT'
    ]
    df_insert = df[columns_to_insert].copy()

    # 4. 准备插入用的记录列表（字典格式）
    records = [
        {
            'SYMBOL': row['SYMBOL'],
            'EXCHANGE': row['EXCHANGE'],
            'NAME': row['NAME'],
            'STATUS': row['STATUS'],
            'ASOF_DATE': row['ASOF_DATE'],
            'SOURCE': row['SOURCE'],
            'CREATED_AT': row['CREATED_AT']
        }
        for _, row in df_insert.iterrows()
    ]

    # 5. 执行插入（兼容所有 Oracle 版本）
    insert_sql = text("""
        INSERT INTO SECOPR.CN_SECURITY_MASTER 
            (SYMBOL, EXCHANGE, NAME, STATUS, ASOF_DATE, SOURCE, CREATED_AT)
        VALUES 
            (:SYMBOL, :EXCHANGE, :NAME, :STATUS, :ASOF_DATE, :SOURCE, :CREATED_AT)
    """)

    print(f"\n准备插入 {len(records)} 条记录...")

    try:
        with engine.connect() as conn:
            # 先删除当天旧数据（幂等）
            conn.execute(
                text("DELETE FROM SECOPR.CN_SECURITY_MASTER WHERE ASOF_DATE = :dt"),
                {"dt": today}
            )

            # 分批插入（防止单次太大导致内存或超时问题）
            batch_size = 800
            inserted_count = 0

            for i in range(0, len(records), batch_size):
                batch = records[i:i + batch_size]
                conn.execute(insert_sql, batch)
                inserted_count += len(batch)
                print(f"已插入 {inserted_count}/{len(records)} 条...")

            conn.commit()
            print(f"\n插入完成！共插入 {inserted_count} 条记录（日期：{today}）")

    except Exception as e:
        print("\n插入失败：", str(e))
        raise

    # 6. 可选：验证插入结果
    try:
        with engine.connect() as conn:
            count = conn.execute(
                text("SELECT COUNT(*) FROM SECOPR.CN_SECURITY_MASTER WHERE ASOF_DATE = :dt"),
                {"dt": today}
            ).scalar()
            print(f"验证：当天记录总数 = {count}")
    except Exception as e:
        print("验证查询失败:", str(e))