from sqlalchemy import create_engine, text, inspect
import pandas as pd

# =====================================================
# 新版 Table DDLs（纯英文列名，完全匹配 AKShare 返回字段）
# =====================================================
TABLE_DDLS = {
    "cn_universe_symbols": """
    CREATE TABLE cn_universe_symbols (
        symbol      VARCHAR2(10) NOT NULL,   -- 长度放宽支持 SH/SZ 前缀（如 sh600000）
        exchange    VARCHAR2(10) NOT NULL,   -- SSE / SZSE
        name        VARCHAR2(100),
        sw_l1       VARCHAR2(100),
        sw_l2       VARCHAR2(100),
        sw_l3       VARCHAR2(100),
        source      VARCHAR2(30),
        created_at  DATE DEFAULT SYSDATE,
        CONSTRAINT pk_cn_universe_symbols PRIMARY KEY (symbol)
    )
    """,

    "cn_stock_daily_price": """
    CREATE TABLE cn_stock_daily_price (
        symbol         VARCHAR2(10)    NOT NULL,
        trade_date     DATE            NOT NULL,
        open           NUMBER(15,4),
        close          NUMBER(15,4),
        pre_close      NUMBER(15,4),          -- 昨收/前收（用于收益率/缺口等计算）
        high           NUMBER(15,4),
        low            NUMBER(15,4),
        volume         NUMBER(20,4),
        amount         NUMBER(20,4),          -- 成交额（原 turnover）
        amplitude      NUMBER(10,4),
        chg_pct        NUMBER(15,8),
        change         NUMBER(15,4),
        turnover_rate  NUMBER(10,4),          -- 换手率
        source         VARCHAR2(30),
        window_start   DATE,
        created_at     DATE DEFAULT SYSDATE,
        exchange       VARCHAR2(10),
        CONSTRAINT pk_cn_stock_daily_price PRIMARY KEY (symbol, trade_date)
    )
    """,

    "cn_index_daily_price": """
    CREATE TABLE cn_index_daily_price (
        index_code     VARCHAR2(20)    NOT NULL,
        trade_date     DATE            NOT NULL,
        open           NUMBER(15,4),
        close          NUMBER(15,4),
        pre_close      NUMBER(15,4),          -- 昨收/前收（用于收益率/缺口等计算）
        high           NUMBER(15,4),
        low            NUMBER(15,4),
        volume         NUMBER(20,4),
        amount         NUMBER(20,4),          -- 成交额
        source         VARCHAR2(30),
        created_at     DATE DEFAULT SYSDATE,
        pre_close      NUMBER(15,4),          -- 保留历史兼容
        chg_pct        NUMBER(15,4),          -- 保留历史兼容
        CONSTRAINT pk_cn_index_daily_price PRIMARY KEY (index_code, trade_date)
    )
    """
}


def table_exists(conn, table_name: str) -> bool:
    sql = text("""
        SELECT COUNT(*)
        FROM user_tables
        WHERE table_name = :t
    """)
    return conn.execute(sql, {"t": table_name.upper()}).scalar_one() > 0


def column_exists(conn, table_name: str, column_name: str) -> bool:
    sql = text("""
        SELECT COUNT(*)
        FROM user_tab_cols
        WHERE table_name = :t AND column_name = :c
    """)
    return conn.execute(sql, {"t": table_name.upper(), "c": column_name.upper()}).scalar_one() > 0


def ensure_column(conn, table_name: str, column_name: str, column_ddl: str):
    """
    如果列不存在则自动 ALTER TABLE ADD
    column_ddl: 如 'exchange VARCHAR2(10)'
    """
    if column_exists(conn, table_name, column_name):
        return
    print(f"[DB] Altering table add column: {table_name}.{column_name}")
    conn.execute(text(f"ALTER TABLE {table_name} ADD {column_name} {column_ddl}"))


def create_tables_if_not_exists(conn):
    """
    conn: sqlalchemy.engine.Connection
    创建表（如果不存在）并确保关键列存在
    """
    for table, ddl in TABLE_DDLS.items():
        if table_exists(conn, table):
            print(f"[DB] Table exists, skip creation: {table}")
            continue
        print(f"[DB] Creating table: {table}")
        conn.execute(text(ddl))
        conn.commit()  # 建表后立即提交

    # 确保旧表迁移时关键列存在（可选自动补列）
    if table_exists(conn, "cn_universe_symbols"):
        ensure_column(conn, "cn_universe_symbols", "exchange", "VARCHAR2(10)")

    if table_exists(conn, "cn_stock_daily_price"):
        ensure_column(conn, "cn_stock_daily_price", "exchange", "VARCHAR2(10)")
        ensure_column(conn, "cn_stock_daily_price", "pre_close", "NUMBER(15,4)")
        # 可根据需要继续补其他列（如 open, high 等），但建议手动迁移新字段

    # Index 表暂不自动补列
    conn.commit()

DB_USER = "secopr"
DB_PASSWORD = "secopr"
DB_HOST = "localhost"
DB_PORT = "1521"
DB_SERVICE = "xe"
SCHEMA_NAME = "SECOPR"



DSN = f"{DB_HOST}:{DB_PORT}/{DB_SERVICE}"
CONNECTION_STRING = f"oracle+oracledb://{DB_USER}:{DB_PASSWORD}@{DSN}"
 
def get_engine():

    engine = create_engine(
        CONNECTION_STRING,
        pool_pre_ping=True,
        future=True,
    )
    return engine
