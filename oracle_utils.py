from sqlalchemy import text

# =====================================================
# Table DDLs (带 exchange 字段)
# =====================================================
TABLE_DDLS = {
    "cn_universe_symbols": """
    CREATE TABLE cn_universe_symbols (
        symbol      VARCHAR2(6)  NOT NULL,
        exchange    VARCHAR2(4)  NOT NULL,
        name        VARCHAR2(50),
        sw_l1       VARCHAR2(50),
        sw_l2       VARCHAR2(50),
        sw_l3       VARCHAR2(50),
        source      VARCHAR2(30),
        created_at  DATE DEFAULT SYSDATE,
        CONSTRAINT pk_cn_universe_symbols PRIMARY KEY (symbol, exchange)
    )
    """,

    "cn_stock_daily_price": """
    CREATE TABLE cn_stock_daily_price (
        symbol        VARCHAR2(6)  NOT NULL,
        exchange      VARCHAR2(4)  NOT NULL,
        trade_date    DATE         NOT NULL,
        close         NUMBER(10,4),
        turnover         NUMBER(10,4),
     
        chg_pct         NUMBER(10,4),
        source        VARCHAR2(30),
        window_start  DATE,
        created_at    DATE DEFAULT SYSDATE,
        CONSTRAINT pk_cn_stock_price PRIMARY KEY (symbol, exchange, trade_date)
    )
    """,

    "cn_index_daily_price": """
    CREATE TABLE cn_index_daily_price (
        index_code    VARCHAR2(20) NOT NULL,
        trade_date    DATE         NOT NULL,
        close         NUMBER(10,4),
        source        VARCHAR2(30),
        created_at    DATE DEFAULT SYSDATE,
        CONSTRAINT pk_cn_index_price PRIMARY KEY (index_code, trade_date)
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
    column_ddl: 如 'exchange VARCHAR2(4)'
    """
    if column_exists(conn, table_name, column_name):
        return
    print(f"[DB] altering table add column: {table_name}.{column_name}")
    conn.execute(text(f"ALTER TABLE {table_name} ADD ({column_ddl})"))


def create_tables_if_not_exists(conn):
    """
    conn: sqlalchemy.engine.Connection
    """
    # 1) Create tables if missing
    for table, ddl in TABLE_DDLS.items():
        if table_exists(conn, table):
            print(f"[DB] table exists, skip: {table}")
            continue
        print(f"[DB] creating table: {table}")
        conn.execute(text(ddl))

    # 2) Ensure critical columns exist (for already-created tables)
    # Universe: exchange + composite PK not auto-migrated, but we at least add column if missing.
    # (PK 迁移属于重建表级别操作，这里不自动做，避免破坏数据)
    if table_exists(conn, "cn_universe_symbols"):
        ensure_column(conn, "cn_universe_symbols", "exchange", "exchange VARCHAR2(4)")

    if table_exists(conn, "cn_stock_daily_price"):
        ensure_column(conn, "cn_stock_daily_price", "exchange", "exchange VARCHAR2(4)")

    # Index table无需补列
