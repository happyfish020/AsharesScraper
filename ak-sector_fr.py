import pandas as pd
from datetime import date
import time
import random
from sqlalchemy import create_engine, text, inspect
import akshare as ak

# 数据库连接配置（全局常量，建议移到配置文件）
DB_CONFIG = {
    'user': "secopr",
    'password': "secopr",
    'host': "localhost",
    'port': "1521",
    'service': "xe"
}

SCHEMA_NAME = "SECOPR"

def get_engine():
    """创建并返回数据库引擎"""
    dsn = f"{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['service']}"
    conn_str = f"oracle+oracledb://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{dsn}"
    return create_engine(
        conn_str,
        pool_pre_ping=True,
        future=True,
    )

def create_table_if_not_exists(engine, table_name: str, create_sql: str, indexes: list = None):
    """检查表是否存在，不存在则创建表及索引"""
    inspector = inspect(engine)
    full_table = f"{SCHEMA_NAME}.{table_name}"
    
    if not inspector.has_table(table_name, schema=SCHEMA_NAME):
        print(f"创建表: {full_table}")
        with engine.connect() as conn:
            conn.execute(text(create_sql))
            if indexes:
                for idx_sql in indexes:
                    conn.execute(text(idx_sql))
            conn.commit()
        print(f"表 {full_table} 创建成功")
    else:
        print(f"表 {full_table} 已存在，跳过创建")


def get_exchange(code: str) -> str:
    """根据股票代码判断交易所"""
    code = str(code).strip()
    if code.startswith(('6', '9')):
        return 'SH'
    elif code.startswith(('0', '3')):
        return 'SZ'
    elif code.startswith(('4', '8', '43', '83', '87', '88')) or len(code) > 6:
        return 'BJ'
    return 'UNKNOWN'


def clean_numeric_series(series):
    """清洗数值列：处理常见无效值并转为 float / None"""
    series = series.replace(['--', '-', '—', '暂无', '新股', '', '涨停', '跌停'], None)
    series = pd.to_numeric(series, errors='coerce')
    return series


# ==================== 功能函数区域 ====================

def collect_and_save_a_stock_basic():
    """采集并保存 A股基础信息（股票列表）"""
    engine = get_engine()
    today = date.today()
    
    # 获取数据
    df = ak.stock_info_a_code_name()
    df['SYMBOL'] = df['code'].astype(str).str.strip()
    df['EXCHANGE'] = df['SYMBOL'].apply(get_exchange)
    df = df.rename(columns={'name': 'NAME'})
    
    df['ASOF_DATE'] = today
    df['STATUS'] = 'ACTIVE'
    df['SOURCE'] = 'AKSHARE'
    df['CREATED_AT'] = pd.Timestamp.now()
    
    columns = ['SYMBOL', 'EXCHANGE', 'NAME', 'STATUS', 'ASOF_DATE', 'SOURCE', 'CREATED_AT']
    
    records = df[columns].to_dict('records')
    
    # 插入（幂等）
    with engine.connect() as conn:
        conn.execute(
            text("DELETE FROM SECOPR.CN_SECURITY_MASTER WHERE ASOF_DATE = :dt"),
            {"dt": today}
        )
        if records:
            conn.execute(
                text("""
                    INSERT INTO SECOPR.CN_SECURITY_MASTER
                    (SYMBOL, EXCHANGE, NAME, STATUS, ASOF_DATE, SOURCE, CREATED_AT)
                    VALUES (:SYMBOL, :EXCHANGE, :NAME, :STATUS, :ASOF_DATE, :SOURCE, :CREATED_AT)
                """),
                records
            )
        conn.commit()
    
    print(f"[A股基础信息] 插入完成：{len(records)} 条")


def collect_and_save_hot_rank():
    """采集并保存东方财富 A股热度排名（人气榜前100）"""
    engine = get_engine()
    today = date.today()
    
    df = ak.stock_hot_rank_em()
    
    # 列名映射（根据实际输出来调整）
    column_map = {
        '当前排名': 'CURRENT_RANK',
        '代码': 'SYMBOL',
        '股票名称': 'STOCK_NAME',
        '最新价': 'LATEST_PRICE',
        '涨跌额': 'CHANGE_AMOUNT',
        '涨跌幅': 'CHANGE_PCT'
    }
    
    # 只保留存在的列并重命名
    rename_dict = {k: v for k, v in column_map.items() if k in df.columns}
    df = df.rename(columns=rename_dict)
    
    # 清洗数值字段
    for col in ['LATEST_PRICE', 'CHANGE_AMOUNT', 'CHANGE_PCT']:
        if col in df.columns:
            df[col] = clean_numeric_series(df[col])
    
    # 统一代码格式（取后6位）
    if 'SYMBOL' in df.columns:
        df['SYMBOL'] = df['SYMBOL'].astype(str).str.strip().str[-6:]
    
    df['RANK_DATE'] = today
    df['ASOF_DATE'] = today
    df['SOURCE'] = 'AKSHARE_EASTMONEY'
    df['CREATED_AT'] = pd.Timestamp.now()
    
    records = df.to_dict('records')
    
    with engine.connect() as conn:
        conn.execute(
            text("DELETE FROM SECOPR.CN_STOCK_HOT_RANK_EM WHERE ASOF_DATE = :dt"),
            {"dt": today}
        )
        
        if records:
            conn.execute(
                text("""
                    INSERT INTO SECOPR.CN_STOCK_HOT_RANK_EM
                    (RANK_DATE, CURRENT_RANK, SYMBOL, STOCK_NAME,
                     LATEST_PRICE, CHANGE_AMOUNT, CHANGE_PCT,
                     ASOF_DATE, SOURCE, CREATED_AT)
                    VALUES
                    (:RANK_DATE, :CURRENT_RANK, :SYMBOL, :STOCK_NAME,
                     :LATEST_PRICE, :CHANGE_AMOUNT, :CHANGE_PCT,
                     :ASOF_DATE, :SOURCE, :CREATED_AT)
                """),
                records
            )
        conn.commit()
    
    print(f"[热度排名] 插入完成：{len(records)} 条")


# 示例：其他功能函数的框架（可继续扩展）
def collect_industry_master():
    """采集并保存行业板块主表"""
    pass  # 实现类似上面逻辑


def collect_concept_master():
    """采集并保存概念板块主表"""
    pass


def collect_board_constituents(board_type='industry', sleep_base=8, max_boards=None):
    """
    采集板块成分股（行业或概念）
    board_type: 'industry' 或 'concept'
    """
    pass  # 实现逐个板块采集 + 防限频逻辑


# ==================== 主调度示例 ====================
if __name__ == "__main__":
    engine = get_engine()
    
    # 1. 创建所有需要的表（可单独运行一次）
    # create_table_if_not_exists(engine, "CN_SECURITY_MASTER", ...)
    # create_table_if_not_exists(engine, "CN_STOCK_HOT_RANK_EM", ...)
    # ... 其他表创建 ...
    
    today = date.today()
    print(f"采集日期: {today}\n")
    
    # 按需调用各个功能
    collect_and_save_a_stock_basic()
    time.sleep(3)
    
    collect_and_save_hot_rank()
    time.sleep(3)
    
    # collect_industry_master()
    # collect_concept_master()
    # collect_board_constituents('industry', sleep_base=10, max_boards=50)
    # collect_board_constituents('concept', sleep_base=12)
    
    print("\n所有指定采集任务完成")