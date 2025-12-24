import re
import akshare as ak
import pandas as pd


def normalize_stock_code(stock_code: str) -> str:
    """
    统一把各种输入形式转换为 6 位股票代码（纯数字）：

    支持输入：
    - "000001"
    - "000001.SZ" / "600000.SH" / "830001.BJ"
    - "sz000001" / "sh600000" / "bj830001"
    - 其他带杂字符的形式（会尝试抽取 6 位数字）

    返回：
    - "000001" 这种 6 位纯数字
    """
    if stock_code is None:
        raise ValueError("stock_code is None")

    s = str(stock_code).strip().upper()

    # 常见格式：000001.SZ / 600000.SH / 830001.BJ
    m = re.match(r"^(\d{6})\.(SZ|SH|BJ)$", s)
    if m:
        return m.group(1)

    # 常见格式：sz000001 / sh600000 / bj830001
    m = re.match(r"^(SZ|SH|BJ)(\d{6})$", s)
    if m:
        return m.group(2)

    # 纯数字
    if s.isdigit() and len(s) == 6:
        return s

    # 兜底：抽取任意连续 6 位数字
    m = re.search(r"(\d{6})", s)
    if m:
        return m.group(1)

    raise ValueError(f"invalid stock_code input: {stock_code}")


# =====================================================
# 股票（日线）- 东财
# =====================================================

def load_stock_price_eastmoney(
    stock_code: str,
    start_date: str,
    end_date: str,
    adjust: str = "qfq",
) -> pd.DataFrame:
    """
    使用东财接口拉取 A 股股票日线行情

    Parameters
    ----------
    stock_code : str
        推荐传 6 位股票代码（如 "000001"）
        也兼容 "000001.SZ" / "sz000001" / "600000.SH" 等
    start_date : str
        YYYYMMDD
    end_date : str
        YYYYMMDD
    adjust : str
        qfq / hfq / None

    Returns
    -------
    DataFrame columns:
        trade_date | close |turnover |chg_pct
    """
    #code6 = normalize_stock_code(stock_code)

    df = ak.stock_zh_a_hist(
        symbol=stock_code,
        period="daily",
        start_date=start_date,
        end_date=end_date,
        adjust=adjust,
    )

    if df is None or df.empty:
        return None

    df = df.rename(columns={"日期": "trade_date", "收盘": "close",  "成交额": "turnover" ,"涨跌幅": "chg_pct"})
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    
    df["turnover"] = pd.to_numeric(df["turnover"], errors="coerce")
    df["chg_pct"] = pd.to_numeric(df["chg_pct"], errors="coerce")
    df = df.dropna(subset=["trade_date", "close", "turnover", "chg_pct"])

    # ===== 2️⃣ 时间排序（非常重要）=====
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.sort_values("trade_date").reset_index(drop=True)

    # ===== 3️⃣ 补 pre_close（上一交易日 close）=====
    df["pre_close"] = df["close"].shift(1)

    return df[["trade_date", "close", "turnover","pre_close", "chg_pct"]]
 



# =====================================================
# 指数（日线）- 东财（保持）
# =====================================================

def load_index_price_em(
    index_code: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    指数行情（日线，东财）
    index_code 例：
      "sh000300", "sh000001", "sz399006"
    """
    if not isinstance(index_code, str) or not index_code.strip():
        raise ValueError("index_code must be non-empty str")

    df = ak.stock_zh_index_daily_em(
        symbol=index_code,
        start_date=start_date,
        end_date=end_date,
    )

    if df is None or df.empty:
        return None

    df = df.rename(columns={"date": "trade_date", "close": "close", "turnover": "turnover", "chg_pct": "chg_pct" })
     
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    
    df["turnover"] = pd.to_numeric(df["turnover"], errors="coerce")
    df["chg_pct"] = pd.to_numeric(df["chg_pct"], errors="coerce")
    df = df.dropna(subset=["trade_date", "close", "turnover", "chg_pct"])


    # ===== 2️⃣ 时间排序（非常重要）=====
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.sort_values("trade_date").reset_index(drop=True)

    # ===== 3️⃣ 补 pre_close（上一交易日 close）=====
    df["pre_close"] = df["close"].shift(1)
    

    return df[["trade_date", "close", "turnover","pre_close", "chg_pct"]]
