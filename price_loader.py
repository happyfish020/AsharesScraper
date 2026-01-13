import re
import akshare as ak
import pandas as pd
import time

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
) -> pd.DataFrame | None:
    """
    使用东财接口拉取 A 股股票日线行情（AKShare: stock_zh_a_hist）

    Returns
    -------
    DataFrame columns（英文列名对齐 DB DDL / AK 返回语义）:
        trade_date, open, close, high, low,
        volume, amount, amplitude, chg_pct, change, turnover_rate, pre_close
    """
    df = ak.stock_zh_a_hist(
        symbol=stock_code,
        period="daily",
        start_date=start_date,
        end_date=end_date,
        adjust=adjust,
    )

    if df is None or df.empty:
        return None

    rename_map = {
        "日期": "trade_date",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "成交额": "amount",
        "振幅": "amplitude",
        "涨跌幅": "chg_pct",
        "涨跌额": "change",
        "换手率": "turnover_rate",
    }
    df = df.rename(columns=rename_map)

    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")

    for col in [
        "open",
        "close",
        "pre_close",
        "high",
        "low",
        "volume",
        "amount",
        "amplitude",
        "chg_pct",
        "change",
        "turnover_rate",
        "pre_close",
        
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["trade_date", "close"]).copy()
    df = df.sort_values("trade_date").reset_index(drop=True)
    
    keep_cols = [
        "trade_date",
        "open",
        "close",
        "pre_close",
        "high",
        "low",
        "volume",
        "amount",
        "amplitude",
        "chg_pct",
        "change",
        "turnover_rate",
    
    ]
    for c in keep_cols:
        if c not in df.columns:
            df[c] = None

    return df[keep_cols]

def load_index_price_em(
    index_code: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame | None:
    """
    指数行情（日线，东财；AKShare: stock_zh_index_daily_em）

    AK 返回列（英文）:
        date, open, close, high, low, volume, amount

    DB DDL 对齐（并保留兼容列）:
        trade_date, open, close, high, low, volume, amount, pre_close, chg_pct

    新增：3 次重试机制
    """
    if not isinstance(index_code, str) or not index_code.strip():
        raise ValueError("index_code must be non-empty str")

    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            df = ak.stock_zh_index_daily_em(
                symbol=index_code,
                start_date=start_date,
                end_date=end_date,
            )

            if df is None or df.empty:
                return None

            df = df.rename(columns={"date": "trade_date"})

            df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
            for col in ["open", "close", "high", "low", "volume", "amount"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            df = df.dropna(subset=["trade_date", "close"]).copy()
            df = df.sort_values("trade_date").reset_index(drop=True)

            df["pre_close"] = df["close"].shift(1)
            df["chg_pct"] = ((df["close"] - df["pre_close"]) / df["pre_close"] * 100).round(4)
            df.loc[0, ["pre_close", "chg_pct"]] = None, None

            keep_cols = [
                "trade_date",
                "open",
                "close",
                "high",
                "low",
                "volume",
                "amount",
                "pre_close",
                "chg_pct",
            ]
            for c in keep_cols:
                if c not in df.columns:
                    df[c] = None
            return df[keep_cols]

        except Exception as e:
            if attempt == max_retries:
                raise RuntimeError(
                    f"获取指数 {index_code} 日线数据失败（已重试 {max_retries} 次）：{str(e)}"
                ) from e
            time.sleep(2)

    return None

