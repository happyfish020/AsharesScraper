import re
import akshare as ak
import pandas as pd
import time



import baostock as bs
import numpy as np

from typing import List, Dict, Optional



from wireguard_helper import activate_tunnel, deactivate_tunnel, switch_wire_guard
from logger import setup_logging, get_logger
LOG = get_logger("Main")


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
    name: str,
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

    # 重命名列
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

    # 日期转换
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")

    # 数值列转换
    numeric_cols = [
        "open", "close", "high", "low", "volume", "amount",
        "amplitude", "chg_pct", "change", "turnover_rate"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 关键修改：计算 pre_close 和 chg_pct（参考 load_index_price_em）
    df = df.sort_values("trade_date").reset_index(drop=True)
    df["pre_close"] = df["close"].shift(1)
    df["chg_pct"] = ((df["close"] - df["pre_close"]) / df["pre_close"] * 100).round(4)

    # 第一行没有前一日数据，置为空
    df.loc[0, ["pre_close", "chg_pct"]] = None, None
    df["name"] = name
    # 保留所有需要的列，如果缺少则补 None
    keep_cols = [
        "trade_date",
        "name",
        "open",
        "close",
        "high",
        "low",
        "volume",
        "amount",
        "amplitude",
        "chg_pct",
        "change",
        "turnover_rate",
        "pre_close",           # 新增
    ]
    for c in keep_cols:
        if c not in df.columns:
            df[c] = None

    df = df.dropna(subset=["trade_date", "close"]).copy()

    return df[keep_cols]


def load_stock_price_eastmoney_20260123_no(
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

    max_retries = 1
    for attempt in range(1, max_retries + 1):
        #deactivate_tunnel("cn")
        #LOG.info("deactivate_tunnel - in load_index_price_em")
         
        #activate_tunnel("cn")
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
                LOG.info(e)
                raise RuntimeError(
                    f"获取指数 {index_code} 日线数据失败（已重试 {max_retries} 次）：{str(e)}"
                ) from e
            #switch_wire_guard("cn")
            #LOG.info("deactivate_tunnel - in load_index_price_em")
            #time.sleep(60)
            #activate_tunnel("cn")
            #LOG.info("activate_tunnel - in load_index_price_em")
    
    #   
            

    return None




def load_stocks_price_baostock_batch(
    stock_codes: List[str],
    start_date: str,
    end_date: str,
    names: Optional[Dict[str, str]] = None,  # 可选：{code: name} 字典
    adjust: str = "qfq",
    sleep_interval: int = 20,                # 每多少只股票 sleep 一次
    sleep_seconds: float = 1.0
) -> Dict[str, Optional[pd.DataFrame]]:
    """
    批量使用 baostock 拉取多只 A 股股票日线行情
    
    参数:
    - stock_codes: List[str]，纯6位数字代码列表，例如 ["600519", "000001", "688111"]
    - names: Optional[Dict[str, str]]，代码到名称的映射，用于填充 name 列
    - adjust: "qfq" 前复权, "hfq" 后复权, 其他/空 不复权
    
    返回:
    Dict[str, Optional[pd.DataFrame]]，key 为原始6位代码，value 为对应 DataFrame 或 None
    """
    results: Dict[str, Optional[pd.DataFrame]] = {}

    # 日期标准化
    def normalize_date_ymd(d: str) -> str:
        d = d.replace("-", "").strip()
        if len(d) == 8 and d.isdigit():
            return f"{d[:4]}-{d[4:6]}-{d[6:8]}"
        raise ValueError(f"Date format error: {d}")

    start_dt = normalize_date_ymd(start_date)
    end_dt   = normalize_date_ymd(end_date)

    # 登录（只登录一次）
    lg = bs.login()
    if lg.error_code != '0':
        print(f"baostock login failed: {lg.error_msg}")
        return results  # 返回空结果

    print("baostock login success")

    # 复权映射
    adjust_map = {"qfq": "2", "hfq": "3"}
    adjustflag = adjust_map.get(adjust, "1")  # 默认不复权

    fields = "date,code,open,high,low,close,preclose,volume,amount,turn,pctChg"

    processed = 0

    for stock_code in stock_codes:
        code_clean = str(stock_code).strip()
        if len(code_clean) != 6 or not code_clean.isdigit():
            print(f"Invalid code format: {stock_code}")
            results[code_clean] = None
            continue

        # 确定交易所前缀
        if code_clean.startswith(('6', '68', '69')):
            symbol = f"sh.{code_clean}"
        elif code_clean.startswith(('0', '3')):
            symbol = f"sz.{code_clean}"
        elif code_clean.startswith(('4', '8')):
            symbol = f"bj.{code_clean}"
        else:
            print(f"Unknown prefix for code: {code_clean}")
            results[code_clean] = None
            continue

        name = names.get(code_clean, "") if names else ""

        try:
            rs = bs.query_history_k_data_plus(
                code=symbol,
                fields=fields,
                start_date=start_dt,
                end_date=end_dt,
                frequency="d",
                adjustflag=adjustflag
            )

            if rs.error_code != '0':
                print(f"Query error {symbol}: {rs.error_code} - {rs.error_msg}")
                results[code_clean] = None
                continue

            df = rs.get_data()
            if df.empty:
                results[code_clean] = None
                continue

            # 重命名
            rename_map = {
                'date':       'trade_date',
                'open':       'open',
                'high':       'high',
                'low':        'low',
                'close':      'close',
                'preclose':   'pre_close',
                'volume':     'volume',
                'amount':     'amount',
                'turn':       'turnover_rate',
                'pctChg':     'chg_pct_raw',
            }
            df = df.rename(columns=rename_map)

            # 转数值
            numeric_cols = ['open', 'high', 'low', 'close', 'pre_close',
                            'volume', 'amount', 'turnover_rate', 'chg_pct_raw']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            # 计算衍生字段
            df = df.sort_values("trade_date").reset_index(drop=True)
            df['amplitude'] = ((df['high'] - df['low']) / df['pre_close'] * 100
                               ).where(df['pre_close'] != 0, np.nan)
            df['change'] = df['close'] - df['pre_close']
            df['chg_pct'] = ((df['close'] - df['pre_close']) / df['pre_close'] * 100).round(4)
            df.loc[0, ['pre_close', 'change', 'chg_pct', 'amplitude']] = None

            df['name'] = name

            keep_cols = [
                "trade_date", "name", "open", "close", "high", "low",
                "volume", "amount", "amplitude", "chg_pct", "change",
                "turnover_rate", "pre_close"
            ]

            for c in keep_cols:
                if c not in df.columns:
                    df[c] = None

            df = df[keep_cols].dropna(subset=["trade_date", "close"]).copy()
            df = df.replace([np.nan, np.inf, -np.inf], None)

            results[code_clean] = df

            print(f"Success: {code_clean} ({symbol})  rows: {len(df)}")

        except Exception as e:
            print(f"Exception {code_clean} ({symbol}): {str(e)}")
            results[code_clean] = None

        # 频率控制
        processed += 1
        if processed % sleep_interval == 0:
            print(f"Processed {processed} stocks, sleeping {sleep_seconds}s ...")
            time.sleep(sleep_seconds)

    bs.logout()
    print("baostock logout")
    print(f"Batch finished. Success: {sum(1 for v in results.values() if v is not None)} / {len(stock_codes)}")

    return results


