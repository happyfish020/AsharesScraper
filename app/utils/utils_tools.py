import re

from sqlalchemy import inspect

 
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

