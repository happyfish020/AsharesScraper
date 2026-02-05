import requests
import json
import pandas as pd
import random
import time
from datetime import datetime

# ==================== 防反爬配置 ====================
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:133.0) Gecko/20100101 Firefox/133.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
]

REFERERS = [
    "https://data.eastmoney.com/zjlx/512880.html",
    "https://data.eastmoney.com/",
    "https://quote.eastmoney.com/",
    "https://www.eastmoney.com/",
]

# 可选：如果你有代理池，可以在这里填入
PROXIES = None   # 示例：{"http": "http://127.0.0.1:7890", "https": "http://127.0.0.1:7890"}

# ==================== 加强版请求函数 ====================
def get_with_anti_block(url, params=None, max_retries=3):
    """
    带随机UA、随机Referer、随机延迟、指数退避重试的请求函数
    """
    for attempt in range(max_retries):
        headers = {
            "User-Agent": random.choice(USER_AGENTS),
            "Referer": random.choice(REFERERS),
            "Accept": "*/*",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "DNT": "1",
        }

        try:
            # 随机延迟 0.6~2.5 秒（更像真人）
            sleep_time = random.uniform(0.6, 2.5)
            print(f"第 {attempt+1} 次尝试，延迟 {sleep_time:.2f} 秒...")
            time.sleep(sleep_time)

            response = requests.get(
                url,
                params=params,
                headers=headers,
                proxies=PROXIES,
                timeout=15
            )

            # 如果被限流（常见 429/403/503），等更长时间再重试
            if response.status_code in [429, 403, 503]:
                wait = (2 ** attempt) * random.uniform(3, 8)
                print(f"被限流，等待 {wait:.1f} 秒后重试...")
                time.sleep(wait)
                continue

            response.raise_for_status()
            return response

        except requests.exceptions.RequestException as e:
            print(f"请求失败: {e}")
            if attempt == max_retries - 1:
                raise
            time.sleep((2 ** attempt) * random.uniform(2, 5))

    return None


# ==================== 主函数：获取 512880 历史资金流向 ====================
def get_512880_zjlx_history(pagesize=500):
    url = "https://push2his.eastmoney.com/api/qt/stock/fflow/kline/get"

    params = {
        "cb": f"jQuery{random.randint(100000000, 999999999)}",
        "secid": "1.512880",
        "fields1": "f1,f2,f3,f5",
        "fields2": "f51,f52,f53,f54,f56,f57,f58,f59,f60,f61,f62,f63,f64",
        "klt": "101",        # 日线
        "lmt": "0",          # 全部历史
        "fqt": "1",
        "_": str(int(datetime.now().timestamp() * 1000))
    }

    print("正在请求东方财富资金流向历史数据...")
    r = get_with_anti_block(url, params=params)

    if r is None:
        print("多次重试后仍失败，请检查网络或稍后重试")
        return None

    text = r.text

    # 去掉 callback
    if text.startswith("jQuery"):
        text = text[text.index("(") + 1 : text.rindex(")")]

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        print("JSON解析失败，可能是接口返回异常")
        return None

    if not data.get("data") or not data["data"].get("klines"):
        print("未获取到数据，可能参数已失效")
        return None

    klines = data["data"]["klines"]

    columns = [
        "日期", "收盘价", "涨跌幅", "未知", "主力净流入(万)", "超大单净额(万)",
        "大单净额(万)", "中单净额(万)", "小单净额(万)",
        "主力净占比", "超大单净占比", "大单净占比", "中单净占比", "小单净占比"
    ]

    rows = [line.split(",") for line in klines]
    df = pd.DataFrame(rows, columns=columns)

    # 类型转换
    numeric_cols = columns[1:]
    df[numeric_cols] = df[numeric_cols].astype(float)
    df["日期"] = pd.to_datetime(df["日期"])

    df = df.sort_values("日期", ascending=False).reset_index(drop=True)

    print(f"成功获取 {len(df)} 条历史资金流向数据（从 {df['日期'].iloc[-1].date()} 到 {df['日期'].iloc[0].date()}）")
    return df


# ==================== 执行 ====================
if __name__ == "__main__":
    df = get_512880_zjlx_history()

    if df is not None:
        print("\n前 10 行预览：")
        print(df.head(10))

        # 保存
        df.to_csv("512880_资金流向历史_加强版.csv", index=False, encoding="utf_8_sig")
        print("\n数据已保存为：512880_资金流向历史_加强版.csv")