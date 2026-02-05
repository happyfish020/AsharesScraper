import requests
import json
import pandas as pd
from datetime import datetime

def get_512880_zjlx_history(pagesize=500, page=1):
    """
    获取 证券ETF(512880) 历史资金流向数据
    """
    url = "https://push2his.eastmoney.com/api/qt/stock/fflow/kline/get"

    params = {
        "cb":         "jQuery随便写点",           # callback 可随意
        "secid":      "1.512880",                 # 沪市etf用1.前缀
        "fields1":    "f1,f2,f3,f5",
        "fields2":    "f51,f52,f53,f54,f56,f57,f58,f59,f60,f61,f62,f63,f64",
        "klt":        "101",                      # 101=日线
        "lmt":        "0",                        # 0=全部历史
        "fqt":        "1",
        "_":          str(int(datetime.now().timestamp()*1000))
    }

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36",
        "Referer":    "https://data.eastmoney.com/",
        "Accept":     "*/*",
        "Accept-Encoding": "gzip, deflate, br",
    }

    try:
        r = requests.get(url, params=params, headers=headers, timeout=10)
        r.raise_for_status()

        text = r.text

        # 去掉 callback 外壳
        if text.startswith("jQuery"):
            text = text[text.index("(")+1 : text.rindex(")")]
        
        data = json.loads(text)
        
        if data.get("data") is None or data["data"].get("klines") is None:
            print("未获取到数据，可能secid或参数有误")
            return None

        klines = data["data"]["klines"]
        
        # 字段对应（根据fields2顺序）
        columns = [
            "日期", "收盘价", "涨跌幅", "?","主力净流入(万)","超大单净额(万)",
            "大单净额(万)","中单净额(万)","小单净额(万)",
            "主力净占比","超大单净占比","大单净占比","中单净占比","小单净占比"
        ]

        rows = [line.split(",") for line in klines]
        
        df = pd.DataFrame(rows, columns=columns[:len(rows[0])])
        # 数值列转float
        numeric_cols = columns[1:]
        df[numeric_cols] = df[numeric_cols].astype(float)
        
        # 日期列转datetime
        df["日期"] = pd.to_datetime(df["日期"])
        df = df.sort_values("日期", ascending=False).reset_index(drop=True)
        
        return df

    except Exception as e:
        print("请求失败:", e)
        return None


if __name__ == "__main__":
    df = get_512880_zjlx_history()
    if df is not None:
        print(df.head(15))
        # df.to_csv("512880_资金流向历史.csv", index=False, encoding="utf_8_sig")