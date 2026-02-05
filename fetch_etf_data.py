# -*- coding: utf-8 -*-
"""
东方财富 ETF 数据日常拉取（读取配置文件中的 ut）

运行前请确保已通过 update_ut_and_test.py 获得有效 ut
"""

import json
import pandas as pd
from seleniumwire import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import time
CONFIG_FILE = "etf_config.json"
FIELDS = "f12,f14,f2,f3,f4,f5,f6,f15,f16,f17,f18,f62"
FS = "b:MK0021,b:MK0022,b:MK0023,b:MK0024,b:MK0827"


def load_ut():
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            return json.load(f).get("ut_value")
    except:
        return None


def fetch_etf(pz=1000):
    ut = load_ut()
    if not ut:
        print(f"请先运行 update_ut_and_test.py 获取有效 ut 并保存到 {CONFIG_FILE}")
        return None

    params = {
        "pn": 1,
        "pz": pz,
        "po": 1,
        "np": 1,
        "ut": ut,
        "fltt": 2,
        "invt": 2,
        "wbp2u": "|0|0|0|web",
        "fid": "f12",
        "fs": FS,
        "fields": FIELDS,
    }

    url = "https://push2.eastmoney.com/api/qt/clist/get?" + "&".join(f"{k}={v}" for k, v in params.items())

    options = Options()
    options.add_argument("--headless=new")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    try:
        print(f"请求: {url[:80]}...")
        driver.get(url)
        time.sleep(3)

        for req in driver.requests:
            if req.url.startswith("https://") and "api/qt/clist/get" in req.url:
                if req.response and req.response.status_code == 200:
                    try:
                        js = req.response.json()
                        if js.get("rc") == 0:
                            diff = js.get("data", {}).get("diff", [])
                            if diff:
                                df = pd.DataFrame(diff)
                                print(f"成功获取 {len(df)} 条数据")
                                return df
                    except:
                        pass
        print("未获取到有效数据")
        return None

    finally:
        driver.quit()


if __name__ == "__main__":
    df = fetch_etf(pz=1000)
    if df is not None:
        print("\n前 10 行：")
        print(df.head(10).to_string(index=False))
        df.to_csv("etf_latest.csv", index=False, encoding="utf-8-sig")
        print("已保存到 etf_latest.csv")