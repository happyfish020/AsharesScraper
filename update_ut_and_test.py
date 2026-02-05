# -*- coding: utf-8 -*-
"""
东方财富 ETF 数据 - 更新/测试 ut_value（使用 selenium-wire + Chrome）

逻辑：
1. 先尝试读取 etf_config.json 中的 ut_value 并测试是否可用
2. 如果可用 → 直接结束（或输出数据预览）
3. 如果不可用 → 打开浏览器捕获最新 ut → 测试新 ut → 成功则写入配置文件
"""

import json
import time
import random
import pandas as pd
from seleniumwire import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

CONFIG_FILE = "etf_config.json"
TARGET_URL = "https://quote.eastmoney.com/center/gridlist.html#fund_etf"

FIELDS = "f12,f14,f2,f3,f4,f5,f6,f15,f16,f17,f18,f62"
FS = "b:MK0021,b:MK0022,b:MK0023,b:MK0024,b:MK0827"


def load_ut_from_config():
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("ut_value")
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return None


def save_ut_to_config(ut_value):
    data = {"ut_value": ut_value, "last_updated": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"已保存 ut 到 {CONFIG_FILE}")


def create_chrome_driver(headless=False):
    options = Options()
    if headless:
        options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1280,800")
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/144.0.0.0 Safari/537.36"
    )

    # selenium-wire 自动拦截请求
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    return driver


def capture_latest_ut(driver):
    print("正在打开页面捕获最新 ut ...")
    driver.get(TARGET_URL)
    time.sleep(8 + random.uniform(0, 4))  # 等待页面发起请求

    for request in driver.requests:
        if "api/qt/clist/get" in request.url:
            print("\n捕获到 clist/get 请求")
            print("URL:", request.url)
            if request.response and request.response.status_code == 200:
                try:
                    data = request.response.json()
                    if data.get("rc") == 0 and data.get("data", {}).get("diff"):
                        ut = request.url.split("ut=")[-1].split("&")[0]
                        print(f"有效 ut: {ut}")
                        return ut
                except Exception:
                    pass
    print("未捕获到有效请求（可能需手动触发）")
    return None


def test_ut_with_selenium_wire(ut_value, pz=500, timeout=25):
    if not ut_value:
        return False, "ut_value 为空"

    params = {
        "pn": 1,
        "pz": pz,
        "po": 1,
        "np": 1,
        "ut": ut_value,
        "fltt": 2,
        "invt": 2,
        "wbp2u": "|0|0|0|web",
        "fid": "f12",
        "fs": FS,
        "fields": FIELDS,
    }

    test_urls = [
        "https://push2.eastmoney.com/api/qt/clist/get",
        "https://88.push2.eastmoney.com/api/qt/clist/get",
        "https://80.push2.eastmoney.com/api/qt/clist/get",
    ]

    driver = create_chrome_driver(headless=True)

    try:
        for base_url in test_urls:
            url = base_url + "?" + "&".join(f"{k}={v}" for k, v in params.items())
            print(f"测试: {base_url} ...")
            driver.get(url)
            time.sleep(2 + random.uniform(0, 2))

            for request in driver.requests:
                if request.url == url and request.response:
                    if request.response.status_code == 200:
                        try:
                            js = request.response.json()
                            if js.get("rc") == 0 and js.get("data", {}).get("diff"):
                                print("测试成功！ rc=0，有 diff 数据")
                                return True, js.get("data", {}).get("diff", [])
                        except:
                            pass
        return False, "所有测试 URL 均失败（无有效响应或 rc≠0）"

    finally:
        driver.quit()


def main():
    print("=== 东方财富 ETF ut 更新与测试 ===\n")

    ut = load_ut_from_config()
    print(f"从配置文件读取 （如果存在）")

    if ut:
        print("先测试现有 ut ...")
        success, result = test_ut_with_selenium_wire(ut, pz=300)
        if success:
            print("现有 ut 有效！")
            if isinstance(result, list) and result:
                df = pd.DataFrame(result)
                print("\n前 8 行预览：")
                print(df.head(8).to_string(index=False))
            return
        else:
            print("测试失败：", result)
            print("→ 将尝试捕获新 ut\n")

    # 捕获新 ut
    driver = create_chrome_driver(headless=False)  # 显示浏览器，便于观察
    try:
        new_ut = capture_latest_ut(driver)
        if new_ut:
            print("\n测试新捕获的 ut ...")
            success, result = test_ut_with_selenium_wire(new_ut, pz=300)
            if success:
                save_ut_to_config(new_ut)
                print("新 ut 已验证有效并保存")
                if isinstance(result, list) and result:
                    df = pd.DataFrame(result)
                    print("\n前 8 行预览：")
                    print(df.head(8).to_string(index=False))
            else:
                print("新 ut 测试失败：", result)
        else:
            print("捕获失败，请手动 F12 → Network 抓取 ut")
    finally:
        driver.quit()


if __name__ == "__main__":
    main()