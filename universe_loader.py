"""
universe_loader.py

职责：
- 加载 A 股股票 universe（申万行业体系）
- 优先从本地静态文件读取
- 本地不存在时，调用 ak 接口一次性生成并缓存

注意：
- universe 被视为“低频静态元数据”
- 不在此模块做任何数据库写入
"""

import os
import json
import akshare as ak
from typing import Dict


# =====================================================
# CONFIG
# =====================================================

DATA_DIR = "data/universe"
UNIVERSE_FILE = os.path.join(DATA_DIR, "sw_universe.json")


# =====================================================
# CORE
# =====================================================

def load_universe() -> Dict[str, dict]:
    """
    返回格式：
    {
        "600519": {
            "name": "...",
            "sw_l1": "...",
            "sw_l2": "...",
            "sw_l3": "...",
            "source": "sw_index_third_cons"
        },
        ...
    }
    """

    # ---------- 1️⃣ 本地缓存优先 ----------
    if os.path.exists(UNIVERSE_FILE):
        with open(UNIVERSE_FILE, "r", encoding="utf-8") as f:
            universe = json.load(f)
        LOG.info(f"[UNIVERSE] loaded from local file: {UNIVERSE_FILE} ({len(universe)})")
        return universe

    # ---------- 2️⃣ 从 ak 构建 ----------
    LOG.info("[UNIVERSE] local file not found, building from akshare ...")

    universe: Dict[str, dict] = {}

    l1_df = ak.sw_index_first_info()
    for _, row in l1_df.iterrows():
        ind_code = row["行业代码"]

        try:
            cons_df = ak.sw_index_third_cons(symbol=ind_code)
        except Exception as e:
            LOG.info(f"[UNIVERSE][WARN] failed to load industry {ind_code}: {e}")
            continue

        for _, r in cons_df.iterrows():
            symbol = r["股票代码"]
            universe[symbol] = {
                "name": r["股票简称"],
                "sw_l1": r["申万1级"],
                "sw_l2": r["申万2级"],
                "sw_l3": r["申万3级"],
                "source": "sw_index_third_cons",
            }

    # ---------- 3️⃣ 写入本地缓存 ----------
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(UNIVERSE_FILE, "w", encoding="utf-8") as f:
        json.dump(universe, f, ensure_ascii=False, indent=2)

    LOG.info(f"[UNIVERSE] built and saved to {UNIVERSE_FILE} ({len(universe)})")

    return universe
