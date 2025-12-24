import json
import os
from datetime import datetime


def load_state(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_state(path: str, data: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)



def mark_failed(state: dict, symbol: str, reason: str):
    now = datetime.now().isoformat(timespec="seconds")
    info = state.get(symbol, {
        "reason": reason,
        "first_failed": now,
        "count": 0
    })
    info["last_failed"] = now
    info["count"] += 1
    state[symbol] = info
