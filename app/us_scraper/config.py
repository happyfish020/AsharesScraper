from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.json"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists() or not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def load_config(config_path: str = "") -> dict[str, Any]:
    candidates: list[Path] = []
    if str(config_path or "").strip():
        candidates.append(Path(config_path).expanduser())
    env_cfg = os.getenv("ASHARE_CONFIG", "").strip() or os.getenv("MARKET_CONFIG", "").strip()
    if env_cfg:
        candidates.append(Path(env_cfg).expanduser())
    candidates.extend([DEFAULT_CONFIG_PATH, PROJECT_ROOT / "config.json"])

    merged: dict[str, Any] = {}
    seen: set[Path] = set()
    for raw in candidates:
        path = raw if raw.is_absolute() else (PROJECT_ROOT / raw)
        path = path.resolve()
        if path in seen:
            continue
        seen.add(path)
        if path.suffix.lower() == ".json":
            merged.update(_read_json(path))
    return merged


def get_tushare_token(config_path: str = "") -> str:
    cfg = load_config(config_path)
    for key in ("tushare_token", "TUSHARE_TOKEN", "ts_token", "TS_TOKEN", "token"):
        value = cfg.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    nested = cfg.get("tushare")
    if isinstance(nested, dict):
        for key in ("token", "tushare_token", "ts_token", "TUSHARE_TOKEN", "TS_TOKEN"):
            value = nested.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    for key in ("TUSHARE_TOKEN", "TS_TOKEN"):
        value = os.getenv(key, "").strip()
        if value:
            return value
    return ""
