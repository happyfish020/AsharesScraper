from __future__ import annotations

import configparser
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests

from data_pipeline.common.cache import CacheStore

TS_URL = "https://api.tushare.pro"


def resolve_tushare_token(cli_token: str = "", config_path: str = "") -> str:
    token = str(cli_token or "").strip()
    if token:
        return token
    for env_key in ("TUSHARE_TOKEN", "TS_TOKEN"):
        env_value = os.getenv(env_key, "").strip()
        if env_value:
            return env_value
    candidates = [Path(config_path)] if str(config_path or "").strip() else []
    candidates.extend([Path(".env"), Path("ts_config.ini"), Path("docs/my-test.ini")])
    for path in candidates:
        if not path.exists() or not path.is_file():
            continue
        if path.suffix.lower() == ".ini":
            parser = configparser.ConfigParser()
            try:
                parser.read(path, encoding="utf-8")
            except configparser.ParsingError:
                continue
            for section in parser.sections():
                for key in ("token", "tushare_token", "ts_token"):
                    value = parser.get(section, key, fallback="").strip()
                    if value:
                        return value
        else:
            for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
                if "=" not in line:
                    continue
                key, value = line.split("=", 1)
                if key.strip().upper() in {"TUSHARE_TOKEN", "TS_TOKEN"} and value.strip():
                    return value.strip().strip("'\"")
    # Fallback: try importing the default token from app tools
    try:
        from app.tools.sync_cn_stock_daily_price_from_tushare import DEFAULT_TUSHARE_TOKEN

        if DEFAULT_TUSHARE_TOKEN:
            return DEFAULT_TUSHARE_TOKEN
    except (ImportError, AttributeError):
        pass
    raise SystemExit("Tushare token is required. Use --token, env TUSHARE_TOKEN/TS_TOKEN, or --config.")


@dataclass(slots=True)
class TushareClient:
    token: str
    logger: object
    min_interval_seconds: float = 0.18
    retries: int = 5
    timeout_seconds: float = 40.0
    session: requests.Session | None = None
    cache: CacheStore | None = None
    _last_call_at: float = 0.0

    def __post_init__(self) -> None:
        if self.session is None:
            object.__setattr__(self, "session", requests.Session())
        self.session.trust_env = False
        if self.cache is None:
            object.__setattr__(self, "cache", CacheStore())

    def _sleep_for_rate_limit(self) -> None:
        elapsed = time.time() - self._last_call_at
        if elapsed < self.min_interval_seconds:
            time.sleep(self.min_interval_seconds - elapsed)

    def call(self, api_name: str, params: dict | None = None, fields: str = "", *, cache_key: str = "", use_cache: bool = True) -> pd.DataFrame:
        key = cache_key or json.dumps({"api": api_name, "params": params or {}, "fields": fields}, sort_keys=True, ensure_ascii=True)
        if use_cache:
            cached = self.cache.load_frame("tushare", key)
            if cached is not None:
                return cached.copy()
        payload = {"api_name": api_name, "token": self.token, "params": params or {}, "fields": fields}
        last_error: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                self._sleep_for_rate_limit()
                response = self.session.post(TS_URL, json=payload, timeout=self.timeout_seconds)
                self._last_call_at = time.time()
                response.raise_for_status()
                data = response.json()
                if int(data.get("code", -1)) != 0:
                    raise RuntimeError(f"{api_name} failed: {data.get('msg')}")
                block = data.get("data") or {}
                frame = pd.DataFrame(block.get("items") or [], columns=block.get("fields") or [])
                if use_cache:
                    self.cache.save_frame("tushare", key, frame)
                return frame
            except Exception as exc:
                last_error = exc
                wait_seconds = min(12.0, 1.5 * attempt)
                self.logger.warning("tushare_retry api=%s attempt=%s/%s wait=%.1fs err=%s", api_name, attempt, self.retries, wait_seconds, exc)
                time.sleep(wait_seconds)
        raise RuntimeError(f"{api_name} failed after retries: {last_error}")

    def paginate(
        self,
        api_name: str,
        params: dict | None,
        fields: str,
        *,
        page_size: int = 5000,
        key_prefix: str = "",
    ) -> pd.DataFrame:
        pages: list[pd.DataFrame] = []
        offset = 0
        while True:
            page_params = dict(params or {})
            page_params["offset"] = offset
            page_params["limit"] = page_size
            key = f"{key_prefix}|offset={offset}|limit={page_size}"
            frame = self.call(api_name, page_params, fields, cache_key=key, use_cache=True)
            if frame.empty:
                break
            pages.append(frame)
            if len(frame.index) < page_size:
                break
            offset += page_size
        if not pages:
            return pd.DataFrame(columns=[item.strip() for item in fields.split(",") if item.strip()])
        return pd.concat(pages, ignore_index=True)
