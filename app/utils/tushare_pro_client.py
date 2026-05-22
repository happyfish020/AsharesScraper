from __future__ import annotations

import json
import os
from functools import partial

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class SessionDataApi:
    def __init__(
        self,
        token: str,
        *,
        timeout: float = 30.0,
        base_url: str | None = None,
        pool_connections: int = 8,
        pool_maxsize: int = 16,
        max_retries: int = 3,
        backoff_factor: float = 1.0,
    ) -> None:
        self._token = str(token or "").strip()
        self._timeout = float(timeout)
        self._http_url = (
            str(base_url or "").strip()
            or os.getenv("TUSHARE_PRO_BASE_URL", "").strip()
            or "http://api.waditu.com/dataapi"
        ).rstrip("/")

        retry = Retry(
            total=max(0, int(max_retries)),
            connect=max(0, int(max_retries)),
            read=max(0, int(max_retries)),
            status=max(0, int(max_retries)),
            backoff_factor=max(0.0, float(backoff_factor)),
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=None,
            raise_on_status=False,
        )
        adapter = HTTPAdapter(
            max_retries=retry,
            pool_connections=max(1, int(pool_connections)),
            pool_maxsize=max(1, int(pool_maxsize)),
        )
        self._session = requests.Session()
        self._session.mount("http://", adapter)
        self._session.mount("https://", adapter)
        self._session.headers.update(
            {
                "User-Agent": "AsharesScraperV2-TusharePro/1.0",
                "Connection": "keep-alive",
                "Content-Type": "application/json",
            }
        )

    def query(self, api_name, fields="", **kwargs):
        kwargs.setdefault("ts_type_name", self._http_url)
        req_params = {
            "api_name": api_name,
            "token": self._token,
            "params": kwargs,
            "fields": fields,
        }

        res = self._session.post(
            f"{self._http_url}/{api_name}",
            json=req_params,
            timeout=self._timeout,
        )
        res.raise_for_status()
        result = json.loads(res.text)
        if result["code"] != 0:
            raise Exception(result["msg"])
        data = result["data"]
        return pd.DataFrame(data["items"], columns=data["fields"])

    def close(self) -> None:
        self._session.close()

    def __getattr__(self, name):
        return partial(self.query, name)


def build_tushare_pro_client(token: str, *, timeout: float | None = None) -> SessionDataApi:
    timeout_value = timeout
    if timeout_value is None:
        timeout_value = float(os.getenv("TUSHARE_PRO_TIMEOUT_SECONDS", "30"))
    return SessionDataApi(
        token=token,
        timeout=float(timeout_value),
        pool_connections=int(os.getenv("TUSHARE_PRO_POOL_CONNECTIONS", "8")),
        pool_maxsize=int(os.getenv("TUSHARE_PRO_POOL_MAXSIZE", "16")),
        max_retries=int(os.getenv("TUSHARE_PRO_HTTP_MAX_RETRIES", "3")),
        backoff_factor=float(os.getenv("TUSHARE_PRO_HTTP_BACKOFF_SECONDS", "1.0")),
    )
