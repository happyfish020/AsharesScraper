from datetime import date
import os
import time

import pandas as pd
import requests

from app.us_scraper.datasources.base import BaseDataSource


class CNNFearGreedSource(BaseDataSource):
    """
    CNN Fear & Greed daily index source.

    Field output:
      - index_value (0-100)
    """

    URL = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"

    def __init__(self, name: str = "fear_greed") -> None:
        super().__init__(name=name)

    def _fetch(self, start_date: date, end_date: date) -> pd.DataFrame:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            ),
            "Accept": "application/json,text/plain,*/*",
            "Referer": "https://edition.cnn.com/markets/fear-and-greed",
            "Origin": "https://edition.cnn.com",
            "Accept-Language": "en-US,en;q=0.9",
        }
        try:
            payload = None
            for _ in range(4):
                resp = requests.get(self.URL, headers=headers, timeout=30)
                if resp.status_code == 418:
                    time.sleep(0.7)
                    continue
                resp.raise_for_status()
                payload = resp.json()
                break
            if payload is None:
                raise ValueError("CNN Fear & Greed endpoint failed after retries.")
            hist = payload.get("fear_and_greed_historical", {})
            data = hist.get("data", []) if isinstance(hist, dict) else []
            if not data:
                raise ValueError("CNN payload contains no historical data.")

            df = pd.DataFrame(data)
            df = df.rename(columns={"x": "date", "y": "index_value"})
            df["date"] = pd.to_datetime(df["date"], unit="ms", utc=True).dt.tz_convert(None)
            df["index_value"] = pd.to_numeric(df["index_value"], errors="coerce")
            df = df.dropna(subset=["date", "index_value"])
            mask = (df["date"].dt.date >= start_date) & (df["date"].dt.date <= end_date)
            df = df.loc[mask, ["date", "index_value"]]
            return df.set_index("date").sort_index()
        except Exception:
            if not self._has_local_cache():
                raise
            local = self._read_local()
            idx = pd.to_datetime(local.index)
            mask = (idx.date >= start_date) & (idx.date <= end_date)
            return local.loc[mask]

    def _has_local_cache(self) -> bool:
        return bool(self.data_path) and os.path.exists(self.data_path)
