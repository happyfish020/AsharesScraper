from datetime import date
import io

import pandas as pd
import requests

from app.us_scraper.datasources.base import BaseDataSource


class CBOEVIXSource(BaseDataSource):
    """VIX daily data from CBOE public CSV."""

    URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"

    def __init__(self) -> None:
        super().__init__("vix")

    def _fetch(self, start_date: date, end_date: date) -> pd.DataFrame:
        resp = requests.get(self.URL, timeout=30)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        if df.empty or "DATE" not in df.columns:
            return df
        df = df.rename(columns=str.lower)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        mask = (df["date"].dt.date >= start_date) & (df["date"].dt.date <= end_date)
        df = df.loc[mask]
        df = df.set_index("date")
        return df
