from datetime import date
import io

import pandas as pd
import requests

from app.us_scraper.datasources.base import BaseDataSource


class VIXTermStructureSource(BaseDataSource):
    """VIX term structure from CBOE public index history CSV files."""

    URLS = {
        "vix9d": "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX9D_History.csv",
        "vix": "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv",
        "vix3m": "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX3M_History.csv",
        "vix6m": "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX6M_History.csv",
        # VXMT may return 403 in some regions/accounts, so treat as optional.
        "vxmt": "https://cdn.cboe.com/api/global/us_indices/daily_prices/VXMT_History.csv",
    }

    def __init__(self) -> None:
        super().__init__("vix_term")

    def _fetch(self, start_date: date, end_date: date) -> pd.DataFrame:
        out = None
        for name, url in self.URLS.items():
            resp = requests.get(url, timeout=30)
            if resp.status_code != 200:
                continue
            df = pd.read_csv(io.StringIO(resp.text))
            if df.empty or "DATE" not in df.columns or "CLOSE" not in df.columns:
                continue
            cur = pd.DataFrame(
                {
                    "date": pd.to_datetime(df["DATE"], errors="coerce"),
                    name: pd.to_numeric(df["CLOSE"], errors="coerce"),
                }
            ).dropna(subset=["date"])
            cur = cur.set_index("date")
            out = cur if out is None else out.join(cur, how="outer")

        if out is None or out.empty:
            return pd.DataFrame()

        mask = (out.index.date >= start_date) & (out.index.date <= end_date)
        out = out.loc[mask].sort_index()

        if {"vix9d", "vix"}.issubset(out.columns):
            out["term_9d_minus_1m"] = out["vix9d"] - out["vix"]
        if {"vix3m", "vix"}.issubset(out.columns):
            out["term_3m_minus_1m"] = out["vix3m"] - out["vix"]

        out.index.name = "date"
        return out
