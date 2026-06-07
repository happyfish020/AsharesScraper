from datetime import date
from urllib.parse import quote

import pandas as pd
import requests

from app.us_scraper.datasources.base import BaseDataSource


class WikipediaPageviewsRiskSource(BaseDataSource):
    """
    Aggregate daily Wikipedia pageviews of market-risk related pages.

    Notes:
    - Wikimedia pageviews API starts from 2015-07.
    - Output columns:
      - views_total: sum of selected pages' daily views
      - views_pages_covered: number of pages returned for each day
    """

    BASE_URL = (
        "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
        "en.wikipedia.org/all-access/user/{article}/daily/{start}/{end}"
    )
    ARTICLES = [
        "S&P_500",
        "CBOE_Volatility_Index",
        "Recession",
        "Federal_Reserve",
        "Credit_spread",
    ]

    def __init__(self) -> None:
        super().__init__("wikipedia_pageviews_risk")

    def _fetch(self, start_date: date, end_date: date) -> pd.DataFrame:
        # API coverage starts mid-2015.
        coverage_start = date(2015, 7, 1)
        effective_start = max(start_date, coverage_start)
        if effective_start > end_date:
            return pd.DataFrame()

        s = requests.Session()
        headers = {"User-Agent": "USScraper/1.0 (market-risk-monitor)"}
        start_s = effective_start.strftime("%Y%m%d")
        end_s = end_date.strftime("%Y%m%d")

        frames: list[pd.DataFrame] = []
        for article in self.ARTICLES:
            url = self.BASE_URL.format(article=quote(article, safe=""), start=start_s, end=end_s)
            resp = s.get(url, headers=headers, timeout=30)
            resp.raise_for_status()
            items = resp.json().get("items", [])
            if not items:
                continue
            rows = []
            for it in items:
                ts = str(it.get("timestamp", ""))[:8]
                d = pd.to_datetime(ts, format="%Y%m%d", errors="coerce")
                v = pd.to_numeric(it.get("views"), errors="coerce")
                if pd.isna(d):
                    continue
                rows.append((d, v))
            if not rows:
                continue
            cur = pd.DataFrame(rows, columns=["date", article]).set_index("date")
            frames.append(cur)

        if not frames:
            return pd.DataFrame()

        out = pd.concat(frames, axis=1).sort_index()
        out["views_total"] = out.sum(axis=1, skipna=True)
        out["views_pages_covered"] = out.notna().sum(axis=1)
        out.index.name = "date"
        return out[["views_total", "views_pages_covered"]]

