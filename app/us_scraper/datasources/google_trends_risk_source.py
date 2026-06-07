import json
import time
from datetime import date

import pandas as pd
import requests

from app.us_scraper.datasources.base import BaseDataSource


class GoogleTrendsRiskSource(BaseDataSource):
    """
    Google Trends interest over time for risk-related keywords (US).

    Uses unofficial public web endpoints:
    - /trends/api/explore -> token
    - /trends/api/widgetdata/multiline -> timeline

    Google may return 429 depending on IP/rate-limit.
    """

    EXPLORE_URL = "https://trends.google.com/trends/api/explore"
    MULTILINE_URL = "https://trends.google.com/trends/api/widgetdata/multiline"
    KEYWORDS = ["recession", "stock market crash", "credit spread", "liquidity crisis"]

    def __init__(self) -> None:
        super().__init__("google_trends_risk")

    @staticmethod
    def _strip_guard_prefix(text: str) -> str:
        prefix = ")]}',"
        if text.startswith(prefix):
            parts = text.split("\n", 1)
            return parts[1] if len(parts) > 1 else ""
        return text

    def _fetch(self, start_date: date, end_date: date) -> pd.DataFrame:
        s = requests.Session()
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        timeframe = f"{start_date.isoformat()} {end_date.isoformat()}"

        comparison = [{"keyword": kw, "geo": "US", "time": timeframe} for kw in self.KEYWORDS]
        req_obj = {"comparisonItem": comparison, "category": 0, "property": ""}

        # Retry a few times for transient 429.
        explore_json = None
        for retry in range(3):
            r = s.get(
                self.EXPLORE_URL,
                params={"hl": "en-US", "tz": "0", "req": json.dumps(req_obj)},
                headers=headers,
                timeout=45,
            )
            if r.status_code == 429 and retry < 2:
                time.sleep(8 + retry * 6)
                continue
            if r.status_code != 200:
                return pd.DataFrame()
            text = self._strip_guard_prefix(r.text)
            try:
                explore_json = json.loads(text)
            except json.JSONDecodeError:
                return pd.DataFrame()
            break

        if not explore_json:
            return pd.DataFrame()

        token = None
        widget_req = None
        for w in explore_json.get("widgets", []):
            if w.get("id") == "TIMESERIES":
                token = w.get("token")
                widget_req = w.get("request")
                break
        if not token or not widget_req:
            return pd.DataFrame()

        r2 = s.get(
            self.MULTILINE_URL,
            params={"hl": "en-US", "tz": "0", "req": json.dumps(widget_req, separators=(",", ":")), "token": token},
            headers=headers,
            timeout=45,
        )
        if r2.status_code != 200:
            return pd.DataFrame()
        try:
            data = json.loads(self._strip_guard_prefix(r2.text))
        except json.JSONDecodeError:
            return pd.DataFrame()
        timeline = data.get("default", {}).get("timelineData", [])
        if not timeline:
            return pd.DataFrame()

        rows = []
        for item in timeline:
            ts = pd.to_datetime(pd.to_numeric(item.get("time"), errors="coerce"), unit="s", errors="coerce")
            values = item.get("value", [])
            if pd.isna(ts) or not isinstance(values, list):
                continue
            numeric_vals = [pd.to_numeric(v, errors="coerce") for v in values]
            if len(numeric_vals) < len(self.KEYWORDS):
                numeric_vals += [None] * (len(self.KEYWORDS) - len(numeric_vals))
            rows.append([ts, *numeric_vals[: len(self.KEYWORDS)]])

        if not rows:
            return pd.DataFrame()

        columns = ["date"] + [f"gt_{i+1}" for i in range(len(self.KEYWORDS))]
        out = pd.DataFrame(rows, columns=columns).set_index("date").sort_index()
        out["gt_risk_mean"] = out.mean(axis=1, skipna=True)
        out.index.name = "date"
        return out
