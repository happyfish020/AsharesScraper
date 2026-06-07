import json
import time
from datetime import date

import pandas as pd
import requests

from app.us_scraper.datasources.base import BaseDataSource


class GDELTRiskSource(BaseDataSource):
    """
    GDELT DOC Timeline volume for market-risk query.

    Output columns:
    - gdelt_count: document count per day matching query
    - gdelt_norm: normalized timeline intensity if returned by API
    """

    API_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
    QUERY = '(recession OR "stock market crash" OR "credit spread" OR "liquidity crisis")'

    def __init__(self) -> None:
        super().__init__("gdelt_risk")

    @staticmethod
    def _dt_str(d: date) -> str:
        return d.strftime("%Y%m%d000000")

    @staticmethod
    def _iter_year_windows(start_date: date, end_date: date) -> list[tuple[date, date]]:
        windows: list[tuple[date, date]] = []
        cur = date(start_date.year, 1, 1)
        if cur < start_date:
            cur = start_date
        while cur <= end_date:
            y_end = date(cur.year, 12, 31)
            w_end = min(y_end, end_date)
            windows.append((cur, w_end))
            cur = date.fromordinal(w_end.toordinal() + 1)
        return windows

    @staticmethod
    def _strip_guard_prefix(text: str) -> str:
        # GDELT may prepend header-like text before JSON.
        i = text.find("{")
        return text[i:] if i >= 0 else text

    def _fetch(self, start_date: date, end_date: date) -> pd.DataFrame:
        s = requests.Session()
        out_frames: list[pd.DataFrame] = []
        windows = self._iter_year_windows(start_date, end_date)

        for i, (w_start, w_end) in enumerate(windows):
            params = {
                "query": self.QUERY,
                "mode": "TimelineVolRaw",
                "format": "json",
                "startdatetime": self._dt_str(w_start),
                "enddatetime": self._dt_str(w_end),
            }

            # GDELT free endpoint rate-limit: <= 1 request / 5 seconds.
            if i > 0:
                time.sleep(5.2)

            payload = None
            for retry in range(2):
                try:
                    resp = s.get(self.API_URL, params=params, timeout=60)
                    if resp.status_code == 429:
                        time.sleep(7 + retry * 4)
                        continue
                    resp.raise_for_status()
                    body = self._strip_guard_prefix(resp.text).strip()
                    if not body or not body.startswith("{"):
                        time.sleep(3 + retry * 2)
                        continue
                    payload = json.loads(body)
                    break
                except Exception:
                    time.sleep(4 + retry * 3)
                    continue
            if not payload:
                continue

            timeline = payload.get("timeline", [])
            if not isinstance(timeline, list) or not timeline:
                continue

            rows: list[tuple[pd.Timestamp, float, float]] = []
            # TimelineVolRaw format:
            # {"timeline":[{"series":"Article Count","data":[{"date":"...","value":...,"norm":...}, ...]}]}
            points = []
            first = timeline[0]
            if isinstance(first, dict) and isinstance(first.get("data"), list):
                points = first.get("data", [])
            else:
                points = timeline

            for point in points:
                if not isinstance(point, dict):
                    continue
                d = pd.to_datetime(point.get("date"), errors="coerce")
                v = pd.to_numeric(point.get("value"), errors="coerce")
                n = pd.to_numeric(point.get("norm"), errors="coerce")
                if pd.isna(d):
                    continue
                rows.append((d, v, n))

            if not rows:
                continue

            cur_df = pd.DataFrame(rows, columns=["date", "gdelt_count", "gdelt_norm"]).set_index("date")
            out_frames.append(cur_df)

        if not out_frames:
            return pd.DataFrame()

        out = pd.concat(out_frames).sort_index()
        out = out[~out.index.duplicated(keep="last")]
        out.index.name = "date"
        return out
