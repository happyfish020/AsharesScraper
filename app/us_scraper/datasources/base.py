import abc
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, date
from typing import Optional

import pandas as pd
from pandas.tseries.offsets import BDay


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _to_date(d: date | datetime | str) -> date:
    if isinstance(d, date) and not isinstance(d, datetime):
        return d
    if isinstance(d, datetime):
        return d.date()
    # string
    return datetime.fromisoformat(d).date()


@dataclass
class DownloadState:
    last_date: Optional[str] = None  # ISO date string


class BaseDataSource(abc.ABC):
    """
    Base class for all data sources.

    Each concrete data source is responsible for implementing the `_fetch`
    method, which returns a DataFrame with a Date index and at least one
    column of data.
    """

    def __init__(
        self,
        name: str,
        data_dir: str = "data/us_scraper",
        state_dir: str = "state/us_scraper",
    ) -> None:
        self.name = name
        self.data_dir = data_dir
        self.state_dir = state_dir

        _ensure_dir(self.data_dir)
        _ensure_dir(self.state_dir)

        safe_name = self.name.replace(" ", "_").lower()
        self.data_path = os.path.join(self.data_dir, f"{safe_name}.csv")
        self.state_path = os.path.join(self.state_dir, f"{safe_name}.json")

    # ---------- public API ----------
    def update(
        self,
        start_date: Optional[date | datetime | str] = None,
        end_date: Optional[date | datetime | str] = None,
        refresh: bool = False,
        overlap_trading_days: int = 7,
    ) -> pd.DataFrame:
        """
        Update local data with resume logic.

        - If `refresh` is True: ignore previous state, redownload full range.
        - Else: continue from the last stored date + 1 day, while re-pulling
          the most recent `overlap_trading_days` business days to backfill misses.
        """
        end_d = _to_date(end_date) if end_date else date.today()
        user_start_d = _to_date(start_date) if start_date else None
        overlap_start = (pd.Timestamp(end_d) - BDay(max(overlap_trading_days, 0))).date()

        if refresh or not os.path.exists(self.data_path):
            start_d = user_start_d if user_start_d else date(end_d.year - 15, end_d.month, end_d.day)
            last_date_in_state = None
        else:
            state = self._load_state()
            last_date_in_state = state.last_date
            if last_date_in_state:
                last_d = _to_date(last_date_in_state)
                resume_start = last_d.fromordinal(last_d.toordinal() + 1)
                start_d = min(resume_start, overlap_start)
            else:
                start_d = overlap_start
            if user_start_d:
                start_d = max(start_d, user_start_d)

        if start_d > end_d:
            # Nothing to fetch; just return existing data
            if os.path.exists(self.data_path):
                return self._read_local()
            else:
                return pd.DataFrame()

        new_df = self._fetch(start_d, end_d)
        if new_df.empty:
            return self._read_local() if os.path.exists(self.data_path) else new_df

        new_df = self._normalize_index(new_df)

        if os.path.exists(self.data_path) and not refresh and last_date_in_state:
            existing = self._read_local()
            combined = pd.concat([existing, new_df])
            combined = combined[~combined.index.duplicated(keep="last")].sort_index()
        else:
            combined = new_df.sort_index()

        self._write_local(combined)

        last_index_date = combined.index.max()
        self._save_state(DownloadState(last_date=last_index_date.strftime("%Y-%m-%d")))

        return combined

    # ---------- methods for subclasses ----------
    @abc.abstractmethod
    def _fetch(self, start_date: date, end_date: date) -> pd.DataFrame:
        """Fetch data from remote source, return DataFrame indexed by Date."""

    def _normalize_index(self, df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(df.index, pd.DatetimeIndex):
            if "Date" in df.columns:
                df = df.set_index("Date")
            elif "date" in df.columns:
                df = df.set_index("date")
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        df.index = df.index.date
        return df

    # ---------- local IO ----------
    @staticmethod
    def _canonical_col_name(col: str) -> str:
        s = str(col).strip().lower()
        if "adj" in s and "close" in s:
            return "adj_close"
        if "volume" in s or re.search(r"\bvol\b", s):
            return "volume"
        if "open" in s:
            return "open"
        if "high" in s:
            return "high"
        if "low" in s:
            return "low"
        if "close" in s:
            return "close"
        s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
        return s

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        norm_cols = []
        for c in df.columns:
            s = str(c).strip().lower()
            m = re.match(r"^\('([^']+)',\s*'[^']+'\)$", s)
            if m:
                s = m.group(1)
            norm_cols.append(self._canonical_col_name(s))
        df = df.copy()
        df.columns = norm_cols
        # Coalesce duplicate columns caused by legacy yfinance tuple headers.
        if df.columns.duplicated().any():
            df = df.T.groupby(level=0).first().T
        return df

    def _read_local(self) -> pd.DataFrame:
        raw = pd.read_csv(self.data_path, low_memory=False)
        if raw.empty:
            return raw

        first_col_name = str(raw.columns[0]).lower()
        if first_col_name in ("price", "date"):
            first_col = raw.iloc[:, 0].astype(str)
            mask = first_col.str.match(r"^\d{4}-\d{2}-\d{2}$", na=False)
            raw = raw.loc[mask].copy()
            if raw.empty:
                return pd.DataFrame()
            dates = pd.to_datetime(raw.iloc[:, 0], errors="coerce")
            raw = raw.iloc[:, 1:].copy()
            raw.index = dates
            raw.index.name = "date"
            raw.columns = [str(c).strip().lower().replace(" ", "_") for c in raw.columns]
        else:
            date_col = "date" if "date" in raw.columns else raw.columns[0]
            raw[date_col] = pd.to_datetime(raw[date_col], errors="coerce")
            raw = raw.dropna(subset=[date_col]).set_index(date_col)
            raw.index.name = "date"
            raw.columns = [str(c).strip().lower().replace(" ", "_") for c in raw.columns]

        raw = self._normalize_columns(raw)
        for c in raw.columns:
            raw[c] = pd.to_numeric(raw[c], errors="coerce")
        raw = raw.sort_index()
        raw.index = raw.index.date
        return raw

    def _write_local(self, df: pd.DataFrame) -> None:
        out = df.copy()
        out.index = pd.to_datetime(out.index)
        out.index.name = "date"
        out.to_csv(self.data_path)

    # ---------- state IO ----------
    def _load_state(self) -> DownloadState:
        if not os.path.exists(self.state_path):
            return DownloadState()
        with open(self.state_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return DownloadState(**data)

    def _save_state(self, state: DownloadState) -> None:
        with open(self.state_path, "w", encoding="utf-8") as f:
            json.dump(state.__dict__, f, indent=2)

