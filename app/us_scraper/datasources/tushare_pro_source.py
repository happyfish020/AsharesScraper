from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional

import pandas as pd

from app.us_scraper.config import get_tushare_token
from app.us_scraper.datasources.base import BaseDataSource
from app.utils.tushare_pro_client import build_tushare_pro_client


DEFAULT_PRICE_FIELDS = "ts_code,trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount"


@dataclass(frozen=True)
class TushareEndpoint:
    api_name: str
    ts_code: str
    fields: str = DEFAULT_PRICE_FIELDS


class TushareProDailySource(BaseDataSource):
    """Daily market data source backed by the shared AshareScraper Tushare HTTP client.

    This class is for non-A-share/global data only. It does not change any existing
    A-share loader task or A-share Tushare calling path.
    """

    def __init__(
        self,
        endpoints: TushareEndpoint | list[TushareEndpoint],
        name: Optional[str] = None,
        data_dir: str = "data/us_scraper",
        state_dir: str = "state/us_scraper",
        config_path: str = "",
    ) -> None:
        if isinstance(endpoints, TushareEndpoint):
            endpoints = [endpoints]
        if not endpoints:
            raise ValueError("At least one Tushare endpoint is required.")
        self.endpoints = endpoints
        self.config_path = config_path
        display_name = name or endpoints[0].ts_code.lower().replace(".", "_").replace("^", "")
        super().__init__(display_name, data_dir=data_dir, state_dir=state_dir)

    def _fetch(self, start_date: date, end_date: date) -> pd.DataFrame:
        token = get_tushare_token(self.config_path)
        if not token:
            raise RuntimeError("Tushare token is required. Fill config/config.json:tushare_token or env TUSHARE_TOKEN.")
        client = build_tushare_pro_client(token)
        start = start_date.strftime("%Y%m%d")
        end = end_date.strftime("%Y%m%d")
        try:
            errors: list[str] = []
            for endpoint in self.endpoints:
                try:
                    raw = client.query(
                        endpoint.api_name,
                        fields=endpoint.fields,
                        ts_code=endpoint.ts_code,
                        start_date=start,
                        end_date=end,
                    )
                except Exception as exc:
                    errors.append(f"{endpoint.api_name}:{endpoint.ts_code}: {exc}")
                    continue
                df = self._normalize_tushare_frame(raw)
                if not df.empty:
                    return df
            if errors:
                raise RuntimeError("; ".join(errors))
            return pd.DataFrame()
        finally:
            client.close()

    @staticmethod
    def _normalize_tushare_frame(raw: pd.DataFrame) -> pd.DataFrame:
        if raw is None or raw.empty or "trade_date" not in raw.columns:
            return pd.DataFrame()
        out = raw.copy()
        out["date"] = pd.to_datetime(out["trade_date"], format="%Y%m%d", errors="coerce")
        out = out.dropna(subset=["date"]).set_index("date").sort_index()
        out = out.rename(columns={"vol": "volume", "pct_chg": "pct_change"})
        for col in ("ts_code", "trade_date"):
            if col in out.columns:
                out = out.drop(columns=[col])
        for col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
        if "adj_close" not in out.columns and "close" in out.columns:
            out["adj_close"] = out["close"]
        return out.dropna(how="all", axis=1)


def us_equity_or_etf_source(symbol: str, *, name: str | None = None, config_path: str = "") -> TushareProDailySource:
    return TushareProDailySource(
        TushareEndpoint("us_daily", symbol.upper()),
        name=name or symbol.lower(),
        config_path=config_path,
    )


def global_index_source(ts_code: str, *, name: str, config_path: str = "") -> TushareProDailySource:
    return TushareProDailySource(
        [
            TushareEndpoint("index_global", ts_code.upper()),
            TushareEndpoint("us_daily", ts_code.upper()),
        ],
        name=name,
        config_path=config_path,
    )
