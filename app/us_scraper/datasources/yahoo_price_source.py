from __future__ import annotations

from datetime import date, timedelta
from io import StringIO
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

from app.us_scraper.datasources.base import BaseDataSource
from app.us_scraper.datasources.tushare_pro_source import us_equity_or_etf_source, global_index_source


class YahooFinanceDailySource(BaseDataSource):
    """Daily OHLCV source for US ETFs/equities using yfinance first, then Yahoo chart CSV.

    The runtime used by this project does not always have Tushare ``us_daily`` permission.
    This source avoids that paid endpoint for ETF proxies such as SPY/QQQ/SOXX/IEF/TLT.
    """

    def __init__(
        self,
        symbol: str,
        *,
        name: Optional[str] = None,
        data_dir: str = "data/us_scraper",
        state_dir: str = "state/us_scraper",
    ) -> None:
        self.symbol = symbol.upper().strip()
        super().__init__(name or self.symbol.lower().replace("^", ""), data_dir=data_dir, state_dir=state_dir)

    def _fetch(self, start_date: date, end_date: date) -> pd.DataFrame:
        errors: list[str] = []
        try:
            df = self._fetch_with_yfinance(start_date, end_date)
            if not df.empty:
                return df
        except Exception as exc:  # pragma: no cover - depends on optional package/network
            errors.append(f"yfinance:{self.symbol}: {exc}")
        try:
            df = self._fetch_with_yahoo_chart_csv(start_date, end_date)
            if not df.empty:
                return df
        except Exception as exc:  # pragma: no cover - depends on network
            errors.append(f"yahoo_chart:{self.symbol}: {exc}")
        raise RuntimeError("; ".join(errors) or f"YahooFinanceDailySource returned no rows for {self.symbol}")

    def _fetch_with_yfinance(self, start_date: date, end_date: date) -> pd.DataFrame:
        import yfinance as yf  # type: ignore

        # yfinance end is exclusive; add one calendar day so the requested end date is included.
        raw = yf.download(
            self.symbol,
            start=start_date.isoformat(),
            end=(end_date + timedelta(days=1)).isoformat(),
            auto_adjust=False,
            progress=False,
            threads=False,
        )
        return self._normalize_price_frame(raw)

    def _fetch_with_yahoo_chart_csv(self, start_date: date, end_date: date) -> pd.DataFrame:
        period1 = int(pd.Timestamp(start_date).timestamp())
        period2 = int(pd.Timestamp(end_date + timedelta(days=1)).timestamp())
        url = (
            f"https://query1.finance.yahoo.com/v7/finance/download/{self.symbol}"
            f"?period1={period1}&period2={period2}&interval=1d&events=history&includeAdjustedClose=true"
        )
        req = Request(url, headers={"User-Agent": "Mozilla/5.0 AshareScraper/USGlobalRisk"})
        try:
            with urlopen(req, timeout=30) as resp:
                text = resp.read().decode("utf-8", errors="replace")
        except (HTTPError, URLError) as exc:
            raise RuntimeError(exc) from exc
        raw = pd.read_csv(StringIO(text))
        return self._normalize_price_frame(raw)

    @staticmethod
    def _normalize_price_frame(raw: pd.DataFrame) -> pd.DataFrame:
        if raw is None or raw.empty:
            return pd.DataFrame()
        df = raw.copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [str(c[0]) for c in df.columns]
        if "Date" in df.columns:
            df["date"] = pd.to_datetime(df["Date"], errors="coerce")
            df = df.drop(columns=["Date"])
        elif "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        else:
            df["date"] = pd.to_datetime(df.index, errors="coerce")
        df = df.dropna(subset=["date"]).set_index("date").sort_index()
        rename = {
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Adj_Close": "adj_close",
            "Volume": "volume",
        }
        df = df.rename(columns=rename)
        keep = [c for c in ("open", "high", "low", "close", "adj_close", "volume") if c in df.columns]
        if not keep:
            return pd.DataFrame()
        out = df[keep].copy()
        for col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
        if "adj_close" not in out.columns and "close" in out.columns:
            out["adj_close"] = out["close"]
        return out.dropna(how="all")


class FallbackDailyPriceSource(BaseDataSource):
    """Composite daily source: primary provider first, then fallbacks.

    Intended for daily risk proxies where missing data is worse than provider purity.
    The final CSV/table name remains stable (for example ``spy.csv`` / ``us_spy``).
    """

    def __init__(self, name: str, providers: list[BaseDataSource]) -> None:
        if not providers:
            raise ValueError("FallbackDailyPriceSource requires at least one provider")
        self.providers = providers
        super().__init__(name, data_dir=providers[0].data_dir, state_dir=providers[0].state_dir)
        for provider in self.providers:
            provider.name = self.name
            provider.data_path = self.data_path
            provider.state_path = self.state_path

    def _fetch(self, start_date: date, end_date: date) -> pd.DataFrame:
        errors: list[str] = []
        for provider in self.providers:
            try:
                df = provider._fetch(start_date, end_date)
                if df is not None and not df.empty:
                    return df
                errors.append(f"{provider.__class__.__name__}: empty")
            except Exception as exc:
                errors.append(f"{provider.__class__.__name__}: {exc}")
        raise RuntimeError("; ".join(errors))


def us_equity_or_etf_fallback_source(symbol: str, *, name: str | None = None, config_path: str = "") -> FallbackDailyPriceSource:
    stable_name = name or symbol.lower()
    return FallbackDailyPriceSource(
        stable_name,
        [
            us_equity_or_etf_source(symbol, name=stable_name, config_path=config_path),
            YahooFinanceDailySource(symbol, name=stable_name),
        ],
    )


def global_index_fallback_source(ts_code: str, *, name: str, yahoo_symbol: str | None = None, config_path: str = "") -> FallbackDailyPriceSource:
    providers: list[BaseDataSource] = [global_index_source(ts_code, name=name, config_path=config_path)]
    if yahoo_symbol:
        providers.append(YahooFinanceDailySource(yahoo_symbol, name=name))
    return FallbackDailyPriceSource(name, providers)
