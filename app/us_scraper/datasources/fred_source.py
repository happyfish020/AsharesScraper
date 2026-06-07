from __future__ import annotations

import io
import logging
import os
import time
from contextlib import contextmanager
from datetime import date
from typing import Iterator

import pandas as pd
import requests

from app.us_scraper.datasources.base import BaseDataSource


LOG = logging.getLogger("Runner")


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "y", "on"}


@contextmanager
def _temporary_disable_wireguard_for_fred() -> Iterator[None]:
    """Temporarily stop WireGuard for FRED requests, then restore it.

    A-share collection still keeps its normal runner-level WireGuard behavior.
    This context is used only by FREDSeriesSource because fred.stlouisfed.org
    can be slow or blocked through the active CN tunnel.

    Environment controls:
      US_FRED_DISABLE_VPN=0      -> do not stop WireGuard for FRED
      US_FRED_WIREGUARD_TUNNEL   -> tunnel name, default: cn
      US_FRED_RESTORE_VPN=0      -> do not restore the tunnel after FRED
    """
    if not _env_flag("US_FRED_DISABLE_VPN", default=True):
        yield
        return

    tunnel_name = os.getenv("US_FRED_WIREGUARD_TUNNEL", "cn").strip() or "cn"
    should_restore = False
    try:
        from app.utils.wireguard_helper import activate_tunnel, deactivate_tunnel, is_wireguard_running

        should_restore = bool(is_wireguard_running(tunnel_name))
        if should_restore:
            LOG.info("[US-GLOBAL][FRED] temporarily disabling WireGuard tunnel=%s for fred.stlouisfed.org", tunnel_name)
            deactivate_tunnel(tunnel_name)
        yield
    except Exception as exc:
        # Do not make FRED unavailable just because VPN control is unavailable.
        LOG.warning("[US-GLOBAL][FRED] WireGuard temporary-disable skipped/failed: %s", exc)
        yield
    finally:
        if should_restore and _env_flag("US_FRED_RESTORE_VPN", default=True):
            try:
                from app.utils.wireguard_helper import activate_tunnel

                activate_tunnel(tunnel_name)
                LOG.info("[US-GLOBAL][FRED] restored WireGuard tunnel=%s after FRED request", tunnel_name)
            except Exception as exc:
                LOG.warning("[US-GLOBAL][FRED] failed to restore WireGuard tunnel=%s: %s", tunnel_name, exc)


class FREDSeriesSource(BaseDataSource):
    """
    Simple CSV-based FRED series downloader (no API key required).

    It uses the public CSV endpoint:
      https://fred.stlouisfed.org/graph/fredgraph.csv?id=SERIES_ID

    For FRED only, requests bypass environment proxies and temporarily disable
    the WireGuard tunnel by default. This lets us test whether FRED timeouts are
    caused by the active VPN while keeping A-share scraping behavior unchanged.
    """

    BASE_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
    REQUEST_TIMEOUT_SECONDS = int(os.getenv("US_FRED_TIMEOUT_SECONDS", "30"))
    MAX_ATTEMPTS = int(os.getenv("US_FRED_MAX_ATTEMPTS", "3"))
    RETRY_SLEEP_SECONDS = float(os.getenv("US_FRED_RETRY_SLEEP_SECONDS", "5"))

    def __init__(self, series_id: str, name: str | None = None) -> None:
        self.series_id = series_id
        display_name = name or f"fred_{series_id.lower()}"
        super().__init__(display_name)

    def _fetch(self, start_date: date, end_date: date) -> pd.DataFrame:
        params = {
            "id": self.series_id,
        }
        resp = self._get_with_retry(params)
        df = pd.read_csv(
            io.StringIO(resp.text),
        )
        df = df.rename(columns={df.columns[0]: "date", self.series_id: "value"})
        df["date"] = pd.to_datetime(df["date"])
        mask = (df["date"].dt.date >= start_date) & (df["date"].dt.date <= end_date)
        df = df.loc[mask]
        df = df.set_index("date")
        return df

    def _get_with_retry(self, params: dict[str, str]) -> requests.Response:
        last_error: Exception | None = None
        with _temporary_disable_wireguard_for_fred():
            session = requests.Session()
            # Bypass HTTP_PROXY/HTTPS_PROXY/REQUESTS_CA_BUNDLE env effects for FRED.
            # This does not replace WireGuard shutdown above; it only prevents proxy env leakage.
            session.trust_env = False
            try:
                for attempt in range(1, max(1, self.MAX_ATTEMPTS) + 1):
                    try:
                        resp = session.get(self.BASE_URL, params=params, timeout=self.REQUEST_TIMEOUT_SECONDS)
                        resp.raise_for_status()
                        return resp
                    except requests.RequestException as exc:
                        last_error = exc
                        if attempt >= max(1, self.MAX_ATTEMPTS):
                            break
                        time.sleep(max(0.0, self.RETRY_SLEEP_SECONDS))
            finally:
                session.close()
        assert last_error is not None
        raise last_error
