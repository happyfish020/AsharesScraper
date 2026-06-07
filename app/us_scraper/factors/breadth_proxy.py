from __future__ import annotations

import os
from typing import Dict

import pandas as pd


def _read_yahoo_price_csv(path: str) -> pd.DataFrame:
    """
    Read local Yahoo CSV saved by this project and return a normalized price frame.
    """
    raw = pd.read_csv(path, low_memory=False)
    if raw.empty:
        return pd.DataFrame()

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

    for c in raw.columns:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    return raw.sort_index()


def _load_adj_close(data_dir: str, symbol_name: str) -> pd.Series:
    path = os.path.join(data_dir, f"{symbol_name.lower()}.csv")
    if not os.path.exists(path):
        legacy_path = os.path.join(data_dir, f"yahoo_{symbol_name.lower()}.csv")
        path = legacy_path if os.path.exists(legacy_path) else path
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required CSV not found: {path}")
    df = _read_yahoo_price_csv(path)
    if "adj_close" in df.columns:
        s = df["adj_close"]
    elif "close" in df.columns:
        s = df["close"]
    else:
        raise ValueError(f"No close/adj_close column in: {path}")
    return s.rename(symbol_name.lower())


def build_breadth_proxy_a_b(data_dir: str = "data/us_scraper") -> Dict[str, pd.DataFrame]:
    """
    Build two breadth proxy variants:

    A: No-breadth baseline (proxy=1, stress=0).
    B: ETF proxy from IBB/XBI/GDX:
       breadth_proxy = mean(asset / MA50(asset))
       breadth_stress = clip(1 - breadth_proxy, 0, 1)
    """
    ibb = _load_adj_close(data_dir, "ibb")
    xbi = _load_adj_close(data_dir, "xbi")
    gdx = _load_adj_close(data_dir, "gdx")

    merged = pd.concat([ibb, xbi, gdx], axis=1, join="inner").dropna()
    if merged.empty:
        raise ValueError("No overlapping history for IBB/XBI/GDX.")

    ma50 = merged.rolling(50, min_periods=50).mean()
    ratio = merged / ma50
    ratio = ratio.dropna()
    if ratio.empty:
        raise ValueError("Not enough history to compute MA50 ratios.")

    b = pd.DataFrame(index=ratio.index)
    b["ibb_ratio"] = ratio["ibb"]
    b["xbi_ratio"] = ratio["xbi"]
    b["gdx_ratio"] = ratio["gdx"]
    b["breadth_proxy"] = (b["ibb_ratio"] + b["xbi_ratio"] + b["gdx_ratio"]) / 3.0
    b["breadth_stress"] = (1.0 - b["breadth_proxy"]).clip(lower=0.0, upper=1.0)

    a = pd.DataFrame(index=b.index)
    a["breadth_proxy"] = 1.0
    a["breadth_stress"] = 0.0

    a.index.name = "date"
    b.index.name = "date"
    return {"a": a, "b": b}


def save_breadth_proxy_csvs(proxies: Dict[str, pd.DataFrame], data_dir: str = "data/us_scraper") -> Dict[str, str]:
    os.makedirs(data_dir, exist_ok=True)
    out: Dict[str, str] = {}
    for key, df in proxies.items():
        path = os.path.join(data_dir, f"breadth_proxy_{key}.csv")
        out_df = df.copy()
        out_df.index = pd.to_datetime(out_df.index)
        out_df.index.name = "date"
        out_df.to_csv(path)
        out[key] = path
    return out
