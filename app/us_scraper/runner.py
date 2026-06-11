from __future__ import annotations

import argparse
import logging
import math
import multiprocessing as mp
import os
import tempfile
import traceback
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from app.us_scraper.datasources.base import BaseDataSource
from app.us_scraper.datasources.cboe_vix_source import CBOEVIXSource
from app.us_scraper.datasources.cnn_fear_greed_source import CNNFearGreedSource
from app.us_scraper.datasources.finra_margin_source import FinraMarginDebtSource
from app.us_scraper.datasources.fred_source import FREDSeriesSource
from app.us_scraper.datasources.gdelt_risk_source import GDELTRiskSource
from app.us_scraper.datasources.tushare_pro_source import global_index_source, us_equity_or_etf_source
from app.us_scraper.datasources.yahoo_price_source import global_index_fallback_source, us_equity_or_etf_fallback_source
from app.us_scraper.datasources.vix_term_structure_source import VIXTermStructureSource
from app.us_scraper.datasources.wikipedia_pageviews_source import WikipediaPageviewsRiskSource
from app.us_scraper.db_loader import upsert_csv_to_mysql
from app.us_scraper.factors.breadth_proxy import build_breadth_proxy_a_b, save_breadth_proxy_csvs


# Default daily set is intentionally narrow and trading-oriented.
# Goal: support 1-4 week risk-reduction / cash-raising decisions, not slow macro research.
#
# P0 daily risk preference signals:
#   - QQQ/SPY: growth / technology risk appetite
#   - XLU/SPY and XLP/SPY: defensive rotation
#   - HYG/LQD: credit risk appetite
#   - VIX: volatility pressure
#   - SOXX: configurable theme/sector proxy risk pressure (default proxy source: SOXX ETF)
#   - IEF/TLT: practical rate-pressure proxies fetched through Tushare US ETF data
#
# FRED is intentionally NOT used by the daily/default path. In the user's current
# network environment fred.stlouisfed.org cannot reliably return data, so daily
# risk reduction decisions must not depend on FRED. When exact yields are needed,
# use an offline/manual macro task later; for daily de-risking, IEF/TLT are enough.
DAILY_SOURCES = [
    "spy", "qqq",
    "xlu", "xlp",
    "hyg", "lqd",
    "vix",
    "soxx",
    "ief", "tlt",
]
RESEARCH_SOURCES = [
    "spx", "iwm", "rut", "gld", "ibb", "xbi", "gdx",
    "vix_term", "margin_debt", "fear_greed", "wikipedia_pageviews_risk", "gdelt_risk",
]
FRED_DISABLED_SOURCES = {"ust10", "us10y", "dgs10", "ust30", "us30y", "dgs30", "tips10", "hy_oas", "ig_oas", "fed_balance_sheet"}
REQUIRE_NON_EMPTY_SOURCES: set[str] = set()
DAILY_WINDOW_DAYS = 7
WEB_SCRAPE_SOURCES = {"fear_greed"}


RISK_PREFERENCE_TABLE = "us_risk_preference_daily"
RISK_BACKTEST_DAILY_TABLE = "us_risk_preference_backtest_daily"
RISK_BACKTEST_SUMMARY_TABLE = "us_risk_preference_backtest_summary"
THEME_RISK_TABLE = "us_theme_risk_daily"
THEME_RISK_BACKTEST_DAILY_TABLE = "us_theme_risk_backtest_daily"
THEME_FAILURE_EVENT_TABLE = "us_theme_failure_event_daily"
THEME_EVENT_CAPTURE_DAILY_TABLE = "us_theme_event_capture_daily"
THEME_EVENT_CAPTURE_SUMMARY_TABLE = "us_theme_event_capture_summary"
THEME_PROXY_SOURCE = "soxx"  # default proxy; table/columns stay generic for future themes
RISK_BACKTEST_HORIZONS = (5, 10, 20)
RISK_BACKTEST_THRESHOLDS = (25, 40, 60)
RISK_RAW_TABLES = {
    "spy": "us_spy",
    "qqq": "us_qqq",
    "soxx": "us_soxx",
    "xlu": "us_xlu",
    "xlp": "us_xlp",
    "hyg": "us_hyg",
    "lqd": "us_lqd",
    "vix": "us_vix",
    "fear_greed": "us_fear_greed",
}


def _mysql_table_exists(engine, table_name: str) -> bool:
    with engine.connect() as conn:
        row = conn.execute(text("SHOW TABLES LIKE :table_name"), {"table_name": table_name}).fetchone()
    return row is not None


def _read_us_close_series(engine, table_name: str, value_column: str = "close") -> pd.Series:
    if not _mysql_table_exists(engine, table_name):
        return pd.Series(dtype="float64")
    try:
        df = pd.read_sql(
            text(f"SELECT `date`, `{value_column}` AS value FROM `{table_name}` WHERE `{value_column}` IS NOT NULL"),
            engine,
        )
    except Exception:
        return pd.Series(dtype="float64")
    if df.empty or "date" not in df.columns:
        return pd.Series(dtype="float64")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["date", "value"]).sort_values("date")
    if df.empty:
        return pd.Series(dtype="float64")
    return df.drop_duplicates(subset=["date"], keep="last").set_index("date")["value"].astype(float)


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    frame = pd.concat([numerator, denominator], axis=1, join="inner")
    if frame.empty:
        return pd.Series(dtype="float64")
    frame = frame.replace([float("inf"), float("-inf")], pd.NA).dropna()
    if frame.empty:
        return pd.Series(dtype="float64")
    den = frame.iloc[:, 1].replace(0, pd.NA)
    out = frame.iloc[:, 0] / den
    return pd.to_numeric(out, errors="coerce").dropna()


def _pct_change(series: pd.Series, periods: int) -> pd.Series:
    # Explicit fill_method=None keeps pandas 3.x behavior stable and avoids
    # silently forward-filling gaps before measuring risk-preference changes.
    return pd.to_numeric(series, errors="coerce").pct_change(periods=periods, fill_method=None)


def _condition_streak(condition: pd.Series) -> pd.Series:
    """Consecutive True-count streak. Missing values break the streak.

    UTRBE-style risk detection values persistence more than one-day shock.
    This helper converts a structural condition such as QQQ/SPY below MA20 into
    a continuous deterioration counter.
    """
    cond = condition.fillna(False).astype(bool)
    group_id = (cond != cond.shift()).cumsum()
    streak = cond.groupby(group_id).cumcount() + 1
    return streak.where(cond, 0).astype("float64")


def _streak_component(streak: pd.Series, *, start: float = 3.0, step: float = 1.6, cap: float = 22.0) -> pd.Series:
    raw = (pd.to_numeric(streak, errors="coerce") - start + 1.0) * step
    return raw.clip(lower=0, upper=cap).fillna(0)


def _trend_slope_component(series: pd.Series, ma: pd.Series, *, below: bool, scale: float = 120.0, cap: float = 10.0) -> pd.Series:
    """Small confirmation from distance to MA20; persistence is scored elsewhere."""
    if below:
        raw = ((ma - series) / ma.replace(0, pd.NA)) * scale
    else:
        raw = ((series - ma) / ma.replace(0, pd.NA)) * scale
    return raw.clip(lower=0, upper=cap).fillna(0)


def _risk_component_below_ma(series: pd.Series, ma: pd.Series, scale: float = 100.0, cap: float = 18.0) -> pd.Series:
    raw = ((ma - series) / ma.replace(0, pd.NA)) * scale
    return raw.clip(lower=0, upper=cap).fillna(0)


def _risk_component_above_ma(series: pd.Series, ma: pd.Series, scale: float = 100.0, cap: float = 18.0) -> pd.Series:
    raw = ((series - ma) / ma.replace(0, pd.NA)) * scale
    return raw.clip(lower=0, upper=cap).fillna(0)


def _risk_component_negative_trend(chg: pd.Series, *, scale: float = 450.0, cap: float = 20.0) -> pd.Series:
    # Example: -3% relative-strength deterioration contributes about 13.5 points.
    return ((-pd.to_numeric(chg, errors="coerce")) * scale).clip(lower=0, upper=cap).fillna(0)


def _risk_component_positive_trend(chg: pd.Series, *, scale: float = 450.0, cap: float = 20.0) -> pd.Series:
    # Defensive relative-strength improvement is a risk-off early warning.
    return (pd.to_numeric(chg, errors="coerce") * scale).clip(lower=0, upper=cap).fillna(0)


def _risk_component_vix_lift(chg: pd.Series, *, scale: float = 90.0, cap: float = 22.0) -> pd.Series:
    # VIX rising 15%-25% over 5/10 sessions is meaningful even before a crash.
    return (pd.to_numeric(chg, errors="coerce") * scale).clip(lower=0, upper=cap).fillna(0)


def _mysql_null_safe(value):
    """Convert pandas/numpy missing values to None before PyMySQL executemany.

    MySQL drivers reject float NaN/inf values. Early rolling-window rows in
    us_risk_preference_daily are expected to have NULL trend fields, so sanitize
    every record explicitly instead of relying on DataFrame.where().
    """
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def _mysql_null_safe_records(df: pd.DataFrame) -> list[dict]:
    return [
        {key: _mysql_null_safe(value) for key, value in row.items()}
        for row in df.to_dict(orient="records")
    ]

def _create_risk_preference_table(engine) -> None:
    ddl = f"""
    CREATE TABLE IF NOT EXISTS `{RISK_PREFERENCE_TABLE}` (
        `trade_date` DATE NOT NULL,
        `qqq_spy_ratio` DOUBLE NULL,
        `qqq_spy_ratio_20d` DOUBLE NULL,
        `qqq_spy_ratio_chg_5d` DOUBLE NULL,
        `qqq_spy_ratio_chg_10d` DOUBLE NULL,
        `qqq_spy_below_ma20_streak` DOUBLE NULL,
        `soxx_spy_ratio` DOUBLE NULL,
        `soxx_spy_ratio_20d` DOUBLE NULL,
        `soxx_spy_ratio_chg_5d` DOUBLE NULL,
        `soxx_spy_ratio_chg_10d` DOUBLE NULL,
        `soxx_spy_below_ma20_streak` DOUBLE NULL,
        `xlu_spy_ratio` DOUBLE NULL,
        `xlu_spy_ratio_20d` DOUBLE NULL,
        `xlu_spy_ratio_chg_5d` DOUBLE NULL,
        `xlu_spy_ratio_chg_10d` DOUBLE NULL,
        `xlu_spy_above_ma20_streak` DOUBLE NULL,
        `xlp_spy_ratio` DOUBLE NULL,
        `xlp_spy_ratio_20d` DOUBLE NULL,
        `xlp_spy_ratio_chg_5d` DOUBLE NULL,
        `xlp_spy_ratio_chg_10d` DOUBLE NULL,
        `xlp_spy_above_ma20_streak` DOUBLE NULL,
        `hyg_lqd_ratio` DOUBLE NULL,
        `hyg_lqd_ratio_20d` DOUBLE NULL,
        `hyg_lqd_ratio_chg_5d` DOUBLE NULL,
        `hyg_lqd_ratio_chg_10d` DOUBLE NULL,
        `hyg_lqd_below_ma20_streak` DOUBLE NULL,
        `vix_close` DOUBLE NULL,
        `vix_ma20` DOUBLE NULL,
        `vix_chg_5d` DOUBLE NULL,
        `vix_chg_10d` DOUBLE NULL,
        `vix_above_ma20_streak` DOUBLE NULL,
        `fear_greed` DOUBLE NULL,
        `risk_off_score` DOUBLE NULL,
        `risk_regime` VARCHAR(32) NULL,
        `updated_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        PRIMARY KEY (`trade_date`)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """
    extra_cols = {
        "qqq_spy_ratio_chg_5d": "DOUBLE NULL",
        "qqq_spy_ratio_chg_10d": "DOUBLE NULL",
        "qqq_spy_below_ma20_streak": "DOUBLE NULL",
        "soxx_spy_ratio_chg_5d": "DOUBLE NULL",
        "soxx_spy_ratio_chg_10d": "DOUBLE NULL",
        "soxx_spy_below_ma20_streak": "DOUBLE NULL",
        "xlu_spy_ratio_chg_5d": "DOUBLE NULL",
        "xlu_spy_ratio_chg_10d": "DOUBLE NULL",
        "xlu_spy_above_ma20_streak": "DOUBLE NULL",
        "xlp_spy_ratio_chg_5d": "DOUBLE NULL",
        "xlp_spy_ratio_chg_10d": "DOUBLE NULL",
        "xlp_spy_above_ma20_streak": "DOUBLE NULL",
        "hyg_lqd_ratio_chg_5d": "DOUBLE NULL",
        "hyg_lqd_ratio_chg_10d": "DOUBLE NULL",
        "hyg_lqd_below_ma20_streak": "DOUBLE NULL",
        "vix_chg_5d": "DOUBLE NULL",
        "vix_chg_10d": "DOUBLE NULL",
        "vix_above_ma20_streak": "DOUBLE NULL",
    }
    with engine.begin() as conn:
        conn.execute(text(ddl))
        existing = {row[0] for row in conn.execute(text(f"SHOW COLUMNS FROM `{RISK_PREFERENCE_TABLE}`"))}
        for col, col_type in extra_cols.items():
            if col not in existing:
                conn.execute(text(f"ALTER TABLE `{RISK_PREFERENCE_TABLE}` ADD COLUMN `{col}` {col_type}"))


def _regime_from_score(score: float | int | None) -> str:
    if score is None or pd.isna(score):
        return "UNKNOWN"
    score = float(score)
    if score >= 75:
        return "PANIC_RISK_OFF"
    if score >= 60:
        return "RISK_OFF_WARNING"
    if score >= 40:
        return "EARLY_RISK_OFF"
    if score >= 25:
        return "CAUTION"
    return "RISK_ON"


def update_us_risk_preference_daily(engine, *, log: logging.Logger | None = None) -> int:
    """Build the US/global risk preference feature table from raw us_* tables.

    This belongs to AshareScraper rather than GrowthAlpha because it is a
    repeatable data/feature engineering layer. GrowthAlpha should consume the
    resulting table directly and focus on interpretation and position guidance.
    """
    logger = log or logging.getLogger("AshareUSScraper")
    spy = _read_us_close_series(engine, RISK_RAW_TABLES["spy"])
    qqq = _read_us_close_series(engine, RISK_RAW_TABLES["qqq"])
    soxx = _read_us_close_series(engine, RISK_RAW_TABLES["soxx"])
    xlu = _read_us_close_series(engine, RISK_RAW_TABLES["xlu"])
    xlp = _read_us_close_series(engine, RISK_RAW_TABLES["xlp"])
    hyg = _read_us_close_series(engine, RISK_RAW_TABLES["hyg"])
    lqd = _read_us_close_series(engine, RISK_RAW_TABLES["lqd"])
    vix = _read_us_close_series(engine, RISK_RAW_TABLES["vix"])
    fear_greed = _read_us_close_series(engine, RISK_RAW_TABLES["fear_greed"], value_column="index_value")

    qqq_spy = _safe_ratio(qqq, spy)
    soxx_spy = _safe_ratio(soxx, spy)
    xlu_spy = _safe_ratio(xlu, spy)
    xlp_spy = _safe_ratio(xlp, spy)
    hyg_lqd = _safe_ratio(hyg, lqd)

    all_series = {
        "qqq_spy_ratio": qqq_spy,
        "soxx_spy_ratio": soxx_spy,
        "xlu_spy_ratio": xlu_spy,
        "xlp_spy_ratio": xlp_spy,
        "hyg_lqd_ratio": hyg_lqd,
        "vix_close": vix,
        "fear_greed": fear_greed,
    }
    base_index = sorted(set().union(*(set(s.index) for s in all_series.values() if not s.empty)))
    if not base_index:
        logger.warning("[US-RISK] skipped: no raw US/global series available")
        return 0

    out = pd.DataFrame(index=pd.DatetimeIndex(base_index, name="trade_date"))
    for col, ser in all_series.items():
        if not ser.empty:
            out[col] = ser.reindex(out.index)

    ratio_cols = ("qqq_spy_ratio", "soxx_spy_ratio", "xlu_spy_ratio", "xlp_spy_ratio", "hyg_lqd_ratio")
    for col in ratio_cols:
        out[f"{col}_20d"] = out[col].rolling(20, min_periods=5).mean()
        out[f"{col}_chg_5d"] = _pct_change(out[col], 5)
        out[f"{col}_chg_10d"] = _pct_change(out[col], 10)
    out["vix_ma20"] = out["vix_close"].rolling(20, min_periods=5).mean()
    out["vix_chg_5d"] = _pct_change(out["vix_close"], 5)
    out["vix_chg_10d"] = _pct_change(out["vix_close"], 10)

    # UTRBE-style persistence features. These are the core of V2.
    # A single crash day should not dominate the score; a 1-2 week structural
    # breakdown should.
    out["qqq_spy_below_ma20_streak"] = _condition_streak(out["qqq_spy_ratio"] < out["qqq_spy_ratio_20d"])
    out["soxx_spy_below_ma20_streak"] = _condition_streak(out["soxx_spy_ratio"] < out["soxx_spy_ratio_20d"])
    out["xlu_spy_above_ma20_streak"] = _condition_streak(out["xlu_spy_ratio"] > out["xlu_spy_ratio_20d"])
    out["xlp_spy_above_ma20_streak"] = _condition_streak(out["xlp_spy_ratio"] > out["xlp_spy_ratio_20d"])
    out["hyg_lqd_below_ma20_streak"] = _condition_streak(out["hyg_lqd_ratio"] < out["hyg_lqd_ratio_20d"])
    out["vix_above_ma20_streak"] = _condition_streak(out["vix_close"] > out["vix_ma20"])

    # Early-warning score: UTRBE.V30-style structural persistence engine.
    # Core idea: persistent relative weakness/defensive leadership is an early
    # warning; VIX panic and one-day selloffs are confirmation only.
    score = pd.Series(0.0, index=out.index)

    # 1) Persistent loss of growth/theme proxy leadership versus SPY.
    score += _streak_component(out["qqq_spy_below_ma20_streak"], start=3, step=1.8, cap=24)
    score += _streak_component(out["soxx_spy_below_ma20_streak"], start=3, step=1.9, cap=26)

    # 2) Persistent defensive rotation. XLU/XLP leading SPY is an early risk-off
    # tell, especially when it persists several sessions.
    score += _streak_component(out["xlu_spy_above_ma20_streak"], start=3, step=1.5, cap=20)
    score += _streak_component(out["xlp_spy_above_ma20_streak"], start=3, step=1.2, cap=16)

    # 3) Credit risk appetite deterioration. HYG/LQD weakness has high weight
    # because it is closer to institutional de-risking than sentiment surveys.
    score += _streak_component(out["hyg_lqd_below_ma20_streak"], start=3, step=2.0, cap=26)

    # 4) Trend direction confirmation. Kept secondary to persistence.
    score += _risk_component_negative_trend(out["qqq_spy_ratio_chg_10d"], scale=360, cap=10)
    score += _risk_component_negative_trend(out["soxx_spy_ratio_chg_10d"], scale=320, cap=10)
    score += _risk_component_positive_trend(out["xlu_spy_ratio_chg_10d"], scale=360, cap=9)
    score += _risk_component_positive_trend(out["xlp_spy_ratio_chg_10d"], scale=300, cap=7)
    score += _risk_component_negative_trend(out["hyg_lqd_ratio_chg_10d"], scale=650, cap=12)

    # 5) Current distance from MA20: small confirmation only.
    score += _trend_slope_component(out["qqq_spy_ratio"], out["qqq_spy_ratio_20d"], below=True, cap=7)
    score += _trend_slope_component(out["soxx_spy_ratio"], out["soxx_spy_ratio_20d"], below=True, cap=7)
    score += _trend_slope_component(out["xlu_spy_ratio"], out["xlu_spy_ratio_20d"], below=False, cap=6)
    score += _trend_slope_component(out["xlp_spy_ratio"], out["xlp_spy_ratio_20d"], below=False, cap=5)
    score += _trend_slope_component(out["hyg_lqd_ratio"], out["hyg_lqd_ratio_20d"], below=True, cap=7)

    # 6) VIX is confirmation, not the engine. This avoids classifying only
    # post-crash panic days as warnings.
    score += _streak_component(out["vix_above_ma20_streak"], start=2, step=0.8, cap=8)
    score += _risk_component_vix_lift(out["vix_chg_10d"], scale=35, cap=8)

    if "fear_greed" in out.columns:
        fg_component = ((55.0 - pd.to_numeric(out["fear_greed"], errors="coerce")) / 1.7).clip(lower=0, upper=18).fillna(0)
        score += fg_component
    out["risk_off_score"] = score.clip(lower=0, upper=100).round(4)
    out["risk_regime"] = out["risk_off_score"].map(_regime_from_score)

    wanted = [
        "qqq_spy_ratio", "qqq_spy_ratio_20d", "qqq_spy_ratio_chg_5d", "qqq_spy_ratio_chg_10d", "qqq_spy_below_ma20_streak",
        "soxx_spy_ratio", "soxx_spy_ratio_20d", "soxx_spy_ratio_chg_5d", "soxx_spy_ratio_chg_10d", "soxx_spy_below_ma20_streak",
        "xlu_spy_ratio", "xlu_spy_ratio_20d", "xlu_spy_ratio_chg_5d", "xlu_spy_ratio_chg_10d", "xlu_spy_above_ma20_streak",
        "xlp_spy_ratio", "xlp_spy_ratio_20d", "xlp_spy_ratio_chg_5d", "xlp_spy_ratio_chg_10d", "xlp_spy_above_ma20_streak",
        "hyg_lqd_ratio", "hyg_lqd_ratio_20d", "hyg_lqd_ratio_chg_5d", "hyg_lqd_ratio_chg_10d", "hyg_lqd_below_ma20_streak",
        "vix_close", "vix_ma20", "vix_chg_5d", "vix_chg_10d", "vix_above_ma20_streak",
        "fear_greed", "risk_off_score", "risk_regime",
    ]
    for col in wanted:
        if col not in out.columns:
            out[col] = pd.NA
    out = out[wanted].reset_index()
    out["trade_date"] = pd.to_datetime(out["trade_date"]).dt.date
    out = out.replace([float("inf"), float("-inf")], pd.NA)

    _create_risk_preference_table(engine)
    # Keep the insert list generated from `wanted` so new UTRBE persistence
    # columns cannot be forgotten in future patches.
    insert_cols = ["trade_date", *wanted]
    column_sql = ", ".join(f"`{col}`" for col in insert_cols)
    value_sql = ", ".join(f":{col}" for col in insert_cols)
    update_sql = ",\n            ".join(
        f"`{col}`=VALUES(`{col}`)" for col in wanted
    )
    insert_sql = text(f"""
        INSERT INTO `{RISK_PREFERENCE_TABLE}` (
            {column_sql}
        ) VALUES (
            {value_sql}
        ) ON DUPLICATE KEY UPDATE
            {update_sql}
    """)
    records = _mysql_null_safe_records(out)
    if not records:
        return 0
    with engine.begin() as conn:
        conn.execute(insert_sql, records)
    logger.info("[US-RISK] table=%s rows=%s latest=%s", RISK_PREFERENCE_TABLE, len(records), out["trade_date"].max())
    return len(records)



def _forward_return(series: pd.Series, horizon: int) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").sort_index()
    denom = s.mask(s == 0)
    return (s.shift(-int(horizon)) / denom) - 1.0


def _forward_max_drawdown(series: pd.Series, horizon: int) -> pd.Series:
    """Worst close-to-close drawdown within the next N observations.

    Uses trading observations rather than calendar days. A value of -0.06 means
    the instrument traded at least 6% below the signal day's close during the
    forward window. Pandas NA scalars are intentionally avoided here because
    pandas/numpy float arrays cannot cast pd.NA reliably under Python 3.14.
    """
    s = pd.to_numeric(series, errors="coerce").sort_index()
    values: list[float] = []
    idx = list(s.index)
    for i, _dt in enumerate(idx):
        base = s.iloc[i]
        if pd.isna(base) or float(base) == 0.0:
            values.append(float("nan"))
            continue
        future = s.iloc[i + 1 : i + 1 + int(horizon)].dropna()
        if future.empty:
            values.append(float("nan"))
        else:
            values.append(float(future.min() / float(base) - 1.0))
    return pd.Series(values, index=s.index, dtype="float64")


def _create_risk_backtest_tables(engine) -> None:
    """Create backtest tables.

    Important: the detail table is intentionally one row per trade_date.
    Older patch versions created one row per (date, horizon, threshold), which
    inflated ad-hoc SQL grouped by risk_regime and made validation misleading.
    Because these are derived validation tables, an incompatible old detail
    table is dropped and rebuilt automatically.
    """
    detail_ddl = f"""
    CREATE TABLE IF NOT EXISTS `{RISK_BACKTEST_DAILY_TABLE}` (
        `trade_date` DATE NOT NULL,
        `risk_off_score` DOUBLE NULL,
        `risk_regime` VARCHAR(32) NULL,
        `future_5d_spy_return` DOUBLE NULL,
        `future_5d_qqq_return` DOUBLE NULL,
        `future_5d_soxx_return` DOUBLE NULL,
        `future_5d_qqq_vs_spy_return` DOUBLE NULL,
        `future_5d_soxx_vs_spy_return` DOUBLE NULL,
        `future_5d_spy_max_drawdown` DOUBLE NULL,
        `future_5d_qqq_max_drawdown` DOUBLE NULL,
        `future_5d_soxx_max_drawdown` DOUBLE NULL,
        `event_5d_flag` TINYINT NULL,
        `future_10d_spy_return` DOUBLE NULL,
        `future_10d_qqq_return` DOUBLE NULL,
        `future_10d_soxx_return` DOUBLE NULL,
        `future_10d_qqq_vs_spy_return` DOUBLE NULL,
        `future_10d_soxx_vs_spy_return` DOUBLE NULL,
        `future_10d_spy_max_drawdown` DOUBLE NULL,
        `future_10d_qqq_max_drawdown` DOUBLE NULL,
        `future_10d_soxx_max_drawdown` DOUBLE NULL,
        `event_10d_flag` TINYINT NULL,
        `future_20d_spy_return` DOUBLE NULL,
        `future_20d_qqq_return` DOUBLE NULL,
        `future_20d_soxx_return` DOUBLE NULL,
        `future_20d_qqq_vs_spy_return` DOUBLE NULL,
        `future_20d_soxx_vs_spy_return` DOUBLE NULL,
        `future_20d_spy_max_drawdown` DOUBLE NULL,
        `future_20d_qqq_max_drawdown` DOUBLE NULL,
        `future_20d_soxx_max_drawdown` DOUBLE NULL,
        `event_20d_flag` TINYINT NULL,
        `spy_max_drawdown_5d` DOUBLE NULL,
        `qqq_max_drawdown_5d` DOUBLE NULL,
        `soxx_max_drawdown_5d` DOUBLE NULL,
        `spy_max_drawdown_10d` DOUBLE NULL,
        `qqq_max_drawdown_10d` DOUBLE NULL,
        `soxx_max_drawdown_10d` DOUBLE NULL,
        `spy_max_drawdown_20d` DOUBLE NULL,
        `qqq_max_drawdown_20d` DOUBLE NULL,
        `soxx_max_drawdown_20d` DOUBLE NULL,
        `updated_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        PRIMARY KEY (`trade_date`),
        KEY `idx_us_risk_bt_regime` (`risk_regime`),
        KEY `idx_us_risk_bt_score` (`risk_off_score`)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """
    summary_ddl = f"""
    CREATE TABLE IF NOT EXISTS `{RISK_BACKTEST_SUMMARY_TABLE}` (
        `horizon_days` INT NOT NULL,
        `warning_threshold` DOUBLE NOT NULL,
        `sample_count` INT NULL,
        `warning_count` INT NULL,
        `event_count` INT NULL,
        `true_positive_count` INT NULL,
        `false_positive_count` INT NULL,
        `false_negative_count` INT NULL,
        `precision_rate` DOUBLE NULL,
        `recall_rate` DOUBLE NULL,
        `false_positive_rate` DOUBLE NULL,
        `avg_spy_forward_return_when_warning` DOUBLE NULL,
        `avg_qqq_forward_return_when_warning` DOUBLE NULL,
        `avg_soxx_forward_return_when_warning` DOUBLE NULL,
        `avg_qqq_vs_spy_when_warning` DOUBLE NULL,
        `avg_soxx_vs_spy_when_warning` DOUBLE NULL,
        `avg_spy_max_drawdown_when_warning` DOUBLE NULL,
        `avg_qqq_max_drawdown_when_warning` DOUBLE NULL,
        `avg_soxx_max_drawdown_when_warning` DOUBLE NULL,
        `verdict` VARCHAR(64) NULL,
        `updated_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        PRIMARY KEY (`horizon_days`, `warning_threshold`)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """
    with engine.begin() as conn:
        if _mysql_table_exists(engine, RISK_BACKTEST_DAILY_TABLE):
            cols = {row[0] for row in conn.execute(text(f"SHOW COLUMNS FROM `{RISK_BACKTEST_DAILY_TABLE}`"))}
            # Old incompatible schema: one row per horizon/threshold.
            if "horizon_days" in cols or "warning_threshold" in cols:
                conn.execute(text(f"DROP TABLE `{RISK_BACKTEST_DAILY_TABLE}`"))
        conn.execute(text(detail_ddl))
        cols = {row[0] for row in conn.execute(text(f"SHOW COLUMNS FROM `{RISK_BACKTEST_DAILY_TABLE}`"))}
        alias_columns = {
            "spy_max_drawdown_5d": "DOUBLE NULL",
            "qqq_max_drawdown_5d": "DOUBLE NULL",
            "soxx_max_drawdown_5d": "DOUBLE NULL",
            "spy_max_drawdown_10d": "DOUBLE NULL",
            "qqq_max_drawdown_10d": "DOUBLE NULL",
            "soxx_max_drawdown_10d": "DOUBLE NULL",
            "spy_max_drawdown_20d": "DOUBLE NULL",
            "qqq_max_drawdown_20d": "DOUBLE NULL",
            "soxx_max_drawdown_20d": "DOUBLE NULL",
        }
        for col_name, col_def in alias_columns.items():
            if col_name not in cols:
                conn.execute(text(f"ALTER TABLE `{RISK_BACKTEST_DAILY_TABLE}` ADD COLUMN `{col_name}` {col_def}"))
        conn.execute(text(summary_ddl))

def _verdict_from_summary(precision: float | None, recall: float | None, warning_count: int) -> str:
    if warning_count <= 0:
        return "NO_WARNINGS"
    if precision is None or recall is None or pd.isna(precision) or pd.isna(recall):
        return "INSUFFICIENT"
    if precision >= 0.55 and recall >= 0.35:
        return "USEFUL_WARNING"
    if precision >= 0.45 and recall >= 0.25:
        return "WATCHLIST_SIGNAL"
    return "WEAK_OR_NOISY"


def update_us_risk_preference_backtest(engine, *, log: logging.Logger | None = None) -> dict[str, int]:
    """Backtest whether risk_off_score warns before future drawdown/underperformance.

    Validation detail is one row per trade_date. Summary remains grouped by
    horizon/threshold. This avoids duplicated averages when users run simple
    SQL like AVG(future_20d_qqq_return) GROUP BY risk_regime.
    """
    logger = log or logging.getLogger("AshareUSScraper")
    if not _mysql_table_exists(engine, RISK_PREFERENCE_TABLE):
        logger.warning("[US-RISK-BT] skipped: %s not found", RISK_PREFERENCE_TABLE)
        return {"detail_rows": 0, "summary_rows": 0}

    risk = pd.read_sql(
        text(f"SELECT `trade_date`, `risk_off_score`, `risk_regime` FROM `{RISK_PREFERENCE_TABLE}` WHERE `risk_off_score` IS NOT NULL"),
        engine,
    )
    if risk.empty:
        logger.warning("[US-RISK-BT] skipped: no risk_off_score rows")
        return {"detail_rows": 0, "summary_rows": 0}
    risk["trade_date"] = pd.to_datetime(risk["trade_date"], errors="coerce")
    risk["risk_off_score"] = pd.to_numeric(risk["risk_off_score"], errors="coerce")
    risk = risk.dropna(subset=["trade_date", "risk_off_score"]).drop_duplicates(subset=["trade_date"], keep="last").set_index("trade_date").sort_index()

    spy = _read_us_close_series(engine, "us_spy")
    qqq = _read_us_close_series(engine, "us_qqq")
    soxx = _read_us_close_series(engine, "us_soxx")
    if spy.empty or qqq.empty or soxx.empty:
        logger.warning("[US-RISK-BT] skipped: missing us_spy/us_qqq/us_soxx close series")
        return {"detail_rows": 0, "summary_rows": 0}

    idx = risk.index.intersection(spy.index).intersection(qqq.index).intersection(soxx.index)
    logger.info("[US-RISK-BT] align rows=%s risk=%s spy=%s qqq=%s soxx=%s", len(idx), len(risk), len(spy), len(qqq), len(soxx))
    if len(idx) < 120:
        logger.warning("[US-RISK-BT] skipped: insufficient aligned rows=%s", len(idx))
        return {"detail_rows": 0, "summary_rows": 0}
    risk = risk.reindex(idx)
    spy = spy.reindex(idx)
    qqq = qqq.reindex(idx)
    soxx = soxx.reindex(idx)

    detail = pd.DataFrame(index=idx)
    detail["risk_off_score"] = risk["risk_off_score"]
    detail["risk_regime"] = risk["risk_regime"]
    summary_records: list[dict] = []

    for horizon in RISK_BACKTEST_HORIZONS:
        h = int(horizon)
        prefix = f"future_{h}d"
        detail[f"{prefix}_spy_return"] = _forward_return(spy, h)
        detail[f"{prefix}_qqq_return"] = _forward_return(qqq, h)
        detail[f"{prefix}_soxx_return"] = _forward_return(soxx, h)
        detail[f"{prefix}_qqq_vs_spy_return"] = detail[f"{prefix}_qqq_return"] - detail[f"{prefix}_spy_return"]
        detail[f"{prefix}_soxx_vs_spy_return"] = detail[f"{prefix}_soxx_return"] - detail[f"{prefix}_spy_return"]
        detail[f"{prefix}_spy_max_drawdown"] = _forward_max_drawdown(spy, h)
        detail[f"{prefix}_qqq_max_drawdown"] = _forward_max_drawdown(qqq, h)
        detail[f"{prefix}_soxx_max_drawdown"] = _forward_max_drawdown(soxx, h)
        # Convenience aliases for validation SQL, e.g. AVG(qqq_max_drawdown_20d).
        detail[f"spy_max_drawdown_{h}d"] = detail[f"{prefix}_spy_max_drawdown"]
        detail[f"qqq_max_drawdown_{h}d"] = detail[f"{prefix}_qqq_max_drawdown"]
        detail[f"soxx_max_drawdown_{h}d"] = detail[f"{prefix}_soxx_max_drawdown"]
        event_col = f"event_{h}d_flag"
        detail[event_col] = (
            (detail[f"{prefix}_spy_max_drawdown"] <= -0.035)
            | (detail[f"{prefix}_qqq_max_drawdown"] <= -0.055)
            | (detail[f"{prefix}_soxx_max_drawdown"] <= -0.080)
            | (detail[f"{prefix}_qqq_vs_spy_return"] <= -0.025)
            | (detail[f"{prefix}_soxx_vs_spy_return"] <= -0.050)
        ).astype("Int64")

        base = pd.DataFrame({
            "risk_off_score": detail["risk_off_score"],
            "spy_forward_return": detail[f"{prefix}_spy_return"],
            "qqq_forward_return": detail[f"{prefix}_qqq_return"],
            "soxx_forward_return": detail[f"{prefix}_soxx_return"],
            "qqq_vs_spy_forward_return": detail[f"{prefix}_qqq_vs_spy_return"],
            "soxx_vs_spy_forward_return": detail[f"{prefix}_soxx_vs_spy_return"],
            "spy_max_drawdown": detail[f"{prefix}_spy_max_drawdown"],
            "qqq_max_drawdown": detail[f"{prefix}_qqq_max_drawdown"],
            "soxx_max_drawdown": detail[f"{prefix}_soxx_max_drawdown"],
            "event_flag": detail[event_col],
        }).dropna(subset=["spy_forward_return", "qqq_forward_return", "soxx_forward_return"])

        for threshold in RISK_BACKTEST_THRESHOLDS:
            warning = (base["risk_off_score"] >= float(threshold)).astype("Int64")
            event = base["event_flag"].astype("Int64")
            true_positive = ((warning == 1) & (event == 1)).astype("Int64")
            false_positive = ((warning == 1) & (event == 0)).astype("Int64")
            sample_count = int(len(base))
            warning_count = int(warning.sum())
            event_count = int(event.sum())
            true_positive_count = int(true_positive.sum())
            false_positive_count = int(false_positive.sum())
            false_negative_count = max(0, event_count - true_positive_count)
            precision = (true_positive_count / warning_count) if warning_count else None
            recall = (true_positive_count / event_count) if event_count else None
            non_event_count = max(0, sample_count - event_count)
            false_positive_rate = (false_positive_count / non_event_count) if non_event_count else None
            warned = base[warning == 1]
            summary_records.append({
                "horizon_days": h,
                "warning_threshold": float(threshold),
                "sample_count": sample_count,
                "warning_count": warning_count,
                "event_count": event_count,
                "true_positive_count": true_positive_count,
                "false_positive_count": false_positive_count,
                "false_negative_count": false_negative_count,
                "precision_rate": precision,
                "recall_rate": recall,
                "false_positive_rate": false_positive_rate,
                "avg_spy_forward_return_when_warning": warned["spy_forward_return"].mean() if not warned.empty else None,
                "avg_qqq_forward_return_when_warning": warned["qqq_forward_return"].mean() if not warned.empty else None,
                "avg_soxx_forward_return_when_warning": warned["soxx_forward_return"].mean() if not warned.empty else None,
                "avg_qqq_vs_spy_when_warning": warned["qqq_vs_spy_forward_return"].mean() if not warned.empty else None,
                "avg_soxx_vs_spy_when_warning": warned["soxx_vs_spy_forward_return"].mean() if not warned.empty else None,
                "avg_spy_max_drawdown_when_warning": warned["spy_max_drawdown"].mean() if not warned.empty else None,
                "avg_qqq_max_drawdown_when_warning": warned["qqq_max_drawdown"].mean() if not warned.empty else None,
                "avg_soxx_max_drawdown_when_warning": warned["soxx_max_drawdown"].mean() if not warned.empty else None,
                "verdict": _verdict_from_summary(precision, recall, warning_count),
            })

    detail = detail.reset_index().rename(columns={"index": "trade_date"})
    detail["trade_date"] = pd.to_datetime(detail["trade_date"]).dt.date
    detail_cols = [
        "trade_date", "risk_off_score", "risk_regime",
        "future_5d_spy_return", "future_5d_qqq_return", "future_5d_soxx_return",
        "future_5d_qqq_vs_spy_return", "future_5d_soxx_vs_spy_return",
        "future_5d_spy_max_drawdown", "future_5d_qqq_max_drawdown", "future_5d_soxx_max_drawdown", "event_5d_flag",
        "future_10d_spy_return", "future_10d_qqq_return", "future_10d_soxx_return",
        "future_10d_qqq_vs_spy_return", "future_10d_soxx_vs_spy_return",
        "future_10d_spy_max_drawdown", "future_10d_qqq_max_drawdown", "future_10d_soxx_max_drawdown", "event_10d_flag",
        "future_20d_spy_return", "future_20d_qqq_return", "future_20d_soxx_return",
        "future_20d_qqq_vs_spy_return", "future_20d_soxx_vs_spy_return",
        "future_20d_spy_max_drawdown", "future_20d_qqq_max_drawdown", "future_20d_soxx_max_drawdown", "event_20d_flag",
        "spy_max_drawdown_5d", "qqq_max_drawdown_5d", "soxx_max_drawdown_5d",
        "spy_max_drawdown_10d", "qqq_max_drawdown_10d", "soxx_max_drawdown_10d",
        "spy_max_drawdown_20d", "qqq_max_drawdown_20d", "soxx_max_drawdown_20d",
    ]
    detail = detail[detail_cols]
    summary = pd.DataFrame(summary_records)

    _create_risk_backtest_tables(engine)
    detail_sql = text(f"""
        INSERT INTO `{RISK_BACKTEST_DAILY_TABLE}` (
            `trade_date`, `risk_off_score`, `risk_regime`,
            `future_5d_spy_return`, `future_5d_qqq_return`, `future_5d_soxx_return`,
            `future_5d_qqq_vs_spy_return`, `future_5d_soxx_vs_spy_return`,
            `future_5d_spy_max_drawdown`, `future_5d_qqq_max_drawdown`, `future_5d_soxx_max_drawdown`, `event_5d_flag`,
            `future_10d_spy_return`, `future_10d_qqq_return`, `future_10d_soxx_return`,
            `future_10d_qqq_vs_spy_return`, `future_10d_soxx_vs_spy_return`,
            `future_10d_spy_max_drawdown`, `future_10d_qqq_max_drawdown`, `future_10d_soxx_max_drawdown`, `event_10d_flag`,
            `future_20d_spy_return`, `future_20d_qqq_return`, `future_20d_soxx_return`,
            `future_20d_qqq_vs_spy_return`, `future_20d_soxx_vs_spy_return`,
            `future_20d_spy_max_drawdown`, `future_20d_qqq_max_drawdown`, `future_20d_soxx_max_drawdown`, `event_20d_flag`,
            `spy_max_drawdown_5d`, `qqq_max_drawdown_5d`, `soxx_max_drawdown_5d`,
            `spy_max_drawdown_10d`, `qqq_max_drawdown_10d`, `soxx_max_drawdown_10d`,
            `spy_max_drawdown_20d`, `qqq_max_drawdown_20d`, `soxx_max_drawdown_20d`
        ) VALUES (
            :trade_date, :risk_off_score, :risk_regime,
            :future_5d_spy_return, :future_5d_qqq_return, :future_5d_soxx_return,
            :future_5d_qqq_vs_spy_return, :future_5d_soxx_vs_spy_return,
            :future_5d_spy_max_drawdown, :future_5d_qqq_max_drawdown, :future_5d_soxx_max_drawdown, :event_5d_flag,
            :future_10d_spy_return, :future_10d_qqq_return, :future_10d_soxx_return,
            :future_10d_qqq_vs_spy_return, :future_10d_soxx_vs_spy_return,
            :future_10d_spy_max_drawdown, :future_10d_qqq_max_drawdown, :future_10d_soxx_max_drawdown, :event_10d_flag,
            :future_20d_spy_return, :future_20d_qqq_return, :future_20d_soxx_return,
            :future_20d_qqq_vs_spy_return, :future_20d_soxx_vs_spy_return,
            :future_20d_spy_max_drawdown, :future_20d_qqq_max_drawdown, :future_20d_soxx_max_drawdown, :event_20d_flag,
            :spy_max_drawdown_5d, :qqq_max_drawdown_5d, :soxx_max_drawdown_5d,
            :spy_max_drawdown_10d, :qqq_max_drawdown_10d, :soxx_max_drawdown_10d,
            :spy_max_drawdown_20d, :qqq_max_drawdown_20d, :soxx_max_drawdown_20d
        ) ON DUPLICATE KEY UPDATE
            `risk_off_score`=VALUES(`risk_off_score`),
            `risk_regime`=VALUES(`risk_regime`),
            `future_5d_spy_return`=VALUES(`future_5d_spy_return`),
            `future_5d_qqq_return`=VALUES(`future_5d_qqq_return`),
            `future_5d_soxx_return`=VALUES(`future_5d_soxx_return`),
            `future_5d_qqq_vs_spy_return`=VALUES(`future_5d_qqq_vs_spy_return`),
            `future_5d_soxx_vs_spy_return`=VALUES(`future_5d_soxx_vs_spy_return`),
            `future_5d_spy_max_drawdown`=VALUES(`future_5d_spy_max_drawdown`),
            `future_5d_qqq_max_drawdown`=VALUES(`future_5d_qqq_max_drawdown`),
            `future_5d_soxx_max_drawdown`=VALUES(`future_5d_soxx_max_drawdown`),
            `event_5d_flag`=VALUES(`event_5d_flag`),
            `future_10d_spy_return`=VALUES(`future_10d_spy_return`),
            `future_10d_qqq_return`=VALUES(`future_10d_qqq_return`),
            `future_10d_soxx_return`=VALUES(`future_10d_soxx_return`),
            `future_10d_qqq_vs_spy_return`=VALUES(`future_10d_qqq_vs_spy_return`),
            `future_10d_soxx_vs_spy_return`=VALUES(`future_10d_soxx_vs_spy_return`),
            `future_10d_spy_max_drawdown`=VALUES(`future_10d_spy_max_drawdown`),
            `future_10d_qqq_max_drawdown`=VALUES(`future_10d_qqq_max_drawdown`),
            `future_10d_soxx_max_drawdown`=VALUES(`future_10d_soxx_max_drawdown`),
            `event_10d_flag`=VALUES(`event_10d_flag`),
            `future_20d_spy_return`=VALUES(`future_20d_spy_return`),
            `future_20d_qqq_return`=VALUES(`future_20d_qqq_return`),
            `future_20d_soxx_return`=VALUES(`future_20d_soxx_return`),
            `future_20d_qqq_vs_spy_return`=VALUES(`future_20d_qqq_vs_spy_return`),
            `future_20d_soxx_vs_spy_return`=VALUES(`future_20d_soxx_vs_spy_return`),
            `future_20d_spy_max_drawdown`=VALUES(`future_20d_spy_max_drawdown`),
            `future_20d_qqq_max_drawdown`=VALUES(`future_20d_qqq_max_drawdown`),
            `future_20d_soxx_max_drawdown`=VALUES(`future_20d_soxx_max_drawdown`),
            `event_20d_flag`=VALUES(`event_20d_flag`),
            `spy_max_drawdown_5d`=VALUES(`spy_max_drawdown_5d`),
            `qqq_max_drawdown_5d`=VALUES(`qqq_max_drawdown_5d`),
            `soxx_max_drawdown_5d`=VALUES(`soxx_max_drawdown_5d`),
            `spy_max_drawdown_10d`=VALUES(`spy_max_drawdown_10d`),
            `qqq_max_drawdown_10d`=VALUES(`qqq_max_drawdown_10d`),
            `soxx_max_drawdown_10d`=VALUES(`soxx_max_drawdown_10d`),
            `spy_max_drawdown_20d`=VALUES(`spy_max_drawdown_20d`),
            `qqq_max_drawdown_20d`=VALUES(`qqq_max_drawdown_20d`),
            `soxx_max_drawdown_20d`=VALUES(`soxx_max_drawdown_20d`)
    """)
    summary_sql = text(f"""
        INSERT INTO `{RISK_BACKTEST_SUMMARY_TABLE}` (
            `horizon_days`, `warning_threshold`, `sample_count`, `warning_count`, `event_count`,
            `true_positive_count`, `false_positive_count`, `false_negative_count`,
            `precision_rate`, `recall_rate`, `false_positive_rate`,
            `avg_spy_forward_return_when_warning`, `avg_qqq_forward_return_when_warning`, `avg_soxx_forward_return_when_warning`,
            `avg_qqq_vs_spy_when_warning`, `avg_soxx_vs_spy_when_warning`,
            `avg_spy_max_drawdown_when_warning`, `avg_qqq_max_drawdown_when_warning`, `avg_soxx_max_drawdown_when_warning`,
            `verdict`
        ) VALUES (
            :horizon_days, :warning_threshold, :sample_count, :warning_count, :event_count,
            :true_positive_count, :false_positive_count, :false_negative_count,
            :precision_rate, :recall_rate, :false_positive_rate,
            :avg_spy_forward_return_when_warning, :avg_qqq_forward_return_when_warning, :avg_soxx_forward_return_when_warning,
            :avg_qqq_vs_spy_when_warning, :avg_soxx_vs_spy_when_warning,
            :avg_spy_max_drawdown_when_warning, :avg_qqq_max_drawdown_when_warning, :avg_soxx_max_drawdown_when_warning,
            :verdict
        ) ON DUPLICATE KEY UPDATE
            `sample_count`=VALUES(`sample_count`),
            `warning_count`=VALUES(`warning_count`),
            `event_count`=VALUES(`event_count`),
            `true_positive_count`=VALUES(`true_positive_count`),
            `false_positive_count`=VALUES(`false_positive_count`),
            `false_negative_count`=VALUES(`false_negative_count`),
            `precision_rate`=VALUES(`precision_rate`),
            `recall_rate`=VALUES(`recall_rate`),
            `false_positive_rate`=VALUES(`false_positive_rate`),
            `avg_spy_forward_return_when_warning`=VALUES(`avg_spy_forward_return_when_warning`),
            `avg_qqq_forward_return_when_warning`=VALUES(`avg_qqq_forward_return_when_warning`),
            `avg_soxx_forward_return_when_warning`=VALUES(`avg_soxx_forward_return_when_warning`),
            `avg_qqq_vs_spy_when_warning`=VALUES(`avg_qqq_vs_spy_when_warning`),
            `avg_soxx_vs_spy_when_warning`=VALUES(`avg_soxx_vs_spy_when_warning`),
            `avg_spy_max_drawdown_when_warning`=VALUES(`avg_spy_max_drawdown_when_warning`),
            `avg_qqq_max_drawdown_when_warning`=VALUES(`avg_qqq_max_drawdown_when_warning`),
            `avg_soxx_max_drawdown_when_warning`=VALUES(`avg_soxx_max_drawdown_when_warning`),
            `verdict`=VALUES(`verdict`)
    """)
    with engine.begin() as conn:
        conn.execute(detail_sql, _mysql_null_safe_records(detail))
        conn.execute(summary_sql, _mysql_null_safe_records(summary))
    logger.info(
        "[US-RISK-BT] detail_table=%s rows=%s summary_table=%s rows=%s",
        RISK_BACKTEST_DAILY_TABLE,
        len(detail),
        RISK_BACKTEST_SUMMARY_TABLE,
        len(summary),
    )
    return {"detail_rows": int(len(detail)), "summary_rows": int(len(summary))}


def _create_theme_risk_table(engine) -> None:
    ddl = f"""
    CREATE TABLE IF NOT EXISTS `{THEME_RISK_TABLE}` (
        `trade_date` DATE NOT NULL,
        `theme_proxy_close` DOUBLE NULL,
        `theme_proxy_ma20` DOUBLE NULL,
        `theme_proxy_chg_5d` DOUBLE NULL,
        `theme_proxy_chg_10d` DOUBLE NULL,
        `theme_proxy_below_ma20_streak` DOUBLE NULL,
        `theme_spy_ratio` DOUBLE NULL,
        `theme_spy_ratio_20d` DOUBLE NULL,
        `theme_spy_ratio_chg_5d` DOUBLE NULL,
        `theme_spy_ratio_chg_10d` DOUBLE NULL,
        `theme_spy_below_ma20_streak` DOUBLE NULL,
        `theme_qqq_ratio` DOUBLE NULL,
        `theme_qqq_ratio_20d` DOUBLE NULL,
        `theme_qqq_ratio_chg_5d` DOUBLE NULL,
        `theme_qqq_ratio_chg_10d` DOUBLE NULL,
        `theme_qqq_below_ma20_streak` DOUBLE NULL,
        `qqq_spy_ratio` DOUBLE NULL,
        `qqq_spy_ratio_20d` DOUBLE NULL,
        `qqq_spy_below_ma20_streak` DOUBLE NULL,
        `hyg_lqd_ratio` DOUBLE NULL,
        `hyg_lqd_ratio_20d` DOUBLE NULL,
        `hyg_lqd_below_ma20_streak` DOUBLE NULL,
        `vix_close` DOUBLE NULL,
        `vix_ma20` DOUBLE NULL,
        `vix_above_ma20_streak` DOUBLE NULL,
        `theme_risk_score` DOUBLE NULL,
        `theme_risk_regime` VARCHAR(32) NULL,
        `updated_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        PRIMARY KEY (`trade_date`)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """
    extra_cols = {
        "theme_proxy_close": "DOUBLE NULL", "theme_proxy_ma20": "DOUBLE NULL", "theme_proxy_chg_5d": "DOUBLE NULL", "theme_proxy_chg_10d": "DOUBLE NULL", "theme_proxy_below_ma20_streak": "DOUBLE NULL",
        "theme_spy_ratio": "DOUBLE NULL", "theme_spy_ratio_20d": "DOUBLE NULL", "theme_spy_ratio_chg_5d": "DOUBLE NULL", "theme_spy_ratio_chg_10d": "DOUBLE NULL", "theme_spy_below_ma20_streak": "DOUBLE NULL",
        "theme_qqq_ratio": "DOUBLE NULL", "theme_qqq_ratio_20d": "DOUBLE NULL", "theme_qqq_ratio_chg_5d": "DOUBLE NULL", "theme_qqq_ratio_chg_10d": "DOUBLE NULL", "theme_qqq_below_ma20_streak": "DOUBLE NULL",
        "qqq_spy_ratio": "DOUBLE NULL", "qqq_spy_ratio_20d": "DOUBLE NULL", "qqq_spy_below_ma20_streak": "DOUBLE NULL",
        "hyg_lqd_ratio": "DOUBLE NULL", "hyg_lqd_ratio_20d": "DOUBLE NULL", "hyg_lqd_below_ma20_streak": "DOUBLE NULL",
        "vix_close": "DOUBLE NULL", "vix_ma20": "DOUBLE NULL", "vix_above_ma20_streak": "DOUBLE NULL",
        "theme_risk_score": "DOUBLE NULL", "theme_risk_regime": "VARCHAR(32) NULL",
    }
    with engine.begin() as conn:
        conn.execute(text(ddl))
        existing = {row[0] for row in conn.execute(text(f"SHOW COLUMNS FROM `{THEME_RISK_TABLE}`"))}
        for col, col_type in extra_cols.items():
            if col not in existing:
                conn.execute(text(f"ALTER TABLE `{THEME_RISK_TABLE}` ADD COLUMN `{col}` {col_type}"))


def update_us_theme_risk_daily(engine, *, log: logging.Logger | None = None) -> int:
    logger = log or logging.getLogger("AshareUSScraper")
    spy = _read_us_close_series(engine, RISK_RAW_TABLES["spy"])
    qqq = _read_us_close_series(engine, RISK_RAW_TABLES["qqq"])
    theme_proxy = _read_us_close_series(engine, RISK_RAW_TABLES[THEME_PROXY_SOURCE])
    hyg = _read_us_close_series(engine, RISK_RAW_TABLES["hyg"])
    lqd = _read_us_close_series(engine, RISK_RAW_TABLES["lqd"])
    vix = _read_us_close_series(engine, RISK_RAW_TABLES["vix"])
    theme_spy = _safe_ratio(theme_proxy, spy)
    theme_qqq = _safe_ratio(theme_proxy, qqq)
    qqq_spy = _safe_ratio(qqq, spy)
    hyg_lqd = _safe_ratio(hyg, lqd)
    all_series = {"theme_proxy_close": theme_proxy, "theme_spy_ratio": theme_spy, "theme_qqq_ratio": theme_qqq, "qqq_spy_ratio": qqq_spy, "hyg_lqd_ratio": hyg_lqd, "vix_close": vix}
    base_index = sorted(set().union(*(set(s.index) for s in all_series.values() if not s.empty)))
    if not base_index:
        logger.warning("[US-THEME-RISK] skipped: no raw US theme proxy series available")
        return 0
    out = pd.DataFrame(index=pd.DatetimeIndex(base_index, name="trade_date"))
    for col, ser in all_series.items():
        if not ser.empty:
            out[col] = ser.reindex(out.index)
    out["theme_proxy_ma20"] = out["theme_proxy_close"].rolling(20, min_periods=5).mean()
    out["theme_proxy_chg_5d"] = _pct_change(out["theme_proxy_close"], 5)
    out["theme_proxy_chg_10d"] = _pct_change(out["theme_proxy_close"], 10)
    out["theme_proxy_below_ma20_streak"] = _condition_streak(out["theme_proxy_close"] < out["theme_proxy_ma20"])
    for col in ("theme_spy_ratio", "theme_qqq_ratio", "qqq_spy_ratio", "hyg_lqd_ratio"):
        out[f"{col}_20d"] = out[col].rolling(20, min_periods=5).mean()
        out[f"{col}_chg_5d"] = _pct_change(out[col], 5)
        out[f"{col}_chg_10d"] = _pct_change(out[col], 10)
    out["theme_spy_below_ma20_streak"] = _condition_streak(out["theme_spy_ratio"] < out["theme_spy_ratio_20d"])
    out["theme_qqq_below_ma20_streak"] = _condition_streak(out["theme_qqq_ratio"] < out["theme_qqq_ratio_20d"])
    out["qqq_spy_below_ma20_streak"] = _condition_streak(out["qqq_spy_ratio"] < out["qqq_spy_ratio_20d"])
    out["hyg_lqd_below_ma20_streak"] = _condition_streak(out["hyg_lqd_ratio"] < out["hyg_lqd_ratio_20d"])
    out["vix_ma20"] = out["vix_close"].rolling(20, min_periods=5).mean()
    out["vix_above_ma20_streak"] = _condition_streak(out["vix_close"] > out["vix_ma20"])
    score = pd.Series(0.0, index=out.index)
    score += _streak_component(out["theme_spy_below_ma20_streak"], start=3, step=2.4, cap=28)
    score += _streak_component(out["theme_qqq_below_ma20_streak"], start=3, step=2.8, cap=32)
    score += _streak_component(out["theme_proxy_below_ma20_streak"], start=3, step=1.6, cap=18)
    score += _streak_component(out["qqq_spy_below_ma20_streak"], start=4, step=1.2, cap=12)
    score += _streak_component(out["hyg_lqd_below_ma20_streak"], start=4, step=1.6, cap=18)
    score += _risk_component_negative_trend(out["theme_spy_ratio_chg_10d"], scale=480, cap=14)
    score += _risk_component_negative_trend(out["theme_qqq_ratio_chg_10d"], scale=560, cap=18)
    score += _risk_component_negative_trend(out["theme_proxy_chg_10d"], scale=220, cap=10)
    score += _risk_component_negative_trend(out["hyg_lqd_ratio_chg_10d"], scale=550, cap=8)
    score += _streak_component(out["vix_above_ma20_streak"], start=3, step=0.7, cap=6)
    out["theme_risk_score"] = score.clip(lower=0, upper=100).round(4)
    out["theme_risk_regime"] = out["theme_risk_score"].map(_regime_from_score)
    wanted = ["theme_proxy_close", "theme_proxy_ma20", "theme_proxy_chg_5d", "theme_proxy_chg_10d", "theme_proxy_below_ma20_streak", "theme_spy_ratio", "theme_spy_ratio_20d", "theme_spy_ratio_chg_5d", "theme_spy_ratio_chg_10d", "theme_spy_below_ma20_streak", "theme_qqq_ratio", "theme_qqq_ratio_20d", "theme_qqq_ratio_chg_5d", "theme_qqq_ratio_chg_10d", "theme_qqq_below_ma20_streak", "qqq_spy_ratio", "qqq_spy_ratio_20d", "qqq_spy_below_ma20_streak", "hyg_lqd_ratio", "hyg_lqd_ratio_20d", "hyg_lqd_below_ma20_streak", "vix_close", "vix_ma20", "vix_above_ma20_streak", "theme_risk_score", "theme_risk_regime"]
    for col in wanted:
        if col not in out.columns:
            out[col] = pd.NA
    out = out[wanted].reset_index()
    out["trade_date"] = pd.to_datetime(out["trade_date"]).dt.date
    out = out.replace([float("inf"), float("-inf")], pd.NA)
    _create_theme_risk_table(engine)
    insert_cols = ["trade_date", *wanted]
    column_sql = ", ".join(f"`{col}`" for col in insert_cols)
    value_sql = ", ".join(f":{col}" for col in insert_cols)
    update_sql = ",\n            ".join(f"`{col}`=VALUES(`{col}`)" for col in wanted)
    insert_sql = text(f"""
        INSERT INTO `{THEME_RISK_TABLE}` ({column_sql})
        VALUES ({value_sql})
        ON DUPLICATE KEY UPDATE {update_sql}
    """)
    records = _mysql_null_safe_records(out)
    if not records:
        return 0
    with engine.begin() as conn:
        conn.execute(insert_sql, records)
    logger.info("[US-THEME-RISK] table=%s rows=%s latest=%s", THEME_RISK_TABLE, len(records), out["trade_date"].max())
    return len(records)


def _create_theme_backtest_table(engine) -> None:
    ddl = f"""
    CREATE TABLE IF NOT EXISTS `{THEME_RISK_BACKTEST_DAILY_TABLE}` (
        `trade_date` DATE NOT NULL,
        `theme_risk_score` DOUBLE NULL,
        `theme_risk_regime` VARCHAR(32) NULL,
        `future_5d_theme_proxy_return` DOUBLE NULL,
        `future_10d_theme_proxy_return` DOUBLE NULL,
        `future_20d_theme_proxy_return` DOUBLE NULL,
        `future_5d_theme_proxy_vs_spy_return` DOUBLE NULL,
        `future_10d_theme_proxy_vs_spy_return` DOUBLE NULL,
        `future_20d_theme_proxy_vs_spy_return` DOUBLE NULL,
        `future_5d_theme_proxy_vs_qqq_return` DOUBLE NULL,
        `future_10d_theme_proxy_vs_qqq_return` DOUBLE NULL,
        `future_20d_theme_proxy_vs_qqq_return` DOUBLE NULL,
        `theme_proxy_max_drawdown_5d` DOUBLE NULL,
        `theme_proxy_max_drawdown_10d` DOUBLE NULL,
        `theme_proxy_max_drawdown_20d` DOUBLE NULL,
        `theme_event_5d_flag` TINYINT NULL,
        `theme_event_10d_flag` TINYINT NULL,
        `theme_event_20d_flag` TINYINT NULL,
        `updated_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        PRIMARY KEY (`trade_date`)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """
    with engine.begin() as conn:
        conn.execute(text(ddl))


def update_us_theme_backtest(engine, *, log: logging.Logger | None = None) -> dict[str, int]:
    logger = log or logging.getLogger("AshareUSScraper")
    if not _mysql_table_exists(engine, THEME_RISK_TABLE):
        logger.warning("[US-THEME-RISK-BT] skipped: %s not found", THEME_RISK_TABLE)
        return {"detail_rows": 0}
    risk = pd.read_sql(text(f"SELECT * FROM `{THEME_RISK_TABLE}` WHERE `theme_risk_score` IS NOT NULL"), engine)
    if risk.empty:
        return {"detail_rows": 0}
    risk["trade_date"] = pd.to_datetime(risk["trade_date"], errors="coerce")
    risk = risk.dropna(subset=["trade_date"]).drop_duplicates(subset=["trade_date"], keep="last").set_index("trade_date").sort_index()
    spy = _read_us_close_series(engine, RISK_RAW_TABLES["spy"])
    qqq = _read_us_close_series(engine, RISK_RAW_TABLES["qqq"])
    theme_proxy = _read_us_close_series(engine, RISK_RAW_TABLES[THEME_PROXY_SOURCE])
    if spy.empty or qqq.empty or theme_proxy.empty:
        logger.warning("[US-THEME-RISK-BT] skipped: missing raw series spy=%s qqq=%s theme_proxy_source=%s rows=%s", len(spy), len(qqq), THEME_PROXY_SOURCE, len(theme_proxy))
        return {"detail_rows": 0}

    idx = risk.index.intersection(spy.index).intersection(qqq.index).intersection(theme_proxy.index)
    logger.info("[US-THEME-RISK-BT] align rows=%s risk=%s spy=%s qqq=%s theme_proxy_source=%s theme_proxy=%s", len(idx), len(risk), len(spy), len(qqq), THEME_PROXY_SOURCE, len(theme_proxy))
    if len(idx) < 120:
        logger.warning("[US-THEME-RISK-BT] skipped: insufficient aligned rows=%s", len(idx))
        return {"detail_rows": 0}
    risk = risk.reindex(idx); spy = spy.reindex(idx); qqq = qqq.reindex(idx); theme_proxy = theme_proxy.reindex(idx)
    detail = pd.DataFrame(index=idx)
    detail["theme_risk_score"] = risk["theme_risk_score"]
    detail["theme_risk_regime"] = risk["theme_risk_regime"]
    for h in RISK_BACKTEST_HORIZONS:
        detail[f"future_{h}d_theme_proxy_return"] = _forward_return(theme_proxy, h)
        detail[f"future_{h}d_theme_proxy_vs_spy_return"] = detail[f"future_{h}d_theme_proxy_return"] - _forward_return(spy, h)
        detail[f"future_{h}d_theme_proxy_vs_qqq_return"] = detail[f"future_{h}d_theme_proxy_return"] - _forward_return(qqq, h)
        detail[f"theme_proxy_max_drawdown_{h}d"] = _forward_max_drawdown(theme_proxy, h)
        detail[f"theme_event_{h}d_flag"] = ((detail[f"theme_proxy_max_drawdown_{h}d"] <= -0.08) | (detail[f"future_{h}d_theme_proxy_vs_spy_return"] <= -0.05) | (detail[f"future_{h}d_theme_proxy_vs_qqq_return"] <= -0.035)).astype("Int64")
    detail = detail.reset_index().rename(columns={"index": "trade_date"})
    detail["trade_date"] = pd.to_datetime(detail["trade_date"]).dt.date
    cols = ["trade_date", "theme_risk_score", "theme_risk_regime", "future_5d_theme_proxy_return", "future_10d_theme_proxy_return", "future_20d_theme_proxy_return", "future_5d_theme_proxy_vs_spy_return", "future_10d_theme_proxy_vs_spy_return", "future_20d_theme_proxy_vs_spy_return", "future_5d_theme_proxy_vs_qqq_return", "future_10d_theme_proxy_vs_qqq_return", "future_20d_theme_proxy_vs_qqq_return", "theme_proxy_max_drawdown_5d", "theme_proxy_max_drawdown_10d", "theme_proxy_max_drawdown_20d", "theme_event_5d_flag", "theme_event_10d_flag", "theme_event_20d_flag"]
    _create_theme_backtest_table(engine)
    column_sql = ", ".join(f"`{c}`" for c in cols)
    value_sql = ", ".join(f":{c}" for c in cols)
    update_sql = ",\n            ".join(f"`{c}`=VALUES(`{c}`)" for c in cols if c != "trade_date")
    detail_sql = text(f"INSERT INTO `{THEME_RISK_BACKTEST_DAILY_TABLE}` ({column_sql}) VALUES ({value_sql}) ON DUPLICATE KEY UPDATE {update_sql}")
    with engine.begin() as conn:
        conn.execute(detail_sql, _mysql_null_safe_records(detail[cols]))
    logger.info("[US-THEME-RISK-BT] detail_table=%s rows=%s", THEME_RISK_BACKTEST_DAILY_TABLE, len(detail))
    return {"detail_rows": int(len(detail))}


def _create_theme_event_tables(engine) -> None:
    event_ddl = f"""
    CREATE TABLE IF NOT EXISTS `{THEME_FAILURE_EVENT_TABLE}` (
        `trade_date` DATE NOT NULL,
        `theme_risk_score` DOUBLE NULL,
        `theme_risk_regime` VARCHAR(32) NULL,
        `theme_proxy_max_drawdown_5d` DOUBLE NULL,
        `theme_proxy_max_drawdown_10d` DOUBLE NULL,
        `theme_proxy_max_drawdown_20d` DOUBLE NULL,
        `future_5d_theme_proxy_vs_spy_return` DOUBLE NULL,
        `future_10d_theme_proxy_vs_spy_return` DOUBLE NULL,
        `future_20d_theme_proxy_vs_spy_return` DOUBLE NULL,
        `future_5d_theme_proxy_vs_qqq_return` DOUBLE NULL,
        `future_10d_theme_proxy_vs_qqq_return` DOUBLE NULL,
        `future_20d_theme_proxy_vs_qqq_return` DOUBLE NULL,
        `failure_event_5d_flag` TINYINT NULL,
        `failure_event_10d_flag` TINYINT NULL,
        `failure_event_20d_flag` TINYINT NULL,
        `event_reason_5d` VARCHAR(128) NULL,
        `event_reason_10d` VARCHAR(128) NULL,
        `event_reason_20d` VARCHAR(128) NULL,
        `updated_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        PRIMARY KEY (`trade_date`),
        KEY `idx_theme_failure_20d` (`failure_event_20d_flag`),
        KEY `idx_theme_failure_regime` (`theme_risk_regime`)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """
    capture_ddl = f"""
    CREATE TABLE IF NOT EXISTS `{THEME_EVENT_CAPTURE_DAILY_TABLE}` (
        `trade_date` DATE NOT NULL,
        `horizon_days` INT NOT NULL,
        `warning_threshold` DOUBLE NOT NULL,
        `failure_event_flag` TINYINT NULL,
        `captured_flag` TINYINT NULL,
        `warning_date` DATE NULL,
        `lead_days` INT NULL,
        `max_warning_score_prior` DOUBLE NULL,
        `event_reason` VARCHAR(128) NULL,
        `updated_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        PRIMARY KEY (`trade_date`, `horizon_days`, `warning_threshold`),
        KEY `idx_theme_capture_horizon_threshold` (`horizon_days`, `warning_threshold`),
        KEY `idx_theme_capture_event` (`failure_event_flag`, `captured_flag`)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """
    summary_ddl = f"""
    CREATE TABLE IF NOT EXISTS `{THEME_EVENT_CAPTURE_SUMMARY_TABLE}` (
        `horizon_days` INT NOT NULL,
        `warning_threshold` DOUBLE NOT NULL,
        `sample_count` INT NULL,
        `event_count` INT NULL,
        `captured_event_count` INT NULL,
        `missed_event_count` INT NULL,
        `warning_count` INT NULL,
        `true_positive_warning_count` INT NULL,
        `false_alarm_count` INT NULL,
        `capture_rate` DOUBLE NULL,
        `false_alarm_rate` DOUBLE NULL,
        `avg_lead_days` DOUBLE NULL,
        `median_lead_days` DOUBLE NULL,
        `avg_event_drawdown` DOUBLE NULL,
        `avg_captured_event_drawdown` DOUBLE NULL,
        `verdict` VARCHAR(64) NULL,
        `updated_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        PRIMARY KEY (`horizon_days`, `warning_threshold`)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """
    event_cols = {
        "theme_risk_score": "DOUBLE NULL", "theme_risk_regime": "VARCHAR(32) NULL",
        "theme_proxy_max_drawdown_5d": "DOUBLE NULL", "theme_proxy_max_drawdown_10d": "DOUBLE NULL", "theme_proxy_max_drawdown_20d": "DOUBLE NULL",
        "future_5d_theme_proxy_vs_spy_return": "DOUBLE NULL", "future_10d_theme_proxy_vs_spy_return": "DOUBLE NULL", "future_20d_theme_proxy_vs_spy_return": "DOUBLE NULL",
        "future_5d_theme_proxy_vs_qqq_return": "DOUBLE NULL", "future_10d_theme_proxy_vs_qqq_return": "DOUBLE NULL", "future_20d_theme_proxy_vs_qqq_return": "DOUBLE NULL",
        "failure_event_5d_flag": "TINYINT NULL", "failure_event_10d_flag": "TINYINT NULL", "failure_event_20d_flag": "TINYINT NULL",
        "event_reason_5d": "VARCHAR(128) NULL", "event_reason_10d": "VARCHAR(128) NULL", "event_reason_20d": "VARCHAR(128) NULL",
    }
    capture_cols = {
        "horizon_days": "INT NOT NULL", "warning_threshold": "DOUBLE NOT NULL", "failure_event_flag": "TINYINT NULL",
        "captured_flag": "TINYINT NULL", "warning_date": "DATE NULL", "lead_days": "INT NULL",
        "max_warning_score_prior": "DOUBLE NULL", "event_reason": "VARCHAR(128) NULL",
    }
    summary_cols = {
        "sample_count": "INT NULL", "event_count": "INT NULL", "captured_event_count": "INT NULL",
        "missed_event_count": "INT NULL", "warning_count": "INT NULL", "true_positive_warning_count": "INT NULL",
        "false_alarm_count": "INT NULL", "capture_rate": "DOUBLE NULL", "false_alarm_rate": "DOUBLE NULL",
        "avg_lead_days": "DOUBLE NULL", "median_lead_days": "DOUBLE NULL", "avg_event_drawdown": "DOUBLE NULL",
        "avg_captured_event_drawdown": "DOUBLE NULL", "verdict": "VARCHAR(64) NULL",
    }
    with engine.begin() as conn:
        conn.execute(text(event_ddl))
        conn.execute(text(capture_ddl))
        conn.execute(text(summary_ddl))
        for table_name, columns in (
            (THEME_FAILURE_EVENT_TABLE, event_cols),
            (THEME_EVENT_CAPTURE_DAILY_TABLE, capture_cols),
            (THEME_EVENT_CAPTURE_SUMMARY_TABLE, summary_cols),
        ):
            existing = {row[0] for row in conn.execute(text(f"SHOW COLUMNS FROM `{table_name}`"))}
            for col_name, col_def in columns.items():
                if col_name not in existing:
                    conn.execute(text(f"ALTER TABLE `{table_name}` ADD COLUMN `{col_name}` {col_def}"))


def _theme_failure_reason(row: pd.Series, horizon: int) -> str:
    reasons: list[str] = []
    dd = row.get(f"theme_proxy_max_drawdown_{horizon}d")
    vs_spy = row.get(f"future_{horizon}d_theme_proxy_vs_spy_return")
    vs_qqq = row.get(f"future_{horizon}d_theme_proxy_vs_qqq_return")
    if pd.notna(dd) and float(dd) <= -0.08:
        reasons.append("DRAWDOWN_LE_8PCT")
    if pd.notna(vs_spy) and float(vs_spy) <= -0.05:
        reasons.append("UNDERPERFORM_SPY_LE_5PCT")
    if pd.notna(vs_qqq) and float(vs_qqq) <= -0.035:
        reasons.append("UNDERPERFORM_QQQ_LE_3_5PCT")
    return ",".join(reasons) if reasons else ""


def _theme_event_verdict(capture_rate: float | None, false_alarm_rate: float | None, event_count: int) -> str:
    if event_count <= 0:
        return "NO_EVENTS"
    if capture_rate is None or pd.isna(capture_rate):
        return "INSUFFICIENT"
    far = 1.0 if false_alarm_rate is None or pd.isna(false_alarm_rate) else float(false_alarm_rate)
    if capture_rate >= 0.65 and far <= 0.75:
        return "USEFUL_EVENT_WARNING"
    if capture_rate >= 0.50 and far <= 0.85:
        return "WATCHLIST_EVENT_SIGNAL"
    return "WEAK_OR_NOISY"


def update_us_theme_event_engine(engine, *, log: logging.Logger | None = None) -> dict[str, int]:
    """Validate theme risk as an event-capture engine, not a return predictor.

    A theme failure event is defined on a signal date by future forward outcome:
    - theme proxy max drawdown <= -8%, or
    - theme proxy underperforms SPY by <= -5%, or
    - theme proxy underperforms QQQ by <= -3.5%.

    Capture checks whether the score crossed a threshold before the event date
    within the same horizon window. This matches the user's objective: early
    warning for de-risking, not reporting after the drawdown already happened.
    """
    logger = log or logging.getLogger("AshareUSScraper")
    if not _mysql_table_exists(engine, THEME_RISK_BACKTEST_DAILY_TABLE):
        logger.warning("[US-THEME-EVENT] skipped: %s not found", THEME_RISK_BACKTEST_DAILY_TABLE)
        return {"event_rows": 0, "capture_rows": 0, "summary_rows": 0}

    df = pd.read_sql(text(f"SELECT * FROM `{THEME_RISK_BACKTEST_DAILY_TABLE}`"), engine)
    if df.empty:
        logger.warning("[US-THEME-EVENT] skipped: no theme backtest rows")
        return {"event_rows": 0, "capture_rows": 0, "summary_rows": 0}
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
    df = df.dropna(subset=["trade_date"]).drop_duplicates(subset=["trade_date"], keep="last").sort_values("trade_date").reset_index(drop=True)
    df["theme_risk_score"] = pd.to_numeric(df.get("theme_risk_score"), errors="coerce")
    if len(df) < 120:
        logger.warning("[US-THEME-EVENT] skipped: insufficient rows=%s", len(df))
        return {"event_rows": 0, "capture_rows": 0, "summary_rows": 0}

    events = pd.DataFrame()
    events["trade_date"] = df["trade_date"].dt.date
    events["theme_risk_score"] = df["theme_risk_score"]
    events["theme_risk_regime"] = df.get("theme_risk_regime")
    for h in RISK_BACKTEST_HORIZONS:
        dd_col = f"theme_proxy_max_drawdown_{h}d"
        vs_spy_col = f"future_{h}d_theme_proxy_vs_spy_return"
        vs_qqq_col = f"future_{h}d_theme_proxy_vs_qqq_return"
        for c in (dd_col, vs_spy_col, vs_qqq_col):
            events[c] = pd.to_numeric(df.get(c), errors="coerce")
        flag = (
            (events[dd_col] <= -0.08)
            | (events[vs_spy_col] <= -0.05)
            | (events[vs_qqq_col] <= -0.035)
        )
        events[f"failure_event_{h}d_flag"] = flag.fillna(False).astype("Int64")
        events[f"event_reason_{h}d"] = [
            _theme_failure_reason(row, h) for _, row in events.iterrows()
        ]

    _create_theme_event_tables(engine)
    event_cols = [
        "trade_date", "theme_risk_score", "theme_risk_regime",
        "theme_proxy_max_drawdown_5d", "theme_proxy_max_drawdown_10d", "theme_proxy_max_drawdown_20d",
        "future_5d_theme_proxy_vs_spy_return", "future_10d_theme_proxy_vs_spy_return", "future_20d_theme_proxy_vs_spy_return",
        "future_5d_theme_proxy_vs_qqq_return", "future_10d_theme_proxy_vs_qqq_return", "future_20d_theme_proxy_vs_qqq_return",
        "failure_event_5d_flag", "failure_event_10d_flag", "failure_event_20d_flag",
        "event_reason_5d", "event_reason_10d", "event_reason_20d",
    ]
    event_insert = text(f"""
        INSERT INTO `{THEME_FAILURE_EVENT_TABLE}` ({', '.join(f'`{c}`' for c in event_cols)})
        VALUES ({', '.join(f':{c}' for c in event_cols)})
        ON DUPLICATE KEY UPDATE
            {', '.join(f'`{c}`=VALUES(`{c}`)' for c in event_cols if c != 'trade_date')}
    """)

    capture_records: list[dict] = []
    summary_records: list[dict] = []
    scores = pd.to_numeric(df["theme_risk_score"], errors="coerce")
    n = len(df)
    for h in RISK_BACKTEST_HORIZONS:
        event_flag = events[f"failure_event_{h}d_flag"].fillna(0).astype(int).to_numpy()
        dd_values = pd.to_numeric(events[f"theme_proxy_max_drawdown_{h}d"], errors="coerce")
        for threshold in RISK_BACKTEST_THRESHOLDS:
            warning = (scores >= float(threshold)).fillna(False).to_numpy()
            captured_flags: list[int | None] = []
            warning_dates: list[date | None] = []
            lead_days_list: list[int | None] = []
            max_prior_scores: list[float | None] = []
            for i in range(n):
                is_event = int(event_flag[i]) == 1
                if not is_event:
                    captured_flags.append(0)
                    warning_dates.append(None)
                    lead_days_list.append(None)
                    max_prior_scores.append(None)
                    continue
                start_i = max(0, i - int(h))
                prior_idx = [j for j in range(start_i, i) if bool(warning[j])]
                if prior_idx:
                    first_j = prior_idx[0]
                    captured_flags.append(1)
                    warning_dates.append(df.loc[first_j, "trade_date"].date())
                    lead_days_list.append(int(i - first_j))
                    max_prior_scores.append(float(scores.iloc[prior_idx].max()))
                else:
                    captured_flags.append(0)
                    warning_dates.append(None)
                    lead_days_list.append(None)
                    window_scores = scores.iloc[start_i:i]
                    max_prior_scores.append(None if window_scores.empty or pd.isna(window_scores.max()) else float(window_scores.max()))

            for i in range(n):
                capture_records.append({
                    "trade_date": df.loc[i, "trade_date"].date(),
                    "horizon_days": int(h),
                    "warning_threshold": float(threshold),
                    "failure_event_flag": int(event_flag[i]),
                    "captured_flag": int(captured_flags[i] or 0),
                    "warning_date": warning_dates[i],
                    "lead_days": lead_days_list[i],
                    "max_warning_score_prior": max_prior_scores[i],
                    "event_reason": events.loc[i, f"event_reason_{h}d"],
                })

            event_count = int(event_flag.sum())
            captured_event_count = int(sum(1 for i, f in enumerate(event_flag) if int(f) == 1 and int(captured_flags[i] or 0) == 1))
            missed_event_count = event_count - captured_event_count
            warning_count = int(warning.sum())
            true_positive_warning_count = 0
            for i in range(n):
                if not bool(warning[i]):
                    continue
                end_i = min(n, i + int(h) + 1)
                if int(event_flag[i + 1:end_i].sum()) > 0:
                    true_positive_warning_count += 1
            false_alarm_count = warning_count - true_positive_warning_count
            capture_rate = captured_event_count / event_count if event_count else None
            false_alarm_rate = false_alarm_count / warning_count if warning_count else None
            lead_values = [x for x in lead_days_list if x is not None]
            avg_lead = float(pd.Series(lead_values).mean()) if lead_values else None
            median_lead = float(pd.Series(lead_values).median()) if lead_values else None
            event_dd = dd_values[event_flag == 1]
            captured_dd = dd_values[[int(f) == 1 and int(captured_flags[i] or 0) == 1 for i, f in enumerate(event_flag)]]
            summary_records.append({
                "horizon_days": int(h),
                "warning_threshold": float(threshold),
                "sample_count": int(n),
                "event_count": event_count,
                "captured_event_count": captured_event_count,
                "missed_event_count": missed_event_count,
                "warning_count": warning_count,
                "true_positive_warning_count": true_positive_warning_count,
                "false_alarm_count": false_alarm_count,
                "capture_rate": capture_rate,
                "false_alarm_rate": false_alarm_rate,
                "avg_lead_days": avg_lead,
                "median_lead_days": median_lead,
                "avg_event_drawdown": None if event_dd.empty else float(event_dd.mean()),
                "avg_captured_event_drawdown": None if captured_dd.empty else float(captured_dd.mean()),
                "verdict": _theme_event_verdict(capture_rate, false_alarm_rate, event_count),
            })

    capture_cols = [
        "trade_date", "horizon_days", "warning_threshold", "failure_event_flag", "captured_flag",
        "warning_date", "lead_days", "max_warning_score_prior", "event_reason",
    ]
    capture_insert = text(f"""
        INSERT INTO `{THEME_EVENT_CAPTURE_DAILY_TABLE}` ({', '.join(f'`{c}`' for c in capture_cols)})
        VALUES ({', '.join(f':{c}' for c in capture_cols)})
        ON DUPLICATE KEY UPDATE
            {', '.join(f'`{c}`=VALUES(`{c}`)' for c in capture_cols if c not in {'trade_date', 'horizon_days', 'warning_threshold'})}
    """)
    summary_cols = [
        "horizon_days", "warning_threshold", "sample_count", "event_count", "captured_event_count", "missed_event_count",
        "warning_count", "true_positive_warning_count", "false_alarm_count", "capture_rate", "false_alarm_rate",
        "avg_lead_days", "median_lead_days", "avg_event_drawdown", "avg_captured_event_drawdown", "verdict",
    ]
    summary_insert = text(f"""
        INSERT INTO `{THEME_EVENT_CAPTURE_SUMMARY_TABLE}` ({', '.join(f'`{c}`' for c in summary_cols)})
        VALUES ({', '.join(f':{c}' for c in summary_cols)})
        ON DUPLICATE KEY UPDATE
            {', '.join(f'`{c}`=VALUES(`{c}`)' for c in summary_cols if c not in {'horizon_days', 'warning_threshold'})}
    """)
    with engine.begin() as conn:
        conn.execute(event_insert, _mysql_null_safe_records(events[event_cols]))
        conn.execute(capture_insert, _mysql_null_safe_records(pd.DataFrame(capture_records)[capture_cols]))
        conn.execute(summary_insert, _mysql_null_safe_records(pd.DataFrame(summary_records)[summary_cols]))
    logger.info(
        "[US-THEME-EVENT] event_table=%s rows=%s capture_table=%s rows=%s summary_table=%s rows=%s",
        THEME_FAILURE_EVENT_TABLE,
        len(events),
        THEME_EVENT_CAPTURE_DAILY_TABLE,
        len(capture_records),
        THEME_EVENT_CAPTURE_SUMMARY_TABLE,
        len(summary_records),
    )
    return {"event_rows": int(len(events)), "capture_rows": int(len(capture_records)), "summary_rows": int(len(summary_records))}

def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def build_source(name: str, *, config_path: str = "") -> BaseDataSource:
    key = name.lower().strip()
    if key == "spx":
        return global_index_fallback_source(".SPX", name="spx", yahoo_symbol="^GSPC", config_path=config_path)
    if key == "spy":
        return us_equity_or_etf_fallback_source("SPY", name="spy", config_path=config_path)
    if key == "qqq":
        return us_equity_or_etf_fallback_source("QQQ", name="qqq", config_path=config_path)
    if key == "hyg":
        return us_equity_or_etf_fallback_source("HYG", name="hyg", config_path=config_path)
    if key == "lqd":
        return us_equity_or_etf_fallback_source("LQD", name="lqd", config_path=config_path)
    if key == "xlu":
        return us_equity_or_etf_fallback_source("XLU", name="xlu", config_path=config_path)
    if key == "xlp":
        return us_equity_or_etf_fallback_source("XLP", name="xlp", config_path=config_path)
    if key == "soxx":
        return us_equity_or_etf_fallback_source("SOXX", name="soxx", config_path=config_path)
    if key == "tlt":
        return us_equity_or_etf_fallback_source("TLT", name="tlt", config_path=config_path)
    if key == "ief":
        return us_equity_or_etf_fallback_source("IEF", name="ief", config_path=config_path)
    if key == "iwm":
        return us_equity_or_etf_fallback_source("IWM", name="iwm", config_path=config_path)
    if key == "rut":
        return global_index_fallback_source(".RUT", name="russell_2000", yahoo_symbol="^RUT", config_path=config_path)
    if key == "gdx":
        return us_equity_or_etf_fallback_source("GDX", name="gdx", config_path=config_path)
    if key == "gld":
        return us_equity_or_etf_fallback_source("GLD", name="gld", config_path=config_path)
    if key == "ibb":
        return us_equity_or_etf_fallback_source("IBB", name="ibb", config_path=config_path)
    if key == "xbi":
        return us_equity_or_etf_fallback_source("XBI", name="xbi", config_path=config_path)
    if key.startswith("us:"):
        symbol = key.split(":", 1)[1]
        return us_equity_or_etf_fallback_source(symbol, name=symbol.lower(), config_path=config_path)
    if key.startswith("idx:"):
        symbol = key.split(":", 1)[1]
        return global_index_fallback_source(symbol, name=symbol.lower().replace(".", "_"), config_path=config_path)
    if key == "vix":
        return CBOEVIXSource()
    if key == "vix_term":
        return VIXTermStructureSource()
    if key == "wikipedia_pageviews_risk":
        return WikipediaPageviewsRiskSource()
    if key == "gdelt_risk":
        return GDELTRiskSource()
    if key == "fear_greed":
        return CNNFearGreedSource(name="fear_greed")
    if key == "margin_debt":
        return FinraMarginDebtSource(name="margin_debt")
    if key in FRED_DISABLED_SOURCES:
        if key in {"ust10", "us10y", "dgs10"}:
            return us_equity_or_etf_fallback_source("IEF", name="ust10_proxy_ief", config_path=config_path)
        if key in {"ust30", "us30y", "dgs30"}:
            return us_equity_or_etf_fallback_source("TLT", name="ust30_proxy_tlt", config_path=config_path)
        raise ValueError(
            f"FRED source '{name}' is disabled for daily AshareScraper runs because FRED is not reliable in this environment. "
            "Use practical ETF proxies instead: ief/tlt, or explicit ust10/ust30 aliases for ust10_proxy_ief/ust30_proxy_tlt."
        )
    if key == "aaii_sentiment":
        raise ValueError("aaii_sentiment is intentionally disabled for the unified AshareScraper global-risk module.")
    raise ValueError(f"Unknown US/global source name: {name}")


def _source_worker(source_name: str, start: str, end: str, refresh: bool, overlap: int, config_path: str) -> tuple[str, bool, str, str, int]:
    try:
        src = build_source(source_name, config_path=config_path)
        df = src.update(start_date=start, end_date=end, refresh=refresh, overlap_trading_days=overlap)
        return source_name, True, src.name, src.data_path, 0 if df is None else len(df)
    except Exception as exc:
        tb_path = os.path.join(tempfile.gettempdir(), f"ashare_us_scraper_{source_name}_{os.getpid()}.log")
        with open(tb_path, "w", encoding="utf-8") as fh:
            fh.write(traceback.format_exc())
        return source_name, False, "", tb_path, 0


def _normalize_date(raw: str, label: str) -> str:
    s = str(raw or "").strip()
    if not s:
        raise ValueError(f"{label} is empty")
    if s.lower() == "latest":
        return date.today().strftime("%Y%m%d")
    if len(s) == 8 and s.isdigit():
        datetime.strptime(s, "%Y%m%d")
        return s
    if len(s) == 10 and s[4] == "-" and s[7] == "-":
        return datetime.strptime(s, "%Y-%m-%d").strftime("%Y%m%d")
    raise ValueError(f"Invalid {label}: {raw}. Expected latest, YYYYMMDD, or YYYY-MM-DD.")


def run_us_scraper(
    *,
    engine=None,
    log: logging.Logger | None = None,
    sources: list[str] | None = None,
    daily: bool = False,
    refresh: bool = False,
    history_backfill: bool = False,
    years: int = 1,
    overlap_trading_days: int = 7,
    source_timeout_seconds: int = 180,
    load_db: bool = True,
    load_only: bool = False,
    compute_breadth_proxy: bool = False,
    risk_backtest: bool = False,
    start_date: str = "",
    end_date: str = "",
    config_path: str = "",
    workers: int = 0,
) -> dict[str, int]:
    logger = log or logging.getLogger("AshareUSScraper")
    selected = [s.strip() for s in (sources or []) if s and s.strip()]
    if daily or not selected:
        selected = list(DAILY_SOURCES) if not selected else selected

    end_s = _normalize_date(end_date or "latest", "--us-end-date")
    if start_date:
        start_s = _normalize_date(start_date, "--us-start-date")
    else:
        end_dt = datetime.strptime(end_s, "%Y%m%d").date()
        start_s = date(end_dt.year - max(1, int(years)), end_dt.month, end_dt.day).strftime("%Y%m%d")
    if start_s > end_s:
        raise ValueError(f"US/global date window invalid: {start_s} > {end_s}")

    logger.info("[US-GLOBAL] selected=%s start=%s end=%s load_db=%s load_only=%s history_backfill=%s", selected, start_s, end_s, load_db, load_only, history_backfill)
    data_paths: dict[str, str] = {}
    rows_by_source: dict[str, int] = {}

    if not load_only:
        if int(workers or 0) > 1 and len(selected) > 1:
            with mp.Pool(processes=int(workers)) as pool:
                async_results = [
                    pool.apply_async(_source_worker, (name, start_s, end_s, refresh, overlap_trading_days, config_path))
                    for name in selected
                ]
                for name, ar in zip(selected, async_results):
                    try:
                        source_name, ok, normalized_name, path, nrows = ar.get(timeout=max(30, int(source_timeout_seconds)))
                    except mp.TimeoutError:
                        raise RuntimeError(f"US/global source timed out: {name}")
                    if not ok:
                        if source_name in REQUIRE_NON_EMPTY_SOURCES:
                            raise RuntimeError(f"US/global required source failed: {source_name}; traceback={path}")
                        logger.warning("[US-GLOBAL] optional source failed source=%s traceback=%s", source_name, path)
                        continue
                    data_paths[normalized_name] = path
                    rows_by_source[normalized_name] = nrows
                    logger.info("[US-GLOBAL] downloaded source=%s rows=%s path=%s", normalized_name, nrows, path)
        else:
            for name in selected:
                try:
                    src = build_source(name, config_path=config_path)
                    df = src.update(start_date=start_s, end_date=end_s, refresh=refresh, overlap_trading_days=overlap_trading_days)
                    data_paths[src.name] = src.data_path
                    rows_by_source[src.name] = 0 if df is None else len(df)
                    logger.info("[US-GLOBAL] downloaded source=%s rows=%s path=%s", src.name, rows_by_source[src.name], src.data_path)
                except Exception as exc:
                    if name in REQUIRE_NON_EMPTY_SOURCES:
                        raise
                    logger.warning("[US-GLOBAL] optional source failed source=%s err=%s", name, exc)

    if compute_breadth_proxy:
        root = _project_root()
        data_dir = root / "data" / "us_scraper"
        proxy_a, proxy_b = build_breadth_proxy_a_b(str(data_dir))
        save_breadth_proxy_csvs(proxy_a, proxy_b, str(data_dir))
        data_paths["breadth_proxy_a"] = str(data_dir / "breadth_proxy_a.csv")
        data_paths["breadth_proxy_b"] = str(data_dir / "breadth_proxy_b.csv")

    if load_only:
        root = _project_root()
        data_dir = root / "data" / "us_scraper"
        for path in sorted(data_dir.glob("*.csv")):
            data_paths[path.stem] = str(path)

    if load_db:
        if engine is None:
            raise RuntimeError("engine is required when load_db=True")
        for source_name, csv_path in sorted(data_paths.items()):
            table = f"us_{source_name}"
            inserted = upsert_csv_to_mysql(engine, csv_path, table, start_date=start_s, end_date=end_s)
            logger.info("[US-GLOBAL][DB cn_market_red] table=%s rows=%s csv=%s", table, inserted, csv_path)
            rows_by_source[f"db:{table}"] = inserted
        # Automatically refresh the derived trend/features table after raw US/global daily data is loaded.
        # GrowthAlpha should consume this table instead of recalculating raw ratios itself.
        risk_rows = update_us_risk_preference_daily(engine, log=logger)
        rows_by_source[f"db:{RISK_PREFERENCE_TABLE}"] = risk_rows
        theme_risk_rows = update_us_theme_risk_daily(engine, log=logger)
        rows_by_source[f"db:{THEME_RISK_TABLE}"] = theme_risk_rows

    if risk_backtest or history_backfill:
        if engine is None:
            raise RuntimeError("engine is required when risk_backtest=True")
        bt_rows = update_us_risk_preference_backtest(engine, log=logger)
        rows_by_source[f"db:{RISK_BACKTEST_DAILY_TABLE}"] = int(bt_rows.get("detail_rows", 0))
        rows_by_source[f"db:{RISK_BACKTEST_SUMMARY_TABLE}"] = int(bt_rows.get("summary_rows", 0))
        theme_bt_rows = update_us_theme_backtest(engine, log=logger)
        rows_by_source[f"db:{THEME_RISK_BACKTEST_DAILY_TABLE}"] = int(theme_bt_rows.get("detail_rows", 0))
        theme_event_rows = update_us_theme_event_engine(engine, log=logger)
        rows_by_source[f"db:{THEME_FAILURE_EVENT_TABLE}"] = int(theme_event_rows.get("event_rows", 0))
        rows_by_source[f"db:{THEME_EVENT_CAPTURE_DAILY_TABLE}"] = int(theme_event_rows.get("capture_rows", 0))
        rows_by_source[f"db:{THEME_EVENT_CAPTURE_SUMMARY_TABLE}"] = int(theme_event_rows.get("summary_rows", 0))

    return rows_by_source


class USGlobalScraperTask:
    name = "USGlobalScraperTask"

    def __init__(
        self,
        *,
        sources: list[str] | None = None,
        daily: bool = True,
        refresh: bool = False,
        history_backfill: bool = False,
        years: int = 1,
        overlap_trading_days: int = 7,
        source_timeout_seconds: int = 180,
        load_db: bool = True,
        load_only: bool = False,
        compute_breadth_proxy: bool = False,
        risk_backtest: bool = False,
        start_date: str = "",
        end_date: str = "",
        config_path: str = "",
        workers: int = 0,
    ) -> None:
        self.sources = sources or []
        self.daily = daily
        self.refresh = refresh
        self.history_backfill = history_backfill
        self.years = years
        self.overlap_trading_days = overlap_trading_days
        self.source_timeout_seconds = source_timeout_seconds
        self.load_db = load_db
        self.load_only = load_only
        self.compute_breadth_proxy = compute_breadth_proxy
        self.risk_backtest = risk_backtest
        self.start_date = start_date
        self.end_date = end_date
        self.config_path = config_path
        self.workers = workers

    def run(self, ctx) -> None:
        run_us_scraper(
            engine=ctx.engine,
            log=ctx.log,
            sources=self.sources,
            daily=self.daily,
            refresh=self.refresh or self.history_backfill,
            history_backfill=self.history_backfill,
            years=self.years,
            overlap_trading_days=self.overlap_trading_days,
            source_timeout_seconds=self.source_timeout_seconds,
            load_db=self.load_db,
            load_only=self.load_only,
            compute_breadth_proxy=self.compute_breadth_proxy,
            risk_backtest=self.risk_backtest,
            # Daily mode follows the normal AshareScraper date window (usually one trading day).
            # History-backfill mode intentionally ignores the A-share date window so --us-years
            # can rebuild a full US/global risk history for trend calculation.
            start_date=(self.start_date if self.history_backfill else (self.start_date or getattr(ctx.config, "start_date", "") or "")),
            end_date=(self.end_date if self.history_backfill else (self.end_date or getattr(ctx.config, "end_date", "") or "")),
            config_path=self.config_path,
            workers=self.workers,
        )
