from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
from sqlalchemy import text

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.settings import build_engine, get_db_name


CORE_TABLES = [
    "cn_local_industry_proxy_daily",
    "cn_ga_stock_role_map_daily",
    "cn_stock_mainline_strength_daily",
    "cn_ga_mainline_radar_daily",
    "cn_ga_market_pulse_daily",
    "cn_mainline_lifecycle_daily",
]


def _as_date(value: object) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    return datetime.strptime(str(value)[:10], "%Y-%m-%d").date()


def _query_scalar(sql: str, params: dict | None = None) -> object:
    engine = build_engine()
    with engine.connect() as conn:
        return conn.execute(text(sql), params or {}).scalar()


def _latest_signal_trade_date() -> str:
    """Resolve the latest trade date that the V8 signal chain can realistically build.

    Stock and index loaders may be fresher than:
    - `cn_stock_daily_basic`
    - the static `LOCAL_FINE` membership horizon in `cn_local_industry_map_hist`

    The signal guard should default to the latest date supported by the full
    mainline chain rather than blindly forcing the latest raw market date.
    """
    stock_date = _query_scalar("SELECT MAX(trade_date) FROM cn_stock_daily_price")
    index_date = _query_scalar("SELECT MAX(trade_date) FROM cn_index_daily_price")
    basic_date = _query_scalar("SELECT MAX(trade_date) FROM cn_stock_daily_basic")
    local_fine_date = _query_scalar(
        """
        SELECT MAX(COALESCE(out_date, in_date))
        FROM cn_local_industry_map_hist
        WHERE industry_level = 'L3'
        """
    )
    candidates = [stock_date, index_date, basic_date, local_fine_date]
    if any(value is None for value in candidates):
        raise SystemExit(
            "missing latest trade-date inputs for signal guard "
            "(stock/index/daily_basic/local_fine_map_hist)"
        )
    resolved = min(_as_date(value) for value in candidates)
    latest_raw = min(_as_date(stock_date), _as_date(index_date))
    if resolved < latest_raw:
        print(
            "[SIGNAL_GUARD] latest raw market date is",
            latest_raw.isoformat(),
            "but signal chain is only buildable through",
            resolved.isoformat(),
            "(bounded by daily_basic / LOCAL_FINE map coverage)",
        )
    return resolved.isoformat()


def _table_count(table: str, trade_date: str) -> int:
    value = _query_scalar(f"SELECT COUNT(*) FROM {table} WHERE trade_date = :trade_date", {"trade_date": trade_date})
    return int(value or 0)


def _query_df(sql: str, params: dict | None = None) -> pd.DataFrame:
    engine = build_engine()
    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn, params=params or {})


def _execute(sql: str, params: dict | None = None) -> int:
    engine = build_engine()
    with engine.begin() as conn:
        result = conn.execute(text(sql), params or {})
        return int(result.rowcount or 0)


def _execute_many(sql: str, rows: list[dict]) -> int:
    if not rows:
        return 0
    engine = build_engine()
    with engine.begin() as conn:
        result = conn.execute(text(sql), rows)
        return int(result.rowcount or 0)


def _f(value: object, default: float = 0.0) -> float:
    val = pd.to_numeric(value, errors="coerce")
    if pd.isna(val):
        return default
    return float(val)


def _db_args() -> list[str]:
    return [
        "--db-host",
        os.getenv("ASHARE_MYSQL_HOST", "localhost"),
        "--db-port",
        os.getenv("ASHARE_MYSQL_PORT", "3306"),
        "--db-user",
        os.getenv("ASHARE_MYSQL_USER", "cn_opr_red"),
        "--db-password",
        os.getenv("ASHARE_MYSQL_PASSWORD", "sec_Bobo123"),
        "--db-name",
        get_db_name(),
    ]


def _run(script: str, args: list[str]) -> None:
    cmd = [sys.executable, str(PROJECT_ROOT / script)] + args
    redacted = ["******" if i > 0 and cmd[i - 1] == "--db-password" else token for i, token in enumerate(cmd)]
    print("[SIGNAL_GUARD] run:", " ".join(redacted))
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def _ensure_layer(table: str, trade_date: str, force: bool, script: str, args: list[str]) -> str:
    before = 0 if force else _table_count(table, trade_date)
    if before > 0:
        return f"exists:{before}"
    _run(script, args)
    after = _table_count(table, trade_date)
    if after <= 0:
        raise RuntimeError(f"{table} still missing for {trade_date} after running {script}")
    return f"built:{after}"


def _fallback_build_radar(trade_date: str, start_date: str) -> str:
    proxy = _query_df(
        """
        SELECT trade_date, industry_id, industry_name, member_count, up_count, down_count,
               up_ratio, ret_eqw, amount_total, top5_concentration
        FROM cn_local_industry_proxy_daily
        WHERE trade_date BETWEEN :start_date AND :trade_date
        """,
        {"start_date": start_date, "trade_date": trade_date},
    )
    if proxy.empty:
        raise RuntimeError(f"cannot fallback-build cn_ga_mainline_radar_daily: proxy empty for {start_date}..{trade_date}")
    proxy["trade_date"] = pd.to_datetime(proxy["trade_date"]).dt.strftime("%Y-%m-%d")
    proxy["ret_eqw"] = pd.to_numeric(proxy["ret_eqw"], errors="coerce").fillna(0.0)
    proxy = proxy.sort_values(["industry_id", "trade_date"])
    proxy["rs_5d"] = proxy.groupby("industry_id")["ret_eqw"].rolling(5, min_periods=1).sum().reset_index(level=0, drop=True) * 100.0
    proxy["rs_20d"] = proxy.groupby("industry_id")["ret_eqw"].rolling(20, min_periods=1).sum().reset_index(level=0, drop=True) * 100.0
    day = proxy[proxy["trade_date"].eq(trade_date)].copy()
    if day.empty:
        raise RuntimeError(f"cannot fallback-build cn_ga_mainline_radar_daily: no proxy rows for {trade_date}")
    day["up_ratio"] = pd.to_numeric(day["up_ratio"], errors="coerce").fillna(0.0)
    day["amount_total"] = pd.to_numeric(day["amount_total"], errors="coerce").fillna(0.0)
    day["member_count"] = pd.to_numeric(day["member_count"], errors="coerce").fillna(0).astype(int)
    day["amount_rank_score"] = day["amount_total"].rank(pct=True).fillna(0.0)
    day["mainline_score"] = (
        day["up_ratio"] * 45.0
        + day["amount_rank_score"] * 20.0
        + (pd.to_numeric(day["rs_20d"], errors="coerce").fillna(0.0) + 10.0).clip(0.0, 20.0)
    ).clip(0.0, 100.0)
    day["rank_no"] = day["mainline_score"].rank(ascending=False, method="first").astype(int)
    day["rs_rank"] = pd.to_numeric(day["rs_20d"], errors="coerce").fillna(0.0).rank(ascending=False, method="first").astype(int)

    def state(score: float, rs20: float, up_ratio: float) -> str:
        if score >= 60.0 and rs20 > 0 and up_ratio >= 0.55:
            return "CONFIRMED"
        if score >= 45.0 and rs20 > 0:
            return "FORMING"
        if score >= 30.0:
            return "ROTATING"
        if score >= 20.0:
            return "EARLY"
        return "FADE"

    rows = []
    for row in day.itertuples():
        score = float(row.mainline_score)
        rows.append(
            {
                "trade_date": trade_date,
                "mainline_id": row.industry_id,
                "mainline_name": row.industry_name,
                "member_count": int(row.member_count),
                "up_count": int(_f(row.up_count)),
                "down_count": int(_f(row.down_count)),
                "up_ratio": float(row.up_ratio),
                "avg_ret": _f(row.ret_eqw) * 100.0,
                "median_ret": None,
                "amount_sum": _f(row.amount_total),
                "amount_up_sum": None,
                "amount_chg_5d": None,
                "rs_5d": _f(row.rs_5d),
                "rs_20d": _f(row.rs_20d),
                "rs_rank": int(row.rs_rank),
                "leader_count": 0,
                "leader_score_avg": 0.0,
                "mainline_score": score,
                "mainline_state": state(score, _f(row.rs_20d), _f(row.up_ratio)),
                "rank_no": int(row.rank_no),
                "reason": f"fallback_from_local_proxy; score={score:.2f}; up_ratio={_f(row.up_ratio):.4f}; rs20={_f(row.rs_20d):.4f}",
                "leader_density": 0.0,
                "new_high_ratio": 0.0,
                "breakout_ratio": 0.0,
                "trend_alignment_score": _f(row.up_ratio),
                "strong_stock_count": int(_f(row.up_count)),
                "mainline_confidence": 0.6,
                "rotation_rank": int(row.rank_no),
                "mainline_phase": state(score, _f(row.rs_20d), _f(row.up_ratio)),
                "rs_60d": None,
                "rs_120d": None,
                "heat_percentile_5d": None,
            }
        )
    _execute("DELETE FROM cn_ga_mainline_radar_daily WHERE trade_date = :trade_date", {"trade_date": trade_date})
    sql = """
    INSERT INTO cn_ga_mainline_radar_daily (
        trade_date, mainline_id, mainline_name, member_count, up_count, down_count,
        up_ratio, avg_ret, median_ret, amount_sum, amount_up_sum, amount_chg_5d,
        rs_5d, rs_20d, rs_rank, leader_count, leader_score_avg, mainline_score,
        mainline_state, rank_no, reason, leader_density, new_high_ratio,
        breakout_ratio, trend_alignment_score, strong_stock_count, mainline_confidence,
        rotation_rank, mainline_phase, rs_60d, rs_120d, heat_percentile_5d
    ) VALUES (
        :trade_date, :mainline_id, :mainline_name, :member_count, :up_count, :down_count,
        :up_ratio, :avg_ret, :median_ret, :amount_sum, :amount_up_sum, :amount_chg_5d,
        :rs_5d, :rs_20d, :rs_rank, :leader_count, :leader_score_avg, :mainline_score,
        :mainline_state, :rank_no, :reason, :leader_density, :new_high_ratio,
        :breakout_ratio, :trend_alignment_score, :strong_stock_count, :mainline_confidence,
        :rotation_rank, :mainline_phase, :rs_60d, :rs_120d, :heat_percentile_5d
    )
    """
    return f"fallback_built:{_execute_many(sql, rows)}"


def _fallback_build_pulse(trade_date: str) -> str:
    radar = _query_df("SELECT * FROM cn_ga_mainline_radar_daily WHERE trade_date = :trade_date", {"trade_date": trade_date})
    if radar.empty:
        raise RuntimeError(f"cannot fallback-build cn_ga_market_pulse_daily: radar empty for {trade_date}")
    states = radar["mainline_state"].fillna("").astype(str).str.upper()
    bullish = float(states.isin(["CONFIRMED", "FORMING"]).mean())
    bearish = float(states.eq("FADE").mean())
    neutral = max(0.0, 1.0 - bullish - bearish)
    breadth = _f(pd.to_numeric(radar["up_ratio"], errors="coerce").mean())
    trend = _f(pd.to_numeric(radar["trend_alignment_score"], errors="coerce").mean(), breadth)
    amount_score = 50.0
    idx = _query_df(
        """
        SELECT index_code, chg_pct
        FROM cn_index_daily_price
        WHERE trade_date = :trade_date AND index_code IN ('sh000300', 'sz399001', 'sz399006', 'sh000688')
        """,
        {"trade_date": trade_date},
    )
    idx_map = {str(r.index_code): _f(r.chg_pct) for r in idx.itertuples()} if not idx.empty else {}
    hs300 = idx_map.get("sh000300", 0.0)
    sz = idx_map.get("sz399001", 0.0)
    cyb = idx_map.get("sz399006", idx_map.get("sh000688", 0.0))
    index_rs_score = max(0.0, min(100.0, 50.0 + (hs300 + sz + cyb) / 3.0 * 10.0))
    market_score = max(0.0, min(100.0, bullish * 45.0 + breadth * 25.0 + index_rs_score * 0.30))
    if bullish >= 0.5 and market_score >= 55.0:
        market_state, exposure, risk = "TREND_STRONG", 0.8, "LOW"
    elif bullish >= 0.3:
        market_state, exposure, risk = "TREND_WEAK", 0.6, "MEDIUM"
    elif bearish >= 0.5:
        market_state, exposure, risk = "RISK_OFF", 0.2, "HIGH"
    else:
        market_state, exposure, risk = "RANGE", 0.4, "MEDIUM"
    row = {
        "trade_date": trade_date,
        "market_score": market_score,
        "market_state": market_state,
        "target_exposure": exposure,
        "breadth_up_ratio": breadth,
        "breadth_down_ratio": 1.0 - breadth,
        "amount_score": amount_score,
        "index_rs_score": index_rs_score,
        "volatility_score": None,
        "hs300_pct_chg": hs300,
        "cyb_pct_chg": cyb,
        "sz_pct_chg": sz,
        "risk_flag": risk,
        "reason": f"fallback_from_radar; bullish={bullish:.4f}; bearish={bearish:.4f}; breadth={breadth:.4f}",
        "bullish_industry_ratio": bullish,
        "neutral_industry_ratio": neutral,
        "bearish_industry_ratio": bearish,
        "rotation_speed": 1.0 - neutral,
        "mainline_stability": max(bullish, bearish, neutral),
        "trend_alignment_avg": trend,
        "industry_expansion_breadth": breadth,
        "top_mainline_count": int((pd.to_numeric(radar["mainline_score"], errors="coerce") >= 60.0).sum()),
        "market_phase": market_state,
    }
    _execute("DELETE FROM cn_ga_market_pulse_daily WHERE trade_date = :trade_date", {"trade_date": trade_date})
    sql = """
    INSERT INTO cn_ga_market_pulse_daily (
        trade_date, market_score, market_state, target_exposure, breadth_up_ratio,
        breadth_down_ratio, amount_score, index_rs_score, volatility_score,
        hs300_pct_chg, cyb_pct_chg, sz_pct_chg, risk_flag, reason,
        bullish_industry_ratio, neutral_industry_ratio, bearish_industry_ratio,
        rotation_speed, mainline_stability, trend_alignment_avg,
        industry_expansion_breadth, top_mainline_count, market_phase
    ) VALUES (
        :trade_date, :market_score, :market_state, :target_exposure, :breadth_up_ratio,
        :breadth_down_ratio, :amount_score, :index_rs_score, :volatility_score,
        :hs300_pct_chg, :cyb_pct_chg, :sz_pct_chg, :risk_flag, :reason,
        :bullish_industry_ratio, :neutral_industry_ratio, :bearish_industry_ratio,
        :rotation_speed, :mainline_stability, :trend_alignment_avg,
        :industry_expansion_breadth, :top_mainline_count, :market_phase
    )
    """
    return f"fallback_built:{_execute_many(sql, [row])}"


def _fallback_build_lifecycle(trade_date: str) -> str:
    radar = _query_df("SELECT * FROM cn_ga_mainline_radar_daily WHERE trade_date = :trade_date", {"trade_date": trade_date})
    pulse = _query_df("SELECT market_state FROM cn_ga_market_pulse_daily WHERE trade_date = :trade_date LIMIT 1", {"trade_date": trade_date})
    if radar.empty:
        raise RuntimeError(f"cannot fallback-build cn_mainline_lifecycle_daily: radar empty for {trade_date}")
    market_state = str(pulse.iloc[0]["market_state"]) if not pulse.empty else "RANGE"
    rows = []
    max_amount = max(_f(pd.to_numeric(radar.get("amount_sum"), errors="coerce").max()), 1.0)
    for row in radar.itertuples():
        score = _f(row.mainline_score)
        if market_state == "RISK_OFF":
            state, risk, reason = "RISK_OFF", "MARKET_RISK_OFF", "market_risk_off"
        elif row.mainline_state in {"CONFIRMED", "FORMING"}:
            state, risk, reason = "TREND_EXPANSION", "NORMAL", "mainline_confirmed_or_forming"
        elif row.mainline_state == "ROTATING":
            state, risk, reason = "DIFFUSION", "WATCH", "mainline_rotating"
        elif row.mainline_state == "EARLY":
            state, risk, reason = "BOTTOM_REPAIR", "WATCH", "mainline_early"
        else:
            state, risk, reason = "TOP_DECAY", "MAINLINE_DECAY", "mainline_fade"
        rows.append(
            {
                "trade_date": trade_date,
                "mainline_id": row.mainline_id,
                "mainline_name": row.mainline_name,
                "mainline_strength": score / 100.0,
                "capital_concentration_score": _f(row.amount_sum) / max_amount,
                "trend_alignment_score": _f(row.trend_alignment_score),
                "breakout_ratio": _f(row.breakout_ratio),
                "new_high_ratio": _f(row.new_high_ratio),
                "leader_density": _f(row.leader_density),
                "rotation_rank": int(_f(row.rank_no, 999.0)),
                "lifecycle_state": state,
                "lifecycle_score": score / 100.0,
                "phase_reason": f"fallback_from_radar_pulse;{reason}",
                "risk_flag": risk,
            }
        )
    _execute("DELETE FROM cn_mainline_lifecycle_daily WHERE trade_date = :trade_date", {"trade_date": trade_date})
    sql = """
    INSERT INTO cn_mainline_lifecycle_daily (
        trade_date, mainline_id, mainline_name, mainline_strength,
        capital_concentration_score, trend_alignment_score, breakout_ratio,
        new_high_ratio, leader_density, rotation_rank, lifecycle_state,
        lifecycle_score, phase_reason, risk_flag
    ) VALUES (
        :trade_date, :mainline_id, :mainline_name, :mainline_strength,
        :capital_concentration_score, :trend_alignment_score, :breakout_ratio,
        :new_high_ratio, :leader_density, :rotation_rank, :lifecycle_state,
        :lifecycle_score, :phase_reason, :risk_flag
    )
    """
    return f"fallback_built:{_execute_many(sql, rows)}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Ensure latest daily market/mainline signal state tables are present.")
    parser.add_argument("--trade-date", default="", help="YYYY-MM-DD; default latest common stock/index trade date")
    parser.add_argument("--lookback-days", type=int, default=30, help="Builder window used for rolling dependencies")
    parser.add_argument("--force", action="store_true", help="Rebuild latest rows even if they exist")
    args = parser.parse_args()

    trade_date = args.trade_date or _latest_signal_trade_date()
    start_date = (_as_date(trade_date) - timedelta(days=max(1, args.lookback_days) - 1)).isoformat()
    db_args = _db_args()

    actions: dict[str, str] = {}
    actions["cn_local_industry_proxy_daily"] = _ensure_layer(
        "cn_local_industry_proxy_daily",
        trade_date,
        args.force,
        "scripts/build_local_industry_proxy_daily.py",
        ["--start", start_date, "--end", trade_date, "--chunk-days", "5"],
    )
    try:
        actions["cn_ga_stock_role_map_daily"] = _ensure_layer(
            "cn_ga_stock_role_map_daily",
            trade_date,
            args.force,
            "scripts/build_ga_stock_role_map_daily.py",
            ["--start", start_date, "--end", trade_date, "--replace", "--chunk-months", "1"] + db_args,
        )
    except RuntimeError as exc:
        actions["cn_ga_stock_role_map_daily"] = f"missing_non_blocking:{exc}"

    try:
        actions["cn_stock_mainline_strength_daily"] = _ensure_layer(
            "cn_stock_mainline_strength_daily",
            trade_date,
            args.force,
            "scripts/build_cn_stock_mainline_strength_daily.py",
            ["--start", start_date, "--end", trade_date, "--replace", "--chunk-months", "1"] + db_args,
        )
    except RuntimeError as exc:
        actions["cn_stock_mainline_strength_daily"] = f"missing_non_blocking:{exc}"

    try:
        actions["cn_ga_mainline_radar_daily"] = _ensure_layer(
            "cn_ga_mainline_radar_daily",
            trade_date,
            args.force,
            "scripts/build_ga_mainline_radar_daily.py",
            ["--start", start_date, "--end", trade_date, "--replace", "--chunk-months", "1"] + db_args,
        )
    except (RuntimeError, subprocess.CalledProcessError) as exc:
        actions["cn_ga_mainline_radar_daily"] = _fallback_build_radar(trade_date, start_date)

    try:
        actions["cn_ga_market_pulse_daily"] = _ensure_layer(
            "cn_ga_market_pulse_daily",
            trade_date,
            args.force,
            "scripts/build_ga_market_pulse_daily.py",
            ["--start", start_date, "--end", trade_date, "--replace", "--chunk-months", "1"] + db_args,
        )
    except (RuntimeError, subprocess.CalledProcessError) as exc:
        actions["cn_ga_market_pulse_daily"] = _fallback_build_pulse(trade_date)

    try:
        actions["cn_mainline_lifecycle_daily"] = _ensure_layer(
            "cn_mainline_lifecycle_daily",
            trade_date,
            args.force,
            "scripts/build_mainline_lifecycle_daily.py",
            ["--start", start_date, "--end", trade_date, "--replace", "--chunk-months", "1"] + db_args,
        )
    except (RuntimeError, subprocess.CalledProcessError) as exc:
        actions["cn_mainline_lifecycle_daily"] = _fallback_build_lifecycle(trade_date)

    counts = {table: _table_count(table, trade_date) for table in CORE_TABLES}
    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "trade_date": trade_date,
        "start_date": start_date,
        "actions": actions,
        "counts": counts,
    }
    out_dir = PROJECT_ROOT / "reports" / "daily_signal_guard"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "latest_daily_signal_guard_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
