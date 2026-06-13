"""Build cn_mainline_lifecycle_daily from clean cn_mainline_strength_fact_daily.

This is the P0H safe rewire path. It does NOT read cn_ga_mainline_radar_daily.
The legacy build_mainline_lifecycle_daily.py is left intact; wire this script in
with V8_USE_FACT_LIFECYCLE=1 after Fact Layer audit passes.
"""
from __future__ import annotations

import argparse
import os
from datetime import date, datetime
from typing import Any

import pandas as pd
from sqlalchemy import URL, create_engine, text
from sqlalchemy.engine import Engine


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build cn_mainline_lifecycle_daily from cn_mainline_strength_fact_daily")
    p.add_argument("--start", default="2024-01-01")
    p.add_argument("--end", default="")
    p.add_argument("--db-name", default="cn_market_red")
    p.add_argument("--db-host", default="127.0.0.1")
    p.add_argument("--db-port", type=int, default=3306)
    p.add_argument("--db-user", default="cn_opr_red")
    p.add_argument("--db-password", default=None)
    p.add_argument("--replace", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p


def _parse_date(v: str) -> date:
    return datetime.strptime(v, "%Y-%m-%d" if "-" in v else "%Y%m%d").date()


def build_engine(host: str, port: int, user: str, password: str, db: str) -> Engine:
    url = URL.create(
        "mysql+pymysql",
        username=user,
        password=password,
        host=host,
        port=port,
        database=db,
        query={"charset": "utf8mb4"},
    )
    return create_engine(url, pool_pre_ping=True, future=True)


def fetch_df(engine: Engine, sql: str, params: dict[str, Any] | None = None) -> pd.DataFrame:
    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn, params=params or {})


def ensure_lifecycle_table(engine: Engine) -> None:
    ddl = """
    CREATE TABLE IF NOT EXISTS cn_mainline_lifecycle_daily (
        trade_date DATE NOT NULL,
        mainline_id VARCHAR(64) NOT NULL,
        mainline_name VARCHAR(128) NULL,
        mainline_strength DOUBLE NULL,
        capital_concentration_score DOUBLE NULL,
        trend_alignment_score DOUBLE NULL,
        breakout_ratio DOUBLE NULL,
        new_high_ratio DOUBLE NULL,
        leader_density DOUBLE NULL,
        rotation_rank INT NULL,
        lifecycle_state VARCHAR(64) NOT NULL DEFAULT 'UNKNOWN',
        lifecycle_score DOUBLE NULL,
        phase_reason VARCHAR(512) NULL,
        risk_flag VARCHAR(64) NULL,
        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        PRIMARY KEY (trade_date, mainline_id),
        KEY idx_mainline_lifecycle_daily_date (trade_date),
        KEY idx_mainline_lifecycle_daily_state (lifecycle_state, trade_date),
        KEY idx_mainline_lifecycle_daily_rank (rotation_rank, trade_date),
        KEY idx_mainline_lifecycle_daily_strength (mainline_strength, trade_date)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """
    with engine.begin() as conn:
        conn.execute(text(ddl))


def load_fact(engine: Engine, start: date, end: date) -> pd.DataFrame:
    sql = """
      SELECT trade_date, mainline_id, mainline_name,
             mainline_strength_score, capital_score, trend_score,
             breakout_ratio, new_high_20d_count, active_member_count,
             leader_count, core_count, rank_no, breadth_score,
             rs_20d, rs_60d, rs_120d, ret_20d, amount_rank_pct,
             data_quality_flag, is_backtest_eligible
      FROM cn_mainline_strength_fact_daily
      WHERE trade_date BETWEEN :start AND :end
    """
    return fetch_df(engine, sql, {"start": start, "end": end})


def _safe_ratio(num: float, den: float) -> float:
    if den is None or den <= 0:
        return 0.0
    return float(num or 0.0) / float(den)


def assign_lifecycle(row: pd.Series) -> tuple[str, float, str, str]:
    strength = float(row.get("mainline_strength_score") or 0.0)
    trend = float(row.get("trend_score") or 0.0)
    capital = float(row.get("capital_score") or 0.0)
    breadth = float(row.get("breadth_score") or 0.0)
    breakout = float(row.get("breakout_ratio") or 0.0)
    rs20 = float(row.get("rs_20d") or 0.0)
    rs60 = row.get("rs_60d")
    rank_no = int(row.get("rank_no") or 9999)
    active = float(row.get("active_member_count") or 0.0)
    leader_count = float(row.get("leader_count") or 0.0)
    leader_density = _safe_ratio(leader_count, active)
    quality = str(row.get("data_quality_flag") or "UNKNOWN")
    eligible = int(row.get("is_backtest_eligible") or 0)

    if eligible != 1 or quality not in {"OK", "INSUFFICIENT_120D_HISTORY"}:
        return "UNKNOWN", min(strength, 40.0), f"quality={quality}; eligible={eligible}; fact-layer data not enough", "DATA_QUALITY_CHECK"

    rs60_val = None if pd.isna(rs60) else float(rs60)
    if strength >= 72 and rank_no <= 8 and trend >= 60 and capital >= 55 and breadth >= 50:
        return "TREND_EXPANSION", strength, "fact strength/top-rank/trend/capital/breadth aligned", "NONE"
    if strength >= 65 and rank_no <= 12 and breadth >= 60 and leader_density < 0.35:
        return "DIFFUSION", strength, "strong fact score with broader participation and non-extreme leader density", "NONE"
    if strength >= 62 and (leader_density >= 0.35 or (trend >= 65 and breadth < 45)):
        return "DIVERGENCE", strength, "strength remains high but internal breadth/leader density is divergent", "YELLOW"
    if strength < 45 and rs20 < 0 and (rs60_val is not None and rs60_val < 0):
        return "TOP_DECAY", strength, "weak fact score with negative medium-term relative strength", "YELLOW"
    if 45 <= strength < 62 and (rs20 > 0 or trend >= 50 or breakout > 0.08):
        return "BOTTOM_REPAIR", strength, "repairing from lower score with improving trend or breakout evidence", "NONE"
    if strength < 35:
        return "RISK_OFF", strength, "very weak fact-layer strength", "RED"
    return "UNKNOWN", strength, "fact evidence mixed; no lifecycle state confirmed", "WATCH"


def transform(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["leader_density"] = out.apply(lambda r: _safe_ratio(r.get("leader_count"), r.get("active_member_count")), axis=1)
    out["new_high_ratio"] = out.apply(lambda r: _safe_ratio(r.get("new_high_20d_count"), r.get("active_member_count")), axis=1)
    assigned = out.apply(assign_lifecycle, axis=1, result_type="expand")
    assigned.columns = ["lifecycle_state", "lifecycle_score", "phase_reason", "risk_flag"]
    out = pd.concat([out, assigned], axis=1)
    out["mainline_strength"] = out["mainline_strength_score"]
    out["capital_concentration_score"] = out["capital_score"]
    out["trend_alignment_score"] = out["trend_score"]
    out["rotation_rank"] = out["rank_no"]
    cols = [
        "trade_date", "mainline_id", "mainline_name", "mainline_strength",
        "capital_concentration_score", "trend_alignment_score", "breakout_ratio",
        "new_high_ratio", "leader_density", "rotation_rank", "lifecycle_state",
        "lifecycle_score", "phase_reason", "risk_flag",
    ]
    return out[cols]


def write_rows(engine: Engine, df: pd.DataFrame, start: date, end: date, replace: bool) -> int:
    if df.empty:
        return 0
    rows = df.astype(object).where(pd.notna(df), None).to_dict(orient="records")
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
    ON DUPLICATE KEY UPDATE
        mainline_name = VALUES(mainline_name),
        mainline_strength = VALUES(mainline_strength),
        capital_concentration_score = VALUES(capital_concentration_score),
        trend_alignment_score = VALUES(trend_alignment_score),
        breakout_ratio = VALUES(breakout_ratio),
        new_high_ratio = VALUES(new_high_ratio),
        leader_density = VALUES(leader_density),
        rotation_rank = VALUES(rotation_rank),
        lifecycle_state = VALUES(lifecycle_state),
        lifecycle_score = VALUES(lifecycle_score),
        phase_reason = VALUES(phase_reason),
        risk_flag = VALUES(risk_flag),
        updated_at = CURRENT_TIMESTAMP
    """
    with engine.begin() as conn:
        if replace:
            conn.execute(text("DELETE FROM cn_mainline_lifecycle_daily WHERE trade_date BETWEEN :start AND :end"), {"start": start, "end": end})
        batch_size = 4000
        total = 0
        for i in range(0, len(rows), batch_size):
            batch = rows[i : i + batch_size]
            conn.execute(text(sql), batch)
            total += len(batch)
    return total


def main() -> None:
    args = build_parser().parse_args()
    start = _parse_date(args.start)
    end = _parse_date(args.end) if args.end else date.today()
    password = args.db_password or os.getenv("ASHARE_MYSQL_PASSWORD", "sec_Bobo123")
    engine = build_engine(args.db_host, args.db_port, args.db_user, password, args.db_name)
    ensure_lifecycle_table(engine)
    raw = load_fact(engine, start, end)
    out = transform(raw)
    print(f"[FACT LIFECYCLE] loaded_fact_rows={len(raw)} output_rows={len(out)} range={start}~{end}")
    if not out.empty:
        print("[FACT LIFECYCLE STATE]\n" + out["lifecycle_state"].value_counts().to_string())
        print("[FACT LIFECYCLE LATEST TOP]\n" + out.sort_values(["trade_date", "rotation_rank"]).tail(20).to_string(index=False))
    if args.dry_run:
        print("[FACT LIFECYCLE DRY-RUN] no write")
        return
    written = write_rows(engine, out, start, end, args.replace)
    print(f"[FACT LIFECYCLE WROTE] rows={written}")


if __name__ == "__main__":
    main()
