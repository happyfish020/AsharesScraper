"""Build cn_mainline_lifecycle_daily from cn_mainline_strength_fact_daily.

P0H Lifecycle Rewire
--------------------
Hard rule: this script does NOT read cn_ga_mainline_radar_daily.
It consumes only the metadata-driven fact layer:
    cn_mainline_strength_fact_daily

Use after cn_mainline_strength_fact_daily has passed coverage audit.

Examples:
  python scripts/build_mainline_lifecycle_daily.py --start 2024-01-01 --end 2026-06-12 --dry-run --strict --db-password <pwd>
  python scripts/build_mainline_lifecycle_daily.py --start 2024-01-01 --end 2026-06-12 --replace --strict --db-password <pwd>
"""
from __future__ import annotations

import argparse
import os
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import URL, create_engine, text
from sqlalchemy.engine import Engine

FACT_TABLE = "cn_mainline_strength_fact_daily"
TARGET_TABLE = "cn_mainline_lifecycle_daily"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build cn_mainline_lifecycle_daily from metadata fact layer")
    p.add_argument("--start", default="2024-01-01")
    p.add_argument("--end", default="")
    p.add_argument("--db-name", default="cn_market_red")
    p.add_argument("--db-host", default="127.0.0.1")
    p.add_argument("--db-port", type=int, default=3306)
    p.add_argument("--db-user", default="cn_opr_red")
    p.add_argument("--db-password", default=None)
    p.add_argument("--replace", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--strict", action="store_true")
    p.add_argument("--expected-daily-mainlines", type=int, default=41)
    p.add_argument("--min-latest-mainlines", type=int, default=41)
    return p


def parse_date(v: str) -> date:
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


def ensure_schema(engine: Engine) -> None:
    ddl = f"""
    CREATE TABLE IF NOT EXISTS {TARGET_TABLE} (
        trade_date DATE NOT NULL,
        mainline_id VARCHAR(64) NOT NULL,
        mainline_name VARCHAR(128) NULL,
        source_layer VARCHAR(64) NULL,
        mainline_strength DOUBLE NULL,
        capital_concentration_score DOUBLE NULL,
        trend_alignment_score DOUBLE NULL,
        breadth_score DOUBLE NULL,
        breakout_ratio DOUBLE NULL,
        new_high_ratio DOUBLE NULL,
        leader_density DOUBLE NULL,
        rotation_rank INT NULL,
        lifecycle_state VARCHAR(64) NOT NULL DEFAULT 'UNKNOWN',
        lifecycle_score DOUBLE NULL,
        phase_reason VARCHAR(512) NULL,
        risk_flag VARCHAR(64) NULL,
        data_quality_flag VARCHAR(64) NULL,
        is_backtest_eligible TINYINT NULL,
        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        PRIMARY KEY (trade_date, mainline_id),
        KEY idx_cml_date_rank (trade_date, rotation_rank),
        KEY idx_cml_mainline_date (mainline_id, trade_date),
        KEY idx_cml_state_date (lifecycle_state, trade_date),
        KEY idx_cml_risk_date (risk_flag, trade_date)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """
    # If the table already exists from the legacy builder, add only the columns this
    # fact-based rewire needs. MySQL 8 supports ADD COLUMN IF NOT EXISTS; for older
    # versions we catch duplicate-column errors by querying INFORMATION_SCHEMA.
    with engine.begin() as conn:
        conn.execute(text(ddl))
        existing = conn.execute(text("""
            SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = :table
        """), {"table": TARGET_TABLE}).fetchall()
        cols = {r[0] for r in existing}
        alters = []
        specs = {
            "source_layer": "VARCHAR(64) NULL",
            "breadth_score": "DOUBLE NULL",
            "data_quality_flag": "VARCHAR(64) NULL",
            "is_backtest_eligible": "TINYINT NULL",
        }
        for col, spec in specs.items():
            if col not in cols:
                alters.append(f"ADD COLUMN `{col}` {spec}")
        if alters:
            conn.execute(text(f"ALTER TABLE {TARGET_TABLE} " + ", ".join(alters)))


def load_fact(engine: Engine, start: date, end: date) -> pd.DataFrame:
    sql = f"""
      SELECT trade_date, mainline_id, mainline_name, source_layer,
             mainline_strength_score, capital_score, trend_score, breadth_score,
             breakout_ratio, new_high_20d_count, active_member_count,
             leader_count, core_count, rank_no, rs_20d, rs_60d, rs_120d,
             ret_1d, ret_5d, ret_20d, ret_60d, ret_120d,
             amount_rank_pct, data_quality_flag, is_backtest_eligible
      FROM {FACT_TABLE}
      WHERE trade_date BETWEEN :start AND :end
      ORDER BY trade_date, rank_no, mainline_id
    """
    df = fetch_df(engine, sql, {"start": start, "end": end})
    if not df.empty:
        df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
    return df


def safe_float(v: Any, default: float = 0.0) -> float:
    if v is None or pd.isna(v):
        return default
    try:
        return float(v)
    except Exception:
        return default


def safe_ratio(num: Any, den: Any) -> float:
    den_f = safe_float(den)
    if den_f <= 0:
        return 0.0
    return safe_float(num) / den_f


def assign_lifecycle(row: pd.Series) -> tuple[str, float, str, str]:
    strength = safe_float(row.get("mainline_strength_score"))
    trend = safe_float(row.get("trend_score"))
    capital = safe_float(row.get("capital_score"))
    breadth = safe_float(row.get("breadth_score"))
    breakout = safe_float(row.get("breakout_ratio"))
    rs20 = safe_float(row.get("rs_20d"))
    rs60 = safe_float(row.get("rs_60d"))
    rs120 = safe_float(row.get("rs_120d"))
    ret5 = safe_float(row.get("ret_5d"))
    ret20 = safe_float(row.get("ret_20d"))
    rank_no = int(safe_float(row.get("rank_no"), 9999))
    active = safe_float(row.get("active_member_count"))
    leader_density = safe_ratio(row.get("leader_count"), active)
    quality = str(row.get("data_quality_flag") or "UNKNOWN")
    eligible = int(safe_float(row.get("is_backtest_eligible")))

    # Data quality gate: keep the row but do not invent a high-confidence phase.
    if quality == "LOW_MEMBER_COVERAGE" or active < 3:
        return "UNKNOWN", min(strength, 45.0), f"quality={quality}; active_members={active:g}; lifecycle confidence limited", "DATA_QUALITY_CHECK"
    if eligible != 1 and quality not in {"OK", "INSUFFICIENT_120D_HISTORY"}:
        return "UNKNOWN", min(strength, 45.0), f"quality={quality}; eligible={eligible}; fact-layer data not enough", "DATA_QUALITY_CHECK"

    # Top / decay first: protect downstream risk decisions.
    if strength < 35 or (rank_no > 30 and trend < 0.30 and breadth < 0.30):
        return "TOP_DECAY", strength, "weak fact strength with poor trend/breadth evidence", "RED"
    if ret20 < -0.08 and trend < 0.40 and rs60 < 0.40:
        return "TOP_DECAY", strength, "20d drawdown plus weak medium-term relative strength", "YELLOW"

    # Divergence / late expansion: strong headline but internal evidence no longer clean.
    if strength >= 62 and (breadth < 0.35 or leader_density >= 0.45 or ret5 < -0.04):
        return "DIVERGENCE", strength, "headline strength remains high but breadth/leader concentration/recent return is divergent", "YELLOW"
    if strength >= 68 and rank_no <= 10 and trend >= 0.65:
        return "LATE_EXPANSION", strength, "top-ranked strong trend; monitor heat and diffusion quality", "WATCH"

    # Clean expansion states.
    if strength >= 58 and rank_no <= 18 and trend >= 0.50 and capital >= 0.45 and breadth >= 0.38:
        return "TREND_EXPANSION", strength, "fact strength, trend, capital and breadth are aligned", "NONE"
    if strength >= 50 and (trend >= 0.45 or rs20 >= 0.55 or breakout >= 0.08):
        return "EARLY_EXPANSION", strength, "strength improving with early trend or breakout evidence", "NONE"

    # Repair and residual.
    if strength >= 38 and (rs20 >= 0.35 or trend >= 0.35 or ret20 > -0.04):
        return "BOTTOM_REPAIR", strength, "low-to-mid strength but repair evidence exists", "WATCH"
    return "UNKNOWN", strength, "fact evidence mixed; no lifecycle state confirmed", "WATCH"


def transform(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    out = df.copy()
    out["leader_density"] = out.apply(lambda r: safe_ratio(r.get("leader_count"), r.get("active_member_count")), axis=1)
    out["new_high_ratio"] = out.apply(lambda r: safe_ratio(r.get("new_high_20d_count"), r.get("active_member_count")), axis=1)
    assigned = out.apply(assign_lifecycle, axis=1, result_type="expand")
    assigned.columns = ["lifecycle_state", "lifecycle_score", "phase_reason", "risk_flag"]
    out = pd.concat([out, assigned], axis=1)
    out["mainline_strength"] = out["mainline_strength_score"]
    out["capital_concentration_score"] = out["capital_score"]
    out["trend_alignment_score"] = out["trend_score"]
    out["rotation_rank"] = out["rank_no"]
    cols = [
        "trade_date", "mainline_id", "mainline_name", "source_layer",
        "mainline_strength", "capital_concentration_score", "trend_alignment_score",
        "breadth_score", "breakout_ratio", "new_high_ratio", "leader_density",
        "rotation_rank", "lifecycle_state", "lifecycle_score", "phase_reason",
        "risk_flag", "data_quality_flag", "is_backtest_eligible",
    ]
    return out[cols].sort_values(["trade_date", "rotation_rank", "mainline_id"])


def write_rows(engine: Engine, df: pd.DataFrame, start: date, end: date, replace: bool) -> int:
    if df.empty:
        return 0
    rows = df.astype(object).where(pd.notna(df), None).to_dict(orient="records")
    insert_cols = list(df.columns)
    columns = ", ".join(f"`{c}`" for c in insert_cols)
    values = ", ".join(f":{c}" for c in insert_cols)
    update_cols = [c for c in insert_cols if c not in ("trade_date", "mainline_id")]
    update_sql = ", ".join(f"`{c}`=VALUES(`{c}`)" for c in update_cols)
    sql = text(f"""
        INSERT INTO {TARGET_TABLE} ({columns})
        VALUES ({values})
        ON DUPLICATE KEY UPDATE {update_sql}, updated_at=CURRENT_TIMESTAMP
    """)
    with engine.begin() as conn:
        if replace:
            conn.execute(text(f"DELETE FROM {TARGET_TABLE} WHERE trade_date BETWEEN :start AND :end"), {"start": start, "end": end})
        for i in range(0, len(rows), 1000):
            conn.execute(sql, rows[i:i + 1000])
    return len(rows)


def strict_checks(raw: pd.DataFrame, out: pd.DataFrame, expected: int, min_latest: int) -> list[str]:
    issues: list[str] = []
    if raw.empty:
        return ["fact source returned 0 rows"]
    latest = raw["trade_date"].max()
    latest_count = int(raw.loc[raw["trade_date"] == latest, "mainline_id"].nunique())
    if latest_count < min_latest:
        issues.append(f"latest mainlines {latest_count} < {min_latest}")
    daily = raw.groupby("trade_date")["mainline_id"].nunique()
    bad = daily[daily != expected]
    if not bad.empty:
        sample = ", ".join([f"{d}:{int(v)}" for d, v in bad.head(10).items()])
        issues.append(f"daily mainline count not equal {expected}; bad_days={len(bad)} sample={sample}")
    if len(out) != len(raw):
        issues.append(f"output rows {len(out)} != fact rows {len(raw)}")
    return issues


def main() -> None:
    args = build_parser().parse_args()
    if args.replace and args.dry_run:
        raise SystemExit("Use only one of --replace or --dry-run")
    start = parse_date(args.start)
    end = parse_date(args.end) if args.end else datetime.today().date()
    password = args.db_password if args.db_password is not None else (os.getenv("ASHARE_MYSQL_PASSWORD") or os.getenv("MYSQL_PASSWORD") or "")
    engine = build_engine(args.db_host, args.db_port, args.db_user, password, args.db_name)
    ensure_schema(engine)
    raw = load_fact(engine, start, end)
    out = transform(raw)

    latest = None if raw.empty else raw["trade_date"].max()
    latest_count = 0 if raw.empty else int(raw.loc[raw["trade_date"] == latest, "mainline_id"].nunique())
    print("[FACT LIFECYCLE]", {
        "source_table": FACT_TABLE,
        "target_table": TARGET_TABLE,
        "loaded_fact_rows": len(raw),
        "output_rows": len(out),
        "latest_trade_date": latest,
        "latest_mainlines": latest_count,
    })
    if not out.empty:
        print("[FACT LIFECYCLE STATE]")
        print(out["lifecycle_state"].value_counts().to_string())
        latest_top = out[out["trade_date"] == latest].sort_values("rotation_rank").head(20)
        print("[FACT LIFECYCLE LATEST TOP]")
        print(latest_top[["trade_date", "rotation_rank", "mainline_id", "mainline_name", "source_layer", "mainline_strength", "lifecycle_state", "risk_flag"]].to_string(index=False))

    issues = strict_checks(raw, out, args.expected_daily_mainlines, args.min_latest_mainlines) if args.strict else []
    if issues:
        print("[FACT LIFECYCLE FAILED]")
        for issue in issues:
            print(f"- {issue}")
        raise SystemExit(1)
    if args.dry_run:
        print("[FACT LIFECYCLE DRY-RUN] no write")
        return
    written = write_rows(engine, out, start, end, replace=args.replace)
    print("[FACT LIFECYCLE WROTE]", {"rows": written, "replace": bool(args.replace)})


if __name__ == "__main__":
    main()
