"""Build/seed canonical mainline registry and source-code map.

Purpose
-------
External data sources use unstable IDs (SW L1 801xxx.SI, SW L3 85xxxx.SI,
EastMoney BKxxxx, custom theme codes). GrowthAlpha should consume only the
system canonical MAINLINE_ID. This script creates two mapping tables and seeds
SW-L1 mappings from cn_local_industry_proxy_daily.

It does not read cn_ga_mainline_radar_daily.

Usage:
  python scripts/build_cn_canonical_mainline_registry.py --start 2024-01-01 --end 2026-06-12 --replace-sw-l1
"""
from __future__ import annotations

import argparse
import os
from datetime import date, datetime
from typing import Any

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, URL


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Seed canonical mainline registry / source code map")
    p.add_argument("--start", default="2024-01-01")
    p.add_argument("--end", default="")
    p.add_argument("--db-name", default="cn_market_red")
    p.add_argument("--db-host", default="127.0.0.1")
    p.add_argument("--db-port", type=int, default=3306)
    p.add_argument("--db-user", default="cn_opr_red")
    p.add_argument("--db-password", default=None)
    p.add_argument("--replace-sw-l1", action="store_true", help="Refresh active SW-L1 mappings discovered from proxy table")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--strict", action="store_true")
    return p


def _parse_date(v: str) -> date:
    return datetime.strptime(v, "%Y-%m-%d" if "-" in v else "%Y%m%d").date()


def build_engine(host: str, port: int, user: str, password: str, db: str) -> Engine:
    url = URL.create("mysql+pymysql", username=user, password=password, host=host, port=port, database=db, query={"charset": "utf8mb4"})
    return create_engine(url, pool_pre_ping=True, future=True)


def fetch_df(engine: Engine, sql: str, params: dict[str, Any] | None = None) -> pd.DataFrame:
    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn, params=params or {})


def ensure_schema(engine: Engine) -> None:
    ddl_registry = """
    CREATE TABLE IF NOT EXISTS cn_canonical_mainline_registry (
        canonical_mainline_id VARCHAR(64) NOT NULL,
        canonical_mainline_name VARCHAR(128) NOT NULL,
        canonical_level VARCHAR(32) NOT NULL DEFAULT 'L1',
        canonical_family VARCHAR(64) NULL,
        description VARCHAR(512) NULL,
        is_active TINYINT NOT NULL DEFAULT 1,
        effective_start_date DATE NOT NULL DEFAULT '2000-01-01',
        effective_end_date DATE NULL,
        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        PRIMARY KEY (canonical_mainline_id),
        KEY idx_ccmr_active_level (is_active, canonical_level)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """
    ddl_map = """
    CREATE TABLE IF NOT EXISTS cn_source_mainline_code_map (
        source_system VARCHAR(64) NOT NULL,
        source_code VARCHAR(64) NOT NULL,
        source_name VARCHAR(128) NULL,
        source_level VARCHAR(32) NULL,
        canonical_mainline_id VARCHAR(64) NOT NULL,
        canonical_mainline_name VARCHAR(128) NOT NULL,
        canonical_level VARCHAR(32) NOT NULL DEFAULT 'L1',
        is_primary_mapping TINYINT NOT NULL DEFAULT 1,
        mapping_confidence DOUBLE NOT NULL DEFAULT 1.0,
        mapping_rule VARCHAR(128) NOT NULL DEFAULT 'AUTO_SEEDED',
        effective_start_date DATE NOT NULL DEFAULT '2000-01-01',
        effective_end_date DATE NULL,
        is_active TINYINT NOT NULL DEFAULT 1,
        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        PRIMARY KEY (source_system, source_code, effective_start_date),
        KEY idx_csmcm_canonical (canonical_mainline_id, is_active),
        KEY idx_csmcm_source_active (source_system, source_level, is_active)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """
    with engine.begin() as conn:
        conn.execute(text(ddl_registry))
        conn.execute(text(ddl_map))


def _canonical_id_for_sw_l1(source_code: str) -> str:
    digits = str(source_code).split(".")[0]
    return f"ML_CN_SWL1_{digits}"


def load_sw_l1_sources(engine: Engine, start: date, end: date) -> pd.DataFrame:
    sql = """
      SELECT industry_id AS source_code,
             MAX(COALESCE(industry_name, industry_id)) AS source_name,
             MAX(COALESCE(industry_level, 'L1')) AS source_level,
             COUNT(DISTINCT trade_date) AS trade_days
      FROM cn_local_industry_proxy_daily
      WHERE trade_date BETWEEN :start AND :end
        AND (industry_level = 'L1' OR industry_id REGEXP '^801[0-9]{3}\\.SI$')
      GROUP BY industry_id
      ORDER BY industry_id
    """
    return fetch_df(engine, sql, {"start": start, "end": end})


def build_seed_rows(src: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if src.empty:
        return pd.DataFrame(), pd.DataFrame()
    src = src.copy()
    src["canonical_mainline_id"] = src["source_code"].astype(str).map(_canonical_id_for_sw_l1)
    src["canonical_mainline_name"] = src["source_name"].fillna(src["source_code"]).astype(str)
    reg = src[["canonical_mainline_id", "canonical_mainline_name"]].drop_duplicates().copy()
    reg["canonical_level"] = "L1"
    reg["canonical_family"] = "SW_L1"
    reg["description"] = "Auto-seeded canonical mainline from SW L1 proxy. GrowthAlpha consumers must use this canonical ID, not the raw source code."
    reg["is_active"] = 1
    map_df = src[["source_code", "source_name", "source_level", "canonical_mainline_id", "canonical_mainline_name"]].copy()
    map_df.insert(0, "source_system", "SW")
    map_df["canonical_level"] = "L1"
    map_df["is_primary_mapping"] = 1
    map_df["mapping_confidence"] = 1.0
    map_df["mapping_rule"] = "AUTO_SW_L1_PROXY"
    map_df["is_active"] = 1
    return reg, map_df


def write_rows(engine: Engine, reg: pd.DataFrame, map_df: pd.DataFrame, replace_sw_l1: bool) -> tuple[int, int]:
    with engine.begin() as conn:
        if replace_sw_l1:
            conn.execute(text("DELETE FROM cn_source_mainline_code_map WHERE source_system='SW' AND source_level='L1' AND mapping_rule='AUTO_SW_L1_PROXY'"))
            conn.execute(text("DELETE FROM cn_canonical_mainline_registry WHERE canonical_family='SW_L1' AND canonical_mainline_id LIKE 'ML_CN_SWL1_%'"))
        if not reg.empty:
            rows = reg.astype(object).where(pd.notna(reg), None).to_dict("records")
            conn.execute(text("""
                INSERT INTO cn_canonical_mainline_registry
                    (canonical_mainline_id, canonical_mainline_name, canonical_level, canonical_family, description, is_active)
                VALUES (:canonical_mainline_id, :canonical_mainline_name, :canonical_level, :canonical_family, :description, :is_active)
                ON DUPLICATE KEY UPDATE
                    canonical_mainline_name=VALUES(canonical_mainline_name),
                    canonical_level=VALUES(canonical_level),
                    canonical_family=VALUES(canonical_family),
                    description=VALUES(description),
                    is_active=VALUES(is_active)
            """), rows)
        if not map_df.empty:
            rows = map_df.astype(object).where(pd.notna(map_df), None).to_dict("records")
            conn.execute(text("""
                INSERT INTO cn_source_mainline_code_map
                    (source_system, source_code, source_name, source_level,
                     canonical_mainline_id, canonical_mainline_name, canonical_level,
                     is_primary_mapping, mapping_confidence, mapping_rule, is_active)
                VALUES (:source_system, :source_code, :source_name, :source_level,
                        :canonical_mainline_id, :canonical_mainline_name, :canonical_level,
                        :is_primary_mapping, :mapping_confidence, :mapping_rule, :is_active)
                ON DUPLICATE KEY UPDATE
                    source_name=VALUES(source_name), source_level=VALUES(source_level),
                    canonical_mainline_id=VALUES(canonical_mainline_id),
                    canonical_mainline_name=VALUES(canonical_mainline_name),
                    canonical_level=VALUES(canonical_level),
                    is_primary_mapping=VALUES(is_primary_mapping),
                    mapping_confidence=VALUES(mapping_confidence),
                    mapping_rule=VALUES(mapping_rule),
                    is_active=VALUES(is_active)
            """), rows)
    return len(reg), len(map_df)


def main() -> None:
    args = build_parser().parse_args()
    start = _parse_date(args.start)
    end = _parse_date(args.end) if args.end else datetime.today().date()
    password = args.db_password if args.db_password is not None else os.getenv("MYSQL_PASSWORD", "")
    engine = build_engine(args.db_host, args.db_port, args.db_user, password, args.db_name)
    ensure_schema(engine)
    src = load_sw_l1_sources(engine, start, end)
    reg, map_df = build_seed_rows(src)
    print(f"[CANONICAL REGISTRY] discovered_sw_l1={len(src)} registry_rows={len(reg)} map_rows={len(map_df)} range={start}~{end}")
    if args.strict and len(map_df) < 25:
        raise SystemExit(f"CANONICAL_SW_L1_TOO_FEW rows={len(map_df)} expected>=25")
    if args.dry_run:
        print(map_df.head(40).to_string(index=False) if not map_df.empty else "[CANONICAL REGISTRY] no rows")
        print("[CANONICAL REGISTRY DRY-RUN] no write")
        return
    r, m = write_rows(engine, reg, map_df, args.replace_sw_l1)
    print(f"[CANONICAL REGISTRY WROTE] registry={r} source_map={m}")


if __name__ == "__main__":
    main()
