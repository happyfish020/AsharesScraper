from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sqlalchemy import inspect

from app.settings import build_engine
from app.tools.sync_cn_gcrl_public_fund_from_tushare import ensure_tables

REQUIRED_TABLES = [
    "cn_gcrl_institution_registry",
    "cn_gcrl_fund_registry",
    "cn_gcrl_position_snapshot",
    "cn_gcrl_position_change",
    "cn_gcrl_data_freshness",
    "cn_gcrl_data_source_status",
]

FORBIDDEN_ANALYSIS_TABLE_FRAGMENTS = ["theme", "score", "signal", "buy", "sell"]


def main() -> int:
    engine = build_engine()
    ensure_tables(engine)
    inspector = inspect(engine)
    existing = set(inspector.get_table_names())
    missing = [t for t in REQUIRED_TABLES if t not in existing]
    if missing:
        raise SystemExit(f"FAIL missing tables: {missing}")
    bad = [t for t in REQUIRED_TABLES if not t.startswith("cn_")]
    if bad:
        raise SystemExit(f"FAIL non-cn prefix tables: {bad}")
    print("GCRL CN public fund schema audit PASS")
    for t in REQUIRED_TABLES:
        print(f"  - {t}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
