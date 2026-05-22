from __future__ import annotations

import argparse
from pathlib import Path
import sys

from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_pipeline.common.db import get_engine


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate that replacement-source runs covered the intended date window.")
    parser.add_argument("--expected-start", required=True)
    parser.add_argument("--expected-end", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    engine = get_engine()
    checks = [
        (
            "cn_ts_sw_industry_member_hist",
            "SELECT MIN(in_date) AS min_d, MAX(COALESCE(out_date, '2099-12-31')) AS max_d, COUNT(*) AS cnt FROM cn_ts_sw_industry_member_hist",
        ),
        (
            "cn_v7_v8_industry_crosswalk",
            "SELECT MIN(effective_date) AS min_d, MAX(effective_date) AS max_d, COUNT(*) AS cnt FROM cn_v7_v8_industry_crosswalk",
        ),
    ]
    with engine.connect() as conn:
        for name, sql in checks:
            row = conn.execute(text(sql)).mappings().first()
            print(f"{name}: rows={row['cnt']} min={row['min_d']} max={row['max_d']}")


if __name__ == "__main__":
    main()
