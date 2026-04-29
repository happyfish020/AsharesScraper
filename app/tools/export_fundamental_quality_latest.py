from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

from app.settings import build_engine


def main() -> None:
    parser = argparse.ArgumentParser(description="Export latest fundamental quality picks to CSV.")
    parser.add_argument("--mode", choices=["core", "margin", "all"], default="core")
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--output", default="", help="Optional output CSV path")
    parser.add_argument("--min-revenue-growth", type=float, default=None, help="Optional extra runtime filter")
    args = parser.parse_args()

    engine = build_engine()
    where_clause = "1=1"
    if args.mode == "core":
        where_clause = "quality_pass_core = 1"
    elif args.mode == "margin":
        where_clause = "quality_pass_with_margin = 1"

    extra_clause = ""
    params: dict[str, object] = {"limit_n": max(1, int(args.limit))}
    if args.min_revenue_growth is not None:
        extra_clause = " AND revenue_growth_pct >= :min_revenue_growth"
        params["min_revenue_growth"] = float(args.min_revenue_growth)

    sql = f"""
        SELECT
            parameter_set,
            symbol,
            basic_trade_date,
            month_key,
            fina_end_date,
            ann_date,
            eps,
            revenue_growth_pct,
            debt_to_eqt,
            grossprofit_margin,
            total_mv,
            pe_ttm,
            pb,
            quality_core_score,
            quality_total_score,
            quality_pass_core,
            quality_pass_with_margin
        FROM cn_stock_fundamental_quality_v1
        WHERE {where_clause}
        {extra_clause}
        ORDER BY quality_total_score DESC, revenue_growth_pct DESC, grossprofit_margin DESC, total_mv DESC
        LIMIT :limit_n
    """

    df = pd.read_sql(sql, engine, params=params)
    out_dir = Path("audit_reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    output = Path(args.output) if str(args.output).strip() else out_dir / f"fundamental_quality_latest_{args.mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(output, index=False, encoding="utf-8-sig")
    print(f"rows={len(df)} output={output}")


if __name__ == "__main__":
    main()
