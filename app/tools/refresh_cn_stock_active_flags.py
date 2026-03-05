from __future__ import annotations

import argparse

from sqlalchemy import text

from app.settings import build_engine


def main() -> None:
    p = argparse.ArgumentParser(description="Refresh cn_stock_universe_status_t active flags")
    p.add_argument("--asof", default=None, help="as-of date YYYY-MM-DD, default max(trade_date)")
    p.add_argument("--recent-days", type=int, default=30, help="lookback window days")
    p.add_argument("--min-trade-days", type=int, default=1, help="minimum distinct trade days in window")
    args = p.parse_args()

    eng = build_engine()
    with eng.begin() as conn:
        conn.execute(
            text(
                """
                CALL sp_refresh_stock_universe_status(
                    :asof_date,
                    :recent_days,
                    :min_trade_days
                )
                """
            ),
            {
                "asof_date": args.asof,
                "recent_days": int(args.recent_days),
                "min_trade_days": int(args.min_trade_days),
            },
        )

        row = conn.execute(
            text(
                """
                SELECT
                    SUM(CASE WHEN IFNULL(is_active,1)=1 THEN 1 ELSE 0 END) AS active_cnt,
                    SUM(CASE WHEN IFNULL(is_active,1)=0 THEN 1 ELSE 0 END) AS inactive_cnt,
                    MAX(updated_at) AS updated_at
                FROM cn_stock_universe_status_t
                """
            )
        ).mappings().one()

    print(
        f"refresh done: active={row['active_cnt']} inactive={row['inactive_cnt']} updated_at={row['updated_at']}"
    )


if __name__ == "__main__":
    main()
