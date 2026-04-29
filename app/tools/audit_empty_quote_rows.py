from __future__ import annotations

import argparse

from sqlalchemy import text

from app.settings import build_engine


def _print_section(title: str) -> None:
    print(f"--- {title} ---")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Audit fully empty quote rows in cn_stock_daily_price and cn_index_daily_price"
    )
    p.add_argument(
        "--fail-on-found",
        action="store_true",
        help="Exit with code 1 when any fully empty quote rows are found",
    )
    args = p.parse_args()

    eng = build_engine()
    with eng.connect() as conn:
        stock_summary = conn.execute(
            text(
                """
                SELECT
                    COUNT(*) AS n_rows,
                    COUNT(DISTINCT symbol) AS n_codes,
                    MIN(trade_date) AS min_d,
                    MAX(trade_date) AS max_d
                FROM cn_stock_daily_price
                WHERE open IS NULL AND high IS NULL AND low IS NULL AND close IS NULL
                  AND pre_close IS NULL AND volume IS NULL AND amount IS NULL AND chg_pct IS NULL
                """
            )
        ).mappings().one()

        index_summary = conn.execute(
            text(
                """
                SELECT
                    COUNT(*) AS n_rows,
                    COUNT(DISTINCT index_code) AS n_codes,
                    MIN(trade_date) AS min_d,
                    MAX(trade_date) AS max_d
                FROM cn_index_daily_price
                WHERE open IS NULL AND high IS NULL AND low IS NULL AND close IS NULL
                  AND pre_close IS NULL AND volume IS NULL AND amount IS NULL AND chg_pct IS NULL
                """
            )
        ).mappings().one()

        stock_by_source = conn.execute(
            text(
                """
                SELECT source, COUNT(*) AS n_rows, COUNT(DISTINCT symbol) AS n_codes
                FROM cn_stock_daily_price
                WHERE open IS NULL AND high IS NULL AND low IS NULL AND close IS NULL
                  AND pre_close IS NULL AND volume IS NULL AND amount IS NULL AND chg_pct IS NULL
                GROUP BY source
                ORDER BY n_rows DESC, n_codes DESC
                """
            )
        ).mappings().all()

        index_by_source = conn.execute(
            text(
                """
                SELECT source, COUNT(*) AS n_rows, COUNT(DISTINCT index_code) AS n_codes
                FROM cn_index_daily_price
                WHERE open IS NULL AND high IS NULL AND low IS NULL AND close IS NULL
                  AND pre_close IS NULL AND volume IS NULL AND amount IS NULL AND chg_pct IS NULL
                GROUP BY source
                ORDER BY n_rows DESC, n_codes DESC
                """
            )
        ).mappings().all()

    _print_section("stock_summary")
    print(
        f"rows={stock_summary['n_rows']} codes={stock_summary['n_codes']} "
        f"min_d={stock_summary['min_d']} max_d={stock_summary['max_d']}"
    )
    _print_section("stock_by_source")
    if stock_by_source:
        for row in stock_by_source:
            print(f"source={row['source']} rows={row['n_rows']} codes={row['n_codes']}")
    else:
        print("none")

    _print_section("index_summary")
    print(
        f"rows={index_summary['n_rows']} codes={index_summary['n_codes']} "
        f"min_d={index_summary['min_d']} max_d={index_summary['max_d']}"
    )
    _print_section("index_by_source")
    if index_by_source:
        for row in index_by_source:
            print(f"source={row['source']} rows={row['n_rows']} codes={row['n_codes']}")
    else:
        print("none")

    total_found = int(stock_summary["n_rows"] or 0) + int(index_summary["n_rows"] or 0)
    print(f"total_fully_empty_quote_rows={total_found}")
    if args.fail_on_found and total_found > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
