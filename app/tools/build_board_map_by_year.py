from __future__ import annotations

import argparse
from datetime import date, datetime

from sqlalchemy import text

from app.settings import build_engine


def _parse_ymd(s: str) -> date:
    s = (s or "").strip()
    if len(s) == 10 and s[4] == "-" and s[7] == "-":
        return datetime.strptime(s, "%Y-%m-%d").date()
    return datetime.strptime(s, "%Y%m%d").date()


def main():
    p = argparse.ArgumentParser(description="Build board member map by year chunks")
    p.add_argument("--start", default="2000-01-01")
    p.add_argument("--end", default=date.today().strftime("%Y-%m-%d"))
    args = p.parse_args()

    d1 = _parse_ymd(args.start)
    d2 = _parse_ymd(args.end)
    if d1 > d2:
        raise SystemExit("invalid range")

    engine = build_engine()
    y = d1.year
    while y <= d2.year:
        ys = date(y, 1, 1)
        ye = date(y, 12, 31)
        if ys < d1:
            ys = d1
        if ye > d2:
            ye = d2
        with engine.begin() as conn:
            conn.execute(text("CALL sp_build_board_member_map(:d1,:d2)"), {"d1": ys, "d2": ye})
        print(f"built_map_year={y} range={ys}..{ye}")
        y += 1

    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT sector_type, COUNT(*) c
                FROM cn_board_member_map_d
                WHERE trade_date BETWEEN :d1 AND :d2
                GROUP BY sector_type
                ORDER BY sector_type
                """
            ),
            {"d1": d1, "d2": d2},
        ).fetchall()
        print("map_counts=", rows)


if __name__ == "__main__":
    main()

