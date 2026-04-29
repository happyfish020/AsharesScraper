from __future__ import annotations

import sys

from app.tools.backfill_sw_industry_history_from_tushare import main as _generic_main


def main() -> None:
    _generic_main(
        [
            "--level",
            "L3",
            "--member-source",
            "tushare_sw_l3",
            "--master-source",
            "TUSHARE_SW2021_L3",
            *sys.argv[1:],
        ]
    )


if __name__ == "__main__":
    main()
