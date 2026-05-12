from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_pipeline.builders.industry_capital_flow import run as run_industry_capital_flow
from data_pipeline.builders.industry_proxy_daily import run as run_industry_proxy_daily
from data_pipeline.builders.mainline_strength_daily import run as run_mainline_strength_daily
from data_pipeline.builders.stock_fundamental_daily import run as run_stock_fundamental_daily
from data_pipeline.builders.sw_industry_master import run as run_sw_industry_master
from data_pipeline.builders.sw_industry_member_hist import run as run_sw_industry_member_hist


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified GA_MAINLINE_DATA_BACKFILL_SYSTEM entry.")
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--end", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--chunk-months", type=int, default=1)
    parser.add_argument("--token", default="")
    parser.add_argument("--config", default="")
    parser.add_argument("--src", default="SW2021")
    parser.add_argument("--industry-level", default="L1", choices=["L1", "L2", "L3"])
    parser.add_argument(
        "--steps",
        default="industry_master,industry_member_hist,industry_proxy_daily,industry_capital_flow,cn_stock_fundamental_daily,cn_mainline_strength_daily",
        help="Comma-separated pipeline steps",
    )
    return parser


def _shared_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        start=args.start,
        end=args.end,
        resume=args.resume,
        workers=args.workers,
        chunk_months=args.chunk_months,
        token=args.token,
        config=args.config,
    )


def main() -> None:
    args = build_parser().parse_args()
    steps = [item.strip() for item in str(args.steps).split(",") if item.strip()]
    print('[SOURCE AUDIT] source data coverage audits are executed inside each selected pipeline step before processing.', flush=True)

    if "industry_master" in steps:
        run_sw_industry_master(argparse.Namespace(**vars(_shared_args(args)), src=args.src))
    if "industry_member_hist" in steps:
        run_sw_industry_member_hist(
            argparse.Namespace(**vars(_shared_args(args)), src=args.src, industry_level=args.industry_level)
        )
    if "industry_proxy_daily" in steps:
        run_industry_proxy_daily(argparse.Namespace(**vars(_shared_args(args)), industry_level=args.industry_level))
    if "industry_capital_flow" in steps:
        run_industry_capital_flow(argparse.Namespace(**vars(_shared_args(args)), industry_level=args.industry_level))
    if "cn_stock_fundamental_daily" in steps:
        run_stock_fundamental_daily(_shared_args(args))
    if "cn_mainline_strength_daily" in steps:
        run_mainline_strength_daily(argparse.Namespace(**vars(_shared_args(args)), industry_level=args.industry_level))


if __name__ == "__main__":
    main()
