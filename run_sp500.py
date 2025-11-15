"""Runner script for executing the S&P 500 agentic workflow."""

from __future__ import annotations

import argparse
import sys
from typing import Optional

from agentic_quant import run_sp500_workflow


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Execute the agentic quant workflow configured for the current "
            "S&P 500 universe fetched from Yahoo Finance."
        )
    )
    parser.add_argument(
        "--max-tickers",
        type=int,
        default=None,
        help=(
            "Limit the number of S&P 500 constituents fetched. Useful for "
            "prototyping or reducing runtime."
        ),
    )
    parser.add_argument(
        "--period",
        type=str,
        default="5y",
        help="Historical window to request from Yahoo Finance (e.g., '1y', 'max').",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Explicit start date (YYYY-MM-DD) to override the period argument.",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="Explicit end date (YYYY-MM-DD) for price history.",
    )
    parser.add_argument(
        "--interval",
        type=str,
        default="1d",
        help="Sampling frequency for prices (e.g., '1d', '1wk', '1mo').",
    )
    parser.add_argument(
        "--min-history",
        type=int,
        default=252,
        help="Minimum number of observations required for each ticker.",
    )
    parser.add_argument(
        "--target-return",
        type=float,
        default=0.12,
        help="Desired annualized return target for the optimizer.",
    )

    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    report = run_sp500_workflow(
        max_tickers=args.max_tickers,
        period=args.period,
        start=args.start,
        end=args.end,
        interval=args.interval,
        min_history=args.min_history,
        target_return=args.target_return,
    )

    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
