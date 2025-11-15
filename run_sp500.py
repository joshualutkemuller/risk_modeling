"""Runner script for executing the S&P 500 agentic workflow via pandas_datareader."""

from __future__ import annotations

import argparse
import sys
from typing import Optional

from agentic_quant import (
    PandasDataReaderDataAgent,
    build_pipeline,
    get_sp500_tickers,
)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Execute the agentic quant workflow configured for the current "
            "S&P 500 universe using pandas_datareader to source historical prices."
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
        "--start",
        type=str,
        default=None,
        help="Start date (YYYY-MM-DD) for price history. Defaults to 5 years ago.",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="Explicit end date (YYYY-MM-DD) for price history.",
    )
    parser.add_argument(
        "--data-source",
        type=str,
        default="stooq",
        help=(
            "pandas_datareader source to use (e.g., 'stooq', 'av', 'fred'). "
            "Defaults to 'stooq'."
        ),
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

    tickers = get_sp500_tickers(limit=args.max_tickers)
    data_agent = PandasDataReaderDataAgent(
        tickers,
        data_source=args.data_source,
        start=args.start,
        end=args.end,
        min_history=args.min_history,
    )
    pipeline = build_pipeline(
        tickers=tickers,
        periods=args.min_history + 1,
        target_return=args.target_return,
        data_agent=data_agent,
    )
    board = pipeline.run()
    report = board["report"]
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
