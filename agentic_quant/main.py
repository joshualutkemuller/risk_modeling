"""Command line entry point for running the agentic quant workflow."""

from __future__ import annotations

from typing import Sequence

from .agents import (
    DataAgent,
    FactorSignalAgent,
    PortfolioConstructionAgent,
    ReportAgent,
    RiskAgent,
    RiskOverlayAgent,
)
from .framework import AgentPipeline


def run_workflow(
    tickers: Sequence[str] = ("TECH", "HEALTH", "ENERGY", "UTIL"),
    periods: int = 504,
    target_return: float | None = 0.12,
) -> str:
    """Execute the full pipeline and return the synthesized report."""

    pipeline = AgentPipeline(
        [
            DataAgent(tickers=tickers, periods=periods),
            FactorSignalAgent(),
            RiskAgent(),
            PortfolioConstructionAgent(target_return=target_return),
            RiskOverlayAgent(),
            ReportAgent(),
        ]
    )
    board = pipeline.run()
    return board["report"]


def main() -> None:
    report = run_workflow()
    print(report)


if __name__ == "__main__":
    main()
