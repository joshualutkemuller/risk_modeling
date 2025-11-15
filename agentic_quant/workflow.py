"""Programmatic interface for the agentic quant workflow."""

from __future__ import annotations

from typing import Sequence

from .agents import (
    DataAgent,
    FactorSignalAgent,
    PortfolioConstructionAgent,
    ReportAgent,
    RiskAgent,
    RiskOverlayAgent,
    YahooFinanceDataAgent,
)
from .framework import Agent, AgentPipeline
from .universes import get_sp500_tickers


def build_pipeline(
    tickers: Sequence[str] = ("TECH", "HEALTH", "ENERGY", "UTIL"),
    periods: int = 504,
    target_return: float | None = 0.12,
    data_agent: Agent | None = None,
) -> AgentPipeline:
    """Construct the pipeline without executing it.

    This is useful when callers want to inspect or swap agents before running
    the workflow.  The returned pipeline can be executed via
    :meth:`AgentPipeline.run`.
    """

    agents: list[Agent] = [
        data_agent if data_agent is not None else DataAgent(tickers=tickers, periods=periods),
        FactorSignalAgent(),
        RiskAgent(),
        PortfolioConstructionAgent(target_return=target_return),
        RiskOverlayAgent(),
        ReportAgent(),
    ]

    return AgentPipeline(agents)


def run_workflow(
    tickers: Sequence[str] = ("TECH", "HEALTH", "ENERGY", "UTIL"),
    periods: int = 504,
    target_return: float | None = 0.12,
    data_agent: Agent | None = None,
) -> str:
    """Execute the full pipeline and return the synthesized report."""

    pipeline = build_pipeline(
        tickers=tickers,
        periods=periods,
        target_return=target_return,
        data_agent=data_agent,
    )
    board = pipeline.run()
    return board["report"]


def build_sp500_pipeline(
    *,
    max_tickers: int | None = None,
    period: str | None = "5y",
    start: str | None = None,
    end: str | None = None,
    interval: str = "1d",
    auto_adjust: bool = True,
    min_history: int = 252,
    target_return: float | None = 0.12,
) -> AgentPipeline:
    """Construct a pipeline configured for the S&P 500 universe.

    This helper retrieves the current S&P 500 constituents via
    :func:`get_sp500_tickers`, downloads price history from Yahoo Finance, and
    wires the resulting data agent into the standard workflow.
    """

    tickers = get_sp500_tickers(limit=max_tickers)
    data_agent = YahooFinanceDataAgent(
        tickers,
        period=period,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=auto_adjust,
        min_history=min_history,
    )

    return build_pipeline(
        tickers=tickers,
        periods=min_history + 1,
        target_return=target_return,
        data_agent=data_agent,
    )


def run_sp500_workflow(
    *,
    max_tickers: int | None = None,
    period: str | None = "5y",
    start: str | None = None,
    end: str | None = None,
    interval: str = "1d",
    auto_adjust: bool = True,
    min_history: int = 252,
    target_return: float | None = 0.12,
) -> str:
    """Execute the S&P 500 configured pipeline and return the report."""

    pipeline = build_sp500_pipeline(
        max_tickers=max_tickers,
        period=period,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=auto_adjust,
        min_history=min_history,
        target_return=target_return,
    )
    board = pipeline.run()
    return board["report"]

