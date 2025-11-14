"""Domain-specific agents that collaborate on quantitative workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np

from .framework import Blackboard
from mean_variance import MeanVarianceOptimizer, PortfolioPerformance


@dataclass(frozen=True)
class MarketData:
    """Historical market data generated or retrieved by the data agent."""

    tickers: Sequence[str]
    prices: np.ndarray
    returns: np.ndarray


@dataclass(frozen=True)
class SignalReport:
    """Expected return estimates and diagnostics from the signal agent."""

    tickers: Sequence[str]
    expected_returns: np.ndarray
    annualization_factor: float


@dataclass(frozen=True)
class RiskReport:
    """Covariance matrix and metadata from the risk agent."""

    tickers: Sequence[str]
    covariance: np.ndarray
    lookback: int


@dataclass(frozen=True)
class PortfolioPlan:
    """Portfolio weights and analytics produced by the optimizer agent."""

    tickers: Sequence[str]
    target_return: float
    weights: np.ndarray
    performance: PortfolioPerformance
    gmvp_weights: np.ndarray
    gmvp_performance: PortfolioPerformance


class DataAgent:
    """Generates synthetic price series for a collection of assets."""

    def __init__(
        self,
        tickers: Sequence[str],
        periods: int = 252,
        seed: int | None = 7,
        drift: Iterable[float] | float = 0.08,
        volatility: Iterable[float] | float = 0.2,
    ) -> None:
        if periods < 2:
            raise ValueError("periods must be at least 2")
        self.name = "data_agent"
        self._tickers = list(tickers)
        self._periods = periods
        self._seed = seed
        self._drift = drift
        self._volatility = volatility

    def run(self, blackboard: Blackboard) -> None:
        rng = np.random.default_rng(self._seed)
        n_assets = len(self._tickers)
        drift = np.broadcast_to(np.asarray(self._drift, dtype=float), n_assets)
        vol = np.broadcast_to(np.asarray(self._volatility, dtype=float), n_assets)

        dt = 1.0 / 252.0
        prices = np.empty((self._periods, n_assets), dtype=float)
        prices[0] = 100.0
        for t in range(1, self._periods):
            shocks = rng.normal(loc=0.0, scale=1.0, size=n_assets)
            growth = (drift - 0.5 * vol**2) * dt + vol * np.sqrt(dt) * shocks
            prices[t] = prices[t - 1] * np.exp(growth)
        returns = np.diff(prices, axis=0) / prices[:-1]

        market_data = MarketData(tickers=self._tickers, prices=prices, returns=returns)
        blackboard["market_data"] = market_data


class FactorSignalAgent:
    """Computes expected returns using a simple momentum factor model."""

    def __init__(self, lookback: int = 63, annualization_factor: float = 252.0) -> None:
        if lookback < 2:
            raise ValueError("lookback must be at least 2")
        self.name = "signal_agent"
        self._lookback = lookback
        self._annualization_factor = float(annualization_factor)

    def run(self, blackboard: Blackboard) -> None:
        blackboard.require("market_data")
        market_data: MarketData = blackboard["market_data"]
        if market_data.returns.shape[0] < self._lookback:
            raise ValueError("Not enough return observations for the requested lookback window")

        window = market_data.returns[-self._lookback :]
        # Use a normalized momentum score as expected return proxy.
        mean_returns = window.mean(axis=0)
        volatility = window.std(axis=0, ddof=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            sharpes = np.divide(mean_returns, volatility, where=volatility > 0)
        sharpes = np.nan_to_num(sharpes)
        expected_returns = mean_returns + 0.5 * sharpes * volatility
        expected_returns = expected_returns * self._annualization_factor

        report = SignalReport(
            tickers=market_data.tickers,
            expected_returns=expected_returns,
            annualization_factor=self._annualization_factor,
        )
        blackboard["signal_report"] = report


class RiskAgent:
    """Estimates covariance matrices from historical returns."""

    def __init__(self, lookback: int = 252) -> None:
        if lookback < 2:
            raise ValueError("lookback must be at least 2")
        self.name = "risk_agent"
        self._lookback = lookback

    def run(self, blackboard: Blackboard) -> None:
        blackboard.require("market_data")
        market_data: MarketData = blackboard["market_data"]
        if market_data.returns.shape[0] < self._lookback:
            raise ValueError("Not enough return observations for the requested lookback window")

        window = market_data.returns[-self._lookback :]
        covariance = np.cov(window, rowvar=False, ddof=1)

        report = RiskReport(tickers=market_data.tickers, covariance=covariance, lookback=self._lookback)
        blackboard["risk_report"] = report


class PortfolioConstructionAgent:
    """Builds an efficient portfolio using the Markowitz optimizer."""

    def __init__(self, target_return: float | None = None) -> None:
        self.name = "portfolio_agent"
        self._target_return = target_return

    def run(self, blackboard: Blackboard) -> None:
        blackboard.require("signal_report", "risk_report")
        signals: SignalReport = blackboard["signal_report"]
        risk: RiskReport = blackboard["risk_report"]

        optimizer = MeanVarianceOptimizer(signals.expected_returns, risk.covariance)
        gmvp_weights = optimizer.global_minimum_variance_weights()
        gmvp_perf = optimizer.portfolio_performance(gmvp_weights)

        if self._target_return is None:
            target_return = float(np.mean(signals.expected_returns))
        else:
            target_return = float(self._target_return)

        weights = optimizer.efficient_weights(target_return)
        performance = optimizer.portfolio_performance(weights)

        plan = PortfolioPlan(
            tickers=signals.tickers,
            target_return=target_return,
            weights=weights,
            performance=performance,
            gmvp_weights=gmvp_weights,
            gmvp_performance=gmvp_perf,
        )
        blackboard["portfolio_plan"] = plan


class RiskOverlayAgent:
    """Applies simple exposure limits to the constructed portfolio."""

    def __init__(self, max_leverage: float = 1.5, max_single_weight: float = 0.5) -> None:
        if max_leverage <= 0:
            raise ValueError("max_leverage must be positive")
        if not (0 < max_single_weight <= 1.0):
            raise ValueError("max_single_weight must be in (0, 1]")
        self.name = "risk_overlay_agent"
        self._max_leverage = float(max_leverage)
        self._max_single_weight = float(max_single_weight)

    def run(self, blackboard: Blackboard) -> None:
        blackboard.require("portfolio_plan", "signal_report", "risk_report")
        plan: PortfolioPlan = blackboard["portfolio_plan"]
        signals: SignalReport = blackboard["signal_report"]
        risk: RiskReport = blackboard["risk_report"]
        weights = plan.weights.copy()
        gross = float(np.sum(np.abs(weights)))
        if gross > self._max_leverage:
            weights *= self._max_leverage / gross

        max_abs_weight = np.max(np.abs(weights))
        if max_abs_weight > self._max_single_weight:
            scale = self._max_single_weight / max_abs_weight
            weights *= scale

        # Renormalize to keep the portfolio fully invested.
        total = float(np.sum(weights))
        if not np.isclose(total, 0.0):
            weights /= total

        optimizer = MeanVarianceOptimizer(signals.expected_returns, risk.covariance)
        performance = optimizer.portfolio_performance(weights)
        optimized_plan = PortfolioPlan(
            tickers=plan.tickers,
            target_return=plan.target_return,
            weights=weights,
            performance=performance,
            gmvp_weights=plan.gmvp_weights,
            gmvp_performance=plan.gmvp_performance,
        )
        blackboard["portfolio_plan"] = optimized_plan
        blackboard["risk_overlay"] = {
            "gross_leverage": float(np.sum(np.abs(weights))),
            "max_single_weight": float(np.max(np.abs(weights))),
        }


class ReportAgent:
    """Synthesizes a human-readable report of the pipeline outputs."""

    def __init__(self) -> None:
        self.name = "report_agent"

    def run(self, blackboard: Blackboard) -> None:
        blackboard.require("portfolio_plan", "risk_overlay")
        plan: PortfolioPlan = blackboard["portfolio_plan"]
        overlay = blackboard["risk_overlay"]

        lines: List[str] = []
        lines.append("Agentic Quantitative Portfolio Construction Report")
        lines.append("=" * 55)
        lines.append(f"Target annualized return: {plan.target_return:.2%}")
        lines.append(
            f"Achieved by {plan.performance.expected_return:.2%} expected return "
            f"with {plan.performance.volatility:.2%} volatility"
        )
        lines.append("\nWeights after risk overlay:")
        for ticker, weight in zip(plan.tickers, plan.weights):
            lines.append(f"  {ticker}: {weight: .2%}")
        lines.append(
            "\nGlobal minimum variance portfolio (reference): "
            f"{plan.gmvp_performance.expected_return:.2%} expected return, "
            f"{plan.gmvp_performance.volatility:.2%} volatility"
        )
        lines.append(
            "Risk overlay diagnostics: "
            f"gross leverage={overlay['gross_leverage']:.2f}, "
            f"max single weight={overlay['max_single_weight']:.2%}"
        )

        blackboard["report"] = "\n".join(lines)
