"""Utilities for working with investable universes."""

from __future__ import annotations

from typing import Iterable, List


def get_sp500_tickers(limit: int | None = None) -> List[str]:
    """Return the current S&P 500 constituents as Yahoo Finance tickers.

    The tickers are sourced via :mod:`yfinance`.  Yahoo represents tickers with
    periods using hyphens (for example ``BRK-B`` instead of ``BRK.B``), so this
    helper normalizes the identifiers accordingly and preserves the published
    order from the index provider.

    Args:
        limit: Optionally cap the number of tickers that are returned.  This can
            be useful for prototyping or reducing the size of downstream data
            downloads.  When ``None`` (the default), all tickers are returned.

    Returns:
        A list of normalized ticker symbols ready to be passed to
        :class:`~agentic_quant.agents.YahooFinanceDataAgent` or other agents.

    Raises:
        RuntimeError: If :mod:`yfinance` is not installed in the environment.
        ValueError: If ``limit`` is provided but not positive, or if no tickers
            are returned from :mod:`yfinance`.
    """

    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive when provided")

    try:
        import yfinance as yf  # type: ignore
    except ImportError as exc:  # pragma: no cover - defensive import
        raise RuntimeError(
            "yfinance is required to retrieve the S&P 500 universe"
        ) from exc

    raw_tickers: Iterable[str]
    tickers = yf.tickers_sp500()
    if isinstance(tickers, str):
        raw_tickers = tickers.split()
    else:
        raw_tickers = tickers

    normalized: List[str] = []
    seen = set()
    for ticker in raw_tickers:
        clean = ticker.strip().upper().replace(".", "-")
        if not clean or clean in seen:
            continue
        seen.add(clean)
        normalized.append(clean)

    if not normalized:
        raise ValueError("No tickers returned for the S&P 500 universe")

    if limit is not None:
        normalized = normalized[:limit]

    return normalized
