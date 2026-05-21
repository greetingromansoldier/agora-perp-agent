"""core data contracts.

frozen dataclasses passed between engine components. timestamps are
stdlib datetime so this module stays dependency-free.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class OhlcBar:
    """one candle.

    Attributes:
        timestamp: candle open time (UTC).
        open: open price.
        high: high price.
        low: low price.
        close: close price.
        volume: traded volume over the candle.
    """

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True, slots=True)
class MarketData:
    """one market snapshot for one tick.

    Attributes:
        asset: market symbol, e.g. "BTC".
        timestamp: snapshot time (UTC).
        mark_price: current mark price.
        funding_rate: current funding rate per its cadence. Its effect is
            near-zero at a minute horizon but it is carried for honest
            accounting on longer holds.
        bars: recent candles, oldest first.
        book_depth: notional within a band of mid, for slippage modeling.
            None when not collected.
    """

    asset: str
    timestamp: datetime
    mark_price: float
    funding_rate: float
    bars: tuple[OhlcBar, ...]
    book_depth: float | None = None


@dataclass(frozen=True, slots=True)
class Forecast:
    """directional forecast for one asset over the next tick.

    ``p_up + p_down`` need not sum to 1; the remaining mass is the
    implicit "flat / no edge" probability.

    Attributes:
        asset: market symbol.
        timestamp: forecast time (the latest bar's time).
        p_up: probability price rises over the horizon (0..1).
        p_down: probability price falls over the horizon (0..1).
        expected_move_bps: signed magnitude estimate in basis points.
        confidence: model self-reported confidence (0..1).
    """

    asset: str
    timestamp: datetime
    p_up: float
    p_down: float
    expected_move_bps: float
    confidence: float
