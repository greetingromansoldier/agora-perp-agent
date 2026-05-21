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
