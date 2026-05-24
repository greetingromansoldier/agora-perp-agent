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


@dataclass(frozen=True, slots=True)
class FeeSchedule:
    """venue fee and slippage parameters used by ``CostModel``.

    Defaults are the Hyperliquid base tier (maker 1.5 bps / taker 4.5 bps,
    hourly funding cadence). ``slippage_k`` is the dimensionless constant in
    the sqrt-law impact formula ``impact ≈ k · √(notional/depth)``; default
    ``0.005`` is a working baseline for top perps (BTC/ETH on HL) — calibrate
    against actual fills per asset before any number leaves the lab.
    ``flat_slippage_bps`` is the conservative per-side fallback applied when
    ``MarketData.book_depth`` is unavailable.

    Attributes:
        maker_bps: maker fee in basis points of notional.
        taker_bps: taker fee in basis points of notional.
        funding_period_hours: cadence at which the venue settles funding.
        slippage_k: dimensionless slope of the sqrt-law impact model.
        flat_slippage_bps: per-side fallback slippage when depth is missing.
    """

    maker_bps: float = 1.5
    taker_bps: float = 4.5
    funding_period_hours: float = 1.0
    slippage_k: float = 0.005
    flat_slippage_bps: float = 30.0


@dataclass(frozen=True, slots=True)
class CostAssessment:
    """round-trip cost breakdown for one candidate trade.

    All ``*_bps`` fields are in basis points of ``notional``. Each leg is
    already round-trip (entry + exit), so ``round_trip_bps`` is just the sum
    of the three legs. ``edge_after_cost_bps`` is the forecast's gross move
    in the trade's direction minus ``round_trip_bps``. ``is_tradeable`` is a
    convenience flag — the real go/no-go gate is L5 risk.

    Attributes:
        asset: market symbol.
        notional: trade size in quote currency.
        fee_bps: round-trip taker fee.
        slippage_bps: round-trip slippage (sqrt-law or flat fallback).
        funding_bps: signed funding over the hold; positive = cost to us.
        round_trip_bps: ``fee_bps + slippage_bps + funding_bps``.
        breakeven_bps: gross move needed to overcome the round-trip cost.
        edge_after_cost_bps: signed expected move minus round-trip cost.
        is_tradeable: ``True`` iff ``edge_after_cost_bps`` is positive.
    """

    asset: str
    notional: float
    fee_bps: float
    slippage_bps: float
    funding_bps: float
    round_trip_bps: float
    breakeven_bps: float
    edge_after_cost_bps: float
    is_tradeable: bool
