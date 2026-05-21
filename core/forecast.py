"""directional forecasting.

A `Forecaster` protocol plus a transparent public baseline. The real
model — its feature set, fusion weights, calibration, and regime map —
is private and implements the same protocol, dropping in behind this
interface without the engine knowing.
"""

from __future__ import annotations

import math
from typing import Protocol

from core.contracts import Forecast, MarketData
from core.indicators import adx, atr, rsi, sma

_EPS = 1e-12


class Forecaster(Protocol):
    """Turns a market snapshot into a directional `Forecast`."""

    def forecast(self, market: MarketData) -> Forecast:
        """Return a forecast for the snapshot's asset.

        Args:
            market: the latest `MarketData` snapshot.

        Returns:
            A `Forecast` with calibrated probabilities and a signed move.
        """
        ...


def _logistic(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class BaselineForecast:
    """Transparent OHLC-only baseline. Not the alpha.

    Combines a trend signal (close vs SMA, scaled by ATR and trend
    strength) with a momentum signal (RSI distance from 50) into a
    logistic probability. Deliberately generic — the real edge lives in
    the private model that implements the same `Forecaster` protocol.
    """

    def __init__(
        self,
        sma_n: int = 20,
        rsi_n: int = 14,
        adx_n: int = 14,
        atr_n: int = 14,
        gain: float = 2.0,
    ) -> None:
        """Configure indicator periods and the logistic gain.

        Args:
            sma_n: SMA window for the trend signal.
            rsi_n: RSI window for the momentum signal.
            adx_n: ADX window for trend-strength scaling.
            atr_n: ATR window for normalising trend and sizing the move.
            gain: slope of the logistic squash applied to the score.
        """
        self._sma_n = sma_n
        self._rsi_n = rsi_n
        self._adx_n = adx_n
        self._atr_n = atr_n
        self._gain = gain

    def forecast(self, market: MarketData) -> Forecast:
        """Compute the baseline forecast for one snapshot.

        Raises:
            ValueError: if the snapshot has too few bars for the indicators
                (ADX needs the most: ``2 * adx_n + 1``).
        """
        bars = market.bars
        price = market.mark_price

        ma = sma(bars, self._sma_n)
        rng = atr(bars, self._atr_n)
        momentum_rsi = rsi(bars, self._rsi_n)
        adx_value, _plus_di, _minus_di = adx(bars, self._adx_n)

        # Trend: how far price sits above/below the SMA, in ATR units,
        # clamped to ~[-1, 1], then scaled by trend strength (ADX).
        trend = (price - ma) / rng if rng > _EPS else 0.0
        trend = max(-3.0, min(3.0, trend)) / 3.0
        trend *= min(adx_value, 50.0) / 50.0

        # Momentum: RSI distance from neutral, in [-1, 1].
        momentum = (momentum_rsi - 50.0) / 50.0

        score = self._gain * (0.5 * trend + 0.5 * momentum)
        p_up = _logistic(score)
        p_down = 1.0 - p_up
        confidence = abs(2.0 * p_up - 1.0)

        # Signed magnitude ~ one ATR, scaled by directional conviction.
        move_bps = (2.0 * p_up - 1.0) * (rng / price) * 10_000.0 if price > _EPS else 0.0

        return Forecast(
            asset=market.asset,
            timestamp=market.timestamp,
            p_up=p_up,
            p_down=p_down,
            expected_move_bps=move_bps,
            confidence=confidence,
        )
