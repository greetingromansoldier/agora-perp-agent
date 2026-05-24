"""edge-after-cost computation.

A pure cost model that turns a directional `Forecast` plus a `MarketData`
snapshot into a fee/slippage/funding-aware `CostAssessment`. No network, no
state, deterministic.

Slippage follows the canonical sqrt-law `impact ≈ k · √(notional/depth)`,
with a conservative flat fallback when ``MarketData.book_depth`` is missing.
Funding is signed by trade side and on full notional. All ``*_bps`` outputs
are round-trip (entry + exit).
"""

from __future__ import annotations

import math

from core.contracts import CostAssessment, FeeSchedule, Forecast, MarketData
from core.eps import gt_eps

_EPS = 1e-12
_BPS = 10_000.0


class CostModel:
    """Per-trade cost model: fees + slippage + funding → edge-after-cost."""

    def __init__(self, schedule: FeeSchedule, hold_minutes: float = 1.0) -> None:
        """Configure the venue fee schedule and the expected hold horizon.

        Args:
            schedule: per-venue parameters (fees, funding cadence, slippage).
            hold_minutes: expected hold in minutes; funding accrual scales
                linearly with this fraction of the venue's funding period.

        Raises:
            ValueError: if ``hold_minutes`` is negative.
        """
        if hold_minutes < 0.0:
            raise ValueError(f"hold_minutes must be non-negative, got {hold_minutes}")
        self._schedule = schedule
        self._hold_minutes = hold_minutes

    def assess(
        self,
        forecast: Forecast,
        market: MarketData,
        notional: float,
    ) -> CostAssessment:
        """Return the round-trip cost of trading ``notional`` on this snapshot.

        Args:
            forecast: directional view; the side is picked from ``p_up`` vs
                ``p_down`` and the signed gross move is ``expected_move_bps``
                taken in that direction.
            market: snapshot of the same asset; provides ``funding_rate`` and
                ``book_depth`` for slippage modelling.
            notional: trade size in quote currency (e.g. USD).

        Returns:
            A `CostAssessment` with the three per-component legs, the total,
            the breakeven, and the signed edge after cost.

        Raises:
            ValueError: if ``notional`` is non-positive, or if the forecast's
                asset does not match the market's asset.
        """
        if notional <= 0.0:
            raise ValueError(f"notional must be positive, got {notional}")
        if forecast.asset != market.asset:
            raise ValueError(
                f"forecast asset {forecast.asset!r} does not match "
                f"market asset {market.asset!r}"
            )

        side = self._direction(forecast)

        fee_bps = 2.0 * self._schedule.taker_bps
        slippage_bps = 2.0 * self._side_slippage_bps(notional, market)
        funding_bps = self._funding_bps(market, side)
        round_trip_bps = fee_bps + slippage_bps + funding_bps

        gross_bps = self._signed_move_bps(forecast, side)
        edge_after_cost_bps = gross_bps - round_trip_bps

        return CostAssessment(
            asset=market.asset,
            notional=notional,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
            funding_bps=funding_bps,
            round_trip_bps=round_trip_bps,
            breakeven_bps=round_trip_bps,
            edge_after_cost_bps=edge_after_cost_bps,
            is_tradeable=gt_eps(edge_after_cost_bps, 0.0),
        )

    @staticmethod
    def _direction(forecast: Forecast) -> str:
        """Pick the trade side: ``"long"`` when ``p_up >= p_down``."""
        if gt_eps(forecast.p_down, forecast.p_up):
            return "short"
        return "long"

    @staticmethod
    def _signed_move_bps(forecast: Forecast, side: str) -> float:
        """Gross expected move in the trade's own direction.

        ``Forecast.expected_move_bps`` is already signed by the forecast's
        directional bias (positive when ``p_up > 0.5``). A long takes it as
        given; a short flips the sign so the move is read as a gain.
        """
        return forecast.expected_move_bps if side == "long" else -forecast.expected_move_bps

    def _side_slippage_bps(self, notional: float, market: MarketData) -> float:
        """One-side slippage in bps via sqrt-law, with flat fallback."""
        depth = market.book_depth
        if depth is None or depth <= _EPS:
            return self._schedule.flat_slippage_bps
        ratio = notional / depth
        return self._schedule.slippage_k * math.sqrt(ratio) * _BPS

    def _funding_bps(self, market: MarketData, side: str) -> float:
        """Funding cost in bps over the hold, signed by trade side.

        Convention: ``MarketData.funding_rate > 0`` means longs pay shorts.
        A long over ``hold_minutes`` accrues
        ``funding_rate × hold_minutes / period_minutes`` (as a fraction),
        scaled to bps; a short receives the same magnitude as a credit.
        """
        period_minutes = self._schedule.funding_period_hours * 60.0
        if period_minutes <= _EPS:
            return 0.0
        fraction = self._hold_minutes / period_minutes
        raw_bps = market.funding_rate * fraction * _BPS
        return raw_bps if side == "long" else -raw_bps
