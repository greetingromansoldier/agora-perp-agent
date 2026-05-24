"""tests for ``core.cost.CostModel``: fees, slippage, funding, edge."""

from __future__ import annotations

import math
from datetime import datetime, timezone

import pytest

from core.contracts import FeeSchedule, Forecast, MarketData, OhlcBar
from core.cost import CostModel

_T0 = datetime(2026, 5, 24, tzinfo=timezone.utc)


def _bar(close: float) -> OhlcBar:
    return OhlcBar(
        timestamp=_T0,
        open=close,
        high=close,
        low=close,
        close=close,
        volume=1.0,
    )


def _market(
    *,
    asset: str = "BTC",
    mark: float = 100_000.0,
    funding: float = 0.0,
    depth: float | None = 5_000_000.0,
) -> MarketData:
    return MarketData(
        asset=asset,
        timestamp=_T0,
        mark_price=mark,
        funding_rate=funding,
        bars=(_bar(mark),),
        book_depth=depth,
    )


def _forecast(
    *,
    asset: str = "BTC",
    p_up: float = 0.6,
    move_bps: float = 20.0,
    confidence: float = 0.2,
) -> Forecast:
    return Forecast(
        asset=asset,
        timestamp=_T0,
        p_up=p_up,
        p_down=1.0 - p_up,
        expected_move_bps=move_bps,
        confidence=confidence,
    )


# ---------------------------------------------------------------- fee leg

def test_fee_is_round_trip_taker() -> None:
    model = CostModel(FeeSchedule(taker_bps=4.5))
    a = model.assess(_forecast(), _market(), 100_000.0)
    assert a.fee_bps == pytest.approx(9.0, rel=1e-12)


# ----------------------------------------------------------- slippage leg

def test_slippage_matches_sqrt_law_at_known_depth() -> None:
    sched = FeeSchedule(slippage_k=0.5)
    a = CostModel(sched).assess(_forecast(), _market(depth=5_000_000.0), 100_000.0)
    one_side = 0.5 * math.sqrt(100_000.0 / 5_000_000.0) * 10_000.0
    assert a.slippage_bps == pytest.approx(2.0 * one_side, rel=1e-9)


def test_slippage_deeper_book_is_smaller() -> None:
    model = CostModel(FeeSchedule(slippage_k=0.5))
    thin = model.assess(_forecast(), _market(depth=1_000_000.0), 100_000.0)
    deep = model.assess(_forecast(), _market(depth=10_000_000.0), 100_000.0)
    assert deep.slippage_bps < thin.slippage_bps


def test_slippage_bigger_size_is_strictly_larger() -> None:
    model = CostModel(FeeSchedule(slippage_k=0.5))
    small = model.assess(_forecast(), _market(), 10_000.0)
    big = model.assess(_forecast(), _market(), 1_000_000.0)
    assert big.slippage_bps > small.slippage_bps


def test_slippage_uses_flat_fallback_when_depth_missing() -> None:
    sched = FeeSchedule(slippage_k=0.5, flat_slippage_bps=30.0)
    a = CostModel(sched).assess(_forecast(), _market(depth=None), 100_000.0)
    assert a.slippage_bps == pytest.approx(60.0, rel=1e-12)


def test_missing_depth_never_understates_cost_vs_a_deep_book() -> None:
    sched = FeeSchedule(slippage_k=0.5, flat_slippage_bps=30.0)
    model = CostModel(sched)
    deep = model.assess(_forecast(), _market(depth=1e12), 100_000.0)
    fallback = model.assess(_forecast(), _market(depth=None), 100_000.0)
    assert fallback.slippage_bps >= deep.slippage_bps


# ------------------------------------------------------------ funding leg

def test_funding_positive_rate_costs_long_credits_short() -> None:
    model = CostModel(FeeSchedule(funding_period_hours=1.0), hold_minutes=60.0)
    long_a = model.assess(_forecast(p_up=0.6), _market(funding=0.0001), 100_000.0)
    short_a = model.assess(_forecast(p_up=0.4), _market(funding=0.0001), 100_000.0)
    assert long_a.funding_bps == pytest.approx(1.0, rel=1e-9)
    assert short_a.funding_bps == pytest.approx(-1.0, rel=1e-9)


def test_funding_is_near_zero_over_one_minute() -> None:
    model = CostModel(FeeSchedule(funding_period_hours=1.0), hold_minutes=1.0)
    a = model.assess(_forecast(p_up=0.6), _market(funding=0.0001), 100_000.0)
    assert a.funding_bps == pytest.approx(1.0 / 60.0, rel=1e-9)


def test_funding_scales_linearly_with_hold_minutes() -> None:
    sched = FeeSchedule(funding_period_hours=1.0)
    short = CostModel(sched, hold_minutes=30.0).assess(
        _forecast(p_up=0.6), _market(funding=0.0001), 100_000.0
    )
    long = CostModel(sched, hold_minutes=120.0).assess(
        _forecast(p_up=0.6), _market(funding=0.0001), 100_000.0
    )
    assert long.funding_bps == pytest.approx(4.0 * short.funding_bps, rel=1e-9)


# ------------------------------------------------------ totals + edge sign

def test_round_trip_equals_sum_of_legs() -> None:
    a = CostModel(FeeSchedule()).assess(
        _forecast(), _market(funding=0.0001), 100_000.0
    )
    assert a.round_trip_bps == pytest.approx(
        a.fee_bps + a.slippage_bps + a.funding_bps, rel=1e-12
    )


def test_tradeable_when_gross_move_exceeds_cost() -> None:
    a = CostModel(FeeSchedule()).assess(
        _forecast(p_up=0.6, move_bps=50.0), _market(), 100_000.0
    )
    assert a.edge_after_cost_bps > 0.0
    assert a.is_tradeable is True


def test_not_tradeable_when_cost_exceeds_move() -> None:
    a = CostModel(FeeSchedule()).assess(
        _forecast(p_up=0.6, move_bps=1.0), _market(), 100_000.0
    )
    assert a.edge_after_cost_bps < 0.0
    assert a.is_tradeable is False


def test_short_side_flips_signed_move() -> None:
    # forecast pointing down: p_up < p_down, expected_move_bps signed negative
    a = CostModel(FeeSchedule()).assess(
        _forecast(p_up=0.4, move_bps=-100.0), _market(), 100_000.0
    )
    # short side reads the move as +100 bps
    assert a.edge_after_cost_bps == pytest.approx(
        100.0 - a.round_trip_bps, rel=1e-9
    )


# --------------------------------------------------------------- guards

def test_assess_raises_on_non_positive_notional() -> None:
    model = CostModel(FeeSchedule())
    with pytest.raises(ValueError, match="notional must be positive"):
        model.assess(_forecast(), _market(), 0.0)
    with pytest.raises(ValueError, match="notional must be positive"):
        model.assess(_forecast(), _market(), -100.0)


def test_assess_raises_on_asset_mismatch() -> None:
    model = CostModel(FeeSchedule())
    with pytest.raises(ValueError, match="does not match"):
        model.assess(_forecast(asset="BTC"), _market(asset="ETH"), 100_000.0)


def test_constructor_raises_on_negative_hold_minutes() -> None:
    with pytest.raises(ValueError, match="hold_minutes must be non-negative"):
        CostModel(FeeSchedule(), hold_minutes=-1.0)
