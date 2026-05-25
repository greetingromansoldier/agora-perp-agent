"""tests for `core.execute.SimExecutor.open_sized` + `check_stops`."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from core.contracts import (
    AllocationCandidate,
    CostAssessment,
    FeeSchedule,
    Forecast,
    LeverageCaps,
    MarketData,
    OhlcBar,
    PortfolioState,
    RegimeTag,
    SizedCandidate,
    StopTakePlan,
    Tier,
)
from core.execute import SimExecutor

_T0 = datetime(2026, 5, 24)


def _bar(close: float, t: datetime = _T0) -> OhlcBar:
    return OhlcBar(timestamp=t, open=close, high=close, low=close, close=close, volume=1.0)


def _market(
    *, asset: str = "BTC", mark: float = 100_000.0,
    funding: float = 0.0, depth: float | None = 1e12, t: datetime = _T0,
) -> MarketData:
    return MarketData(
        asset=asset, timestamp=t, mark_price=mark,
        funding_rate=funding, bars=(_bar(mark, t),), book_depth=depth,
    )


def _candidate(asset: str = "BTC", side: str = "long") -> AllocationCandidate:
    return AllocationCandidate(
        asset=asset,
        side=side,
        notional=100.0,
        forecast=Forecast(asset, _T0, 0.7, 0.3, 20.0, 0.3),
        cost=CostAssessment(
            asset=asset, notional=100.0, fee_bps=9.0, slippage_bps=14.0,
            funding_bps=0.0, round_trip_bps=23.0, breakeven_bps=23.0,
            edge_after_cost_bps=5.0, is_tradeable=True,
        ),
        rank=0,
    )


def _sized(
    *, asset: str = "BTC", side: str = "long",
    qty: float = 0.01, mark: float = 100_000.0,
    stop_distance: float = 2_000.0, take_distance: float = 5_000.0,
) -> SizedCandidate:
    cand = _candidate(asset, side)
    if side == "long":
        stop_price = mark - stop_distance
        take_price = mark + take_distance
    else:
        stop_price = mark + stop_distance
        take_price = mark - take_distance
    plan = StopTakePlan(
        stop_distance=stop_distance,
        take_distance=take_distance,
        r_multiple=take_distance / stop_distance,
        scaled_exit=True,
        trail_after_first_take=True,
        stop_hardening="limit_stop",
    )
    return SizedCandidate(
        candidate=cand,
        tier=Tier.T1,
        regime=RegimeTag("UP", "NORMAL", "NEUTRAL"),
        qty=qty,
        notional=qty * mark,
        leverage=5.0,
        margin_required=(qty * mark) / 5.0,
        stop_price=stop_price,
        take_price=take_price,
        stop_take_plan=plan,
        leverage_caps=LeverageCaps(40.0, 5.0, 19.0),
        sizing_audit={},
    )


def _executor(k: float = 0.0) -> SimExecutor:
    return SimExecutor(FeeSchedule(taker_bps=4.5, slippage_k=k))


# ----------------------------------------------------- open_sized


def test_open_sized_uses_sized_qty() -> None:
    pf = PortfolioState.empty(1_000_000.0)
    ex = _executor()
    sized = _sized(qty=0.05)
    fill = ex.open_sized(sized, _market(), pf)
    assert pf.has("BTC")
    assert pf.positions["BTC"].qty == pytest.approx(0.05)
    assert fill.qty == pytest.approx(0.05)
    assert fill.is_open is True


def test_open_sized_registers_stops_on_position() -> None:
    pf = PortfolioState.empty(1_000_000.0)
    ex = _executor()
    sized = _sized(stop_distance=3_000.0, take_distance=6_000.0)
    ex.open_sized(sized, _market(), pf)
    pos = pf.positions["BTC"]
    assert pos.stop_price == pytest.approx(97_000.0)
    assert pos.take_price == pytest.approx(106_000.0)
    assert pos.stop_take_plan is sized.stop_take_plan


def test_open_sized_short_registers_inverted_stops() -> None:
    pf = PortfolioState.empty(1_000_000.0)
    ex = _executor()
    sized = _sized(side="short", stop_distance=3_000.0, take_distance=6_000.0)
    ex.open_sized(sized, _market(), pf)
    pos = pf.positions["BTC"]
    assert pos.stop_price == pytest.approx(103_000.0)
    assert pos.take_price == pytest.approx(94_000.0)


def test_open_sized_raises_on_zero_qty() -> None:
    pf = PortfolioState.empty(1_000_000.0)
    ex = _executor()
    sized = _sized(qty=0.0)
    with pytest.raises(ValueError, match="sized.qty must be positive"):
        ex.open_sized(sized, _market(), pf)


def test_open_sized_raises_when_position_already_open() -> None:
    pf = PortfolioState.empty(1_000_000.0)
    ex = _executor()
    ex.open_sized(_sized(), _market(), pf)
    with pytest.raises(ValueError, match="already open"):
        ex.open_sized(_sized(), _market(), pf)


def test_open_sized_raises_on_asset_mismatch() -> None:
    pf = PortfolioState.empty(1_000_000.0)
    ex = _executor()
    with pytest.raises(ValueError, match="does not match"):
        ex.open_sized(_sized(asset="BTC"), _market(asset="ETH"), pf)


# ----------------------------------------------------- check_stops


def test_check_stops_fires_on_long_stop_hit() -> None:
    pf = PortfolioState.empty(1_000_000.0)
    ex = _executor()
    ex.open_sized(_sized(stop_distance=2_000.0), _market(mark=100_000.0), pf)
    # Mark drops to 98_000 — stop at 98_000 → fired.
    fill = ex.check_stops(
        _market(mark=98_000.0, t=_T0 + timedelta(minutes=5)), pf,
    )
    assert fill is not None
    assert fill.is_open is False
    assert "BTC" not in pf.positions


def test_check_stops_fires_on_long_take_hit() -> None:
    pf = PortfolioState.empty(1_000_000.0)
    ex = _executor()
    ex.open_sized(_sized(take_distance=5_000.0), _market(mark=100_000.0), pf)
    fill = ex.check_stops(
        _market(mark=105_000.0, t=_T0 + timedelta(minutes=5)), pf,
    )
    assert fill is not None
    assert "BTC" not in pf.positions


def test_check_stops_fires_on_short_stop_hit() -> None:
    pf = PortfolioState.empty(1_000_000.0)
    ex = _executor()
    ex.open_sized(
        _sized(side="short", stop_distance=2_000.0),
        _market(mark=100_000.0), pf,
    )
    # Mark rises to 102_000 — short stop at 102_000 → fired.
    fill = ex.check_stops(
        _market(mark=102_000.0, t=_T0 + timedelta(minutes=5)), pf,
    )
    assert fill is not None
    assert "BTC" not in pf.positions


def test_check_stops_noop_when_inside_range() -> None:
    pf = PortfolioState.empty(1_000_000.0)
    ex = _executor()
    ex.open_sized(_sized(), _market(mark=100_000.0), pf)
    fill = ex.check_stops(
        _market(mark=100_500.0, t=_T0 + timedelta(minutes=5)), pf,
    )
    assert fill is None
    assert pf.has("BTC")


def test_check_stops_noop_when_no_position() -> None:
    pf = PortfolioState.empty(1_000_000.0)
    ex = _executor()
    fill = ex.check_stops(_market(asset="ETH"), pf)
    assert fill is None
