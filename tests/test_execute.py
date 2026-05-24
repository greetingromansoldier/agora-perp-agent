"""tests for `core.execute.SimExecutor` and portfolio state mutations."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from core.contracts import (
    AllocationCandidate,
    CostAssessment,
    FeeSchedule,
    Fill,
    Forecast,
    MarketData,
    OhlcBar,
    PortfolioState,
)
from core.execute import SimExecutor

_T0 = datetime(2026, 5, 24, tzinfo=timezone.utc)


def _bar(close: float, t: datetime = _T0) -> OhlcBar:
    return OhlcBar(
        timestamp=t, open=close, high=close, low=close, close=close, volume=1.0
    )


def _market(
    *,
    asset: str = "BTC",
    mark: float = 100_000.0,
    funding: float = 0.0,
    depth: float | None = 10_000_000.0,
    t: datetime = _T0,
) -> MarketData:
    return MarketData(
        asset=asset,
        timestamp=t,
        mark_price=mark,
        funding_rate=funding,
        bars=(_bar(mark, t),),
        book_depth=depth,
    )


def _forecast(asset: str = "BTC", p_up: float = 0.7) -> Forecast:
    return Forecast(
        asset=asset,
        timestamp=_T0,
        p_up=p_up,
        p_down=1.0 - p_up,
        expected_move_bps=20.0,
        confidence=0.2,
    )


def _cost(
    asset: str = "BTC", *, notional: float = 100_000.0, edge: float = 5.0
) -> CostAssessment:
    return CostAssessment(
        asset=asset,
        notional=notional,
        fee_bps=9.0,
        slippage_bps=14.0,
        funding_bps=0.0,
        round_trip_bps=23.0,
        breakeven_bps=23.0,
        edge_after_cost_bps=edge,
        is_tradeable=edge > 0.0,
    )


def _candidate(
    asset: str = "BTC",
    side: str = "long",
    notional: float = 100_000.0,
) -> AllocationCandidate:
    return AllocationCandidate(
        asset=asset,
        side=side,
        notional=notional,
        forecast=_forecast(asset, p_up=0.7 if side == "long" else 0.3),
        cost=_cost(asset, notional=notional, edge=5.0),
        rank=0,
    )


def _executor(
    taker_bps: float = 4.5, k: float = 0.005, period_h: float = 1.0
) -> SimExecutor:
    return SimExecutor(
        FeeSchedule(taker_bps=taker_bps, slippage_k=k, funding_period_hours=period_h)
    )


# ----------------------------------------------------------------- open

def test_open_creates_position_and_deducts_fee() -> None:
    pf = PortfolioState.empty(1_000_000.0)
    ex = _executor()
    fill = ex.open(_candidate(), _market(), pf)
    assert pf.has("BTC")
    pos = pf.positions["BTC"]
    assert pos.side == "long"
    assert pos.entry_price > 100_000.0  # long fills with positive slippage
    expected_fee = 100_000.0 * 4.5 / 10_000.0
    assert pf.balance_usd == pytest.approx(1_000_000.0 - expected_fee, rel=1e-12)
    assert fill.is_open is True
    assert fill.realized_pnl_usd == 0.0
    assert fill.fee_paid_usd == pytest.approx(expected_fee, rel=1e-12)


def test_open_short_fills_below_mark() -> None:
    pf = PortfolioState.empty(1_000_000.0)
    ex = _executor()
    fill = ex.open(_candidate(side="short"), _market(), pf)
    assert fill.price < 100_000.0


def test_open_raises_when_position_already_exists() -> None:
    pf = PortfolioState.empty(1_000_000.0)
    ex = _executor()
    ex.open(_candidate(), _market(), pf)
    with pytest.raises(ValueError, match="already open"):
        ex.open(_candidate(), _market(), pf)


def test_open_raises_on_asset_mismatch() -> None:
    pf = PortfolioState.empty(1_000_000.0)
    ex = _executor()
    with pytest.raises(ValueError, match="does not match"):
        ex.open(_candidate(asset="BTC"), _market(asset="ETH"), pf)


# ---------------------------------------------------------------- close

def test_close_removes_position_and_realizes_pnl() -> None:
    pf = PortfolioState.empty(1_000_000.0)
    ex = _executor()
    ex.open(_candidate(), _market(mark=100_000.0), pf)
    later = _market(mark=101_000.0, t=_T0 + timedelta(minutes=1))
    fill = ex.close("BTC", later, pf)
    assert not pf.has("BTC")
    assert fill.is_open is False
    assert fill.realized_pnl_usd > 0.0  # long won when price went up


def test_close_balance_moves_by_exactly_realized_pnl() -> None:
    pf = PortfolioState.empty(1_000_000.0)
    ex = _executor()
    ex.open(_candidate(), _market(mark=100.0, depth=1e12), pf)
    open_fee = 100_000.0 * 4.5 / 10_000.0
    fill = ex.close(
        "BTC",
        _market(mark=105.0, depth=1e12, t=_T0 + timedelta(minutes=1)),
        pf,
    )
    expected_balance = 1_000_000.0 - open_fee + fill.realized_pnl_usd
    assert pf.balance_usd == pytest.approx(expected_balance, rel=1e-12)


def test_close_raises_when_no_position() -> None:
    pf = PortfolioState.empty(1_000_000.0)
    ex = _executor()
    with pytest.raises(ValueError, match="no open position"):
        ex.close("BTC", _market(), pf)


def test_close_raises_on_asset_mismatch() -> None:
    pf = PortfolioState.empty(1_000_000.0)
    ex = _executor()
    ex.open(_candidate(), _market(), pf)
    with pytest.raises(ValueError, match="does not match"):
        ex.close("BTC", _market(asset="ETH"), pf)


# ---------------------------------------------------------- unrealized PnL

def test_long_unrealized_positive_when_mark_up() -> None:
    pf = PortfolioState.empty(1_000_000.0)
    ex = _executor(k=0.0)  # zero slippage for clean PnL signs
    ex.open(_candidate(), _market(mark=100.0, depth=1e12), pf)
    ex.tick(_market(mark=105.0, depth=1e12, t=_T0 + timedelta(minutes=1)), pf)
    assert pf.positions["BTC"].unrealized_pnl_usd() > 0.0


def test_long_unrealized_negative_when_mark_down() -> None:
    pf = PortfolioState.empty(1_000_000.0)
    ex = _executor(k=0.0)
    ex.open(_candidate(), _market(mark=100.0, depth=1e12), pf)
    ex.tick(_market(mark=95.0, depth=1e12, t=_T0 + timedelta(minutes=1)), pf)
    assert pf.positions["BTC"].unrealized_pnl_usd() < 0.0


def test_short_unrealized_positive_when_mark_down() -> None:
    pf = PortfolioState.empty(1_000_000.0)
    ex = _executor(k=0.0)
    ex.open(_candidate(side="short"), _market(mark=100.0, depth=1e12), pf)
    ex.tick(_market(mark=95.0, depth=1e12, t=_T0 + timedelta(minutes=1)), pf)
    assert pf.positions["BTC"].unrealized_pnl_usd() > 0.0


def test_short_unrealized_negative_when_mark_up() -> None:
    pf = PortfolioState.empty(1_000_000.0)
    ex = _executor(k=0.0)
    ex.open(_candidate(side="short"), _market(mark=100.0, depth=1e12), pf)
    ex.tick(_market(mark=105.0, depth=1e12, t=_T0 + timedelta(minutes=1)), pf)
    assert pf.positions["BTC"].unrealized_pnl_usd() < 0.0


# --------------------------------------------------------------- funding

def test_long_pays_positive_funding() -> None:
    pf = PortfolioState.empty(1_000_000.0)
    ex = _executor(period_h=1.0)
    ex.open(_candidate(), _market(mark=100.0, depth=1e12), pf)
    later = _market(
        mark=100.0, funding=0.0001, depth=1e12, t=_T0 + timedelta(minutes=60)
    )
    ex.tick(later, pf)
    assert pf.positions["BTC"].accrued_funding_usd < 0.0


def test_short_receives_positive_funding() -> None:
    pf = PortfolioState.empty(1_000_000.0)
    ex = _executor(period_h=1.0)
    ex.open(_candidate(side="short"), _market(mark=100.0, depth=1e12), pf)
    later = _market(
        mark=100.0, funding=0.0001, depth=1e12, t=_T0 + timedelta(minutes=60)
    )
    ex.tick(later, pf)
    assert pf.positions["BTC"].accrued_funding_usd > 0.0


def test_funding_scales_linearly_with_elapsed_minutes() -> None:
    pf_short = PortfolioState.empty(1_000_000.0)
    pf_long = PortfolioState.empty(1_000_000.0)
    ex = _executor(period_h=1.0)
    ex.open(_candidate(), _market(mark=100.0, depth=1e12), pf_short)
    ex.open(_candidate(), _market(mark=100.0, depth=1e12), pf_long)
    ex.tick(
        _market(mark=100.0, funding=0.0001, depth=1e12, t=_T0 + timedelta(minutes=30)),
        pf_short,
    )
    ex.tick(
        _market(mark=100.0, funding=0.0001, depth=1e12, t=_T0 + timedelta(minutes=60)),
        pf_long,
    )
    assert pf_long.positions["BTC"].accrued_funding_usd == pytest.approx(
        2.0 * pf_short.positions["BTC"].accrued_funding_usd, rel=1e-9
    )


def test_tick_is_idempotent_on_same_timestamp() -> None:
    pf = PortfolioState.empty(1_000_000.0)
    ex = _executor(period_h=1.0)
    ex.open(_candidate(), _market(mark=100.0, depth=1e12), pf)
    later = _market(
        mark=100.0, funding=0.0001, depth=1e12, t=_T0 + timedelta(minutes=30)
    )
    ex.tick(later, pf)
    once = pf.positions["BTC"].accrued_funding_usd
    ex.tick(later, pf)
    twice = pf.positions["BTC"].accrued_funding_usd
    assert once == twice


def test_tick_refreshes_mark_even_on_same_timestamp() -> None:
    pf = PortfolioState.empty(1_000_000.0)
    ex = _executor(period_h=1.0)
    ex.open(_candidate(), _market(mark=100.0, depth=1e12), pf)
    # later mark, same timestamp as last_funding_ts (which is the open time)
    same_ts_new_mark = _market(mark=110.0, depth=1e12, t=_T0)
    ex.tick(same_ts_new_mark, pf)
    assert pf.positions["BTC"].last_mark == 110.0


def test_tick_is_noop_when_no_position_on_asset() -> None:
    pf = PortfolioState.empty(1_000_000.0)
    ex = _executor()
    ex.tick(_market(asset="ETH"), pf)
    assert pf.positions == {}


# ----------------------------------------------------------- equity identity

def test_equity_equals_balance_plus_unrealized_plus_funding() -> None:
    pf = PortfolioState.empty(1_000_000.0)
    ex = _executor(period_h=1.0)
    ex.open(_candidate(), _market(mark=100.0, depth=1e12), pf)
    ex.tick(
        _market(mark=110.0, funding=0.0001, depth=1e12, t=_T0 + timedelta(minutes=60)),
        pf,
    )
    pos = pf.positions["BTC"]
    expected = pf.balance_usd + pos.unrealized_pnl_usd() + pos.accrued_funding_usd
    assert pf.equity_usd() == pytest.approx(expected, rel=1e-12)


# ------------------------------------------------------------- round-trip

def test_round_trip_realized_pnl_matches_hand_count() -> None:
    pf = PortfolioState.empty(1_000_000.0)
    # k=0 → no slippage; depth huge anyway. taker 10 bps for clean arithmetic.
    ex = _executor(taker_bps=10.0, k=0.0, period_h=1.0)
    ex.open(_candidate(notional=1_000.0), _market(mark=100.0, depth=1e18), pf)
    # Move price to 110 over 30 minutes with funding +0.0001/h.
    close_market = _market(
        mark=110.0, funding=0.0001, depth=1e18, t=_T0 + timedelta(minutes=30)
    )
    fill = ex.close("BTC", close_market, pf)

    # Hand count:
    #   entry_price = 100      (k=0 → no slippage)
    #   qty         = 1000/100 = 10
    #   exit_price  = 110
    #   price_pnl   = (110 - 100) × 10                       = 100.00
    #   notional_at_mark = 10 × 110 = 1100
    #   funding cashflow = -1 × 0.0001 × 1100 × (30/60)      = −0.055
    #   exit_fee    = 1100 × 10 / 10_000                     = 1.10
    #   realized    = 100 + (-0.055) - 1.10                  = 98.845
    assert fill.realized_pnl_usd == pytest.approx(98.845, rel=1e-9)


def test_realized_pnl_includes_accrued_funding_on_close() -> None:
    pf = PortfolioState.empty(1_000_000.0)
    ex = _executor(taker_bps=0.0, k=0.0, period_h=1.0)  # only funding contributes
    ex.open(_candidate(notional=1_000.0), _market(mark=100.0, depth=1e18), pf)
    fill = ex.close(
        "BTC",
        _market(
            mark=100.0, funding=0.0001, depth=1e18, t=_T0 + timedelta(minutes=60)
        ),
        pf,
    )
    # No price move, no fees: realized = accrued funding only.
    # long pays positive funding: cashflow = -1 × 0.0001 × 1000 × 1.0 = -0.10
    assert fill.realized_pnl_usd == pytest.approx(-0.10, rel=1e-9)


# --------------------------------------------------------- account derived

def test_portfolio_realized_pnl_property_tracks_balance() -> None:
    pf = PortfolioState.empty(1_000_000.0)
    ex = _executor()
    ex.open(_candidate(), _market(), pf)
    open_fee = 100_000.0 * 4.5 / 10_000.0
    assert pf.realized_pnl_usd == pytest.approx(-open_fee, rel=1e-12)


def test_exposure_sums_qty_times_last_mark_across_positions() -> None:
    pf = PortfolioState.empty(1_000_000.0)
    ex = _executor(k=0.0)
    ex.open(_candidate(asset="BTC", notional=100_000.0), _market(asset="BTC", mark=100.0, depth=1e12), pf)
    ex.open(
        _candidate(asset="ETH", notional=50_000.0),
        _market(asset="ETH", mark=3_000.0, depth=1e12),
        pf,
    )
    # mark stays at open prices since no tick yet
    expected = pf.positions["BTC"].qty * 100.0 + pf.positions["ETH"].qty * 3_000.0
    assert pf.exposure_usd() == pytest.approx(expected, rel=1e-12)
