"""tests for `core.risk.RiskGate`: rules + first-failure ordering."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from core.contracts import (
    AllocationCandidate,
    CostAssessment,
    Forecast,
    PortfolioState,
    Position,
    RiskConfig,
)
from core.risk import RiskGate

_T0 = datetime(2026, 5, 24, tzinfo=timezone.utc)


def _cost(
    asset: str = "BTC",
    *,
    edge_bps: float = 5.0,
    notional: float = 100_000.0,
) -> CostAssessment:
    return CostAssessment(
        asset=asset,
        notional=notional,
        fee_bps=9.0,
        slippage_bps=14.0,
        funding_bps=0.0,
        round_trip_bps=23.0,
        breakeven_bps=23.0,
        edge_after_cost_bps=edge_bps,
        is_tradeable=edge_bps > 0.0,
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


def _candidate(
    asset: str = "BTC",
    side: str = "long",
    notional: float = 100_000.0,
    edge_bps: float = 5.0,
) -> AllocationCandidate:
    return AllocationCandidate(
        asset=asset,
        side=side,
        notional=notional,
        forecast=_forecast(asset),
        cost=_cost(asset, edge_bps=edge_bps, notional=notional),
        rank=0,
    )


def _position(asset: str, notional_at_mark: float = 100_000.0) -> Position:
    qty = notional_at_mark / 100.0  # entry/last_mark = 100 → easy notional math
    return Position(
        asset=asset,
        side="long",
        qty=qty,
        entry_price=100.0,
        entry_time=_T0,
        last_mark=100.0,
        last_funding_ts=_T0,
    )


# ----------------------------------------------------------------- approve

def test_clean_candidate_on_empty_portfolio_is_approved() -> None:
    gate = RiskGate(RiskConfig())
    verdict = gate.evaluate(_candidate(), PortfolioState.empty())
    assert verdict.approved is True
    assert verdict.reason == "ok"
    assert verdict.adjusted_size is None


# -------------------------------------------------------------- individual rules

def test_veto_when_edge_below_threshold() -> None:
    gate = RiskGate(RiskConfig(min_edge_after_cost_bps=10.0))
    verdict = gate.evaluate(_candidate(edge_bps=5.0), PortfolioState.empty())
    assert verdict.approved is False
    assert "edge" in verdict.reason
    assert "below threshold" in verdict.reason


def test_veto_when_asset_already_held() -> None:
    pf = PortfolioState.empty()
    pf.positions["BTC"] = _position("BTC")
    verdict = RiskGate(RiskConfig()).evaluate(_candidate(asset="BTC"), pf)
    assert verdict.approved is False
    assert "already open" in verdict.reason
    assert "BTC" in verdict.reason


def test_veto_when_max_positions_reached() -> None:
    pf = PortfolioState.empty()
    pf.positions["ETH"] = _position("ETH")
    pf.positions["SOL"] = _position("SOL")
    verdict = RiskGate(RiskConfig(max_positions=2)).evaluate(
        _candidate(asset="BTC"), pf
    )
    assert verdict.approved is False
    assert "max positions" in verdict.reason


def test_veto_when_notional_exceeds_per_position_cap() -> None:
    gate = RiskGate(RiskConfig(max_notional_per_position_usd=50_000.0))
    verdict = gate.evaluate(
        _candidate(notional=100_000.0), PortfolioState.empty()
    )
    assert verdict.approved is False
    assert "per-position cap" in verdict.reason


def test_veto_when_total_exposure_would_be_exceeded() -> None:
    pf = PortfolioState.empty()
    pf.positions["ETH"] = _position("ETH", notional_at_mark=100_000.0)
    verdict = RiskGate(RiskConfig(max_total_exposure_usd=150_000.0)).evaluate(
        _candidate(notional=100_000.0), pf
    )
    assert verdict.approved is False
    assert "total cap" in verdict.reason


# ------------------------------------------------------ first-failure ordering

def test_edge_check_runs_before_uniqueness_check() -> None:
    # Both rules would fail; edge fires first per documented order.
    pf = PortfolioState.empty()
    pf.positions["BTC"] = _position("BTC")
    verdict = RiskGate(RiskConfig(min_edge_after_cost_bps=10.0)).evaluate(
        _candidate(asset="BTC", edge_bps=5.0), pf
    )
    assert verdict.approved is False
    assert "edge" in verdict.reason
    assert "already open" not in verdict.reason


def test_uniqueness_check_runs_before_max_positions_check() -> None:
    # max_positions also at cap, but uniqueness fires first.
    pf = PortfolioState.empty()
    pf.positions["BTC"] = _position("BTC")
    verdict = RiskGate(RiskConfig(max_positions=1)).evaluate(
        _candidate(asset="BTC"), pf
    )
    assert verdict.approved is False
    assert "already open" in verdict.reason
    assert "max positions" not in verdict.reason


# ---------------------------------------------------------------- invariants

def test_adjusted_size_always_none_at_mvp() -> None:
    ok = RiskGate(RiskConfig()).evaluate(_candidate(), PortfolioState.empty())
    no = RiskGate(RiskConfig(min_edge_after_cost_bps=100.0)).evaluate(
        _candidate(edge_bps=1.0), PortfolioState.empty()
    )
    assert ok.adjusted_size is None
    assert no.adjusted_size is None


def test_veto_reason_is_non_empty_and_not_ok() -> None:
    verdict = RiskGate(RiskConfig(min_edge_after_cost_bps=100.0)).evaluate(
        _candidate(edge_bps=1.0), PortfolioState.empty()
    )
    assert verdict.reason
    assert verdict.reason != "ok"
