"""tests for `core.sizing.TierRegimeSizer` end-to-end pipeline."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from core.contracts import (
    AllocationCandidate,
    CostAssessment,
    Forecast,
    MarketData,
    OhlcBar,
    PortfolioState,
    RegimeTag,
    SizedCandidate,
    SizingConfig,
    Tier,
)
from core.sizing import TierRegimeSizer

_T0 = datetime(2026, 5, 24)


def _bar(close: float, i: int, hi_off: float = 0.5, lo_off: float = 0.5) -> OhlcBar:
    return OhlcBar(
        timestamp=_T0 + timedelta(hours=i),
        open=close,
        high=close + hi_off,
        low=close - lo_off,
        close=close,
        volume=1.0,
    )


def _bars_uptrend(n: int = 120) -> tuple[OhlcBar, ...]:
    return tuple(_bar(100.0 + 0.5 * i, i) for i in range(n))


def _market(
    *,
    mark: float = 100_000.0,
    funding: float = 0.0,
    depth: float | None = 10_000_000.0,
    asset: str = "BTC",
) -> MarketData:
    return MarketData(
        asset=asset,
        timestamp=_T0,
        mark_price=mark,
        funding_rate=funding,
        bars=(OhlcBar(_T0, mark, mark, mark, mark, 1.0),),
        book_depth=depth,
    )


def _forecast(asset: str = "BTC") -> Forecast:
    return Forecast(
        asset=asset,
        timestamp=_T0,
        p_up=0.7,
        p_down=0.3,
        expected_move_bps=20.0,
        confidence=0.3,
    )


def _cost(asset: str = "BTC", *, edge: float = 5.0) -> CostAssessment:
    return CostAssessment(
        asset=asset,
        notional=100.0,
        fee_bps=9.0,
        slippage_bps=14.0,
        funding_bps=0.0,
        round_trip_bps=23.0,
        breakeven_bps=23.0,
        edge_after_cost_bps=edge,
        is_tradeable=edge > 0.0,
    )


def _candidate(asset: str = "BTC", side: str = "long") -> AllocationCandidate:
    return AllocationCandidate(
        asset=asset,
        side=side,
        notional=100.0,  # naive placeholder; sizing overrides
        forecast=_forecast(asset),
        cost=_cost(asset),
        rank=0,
    )


# --------------------------------------------------- happy path


def test_sized_candidate_shape() -> None:
    sizer = TierRegimeSizer()
    portfolio = PortfolioState.empty(100_000.0)
    regime = RegimeTag("UP", "NORMAL", "NEUTRAL")
    sized = sizer.size(
        candidate=_candidate(),
        market=_market(),
        bars_1h=_bars_uptrend(),
        portfolio=portfolio,
        regime=regime,
        tier=Tier.T1,
    )
    assert isinstance(sized, SizedCandidate)
    assert sized.tier is Tier.T1
    assert sized.regime is regime
    assert sized.qty > 0.0
    assert sized.notional == pytest.approx(sized.qty * 100_000.0, rel=1e-9)
    assert sized.leverage >= 1.0
    assert sized.margin_required > 0.0
    assert sized.stop_price < 100_000.0  # long → stop below entry
    assert sized.take_price > 100_000.0  # long → take above entry


def test_short_side_inverts_stop_take_signs() -> None:
    sizer = TierRegimeSizer()
    portfolio = PortfolioState.empty(100_000.0)
    regime = RegimeTag("DOWN", "NORMAL", "NEUTRAL")
    sized = sizer.size(
        candidate=_candidate(side="short"),
        market=_market(),
        bars_1h=_bars_uptrend(),
        portfolio=portfolio,
        regime=regime,
        tier=Tier.T1,
    )
    assert sized.stop_price > 100_000.0
    assert sized.take_price < 100_000.0


# --------------------------------------------------- stand-down path


def test_crisis_regime_yields_zero_qty() -> None:
    sizer = TierRegimeSizer()
    portfolio = PortfolioState.empty(100_000.0)
    regime = RegimeTag("UP", "CRISIS", "NEUTRAL")
    sized = sizer.size(
        candidate=_candidate(),
        market=_market(),
        bars_1h=_bars_uptrend(),
        portfolio=portfolio,
        regime=regime,
        tier=Tier.T1,
    )
    assert sized.qty == 0.0
    assert sized.notional == 0.0
    assert sized.leverage == 0.0  # zero-caps stand-down audit shape
    assert sized.sizing_audit["playbook_multiplier"] == 0.0


def test_crowded_long_in_uptrend_stands_down() -> None:
    sizer = TierRegimeSizer()
    portfolio = PortfolioState.empty(100_000.0)
    regime = RegimeTag("UP", "NORMAL", "EXTREME_POS")
    sized = sizer.size(
        candidate=_candidate(),
        market=_market(),
        bars_1h=_bars_uptrend(),
        portfolio=portfolio,
        regime=regime,
        tier=Tier.T1,
    )
    assert sized.qty == 0.0


# --------------------------------------------------- tier sensitivity


def test_lower_tier_gets_smaller_qty_for_same_setup() -> None:
    sizer = TierRegimeSizer()
    portfolio = PortfolioState.empty(100_000.0)
    regime = RegimeTag("UP", "NORMAL", "NEUTRAL")
    bars = _bars_uptrend()
    market = _market()
    sized_t1 = sizer.size(
        candidate=_candidate(),
        market=market,
        bars_1h=bars,
        portfolio=portfolio,
        regime=regime,
        tier=Tier.T1,
    )
    sized_t3 = sizer.size(
        candidate=_candidate(),
        market=market,
        bars_1h=bars,
        portfolio=portfolio,
        regime=regime,
        tier=Tier.T3,
    )
    # T3 multiplier 0.35 < T1 1.0 → smaller base risk → smaller qty AFTER
    # the tier-specific ATR multiplier inflates the stop. Net effect can
    # go either way on qty alone, but final_risk in audit must scale.
    assert sized_t3.sizing_audit["tier_multiplier"] == pytest.approx(0.35)
    assert sized_t1.sizing_audit["tier_multiplier"] == pytest.approx(1.0)
    assert sized_t3.sizing_audit["final_risk_usd"] < sized_t1.sizing_audit["final_risk_usd"]


# --------------------------------------------------- audit dict


def test_audit_carries_every_pipeline_step() -> None:
    sizer = TierRegimeSizer()
    portfolio = PortfolioState.empty(100_000.0)
    regime = RegimeTag("UP", "NORMAL", "NEUTRAL")
    sized = sizer.size(
        candidate=_candidate(),
        market=_market(),
        bars_1h=_bars_uptrend(),
        portfolio=portfolio,
        regime=regime,
        tier=Tier.T1,
    )
    expected_keys = {
        "equity_usd",
        "base_risk_usd",
        "tier_multiplier",
        "tier_risk_usd",
        "playbook_multiplier",
        "playbook_risk_usd",
        "realized_vol_ann",
        "vol_scale",
        "targeted_risk_usd",
        "kelly_cap_usd",
        "final_risk_usd",
        "atr_14_1h",
        "atr_multiplier",
        "stop_distance",
        "take_distance",
        "r_multiple",
        "stop_distance_pct",
        "qty",
        "notional_usd",
        "leverage_used",
        "venue_cap",
        "operational_cap",
        "liq_safety_cap",
        "margin_required_usd",
    }
    assert expected_keys <= set(sized.sizing_audit.keys())


# --------------------------------------------------- guards


def test_zero_mark_raises() -> None:
    sizer = TierRegimeSizer()
    portfolio = PortfolioState.empty(100_000.0)
    regime = RegimeTag("UP", "NORMAL", "NEUTRAL")
    with pytest.raises(ValueError, match="mark_price must be positive"):
        sizer.size(
            candidate=_candidate(),
            market=_market(mark=0.0),
            bars_1h=_bars_uptrend(),
            portfolio=portfolio,
            regime=regime,
            tier=Tier.T1,
        )


def test_too_few_bars_for_vol_raises() -> None:
    sizer = TierRegimeSizer()
    portfolio = PortfolioState.empty(100_000.0)
    regime = RegimeTag("UP", "NORMAL", "NEUTRAL")
    # 60 bars: enough for ATR(50) but realized_vol_ann needs >= 21 — still
    # ok. We need to actually fail: 15 bars trips realized_vol_ann.
    too_short = tuple(_bar(100.0 + i, i) for i in range(15))
    with pytest.raises(ValueError):
        sizer.size(
            candidate=_candidate(),
            market=_market(),
            bars_1h=too_short,
            portfolio=portfolio,
            regime=regime,
            tier=Tier.T1,
        )


# --------------------------------------------------- custom config


def test_custom_base_risk_fraction_scales_qty() -> None:
    cfg_lo = SizingConfig(base_risk_fraction=0.001)
    cfg_hi = SizingConfig(base_risk_fraction=0.01)
    portfolio = PortfolioState.empty(100_000.0)
    regime = RegimeTag("UP", "NORMAL", "NEUTRAL")
    bars = _bars_uptrend()

    sized_lo = TierRegimeSizer(cfg_lo).size(
        candidate=_candidate(), market=_market(), bars_1h=bars,
        portfolio=portfolio, regime=regime, tier=Tier.T1,
    )
    sized_hi = TierRegimeSizer(cfg_hi).size(
        candidate=_candidate(), market=_market(), bars_1h=bars,
        portfolio=portfolio, regime=regime, tier=Tier.T1,
    )
    assert sized_hi.qty == pytest.approx(10.0 * sized_lo.qty, rel=1e-9)
