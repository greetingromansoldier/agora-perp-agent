"""tests for `core.stops.derive_stops_takes`."""

from __future__ import annotations

import pytest

from core.contracts import RegimeTag, Tier
from core.stops import derive_stops_takes

_NORMAL_TREND_UP = RegimeTag("UP", "NORMAL", "NEUTRAL")
_NORMAL_TREND_DOWN = RegimeTag("DOWN", "NORMAL", "NEUTRAL")
_NORMAL_RANGE = RegimeTag("RANGE", "NORMAL", "NEUTRAL")
_COMPRESSED_RANGE = RegimeTag("RANGE", "COMPRESSED", "NEUTRAL")
_EXPANDED_TREND = RegimeTag("UP", "EXPANDED", "NEUTRAL")
_TAILWIND_UP = RegimeTag("UP", "NORMAL", "EXTREME_NEG")
_TAILWIND_DOWN = RegimeTag("DOWN", "NORMAL", "EXTREME_POS")


# ---------------------------------------------------- ATR multipliers per tier


def test_t1_normal_uses_1_5x_atr() -> None:
    plan = derive_stops_takes(
        entry=100.0, atr14_1h=2.0, regime=_NORMAL_TREND_UP, tier=Tier.T1, side="long"
    )
    assert plan.stop_distance == pytest.approx(3.0)  # 1.5 × 2.0


def test_t1_expanded_uses_2_0x_atr() -> None:
    plan = derive_stops_takes(
        entry=100.0, atr14_1h=2.0, regime=_EXPANDED_TREND, tier=Tier.T1, side="long"
    )
    assert plan.stop_distance == pytest.approx(4.0)  # 2.0 × 2.0


def test_t2_normal_uses_2_0x_atr() -> None:
    plan = derive_stops_takes(
        entry=100.0, atr14_1h=2.0, regime=_NORMAL_TREND_UP, tier=Tier.T2, side="long"
    )
    assert plan.stop_distance == pytest.approx(4.0)


def test_t3_expanded_uses_3_0x_atr() -> None:
    plan = derive_stops_takes(
        entry=100.0, atr14_1h=2.0, regime=_EXPANDED_TREND, tier=Tier.T3, side="long"
    )
    assert plan.stop_distance == pytest.approx(6.0)


def test_t4_normal_uses_2_5x_atr() -> None:
    plan = derive_stops_takes(
        entry=100.0, atr14_1h=2.0, regime=_NORMAL_TREND_UP, tier=Tier.T4, side="long"
    )
    assert plan.stop_distance == pytest.approx(5.0)


def test_t4_crisis_uses_3_5x_atr() -> None:
    crisis_regime = RegimeTag("UP", "CRISIS", "NEUTRAL")
    plan = derive_stops_takes(
        entry=100.0, atr14_1h=2.0, regime=crisis_regime, tier=Tier.T4, side="long"
    )
    assert plan.stop_distance == pytest.approx(7.0)


# ---------------------------------------------------------- R-multiples


def test_trend_take_is_2_5x_stop() -> None:
    plan = derive_stops_takes(
        entry=100.0, atr14_1h=2.0, regime=_NORMAL_TREND_UP, tier=Tier.T1, side="long"
    )
    assert plan.r_multiple == pytest.approx(2.5)
    assert plan.take_distance == pytest.approx(2.5 * plan.stop_distance)


def test_tailwind_up_take_is_3_0x_stop() -> None:
    plan = derive_stops_takes(
        entry=100.0, atr14_1h=2.0, regime=_TAILWIND_UP, tier=Tier.T1, side="long"
    )
    assert plan.r_multiple == pytest.approx(3.0)


def test_tailwind_down_take_is_3_0x_stop() -> None:
    plan = derive_stops_takes(
        entry=100.0, atr14_1h=2.0, regime=_TAILWIND_DOWN, tier=Tier.T1, side="short"
    )
    assert plan.r_multiple == pytest.approx(3.0)


def test_mean_rev_take_is_1_5x_stop() -> None:
    plan = derive_stops_takes(
        entry=100.0, atr14_1h=2.0, regime=_NORMAL_RANGE, tier=Tier.T1, side="long"
    )
    assert plan.r_multiple == pytest.approx(1.5)


def test_mean_rev_coil_take_is_2_0x_stop() -> None:
    plan = derive_stops_takes(
        entry=100.0, atr14_1h=2.0, regime=_COMPRESSED_RANGE, tier=Tier.T1, side="long"
    )
    assert plan.r_multiple == pytest.approx(2.0)


# -------------------------------------------------------- scaled exit flag


def test_scaled_exit_true_in_trend() -> None:
    plan = derive_stops_takes(
        entry=100.0, atr14_1h=2.0, regime=_NORMAL_TREND_UP, tier=Tier.T1, side="long"
    )
    assert plan.scaled_exit is True


def test_scaled_exit_false_in_range() -> None:
    plan = derive_stops_takes(
        entry=100.0, atr14_1h=2.0, regime=_NORMAL_RANGE, tier=Tier.T1, side="long"
    )
    assert plan.scaled_exit is False


# ---------------------------------------------------- stop hardening


def test_hardening_t1_is_limit_stop() -> None:
    plan = derive_stops_takes(
        entry=100.0, atr14_1h=2.0, regime=_NORMAL_TREND_UP, tier=Tier.T1, side="long"
    )
    assert plan.stop_hardening == "limit_stop"


def test_hardening_t2_is_limit_stop() -> None:
    plan = derive_stops_takes(
        entry=100.0, atr14_1h=2.0, regime=_NORMAL_TREND_UP, tier=Tier.T2, side="long"
    )
    assert plan.stop_hardening == "limit_stop"


def test_hardening_t4_normal_is_market_stop() -> None:
    plan = derive_stops_takes(
        entry=100.0, atr14_1h=2.0, regime=_NORMAL_TREND_UP, tier=Tier.T4, side="long"
    )
    assert plan.stop_hardening == "market_stop"


def test_hardening_t4_expanded_is_soft_stop_scaled() -> None:
    plan = derive_stops_takes(
        entry=100.0, atr14_1h=2.0, regime=_EXPANDED_TREND, tier=Tier.T4, side="long"
    )
    assert plan.stop_hardening == "soft_stop_scaled"


# -------------------------------------------------------------- guards


def test_invalid_side_raises() -> None:
    with pytest.raises(ValueError, match="side must be"):
        derive_stops_takes(
            entry=100.0, atr14_1h=2.0, regime=_NORMAL_TREND_UP, tier=Tier.T1, side="bogus"
        )


def test_zero_atr_raises() -> None:
    with pytest.raises(ValueError, match="atr14_1h must be positive"):
        derive_stops_takes(
            entry=100.0, atr14_1h=0.0, regime=_NORMAL_TREND_UP, tier=Tier.T1, side="long"
        )


def test_zero_entry_raises() -> None:
    with pytest.raises(ValueError, match="entry must be positive"):
        derive_stops_takes(
            entry=0.0, atr14_1h=2.0, regime=_NORMAL_TREND_UP, tier=Tier.T1, side="long"
        )


def test_trail_after_first_take_always_true() -> None:
    plan = derive_stops_takes(
        entry=100.0, atr14_1h=2.0, regime=_NORMAL_TREND_UP, tier=Tier.T1, side="long"
    )
    assert plan.trail_after_first_take is True
