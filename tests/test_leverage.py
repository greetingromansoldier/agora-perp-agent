"""tests for `core.leverage.choose_leverage` + venue-cap lookup."""

from __future__ import annotations

import pytest

from core.contracts import Tier
from core.leverage import choose_leverage, venue_max_leverage


# ------------------------------------------------------------- venue cap


def test_venue_max_btc_is_40() -> None:
    assert venue_max_leverage("BTC") == 40.0


def test_venue_max_eth_is_25() -> None:
    assert venue_max_leverage("ETH") == 25.0


def test_venue_max_sol_is_20() -> None:
    assert venue_max_leverage("SOL") == 20.0


def test_venue_max_unknown_defaults_10() -> None:
    assert venue_max_leverage("MYSTERY") == 10.0


# -------------------------------------------------------- choose_leverage


def test_operational_cap_binds_on_t1_with_tight_stop() -> None:
    # 1% stop on BTC: liq_safety = 1/(0.04 + 0.0125) ≈ 19. Venue = 40,
    # operational = 5. min → 5.
    chosen, caps = choose_leverage(
        asset="BTC",
        notional=10_000.0,
        tier=Tier.T1,
        stop_distance_pct=0.01,
        funding_rate=0.0,
        hold_hours=1.0,
    )
    assert chosen == pytest.approx(5.0)
    assert caps.venue_cap == 40.0
    assert caps.operational_cap == 5.0
    assert caps.liq_safety_cap > 5.0


def test_liq_safety_cap_binds_with_wide_stop() -> None:
    # 25% stop on BTC: liq_safety = 1/(1.0 + 0.0125) ≈ 0.99 → ~1.
    chosen, _ = choose_leverage(
        asset="BTC",
        notional=10_000.0,
        tier=Tier.T1,
        stop_distance_pct=0.25,
        funding_rate=0.0,
        hold_hours=1.0,
    )
    assert chosen == pytest.approx(1.0, rel=0.05)


def test_t4_operational_cap_is_1_5() -> None:
    chosen, caps = choose_leverage(
        asset="MYSTERY",
        notional=1_000.0,
        tier=Tier.T4,
        stop_distance_pct=0.05,
        funding_rate=0.0,
        hold_hours=1.0,
    )
    assert caps.operational_cap == 1.5
    assert chosen <= 1.5


def test_funding_drag_reduces_leverage() -> None:
    # At 0.3 %/h × 24h × 5× = 0.36 — exceeds 0.20 cap. Iterates ×0.8 until
    # under cap. Threshold for reduction: `lev × funding × hold > 0.20`.
    chosen_drag, _ = choose_leverage(
        asset="BTC",
        notional=10_000.0,
        tier=Tier.T1,
        stop_distance_pct=0.01,
        funding_rate=0.003,
        hold_hours=24.0,
        funding_drag_max_pct=0.20,
    )
    chosen_zero, _ = choose_leverage(
        asset="BTC",
        notional=10_000.0,
        tier=Tier.T1,
        stop_distance_pct=0.01,
        funding_rate=0.0,
        hold_hours=24.0,
    )
    assert chosen_drag < chosen_zero
    assert chosen_zero == pytest.approx(5.0)  # operational cap binds


def test_leverage_never_below_1() -> None:
    chosen, _ = choose_leverage(
        asset="BTC",
        notional=10_000.0,
        tier=Tier.T1,
        stop_distance_pct=0.01,
        funding_rate=0.5,  # absurdly high funding
        hold_hours=100.0,
        funding_drag_max_pct=0.20,
    )
    assert chosen >= 1.0


def test_caps_object_carries_all_three() -> None:
    _, caps = choose_leverage(
        asset="ETH",
        notional=5_000.0,
        tier=Tier.T1,
        stop_distance_pct=0.02,
        funding_rate=0.0,
        hold_hours=1.0,
    )
    assert caps.venue_cap == 25.0
    assert caps.operational_cap == 5.0
    assert caps.liq_safety_cap > 0.0


# -------------------------------------------------------------- guards


def test_non_positive_notional_raises() -> None:
    with pytest.raises(ValueError, match="notional must be positive"):
        choose_leverage(
            asset="BTC",
            notional=0.0,
            tier=Tier.T1,
            stop_distance_pct=0.01,
            funding_rate=0.0,
            hold_hours=1.0,
        )


def test_non_positive_stop_pct_raises() -> None:
    with pytest.raises(ValueError, match="stop_distance_pct must be positive"):
        choose_leverage(
            asset="BTC",
            notional=10_000.0,
            tier=Tier.T1,
            stop_distance_pct=0.0,
            funding_rate=0.0,
            hold_hours=1.0,
        )


def test_negative_hold_hours_raises() -> None:
    with pytest.raises(ValueError, match="hold_hours must be non-negative"):
        choose_leverage(
            asset="BTC",
            notional=10_000.0,
            tier=Tier.T1,
            stop_distance_pct=0.01,
            funding_rate=0.0,
            hold_hours=-1.0,
        )
