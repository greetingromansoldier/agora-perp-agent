"""tests for `core.tiers.classify_tier`."""

from __future__ import annotations

from core.contracts import Tier
from core.tiers import classify_tier


def test_btc_is_t1() -> None:
    assert classify_tier("BTC") is Tier.T1


def test_eth_is_t1() -> None:
    assert classify_tier("ETH") is Tier.T1


def test_sol_is_t2() -> None:
    assert classify_tier("SOL") is Tier.T2


def test_bnb_is_t2() -> None:
    assert classify_tier("BNB") is Tier.T2


def test_xrp_is_t2() -> None:
    assert classify_tier("XRP") is Tier.T2


def test_link_is_t3() -> None:
    assert classify_tier("LINK") is Tier.T3


def test_doge_is_t3_by_liquidity() -> None:
    # `coin-tiers.md` §1.3: "T3 by liquidity, T4 by behaviour" — we tag T3
    # here; the regime classifier handles vol-driven re-tier.
    assert classify_tier("DOGE") is Tier.T3


def test_sui_is_t4() -> None:
    assert classify_tier("SUI") is Tier.T4


def test_unknown_asset_defaults_to_t4() -> None:
    # `coin-tiers.md` §0: "coins with no tier are skipped" — we return T4
    # so the default-reject path kicks in upstream.
    assert classify_tier("MYSTERYCOIN") is Tier.T4


def test_classify_is_pure() -> None:
    # Repeated calls must produce the same result without side effects.
    assert classify_tier("BTC") is classify_tier("BTC")
