"""tests for `core.regime`: BaselineRegimeClassifier + BTC override + playbook."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from core.contracts import OhlcBar, RegimeTag, Tier
from core.regime import (
    BaselineRegimeClassifier,
    apply_btc_override,
    playbook_multiplier,
)

_T0 = datetime(2026, 5, 24)


def _bar(close: float, i: int, *, hi_off: float = 1.0, lo_off: float = 1.0) -> OhlcBar:
    return OhlcBar(
        timestamp=_T0 + timedelta(hours=i),
        open=close,
        high=close + hi_off,
        low=close - lo_off,
        close=close,
        volume=1.0,
    )


def _series(closes: list[float]) -> tuple[OhlcBar, ...]:
    return tuple(_bar(c, i) for i, c in enumerate(closes))


def _uptrend(n: int = 120, slope: float = 1.0, start: float = 100.0) -> tuple[OhlcBar, ...]:
    return _series([start + slope * i for i in range(n)])


def _downtrend(n: int = 120, slope: float = 1.0, start: float = 200.0) -> tuple[OhlcBar, ...]:
    return _series([start - slope * i for i in range(n)])


def _sideways(n: int = 120, mid: float = 100.0) -> tuple[OhlcBar, ...]:
    # Tight oscillation around `mid` — ADX should stay below 25.
    return _series([mid + (0.2 if i % 2 == 0 else -0.2) for i in range(n)])


# ----------------------------------------------------------- trend axis


def test_trend_up_in_clean_uptrend() -> None:
    bars = _uptrend()
    assert BaselineRegimeClassifier()._trend(bars) == "UP"


def test_trend_down_in_clean_downtrend() -> None:
    bars = _downtrend()
    assert BaselineRegimeClassifier()._trend(bars) == "DOWN"


def test_trend_range_in_sideways() -> None:
    bars = _sideways()
    assert BaselineRegimeClassifier()._trend(bars) == "RANGE"


# ------------------------------------------------------------- vol axis


def test_vol_compressed_when_recent_atr_low() -> None:
    # Constant series → ATR is constant → ratio ~1.0 → NORMAL. To get
    # COMPRESSED we need recent ATR strictly smaller than long-window ATR.
    closes = [100.0 + (5.0 if i < 70 else 0.05) * ((i % 2) * 2 - 1) for i in range(120)]
    bars = _series(closes)
    # Recent 10 bars have tiny range, older 50 have larger range → ratio < 0.65
    assert BaselineRegimeClassifier()._vol(bars) == "COMPRESSED"


def test_vol_expanded_when_recent_atr_elevated() -> None:
    # Tail of the series has bigger ranges than head → ratio > 1.3.
    closes = [100.0 + (0.5 if i < 70 else 10.0) * ((i % 2) * 2 - 1) for i in range(120)]
    bars = _series(closes)
    vol = BaselineRegimeClassifier()._vol(bars)
    assert vol in {"EXPANDED", "CRISIS"}


def test_vol_normal_in_steady_series() -> None:
    bars = _uptrend(n=120, slope=1.0)
    vol = BaselineRegimeClassifier()._vol(bars)
    # Steady linear rise has uniform TR → ratio ~ 1.0.
    assert vol == "NORMAL"


# ---------------------------------------------------------- funding axis


def test_funding_neutral_t1_below_001pct() -> None:
    assert BaselineRegimeClassifier()._funding(0.00005, Tier.T1) == "NEUTRAL"


def test_funding_bull_bias_t1_at_003pct() -> None:
    assert BaselineRegimeClassifier()._funding(0.0003, Tier.T1) == "BULL_BIAS"


def test_funding_bear_bias_t1_at_minus_003pct() -> None:
    assert BaselineRegimeClassifier()._funding(-0.0003, Tier.T1) == "BEAR_BIAS"


def test_funding_extreme_pos_t1_at_01pct() -> None:
    assert BaselineRegimeClassifier()._funding(0.001, Tier.T1) == "EXTREME_POS"


def test_funding_extreme_neg_t1_at_minus_01pct() -> None:
    assert BaselineRegimeClassifier()._funding(-0.001, Tier.T1) == "EXTREME_NEG"


def test_funding_bins_widen_for_t3() -> None:
    # T3 NEUTRAL extends to 0.04% (4× T1) — a rate that's BULL_BIAS on T1
    # is still NEUTRAL on T3.
    rate = 0.0003  # 0.03 %/8h
    assert BaselineRegimeClassifier()._funding(rate, Tier.T1) == "BULL_BIAS"
    assert BaselineRegimeClassifier()._funding(rate, Tier.T3) == "NEUTRAL"


# --------------------------------------------------------------- full classify


def test_classify_emits_regime_tag() -> None:
    bars = _uptrend()
    tag = BaselineRegimeClassifier().classify(bars, 0.0, Tier.T1)
    assert isinstance(tag, RegimeTag)
    assert tag.trend in {"UP", "DOWN", "RANGE"}
    assert tag.vol in {"COMPRESSED", "NORMAL", "EXPANDED", "CRISIS"}
    assert tag.funding in {
        "NEUTRAL", "BULL_BIAS", "BEAR_BIAS", "EXTREME_POS", "EXTREME_NEG",
    }


# ----------------------------------------------------------- BTC override


def test_btc_crisis_forces_alt_vol_crisis() -> None:
    alt = RegimeTag("UP", "NORMAL", "NEUTRAL")
    btc = RegimeTag("RANGE", "CRISIS", "NEUTRAL")
    out = apply_btc_override(alt, btc)
    assert out.vol == "CRISIS"
    assert out.trend == "UP"


def test_btc_down_expanded_blocks_non_short_alts() -> None:
    alt = RegimeTag("UP", "NORMAL", "NEUTRAL")
    btc = RegimeTag("DOWN", "EXPANDED", "NEUTRAL")
    out = apply_btc_override(alt, btc)
    # Forced to CRISIS-vol → playbook returns 0.
    assert out.vol == "CRISIS"


def test_btc_extreme_funding_bumps_alt_funding() -> None:
    alt = RegimeTag("UP", "NORMAL", "NEUTRAL")
    btc = RegimeTag("UP", "NORMAL", "EXTREME_POS")
    out = apply_btc_override(alt, btc)
    assert out.funding == "EXTREME_POS"


def test_btc_override_identity_when_no_rule_fires() -> None:
    alt = RegimeTag("UP", "NORMAL", "NEUTRAL")
    btc = RegimeTag("UP", "NORMAL", "NEUTRAL")
    out = apply_btc_override(alt, btc)
    assert out == alt


def test_btc_override_does_not_double_bump_funding() -> None:
    # If alt is already EXTREME, BTC EXTREME bump is a no-op.
    alt = RegimeTag("UP", "NORMAL", "EXTREME_NEG")
    btc = RegimeTag("UP", "NORMAL", "EXTREME_POS")
    out = apply_btc_override(alt, btc)
    assert out.funding == "EXTREME_NEG"


# ----------------------------------------------------------- playbook


def test_playbook_trend_baseline() -> None:
    tag = RegimeTag("UP", "NORMAL", "NEUTRAL")
    assert playbook_multiplier(tag) == pytest.approx(1.0)


def test_playbook_crowded_long_stands_down() -> None:
    tag = RegimeTag("UP", "NORMAL", "EXTREME_POS")
    assert playbook_multiplier(tag) == 0.0


def test_playbook_short_squeeze_tailwind() -> None:
    tag = RegimeTag("UP", "NORMAL", "EXTREME_NEG")
    assert playbook_multiplier(tag) == pytest.approx(1.3)


def test_playbook_mean_reversion_baseline() -> None:
    tag = RegimeTag("RANGE", "NORMAL", "NEUTRAL")
    assert playbook_multiplier(tag) == pytest.approx(0.7)


def test_playbook_vol_coil() -> None:
    tag = RegimeTag("RANGE", "COMPRESSED", "NEUTRAL")
    assert playbook_multiplier(tag) == pytest.approx(0.9)


def test_playbook_crisis_always_zero() -> None:
    tag = RegimeTag("UP", "CRISIS", "NEUTRAL")
    assert playbook_multiplier(tag) == 0.0


def test_playbook_unlisted_combo_stands_down() -> None:
    # `RANGE × EXPANDED × any` is not in the §4 table → stand-down.
    tag = RegimeTag("RANGE", "EXPANDED", "NEUTRAL")
    assert playbook_multiplier(tag) == 0.0
