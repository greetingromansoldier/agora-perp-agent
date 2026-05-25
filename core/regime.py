"""three-axis regime classification: trend × vol × funding.

Implements `docs/trading-agent-context/regime-classification.md` §1-§5.
All inputs are on **1-hour candles**; the docs are unambiguous on this
and mixed cadence breaks the EMA-cross + ADX + ATR-ratio thresholds.
Callers that only have 5-minute candles must aggregate or fetch a
parallel 1h stream — see `tmp/spec-mapping.md` §5 Q3.

Funding bins are tier-aware (§3 table): T1 has tighter bins than
T2-T4 because the impact-notional denominator is larger for majors,
making funding less reactive to small flows.
"""

from __future__ import annotations

from typing import Protocol

from core.contracts import OhlcBar, RegimeTag, Tier
from core.indicators import adx, atr, ema

# §2 vol-ratio thresholds.
_VOL_COMPRESSED_MAX = 0.65
_VOL_NORMAL_MAX = 1.30
_VOL_EXPANDED_MAX = 1.80

# §1 thresholds for the trend axis.
_TREND_EMA_SEPARATION = 0.005  # 0.5%
_TREND_ADX_THRESHOLD = 25.0

# §3 funding bins per tier, expressed as fractional per-8h rates.
# 0.0001 = 0.01%/8h. Values converted from the §3 table verbatim.
_FUNDING_BINS: dict[Tier, dict[str, float]] = {
    Tier.T1: {"neutral": 0.0001, "extreme": 0.0005},
    Tier.T2: {"neutral": 0.0002, "extreme": 0.0008},
    Tier.T3: {"neutral": 0.0004, "extreme": 0.0015},
    Tier.T4: {"neutral": 0.0004, "extreme": 0.0015},
}

# §4 playbook table (the 12 traded combinations). Key:
# `(trend, vol, funding)`. Value: playbook multiplier on `tier_risk`.
# Combinations not listed map to stand-down (0.0).
_PLAYBOOK_TABLE: dict[tuple[str, str, str], float] = {
    # UP regimes
    ("UP", "NORMAL", "NEUTRAL"): 1.0,
    ("UP", "EXPANDED", "NEUTRAL"): 0.5,
    ("UP", "COMPRESSED", "NEUTRAL"): 0.3,  # wait for break; small probe
    ("UP", "NORMAL", "BULL_BIAS"): 1.0,
    ("UP", "NORMAL", "EXTREME_POS"): 0.0,  # crowded longs → stand down
    ("UP", "NORMAL", "EXTREME_NEG"): 1.3,  # shorts squeezed → tail-wind
    # DOWN regimes
    ("DOWN", "NORMAL", "NEUTRAL"): 1.0,
    ("DOWN", "EXPANDED", "NEUTRAL"): 0.5,
    ("DOWN", "NORMAL", "BEAR_BIAS"): 1.0,
    ("DOWN", "NORMAL", "EXTREME_POS"): 1.3,  # longs liquidating → tail-wind
    ("DOWN", "NORMAL", "EXTREME_NEG"): 0.0,  # crowded shorts → stand down
    # RANGE regimes
    ("RANGE", "NORMAL", "NEUTRAL"): 0.7,
    ("RANGE", "COMPRESSED", "NEUTRAL"): 0.9,  # vol-coil mean-reversion
}


class RegimeClassifier(Protocol):
    """Turns 1h bars + funding rate + tier into a `RegimeTag`."""

    def classify(
        self,
        bars_1h: tuple[OhlcBar, ...],
        funding_rate: float,
        tier: Tier,
    ) -> RegimeTag:
        """Return the three-axis tag.

        Args:
            bars_1h: trailing 1h candles, oldest first.
            funding_rate: per-8h-equivalent rate (HL is per-hour native;
                caller multiplies if comparing across venues).
            tier: from `classify_tier`.

        Returns:
            A `RegimeTag` with three string axes.
        """
        ...


class BaselineRegimeClassifier:
    """docs-literal 3-axis classifier with the §1-§3 thresholds.

    No tuning yet — that lives in `alpha/` once we have data to fit
    against. Public per the project's split.
    """

    def classify(
        self,
        bars_1h: tuple[OhlcBar, ...],
        funding_rate: float,
        tier: Tier,
    ) -> RegimeTag:
        """Compute the three-axis tag for one asset at one tick.

        Raises:
            ValueError: insufficient bars for any of the indicators
                (EMA-50, ATR(50), ADX(14) all need their respective
                minimums per `core/indicators.py`).
        """
        return RegimeTag(
            trend=self._trend(bars_1h),
            vol=self._vol(bars_1h),
            funding=self._funding(funding_rate, tier),
        )

    @staticmethod
    def _trend(bars: tuple[OhlcBar, ...]) -> str:
        """EMA-20 vs EMA-50 + ADX(14) trend tag (§1)."""
        ema_20 = ema(bars, 20)
        ema_50 = ema(bars, 50)
        adx_val, _, _ = adx(bars, 14)
        separation = (
            (ema_20 - ema_50) / ema_50 if ema_50 > 0.0 else 0.0
        )
        if adx_val < _TREND_ADX_THRESHOLD:
            return "RANGE"
        if separation >= _TREND_EMA_SEPARATION:
            return "UP"
        if separation <= -_TREND_EMA_SEPARATION:
            return "DOWN"
        return "RANGE"

    @staticmethod
    def _vol(bars: tuple[OhlcBar, ...]) -> str:
        """ATR(10) / ATR(50) ratio bucket (§2)."""
        atr_10 = atr(bars, 10)
        atr_50 = atr(bars, 50)
        ratio = atr_10 / atr_50 if atr_50 > 0.0 else 1.0
        if ratio < _VOL_COMPRESSED_MAX:
            return "COMPRESSED"
        if ratio < _VOL_NORMAL_MAX:
            return "NORMAL"
        if ratio < _VOL_EXPANDED_MAX:
            return "EXPANDED"
        return "CRISIS"

    @staticmethod
    def _funding(rate: float, tier: Tier) -> str:
        """Tier-aware funding bin (§3 table). Splits `EXTREME` by sign
        so playbook lookup at §4 is unambiguous."""
        bins = _FUNDING_BINS[tier]
        abs_rate = abs(rate)
        if abs_rate < bins["neutral"]:
            return "NEUTRAL"
        if abs_rate < bins["extreme"]:
            return "BULL_BIAS" if rate > 0.0 else "BEAR_BIAS"
        return "EXTREME_POS" if rate > 0.0 else "EXTREME_NEG"


def apply_btc_override(
    regime: RegimeTag, btc_regime: RegimeTag
) -> RegimeTag:
    """Apply BTC-dominance rules per §5.

    1. `btc.vol == CRISIS` → force the alt's vol to `CRISIS`.
    2. `btc.trend == DOWN` and `btc.vol == EXPANDED` → if alt isn't
       trending DOWN, force `vol = CRISIS` (stand-down via §4 reading).
    3. `btc.funding ∈ {EXTREME_POS, EXTREME_NEG}` → if alt's funding is
       not already EXTREME, bump it one bucket toward BTC's sign.

    Args:
        regime: the alt's classifier output (pre-override).
        btc_regime: BTC's classifier output for the same tick.

    Returns:
        Adjusted `RegimeTag`. Identity if no rule fires.
    """
    if btc_regime.vol == "CRISIS":
        return RegimeTag(regime.trend, "CRISIS", regime.funding)
    if (
        btc_regime.trend == "DOWN"
        and btc_regime.vol == "EXPANDED"
        and regime.trend != "DOWN"
    ):
        return RegimeTag(regime.trend, "CRISIS", regime.funding)
    if btc_regime.funding in {"EXTREME_POS", "EXTREME_NEG"}:
        if regime.funding in {"NEUTRAL", "BULL_BIAS", "BEAR_BIAS"}:
            return RegimeTag(regime.trend, regime.vol, btc_regime.funding)
    return regime


def playbook_multiplier(regime: RegimeTag) -> float:
    """Return the playbook risk-multiplier per `§4` 12-row table.

    Combinations not listed in §4 → 0.0 (stand-down per §4 reading
    note). `CRISIS` vol → 0.0 regardless of trend/funding.
    """
    if regime.vol == "CRISIS":
        return 0.0
    return _PLAYBOOK_TABLE.get(
        (regime.trend, regime.vol, regime.funding), 0.0
    )
