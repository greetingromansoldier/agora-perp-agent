"""stop-loss and take-profit derivation.

Implements `docs/trading-agent-context/risk-stops-and-exits.md` §1.1
(ATR-based stop distance), §3.1 (R-multiples), §3.2 (scaled-exit policy),
§3.3 (trailing after first take), and §2 (stop hardening). Pure function;
no state, no I/O.
"""

from __future__ import annotations

from core.contracts import RegimeTag, StopTakePlan, Tier

# `risk-stops-and-exits.md` §1.1 ATR-multiplier table per tier × vol-regime.
# COMPRESSED reuses NORMAL (stops should not over-tighten in coil), CRISIS
# reuses EXPANDED.
_ATR_MULTIPLIERS: dict[tuple[Tier, str], float] = {
    (Tier.T1, "COMPRESSED"): 1.5,
    (Tier.T1, "NORMAL"): 1.5,
    (Tier.T1, "EXPANDED"): 2.0,
    (Tier.T1, "CRISIS"): 2.0,
    (Tier.T2, "COMPRESSED"): 2.0,
    (Tier.T2, "NORMAL"): 2.0,
    (Tier.T2, "EXPANDED"): 2.5,
    (Tier.T2, "CRISIS"): 2.5,
    (Tier.T3, "COMPRESSED"): 2.0,
    (Tier.T3, "NORMAL"): 2.0,
    (Tier.T3, "EXPANDED"): 3.0,
    (Tier.T3, "CRISIS"): 3.0,
    (Tier.T4, "COMPRESSED"): 2.5,
    (Tier.T4, "NORMAL"): 2.5,
    (Tier.T4, "EXPANDED"): 3.5,
    (Tier.T4, "CRISIS"): 3.5,
}

# `risk-stops-and-exits.md` §3.1 R-multiples per regime.
_R_MULT_TREND = 2.5
_R_MULT_TREND_TAILWIND = 3.0
_R_MULT_MEAN_REV = 1.5
_R_MULT_MEAN_REV_COIL = 2.0


def derive_stops_takes(
    *,
    entry: float,
    atr14_1h: float,
    regime: RegimeTag,
    tier: Tier,
    side: str,
) -> StopTakePlan:
    """Compute the stop, take, and hardening plan for one open.

    Args:
        entry: intended entry price (`market.mark_price` at fill time).
            The stop is offset relative to this; the field is kept for
            interface symmetry even though the function doesn't use it
            directly — entry-relative pricing happens at the call site.
        atr14_1h: ATR(14) on 1h candles in price units.
        regime: the asset's `RegimeTag` (post-BTC-override).
        tier: the asset's tier.
        side: ``"long"`` or ``"short"``.

    Returns:
        A `StopTakePlan` carrying `stop_distance`, `take_distance`, the
        R-multiple chosen, the scaled-exit flag, trail-after-first-take
        flag, and stop-hardening choice.

    Raises:
        ValueError: invalid `side` or non-positive ATR.
    """
    if side not in {"long", "short"}:
        raise ValueError(f"side must be 'long' or 'short', got {side!r}")
    if atr14_1h <= 0.0:
        raise ValueError(f"atr14_1h must be positive, got {atr14_1h}")
    if entry <= 0.0:
        raise ValueError(f"entry must be positive, got {entry}")

    atr_mult = _ATR_MULTIPLIERS[(tier, regime.vol)]
    stop_distance = atr_mult * atr14_1h

    r_mult = _r_multiple(regime)
    take_distance = r_mult * stop_distance

    return StopTakePlan(
        stop_distance=stop_distance,
        take_distance=take_distance,
        r_multiple=r_mult,
        scaled_exit=regime.trend in {"UP", "DOWN"},
        trail_after_first_take=True,
        stop_hardening=_stop_hardening(tier, regime),
    )


def _r_multiple(regime: RegimeTag) -> float:
    """Pick R-multiple per `§3.1 table`.

    Trend-follow with funding tail-wind (crowded other-side liquidating)
    gets the wider 3.0× target; baseline trend is 2.5×; baseline
    mean-reversion is 1.5×; mean-reversion in a vol-coil is 2.0×.
    """
    if regime.trend in {"UP", "DOWN"}:
        if (
            regime.trend == "UP"
            and regime.funding == "EXTREME_NEG"
        ) or (
            regime.trend == "DOWN"
            and regime.funding == "EXTREME_POS"
        ):
            return _R_MULT_TREND_TAILWIND
        return _R_MULT_TREND
    if regime.vol == "COMPRESSED":
        return _R_MULT_MEAN_REV_COIL
    return _R_MULT_MEAN_REV


def _stop_hardening(tier: Tier, regime: RegimeTag) -> str:
    """Pick stop hardening per `§2 table`.

    - T1/T2 → limit-stop (deep books survive slippage).
    - T3/T4 in EXPANDED/CRISIS → soft-stop scaled exit (stops in cascade
      fill 5-8% worse than trigger per §2).
    - Otherwise → reduce-only market-stop.
    """
    if tier in {Tier.T1, Tier.T2}:
        return "limit_stop"
    if regime.vol in {"EXPANDED", "CRISIS"}:
        return "soft_stop_scaled"
    return "market_stop"
