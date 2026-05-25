"""leverage-cap selection.

Per `docs/trading-agent-context/risk-leverage-and-margin.md` §6: the chosen
leverage is `min(venue_cap, operational_cap, liq_safety_cap)`, then reduced
further if `leverage × |funding| × intervals` would exceed the configured
funding-drag ceiling (§5).

Venue caps come from a static fallback table (`coin-tiers.md` §3); the live
HL `meta` payload should be preferred when L1 carries it. See
`tmp/spec-mapping.md` §5 Q4.
"""

from __future__ import annotations

import math

from core.contracts import LeverageCaps, Tier

# `coin-tiers.md` §3 — HL max leverage at small notional. Verify against
# live HL meta at session start; this is the published fallback.
_HL_MAX_LEVERAGE_FALLBACK: dict[str, float] = {
    "BTC": 40.0,
    "ETH": 25.0,
    "SOL": 20.0,
    "XRP": 20.0,
    "BNB": 20.0,
}
_HL_DEFAULT_MAX_LEVERAGE = 10.0  # §3 "Most other listed coins"

# `risk-leverage-and-margin.md` §0 headline operational caps per tier.
# These are the agent's self-imposed ceilings, not venue limits.
_OPERATIONAL_CAP: dict[Tier, float] = {
    Tier.T1: 5.0,
    Tier.T2: 3.0,
    Tier.T3: 3.0,
    Tier.T4: 1.5,
}

# `risk-leverage-and-margin.md` §3 — liq safety multiplier: stop × 4.
_LIQ_SAFETY_STOP_BUFFER = 4.0

# HL hourly funding cadence per `08a-funding-and-basis.md` §1.
_HL_FUNDING_INTERVAL_HOURS = 1.0


def venue_max_leverage(asset: str) -> float:
    """Return HL's published max-leverage at small notional.

    Falls back to ``10`` per `coin-tiers.md` §3 ("Most other listed
    coins") for unknown assets. Static — live HL meta is the right
    source when available.
    """
    return _HL_MAX_LEVERAGE_FALLBACK.get(asset, _HL_DEFAULT_MAX_LEVERAGE)


def choose_leverage(
    *,
    asset: str,
    notional: float,
    tier: Tier,
    stop_distance_pct: float,
    funding_rate: float,
    hold_hours: float,
    funding_drag_max_pct: float = 0.20,
) -> tuple[float, LeverageCaps]:
    """Pick leverage for one trade and report the three caps for audit.

    Three-cap pipeline per `risk-leverage-and-margin.md` §6, then the
    funding-drag check per §5. Leverage is reduced 20% at a time until
    `leverage × |funding| × intervals ≤ funding_drag_max_pct`, with a
    floor of 1× (the trade is never down-shifted below 1×; the LLM/risk
    gate may still veto if 1× is unworkable).

    Args:
        asset: HL coin shortcode (for venue cap lookup).
        notional: trade notional in USD (qty × mark).
        tier: from `classify_tier`.
        stop_distance_pct: `stop_distance / mark_price` — the fraction
            of mark above/below which the stop will fire.
        funding_rate: current per-hour funding rate (HL native cadence).
        hold_hours: expected hold in hours.
        funding_drag_max_pct: ceiling on `leverage × |funding| × hold`
            (§5). Defaults to `0.20` (20% of margin per hold).

    Returns:
        `(chosen_leverage, LeverageCaps)`. `LeverageCaps` carries all
        three caps for the audit trail.

    Raises:
        ValueError: non-positive `stop_distance_pct` or `notional`, or
            negative `hold_hours`.
    """
    if notional <= 0.0:
        raise ValueError(f"notional must be positive, got {notional}")
    if stop_distance_pct <= 0.0:
        raise ValueError(
            f"stop_distance_pct must be positive, got {stop_distance_pct}"
        )
    if hold_hours < 0.0:
        raise ValueError(f"hold_hours must be non-negative, got {hold_hours}")

    venue = venue_max_leverage(asset)
    operational = _OPERATIONAL_CAP[tier]

    # §1 identity: MMR = IMR / 2 = (1/L) / 2.
    mm_rate = 1.0 / (2.0 * venue) if venue > 0.0 else 0.05
    liq_safety_denom = _LIQ_SAFETY_STOP_BUFFER * stop_distance_pct + mm_rate
    liq_safety = (
        1.0 / liq_safety_denom if liq_safety_denom > 0.0 else venue
    )

    caps = LeverageCaps(
        venue_cap=venue,
        operational_cap=operational,
        liq_safety_cap=liq_safety,
    )

    chosen = min(venue, operational, liq_safety)

    # §5 funding-drag iteration.
    intervals = max(1, math.ceil(hold_hours / _HL_FUNDING_INTERVAL_HOURS))
    drag = chosen * abs(funding_rate) * intervals
    while drag > funding_drag_max_pct and chosen > 1.0:
        chosen *= 0.8
        drag = chosen * abs(funding_rate) * intervals
    chosen = max(chosen, 1.0)

    return chosen, caps
