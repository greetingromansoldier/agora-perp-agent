"""tier classification for the universe.

Maps Hyperliquid coin symbols to one of `Tier.T1`-`Tier.T4` per
`docs/trading-agent-context/coin-tiers.md` §2. The §2 ordered classifier
takes `(spot_volume_30d_usd, hl_max_leverage, realized_vol_90d_ann)` inputs
that our current `MarketData` does not carry — for MVP we ship a hardcoded
table seeded from the doc's per-tier example lists (§1.1-§1.4) and accept
that re-tier on live data is a follow-up task.

The hardcoded mapping is deliberately conservative: unknown coins default
to `Tier.T4` (default-skip per §0 "coins with no tier are skipped").
"""

from __future__ import annotations

from core.contracts import Tier

# Per-asset tier sourced from `coin-tiers.md` §1.1-§1.4 examples.
# Re-derive from live HL meta + 90d-vol once L1 carries those fields.
_DEFAULT_TIER_TABLE: dict[str, Tier] = {
    # T1 — majors per §1.1
    "BTC": Tier.T1,
    "ETH": Tier.T1,
    # T2 — large alts per §1.2
    "SOL": Tier.T2,
    "BNB": Tier.T2,
    "XRP": Tier.T2,
    # T3 — mid alts per §1.3 (DOGE is "T3 by liquidity, T4 by behaviour" per
    # §1.3 — the regime classifier handles the vol-driven re-tier).
    "LINK": Tier.T3,
    "AVAX": Tier.T3,
    "OP": Tier.T3,
    "ARB": Tier.T3,
    "DOGE": Tier.T3,
    # T4 — thin/meme/new per §1.4 (illustrative — production needs a
    # symbol allow-list, which lives in the trading loop, not here).
    "SUI": Tier.T4,
    "TIA": Tier.T4,
}


def classify_tier(asset: str) -> Tier:
    """Return the tier label for `asset`; default `Tier.T4` if unknown.

    Args:
        asset: HL coin shortcode (uppercase, e.g. ``"BTC"``).

    Returns:
        One of `Tier.T1`-`Tier.T4`. Unknown coins map to `T4` per the
        "coins with no tier are skipped" default-reject convention in
        `coin-tiers.md` §0.
    """
    return _DEFAULT_TIER_TABLE.get(asset, Tier.T4)
