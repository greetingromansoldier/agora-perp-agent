"""position-sizing pipeline.

Implements `docs/trading-agent-context/risk-sizing-methods.md` §5 (the
8-step composed pipeline) by orchestrating the tier multiplier
(`coin-tiers.md` §0), playbook multiplier (`regime-classification.md`
§4), vol-target scaling (§3), ATR-based stop distance
(`risk-stops-and-exits.md` §1.1), and the three-cap leverage choice
(`risk-leverage-and-margin.md` §6).

Output is a `SizedCandidate` carrying the full per-step audit dict so L9
receipts hash a stable schema. Public per the "alpha is the tuning, the
math is honest" split documented in `tmp/spec-mapping.md` §1.

The Kelly ceiling (§5 step 5) is skipped at MVP — it requires ≥ 100
in-regime trades of audit history per §4 to derive `(p, b)`, and we don't
have that yet. The hook is left in place so the future Kelly addition is
a single-step extension.
"""

from __future__ import annotations

import math
from typing import Protocol

from core.contracts import (
    AllocationCandidate,
    MarketData,
    OhlcBar,
    PortfolioState,
    RegimeTag,
    SizedCandidate,
    SizingConfig,
    Tier,
)
from core.eps import gt_eps
from core.indicators import atr, realized_vol
from core.leverage import choose_leverage
from core.regime import playbook_multiplier
from core.stops import derive_stops_takes

# `coin-tiers.md` §0 risk-multipliers.
_TIER_RISK_MULTIPLIER: dict[Tier, float] = {
    Tier.T1: 1.0,
    Tier.T2: 0.6,
    Tier.T3: 0.35,
    Tier.T4: 0.2,
}

# 1h bars × hours_per_year for annualising sigma from per-bar log returns.
_HOURS_PER_YEAR = 24.0 * 365.0


class PositionSizer(Protocol):
    """Turns a candidate + market + portfolio into a `SizedCandidate`."""

    def size(
        self,
        candidate: AllocationCandidate,
        market: MarketData,
        bars_1h: tuple[OhlcBar, ...],
        portfolio: PortfolioState,
        regime: RegimeTag,
        tier: Tier,
    ) -> SizedCandidate:
        """Produce a sized trade plan.

        Args:
            candidate: ranked candidate from L4.
            market: market snapshot (mark, depth, funding).
            bars_1h: trailing 1h candles for ATR + realized-vol.
            portfolio: live portfolio (read for equity).
            regime: classified regime tag (post-BTC-override).
            tier: classified tier.

        Returns:
            A `SizedCandidate` ready for the executor.
        """
        ...


class TierRegimeSizer:
    """docs-literal sizing pipeline composing tier × regime × vol-target.

    Implements `risk-sizing-methods.md` §5 steps 1-7 (Kelly ceiling at
    step 5 is hooked but skipped — see module docstring).
    """

    def __init__(
        self,
        config: SizingConfig | None = None,
        hold_hours: float = 1.0,
    ) -> None:
        """Bind a sizing config and an expected hold horizon.

        Args:
            config: pipeline parameters; defaults follow the docs.
            hold_hours: expected hold in hours; drives the funding-drag
                check inside `choose_leverage`. Day-trade defaults at 1h;
                LFT at 8-24h+.
        """
        self._cfg = config or SizingConfig()
        self._hold_hours = hold_hours

    def size(
        self,
        candidate: AllocationCandidate,
        market: MarketData,
        bars_1h: tuple[OhlcBar, ...],
        portfolio: PortfolioState,
        regime: RegimeTag,
        tier: Tier,
    ) -> SizedCandidate:
        """Run the 8-step pipeline and emit a `SizedCandidate`.

        Steps follow `risk-sizing-methods.md` §5 in order. The audit dict
        carries every named field from §6 so L9 receipts can hash a
        stable, post-hoc-attributable schema.

        Raises:
            ValueError: insufficient 1h bars for ATR(14) or realized
                vol; non-positive mark price.
        """
        if market.mark_price <= 0.0:
            raise ValueError(
                f"mark_price must be positive, got {market.mark_price}"
            )

        # Step 1 — base dollar risk.
        equity = portfolio.equity_usd()
        base_risk = equity * self._cfg.base_risk_fraction

        # Step 2 — tier multiplier.
        tier_mult = _TIER_RISK_MULTIPLIER[tier]
        tier_risk = base_risk * tier_mult

        # Step 3 — playbook multiplier (regime-aware).
        playbook_mult = playbook_multiplier(regime)
        playbook_risk = tier_risk * playbook_mult

        # Step 4 — vol-target scaling.
        realized_vol_ann = self._realized_vol_annualized(bars_1h)
        if gt_eps(realized_vol_ann, 0.0):
            vol_scale = min(
                self._cfg.vol_target_annualized / realized_vol_ann,
                self._cfg.vol_scale_cap,
            )
        else:
            vol_scale = self._cfg.vol_scale_cap
        targeted_risk = playbook_risk * vol_scale

        # Step 5 — Kelly ceiling (skipped at MVP; no audit history yet).
        final_risk = targeted_risk
        kelly_cap = float("inf")

        # Step 6 — ATR converts dollar risk → qty via stop distance.
        atr_14 = atr(bars_1h, self._cfg.atr_period_stops)
        plan = derive_stops_takes(
            entry=market.mark_price,
            atr14_1h=atr_14,
            regime=regime,
            tier=tier,
            side=candidate.side,
        )
        if not gt_eps(plan.stop_distance, 0.0):
            qty = 0.0
        else:
            qty = final_risk / plan.stop_distance

        if candidate.side == "long":
            stop_price = market.mark_price - plan.stop_distance
            take_price = market.mark_price + plan.take_distance
        else:
            stop_price = market.mark_price + plan.stop_distance
            take_price = market.mark_price - plan.take_distance

        notional = qty * market.mark_price
        stop_distance_pct = plan.stop_distance / market.mark_price

        # Step 7 — leverage cap with funding-drag check.
        if notional > 0.0:
            leverage, caps = choose_leverage(
                asset=candidate.asset,
                notional=notional,
                tier=tier,
                stop_distance_pct=stop_distance_pct,
                funding_rate=market.funding_rate,
                hold_hours=self._hold_hours,
                funding_drag_max_pct=self._cfg.funding_drag_max_pct,
            )
            margin_required = notional / leverage
        else:
            # Stand-down path: keep audit shape, leverage and margin are 0.
            leverage = 0.0
            margin_required = 0.0
            from core.leverage import venue_max_leverage  # noqa: PLC0415

            caps = self._zero_caps(
                venue_max_leverage(candidate.asset),
                tier,
            )

        atr_multiplier = (
            plan.stop_distance / atr_14 if atr_14 > 0.0 else 0.0
        )

        return SizedCandidate(
            candidate=candidate,
            tier=tier,
            regime=regime,
            qty=qty,
            notional=notional,
            leverage=leverage,
            margin_required=margin_required,
            stop_price=stop_price,
            take_price=take_price,
            stop_take_plan=plan,
            leverage_caps=caps,
            sizing_audit={
                "equity_usd": equity,
                "base_risk_usd": base_risk,
                "tier_multiplier": tier_mult,
                "tier_risk_usd": tier_risk,
                "playbook_multiplier": playbook_mult,
                "playbook_risk_usd": playbook_risk,
                "realized_vol_ann": realized_vol_ann,
                "vol_scale": vol_scale,
                "targeted_risk_usd": targeted_risk,
                "kelly_cap_usd": kelly_cap,
                "final_risk_usd": final_risk,
                "atr_14_1h": atr_14,
                "atr_multiplier": atr_multiplier,
                "stop_distance": plan.stop_distance,
                "take_distance": plan.take_distance,
                "r_multiple": plan.r_multiple,
                "stop_distance_pct": stop_distance_pct,
                "qty": qty,
                "notional_usd": notional,
                "leverage_used": leverage,
                "venue_cap": caps.venue_cap,
                "operational_cap": caps.operational_cap,
                "liq_safety_cap": caps.liq_safety_cap,
                "margin_required_usd": margin_required,
            },
        )

    def _realized_vol_annualized(
        self, bars: tuple[OhlcBar, ...]
    ) -> float:
        """Estimate annualised vol from 1h log returns.

        Uses the largest window we can fit, up to 720 bars (≈ 30 days
        on 1h candles). Annualises by ``sqrt(8760)``.

        Stand-in for the docs' "pull realized_vol_30d_ann from feed" —
        the project computes it from bars until L1 carries the field.

        Raises:
            ValueError: fewer than 21 bars available.
        """
        if len(bars) < 21:
            raise ValueError(
                f"realized vol needs >= 21 1h bars, got {len(bars)}"
            )
        target_n = min(720, len(bars) - 1)
        sigma_per_bar = realized_vol(bars, target_n)
        return sigma_per_bar * math.sqrt(_HOURS_PER_YEAR)

    @staticmethod
    def _zero_caps(venue: float, tier: Tier):
        """Build a `LeverageCaps` for the stand-down path.

        Used when notional is 0 (playbook said stand-down or qty rounded
        to 0). Carries the venue and operational caps for audit, with
        `liq_safety_cap = venue` (degenerate but consistent shape).
        """
        from core.contracts import LeverageCaps  # noqa: PLC0415
        from core.leverage import _OPERATIONAL_CAP  # noqa: PLC0415

        return LeverageCaps(
            venue_cap=venue,
            operational_cap=_OPERATIONAL_CAP[tier],
            liq_safety_cap=venue,
        )
