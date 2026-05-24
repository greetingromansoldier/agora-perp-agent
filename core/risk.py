"""deterministic risk gate.

Sits between L4 (allocate — picked top candidate) and L8 (execute — opens
the position). A pure function of `AllocationCandidate` and the current
`PortfolioState`: it can only approve or veto with a reason. No I/O, no
state.

The gate is the hard backstop the LLM agent at L6 cannot override — the
agent may read the verdict's ``reason`` for its own rationale trace, but
a veto cannot be bypassed.
"""

from __future__ import annotations

from core.contracts import (
    AllocationCandidate,
    PortfolioState,
    RiskConfig,
    RiskVerdict,
)


class RiskGate:
    """Approve or veto a candidate against `RiskConfig` + live portfolio."""

    def __init__(self, config: RiskConfig) -> None:
        """Bind the gate to a risk configuration.

        Args:
            config: limits applied on every `evaluate()` call.
        """
        self._config = config

    def evaluate(
        self,
        candidate: AllocationCandidate,
        portfolio: PortfolioState,
    ) -> RiskVerdict:
        """Run the risk rules in fixed order; first failure wins.

        Order (load-bearing; tests assert it):
            1. edge below ``min_edge_after_cost_bps``,
            2. position already open on the same asset,
            3. ``len(portfolio.positions) >= max_positions``,
            4. ``candidate.notional > max_notional_per_position_usd``,
            5. projected total exposure exceeds ``max_total_exposure_usd``.

        Args:
            candidate: ranked candidate from L4.
            portfolio: live account state from L8.

        Returns:
            A `RiskVerdict`; on approval ``reason == "ok"``. ``adjusted_size``
            is always ``None`` at MVP — callers use ``candidate.notional``.
        """
        cfg = self._config

        edge = candidate.cost.edge_after_cost_bps
        if edge < cfg.min_edge_after_cost_bps:
            return RiskVerdict(
                approved=False,
                reason=(
                    f"edge {edge:.2f} bps below threshold "
                    f"{cfg.min_edge_after_cost_bps:.2f} bps"
                ),
            )

        if portfolio.has(candidate.asset):
            return RiskVerdict(
                approved=False,
                reason=f"position already open on {candidate.asset}",
            )

        if len(portfolio.positions) >= cfg.max_positions:
            return RiskVerdict(
                approved=False,
                reason=f"max positions ({cfg.max_positions}) already open",
            )

        if candidate.notional > cfg.max_notional_per_position_usd:
            return RiskVerdict(
                approved=False,
                reason=(
                    f"notional ${candidate.notional:,.0f} exceeds "
                    f"per-position cap ${cfg.max_notional_per_position_usd:,.0f}"
                ),
            )

        projected = portfolio.exposure_usd() + candidate.notional
        if projected > cfg.max_total_exposure_usd:
            return RiskVerdict(
                approved=False,
                reason=(
                    f"projected exposure ${projected:,.0f} exceeds "
                    f"total cap ${cfg.max_total_exposure_usd:,.0f}"
                ),
            )

        return RiskVerdict(approved=True, reason="ok")
