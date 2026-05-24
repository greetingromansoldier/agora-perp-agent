"""synthesizer interface and a deterministic rule-based implementation.

A `Synthesizer.decide(state)` consumes an `AgentState` and produces
`Decisions` — per-asset actions plus a short summary. The Protocol lets
the engine swap a mechanical fallback (`RuleSynthesizer`) for an
LLM-driven implementation (`GeminiSynthesizer`, lands in a follow-up
commit) with one line and no engine changes.

`RuleSynthesizer` reproduces the basic mechanical heuristic: for each
ranked candidate without an open position, propose `enter`; for each
open position, propose `hold`. Deterministic, no I/O, used in tests and
as the default when no LLM API key is available.

Every decision carries a structured `TradeRationale` — fixed-shape so
the trace stays comparable across ticks and Arc receipts (L9) hash a
stable schema.
"""

from __future__ import annotations

from typing import Protocol

from core.contracts import (
    AgentState,
    Decision,
    Decisions,
    TradeRationale,
)


class Synthesizer(Protocol):
    """Decision interface consumed by the trading loop."""

    def decide(self, state: AgentState) -> Decisions:
        """Return per-asset decisions plus a one-line summary for this tick."""
        ...


class RuleSynthesizer:
    """Deterministic mechanical synthesizer; no LLM, no tool calls.

    For each ranked candidate whose asset has no open position, emits
    ``enter`` with the candidate's side. For each open position, emits
    ``hold``. Other assets are implicitly ``skip`` (omitted from output).
    """

    def __init__(self) -> None:
        """Start a fresh decision counter for monotonic ``decision_id``s."""
        self._counter = 0

    def decide(self, state: AgentState) -> Decisions:
        """Build mechanical `Decisions` for this state.

        Args:
            state: live `AgentState` from the trading loop.

        Returns:
            A `Decisions` whose tuple length is
            ``len(candidates_not_held) + len(positions)``.
        """
        decisions: list[Decision] = []
        held = set(state.portfolio.positions)

        for cand in state.candidates:
            if cand.asset in held:
                continue
            cons: tuple[str, ...] = ()
            if not cand.cost.is_tradeable:
                cons = ("edge negative — mechanical rule proposed it anyway",)
            rationale = TradeRationale(
                decision_id=self._next_id(),
                indicators={
                    "p_up": cand.forecast.p_up,
                    "confidence": cand.forecast.confidence,
                },
                numbers={
                    "edge_bps": cand.cost.edge_after_cost_bps,
                    "round_trip_cost_bps": cand.cost.round_trip_bps,
                    "expected_move_bps": cand.forecast.expected_move_bps,
                },
                reasoning=(
                    f"rank {cand.rank + 1} on edge-after-cost; mechanical "
                    f"rule opens the best ranked candidate."
                ),
                pros=(
                    f"best edge in ranked set: "
                    f"{cand.cost.edge_after_cost_bps:+.2f} bps",
                ),
                cons=cons,
            )
            decisions.append(
                Decision(
                    asset=cand.asset,
                    action="enter",
                    side=cand.side,
                    rationale=rationale,
                )
            )

        for asset in held:
            pos = state.portfolio.positions[asset]
            inds: dict[str, float] = {}
            nums: dict[str, float] = {
                "unrealized_pnl_usd": pos.unrealized_pnl_usd(),
                "accrued_funding_usd": pos.accrued_funding_usd,
                "qty": pos.qty,
            }
            board_entry = state.board.get(asset)
            if board_entry is not None:
                forecast, cost = board_entry
                inds["p_up"] = forecast.p_up
                nums["edge_bps"] = cost.edge_after_cost_bps
            rationale = TradeRationale(
                decision_id=self._next_id(),
                indicators=inds,
                numbers=nums,
                reasoning=(
                    "position open; mechanical rule holds — closing is the "
                    "agent's job, not a flip-rule."
                ),
                pros=("no churn cost from forecast jitter",),
                cons=("no flip-signal reaction; can sit through clear reversals",),
            )
            decisions.append(
                Decision(
                    asset=asset,
                    action="hold",
                    side=None,
                    rationale=rationale,
                )
            )

        n_enter = sum(1 for d in decisions if d.action == "enter")
        n_hold = sum(1 for d in decisions if d.action == "hold")
        summary = f"mechanical: {n_enter} enter, {n_hold} hold"
        return Decisions(decisions=tuple(decisions), summary=summary)

    def _next_id(self) -> str:
        self._counter += 1
        return f"d-{self._counter:04d}"
