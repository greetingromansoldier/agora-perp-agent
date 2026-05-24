"""edge-after-cost ranking for the board.

A pure function over the per-asset `(Forecast, CostAssessment)` board.
Filters out non-tradeable candidates (unless asked otherwise), sorts by
edge desc with deterministic tie-breaks, and returns the top-K as
`AllocationCandidate`s with ``rank`` 0..N-1.

No I/O, no state, no sizing — sizing belongs to the risk gate (L5) and
the LLM agent (L6). This layer is just the selector that says "of what's
on the board now, here are the K best opportunities, in order."
"""

from __future__ import annotations

from core.contracts import AllocationCandidate, CostAssessment, Forecast
from core.eps import gt_eps


def allocate(
    board: dict[str, tuple[Forecast, CostAssessment]],
    max_positions: int = 3,
    require_tradeable: bool = True,
) -> tuple[AllocationCandidate, ...]:
    """Return the top-K candidates ranked by edge-after-cost descending.

    Args:
        board: snapshot per asset of its forecast and cost assessment.
        max_positions: hard cap on the returned tuple length.
        require_tradeable: when True (default), drop candidates whose
            ``cost.is_tradeable`` is False. When False, runners-up with
            negative edge are included — useful for agent diagnostics.

    Returns:
        A tuple of `AllocationCandidate` sorted by edge desc (primary key),
        confidence desc, then asset asc. Length never exceeds
        ``max_positions``. ``rank`` is 0 for the best.

    Raises:
        ValueError: if ``max_positions`` is negative.
    """
    if max_positions < 0:
        raise ValueError(f"max_positions must be non-negative, got {max_positions}")

    items = list(board.items())
    if require_tradeable:
        items = [(a, (f, c)) for a, (f, c) in items if c.is_tradeable]

    items.sort(
        key=lambda item: (
            -item[1][1].edge_after_cost_bps,   # primary: edge desc
            -item[1][0].confidence,             # tie-break: confidence desc
            item[0],                            # then: asset name asc
        )
    )

    top = items[:max_positions]
    return tuple(
        AllocationCandidate(
            asset=asset,
            side=_side(forecast),
            notional=cost.notional,
            forecast=forecast,
            cost=cost,
            rank=rank,
        )
        for rank, (asset, (forecast, cost)) in enumerate(top)
    )


def _side(forecast: Forecast) -> str:
    """Pick trade side: ``"long"`` when ``p_up >= p_down`` (matches `CostModel`)."""
    if gt_eps(forecast.p_down, forecast.p_up):
        return "short"
    return "long"
