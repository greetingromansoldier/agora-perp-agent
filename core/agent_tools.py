"""tool registry exposed to LLM synthesizers.

Five typed read-only functions over the live `AgentState`. Each returns a
small JSON-serializable dict; the LLM never sees a Python object directly
and can never mutate engine state through these. No new logic lives
here — these are typed views over existing L1-L8 data, suitable both
for direct prompt embedding and for Google Gemini function-calling.
"""

from __future__ import annotations

from core.contracts import AgentState


def read_board(state: AgentState) -> dict[str, dict[str, float | bool]]:
    """Compact per-asset view of everything scanned this tick.

    Args:
        state: current `AgentState`.

    Returns:
        Mapping ``{asset: {p_up, expected_move_bps, round_trip_cost_bps,
        edge_after_cost_bps, is_tradeable}}``.
    """
    return {
        asset: {
            "p_up": forecast.p_up,
            "expected_move_bps": forecast.expected_move_bps,
            "round_trip_cost_bps": cost.round_trip_bps,
            "edge_after_cost_bps": cost.edge_after_cost_bps,
            "is_tradeable": cost.is_tradeable,
        }
        for asset, (forecast, cost) in state.board.items()
    }


def read_candidates(state: AgentState) -> list[dict[str, object]]:
    """The ranked top-K from `allocate()`; best (rank 0) first.

    Args:
        state: current `AgentState`.

    Returns:
        List of dicts with ``{rank, asset, side, notional_usd,
        edge_after_cost_bps}``.
    """
    return [
        {
            "rank": c.rank,
            "asset": c.asset,
            "side": c.side,
            "notional_usd": c.notional,
            "edge_after_cost_bps": c.cost.edge_after_cost_bps,
        }
        for c in state.candidates
    ]


def read_portfolio(state: AgentState) -> dict[str, object]:
    """Account view: balance, PnL aggregates, and open positions.

    Args:
        state: current `AgentState`.

    Returns:
        Dict with ``balance_usd, realized_pnl_usd, unrealized_usd,
        equity_usd, exposure_usd, open_positions[]``.
    """
    pf = state.portfolio
    return {
        "balance_usd": pf.balance_usd,
        "realized_pnl_usd": pf.realized_pnl_usd,
        "unrealized_usd": pf.unrealized_usd(),
        "equity_usd": pf.equity_usd(),
        "exposure_usd": pf.exposure_usd(),
        "open_positions": [
            {
                "asset": p.asset,
                "side": p.side,
                "qty": p.qty,
                "entry_price": p.entry_price,
                "last_mark": p.last_mark,
                "unrealized_pnl_usd": p.unrealized_pnl_usd(),
                "accrued_funding_usd": p.accrued_funding_usd,
            }
            for p in pf.positions.values()
        ],
    }


def get_forecast(state: AgentState, asset: str) -> dict[str, float | str]:
    """Detail view of one asset's directional forecast.

    Args:
        state: current `AgentState`.
        asset: market symbol.

    Returns:
        Dict with the asset's forecast fields, or ``{"error": "..."}``
        if the asset is not in the board.
    """
    entry = state.board.get(asset)
    if entry is None:
        return {"error": f"no board entry for {asset}"}
    forecast, _ = entry
    return {
        "asset": asset,
        "p_up": forecast.p_up,
        "p_down": forecast.p_down,
        "expected_move_bps": forecast.expected_move_bps,
        "confidence": forecast.confidence,
    }


def get_position(state: AgentState, asset: str) -> dict[str, object]:
    """Detail view of one open position; empty dict if none.

    Args:
        state: current `AgentState`.
        asset: market symbol.

    Returns:
        Dict describing the open position, or ``{}`` when no position
        is open on the asset.
    """
    pos = state.portfolio.positions.get(asset)
    if pos is None:
        return {}
    return {
        "asset": pos.asset,
        "side": pos.side,
        "qty": pos.qty,
        "entry_price": pos.entry_price,
        "last_mark": pos.last_mark,
        "unrealized_pnl_usd": pos.unrealized_pnl_usd(),
        "accrued_funding_usd": pos.accrued_funding_usd,
    }
