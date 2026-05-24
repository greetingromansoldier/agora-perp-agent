"""tests for `core.allocate.allocate`: filter, ordering, side, guards."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from core.allocate import allocate
from core.contracts import CostAssessment, Forecast

_T0 = datetime(2026, 5, 24, tzinfo=timezone.utc)


def _forecast(
    asset: str = "BTC",
    *,
    p_up: float = 0.6,
    move_bps: float = 20.0,
    confidence: float = 0.2,
) -> Forecast:
    return Forecast(
        asset=asset,
        timestamp=_T0,
        p_up=p_up,
        p_down=1.0 - p_up,
        expected_move_bps=move_bps,
        confidence=confidence,
    )


def _cost(
    asset: str = "BTC",
    *,
    edge_bps: float = 5.0,
    notional: float = 100_000.0,
) -> CostAssessment:
    fee, slip, funding = 9.0, 14.0, 0.0
    total = fee + slip + funding
    return CostAssessment(
        asset=asset,
        notional=notional,
        fee_bps=fee,
        slippage_bps=slip,
        funding_bps=funding,
        round_trip_bps=total,
        breakeven_bps=total,
        edge_after_cost_bps=edge_bps,
        is_tradeable=edge_bps > 0.0,
    )


# ---------------------------------------------- empty / filter behaviour

def test_empty_board_returns_empty_tuple() -> None:
    assert allocate({}) == ()


def test_all_non_tradeable_filtered_by_default() -> None:
    board = {
        "BTC": (_forecast("BTC"), _cost("BTC", edge_bps=-10.0)),
        "ETH": (_forecast("ETH"), _cost("ETH", edge_bps=-5.0)),
    }
    assert allocate(board) == ()


def test_all_non_tradeable_included_when_filter_off() -> None:
    board = {
        "BTC": (_forecast("BTC"), _cost("BTC", edge_bps=-10.0)),
        "ETH": (_forecast("ETH"), _cost("ETH", edge_bps=-5.0)),
    }
    result = allocate(board, require_tradeable=False)
    assert [c.asset for c in result] == ["ETH", "BTC"]


def test_mixed_keeps_only_tradeable_when_filtered() -> None:
    board = {
        "BTC": (_forecast("BTC"), _cost("BTC", edge_bps=-10.0)),
        "ETH": (_forecast("ETH"), _cost("ETH", edge_bps=+8.0)),
        "SOL": (_forecast("SOL"), _cost("SOL", edge_bps=+3.0)),
    }
    assert tuple(c.asset for c in allocate(board)) == ("ETH", "SOL")


# -------------------------------------------------------------- ordering

def test_higher_edge_ranks_first() -> None:
    board = {
        "BTC": (_forecast("BTC"), _cost("BTC", edge_bps=5.0)),
        "ETH": (_forecast("ETH"), _cost("ETH", edge_bps=12.0)),
        "SOL": (_forecast("SOL"), _cost("SOL", edge_bps=2.0)),
    }
    result = allocate(board, max_positions=5)
    assert [c.asset for c in result] == ["ETH", "BTC", "SOL"]
    assert [c.rank for c in result] == [0, 1, 2]


def test_tie_break_by_confidence_then_asset_name() -> None:
    board = {
        "ZZZ": (_forecast("ZZZ", confidence=0.1), _cost("ZZZ", edge_bps=5.0)),
        "AAA": (_forecast("AAA", confidence=0.9), _cost("AAA", edge_bps=5.0)),
        "MMM": (_forecast("MMM", confidence=0.9), _cost("MMM", edge_bps=5.0)),
    }
    result = allocate(board, max_positions=5)
    # AAA and MMM tie on edge AND confidence → asset name breaks (A < M).
    # ZZZ has lower confidence so it sorts last despite same edge.
    assert [c.asset for c in result] == ["AAA", "MMM", "ZZZ"]


def test_max_positions_caps_length() -> None:
    board = {
        coin: (_forecast(coin), _cost(coin, edge_bps=float(10 - i)))
        for i, coin in enumerate(["BTC", "ETH", "SOL", "ARB", "AVAX"])
    }
    assert len(allocate(board, max_positions=2)) == 2


def test_ranks_are_zero_to_n_minus_one_with_no_gaps() -> None:
    board = {
        "BTC": (_forecast("BTC"), _cost("BTC", edge_bps=10.0)),
        "ETH": (_forecast("ETH"), _cost("ETH", edge_bps=5.0)),
    }
    assert [c.rank for c in allocate(board)] == [0, 1]


# -------------------------------------------------------- side derivation

def test_side_is_long_when_p_up_dominates() -> None:
    board = {"BTC": (_forecast("BTC", p_up=0.7), _cost("BTC", edge_bps=5.0))}
    assert allocate(board)[0].side == "long"


def test_side_is_short_when_p_down_dominates() -> None:
    board = {"BTC": (_forecast("BTC", p_up=0.3), _cost("BTC", edge_bps=5.0))}
    assert allocate(board)[0].side == "short"


def test_side_is_long_when_probabilities_tie() -> None:
    board = {"BTC": (_forecast("BTC", p_up=0.5), _cost("BTC", edge_bps=5.0))}
    assert allocate(board)[0].side == "long"


# ----------------------------------------------------------- determinism

def test_repeated_call_returns_identical_result() -> None:
    board = {
        "BTC": (_forecast("BTC"), _cost("BTC", edge_bps=10.0)),
        "ETH": (_forecast("ETH"), _cost("ETH", edge_bps=5.0)),
    }
    assert allocate(board) == allocate(board)


# ----------------------------------------------------------------- guards

def test_negative_max_positions_raises() -> None:
    with pytest.raises(ValueError, match="max_positions must be non-negative"):
        allocate({}, max_positions=-1)
