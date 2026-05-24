"""tests for `core.synthesis`: protocol + RuleSynthesizer + GeminiSynthesizer.

GeminiSynthesizer tests cover error paths and the JSON parser only; we
don't hit the live API in unit tests. A live integration check would be
``@pytest.mark.slow`` and is not part of this suite.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

import pytest

from core.contracts import (
    AgentState,
    AllocationCandidate,
    CostAssessment,
    Decision,
    Decisions,
    Forecast,
    PortfolioState,
    Position,
    TradeRationale,
)
from core.synthesis import GeminiSynthesizer, RuleSynthesizer

_T0 = datetime(2026, 5, 25, tzinfo=timezone.utc)


def _forecast(asset: str = "BTC", *, p_up: float = 0.7) -> Forecast:
    return Forecast(
        asset=asset,
        timestamp=_T0,
        p_up=p_up,
        p_down=1.0 - p_up,
        expected_move_bps=20.0,
        confidence=0.3,
    )


def _cost(asset: str = "BTC", *, edge: float = 5.0) -> CostAssessment:
    return CostAssessment(
        asset=asset,
        notional=100.0,
        fee_bps=9.0,
        slippage_bps=14.0,
        funding_bps=0.0,
        round_trip_bps=23.0,
        breakeven_bps=23.0,
        edge_after_cost_bps=edge,
        is_tradeable=edge > 0.0,
    )


def _candidate(asset: str = "BTC", *, edge: float = 5.0) -> AllocationCandidate:
    return AllocationCandidate(
        asset=asset,
        side="long",
        notional=100.0,
        forecast=_forecast(asset),
        cost=_cost(asset, edge=edge),
        rank=0,
    )


def _position(asset: str = "ETH") -> Position:
    return Position(
        asset=asset,
        side="long",
        qty=0.05,
        entry_price=2_000.0,
        entry_time=_T0,
        last_mark=2_010.0,
        last_funding_ts=_T0,
    )


def _state(
    *,
    candidates: tuple[AllocationCandidate, ...] = (),
    positions: tuple[Position, ...] = (),
    board_extra: dict[str, tuple[Forecast, CostAssessment]] | None = None,
) -> AgentState:
    pf = PortfolioState.empty(1_000.0)
    for p in positions:
        pf.positions[p.asset] = p
    board: dict[str, tuple[Forecast, CostAssessment]] = {}
    for c in candidates:
        board[c.asset] = (c.forecast, c.cost)
    if board_extra:
        board.update(board_extra)
    return AgentState(
        timestamp=_T0,
        board=board,
        candidates=candidates,
        portfolio=pf,
        recent_fills=(),
    )


# ------------------------------------------------------------ RuleSynthesizer

def test_rule_synthesizer_empty_state_yields_empty_decisions() -> None:
    rs = RuleSynthesizer()
    d = rs.decide(_state())
    assert d.decisions == ()
    assert "0 enter" in d.summary
    assert "0 hold" in d.summary


def test_rule_synthesizer_proposes_enter_for_each_ranked_candidate() -> None:
    rs = RuleSynthesizer()
    state = _state(candidates=(_candidate("BTC"), _candidate("ETH")))
    d = rs.decide(state)
    actions = [(x.asset, x.action) for x in d.decisions]
    assert actions == [("BTC", "enter"), ("ETH", "enter")]


def test_rule_synthesizer_proposes_hold_for_each_open_position() -> None:
    rs = RuleSynthesizer()
    state = _state(positions=(_position("ETH"),))
    d = rs.decide(state)
    assert [x.asset for x in d.decisions] == ["ETH"]
    assert d.decisions[0].action == "hold"
    assert d.decisions[0].side is None


def test_rule_synthesizer_skips_candidates_already_held() -> None:
    rs = RuleSynthesizer()
    state = _state(
        candidates=(_candidate("BTC"), _candidate("ETH")),
        positions=(_position("BTC"),),
    )
    d = rs.decide(state)
    enter_assets = [x.asset for x in d.decisions if x.action == "enter"]
    hold_assets = [x.asset for x in d.decisions if x.action == "hold"]
    assert enter_assets == ["ETH"]
    assert hold_assets == ["BTC"]


def test_rule_synthesizer_decision_ids_are_monotonic() -> None:
    rs = RuleSynthesizer()
    state = _state(candidates=(_candidate("BTC"), _candidate("ETH"), _candidate("SOL")))
    d = rs.decide(state)
    ids = [x.rationale.decision_id for x in d.decisions]
    assert ids == ["d-0001", "d-0002", "d-0003"]


def test_rule_synthesizer_rationale_carries_indicators_and_numbers() -> None:
    rs = RuleSynthesizer()
    state = _state(candidates=(_candidate("BTC", edge=5.5),))
    r = rs.decide(state).decisions[0].rationale
    assert "p_up" in r.indicators
    assert "edge_bps" in r.numbers
    assert r.numbers["edge_bps"] == pytest.approx(5.5)
    assert r.reasoning  # non-empty
    assert r.pros  # at least one pro


def test_rule_synthesizer_marks_negative_edge_as_con() -> None:
    rs = RuleSynthesizer()
    state = _state(candidates=(_candidate("BTC", edge=-10.0),))
    r = rs.decide(state).decisions[0].rationale
    assert any("edge negative" in c for c in r.cons)


# ----------------------------------------------------------- GeminiSynthesizer

def test_gemini_synthesizer_raises_without_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        GeminiSynthesizer()


def test_gemini_synthesizer_parses_well_formed_response() -> None:
    # Build a synth that bypasses __init__ (no API key needed for parser test).
    synth = GeminiSynthesizer.__new__(GeminiSynthesizer)
    synth._counter = 0  # type: ignore[attr-defined]
    raw = (
        '{"summary": "test", "decisions": ['
        ' {"asset": "BTC", "action": "enter", "side": "long",'
        '  "rationale": {"decision_id": "d-0001",'
        '   "indicators": {"p_up": 0.72},'
        '   "numbers": {"edge_bps": 3.5, "cost_bps": 20.0},'
        '   "reasoning": "top edge after cost",'
        '   "pros": ["best edge"], "cons": []}}]}'
    )
    decisions = synth._parse(raw)  # type: ignore[attr-defined]
    assert isinstance(decisions, Decisions)
    assert decisions.summary == "test"
    assert len(decisions.decisions) == 1
    d = decisions.decisions[0]
    assert d.asset == "BTC"
    assert d.action == "enter"
    assert d.side == "long"
    assert d.rationale.decision_id == "d-0001"
    assert d.rationale.indicators == {"p_up": 0.72}
    assert d.rationale.numbers == {"edge_bps": 3.5, "cost_bps": 20.0}
    assert d.rationale.reasoning == "top edge after cost"
    assert d.rationale.pros == ("best edge",)
    assert d.rationale.cons == ()


def test_gemini_synthesizer_raises_on_non_json() -> None:
    synth = GeminiSynthesizer.__new__(GeminiSynthesizer)
    synth._counter = 0  # type: ignore[attr-defined]
    with pytest.raises(RuntimeError, match="non-JSON"):
        synth._parse("hello, not JSON")  # type: ignore[attr-defined]


def test_gemini_synthesizer_raises_on_empty_response() -> None:
    synth = GeminiSynthesizer.__new__(GeminiSynthesizer)
    synth._counter = 0  # type: ignore[attr-defined]
    with pytest.raises(RuntimeError, match="empty response"):
        synth._parse("")  # type: ignore[attr-defined]


def test_gemini_synthesizer_raises_when_decisions_field_missing() -> None:
    synth = GeminiSynthesizer.__new__(GeminiSynthesizer)
    synth._counter = 0  # type: ignore[attr-defined]
    with pytest.raises(RuntimeError, match="missing"):
        synth._parse('{"summary": "x"}')  # type: ignore[attr-defined]


def test_gemini_synthesizer_normalises_invalid_side_to_none() -> None:
    synth = GeminiSynthesizer.__new__(GeminiSynthesizer)
    synth._counter = 0  # type: ignore[attr-defined]
    raw = (
        '{"summary": "x", "decisions": ['
        ' {"asset": "BTC", "action": "hold", "side": "bogus",'
        '  "rationale": {"decision_id": "d-0001",'
        '   "indicators": {}, "numbers": {},'
        '   "reasoning": "", "pros": [], "cons": []}}]}'
    )
    decisions = synth._parse(raw)  # type: ignore[attr-defined]
    assert decisions.decisions[0].side is None


def test_gemini_synthesizer_assigns_fallback_decision_id() -> None:
    synth = GeminiSynthesizer.__new__(GeminiSynthesizer)
    synth._counter = 0  # type: ignore[attr-defined]
    # rationale.decision_id missing in payload
    raw = (
        '{"summary": "x", "decisions": ['
        ' {"asset": "BTC", "action": "hold", "side": null,'
        '  "rationale": {"indicators": {}, "numbers": {},'
        '   "reasoning": "", "pros": [], "cons": []}}]}'
    )
    decisions = synth._parse(raw)  # type: ignore[attr-defined]
    assert decisions.decisions[0].rationale.decision_id == "d-0001"
