"""synthesizer interface, a deterministic rule, and a Gemini-backed LLM impl.

A `Synthesizer.decide(state)` consumes an `AgentState` and produces
`Decisions` — per-asset actions plus a short summary. The Protocol lets
the engine swap implementations with one line and no engine changes.

Two concrete synthesizers ship here:

- `RuleSynthesizer` — deterministic, no LLM, no tool calls. Default
  fallback when no LLM API key is set; used by every unit test.
- `GeminiSynthesizer` — wraps Google's `google-genai` SDK. Sends the
  whole `AgentState` as JSON context, forces JSON-only output, parses
  it strictly into `Decisions`. Reads ``GEMINI_API_KEY`` from env.

Every decision carries a structured `TradeRationale` — fixed-shape so
the trace stays comparable across ticks and Arc receipts (L9) hash a
stable schema.
"""

from __future__ import annotations

import json
import os
from typing import Any, Protocol

from core.agent_tools import (
    read_board,
    read_candidates,
    read_portfolio,
)
from core.contracts import (
    AgentState,
    Decision,
    Decisions,
    Fill,
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


class GeminiSynthesizer:
    """LLM-driven synthesizer using Google Gemini.

    Reads ``GEMINI_API_KEY`` from env. One synchronous API call per
    ``decide()``: serialises the full `AgentState` as JSON context with
    a fixed system prompt, forces JSON-only output, parses strictly into
    `Decisions`.

    This is the MVP single-call shape. Multi-step ReAct (think → tool →
    observe → ...) is a future iteration; the `Synthesizer` Protocol is
    unchanged.

    The system prompt below is generic and ships in the public repo. The
    tuned prompt framing — the alpha — lives in `alpha/prompts/` in the
    private repo and is injected by replacing `SYSTEM_PROMPT` at runtime.
    """

    DEFAULT_MODEL = "gemini-3.1-pro-preview"
    SYSTEM_PROMPT = (
        "You are an autonomous perpetual-futures trading agent on Hyperliquid.\n"
        "Every tick you receive the full board (every coin we scanned), the "
        "ranked candidates from the allocator, and the current paper portfolio.\n"
        "Decide what to do per asset.\n"
        "\n"
        "Hard rules:\n"
        "1. Never invent numbers. Cite only values present in the state JSON.\n"
        "2. Output STRICTLY the JSON schema described below — no prose around it.\n"
        "3. Every decision must include a structured rationale; do not skip fields.\n"
        "4. `side` is required for `enter` and `flip`; must be null otherwise.\n"
        "5. Be honest: if cost dominates edge, propose `skip`.\n"
    )

    def __init__(
        self,
        model: str | None = None,
        api_key_env: str = "GEMINI_API_KEY",
        max_output_tokens: int = 16_384,
        system_prompt: str | None = None,
    ) -> None:
        """Create the Gemini client; raise if the API key is missing.

        Args:
            model: Gemini model id (default ``"gemini-3.1-pro-preview"``).
            api_key_env: env var name holding the API key.
            max_output_tokens: cap on the response size.
            system_prompt: override the default honest-mode system prompt
                with a custom one (e.g. an exploration / demo prompt).

        Raises:
            RuntimeError: if ``api_key_env`` is not set in the environment
                or the `google-genai` SDK is not installed.
        """
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(
                f"GeminiSynthesizer requires env var {api_key_env!r}. "
                f"Set it (e.g. `export {api_key_env}=...`) and retry."
            )
        try:
            from google import genai  # lazy import — keeps unit tests dep-free
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "google-genai SDK is not installed. Run "
                "`uv add google-genai` (or `pip install google-genai`)."
            ) from e
        self._client = genai.Client(api_key=api_key)
        self._model = model or self.DEFAULT_MODEL
        self._max_output_tokens = max_output_tokens
        self._system_prompt = system_prompt or self.SYSTEM_PROMPT
        self._counter = 0

    def decide(self, state: AgentState) -> Decisions:
        """Round-trip the state through Gemini and parse the response.

        Args:
            state: live `AgentState`.

        Returns:
            Parsed `Decisions` with every decision carrying a structured
            `TradeRationale`.

        Raises:
            RuntimeError: if the model returns non-JSON or a malformed
                payload.
        """
        from google.genai import types as gtypes  # lazy import

        prompt = self._build_prompt(state)
        config = gtypes.GenerateContentConfig(
            response_mime_type="application/json",
            max_output_tokens=self._max_output_tokens,
            system_instruction=self._system_prompt,
        )
        response = self._client.models.generate_content(
            model=self._model,
            contents=prompt,
            config=config,
        )
        return self._parse(response.text or "")

    def _build_prompt(self, state: AgentState) -> str:
        """Compose the user prompt: state JSON + the output schema."""
        ctx = {
            "timestamp": state.timestamp.isoformat(),
            "board": read_board(state),
            "candidates": read_candidates(state),
            "portfolio": read_portfolio(state),
            "recent_fills": [_fill_to_dict(f) for f in state.recent_fills],
        }
        return (
            "STATE:\n"
            f"{json.dumps(ctx, indent=2, default=float)}\n"
            "\n"
            "Emit exactly this JSON shape:\n"
            "{\n"
            '  "summary": "<one-line overall rationale, ≤120 chars>",\n'
            '  "decisions": [\n'
            "    {\n"
            '      "asset": "<symbol>",\n'
            '      "action": "enter|hold|cut|flip|skip",\n'
            '      "side": "long|short|null",\n'
            '      "rationale": {\n'
            '        "decision_id": "d-NNNN",\n'
            '        "indicators": {"p_up": ..., ...},\n'
            '        "numbers": {"edge_bps": ..., "cost_bps": ..., ...},\n'
            '        "reasoning": "<≤2 sentences>",\n'
            '        "pros": ["<bullet>", ...],\n'
            '        "cons": ["<bullet>", ...]\n'
            "      }\n"
            "    }\n"
            "  ]\n"
            "}\n"
        )

    def _parse(self, raw: str) -> Decisions:
        """Parse the model output into `Decisions`; raise on shape errors."""
        text = (raw or "").strip()
        if not text:
            raise RuntimeError("agent returned empty response")
        try:
            data: Any = json.loads(text)
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"agent returned non-JSON: {e}; got: {text[:200]!r}"
            ) from e
        if not isinstance(data, dict) or "decisions" not in data:
            raise RuntimeError(
                f"agent JSON missing `decisions`: {text[:200]!r}"
            )

        decisions: list[Decision] = []
        for d in data["decisions"]:
            r = d.get("rationale", {}) or {}
            decision_id = r.get("decision_id") or self._next_id()
            rationale = TradeRationale(
                decision_id=str(decision_id),
                indicators=_to_float_dict(r.get("indicators")),
                numbers=_to_float_dict(r.get("numbers")),
                reasoning=str(r.get("reasoning", "")).strip(),
                pros=tuple(str(x) for x in (r.get("pros") or ())),
                cons=tuple(str(x) for x in (r.get("cons") or ())),
            )
            side = d.get("side")
            decisions.append(
                Decision(
                    asset=str(d["asset"]),
                    action=str(d["action"]),
                    side=side if side in ("long", "short") else None,
                    rationale=rationale,
                )
            )

        return Decisions(
            decisions=tuple(decisions),
            summary=str(data.get("summary", "")).strip(),
        )

    def _next_id(self) -> str:
        self._counter += 1
        return f"d-{self._counter:04d}"


def _fill_to_dict(f: Fill) -> dict[str, Any]:
    return {
        "asset": f.asset,
        "timestamp": f.timestamp.isoformat(),
        "side": f.side,
        "qty": f.qty,
        "price": f.price,
        "fee_paid_usd": f.fee_paid_usd,
        "is_open": f.is_open,
        "realized_pnl_usd": f.realized_pnl_usd,
    }


def _to_float_dict(value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    out: dict[str, float] = {}
    for k, v in value.items():
        try:
            out[str(k)] = float(v)
        except (TypeError, ValueError):
            continue
    return out
