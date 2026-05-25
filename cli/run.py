"""Canonical paper-trader runner — public, open-source entry point.

End-to-end loop on live Hyperliquid data. Each tick:
    1. Fetch the board for N coins.
    2. Tick every open position (mark + funding accrual + stop/take check).
    3. Build `AgentState`, hand it to the synthesizer.
    4. Apply each decision: enter / hold / cut / flip / skip. On enter +
       flip, the sizing layer (tier × regime × ATR stops × leverage) turns
       the naive candidate into a `SizedCandidate`, then the risk gate
       vetoes by exposure / uniqueness / edge.
    5. Render board + open positions (with stop/take/lev columns) +
       agent trace (sizing breakdown on every ENTER decision).
    6. Persist every EXECUTE decision to an audit sqlite store and (if
       `--on-chain` enabled) anchor it on Arc Testnet via Circle's
       Contract Execution API. Periodically dump a JSON snapshot to
       `dashboard/data/snapshot.json` for the static GitHub Pages
       dashboard to read.

Holds a `PortfolioState` + recent-fills deque in memory; durable history
lives in the audit sqlite + on-chain anchors.

The sizing layer needs 1-hour candles; we fetch 5m candles and aggregate
locally (`_aggregate_to_1h`). Default `--window 720` = 60 hours of 5m
= 60+ 1h bars after aggregation = ATR-50 / EMA-50 / ADX-14 all happy
from tick 1 (assuming HL serves enough history on the first WS poll).

Loads `GEMINI_API_KEY` (and `CIRCLE_*` if `--on-chain`) from a `.env`
file in the repo root. The `.env` file is in `.gitignore` — never
committed. See `.env.example` for the expected layout.

Run from the repo root:

    # rule-based, no LLM key needed, no on-chain anchoring
    uv run --with httpx --with ccxt python cli/run.py

    # Gemini agent, paper-only, no on-chain
    uv run --with httpx --with ccxt --with google-genai \\
        python cli/run.py --agent gemini

    # full: Gemini + on-chain anchors to Arc Testnet
    uv run --with httpx --with ccxt --with google-genai \\
        --with cryptography --with eth_utils --with pycryptodome \\
        python cli/run.py --agent gemini --on-chain

    Ctrl-C to exit.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

# Make the repo root importable so `core.*` and `agent.*` resolve when
# running this file directly (no install step needed).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.allocate import allocate  # noqa: E402
from core.contracts import (  # noqa: E402
    AgentState,
    AllocationCandidate,
    CostAssessment,
    Decision,
    Decisions,
    FeeSchedule,
    Fill,
    Forecast,
    MarketData,
    OhlcBar,
    PortfolioState,
    RegimeTag,
    RiskConfig,
    SizedCandidate,
    SizingConfig,
)
from core.cost import CostModel  # noqa: E402
from core.data import HyperliquidSource, fetch_board  # noqa: E402
from core.execute import SimExecutor  # noqa: E402
from core.forecast import BaselineForecast  # noqa: E402
from core.regime import (  # noqa: E402
    BaselineRegimeClassifier,
    apply_btc_override,
)
from core.risk import RiskGate  # noqa: E402
from core.sizing import TierRegimeSizer  # noqa: E402
from core.synthesis import GeminiSynthesizer, RuleSynthesizer, Synthesizer  # noqa: E402
from core.tiers import classify_tier  # noqa: E402
from core.ws import HyperliquidWsSource  # noqa: E402

# Agent infra (audit, on-chain anchor, snapshot writer, auto-pusher).
from agent import audit, snapshot  # noqa: E402
from agent.anchor_worker import AnchorWorker  # noqa: E402
from agent.arc_constants import CHAIN_ID, arcscan_tx  # noqa: E402
from agent.auto_push import AutoPusher  # noqa: E402
from agent.on_chain import CircleAnchor  # noqa: E402

DEFAULT_COINS = [
    "BTC", "ETH", "SOL", "BNB", "DOGE",
    "XRP", "ARB", "AVAX", "LINK", "OP",
]

# Module-level cross-tick render state: the most recent `SizedCandidate`
# per asset, populated when `_apply` opens a position via the sized path.
# Cleared when the position closes (stop/take/cut/flip).
_LAST_SIZED: dict[str, SizedCandidate] = {}

# Minimum 1h bars required by the regime classifier (EMA-50, ATR-50).
_MIN_1H_BARS = 51

_GREEN = "\033[32m"
_DIM = "\033[2m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_CYAN = "\033[36m"
_RESET = "\033[0m"


def _clear() -> None:
    print("\033[2J\033[H", end="", flush=True)


def _load_dotenv() -> None:
    """Minimal dotenv loader: read KEY=VALUE pairs from ../.env into env.

    `.env` takes precedence over already-set shell vars — unusual for
    dotenv libraries but more predictable for our dev workflow where the
    user edits `.env` and expects the next run to use it, regardless of
    stale ``export`` lines in their shell rc.
    """
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ[key] = value


def _age(now: datetime, then: datetime) -> str:
    secs = max(0, int((now - then).total_seconds()))
    m, s = divmod(secs, 60)
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


_EXPLORE_PROMPT = (
    "You are an autonomous perpetual-futures trading agent on Hyperliquid in "
    "EXPLORATION mode. The user is running a paper demo on short candles "
    "(1m–5m) where per-bar expected moves are small (~1-5 bps) and round-trip "
    "costs are larger (~10-15 bps). Single-bar edge will often be negative or "
    "marginal — that is the math, not a bug.\n"
    "\n"
    "Portfolio sizing rules:\n"
    "- ENTER **every** candidate whose edge after cost is within ~30 bps of "
    "zero AND which the portfolio doesn't already hold. Multiple parallel "
    "positions are explicitly desired in this mode — diversification is "
    "useful, and the user wants to see the loop trading actively.\n"
    "- The portfolio's `max_positions` (visible via tool / state) is the "
    "hard cap; you don't need to ration entries below it.\n"
    "- HOLD an open position unless its forecast has clearly flipped "
    "(p_up < 0.4 for longs, > 0.6 for shorts) — then CUT or FLIP.\n"
    "- SKIP only when edge is far worse than zero (< -30 bps): cost "
    "dominates even with optimistic hold extrapolation.\n"
    "\n"
    "Hard rules (still apply):\n"
    "1. Never invent numbers; cite only values from the state JSON.\n"
    "2. Acknowledge negative edge openly in `cons`.\n"
    "3. Output STRICTLY the JSON schema — no prose around it.\n"
    "4. `side` is required for `enter` and `flip`; null otherwise.\n"
    "5. One Decision per asset; do not duplicate.\n"
)


def _make_synthesizer(
    name: str, *, mode: str = "honest", model: str | None = None,
) -> Synthesizer:
    if name == "rule":
        return RuleSynthesizer()
    if name == "gemini":
        kwargs: dict = {}
        if model:
            kwargs["model"] = model
        if mode == "explore":
            kwargs["system_prompt"] = _EXPLORE_PROMPT
        return GeminiSynthesizer(**kwargs)
    raise ValueError(f"unknown synthesizer: {name!r}")


class _AgentScheduler:
    """Decides when to launch the synthesizer; tracks the last decision.

    Async-friendly: separate hooks for *call started* (when we hand work
    off to the background thread) and *call completed* (when its result
    returns). The main render loop is never blocked by the synthesizer.
    """

    def __init__(self, period_s: float) -> None:
        self._period_s = period_s
        self._last_started: datetime | None = None
        self._last_completed: datetime | None = None
        self._last_decisions: Decisions = Decisions(
            decisions=(), summary="(agent has not run yet)"
        )

    def should_call(self, now: datetime) -> bool:
        if self._last_started is None:
            return True
        return (now - self._last_started).total_seconds() >= self._period_s

    def mark_started(self, now: datetime) -> None:
        self._last_started = now

    def record_completion(self, now: datetime, decisions: Decisions) -> None:
        self._last_completed = now
        self._last_decisions = decisions

    @property
    def last(self) -> Decisions:
        return self._last_decisions

    def status_line(self, now: datetime, in_flight: bool) -> str:
        if self._last_started is None:
            return f"agent: idle (period {self._period_s:.0f}s)"
        elapsed_started = (now - self._last_started).total_seconds()
        if in_flight:
            return f"agent: thinking… ({elapsed_started:.0f}s elapsed)"
        if self._last_completed is None:
            return (
                f"agent: started {elapsed_started:.0f}s ago, "
                f"awaiting result"
            )
        elapsed_done = (now - self._last_completed).total_seconds()
        remaining = max(0.0, self._period_s - elapsed_started)
        return (
            f"agent: decided {elapsed_done:.0f}s ago, "
            f"next call in {remaining:.0f}s (every {self._period_s:.0f}s)"
        )


class _AsyncAgent:
    """Run `Synthesizer.decide()` on a background thread.

    `start(state)` hands a state off to a single worker thread; `poll()`
    returns the `Decisions` only when the worker is done. The main loop
    polls every tick and is never blocked by the LLM call — it keeps
    refreshing prices and ticking positions at full speed while Gemini
    thinks for a few seconds.
    """

    def __init__(self, synthesizer: Synthesizer) -> None:
        self._synthesizer = synthesizer
        self._pool = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="agent"
        )
        self._future: Future[Decisions] | None = None

    def in_flight(self) -> bool:
        return self._future is not None and not self._future.done()

    def start(self, state: AgentState) -> None:
        """Submit a new decision. No-op if a previous call is still pending."""
        if self.in_flight():
            return
        self._future = self._pool.submit(self._synthesizer.decide, state)

    def poll(self) -> Decisions | None:
        """Return decisions once ready; None while pending or idle."""
        if self._future is None or not self._future.done():
            return None
        try:
            result = self._future.result()
        except Exception as e:  # noqa: BLE001
            print(f"\n[agent error] {e}", file=sys.stderr)
            result = Decisions(
                decisions=(), summary=f"agent error: {e}"
            )
        finally:
            self._future = None
        return result

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)


# --------------------------------------------------------------- rendering

_TREND_SHORT = {"UP": "UP", "DOWN": "DN", "RANGE": "RG"}
_VOL_SHORT = {"COMPRESSED": "COMP", "NORMAL": "NORM", "EXPANDED": "EXP", "CRISIS": "CRIS"}
_FUNDING_SHORT = {
    "NEUTRAL": "NEU", "BULL_BIAS": "B+", "BEAR_BIAS": "B-",
    "EXTREME_POS": "E+", "EXTREME_NEG": "E-",
}


def _fmt_regime(regime: RegimeTag | None) -> str:
    if regime is None:
        return "—"
    return (
        f"{_TREND_SHORT.get(regime.trend, '?')}·"
        f"{_VOL_SHORT.get(regime.vol, '?')}·"
        f"{_FUNDING_SHORT.get(regime.funding, '?')}"
    )


def render(
    coins: list[str],
    snapshots: dict[str, MarketData],
    forecasts: dict[str, Forecast | None],
    assessments: dict[str, CostAssessment | None],
    candidates: tuple[AllocationCandidate, ...],
    decisions: Decisions,
    portfolio: PortfolioState,
    btc_regime: RegimeTag | None,
    refresh: float,
    agent_name: str,
    agent_status: str,
    ts: datetime,
) -> str:
    out: list[str] = [""]
    eq = portfolio.equity_usd()
    btc_tag = _fmt_regime(btc_regime)
    out.append(
        f"  agora-perp-agent  paper-trader   "
        f"{ts.strftime('%Y-%m-%d %H:%M:%SZ')}    "
        f"agent={agent_name}    btc={btc_tag}    "
        f"equity ${eq:,.2f}"
    )
    out.append("")

    # ---- Opportunities ----
    out.append("  OPPORTUNITIES (ranked by edge after cost)")
    rule = "─" * 92
    out.append("  " + rule)
    out.append(
        f"  {'rank':>4}  {'coin':<6} {'mark':>13}   {'p_up':>5}   "
        f"{'move':>7}   {'cost':>7}   {'edge':>7}   {'act':<5}"
    )
    out.append("  " + rule)

    cand_by_asset = {c.asset: c for c in candidates}
    ranked_order = [c.asset for c in candidates]
    others = [c for c in coins if c not in cand_by_asset]

    for coin in ranked_order:
        snap = snapshots.get(coin)
        cand = cand_by_asset[coin]
        if snap is None:
            continue
        line = (
            f"  {cand.rank + 1:>4}  {coin:<6} {snap.mark_price:>13,.4f}   "
            f"{cand.forecast.p_up:>5.3f}   "
            f"{cand.forecast.expected_move_bps:>+7.2f}   "
            f"{cand.cost.round_trip_bps:>7.2f}   "
            f"{cand.cost.edge_after_cost_bps:>+7.2f}   "
            f"{cand.side.upper():<5}"
        )
        out.append(_GREEN + line + _RESET)

    for coin in others:
        snap = snapshots.get(coin)
        fc = forecasts.get(coin)
        ca = assessments.get(coin)
        if snap is None:
            out.append(_DIM + f"  {'—':>4}  {coin:<6}   (no data)" + _RESET)
            continue
        if fc is None or ca is None:
            out.append(
                _DIM
                + f"  {'—':>4}  {coin:<6} {snap.mark_price:>13,.4f}     —      —         —         —      err  "
                + _RESET
            )
            continue
        line = (
            f"  {'—':>4}  {coin:<6} {snap.mark_price:>13,.4f}   "
            f"{fc.p_up:>5.3f}   {fc.expected_move_bps:>+7.2f}   "
            f"{ca.round_trip_bps:>7.2f}   {ca.edge_after_cost_bps:>+7.2f}   "
            f"skip "
        )
        out.append(_DIM + line + _RESET)

    out.append("")

    # ---- Open positions ----
    out.append("  OPEN POSITIONS (paper, sized)")
    pos_rule = "─" * 118
    out.append("  " + pos_rule)
    if not portfolio.positions:
        out.append(_DIM + "  (none)" + _RESET)
    else:
        out.append(
            f"  {'coin':<6} {'side':<4} {'lev':>5}  {'qty':>12}   "
            f"{'mark':>12}   {'stop':>12}   {'take':>12}   "
            f"{'pnl':>10}   {'fund':>8}   {'age':>6}"
        )
        for asset, pos in portfolio.positions.items():
            unreal = pos.unrealized_pnl_usd()
            lev_str = "—"
            if asset in _LAST_SIZED:
                lev_str = f"{_LAST_SIZED[asset].leverage:.1f}×"
            stop_str = f"{pos.stop_price:,.2f}" if pos.stop_price is not None else "—"
            take_str = f"{pos.take_price:,.2f}" if pos.take_price is not None else "—"
            line = (
                f"  {asset:<6} {pos.side.upper():<4} {lev_str:>5}  "
                f"{pos.qty:>12,.6f}   {pos.last_mark:>12,.4f}   "
                f"{stop_str:>12}   {take_str:>12}   "
                f"{unreal:>+10.4f}   {pos.accrued_funding_usd:>+8.4f}   "
                f"{_age(ts, pos.entry_time):>6}"
            )
            color = _GREEN if unreal > 0 else _RED
            out.append(color + line + _RESET)

    out.append("")

    # ---- Agent trace ----
    out.append(f"  AGENT TRACE  ({agent_name})   {agent_status}")
    out.append("  " + rule)
    summary_line = decisions.summary or "(no summary)"
    out.append(f"  summary: {summary_line}")
    if not decisions.decisions:
        out.append(_DIM + "  (no decisions this tick)" + _RESET)
    for d in decisions.decisions:
        side = d.side.upper() if d.side else "—"
        edge = d.rationale.numbers.get("edge_bps")
        edge_str = f"{edge:+.2f}" if edge is not None else "  ?  "
        head = (
            f"  {d.rationale.decision_id}  {d.asset:<6} "
            f"{d.action.upper():<6} {side:<5}  edge {edge_str:>7}"
        )
        if d.vetoed_reason:
            head_color = _YELLOW
            head += f"  ✗ veto: {d.vetoed_reason}"
        elif d.action == "enter":
            head_color = _CYAN
            head += "  ✓"
        elif d.action == "cut":
            head_color = _RED
            head += "  ✓"
        elif d.action == "flip":
            head_color = _YELLOW
            head += "  ✓"
        else:
            head_color = _DIM
            head += "  ✓"
        out.append(head_color + head + _RESET)

        # On a non-vetoed ENTER, show the sizing one-liner so the user
        # sees tier / regime / lev / stop / take / notional that the
        # math derived. Pulled from `_LAST_SIZED` (set in `_enter_sized`).
        if (
            not d.vetoed_reason
            and d.action == "enter"
            and d.asset in _LAST_SIZED
        ):
            sz = _LAST_SIZED[d.asset]
            out.append(
                _DIM
                + (
                    f"      sized: {sz.tier} · {_fmt_regime(sz.regime)} · "
                    f"{sz.leverage:.1f}× · "
                    f"stop ${sz.stop_price:,.2f} take ${sz.take_price:,.2f} · "
                    f"${sz.notional:,.2f} notional"
                )
                + _RESET
            )

        # Reasoning is only shown for state-changing or vetoed decisions —
        # HOLD/SKIP would just bloat the trace without new information.
        show_reasoning = bool(d.vetoed_reason) or d.action in ("enter", "cut", "flip")
        if show_reasoning and d.rationale.reasoning:
            text = d.rationale.reasoning.strip().replace("\n", " ")
            if len(text) > 110:
                text = text[:107] + "..."
            out.append(_DIM + f"      {text}" + _RESET)

    out.append("")
    out.append(
        f"  balance: ${portfolio.balance_usd:,.2f}   "
        f"realized: ${portfolio.realized_pnl_usd:+,.2f}   "
        f"unrealized: ${portfolio.unrealized_usd():+,.2f}   "
        f"equity: ${eq:,.2f}"
    )
    out.append(f"  (live HL, refresh {refresh:.1f}s, Ctrl-C to exit)")
    return "\n".join(out)


# --------------------------------------------------------------- fetching

def _fetch_and_score(
    coins: list[str],
    source: HyperliquidSource,
    forecaster: BaselineForecast,
    model: CostModel,
    notional: float,
    window: int,
) -> tuple[
    dict[str, MarketData],
    dict[str, Forecast | None],
    dict[str, CostAssessment | None],
]:
    try:
        snapshots = fetch_board(source, coins, window=window)
    except Exception as e:  # noqa: BLE001
        print(f"\n[fetch error] {e}", file=sys.stderr)
        return {}, {}, {}
    forecasts: dict[str, Forecast | None] = {}
    assessments: dict[str, CostAssessment | None] = {}
    for coin, snap in snapshots.items():
        try:
            fc = forecaster.forecast(snap)
        except Exception:  # noqa: BLE001
            forecasts[coin] = None
            assessments[coin] = None
            continue
        forecasts[coin] = fc
        try:
            assessments[coin] = model.assess(fc, snap, notional)
        except Exception:  # noqa: BLE001
            assessments[coin] = None
    return snapshots, forecasts, assessments


# --------------------------------------------------------------- 1h aggregation


def _aggregate_to_1h(bars_5m: tuple[OhlcBar, ...]) -> tuple[OhlcBar, ...]:
    """Roll 5-minute bars into 1-hour bars by hour-of-day bucket.

    Per `regime-classification.md` §1-§3 every indicator runs on 1h
    candles. Since our WS source streams 5m natively, we aggregate
    locally: each hour bucket gets open=first.open, high=max(highs),
    low=min(lows), close=last.close, volume=sum, timestamp=hour-start.

    The trailing partial hour is dropped (< 12 5m bars) so the last
    aggregated bar is fully formed and the regime classifier sees a
    consistent candle set on every tick.
    """
    if not bars_5m:
        return ()
    buckets: dict[datetime, list[OhlcBar]] = {}
    for bar in bars_5m:
        hour = bar.timestamp.replace(minute=0, second=0, microsecond=0)
        buckets.setdefault(hour, []).append(bar)
    hours_sorted = sorted(buckets.keys())
    aggregated: list[OhlcBar] = []
    for i, hour in enumerate(hours_sorted):
        group = buckets[hour]
        if i == len(hours_sorted) - 1 and len(group) < 12:
            continue
        aggregated.append(
            OhlcBar(
                timestamp=hour,
                open=group[0].open,
                high=max(b.high for b in group),
                low=min(b.low for b in group),
                close=group[-1].close,
                volume=sum(b.volume for b in group),
            )
        )
    return tuple(aggregated)


def _compute_btc_regime(
    snapshots: dict[str, MarketData],
    classifier: BaselineRegimeClassifier,
) -> RegimeTag | None:
    """Compute BTC's regime once per tick for use in the alt override.

    Returns ``None`` when BTC is missing from the board or has too few
    1h bars after aggregation (`_MIN_1H_BARS`). Callers must skip the
    BTC-override step in that case — typical only during warm-up.
    """
    btc = snapshots.get("BTC")
    if btc is None:
        return None
    bars_1h = _aggregate_to_1h(btc.bars)
    if len(bars_1h) < _MIN_1H_BARS:
        return None
    try:
        return classifier.classify(
            bars_1h, btc.funding_rate, classify_tier("BTC")
        )
    except (ValueError, ZeroDivisionError):
        return None


# --------------------------------------------------------------- decision apply


def _enter_sized(
    decision: Decision,
    snap: MarketData,
    board: dict[str, tuple[Forecast, CostAssessment]],
    *,
    gate: RiskGate,
    executor: SimExecutor,
    cost_model: CostModel,
    sizer: TierRegimeSizer,
    classifier: BaselineRegimeClassifier,
    portfolio: PortfolioState,
    recent_fills: deque[Fill],
    btc_regime: RegimeTag | None,
    audit_store: "audit.AuditStore | None" = None,
    anchor_worker: AnchorWorker | None = None,
    snap_state: "snapshot.SnapshotState | None" = None,
    agent_wallet: str = "",
) -> Decision:
    """Open a position via the L5.5 sizing path.

    Pipeline: classify_tier → aggregate 5m→1h → regime classify →
    BTC-override (alts only) → TierRegimeSizer.size → stand-down check
    → cost-aware risk gate → SimExecutor.open_sized. Veto reasons are
    explicit so the trace tells you why nothing happened.
    """
    if decision.side is None:
        return replace(decision, vetoed_reason="enter without side")
    entry = board.get(decision.asset)
    if entry is None:
        return replace(decision, vetoed_reason="no board entry")
    fc, ca = entry

    bars_1h = _aggregate_to_1h(snap.bars)
    if len(bars_1h) < _MIN_1H_BARS:
        return replace(
            decision,
            vetoed_reason=f"warm-up ({len(bars_1h)}/{_MIN_1H_BARS} 1h bars)",
        )

    tier = classify_tier(decision.asset)
    try:
        local_regime = classifier.classify(
            bars_1h, snap.funding_rate, tier
        )
    except (ValueError, ZeroDivisionError) as e:
        return replace(decision, vetoed_reason=f"regime err: {e}")

    if decision.asset != "BTC" and btc_regime is not None:
        regime = apply_btc_override(local_regime, btc_regime)
    else:
        regime = local_regime

    seed_cand = AllocationCandidate(
        asset=decision.asset,
        side=decision.side,
        notional=ca.notional,
        forecast=fc,
        cost=ca,
        rank=0,
    )
    try:
        sized = sizer.size(
            seed_cand, snap, bars_1h, portfolio, regime, tier,
        )
    except (ValueError, ZeroDivisionError) as e:
        return replace(decision, vetoed_reason=f"sizing err: {e}")

    if sized.qty <= 0.0:
        return replace(
            decision,
            vetoed_reason=(
                f"playbook stand-down "
                f"({regime.trend}×{regime.vol}×{regime.funding})"
            ),
        )

    # Risk gate sees the sized notional; cost rescored against it.
    sized_cost = cost_model.assess(fc, snap, sized.notional)
    gate_cand = AllocationCandidate(
        asset=decision.asset,
        side=decision.side,
        notional=sized.notional,
        forecast=fc,
        cost=sized_cost,
        rank=0,
    )
    verdict = gate.evaluate(gate_cand, portfolio)
    if not verdict.approved:
        return replace(decision, vetoed_reason=verdict.reason)

    fill = executor.open_sized(sized, snap, portfolio)
    recent_fills.append(fill)
    _LAST_SIZED[decision.asset] = sized

    # Audit + on-chain anchor + snapshot patch (best-effort: never blocks
    # the fill on Circle API latency).
    if audit_store is not None:
        record = audit.new_record(
            decision_id=decision.rationale.decision_id,
            agent_address=agent_wallet,
            chain_id=CHAIN_ID,
            venue="hyperliquid-paper",
            asset=decision.asset,
            side=decision.side,
            verdict=audit.Verdict.EXECUTE,
            input_hashes={
                "market": audit.hash_input(
                    {
                        "asset": snap.asset,
                        "mark": snap.mark_price,
                        "funding_rate": snap.funding_rate,
                        "book_depth": snap.book_depth,
                        "timestamp": snap.timestamp.isoformat(),
                    }
                ),
                "forecast": audit.hash_input(
                    {
                        "p_up": fc.p_up,
                        "expected_move_bps": fc.expected_move_bps,
                        "confidence": fc.confidence,
                    }
                ),
                "cost": audit.hash_input(
                    {
                        "round_trip_bps": ca.round_trip_bps,
                        "edge_after_cost_bps": ca.edge_after_cost_bps,
                    }
                ),
            },
            sized={
                "qty": sized.qty,
                "notional_usd": sized.notional,
                "leverage": sized.leverage,
                "stop_price": sized.stop_price,
                "take_price": sized.take_price,
                "tier": str(sized.tier),
                "regime_trend": sized.regime.trend,
                "regime_vol": sized.regime.vol,
                "regime_funding": sized.regime.funding,
            },
            reasoning=decision.rationale.reasoning,
        )
        audit_store.append(record)
        regime_short = _fmt_regime(sized.regime)
        if snap_state is not None:
            snap_state.record_open(
                asset=decision.asset,
                side=decision.side,
                qty=sized.qty,
                leverage=sized.leverage,
                entry_price=fill.price,
                stop_price=sized.stop_price,
                take_price=sized.take_price,
                tier=str(sized.tier),
                regime=regime_short,
                notional_usd=sized.notional,
                audit_id=record.audit_id,
                decision_id=decision.rationale.decision_id,
            )
            snap_state.record_decision(
                {
                    "audit_id": record.audit_id,
                    "decision_id": decision.rationale.decision_id,
                    "asset": decision.asset,
                    "side": decision.side,
                    "action": "enter",
                    "verdict": audit.Verdict.EXECUTE.value,
                    "tier": str(sized.tier),
                    "regime": regime_short,
                    "leverage": sized.leverage,
                    "stop_price": sized.stop_price,
                    "take_price": sized.take_price,
                    "notional_usd": sized.notional,
                    "qty": sized.qty,
                    "entry_price": fill.price,
                    "edge_after_cost_bps": ca.edge_after_cost_bps,
                    "reasoning": decision.rationale.reasoning,
                    "decided_at_iso": datetime.now(timezone.utc).isoformat(),
                    "anchor_state": "pending"
                    if anchor_worker is not None else "off",
                    "arc_tx_hash": None,
                    "arcscan_url": None,
                }
            )
        if anchor_worker is not None:
            anchor_worker.submit(record, asset=decision.asset)

    return decision


def _apply(
    decision: Decision,
    snap: MarketData,
    board: dict[str, tuple[Forecast, CostAssessment]],
    *,
    gate: RiskGate,
    executor: SimExecutor,
    cost_model: CostModel,
    sizer: TierRegimeSizer,
    classifier: BaselineRegimeClassifier,
    portfolio: PortfolioState,
    recent_fills: deque[Fill],
    btc_regime: RegimeTag | None,
    audit_store: "audit.AuditStore | None" = None,
    anchor_worker: AnchorWorker | None = None,
    snap_state: "snapshot.SnapshotState | None" = None,
    agent_wallet: str = "",
) -> Decision:
    """Apply one decision to the portfolio. Return the (possibly vetoed) decision."""
    if decision.action == "enter":
        if portfolio.has(decision.asset):
            return replace(decision, vetoed_reason="already held")
        return _enter_sized(
            decision, snap, board,
            gate=gate, executor=executor, cost_model=cost_model,
            sizer=sizer, classifier=classifier,
            portfolio=portfolio, recent_fills=recent_fills,
            btc_regime=btc_regime,
            audit_store=audit_store, anchor_worker=anchor_worker,
            snap_state=snap_state, agent_wallet=agent_wallet,
        )

    if decision.action == "cut":
        if not portfolio.has(decision.asset):
            return replace(decision, vetoed_reason="no position to cut")
        pos = portfolio.positions[decision.asset]
        entry_time_iso = pos.entry_time.isoformat()
        entry_price_pre = pos.entry_price
        qty_pre = pos.qty
        side_pre = pos.side
        funding_pre = pos.accrued_funding_usd
        fill = executor.close(decision.asset, snap, portfolio)
        recent_fills.append(fill)
        _LAST_SIZED.pop(decision.asset, None)
        if snap_state is not None:
            snap_state.record_close(
                asset=decision.asset,
                side=side_pre,
                qty=qty_pre,
                entry_price=entry_price_pre,
                exit_price=fill.price,
                realized_pnl_usd=fill.realized_pnl_usd,
                entry_time_iso=entry_time_iso,
                exit_time_iso=fill.timestamp.isoformat(),
                accrued_funding_usd=funding_pre,
            )
        return decision

    if decision.action == "flip":
        if portfolio.has(decision.asset):
            pos = portfolio.positions[decision.asset]
            entry_time_iso = pos.entry_time.isoformat()
            entry_price_pre = pos.entry_price
            qty_pre = pos.qty
            side_pre = pos.side
            funding_pre = pos.accrued_funding_usd
            fill = executor.close(decision.asset, snap, portfolio)
            recent_fills.append(fill)
            _LAST_SIZED.pop(decision.asset, None)
            if snap_state is not None:
                snap_state.record_close(
                    asset=decision.asset,
                    side=side_pre,
                    qty=qty_pre,
                    entry_price=entry_price_pre,
                    exit_price=fill.price,
                    realized_pnl_usd=fill.realized_pnl_usd,
                    entry_time_iso=entry_time_iso,
                    exit_time_iso=fill.timestamp.isoformat(),
                    accrued_funding_usd=funding_pre,
                )
        return _enter_sized(
            decision, snap, board,
            gate=gate, executor=executor, cost_model=cost_model,
            sizer=sizer, classifier=classifier,
            portfolio=portfolio, recent_fills=recent_fills,
            btc_regime=btc_regime,
            audit_store=audit_store, anchor_worker=anchor_worker,
            snap_state=snap_state, agent_wallet=agent_wallet,
        )

    # hold / skip → no-op
    return decision


# --------------------------------------------------------------- step

def step(
    coins: list[str],
    source: HyperliquidSource,
    forecaster: BaselineForecast,
    cost_model: CostModel,
    executor: SimExecutor,
    gate: RiskGate,
    sizer: TierRegimeSizer,
    classifier: BaselineRegimeClassifier,
    agent: _AsyncAgent,
    scheduler: _AgentScheduler,
    portfolio: PortfolioState,
    recent_fills: deque[Fill],
    notional: float,
    window: int,
    max_positions: int,
    *,
    audit_store: "audit.AuditStore | None" = None,
    anchor_worker: AnchorWorker | None = None,
    snap_state: "snapshot.SnapshotState | None" = None,
    agent_wallet: str = "",
) -> tuple[
    dict[str, MarketData],
    dict[str, Forecast | None],
    dict[str, CostAssessment | None],
    tuple[AllocationCandidate, ...],
    Decisions,
    RegimeTag | None,
]:
    """One tick: fetch → tick → check stops → drain agent → maybe-launch agent.

    The synthesizer runs on a background thread (`_AsyncAgent`). Each
    tick we drain any decisions that just completed and apply them, then
    launch a new call only if we're idle and the scheduler is due. The
    LLM never blocks the render loop.

    Returns the BTC regime tag in addition to the usual board state so
    the renderer can show it next to the BTC-override-affected alts.
    """
    snapshots, forecasts, assessments = _fetch_and_score(
        coins, source, forecaster, cost_model, notional, window
    )

    # Mark + funding accrual + stop/take check on every open position.
    for asset in list(portfolio.positions):
        snap = snapshots.get(asset)
        if snap is None:
            continue
        executor.tick(snap, portfolio)
        # Capture pre-close state for snapshot history before check_stops
        # closes the position (post-close, the Position dict has no entry).
        pos = portfolio.positions.get(asset)
        if pos is not None:
            pre_entry_time_iso = pos.entry_time.isoformat()
            pre_entry_price = pos.entry_price
            pre_qty = pos.qty
            pre_side = pos.side
            pre_funding = pos.accrued_funding_usd
        else:
            pre_entry_time_iso = pre_entry_price = pre_qty = pre_side = pre_funding = None  # type: ignore[assignment]
        closed = executor.check_stops(snap, portfolio)
        if closed is not None:
            recent_fills.append(closed)
            _LAST_SIZED.pop(asset, None)
            if snap_state is not None and pre_qty is not None:
                snap_state.record_close(
                    asset=asset,
                    side=pre_side,
                    qty=pre_qty,
                    entry_price=pre_entry_price,
                    exit_price=closed.price,
                    realized_pnl_usd=closed.realized_pnl_usd,
                    entry_time_iso=pre_entry_time_iso,
                    exit_time_iso=closed.timestamp.isoformat(),
                    accrued_funding_usd=pre_funding,
                )

    board: dict[str, tuple[Forecast, CostAssessment]] = {}
    for asset, fc in forecasts.items():
        ca = assessments.get(asset)
        if fc is not None and ca is not None:
            board[asset] = (fc, ca)

    candidates = allocate(
        board, max_positions=max_positions, require_tradeable=False
    )

    now = datetime.now(timezone.utc)

    # BTC regime computed once per tick (drives the alt override).
    btc_regime = _compute_btc_regime(snapshots, classifier)

    # 1. Drain any decision that completed since the last tick, and apply it.
    completed = agent.poll()
    if completed is not None:
        applied: list[Decision] = []
        for dec in completed.decisions:
            snap = snapshots.get(dec.asset)
            if snap is None:
                applied.append(replace(dec, vetoed_reason="no snapshot"))
                continue
            applied.append(
                _apply(
                    dec, snap, board,
                    gate=gate, executor=executor, cost_model=cost_model,
                    sizer=sizer, classifier=classifier,
                    portfolio=portfolio, recent_fills=recent_fills,
                    btc_regime=btc_regime,
                    audit_store=audit_store, anchor_worker=anchor_worker,
                    snap_state=snap_state, agent_wallet=agent_wallet,
                )
            )
        scheduler.record_completion(
            now,
            Decisions(decisions=tuple(applied), summary=completed.summary),
        )

    # 2. Launch a new agent call if idle + due + we have signal.
    have_signal = bool(candidates) or bool(portfolio.positions)
    if (
        not agent.in_flight()
        and scheduler.should_call(now)
        and have_signal
    ):
        state = AgentState(
            timestamp=now,
            board=board,
            candidates=candidates,
            portfolio=portfolio,
            recent_fills=tuple(recent_fills),
        )
        scheduler.mark_started(now)
        agent.start(state)

    while len(recent_fills) > 20:
        recent_fills.popleft()

    if snap_state is not None:
        snap_state.tick()
        snap_state.record_equity(portfolio.equity_usd())
        snap_state.agent_status = (
            "thinking" if agent.in_flight() else "idle"
        )

    return (
        snapshots,
        forecasts,
        assessments,
        candidates,
        scheduler.last,
        btc_regime,
    )


# --------------------------------------------------------------- main

def main() -> int:
    _load_dotenv()

    ap = argparse.ArgumentParser(description="agora-perp-agent paper trader")
    ap.add_argument(
        "--coins",
        nargs="+",
        default=DEFAULT_COINS,
        help=f"HL coin symbols (default: {' '.join(DEFAULT_COINS)})",
    )
    ap.add_argument(
        "--refresh",
        type=float,
        default=1.0,
        help="seconds per tick; at 1s × 10 coins ≈ 21 RPS to HL /info",
    )
    ap.add_argument(
        "--notional", type=float, default=100.0,
        help=(
            "Baseline notional USD passed to the cost model for the "
            "pre-decision edge_after_cost ranking. The actual fill "
            "notional comes from the L5.5 sizer (tier × regime × ATR)."
        ),
    )
    ap.add_argument(
        "--window", type=int, default=720,
        help=(
            "5-minute OHLC bars per fetch. Defaults to 720 (= 60 hours) "
            "because the L5.5 sizing layer aggregates 5m→1h locally and "
            "the regime classifier needs ≥ 51 1h bars (EMA-50/ATR-50)."
        ),
    )
    ap.add_argument(
        "--hold-hours", type=float, default=4.0,
        help=(
            "Expected hold horizon for the sizer's funding-drag check. "
            "Day-trade default 4h; LFT 24h+."
        ),
    )
    ap.add_argument(
        "--base-risk", type=float, default=0.005,
        help=(
            "Equity fraction risked per trade. Default 0.005 = 0.5%% per "
            "`risk-sizing-methods.md` §1. Bump to 0.02-0.03 for demo "
            "(more visible positions); the docs cap at 0.01 long-term."
        ),
    )
    ap.add_argument(
        "--max-leverage", type=float, default=None,
        help=(
            "Override the per-tier operational cap (default 5× T1, 3× "
            "T2-3, 1.5× T4) with a single ceiling. Useful for demo to "
            "let `liq_safety_cap` do the binding instead of the "
            "conservative tier cap. Try 20 for high-visibility runs."
        ),
    )
    ap.add_argument(
        "--max-positions", type=int, default=20,
        help="hard cap on concurrent positions (default 20 = effectively unlimited at 10 coins)",
    )
    ap.add_argument(
        "--max-notional", type=float, default=250.0,
        help=(
            "Risk-gate cap on per-position notional USD. Default 250 "
            "fits the $1000 starting balance; bump alongside "
            "--starting-balance for demo runs."
        ),
    )
    ap.add_argument(
        "--max-exposure", type=float, default=750.0,
        help=(
            "Risk-gate cap on aggregate open notional USD. Default 750 "
            "is the conservative paper-trader floor; bump for demo."
        ),
    )
    ap.add_argument("--starting-balance", type=float, default=1_000.0)
    ap.add_argument(
        "--min-edge-bps", type=float, default=-15.0,
        help=(
            "Risk-gate edge floor in bps. Trades with edge_after_cost "
            "below this value are vetoed. Default -15 lets the agent "
            "open marginal-negative-edge trades on short candles where "
            "single-bar expected moves are smaller than round-trip "
            "costs — useful for demos. Set to 0 for strict-positive-"
            "edge filtering."
        ),
    )
    ap.add_argument(
        "--agent", choices=("rule", "gemini"), default="rule",
        help="synthesizer: 'rule' (deterministic, no API) or 'gemini' (LLM).",
    )
    ap.add_argument(
        "--model", default=None,
        help=(
            "Gemini model id; default is the SDK pro preview. "
            "Use 'gemini-3.1-flash-lite' or 'gemini-2.5-flash' for ~10× "
            "faster, ~10× cheaper decisions with slightly shallower reasoning."
        ),
    )
    ap.add_argument(
        "--mode", choices=("honest", "explore"), default="honest",
        help=(
            "Gemini system prompt. 'honest' skips negative-edge trades; "
            "'explore' is willing to enter marginal-negative edges on short "
            "candles, useful for demos. Ignored for --agent rule."
        ),
    )
    ap.add_argument(
        "--agent-every", type=float, default=30.0,
        help="seconds between synthesizer calls; default 30s (board still refreshes every tick).",
    )
    ap.add_argument(
        "--interval", default="5m",
        help="HL candle interval driving every indicator (default 5m).",
    )
    ap.add_argument(
        "--source", choices=("rest", "ws"), default="ws",
        help=(
            "Market data backend. 'ws' (default) streams the L2 book over "
            "WebSocket via ccxt and polls OHLCV every 30s — comfortably "
            "inside HL rate limits even at 10+ coins. 'rest' uses our "
            "native HTTP client (1s × 10 coins is right at the rate limit)."
        ),
    )
    ap.add_argument(
        "--audit-db", default=str(_REPO_ROOT / "audit.sqlite"),
        help=(
            "Path to the append-only sqlite audit store. Every decision "
            "(EXECUTE/DEFER/CHALLENGE/REJECT) is written here before any "
            "external effect — see `agent/audit.py` for the schema."
        ),
    )
    ap.add_argument(
        "--on-chain", action="store_true",
        help=(
            "Anchor every EXECUTE-action open to Arc Testnet via Circle's "
            "Contract Execution API (ERC-8183 createJob). Requires "
            "CIRCLE_API_KEY, CIRCLE_ENTITY_SECRET, CIRCLE_WALLET_ID, "
            "CIRCLE_WALLET_ADDRESS in .env (run scripts/circle_setup.py)."
        ),
    )
    ap.add_argument(
        "--snapshot-out", default=str(_REPO_ROOT / "dashboard" / "data" / "snapshot.json"),
        help=(
            "JSON snapshot path the dashboard reads. Written atomically "
            "every `--snapshot-every` seconds with current portfolio + "
            "decision history + trade history + equity curve."
        ),
    )
    ap.add_argument(
        "--snapshot-every", type=float, default=5.0,
        help="seconds between snapshot.json writes (default 5).",
    )
    ap.add_argument(
        "--no-snapshot", action="store_true",
        help="Skip writing the JSON snapshot (run pure terminal).",
    )
    ap.add_argument(
        "--auto-push", action="store_true",
        help=(
            "Background-commit + push `dashboard/data/snapshot.json` "
            "every `--push-every` seconds so the deployed GitHub Pages "
            "dashboard stays fresh. Requires this checkout's `origin` "
            "to already authenticate (SSH key / token in place)."
        ),
    )
    ap.add_argument(
        "--push-every", type=float, default=300.0,
        help=(
            "Seconds between snapshot commits + pushes; clamped to a "
            "60s floor. Default 300 (= 5 min); faster than ~120s is "
            "wasted bandwidth since GH Pages takes ~60s to redeploy."
        ),
    )
    args = ap.parse_args()

    # `--max-leverage` overrides the per-tier operational cap (kept in
    # `core.leverage._OPERATIONAL_CAP`). Monkey-patched here at startup
    # so every subsequent `choose_leverage` call sees the new ceiling.
    # Other caps (venue, liq-safety) still bind — typical effect for
    # BTC/ETH is the liq-safety cap kicking in around 12-18× at our
    # stop distances.
    if args.max_leverage is not None:
        from core import leverage as _leverage_mod
        for _tier in list(_leverage_mod._OPERATIONAL_CAP.keys()):
            _leverage_mod._OPERATIONAL_CAP[_tier] = args.max_leverage

    source: HyperliquidSource | HyperliquidWsSource
    if args.source == "ws":
        source = HyperliquidWsSource(
            args.coins,
            interval=args.interval,
            bars_limit=args.window,
        )
        print("  warming up ws subscriptions…", file=sys.stderr)
        if not source.wait_ready(timeout=15.0):
            print(
                "  ws warm-up timed out after 15s — proceeding with "
                "partial cache; missing coins will show (no data) until "
                "their first push arrives.",
                file=sys.stderr,
            )
    else:
        source = HyperliquidSource(interval=args.interval)
    forecaster = BaselineForecast()
    schedule = FeeSchedule()
    cost_model = CostModel(schedule, hold_minutes=1.0)
    executor = SimExecutor(schedule)
    gate = RiskGate(
        RiskConfig(
            max_positions=args.max_positions,
            min_edge_after_cost_bps=args.min_edge_bps,
            max_notional_per_position_usd=args.max_notional,
            max_total_exposure_usd=args.max_exposure,
        )
    )
    sizer = TierRegimeSizer(
        config=SizingConfig(base_risk_fraction=args.base_risk),
        hold_hours=args.hold_hours,
    )
    classifier = BaselineRegimeClassifier()
    portfolio = PortfolioState.empty(starting_balance=args.starting_balance)
    recent_fills: deque[Fill] = deque(maxlen=20)

    try:
        synthesizer = _make_synthesizer(
            args.agent, mode=args.mode, model=args.model
        )
    except RuntimeError as e:
        print(f"  could not init agent: {e}", file=sys.stderr)
        return 2

    scheduler = _AgentScheduler(period_s=args.agent_every)
    agent_runner = _AsyncAgent(synthesizer)

    # --- audit + on-chain anchor + snapshot wiring ----------------------
    audit_store = audit.AuditStore(args.audit_db)
    agent_wallet = os.environ.get("CIRCLE_WALLET_ADDRESS", "")

    snap_state: snapshot.SnapshotState | None = None
    snapshot_path: Path | None = None
    if not args.no_snapshot:
        snapshot_path = Path(args.snapshot_out)
        snap_state = snapshot.SnapshotState(
            starting_balance_usd=args.starting_balance,
            agent_wallet=agent_wallet,
        )
        # Hydrate decision history from the persistent audit store so the
        # dashboard shows continuous history across restarts.
        try:
            snapshot.hydrate_from_audit_store(snap_state, audit_store)
            print(
                f"  hydrated {len(snap_state.recent_decisions)} prior "
                f"decisions from {args.audit_db}",
                file=sys.stderr,
            )
        except Exception as e:  # noqa: BLE001
            print(f"  hydrate failed (continuing): {e}", file=sys.stderr)

    anchor_client: CircleAnchor | None = None
    anchor_worker: AnchorWorker | None = None
    if args.on_chain:
        try:
            anchor_client = CircleAnchor()  # reads creds from env
        except RuntimeError as e:
            print(
                f"  could not init Circle anchor (--on-chain disabled): {e}",
                file=sys.stderr,
            )
        else:
            def _on_anchor_complete(audit_id, asset, result, err):
                if snap_state is None:
                    return
                if result is not None and result.arc_tx_hash:
                    snap_state.update_anchor(
                        asset=asset,
                        audit_id=audit_id,
                        arc_tx_hash=result.arc_tx_hash,
                        arcscan_url=arcscan_tx(result.arc_tx_hash),
                        state="complete",
                    )
                elif err is not None:
                    snap_state.update_anchor(
                        asset=asset, audit_id=audit_id,
                        arc_tx_hash=None, arcscan_url=None, state="error",
                    )

            anchor_worker = AnchorWorker(
                anchor_client, audit_store, on_complete=_on_anchor_complete,
            )
            print(
                f"  on-chain anchoring enabled (wallet {agent_wallet})",
                file=sys.stderr,
            )

    last_snapshot_at = 0.0

    # --- snapshot auto-pusher to GitHub Pages ---------------------------
    auto_pusher: AutoPusher | None = None
    if args.auto_push and not args.no_snapshot and snapshot_path is not None:
        auto_pusher = AutoPusher(
            snapshot_path=snapshot_path,
            repo_root=_REPO_ROOT,
            push_every_s=args.push_every,
        )
        auto_pusher.start()
        print(
            f"  auto-push enabled (every {args.push_every:.0f}s) → origin",
            file=sys.stderr,
        )

    try:
        while True:
            loop_start = time.monotonic()
            (
                snapshots, forecasts, assessments,
                candidates, decisions, btc_regime,
            ) = step(
                args.coins,
                source, forecaster, cost_model, executor, gate,
                sizer, classifier,
                agent_runner, scheduler, portfolio, recent_fills,
                args.notional, args.window, args.max_positions,
                audit_store=audit_store, anchor_worker=anchor_worker,
                snap_state=snap_state, agent_wallet=agent_wallet,
            )
            now = datetime.now(timezone.utc)
            _clear()
            print(
                render(
                    args.coins, snapshots, forecasts, assessments,
                    candidates, decisions, portfolio,
                    btc_regime, args.refresh, args.agent,
                    scheduler.status_line(now, agent_runner.in_flight()),
                    now,
                )
            )
            # Snapshot dump (rate-limited).
            if (
                snap_state is not None and snapshot_path is not None
                and time.monotonic() - last_snapshot_at >= args.snapshot_every
            ):
                try:
                    snapshot.write(snap_state, portfolio, snapshot_path)
                    last_snapshot_at = time.monotonic()
                except OSError as e:
                    print(
                        f"\n[snapshot write failed] {e}", file=sys.stderr
                    )

            # Sleep only the remainder of the target interval — fetch may
            # already have consumed all of it on slow ticks.
            elapsed = time.monotonic() - loop_start
            time.sleep(max(0.0, args.refresh - elapsed))
    except KeyboardInterrupt:
        print(
            f"\n  bye. final equity ${portfolio.equity_usd():,.2f}, "
            f"realized ${portfolio.realized_pnl_usd:+,.2f}"
        )
    finally:
        # Flush a final snapshot so the dashboard reflects the last state.
        if snap_state is not None and snapshot_path is not None:
            try:
                snapshot.write(snap_state, portfolio, snapshot_path)
            except OSError:
                pass
        if auto_pusher is not None:
            auto_pusher.close()
        if anchor_worker is not None:
            anchor_worker.close()
        if anchor_client is not None:
            anchor_client.close()
        audit_store.close()
        agent_runner.close()
        if isinstance(source, HyperliquidWsSource):
            source.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
