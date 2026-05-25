"""core data contracts.

Frozen dataclasses passed between engine components, plus the mutable
state that paper trading needs (`Position`, `PortfolioState`). Timestamps
are stdlib `datetime` so this module stays dependency-free.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum


@dataclass(frozen=True, slots=True)
class OhlcBar:
    """one candle.

    Attributes:
        timestamp: candle open time (UTC).
        open: open price.
        high: high price.
        low: low price.
        close: close price.
        volume: traded volume over the candle.
    """

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True, slots=True)
class MarketData:
    """one market snapshot for one tick.

    Attributes:
        asset: market symbol, e.g. "BTC".
        timestamp: snapshot time (UTC).
        mark_price: current mark price.
        funding_rate: current funding rate per its cadence. Its effect is
            near-zero at a minute horizon but it is carried for honest
            accounting on longer holds.
        bars: recent candles, oldest first.
        book_depth: notional within a band of mid, for slippage modeling.
            None when not collected.
    """

    asset: str
    timestamp: datetime
    mark_price: float
    funding_rate: float
    bars: tuple[OhlcBar, ...]
    book_depth: float | None = None


@dataclass(frozen=True, slots=True)
class Forecast:
    """directional forecast for one asset over the next tick.

    ``p_up + p_down`` need not sum to 1; the remaining mass is the
    implicit "flat / no edge" probability.

    Attributes:
        asset: market symbol.
        timestamp: forecast time (the latest bar's time).
        p_up: probability price rises over the horizon (0..1).
        p_down: probability price falls over the horizon (0..1).
        expected_move_bps: signed magnitude estimate in basis points.
        confidence: model self-reported confidence (0..1).
    """

    asset: str
    timestamp: datetime
    p_up: float
    p_down: float
    expected_move_bps: float
    confidence: float


@dataclass(frozen=True, slots=True)
class FeeSchedule:
    """venue fee and slippage parameters used by ``CostModel``.

    Defaults are the Hyperliquid base tier (maker 1.5 bps / taker 4.5 bps,
    hourly funding cadence). ``slippage_k`` is the dimensionless constant in
    the sqrt-law impact formula ``impact ≈ k · √(notional/depth)``; default
    ``0.005`` is a working baseline for top perps (BTC/ETH on HL) — calibrate
    against actual fills per asset before any number leaves the lab.
    ``flat_slippage_bps`` is the conservative per-side fallback applied when
    ``MarketData.book_depth`` is unavailable.

    Attributes:
        maker_bps: maker fee in basis points of notional.
        taker_bps: taker fee in basis points of notional.
        funding_period_hours: cadence at which the venue settles funding.
        slippage_k: dimensionless slope of the sqrt-law impact model.
        flat_slippage_bps: per-side fallback slippage when depth is missing.
    """

    maker_bps: float = 1.5
    taker_bps: float = 4.5
    funding_period_hours: float = 1.0
    slippage_k: float = 0.005
    flat_slippage_bps: float = 30.0


@dataclass(frozen=True, slots=True)
class CostAssessment:
    """round-trip cost breakdown for one candidate trade.

    All ``*_bps`` fields are in basis points of ``notional``. Each leg is
    already round-trip (entry + exit), so ``round_trip_bps`` is just the sum
    of the three legs. ``edge_after_cost_bps`` is the forecast's gross move
    in the trade's direction minus ``round_trip_bps``. ``is_tradeable`` is a
    convenience flag — the real go/no-go gate is L5 risk.

    Attributes:
        asset: market symbol.
        notional: trade size in quote currency.
        fee_bps: round-trip taker fee.
        slippage_bps: round-trip slippage (sqrt-law or flat fallback).
        funding_bps: signed funding over the hold; positive = cost to us.
        round_trip_bps: ``fee_bps + slippage_bps + funding_bps``.
        breakeven_bps: gross move needed to overcome the round-trip cost.
        edge_after_cost_bps: signed expected move minus round-trip cost.
        is_tradeable: ``True`` iff ``edge_after_cost_bps`` is positive.
    """

    asset: str
    notional: float
    fee_bps: float
    slippage_bps: float
    funding_bps: float
    round_trip_bps: float
    breakeven_bps: float
    edge_after_cost_bps: float
    is_tradeable: bool


@dataclass(frozen=True, slots=True)
class AllocationCandidate:
    """one ranked trade idea from `allocate()`.

    Carries the trade's direction and intended size alongside the inputs
    that justified it (`forecast`, `cost`) and its rank in the board (0 =
    best). It is a *proposed* trade — the risk gate and the LLM agent still
    get to veto or override.

    Attributes:
        asset: market symbol.
        side: ``"long"`` or ``"short"``, derived from the forecast.
        notional: intended trade size (same as the value the cost model
            was evaluated at).
        forecast: the directional view that drove the candidate.
        cost: the cost assessment for ``notional`` on this snapshot.
        rank: position in the ranked board, 0-indexed; 0 = best edge.
    """

    asset: str
    side: str
    notional: float
    forecast: Forecast
    cost: CostAssessment
    rank: int


@dataclass(slots=True)
class Position:
    """one open paper position; mutable across ticks.

    Unlike the rest of `core/contracts.py`, this dataclass is *not* frozen —
    a live position genuinely is state: its mark moves each tick, funding
    accrues, and unrealized PnL is read on demand. Mutability is honest here.

    Attributes:
        asset: market symbol.
        side: ``"long"`` or ``"short"``.
        qty: base units (e.g. BTC), always positive; direction is in `side`.
        entry_price: realized fill price at open (mark ± entry slippage).
        entry_time: timestamp at open.
        last_mark: latest mark observed; the basis for unrealized PnL.
        last_funding_ts: timestamp of the most recent funding accrual.
        accrued_funding_usd: cumulative funding cashflow, signed. Positive
            means we have collected; negative means we have paid.
    """

    asset: str
    side: str
    qty: float
    entry_price: float
    entry_time: datetime
    last_mark: float
    last_funding_ts: datetime
    accrued_funding_usd: float = 0.0
    stop_price: float | None = None
    take_price: float | None = None
    stop_take_plan: StopTakePlan | None = None

    def unrealized_pnl_usd(self) -> float:
        """Mark-to-market PnL on the price leg, in USD; excludes funding."""
        if self.side == "long":
            return (self.last_mark - self.entry_price) * self.qty
        return (self.entry_price - self.last_mark) * self.qty


@dataclass(slots=True)
class PortfolioState:
    """mutable account state for paper trading.

    Carries the free cash balance and a map of open positions, keyed by
    asset. ``balance_usd`` updates on every realized cashflow (fee at open,
    realized PnL at close). Unrealized PnL and accrued funding live on the
    positions themselves; equity sums everything.

    Attributes:
        starting_balance_usd: initial cash; never changes after creation.
        balance_usd: current free cash, including all realized cashflows.
        positions: open positions keyed by asset.
    """

    starting_balance_usd: float
    balance_usd: float
    positions: dict[str, Position] = field(default_factory=dict)

    @classmethod
    def empty(cls, starting_balance: float = 1_000_000.0) -> PortfolioState:
        """Create an empty portfolio whose balance equals `starting_balance`."""
        return cls(
            starting_balance_usd=starting_balance,
            balance_usd=starting_balance,
        )

    def has(self, asset: str) -> bool:
        """True iff there is an open position on `asset`."""
        return asset in self.positions

    @property
    def realized_pnl_usd(self) -> float:
        """Cumulative realized PnL since inception, in USD."""
        return self.balance_usd - self.starting_balance_usd

    def unrealized_usd(self) -> float:
        """Sum of price-leg unrealized PnL plus accrued funding, across positions."""
        return sum(
            p.unrealized_pnl_usd() + p.accrued_funding_usd
            for p in self.positions.values()
        )

    def equity_usd(self) -> float:
        """Total account value: ``balance + Σ(unrealized + accrued_funding)``."""
        return self.balance_usd + self.unrealized_usd()

    def exposure_usd(self) -> float:
        """Sum of mark-priced notional across open positions, in USD."""
        return sum(p.qty * p.last_mark for p in self.positions.values())


@dataclass(frozen=True, slots=True)
class Fill:
    """historical execution event; frozen because it is a fact of the past.

    Emitted by `SimExecutor.open` and `SimExecutor.close`, appended to the
    trace log, and (later, at L9) hashed onto Arc as a verifiable receipt.

    Attributes:
        asset: market symbol.
        timestamp: execution time (the market snapshot's timestamp at fill).
        side: the position's side, ``"long"`` or ``"short"`` (not the action).
        qty: base units filled.
        price: realized fill price after slippage.
        fee_paid_usd: taker fee paid on this fill.
        is_open: True if this fill opened the position; False if it closed.
        realized_pnl_usd: ``0`` for opening fills; signed realized PnL on
            closing fills (``price_pnl + accrued_funding − exit_fee``).
    """

    asset: str
    timestamp: datetime
    side: str
    qty: float
    price: float
    fee_paid_usd: float
    is_open: bool
    realized_pnl_usd: float = 0.0


@dataclass(frozen=True, slots=True)
class RiskConfig:
    """deterministic limits applied by `RiskGate` before each open.

    Defaults are paper-trader-conservative — they prevent the demo from
    blowing up but are not calibrated against real risk capacity. Tune per
    deployment.

    Attributes:
        max_positions: hard cap on the count of concurrent open positions.
        min_edge_after_cost_bps: candidates with edge below this are vetoed.
        max_notional_per_position_usd: per-position notional ceiling.
        max_total_exposure_usd: ceiling on the sum of mark-priced open notional.
    """

    max_positions: int = 3
    min_edge_after_cost_bps: float = 0.0
    max_notional_per_position_usd: float = 1_000_000.0
    max_total_exposure_usd: float = 5_000_000.0


@dataclass(frozen=True, slots=True)
class RiskVerdict:
    """outcome of running an `AllocationCandidate` through `RiskGate`.

    `reason` is a short, human-readable string for the trace log: ``"ok"``
    on approval, or a description of the rule that fired on veto.
    `adjusted_size` is reserved for future proportional sizing; at MVP it
    is always ``None`` (callers should use ``candidate.notional`` unchanged).

    Attributes:
        approved: True iff every risk rule passed.
        reason: rationale string for the trace log.
        adjusted_size: optional size override; ``None`` at MVP.
    """

    approved: bool
    reason: str
    adjusted_size: float | None = None


@dataclass(frozen=True, slots=True)
class AgentState:
    """snapshot the agent reads at the start of each tick.

    `portfolio` is intentionally a live reference, not a copy — in a
    single-threaded loop the agent reads it during `decide()`, then the
    engine continues mutating it on open/close. The frozen-dataclass
    wrapper just guarantees the agent gets *one* coherent state object,
    not that the dict inside it is immutable.

    Attributes:
        timestamp: tick time.
        board: per-asset (forecast, cost) for everything scanned this tick.
        candidates: ranked top-K from `allocate()`, best first.
        portfolio: live portfolio (read-only by convention).
        recent_fills: last few fills, oldest first; context for the agent.
    """

    timestamp: datetime
    board: dict[str, tuple["Forecast", "CostAssessment"]]
    candidates: tuple["AllocationCandidate", ...]
    portfolio: "PortfolioState"
    recent_fills: tuple["Fill", ...]


@dataclass(frozen=True, slots=True)
class TradeRationale:
    """structured per-decision reasoning record; fixed shape across ticks.

    The schema is rigid on purpose: trace logs stay comparable across
    decisions, Arc receipts (L9) commit to a stable hash shape, and the
    LLM is prevented from drifting into stream-of-consciousness prose.
    Every field is required so omissions are caught at parse time.

    Attributes:
        decision_id: monotonic identifier within the process run, e.g.
            ``"d-0007"``. Carried through veto / skip / execute so the
            trace can be joined later.
        indicators: feature values the agent cited; ``name → value`` (e.g.
            ``{"p_up": 0.72, "rsi14": 65.4, "funding_rate": 0.0001}``).
            Only inputs, not derived numbers.
        numbers: derived metrics the agent cited; ``name → value`` (e.g.
            ``{"edge_bps": -15.3, "round_trip_cost_bps": 20.5}``). These
            must come from tool calls — the LLM is forbidden from inventing.
        reasoning: one short paragraph (1–2 sentences) — the agent's
            actual conclusion. No Tolstoy.
        pros: 1–3 short bullets supporting the action.
        cons: 0–3 short bullets acknowledging counter-points.
    """

    decision_id: str
    indicators: dict[str, float]
    numbers: dict[str, float]
    reasoning: str
    pros: tuple[str, ...]
    cons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Decision:
    """one per-asset action the agent proposes for this tick.

    The risk gate may veto an `enter` or `flip` — in that case the
    decision is still recorded for the trace log with ``vetoed_reason``
    set; no fill is produced. ``side`` is required for `enter`/`flip`
    and ``None`` for `hold`/`cut`/`skip`.

    Attributes:
        asset: market symbol.
        action: one of ``"enter" | "hold" | "cut" | "flip" | "skip"``.
        side: ``"long"`` or ``"short"`` for enter/flip; ``None`` otherwise.
        rationale: structured per-decision rationale (`TradeRationale`).
        vetoed_reason: filled by the executor wrapper if the risk gate
            vetoed an `enter`/`flip`; otherwise ``None``.
    """

    asset: str
    action: str
    side: str | None
    rationale: TradeRationale
    vetoed_reason: str | None = None


@dataclass(frozen=True, slots=True)
class Decisions:
    """ordered per-asset decisions plus a top-level summary for the trace.

    Attributes:
        decisions: tuple of `Decision`, one per asset the agent considered.
        summary: one-line agent-authored summary of the tick's reasoning.
    """

    decisions: tuple[Decision, ...]
    summary: str


class Tier(StrEnum):
    """universe taxonomy bucket per `coin-tiers.md` §0.

    Drives sizing multipliers, leverage caps, ATR-based stop distances, and
    the funding-rate decision matrix. Coins outside the taxonomy default to
    `T4` (skip-by-default per §0).
    """

    T1 = "T1"
    T2 = "T2"
    T3 = "T3"
    T4 = "T4"


@dataclass(frozen=True, slots=True)
class RegimeTag:
    """three-axis regime label per `regime-classification.md` §0.

    The funding axis carries five values; `EXTREME` is split by sign
    (`EXTREME_POS`, `EXTREME_NEG`) so the playbook lookup at §4 — which
    uses `EXTREME(+)` / `EXTREME(-)` notation — has a deterministic key.

    Attributes:
        trend: `"UP" | "DOWN" | "RANGE"` from EMA-20/50 + ADX-14 (§1).
        vol: `"COMPRESSED" | "NORMAL" | "EXPANDED" | "CRISIS"` from
            ATR(10)/ATR(50) ratio + flash-crash override (§2).
        funding: `"NEUTRAL" | "BULL_BIAS" | "BEAR_BIAS" | "EXTREME_POS" |
            "EXTREME_NEG"` per tier-aware bins (§3).
    """

    trend: str
    vol: str
    funding: str


@dataclass(frozen=True, slots=True)
class SizingConfig:
    """sizing-pipeline parameters per `risk-sizing-methods.md` §5.

    Defaults follow the docs literally. Tuning machinery is private alpha;
    public ships only this baseline. See `tmp/spec-mapping.md` §1.

    Attributes:
        base_risk_fraction: equity fraction per trade (§1 floor 0.005;
            never exceed 0.01 per §7).
        vol_target_annualized: annualised target for vol-targeting (§3).
        vol_scale_cap: ceiling on `vol_target / realized_vol` so
            COMPRESSED-vol coins don't get runaway leverage (§3, §7 #4).
        kelly_fraction: quarter-Kelly default (§4).
        min_kelly_history: in-regime trades before Kelly engages (§4).
        atr_period_stops: ATR period on 1h candles for stop distance
            (`risk-stops-and-exits.md` §1.1).
        atr_period_regime_fast: short ATR for vol ratio (§2).
        atr_period_regime_slow: long ATR for vol ratio (§2).
        funding_drag_max_pct: ceiling on
            `leverage × |funding| × intervals` (`risk-leverage-and-margin.md`
            §5).
        funding_cost_to_pnl_max: per `funding-as-signal.md` §6.
    """

    base_risk_fraction: float = 0.005
    vol_target_annualized: float = 0.30
    vol_scale_cap: float = 2.5
    kelly_fraction: float = 0.25
    min_kelly_history: int = 100
    atr_period_stops: int = 14
    atr_period_regime_fast: int = 10
    atr_period_regime_slow: int = 50
    funding_drag_max_pct: float = 0.20
    funding_cost_to_pnl_max: float = 0.30


@dataclass(frozen=True, slots=True)
class LeverageCaps:
    """three leverage caps per `risk-leverage-and-margin.md` §6.

    Chosen leverage is `min(venue_cap, operational_cap, liq_safety_cap)`,
    then potentially reduced by the funding-drag check (§5).

    Attributes:
        venue_cap: HL's per-asset max-leverage at the candidate notional.
        operational_cap: self-imposed cap by tier per §0 (T1=5×, T2/T3=3×,
            T4=1.5×).
        liq_safety_cap: `1 / (4 × stop_distance_pct + mm_rate)` so the liq
            level sits ≥ 4× stop distance from entry (§3).
    """

    venue_cap: float
    operational_cap: float
    liq_safety_cap: float


@dataclass(frozen=True, slots=True)
class StopTakePlan:
    """stop / take-profit / hardening plan per `risk-stops-and-exits.md`.

    Attributes:
        stop_distance: in price units; `atr_multiplier × ATR(14, 1h)`
            (§1.1 tier × vol-regime table).
        take_distance: `r_multiple × stop_distance` (§3.1).
        r_multiple: 2.5 trend / 3.0 tail-wind / 1.5 mean-rev / 2.0
            mean-rev vol-coil (§3.1).
        scaled_exit: `True` = 50% at +1R, 50% at +2.5R with breakeven
            shift after the first take (§3.2). Default for trend regimes.
        trail_after_first_take: trail remaining 50% at `k × ATR(14, 1h)`
            after the +1R take fills (§3.3).
        stop_hardening: `"limit_stop"` | `"market_stop"` |
            `"soft_stop_scaled"` (§2 table).
    """

    stop_distance: float
    take_distance: float
    r_multiple: float
    scaled_exit: bool
    trail_after_first_take: bool
    stop_hardening: str


@dataclass(frozen=True, slots=True)
class SizedCandidate:
    """post-sizing trade plan; the executor's input.

    Replaces `AllocationCandidate`'s naive fixed `notional` with a
    docs-derived qty, leverage, and full exit plan. Carries the audit
    trail so L9 receipts can hash a stable schema.

    Attributes:
        candidate: source `AllocationCandidate` (carries forecast + cost).
        tier: from `classify_tier`.
        regime: from `compute_regime_tag` (post-BTC-override).
        qty: base-unit quantity to fill.
        notional: `qty × mark_price` in USD.
        leverage: chosen leverage, capped per `LeverageCaps` and adjusted
            for funding drag (§5).
        margin_required: `notional / leverage`.
        stop_price: signed by side; `entry - stop_distance` for long,
            `entry + stop_distance` for short.
        take_price: first-take level (50% exit if `scaled_exit`).
        stop_take_plan: full exit plan including hardening and trail.
        leverage_caps: the three caps that informed `leverage`.
        sizing_audit: per-step breakdown of the §5 pipeline (every named
            field from §6 of `risk-sizing-methods.md` for receipts).
    """

    candidate: AllocationCandidate
    tier: Tier
    regime: RegimeTag
    qty: float
    notional: float
    leverage: float
    margin_required: float
    stop_price: float
    take_price: float
    stop_take_plan: StopTakePlan
    leverage_caps: LeverageCaps
    sizing_audit: dict[str, float]
