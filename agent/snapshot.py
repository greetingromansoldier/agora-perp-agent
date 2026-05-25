"""Serialise live trading state to dashboard-readable JSON.

The dashboard at `agora-perp-agent/dashboard/` polls a static JSON file
that the running paper-trader writes. This module is the writer:
takes the agent's current `PortfolioState` + decision history + fill
log + equity curve, produces a flat JSON document the dashboard JS can
render without further transformation.

Writes are atomic via a `.tmp` rename so the dashboard never reads a
half-written file.

Schema is documented in this module's `_SCHEMA_VERSION` comment and on
the dashboard side in `dashboard/app.js`. Bump the version when adding
breaking changes; keep adding fields is back-compat (dashboard renders
what it knows, ignores rest).
"""

from __future__ import annotations

import json
import os
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Bump on breaking schema changes; back-compat additions stay at v1.
_SCHEMA_VERSION = "v1"

# Equity curve trims to the last N points to keep snapshot.json small —
# 720 = 12 hours at 1-min ticks, or 1 hour at 5s ticks.
_EQUITY_CURVE_MAX = 720
# Trade history truncates to last N closed trades.
_TRADE_HISTORY_MAX = 50
# Recent-decisions truncates to last N decisions.
_RECENT_DECISIONS_MAX = 30


@dataclass(slots=True)
class SnapshotState:
    """Cross-tick accumulator the trading loop maintains in memory.

    Built up from board.py's existing state: appended on each fill and
    each LLM completion, periodically dumped to disk via `write`. Kept
    deliberately decoupled from `PortfolioState` so the dashboard sees
    a richer history than `PortfolioState` retains.

    Attributes:
        starting_balance_usd: locked at constructor; equity curve normalises
            to this.
        agent_wallet: signer address for the on-chain anchor cross-links.
        equity_curve: deque of `(iso_ts, equity_usd)` tuples, capped at
            `_EQUITY_CURVE_MAX`.
        trade_history: deque of closed-trade dicts, capped at
            `_TRADE_HISTORY_MAX`.
        recent_decisions: deque of decision dicts (each entry has the
            structured rationale + sizing + on-chain anchor info), capped
            at `_RECENT_DECISIONS_MAX`.
        audit_links: per-asset map of the most recent open's anchor info
            so `open_positions` rows can carry an arcscan URL.
        ticks_processed: monotonic counter for diagnostics.
    """

    starting_balance_usd: float
    agent_wallet: str
    equity_curve: deque = field(default_factory=lambda: deque(maxlen=_EQUITY_CURVE_MAX))
    trade_history: deque = field(default_factory=lambda: deque(maxlen=_TRADE_HISTORY_MAX))
    recent_decisions: deque = field(default_factory=lambda: deque(maxlen=_RECENT_DECISIONS_MAX))
    audit_links: dict[str, dict[str, Any]] = field(default_factory=dict)
    ticks_processed: int = 0
    agent_status: str = "idle"

    def record_equity(self, equity_usd: float) -> None:
        """Append one equity-curve sample with the current UTC timestamp."""
        self.equity_curve.append(
            {
                "timestamp_iso": datetime.now(timezone.utc).isoformat(),
                "equity_usd": float(equity_usd),
            }
        )

    def record_open(
        self,
        *,
        asset: str,
        side: str,
        qty: float,
        leverage: float,
        entry_price: float,
        stop_price: float | None,
        take_price: float | None,
        tier: str,
        regime: str,
        notional_usd: float,
        audit_id: str,
        decision_id: str,
    ) -> None:
        """Track per-asset audit linkage for the open positions table."""
        self.audit_links[asset] = {
            "audit_id": audit_id,
            "decision_id": decision_id,
            "tier": tier,
            "regime": regime,
            "leverage": leverage,
            "stop_price": stop_price,
            "take_price": take_price,
            "entry_price": entry_price,
            "qty_open": qty,
            "notional_open_usd": notional_usd,
            "side": side,
            "arc_tx_hash": None,
            "arcscan_url": None,
            "anchor_state": "pending",
        }

    def update_anchor(
        self,
        *,
        asset: str | None,
        audit_id: str,
        arc_tx_hash: str | None,
        arcscan_url: str | None,
        state: str,
    ) -> None:
        """Patch in arcscan info once `AnchorWorker` returns.

        Updates both `audit_links` (open-position-side render) and the
        matching entry in `recent_decisions` so the dashboard can show
        the same tx_hash in two places.
        """
        if asset is not None and asset in self.audit_links:
            link = self.audit_links[asset]
            if link.get("audit_id") == audit_id:
                link["arc_tx_hash"] = arc_tx_hash
                link["arcscan_url"] = arcscan_url
                link["anchor_state"] = state
        for entry in self.recent_decisions:
            if entry.get("audit_id") == audit_id:
                entry["arc_tx_hash"] = arc_tx_hash
                entry["arcscan_url"] = arcscan_url
                entry["anchor_state"] = state

    def record_close(
        self,
        *,
        asset: str,
        side: str,
        qty: float,
        entry_price: float,
        exit_price: float,
        realized_pnl_usd: float,
        entry_time_iso: str,
        exit_time_iso: str,
        accrued_funding_usd: float,
    ) -> None:
        """Append one closed-trade entry to trade_history; clear audit link."""
        link = self.audit_links.pop(asset, {})
        self.trade_history.appendleft(
            {
                "asset": asset,
                "side": side,
                "qty": float(qty),
                "entry_price": float(entry_price),
                "exit_price": float(exit_price),
                "realized_pnl_usd": float(realized_pnl_usd),
                "accrued_funding_usd": float(accrued_funding_usd),
                "entry_time_iso": entry_time_iso,
                "exit_time_iso": exit_time_iso,
                "audit_id": link.get("audit_id"),
                "decision_id": link.get("decision_id"),
                "leverage_at_open": link.get("leverage"),
                "tier": link.get("tier"),
                "regime": link.get("regime"),
                "arc_tx_hash": link.get("arc_tx_hash"),
                "arcscan_url": link.get("arcscan_url"),
            }
        )

    def record_decision(self, entry: dict[str, Any]) -> None:
        """Append one decision (from LLM emit, post-`_apply`)."""
        self.recent_decisions.appendleft(entry)

    def tick(self) -> None:
        self.ticks_processed += 1


def hydrate_from_audit_store(
    state: "SnapshotState", store: Any,
) -> None:
    """Pre-fill `recent_decisions` history from a persistent sqlite store.

    Used at startup to give continuous history across board.py restarts:
    every prior EXECUTE/DEFER/REJECT decision shows up in the dashboard's
    decision feed even though the in-memory deques are fresh.

    Reads all records, sorts oldest-first, replays them onto `state`.
    Trade history reconstruction (open + close pairing) is best-effort;
    we just surface the EXECUTE entries.

    Args:
        state: empty `SnapshotState` to populate.
        store: an `agent.audit.AuditStore`.
    """
    rows = store._conn.execute(  # noqa: SLF001 — internal cross-module access
        """
        SELECT record_json FROM audit_records
        ORDER BY rowid ASC
        """
    ).fetchall()
    import json as _json  # noqa: PLC0415

    for (blob,) in rows:
        rec = _json.loads(blob)
        if rec.get("verdict") != "EXECUTE":
            continue
        sized = rec.get("sized") or {}
        tx_hash = rec.get("arc_tx_hash")
        state.record_decision(
            {
                "audit_id": rec.get("audit_id"),
                "decision_id": rec.get("decision_id"),
                "asset": rec.get("asset"),
                "side": rec.get("side"),
                "action": "enter",
                "verdict": rec.get("verdict"),
                "tier": sized.get("tier"),
                "regime": (
                    f"{sized.get('regime_trend','?')}·"
                    f"{sized.get('regime_vol','?')}·"
                    f"{sized.get('regime_funding','?')}"
                ),
                "leverage": sized.get("leverage"),
                "stop_price": sized.get("stop_price"),
                "take_price": sized.get("take_price"),
                "notional_usd": sized.get("notional_usd"),
                "qty": sized.get("qty"),
                "entry_price": None,  # not stored on the record itself
                "edge_after_cost_bps": None,
                "reasoning": rec.get("reasoning"),
                "decided_at_iso": rec.get("decided_at_iso"),
                "anchor_state": "complete" if tx_hash else "unknown",
                "arc_tx_hash": tx_hash,
                "arcscan_url": (
                    f"https://testnet.arcscan.app/tx/{tx_hash}"
                    if tx_hash else None
                ),
            }
        )


def write(state: SnapshotState, portfolio, path: Path | str) -> None:
    """Atomically write the snapshot JSON for `portfolio` + history.

    Args:
        state: cross-tick accumulator.
        portfolio: live `PortfolioState`; read for `positions`, `balance`,
            `realized_pnl`, `unrealized`, `equity`.
        path: target file. Written via `<path>.tmp` then `os.replace` so
            the dashboard never sees a partial file.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": _SCHEMA_VERSION,
        "generated_at_iso": datetime.now(timezone.utc).isoformat(),
        "agent_wallet": state.agent_wallet,
        "starting_balance_usd": state.starting_balance_usd,
        "equity_usd": float(portfolio.equity_usd()),
        "balance_usd": float(portfolio.balance_usd),
        "realized_pnl_usd": float(portfolio.realized_pnl_usd),
        "unrealized_pnl_usd": float(portfolio.unrealized_usd()),
        "agent_status": state.agent_status,
        "ticks_processed": state.ticks_processed,
        "open_positions": _serialise_positions(portfolio, state.audit_links),
        "recent_decisions": list(state.recent_decisions),
        "trade_history": list(state.trade_history),
        "equity_curve": list(state.equity_curve),
    }
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=_json_default))
    os.replace(tmp, target)


def _serialise_positions(
    portfolio, audit_links: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for asset, pos in portfolio.positions.items():
        link = audit_links.get(asset, {})
        notional = float(pos.qty * pos.last_mark)
        rows.append(
            {
                "asset": asset,
                "side": pos.side,
                "qty": float(pos.qty),
                "entry_price": float(pos.entry_price),
                "mark_price": float(pos.last_mark),
                "notional_usd": notional,
                "unrealized_pnl_usd": float(pos.unrealized_pnl_usd()),
                "accrued_funding_usd": float(pos.accrued_funding_usd),
                "stop_price": _opt_float(pos.stop_price),
                "take_price": _opt_float(pos.take_price),
                "entry_time_iso": pos.entry_time.isoformat()
                if hasattr(pos.entry_time, "isoformat") else str(pos.entry_time),
                "tier": link.get("tier"),
                "regime": link.get("regime"),
                "leverage": link.get("leverage"),
                "audit_id": link.get("audit_id"),
                "decision_id": link.get("decision_id"),
                "arc_tx_hash": link.get("arc_tx_hash"),
                "arcscan_url": link.get("arcscan_url"),
                "anchor_state": link.get("anchor_state", "n/a"),
            }
        )
    return rows


def _opt_float(value: Any) -> float | None:
    return float(value) if value is not None else None


def _json_default(value: Any) -> Any:
    """Fallback for json.dumps — datetimes → iso, sets → list, else str."""
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, set):
        return list(value)
    return str(value)
