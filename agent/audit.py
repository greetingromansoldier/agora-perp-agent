"""AuditRecord schema + write-ahead sqlite store.

Implements `architecture/audit-record-and-erc8183-anchor.md` §2 (the
record) and §3 (the write-ahead invariant). Each decision the agent
makes — whether EXECUTE, DEFER, CHALLENGE, or REJECT — produces an
`AuditRecord` instance, which is persisted to sqlite *before* any
external effect (venue order submission, on-chain anchor).

The schema is intentionally narrow for MVP: every field below maps to
something we already have in `core/contracts.py` or compute on the way
to a fill. The full §2 schema includes more environment / identity
fields; we add them as we wire L9 deeper.

Anchor hash is `keccak256(canonical_json(record) || nonce)` per §4.
Nonce is 32 random bytes generated at record-construction time and
included in the canonical JSON so the hash is preimage-resistant
against a public schema attacker who knows everything else.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from secrets import token_bytes
from typing import Any

from eth_utils import keccak

_SCHEMA_VERSION = "v1"


class Verdict(StrEnum):
    """The four decision verdicts per §2.

    ``EXECUTE`` is the only verdict that produces a venue order. The
    other three are first-class entries — the audit log records *why*
    nothing happened with the same weight as *what* did happen.
    """

    EXECUTE = "EXECUTE"
    DEFER = "DEFER"
    CHALLENGE = "CHALLENGE"
    REJECT = "REJECT"


class Outcome(StrEnum):
    """Terminal-state vocabulary for the async fill leg per §2.

    ``PAID`` is a separate state post `PaymentReleased` on Arc — kept
    distinct from `FILLED` (which is the venue-side terminal) per the
    Arcadia state-machine reading (`audit-record...md` §7).
    """

    PENDING = "PENDING"
    FILLED = "FILLED"
    PARTIAL = "PARTIAL"
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"
    ERRORED = "ERRORED"
    PAID = "PAID"


@dataclass(frozen=True, slots=True)
class AuditRecord:
    """one decision frozen into a hashable, anchor-ready record.

    Created at decision time, sealed before write-ahead, never mutated.
    Async outcome arrives via a *new* record that references this one
    by `audit_id` (the §6 v1 retry queue pattern).

    Attributes:
        schema_version: bump for breaking schema changes; recorded on
            every record so off-chain readers know how to parse.
        audit_id: UUID v4 string; primary key.
        decision_id: short human-readable id from the agent (`d-0042`).
        agent_address: signer wallet address (`0x...`); ties this
            record to an ERC-8004 identity in v2.
        chain_id: Arc Testnet = 5042002. Audit records survive chain
            migrations.
        venue: HL paper-only at MVP (``"hyperliquid-paper"``).
        decided_at_iso: ISO-8601 UTC timestamp at decision emission.
        input_hashes: stable digest of the inputs the agent was given
            (market snapshot, forecast, cost assessment). Each value
            is hex-encoded keccak256 of the canonical-serialized
            source object.
        asset: HL coin shortcode (``"BTC"``).
        side: ``"long" | "short" | None``. None for non-EXECUTE
            verdicts (or for HOLD / SKIP actions when those become
            first-class).
        verdict: from `Verdict`.
        sized: post-sizing breakdown — qty, notional, leverage, stop,
            take, plus tier and three regime axes. `None` for non-
            EXECUTE verdicts.
        reasoning: LLM's free-text rationale (truncated to 2 KB).
        outcome: from `Outcome`; starts `PENDING`, updated when the
            fill / anchor / payment legs complete.
        fill: post-fill numbers — qty, avg_px, realized_pnl_usd. None
            until outcome leaves PENDING.
        venue_order_id: paper-trader internal fill id.
        arc_tx_hash: tx hash of the ERC-8183 anchor on Arc.
        arc_job_id: ERC-8183 jobId returned by `createJob`.
        nonce_hex: 32-byte hex; the anti-preimage salt for the anchor
            hash. Generated fresh per record.
    """

    schema_version: str
    audit_id: str
    decision_id: str
    agent_address: str
    chain_id: int
    venue: str
    decided_at_iso: str
    input_hashes: dict[str, str]
    asset: str
    side: str | None
    verdict: str
    sized: dict[str, Any] | None
    reasoning: str
    outcome: str
    fill: dict[str, float] | None
    venue_order_id: str | None
    arc_tx_hash: str | None
    arc_job_id: int | None
    nonce_hex: str


def new_record(
    *,
    decision_id: str,
    agent_address: str,
    chain_id: int,
    venue: str,
    asset: str,
    side: str | None,
    verdict: Verdict,
    input_hashes: dict[str, str],
    sized: dict[str, Any] | None,
    reasoning: str,
    decided_at: datetime | None = None,
) -> AuditRecord:
    """Construct a fresh `AuditRecord` at decision time.

    Pre-fills `audit_id`, `nonce_hex`, `decided_at_iso`, and
    `outcome=PENDING`. Caller updates outcome by writing a *new*
    record (records are frozen).
    """
    when = decided_at or datetime.now(timezone.utc)
    return AuditRecord(
        schema_version=_SCHEMA_VERSION,
        audit_id=str(uuid.uuid4()),
        decision_id=decision_id,
        agent_address=agent_address,
        chain_id=chain_id,
        venue=venue,
        decided_at_iso=when.isoformat(),
        input_hashes=input_hashes,
        asset=asset,
        side=side,
        verdict=verdict.value,
        sized=sized,
        reasoning=reasoning[:2048],
        outcome=Outcome.PENDING.value,
        fill=None,
        venue_order_id=None,
        arc_tx_hash=None,
        arc_job_id=None,
        nonce_hex=token_bytes(32).hex(),
    )


def canonical_json(record: AuditRecord) -> bytes:
    """Serialize a record to canonical JSON (sorted keys, no whitespace).

    Determinism is the whole point — two implementations that produce
    different bytes break the anchor hash. Sort keys, drop whitespace,
    UTF-8 encode. `None` becomes `null`, ints stay native ints (JSON
    has no integer / float distinction but `dataclasses.asdict`
    preserves Python types into the dict).
    """
    return json.dumps(
        asdict(record),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def anchor_hash(record: AuditRecord) -> bytes:
    """Return the 32-byte keccak256 anchor hash per §4.

    ``keccak256(canonical_json(record) || nonce_bytes)``. The nonce is
    *also* in the canonical JSON (so the hash is deterministic given
    the record), but it makes the JSON itself unpredictable to a
    public-schema attacker — preventing a pre-compute-all-decisions
    style brute force on the published hash.
    """
    payload = canonical_json(record)
    nonce_bytes = bytes.fromhex(record.nonce_hex)
    return keccak(payload + nonce_bytes)


# ------------------------------------------------------------ storage


class AuditStore:
    """Append-only sqlite store for audit records.

    Each row is a frozen snapshot of an `AuditRecord` at one point in
    its lifecycle (decided / submitted / filled / paid). When the
    state changes, the application calls `append(record)` with a new
    record sharing the same `audit_id`. `latest(audit_id)` returns the
    most recent state.

    The schema deliberately stores the full record as a JSON blob plus
    a few extracted columns for indexing. This keeps the audit shape
    flexible without schema migrations every time we add a field.
    """

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS audit_records (
        rowid INTEGER PRIMARY KEY AUTOINCREMENT,
        audit_id TEXT NOT NULL,
        decision_id TEXT NOT NULL,
        agent_address TEXT NOT NULL,
        asset TEXT NOT NULL,
        verdict TEXT NOT NULL,
        outcome TEXT NOT NULL,
        decided_at_iso TEXT NOT NULL,
        anchor_hash_hex TEXT NOT NULL,
        arc_tx_hash TEXT,
        arc_job_id INTEGER,
        record_json TEXT NOT NULL
    );

    CREATE INDEX IF NOT EXISTS ix_audit_records_audit_id
        ON audit_records (audit_id);
    CREATE INDEX IF NOT EXISTS ix_audit_records_outcome
        ON audit_records (outcome);
    CREATE INDEX IF NOT EXISTS ix_audit_records_asset
        ON audit_records (asset);
    """

    def __init__(self, db_path: Path | str = "audit.sqlite") -> None:
        self._db_path = Path(db_path)
        self._conn = sqlite3.connect(
            self._db_path, isolation_level=None, check_same_thread=False
        )
        # WAL gives us crash-safe append-only writes with concurrent reads.
        self._conn.execute("PRAGMA journal_mode = WAL;")
        for stmt in self._SCHEMA.strip().split(";"):
            if stmt.strip():
                self._conn.execute(stmt)

    def append(self, record: AuditRecord) -> int:
        """Insert one record. Returns the rowid for ordering.

        Idempotency: `audit_id` is *not* unique — successive states of
        the same decision share an `audit_id` and differ only in
        `outcome`, `fill`, `arc_tx_hash`, etc. Use `latest(audit_id)`
        to get the most-recent state.
        """
        hash_hex = anchor_hash(record).hex()
        blob = canonical_json(record).decode("utf-8")
        cur = self._conn.execute(
            """
            INSERT INTO audit_records (
                audit_id, decision_id, agent_address, asset, verdict,
                outcome, decided_at_iso, anchor_hash_hex,
                arc_tx_hash, arc_job_id, record_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.audit_id,
                record.decision_id,
                record.agent_address,
                record.asset,
                record.verdict,
                record.outcome,
                record.decided_at_iso,
                hash_hex,
                record.arc_tx_hash,
                record.arc_job_id,
                blob,
            ),
        )
        return int(cur.lastrowid or 0)

    def latest(self, audit_id: str) -> AuditRecord | None:
        """Return the most-recent state for `audit_id`, or ``None``."""
        row = self._conn.execute(
            """
            SELECT record_json FROM audit_records
            WHERE audit_id = ?
            ORDER BY rowid DESC
            LIMIT 1
            """,
            (audit_id,),
        ).fetchone()
        if row is None:
            return None
        return _record_from_json(row[0])

    def list_pending(self, max_age_seconds: float | None = None) -> list[AuditRecord]:
        """Return EXECUTE-verdict records still in `PENDING` outcome.

        Used by the watchdog (`agent/watchdog.py`) to alert on records
        that should have finalised by now. Optional `max_age_seconds`
        filter — pass to only get records older than N seconds.
        """
        rows = self._conn.execute(
            """
            SELECT record_json FROM audit_records
            WHERE outcome = 'PENDING' AND verdict = 'EXECUTE'
            ORDER BY rowid ASC
            """
        ).fetchall()
        records = [_record_from_json(r[0]) for r in rows]
        if max_age_seconds is None:
            return records
        now = datetime.now(timezone.utc)
        cutoff = max_age_seconds
        result = []
        for r in records:
            decided = datetime.fromisoformat(r.decided_at_iso)
            age = (now - decided).total_seconds()
            if age >= cutoff:
                result.append(r)
        return result

    def count(self) -> int:
        """Return total record count (across all audit_ids and states)."""
        row = self._conn.execute(
            "SELECT COUNT(*) FROM audit_records"
        ).fetchone()
        return int(row[0]) if row else 0

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> AuditStore:
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


def _record_from_json(blob: str) -> AuditRecord:
    """Inverse of `canonical_json` — JSON → `AuditRecord`."""
    data = json.loads(blob)
    return AuditRecord(**data)


# -------------------------------------------------------- input hashing


def hash_input(payload: Any) -> str:
    """Stable hex digest of an input payload for `input_hashes` map.

    Accepts a dict / dataclass-asdict / primitive. Serializes via
    `json.dumps(..., sort_keys=True, default=str)` so datetimes and
    Decimal etc. survive as strings, keccak256s the result, returns
    hex string.

    Caller's responsibility to pre-shape complex objects (e.g.
    `core.contracts.MarketData`) into the dict they want hashed —
    typically `asdict(market_data)`.
    """
    blob = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=str,
    ).encode("utf-8")
    return keccak(blob).hex()
