"""Anchor audit records to Arc Testnet via ERC-8183 commerce protocol.

Uses Circle's Developer-Controlled Wallets **Contract Execution API**
to call ERC-8183's `createJob` (and optionally `submit`) on our behalf.
We never hold a private key — Circle's MPC backend signs the EVM
transaction after we authenticate with a fresh entity-secret
ciphertext.

For each audit record we want to anchor:

    anchor_record(audit_record) → AnchorResult(
        circle_tx_id, arc_tx_hash, arc_job_id, description, hash_hex
    )

The MVP flow is just `createJob`: the audit anchor hash is embedded in
the job's `description` field as `"agora:<audit_id>:0x<hash>"`. That
gives us:

* a permanent on-chain artifact (tx_hash on arcscan)
* the anchor hash readable in the `JobCreated` event's transaction
  input data
* an ERC-8183 jobId that ties the decision to Arc's commerce framework
  without paying USDC for fund/submit/complete

`submit_record_hash(audit_record, job_id)` is provided for the v2
upgrade path — once we want the hash in `JobSubmitted`'s indexed
`bytes32 deliverable` topic (Arcadia indexes that field), we call
`submit` with the same `audit_id`.

Circle docs:
    https://developers.circle.com/w3s/contracts-create-contract-execution-tx
"""

from __future__ import annotations

import base64
import os
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

from agent.arc_constants import (
    ABI_CREATE_JOB,
    ABI_SUBMIT,
    CHAIN_NAME,
    ERC8183_COMMERCE_PROXY,
    arcscan_tx,
)
from agent.audit import AuditRecord, anchor_hash

_CIRCLE_API_BASE = "https://api.circle.com"

# Per Circle's tx-lifecycle docs: state transitions go INITIATED →
# WAITING → QUEUED → CLEARED → SENT → CONFIRMED → COMPLETE.
# Anything after SENT carries a populated `txHash`.
_TERMINAL_OK = {"CONFIRMED", "COMPLETE"}
_TERMINAL_BAD = {"FAILED", "DENIED", "CANCELLED"}
_HASH_AVAILABLE_FROM = {"SENT", "CONFIRMED", "COMPLETE"}


@dataclass(frozen=True, slots=True)
class AnchorResult:
    """The on-chain artifact produced for one audit record.

    Attributes:
        circle_tx_id: Circle's UUID for the transaction (used to poll
            and reconcile via their REST API).
        arc_tx_hash: ``0x``-prefixed 32-byte Arc transaction hash;
            populated once Circle's backend broadcasts.
        arc_job_id: ERC-8183 jobId once the tx confirms and the event
            is parseable. ``None`` for MVP `createJob`-only flow
            without event polling.
        description: the exact string written into the
            `JobCreated.description` field — operators can grep on it.
        anchor_hash_hex: ``0x``-prefixed keccak256 of the audit record;
            also embedded in `description`.
        state: Circle's terminal lifecycle state at the time we
            stopped polling. `COMPLETE` is the happy path.
        arcscan_url: convenience URL for the demo + Loom recording.
    """

    circle_tx_id: str
    arc_tx_hash: str | None
    arc_job_id: int | None
    description: str
    anchor_hash_hex: str
    state: str
    arcscan_url: str | None


class CircleAnchor:
    """Submit ERC-8183 `createJob` transactions via Circle's API.

    Reads credentials from env vars `CIRCLE_API_KEY`,
    `CIRCLE_ENTITY_SECRET`, `CIRCLE_WALLET_ID`, `CIRCLE_WALLET_ADDRESS`
    by default — override via constructor args for testing.

    Not thread-safe: re-fetches the Circle RSA public key on
    construction and reuses it for the lifetime of the instance. Each
    `anchor_record` call generates a fresh OAEP ciphertext (Circle
    rejects re-used ciphertexts).
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        entity_secret: str | None = None,
        wallet_id: str | None = None,
        wallet_address: str | None = None,
        timeout_s: float = 30.0,
        poll_max_s: float = 60.0,
        poll_interval_s: float = 2.0,
    ) -> None:
        self._api_key = api_key or os.environ.get("CIRCLE_API_KEY", "")
        self._entity_secret = (
            entity_secret or os.environ.get("CIRCLE_ENTITY_SECRET", "")
        )
        self._wallet_id = wallet_id or os.environ.get("CIRCLE_WALLET_ID", "")
        self._wallet_address = (
            wallet_address or os.environ.get("CIRCLE_WALLET_ADDRESS", "")
        )
        for label, value in [
            ("CIRCLE_API_KEY", self._api_key),
            ("CIRCLE_ENTITY_SECRET", self._entity_secret),
            ("CIRCLE_WALLET_ID", self._wallet_id),
            ("CIRCLE_WALLET_ADDRESS", self._wallet_address),
        ]:
            if not value:
                raise RuntimeError(
                    f"{label} missing — run scripts/circle_setup.py first"
                )

        self._timeout_s = timeout_s
        self._poll_max_s = poll_max_s
        self._poll_interval_s = poll_interval_s
        self._client = httpx.Client(timeout=timeout_s)
        self._public_key_pem = self._fetch_public_key()

    # -------------------------------------------------------- public API

    def anchor_record(
        self,
        record: AuditRecord,
        *,
        expiry_hours: int = 1,
    ) -> AnchorResult:
        """Anchor one `AuditRecord` to Arc via `createJob`.

        The audit record's `anchor_hash` (keccak256 over canonical
        JSON + nonce) is embedded into the on-chain job's
        `description` field as ``"agora:<audit_id>:0x<hash>"``. The
        provider and evaluator are both our wallet (self-loop) since
        we're not paying anyone for the audit itself.

        Args:
            record: the frozen `AuditRecord` to anchor.
            expiry_hours: hours from now until the job auto-expires
                on-chain. ERC-8183 enforces `expiredAt > now`; we set
                a 1-hour default so the job is short-lived and doesn't
                clutter Arcadia's active-job index.

        Returns:
            `AnchorResult` with `arc_tx_hash` populated once Circle
            broadcasts. Caller can persist this back onto the audit
            record for L10 dashboard linking.

        Raises:
            httpx.HTTPStatusError: on Circle API 4xx/5xx.
            RuntimeError: on terminal-bad lifecycle states
                (`FAILED`, `DENIED`, `CANCELLED`).
        """
        hash_bytes = anchor_hash(record)
        hash_hex = "0x" + hash_bytes.hex()
        description = f"agora:{record.audit_id}:{hash_hex}"
        expired_at = int(
            (
                datetime.now(timezone.utc) + timedelta(hours=expiry_hours)
            ).timestamp()
        )
        zero_addr = "0x" + "0" * 40

        circle_tx_id = self._post_contract_execution(
            abi_signature=ABI_CREATE_JOB,
            abi_parameters=[
                self._wallet_address,
                self._wallet_address,
                str(expired_at),
                description,
                zero_addr,
            ],
        )
        state, arc_tx_hash = self._poll_until_terminal(circle_tx_id)

        return AnchorResult(
            circle_tx_id=circle_tx_id,
            arc_tx_hash=arc_tx_hash,
            arc_job_id=None,  # parsing JobCreated event is a v2 follow-up
            description=description,
            anchor_hash_hex=hash_hex,
            state=state,
            arcscan_url=arcscan_tx(arc_tx_hash) if arc_tx_hash else None,
        )

    def submit_record_hash(
        self, record: AuditRecord, job_id: int,
    ) -> AnchorResult:
        """Submit the anchor hash as ERC-8183 deliverable on an existing job.

        Provided for the v2 upgrade path where we want the hash in
        `JobSubmitted`'s indexed `bytes32` topic (which Arcadia
        natively indexes and Botozen's job #19091 used). MVP path is
        `anchor_record` alone — call this once we wire fund + submit.

        Args:
            record: the `AuditRecord` whose hash to submit.
            job_id: existing ERC-8183 jobId (from a prior `createJob`
                that is now in `Funded` state).

        Returns:
            `AnchorResult` for the submission tx.
        """
        hash_bytes = anchor_hash(record)
        hash_hex = "0x" + hash_bytes.hex()
        empty_bytes = "0x"

        circle_tx_id = self._post_contract_execution(
            abi_signature=ABI_SUBMIT,
            abi_parameters=[str(job_id), hash_hex, empty_bytes],
        )
        state, arc_tx_hash = self._poll_until_terminal(circle_tx_id)

        return AnchorResult(
            circle_tx_id=circle_tx_id,
            arc_tx_hash=arc_tx_hash,
            arc_job_id=job_id,
            description=f"submit:job={job_id}:{hash_hex}",
            anchor_hash_hex=hash_hex,
            state=state,
            arcscan_url=arcscan_tx(arc_tx_hash) if arc_tx_hash else None,
        )

    def close(self) -> None:
        self._client.close()

    # -------------------------------------------------------- internals

    def _fetch_public_key(self) -> str:
        r = self._client.get(
            f"{_CIRCLE_API_BASE}/v1/w3s/config/entity/publicKey",
            headers={"Authorization": f"Bearer {self._api_key}"},
        )
        r.raise_for_status()
        return r.json()["data"]["publicKey"]

    def _fresh_ciphertext(self) -> str:
        """Encrypt our entity_secret freshly for one API call."""
        pk = serialization.load_pem_public_key(self._public_key_pem.encode())
        secret_bytes = bytes.fromhex(self._entity_secret)
        ct = pk.encrypt(
            secret_bytes,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None,
            ),
        )
        return base64.b64encode(ct).decode()

    def _post_contract_execution(
        self, *, abi_signature: str, abi_parameters: list[str],
    ) -> str:
        """Submit a contract-execution tx. Returns Circle's tx id.

        Note on fee format: Circle accepts `feeLevel` as a flat
        top-level field for Developer-Controlled wallet calls (the
        nested `fee.config.feeLevel` form is SCP-specific and yields a
        4xx here). Verified by trial against `400 invalid_value /
        gasPrice / gasLimit` responses on the nested form.
        """
        body = {
            "idempotencyKey": str(uuid.uuid4()),
            "walletId": self._wallet_id,
            "contractAddress": ERC8183_COMMERCE_PROXY,
            "abiFunctionSignature": abi_signature,
            "abiParameters": abi_parameters,
            "feeLevel": "MEDIUM",
            "entitySecretCiphertext": self._fresh_ciphertext(),
        }
        r = self._client.post(
            f"{_CIRCLE_API_BASE}/v1/w3s/developer/transactions/contractExecution",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json=body,
        )
        if r.status_code >= 400:
            raise RuntimeError(
                f"contractExecution failed [{r.status_code}]: {r.text}"
            )
        return r.json()["data"]["id"]

    def _poll_until_terminal(self, tx_id: str) -> tuple[str, str | None]:
        """Poll Circle's tx-status endpoint until terminal or timeout.

        Returns ``(state, tx_hash_or_None)``. `tx_hash` is populated
        once Circle broadcasts the tx (state >= SENT); we stop polling
        early at that point if we don't care about CONFIRMED.
        """
        deadline = time.monotonic() + self._poll_max_s
        last_state = "UNKNOWN"
        last_hash: str | None = None
        while time.monotonic() < deadline:
            r = self._client.get(
                f"{_CIRCLE_API_BASE}/v1/w3s/transactions/{tx_id}",
                headers={"Authorization": f"Bearer {self._api_key}"},
            )
            r.raise_for_status()
            data = r.json()["data"]["transaction"]
            last_state = data.get("state", "UNKNOWN")
            last_hash = data.get("txHash") or last_hash
            if last_state in _TERMINAL_OK:
                return last_state, last_hash
            if last_state in _TERMINAL_BAD:
                raise RuntimeError(
                    f"tx {tx_id} terminal-bad: state={last_state}, "
                    f"data={data}"
                )
            if last_state == "SENT" and last_hash:
                # MVP: SENT + tx_hash is sufficient for arcscan demo.
                # We can re-poll later to confirm finality.
                return last_state, last_hash
            time.sleep(self._poll_interval_s)
        return last_state, last_hash


# ------------------------------------------------------------ helpers


def _load_env(env_path: Path) -> dict[str, str]:
    """Parse `KEY=VALUE` lines into a dict — same as scripts/circle_setup."""
    env: dict[str, str] = {}
    if not env_path.exists():
        return env
    for line in env_path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        env[key.strip()] = value.strip().strip('"').strip("'")
    return env


def from_env_file(env_path: Path | str) -> CircleAnchor:
    """Construct a `CircleAnchor` reading creds from a specific .env file.

    Useful when running scripts that aren't launched from the project
    root and so don't have the env vars already exported.
    """
    env = _load_env(Path(env_path))
    return CircleAnchor(
        api_key=env.get("CIRCLE_API_KEY"),
        entity_secret=env.get("CIRCLE_ENTITY_SECRET"),
        wallet_id=env.get("CIRCLE_WALLET_ID"),
        wallet_address=env.get("CIRCLE_WALLET_ADDRESS"),
    )


__all__ = [
    "AnchorResult",
    "CircleAnchor",
    "from_env_file",
    "Verdict",
]


# Re-export `Verdict` so callers don't need to import from two modules.
from agent.audit import Verdict  # noqa: E402
