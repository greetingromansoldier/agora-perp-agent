"""Live smoke test: anchor one synthetic audit record to Arc Testnet.

Constructs a hand-crafted `AuditRecord`, calls Circle's Contract
Execution API to create an ERC-8183 job on Arc Testnet, polls until
the tx is broadcast (state ≥ SENT), and prints the arcscan URL.

Prerequisites: `CIRCLE_API_KEY`, `CIRCLE_ENTITY_SECRET`,
`CIRCLE_WALLET_ID`, `CIRCLE_WALLET_ADDRESS` populated in `.env`
(provision via `scripts/circle_setup.py`), and the wallet funded with
at least a few microUSDC of testnet USDC for gas (faucet.circle.com).

Run from project root:

    uv run --with httpx --with cryptography --with eth_utils \\
           --with pycryptodome python scripts/smoke_anchor.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Make the project root importable so `agent.*` resolves without
# needing a `pyproject.toml` install. uv-run launches us from cwd but
# Python doesn't add it; we do.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from agent import audit  # noqa: E402
from agent.arc_constants import CHAIN_ID, arcscan_address  # noqa: E402
from agent.on_chain import from_env_file  # noqa: E402


def _load_dotenv(env_path: Path) -> None:
    """Pop .env into os.environ before constructing the anchor client."""
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, _, v = s.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    _load_dotenv(repo_root / ".env")

    record = audit.new_record(
        decision_id="smoke-0001",
        agent_address=os.environ["CIRCLE_WALLET_ADDRESS"],
        chain_id=CHAIN_ID,
        venue="hyperliquid-paper",
        asset="BTC",
        side="long",
        verdict=audit.Verdict.EXECUTE,
        input_hashes={
            "market": audit.hash_input({"asset": "BTC", "mark": 100_000.0}),
            "forecast": audit.hash_input({"p_up": 0.62}),
            "cost": audit.hash_input({"breakeven_bps": 23.0}),
        },
        sized={
            "qty": 0.0123,
            "notional_usd": 1230.0,
            "leverage": 3.0,
            "stop_price": 97_000.0,
            "take_price": 108_000.0,
            "tier": "T1",
            "regime": "UP·NORM·NEU",
        },
        reasoning="smoke test: synthetic record, not a real trade.",
    )

    print(f"→ audit_id:     {record.audit_id}")
    print(f"→ anchor_hash:  0x{audit.anchor_hash(record).hex()}")
    print(f"→ wallet:       {os.environ['CIRCLE_WALLET_ADDRESS']}")
    print(f"  ({arcscan_address(os.environ['CIRCLE_WALLET_ADDRESS'])})")
    print()
    print("→ submitting createJob via Circle Contract Execution API…")

    anchor = from_env_file(repo_root / ".env")
    try:
        result = anchor.anchor_record(record)
    finally:
        anchor.close()

    print()
    print(f"  state:         {result.state}")
    print(f"  circle_tx_id:  {result.circle_tx_id}")
    print(f"  arc_tx_hash:   {result.arc_tx_hash}")
    print(f"  description:   {result.description}")
    if result.arcscan_url:
        print(f"  arcscan:       {result.arcscan_url}")
    print()
    if result.arc_tx_hash:
        print("✓ on-chain anchor succeeded; click the arcscan link above.")
        return 0
    print("✗ no tx_hash yet — Circle may still be queuing. Re-run later.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
