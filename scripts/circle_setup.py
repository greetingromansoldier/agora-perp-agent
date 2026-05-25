"""Provision a Circle Developer-Controlled Wallet on Arc Testnet.

Idempotent: re-runs reuse the entity secret, wallet set, and wallet
already persisted in `.env` rather than creating duplicates.

Flow (per Circle's Developer-Controlled Wallets v1 API):

    1. Read CIRCLE_API_KEY from .env.
    2. Generate a 32-byte entity_secret (hex) — or reuse the one already
       in .env if present.
    3. Fetch Circle's RSA public key via /v1/w3s/config/entity/publicKey.
    4. Encrypt entity_secret with that key (RSA-OAEP-SHA256). Each API
       call needs its own fresh ciphertext — Circle treats each as
       single-use.
    5. Create a wallet set (or reuse the saved one).
    6. Create one wallet on ARC-TESTNET in that set (or reuse the saved
       one). `accountType=EOA` so the wallet is a plain externally-owned
       account, signable via Circle's MPC backend with no smart-contract
       deployment cost.
    7. Persist entity_secret + wallet_set_id + wallet_id + wallet_address
       back to .env so subsequent script runs and the trading agent can
       read them.
    8. Print the faucet URL the operator needs to hit next.

Run from the project root:

    uv run --with httpx --with cryptography python scripts/circle_setup.py
"""

from __future__ import annotations

import base64
import secrets
import sys
import uuid
from pathlib import Path

import httpx
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

_CIRCLE_API_BASE = "https://api.circle.com"
_BLOCKCHAIN = "ARC-TESTNET"
_FAUCET_URL = "https://faucet.circle.com"


def _load_env(env_path: Path) -> dict[str, str]:
    """Parse `KEY=VALUE` lines into a dict. Comments / blanks skipped."""
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


def _write_env(env_path: Path, updates: dict[str, str]) -> None:
    """Update specific keys in-place, preserving comments and order.

    If a key in `updates` doesn't yet exist in the file, append it at the
    end. Only the `KEY=` lines we know about are rewritten — everything
    else (comments, blank lines, other keys) stays exactly as it was.
    """
    existing_lines = (
        env_path.read_text().splitlines() if env_path.exists() else []
    )
    keys_seen: set[str] = set()
    new_lines: list[str] = []
    for line in existing_lines:
        stripped = line.strip()
        if "=" in stripped and not stripped.startswith("#"):
            key = stripped.split("=", 1)[0].strip()
            if key in updates:
                new_lines.append(f"{key}={updates[key]}")
                keys_seen.add(key)
                continue
        new_lines.append(line)
    # Append any keys we didn't see.
    for key, value in updates.items():
        if key not in keys_seen:
            new_lines.append(f"{key}={value}")
    env_path.write_text("\n".join(new_lines) + "\n")


def _encrypt_entity_secret(secret_hex: str, public_key_pem: str) -> str:
    """Encrypt 32-byte secret with Circle's RSA-OAEP-SHA256 key.

    Returns base64-encoded ciphertext suitable for Circle's
    `entitySecretCiphertext` field. Each ciphertext is single-use — the
    caller generates a fresh one per API call.
    """
    public_key = serialization.load_pem_public_key(public_key_pem.encode())
    secret_bytes = bytes.fromhex(secret_hex)
    if len(secret_bytes) != 32:
        raise ValueError(
            f"entity secret must be 32 bytes, got {len(secret_bytes)}"
        )
    ciphertext = public_key.encrypt(
        secret_bytes,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None,
        ),
    )
    return base64.b64encode(ciphertext).decode()


def _fetch_public_key(client: httpx.Client, api_key: str) -> str:
    r = client.get(
        f"{_CIRCLE_API_BASE}/v1/w3s/config/entity/publicKey",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    r.raise_for_status()
    return r.json()["data"]["publicKey"]


def _print_register_instructions(
    entity_secret: str, public_key_pem: str,
) -> None:
    """Print manual entity-secret registration instructions.

    Circle's entity-secret registration is a one-time console-UI action
    (no public API endpoint as of testnet v1). We compute the
    ciphertext locally and ask the operator to paste it into the
    console. Re-running this script after registration completes will
    pick up where it left off — `CIRCLE_ENTITY_SECRET` is already
    persisted in .env so the same plaintext is used.
    """
    ciphertext = _encrypt_entity_secret(entity_secret, public_key_pem)
    print()
    print("=" * 68)
    print("ENTITY SECRET NEEDS ONE-TIME REGISTRATION IN CIRCLE CONSOLE")
    print("=" * 68)
    print()
    print("1. open https://console.circle.com")
    print("2. left sidebar → 'Configurator' → 'Developer-Controlled Wallets'")
    print("   (or search 'entity secret' in the console)")
    print("3. paste this ciphertext when prompted:")
    print()
    print(ciphertext)
    print()
    print("4. click 'Register' / 'Confirm'.")
    print("5. re-run this script — it'll resume from wallet creation.")
    print("=" * 68)


def _create_wallet_set(
    client: httpx.Client,
    api_key: str,
    ciphertext: str,
    name: str,
) -> str:
    r = client.post(
        f"{_CIRCLE_API_BASE}/v1/w3s/developer/walletSets",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "idempotencyKey": str(uuid.uuid4()),
            "name": name,
            "entitySecretCiphertext": ciphertext,
        },
    )
    if r.status_code >= 400:
        raise RuntimeError(
            f"create_wallet_set failed [{r.status_code}]: {r.text}"
        )
    return r.json()["data"]["walletSet"]["id"]


def _create_wallet(
    client: httpx.Client,
    api_key: str,
    ciphertext: str,
    wallet_set_id: str,
) -> tuple[str, str]:
    r = client.post(
        f"{_CIRCLE_API_BASE}/v1/w3s/developer/wallets",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "idempotencyKey": str(uuid.uuid4()),
            "blockchains": [_BLOCKCHAIN],
            "count": 1,
            "walletSetId": wallet_set_id,
            "entitySecretCiphertext": ciphertext,
            "accountType": "EOA",
        },
    )
    if r.status_code >= 400:
        raise RuntimeError(f"create_wallet failed [{r.status_code}]: {r.text}")
    wallet = r.json()["data"]["wallets"][0]
    return wallet["id"], wallet["address"]


def main() -> int:
    env_path = Path(__file__).resolve().parent.parent / ".env"
    env = _load_env(env_path)

    api_key = env.get("CIRCLE_API_KEY", "").strip()
    if not api_key:
        print(
            "error: CIRCLE_API_KEY missing from .env. "
            "Generate a TEST_API_KEY at console.circle.com.",
            file=sys.stderr,
        )
        return 1

    entity_secret = env.get("CIRCLE_ENTITY_SECRET", "").strip()
    if not entity_secret:
        entity_secret = secrets.token_hex(32)
        print(f"→ generated entity_secret ({len(entity_secret)} hex chars)")
    else:
        print("→ reusing existing CIRCLE_ENTITY_SECRET")

    updates: dict[str, str] = {"CIRCLE_ENTITY_SECRET": entity_secret}

    # Persist the entity_secret to .env *before* any API call so a
    # crash here doesn't leave us with an in-memory-only secret.
    _write_env(env_path, {"CIRCLE_ENTITY_SECRET": entity_secret})

    with httpx.Client(timeout=30.0) as client:
        print("→ fetching Circle RSA public key")
        public_key_pem = _fetch_public_key(client, api_key)

        wallet_set_id = env.get("CIRCLE_WALLET_SET_ID", "").strip()
        if not wallet_set_id:
            print("→ creating wallet set")
            ciphertext = _encrypt_entity_secret(entity_secret, public_key_pem)
            try:
                wallet_set_id = _create_wallet_set(
                    client, api_key, ciphertext, name="agora-perp-agent",
                )
            except RuntimeError as e:
                if "156016" in str(e):
                    _print_register_instructions(
                        entity_secret, public_key_pem,
                    )
                    return 2
                raise
            print(f"  wallet_set_id: {wallet_set_id}")
        else:
            print(f"→ reusing wallet set: {wallet_set_id}")
        updates["CIRCLE_WALLET_SET_ID"] = wallet_set_id

        wallet_id = env.get("CIRCLE_WALLET_ID", "").strip()
        wallet_address = env.get("CIRCLE_WALLET_ADDRESS", "").strip()
        if not wallet_id or not wallet_address:
            print(f"→ provisioning wallet on {_BLOCKCHAIN}")
            ciphertext = _encrypt_entity_secret(entity_secret, public_key_pem)
            wallet_id, wallet_address = _create_wallet(
                client, api_key, ciphertext, wallet_set_id,
            )
            print(f"  wallet_id: {wallet_id}")
            print(f"  wallet_address: {wallet_address}")
        else:
            print(f"→ reusing wallet: {wallet_address}")
        updates["CIRCLE_WALLET_ID"] = wallet_id
        updates["CIRCLE_WALLET_ADDRESS"] = wallet_address

    _write_env(env_path, updates)
    print("→ .env updated.")
    print()
    print("=" * 68)
    print("next step — claim testnet USDC for the new wallet:")
    print(f"  1. open {_FAUCET_URL}")
    print(f"  2. paste address: {wallet_address}")
    print("  3. choose 'Arc Testnet'")
    print("  4. click claim. allow ~30 seconds for the mint to land.")
    print()
    print("verify the balance afterwards:")
    print(
        f"  https://testnet.arcscan.app/address/{wallet_address}"
    )
    print("=" * 68)
    return 0


if __name__ == "__main__":
    sys.exit(main())
