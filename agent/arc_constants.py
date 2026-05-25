"""Arc Testnet network constants.

Addresses and event topic hashes for Arc Testnet (chain id 5042002).
Sourced from the Arcadia indexer (Apache-2.0), which has these wired
into a running production indexer — the live ``agent_jobs`` and
``agents`` collections are implicit attestations that the addresses
resolve correctly.

Re-verify against ``testnet.arcscan.app`` before any production use;
Circle may rotate impl addresses behind the proxies on testnet without
notice.

Attribution: derived from
`github.com/magooney-loon/arcadia/internal/chain/arc/arc.go`
(Apache-2.0, https://github.com/magooney-loon/arcadia/blob/main/LICENSE).
"""

from __future__ import annotations

from eth_utils import keccak

# Network identity
CHAIN_ID = 5042002  # 0x4CE292
CHAIN_NAME = "ARC-TESTNET"
CCTP_DOMAIN = 26

# Public RPC endpoints (read-only; write transactions go through Circle's
# Contract Execution API which routes through their own infrastructure).
RPC_HTTP_PRIMARY = "https://rpc.testnet.arc.network"
RPC_HTTP_DRPC = "https://rpc.drpc.testnet.arc.network"
RPC_HTTP_QUICKNODE = "https://rpc.quicknode.testnet.arc.network"

# ---------------------------------------------------- stablecoins

USDC = "0x3600000000000000000000000000000000000000"  # native gas + ERC-20
EURC = "0x89B50855Aa3bE2F677cD6303Cec089B5F319D72a"

# ---------------------------------------------------- CCTP v2

CCTP_TOKEN_MESSENGER = "0x8FE6B999Dc680CcFDD5Bf7EB0974218be2542DAA"
CCTP_MESSAGE_TRANSMITTER = "0xE737e5cEBEEBa77EFE34D4aa090756590b1CE275"
CCTP_TOKEN_MINTER = "0xb43db544E2c27092c107639Ad201b3dEfAbcF192"

# ---------------------------------------------------- Gateway

GATEWAY_WALLET = "0x0077777d7EBA4688BDeF3E311b846F25870A19B9"
GATEWAY_MINTER = "0x0022222ABE238Cc2C7Bb1f21003F0a260052475B"

# ---------------------------------------------------- ERC-8004 (identity)

ERC8004_IDENTITY = "0x8004A818BFB912233c491871b3d84c89A494BD9e"
ERC8004_REPUTATION = "0x8004B663056A597Dffe9eCcC1965A193B7388713"
ERC8004_VALIDATION = "0x8004Cb1BF31DAf7788923b405b754f57acEB4272"

# ---------------------------------------------------- ERC-8183 (commerce / jobs)

ERC8183_COMMERCE_PROXY = "0x0747EEf0706327138c69792bF28Cd525089e4583"
ERC8183_COMMERCE_IMPL = "0xA316fd02827242D537F84730F8a37D0BA5fd351a"

# ---------------------------------------------------- ERC-8183 function ABIs

# Per the ERC-8183 standard (eips.ethereum.org/EIPS/eip-8183).
ABI_CREATE_JOB = (
    "createJob(address,address,uint256,string,address)"
)
ABI_FUND = "fund(uint256,bytes)"
ABI_SUBMIT = "submit(uint256,bytes32,bytes)"
ABI_COMPLETE = "complete(uint256,bytes32,bytes)"
ABI_REJECT = "reject(uint256,bytes32,bytes)"

# USDC ERC-20 (standard).
ABI_APPROVE = "approve(address,uint256)"
ABI_ALLOWANCE = "allowance(address,address)"
ABI_BALANCE_OF = "balanceOf(address)"

# ---------------------------------------------------- ERC-8183 event topics

# `keccak256(event signature)` — the canonical topic0 for `eth_getLogs`
# filtering. Field types are the indexed and non-indexed arguments in
# declaration order. Verified against Arcadia's `events.go`.
TOPIC_JOB_CREATED = keccak(
    text="JobCreated(uint256,address,address,address,uint256,address)"
)
TOPIC_JOB_FUNDED = keccak(
    text="JobFunded(uint256,address,uint256)"
)
TOPIC_JOB_SUBMITTED = keccak(
    text="JobSubmitted(uint256,address,bytes32)"
)
TOPIC_JOB_COMPLETED = keccak(
    text="JobCompleted(uint256,address,bytes32)"
)
TOPIC_JOB_REJECTED = keccak(
    text="JobRejected(uint256,address,bytes32)"
)
TOPIC_PAYMENT_RELEASED = keccak(
    text="PaymentReleased(uint256,address,uint256)"
)
TOPIC_JOB_EXPIRED = keccak(text="JobExpired(uint256)")

# ---------------------------------------------------- ERC-8183 state machine

# Outcome states observed in Arcadia's `save_agent.go`. The lifecycle is:
#     created → funded → submitted → completed | rejected | expired → paid
# `paid` is a separate state post `PaymentReleased`, NOT bundled into
# `JobCompleted` — important for the audit record's `Outcome` enum.
JOB_STATE_CREATED = "created"
JOB_STATE_FUNDED = "funded"
JOB_STATE_SUBMITTED = "submitted"
JOB_STATE_COMPLETED = "completed"
JOB_STATE_REJECTED = "rejected"
JOB_STATE_EXPIRED = "expired"
JOB_STATE_PAID = "paid"

# ---------------------------------------------------- arcscan URL builders

def arcscan_tx(tx_hash: str) -> str:
    """Return the testnet arcscan URL for a tx hash."""
    return f"https://testnet.arcscan.app/tx/{tx_hash}"


def arcscan_address(addr: str) -> str:
    """Return the testnet arcscan URL for an address."""
    return f"https://testnet.arcscan.app/address/{addr}"
