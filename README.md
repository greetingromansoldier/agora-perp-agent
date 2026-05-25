# agora-perp-agent

An autonomous AI trading agent for crypto perpetual futures. Reads live
Hyperliquid markets, decides through an LLM grounded by a deterministic quant
core, and anchors every decision on **Arc testnet** as an
[ERC-8183](https://eips.ethereum.org/EIPS/eip-8183) commerce job so the track
record is independently verifiable on-chain.

Built for the **Agora Agents Hackathon** (Canteen × Circle × Arc, May 2026).
Paper-trading mode only — prices come from a live public feed, positions track
honest PnL, no real funds are at risk.

## Live proof on Arc Testnet

A working anchor (smoke-test synthetic record):

- **Tx**:
  [`0x3009e228…7039`](https://testnet.arcscan.app/tx/0x3009e228d1a0f8f35a54509fc0775228a4b83fd012a5bda3dd64177c928e7039)
- **From**: agent wallet `0x5b09…cf44` (Circle Developer-Controlled)
- **To**: ERC-8183 commerce proxy `0x0747…4583`
- **Function**: `createJob(provider, evaluator, expiredAt, description, hook)`
- **Description on-chain**: `agora:7cf484a9-…:0x9e8b9ad8…` — the audit record's
  keccak256 anchor hash, committed publicly.

Anyone can recompute `keccak256(canonical_json(record) || nonce)` on the
published record and compare it to the `description` field. If the hashes match,
the decision was committed at that block height — not edited later.

## What it does, end to end

1. **Stream** Hyperliquid markets over WebSocket (10 coins by default; L2
   orderbook + 5-minute candles + funding rates).
2. **Forecast** each coin's next-bar move with a baseline model composed of EMA
   / SMA / RSI / ADX / ATR signals fed into a logistic classifier
   (`core/forecast.py`).
3. **Cost-model** the round trip honestly: maker/taker bps + sqrt-law slippage
   based on book depth + signed funding for the hold horizon (`core/cost.py`).
4. **Rank** candidates by edge after cost (`core/allocate.py`), then hand the
   board to the LLM agent.
5. **LLM agent** (Gemini 3.1 Pro by default; rule-based fallback for tests)
   emits a structured `Decisions` JSON: per asset, one of
   `enter / hold / cut / flip / skip` with rationale, cited numbers, and a
   self-grounded confidence. Never invents data; reasoning is the only thing it
   produces from latent space.
6. **Size** each `enter` through the trading-context layer: coin-tier classifier
   → 3-axis regime tag (trend × vol × funding) → playbook multiplier →
   vol-targeted dollar risk → ATR-derived stop distance → three-cap leverage
   (venue / operational / liq-safety) → funding-drag check. Result: a
   `SizedCandidate` with `qty`, `leverage`, `stop_price`, `take_price`, and a
   full per-step audit dict.
7. **Risk-gate** the sized candidate against per-position cap, total exposure
   cap, edge threshold, and uniqueness (`core/risk.py`).
8. **Execute** in paper mode (`core/execute.py`): `open_sized` records stops on
   the position, `check_stops` fires every tick if mark breaches either level,
   `tick` accrues funding and marks PnL.
9. **Anchor** the decision on Arc through Circle's Developer-Controlled-Wallets
   Contract Execution API. Each audit record is hashed with a fresh nonce, the
   hash goes into an ERC-8183 `createJob.description`, and the tx hash links
   back to arcscan.

## Why this design

- **The math is honest, the LLM is the consultant.** Sizing, leverage, stops,
  and PnL are pure deterministic Python — no model hallucination touches a
  number. The LLM picks *which* trades to take and explains *why*; it never
  picks *how big*.
- **The decision is verifiable.** The keccak256 hash anchors the *exact* state
  (market snapshot, forecast, cost assessment, sizing inputs) the decision was
  made on. Anyone holding the off-chain record can prove or disprove that we
  made the call at that block.
- **Settlement is cheap.** Arc denominates gas in USDC and fees are micro-cents
  per anchor. Circle's MPC custody signs every tx — we never hold a private key.

## Engine architecture

```
data (WS L2 + 5m candles + funding)
      |
      v
forecast (logistic over ATR/EMA/SMA/RSI/ADX/realized-vol)
      |
      v
cost (fees + sqrt-law slippage + signed funding)
      |
      v
allocate (rank by edge after cost)
      |
      v
LLM agent (Gemini 3.1 Pro / rule fallback)
      |
      v
sizing layer
      |  classify_tier -> regime tag -> playbook mult -> vol target
      |  -> ATR stops -> 3-cap leverage -> funding drag
      v
risk gate (veto on edge / uniqueness / exposure / per-position cap)
      |
      v
SimExecutor.open_sized -> Position with stop/take registered
      |
      v
audit record (keccak256 + nonce, sqlite write-ahead)
      |
      v
ERC-8183 createJob on Arc via Circle Contract Execution API
      |
      v
tx_hash on arcscan -> dashboard
```

## Tech stack

- **Arc** (Circle's L1, USDC-as-gas, sub-second finality) — settlement
- **Circle Wallets** (Developer-Controlled, MPC custody) — signing
- **ERC-8183 commerce framework** — verifiable decision anchor
- **Hyperliquid public feeds** — market data, no API key needed
- **Gemini 3.1 Pro** — LLM agent (rule-based fallback included)
- **Python 3.12 + uv** — engine, async agent loop, sqlite audit store
- **ccxt.pro** — WebSocket streaming
- **eth-utils + cryptography** — keccak, RSA-OAEP entity-secret crypto

## Status

| Layer | Status | | --- | --- | | L0 — scaffold, tests, eps helpers | ✓ | | L1
— data ingestion (REST + WS, multi-coin parallel) | ✓ | | L2 — forecast
(baseline) | ✓ | | L3 — cost model (sqrt-law slippage + funding) | ✓ | | L4 —
greedy allocation | ✓ | | L5 — risk gate (5-rule veto) | ✓ | | L5.5 — sizing
layer (tier × regime × ATR stops × leverage) | ✓ | | L6 — LLM agent (Gemini +
structured rationale) | ✓ | | L7 — bounded backtest as agent tool | planned | |
L8 — sim execution (open / close / tick / check_stops) | ✓ | | L9 — Arc receipts
via ERC-8183 (proven live) | ✓ MVP | | L10 — static dashboard reading Arc | in
progress | | L11 — Loom + Agora submission | planned |

**Test coverage**: 206 tests across the engine, all passing under
`uv run pytest`.

## Project layout

```
core/                   # engine — deterministic quant math
  data.py               # HyperliquidSource (REST) + parallel fetch_board
  ws.py                 # HyperliquidWsSource (ccxt.pro)
  contracts.py          # frozen dataclasses
  forecast.py           # BaselineForecast (logistic)
  cost.py               # CostModel (sqrt-law slippage + funding)
  allocate.py           # greedy ranked allocation
  risk.py               # 5-rule veto gate
  sizing.py             # TierRegimeSizer
  regime.py             # BaselineRegimeClassifier + BTC override
  tiers.py              # classify_tier
  stops.py              # derive_stops_takes
  leverage.py           # choose_leverage (3-cap pipeline)
  execute.py            # SimExecutor (paper trading)
  synthesis.py          # Synthesizer Protocol + Rule / Gemini impls
  agent_tools.py        # LLM tool surface
  eps.py / indicators.py  # epsilon helpers + SMA/EMA/ATR/RSI/ADX
agent/                  # on-chain anchor + audit + snapshot infra
  audit.py              # AuditRecord + canonical_json + keccak256 + sqlite
  on_chain.py           # CircleAnchor — ERC-8183 createJob via Circle API
  anchor_worker.py      # background thread draining records to Arc
  snapshot.py           # dashboard JSON writer + hydrate from sqlite
  arc_constants.py      # verified Arc Testnet addresses + topic hashes
cli/run.py              # canonical agent runner (live paper trader)
scripts/
  circle_setup.py       # one-time Circle Wallet provisioning
  smoke_anchor.py       # smoke-test the full anchor pipeline
dashboard/              # static GitHub Pages site
  index.html / styles.css / app.js
  data/snapshot.json    # written by the running agent (gitignored output)
tests/                  # pytest — 206 tests
```

The trading agent's tuned alpha (regime-conditioned weight sets, Optuna sweep
outputs) lives in a private repo. The public engine here ships the baseline
implementation that follows the documented quant-research literature literally —
the tuning is what we protect, the math itself isn't.

## Run it locally

```bash
# 1. install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. clone + test
git clone https://github.com/greetingromansoldier/agora-perp-agent
cd agora-perp-agent
uv run --with pytest python -m pytest          # 206 tests should pass

# 3. credentials
cp .env.example .env
# Edit .env: GEMINI_API_KEY (free at aistudio.google.com).
# For on-chain anchoring add CIRCLE_API_KEY too (console.circle.com,
# choose 'Test API Key' — testnet, no real money at risk).

# 4. one-time Circle Wallet provisioning (only if you want --on-chain)
uv run --with httpx --with cryptography python scripts/circle_setup.py
# Follow the printed instructions: paste the ciphertext into Circle
# Console → Web3 Services → Wallets → Register Entity Secret
# Ciphertext. Re-run the script. Then visit faucet.circle.com and
# claim testnet USDC for the wallet address it printed.

# 5. smoke-test the anchor pipeline (optional)
uv run --with httpx --with ccxt --with cryptography \
       --with eth_utils --with pycryptodome \
       python scripts/smoke_anchor.py
# Prints an arcscan URL — open it; you'll see your first ERC-8183
# anchor on Arc Testnet.

# 6. run the live paper trader
uv run --with httpx --with ccxt --with google-genai \
       --with cryptography --with eth_utils --with pycryptodome \
       python cli/run.py --agent gemini --on-chain \
                         --starting-balance 10000 --base-risk 0.02 \
                         --max-leverage 20 --max-notional 50000 \
                         --max-exposure 200000

# Terminal shows the live board + AGENT TRACE. The dashboard at
# dashboard/index.html (serve via `python -m http.server -d dashboard`
# or via deployed GitHub Pages) shows positions, equity curve,
# decisions with on-chain links, and trade history. The dashboard
# updates every 5 seconds from dashboard/data/snapshot.json.
```

`cli/run.py --help` lists every flag (notional, hold horizon, max leverage /
notional / exposure, agent cadence, LLM model, etc.).

### No on-chain mode

Omit `--on-chain` and skip steps 4-5 if you just want to watch the agent trade.
Everything except the Arc anchor still works end-to-end against real-time
Hyperliquid prices.

## License

MIT — see [`LICENSE`](LICENSE).

Arc Testnet constants and ERC-8183 event topic hashes were verified against the
[Arcadia](https://github.com/magooney-loon/arcadia) indexer (Apache-2.0). The
Arcadia codebase is the implicit attestation that the proxy addresses resolve
correctly.
