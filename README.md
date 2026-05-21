# agora-perp-agent

An autonomous AI trading agent for crypto perpetuals. It runs on Arc testnet and
settles in USDC. Built for the Agora Agents Hackathon.

Right now it trades in paper mode (no real money). Prices come from a live
public feed, so the numbers stay honest.

## What it does

Once a minute, the agent:

1. Reads prices for several coins from a public market feed.
2. Forecasts which way each coin is likely to move, and by how much.
3. Subtracts trading costs (fees and slippage) to get the real edge.
4. Picks the best set of trades to hold, under a risk budget.
5. An AI agent looks at the whole board, plus news and sentiment, and decides
   what to open, hold, cut, or skip. It explains every decision.
6. Opens paper positions and tracks honest profit and loss.
7. Writes each decision and result to Arc as a small on-chain receipt, so anyone
   can verify what the agent did.

A live dashboard reads those receipts straight from the chain. It shows the
probability for each coin, the open positions, and the trade history with a link
to each on-chain proof.

## Why this design

- **The AI agent is the decision-maker.** It uses tools (forecast, cost,
  backtest, news) to gather what it needs, then decides and explains. It never
  invents numbers. The math is done by code, so the agent stays honest.
- **Every trade is verifiable.** Decisions and PnL are hashed onto Arc. You do
  not have to trust a screenshot. You can check the chain.
- **It is cheap.** Receipts settle on Arc for a fraction of a cent, paid in
  USDC.

We do not claim it prints money on a one-minute horizon. That is mostly noise.
What we show is an honest, autonomous, verifiable trading agent.

## Stack

- **Arc** (Circle's L1): settlement and USDC-denominated gas.
- **Circle Wallets**: the agent's wallet. Keys are never exposed to the model.
- **USDC**: collateral accounting and on-chain fees.
- **LLM agent**: reasoning, tool use, and decision explanations.
- **Python** engine: data, forecasting, cost model, risk, paper execution.

## Status

Work in progress during the hackathon. Building in public.

## Repository layout

```
core/       shared engine: forecasting, cost, risk, execution
agent_app/  the live agent and its tools
tests/      unit and parity tests
```

## License

MIT, see [`LICENSE`](LICENSE).
