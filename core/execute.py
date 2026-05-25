"""paper-trade execution + funding accrual + mark-to-market.

A `SimExecutor` opens and closes virtual positions against a `MarketData`
snapshot, applying the same sqrt-law slippage and the same fee schedule
the cost model uses. There is no network and no venue; PnL is honest
because the slippage and funding formulas are identical to what L3
predicted.

The executor is *stateless* (only configuration); one instance is safe
to share across many ticks. The mutable state lives in `PortfolioState`,
which the executor receives by reference and updates in place.
"""

from __future__ import annotations

import math

from core.contracts import (
    AllocationCandidate,
    FeeSchedule,
    Fill,
    MarketData,
    PortfolioState,
    Position,
    SizedCandidate,
)

_EPS = 1e-12
_BPS = 10_000.0


class SimExecutor:
    """Paper executor: opens / closes positions, accrues funding, marks PnL."""

    def __init__(self, schedule: FeeSchedule) -> None:
        """Configure the venue fee schedule (also drives slippage and cadence).

        Args:
            schedule: per-venue parameters; ``taker_bps`` is charged at
                open and close, ``slippage_k`` and ``flat_slippage_bps``
                drive the fill price vs mark, and ``funding_period_hours``
                is the cadence used to prorate funding accrual.
        """
        self._schedule = schedule

    def open(
        self,
        candidate: AllocationCandidate,
        market: MarketData,
        portfolio: PortfolioState,
    ) -> Fill:
        """Open a paper position from a ranked candidate.

        Args:
            candidate: asset/side/notional from L4.
            market: snapshot whose ``mark_price`` and ``book_depth`` drive
                the fill price.
            portfolio: account state; mutated on success (fee deducted,
                position inserted).

        Returns:
            A `Fill` with ``is_open=True`` and ``realized_pnl_usd=0``.

        Raises:
            ValueError: if a position is already open on ``candidate.asset``
                or the candidate's asset does not match the market's.
        """
        if portfolio.has(candidate.asset):
            raise ValueError(
                f"position already open for `{candidate.asset}`; close it first"
            )
        if candidate.asset != market.asset:
            raise ValueError(
                f"candidate asset {candidate.asset!r} does not match "
                f"market asset {market.asset!r}"
            )

        slip_frac = self._side_slippage_bps(candidate.notional, market) / _BPS
        if candidate.side == "long":
            fill_price = market.mark_price * (1.0 + slip_frac)
        else:
            fill_price = market.mark_price * (1.0 - slip_frac)

        qty = candidate.notional / fill_price
        fee_usd = candidate.notional * self._schedule.taker_bps / _BPS

        portfolio.balance_usd -= fee_usd
        portfolio.positions[candidate.asset] = Position(
            asset=candidate.asset,
            side=candidate.side,
            qty=qty,
            entry_price=fill_price,
            entry_time=market.timestamp,
            last_mark=market.mark_price,
            last_funding_ts=market.timestamp,
        )

        return Fill(
            asset=candidate.asset,
            timestamp=market.timestamp,
            side=candidate.side,
            qty=qty,
            price=fill_price,
            fee_paid_usd=fee_usd,
            is_open=True,
            realized_pnl_usd=0.0,
        )

    def open_sized(
        self,
        sized: SizedCandidate,
        market: MarketData,
        portfolio: PortfolioState,
    ) -> Fill:
        """Open a position from a docs-derived `SizedCandidate`.

        Like `open()` but uses `sized.qty` directly instead of computing
        qty from a naive notional, and registers `stop_price`,
        `take_price`, and `stop_take_plan` on the `Position` so
        `check_stops` can fire stops/takes on subsequent ticks.

        Args:
            sized: post-sizing trade plan from `TierRegimeSizer.size`.
            market: snapshot whose `mark_price` and `book_depth` drive
                the fill price.
            portfolio: account state; mutated on success.

        Returns:
            A `Fill` with `is_open=True` and `realized_pnl_usd=0`.

        Raises:
            ValueError: position already open on this asset, asset
                mismatch with market, or non-positive sized qty.
        """
        candidate = sized.candidate
        if portfolio.has(candidate.asset):
            raise ValueError(
                f"position already open for `{candidate.asset}`; close it first"
            )
        if candidate.asset != market.asset:
            raise ValueError(
                f"candidate asset {candidate.asset!r} does not match "
                f"market asset {market.asset!r}"
            )
        if sized.qty <= 0.0:
            raise ValueError(
                f"sized.qty must be positive, got {sized.qty} "
                f"(playbook may be stand-down — caller should skip)"
            )

        slip_frac = self._side_slippage_bps(sized.notional, market) / _BPS
        if candidate.side == "long":
            fill_price = market.mark_price * (1.0 + slip_frac)
        else:
            fill_price = market.mark_price * (1.0 - slip_frac)

        fee_usd = sized.notional * self._schedule.taker_bps / _BPS

        portfolio.balance_usd -= fee_usd
        portfolio.positions[candidate.asset] = Position(
            asset=candidate.asset,
            side=candidate.side,
            qty=sized.qty,
            entry_price=fill_price,
            entry_time=market.timestamp,
            last_mark=market.mark_price,
            last_funding_ts=market.timestamp,
            stop_price=sized.stop_price,
            take_price=sized.take_price,
            stop_take_plan=sized.stop_take_plan,
        )

        return Fill(
            asset=candidate.asset,
            timestamp=market.timestamp,
            side=candidate.side,
            qty=sized.qty,
            price=fill_price,
            fee_paid_usd=fee_usd,
            is_open=True,
            realized_pnl_usd=0.0,
        )

    def close(
        self,
        asset: str,
        market: MarketData,
        portfolio: PortfolioState,
    ) -> Fill:
        """Close an open paper position at the current mark (with exit slippage).

        Funding is accrued first so the realized cashflow includes the
        latest sliver. After this the position is removed and the realized
        PnL is added to ``balance_usd``.

        Args:
            asset: market symbol to close.
            market: snapshot for exit price and final funding accrual; its
                ``asset`` field must match ``asset``.
            portfolio: account state; mutated.

        Returns:
            A `Fill` with ``is_open=False`` and signed
            ``realized_pnl_usd = price_pnl + accrued_funding − exit_fee``.

        Raises:
            ValueError: if no position is open on the asset, or if the
                market's asset does not match the close request.
        """
        if asset != market.asset:
            raise ValueError(
                f"market asset {market.asset!r} does not match close request {asset!r}"
            )
        pos = portfolio.positions.get(asset)
        if pos is None:
            raise ValueError(f"no open position for `{asset}` to close")

        # Roll funding forward to `market.timestamp` so the realized
        # cashflow on close includes any accrual since the last tick.
        self._accrue_funding(pos, market)

        exit_notional = pos.qty * market.mark_price
        slip_frac = self._side_slippage_bps(exit_notional, market) / _BPS
        if pos.side == "long":
            exit_price = market.mark_price * (1.0 - slip_frac)
            price_pnl = (exit_price - pos.entry_price) * pos.qty
        else:
            exit_price = market.mark_price * (1.0 + slip_frac)
            price_pnl = (pos.entry_price - exit_price) * pos.qty

        exit_fee_usd = pos.qty * exit_price * self._schedule.taker_bps / _BPS
        realized = price_pnl + pos.accrued_funding_usd - exit_fee_usd

        portfolio.balance_usd += realized
        del portfolio.positions[asset]

        return Fill(
            asset=asset,
            timestamp=market.timestamp,
            side=pos.side,
            qty=pos.qty,
            price=exit_price,
            fee_paid_usd=exit_fee_usd,
            is_open=False,
            realized_pnl_usd=realized,
        )

    def tick(self, market: MarketData, portfolio: PortfolioState) -> None:
        """Mark and accrue funding for the position on ``market.asset``.

        No-op when there is no open position on the asset. Idempotent on
        repeated calls with the same ``market.timestamp`` (funding is
        accrued at most once per timestamp); ``last_mark`` is still
        refreshed in that case so unrealized PnL reflects the snapshot.
        """
        pos = portfolio.positions.get(market.asset)
        if pos is None:
            return
        # Always refresh mark so unrealized PnL is current.
        pos.last_mark = market.mark_price
        if market.timestamp <= pos.last_funding_ts:
            return
        self._accrue_funding(pos, market)

    def check_stops(
        self,
        market: MarketData,
        portfolio: PortfolioState,
    ) -> Fill | None:
        """Fire stop or take on `market.asset` if the level was breached.

        Inspects the position's `stop_price` and `take_price` (set at
        `open_sized` time) against the current `market.mark_price` and
        closes the position via `close()` if either is hit. Returns the
        closing `Fill` on a hit, `None` otherwise.

        No-op when there is no open position on the asset, no stop/take
        registered, or neither level has been crossed. The stop check
        runs before the take check — if both would fire on the same
        tick (price gap), the stop wins (worst case is recorded).

        Args:
            market: current snapshot; `mark_price` is the trigger basis.
            portfolio: account state; mutated if the position closes.

        Returns:
            The closing `Fill` if a stop or take fired; `None` if no
            action taken.
        """
        pos = portfolio.positions.get(market.asset)
        if pos is None:
            return None
        if pos.stop_price is not None:
            if pos.side == "long" and market.mark_price <= pos.stop_price:
                return self.close(market.asset, market, portfolio)
            if pos.side == "short" and market.mark_price >= pos.stop_price:
                return self.close(market.asset, market, portfolio)
        if pos.take_price is not None:
            if pos.side == "long" and market.mark_price >= pos.take_price:
                return self.close(market.asset, market, portfolio)
            if pos.side == "short" and market.mark_price <= pos.take_price:
                return self.close(market.asset, market, portfolio)
        return None

    # ------------------------------------------------------------------ internals

    def _accrue_funding(self, pos: Position, market: MarketData) -> None:
        """Add the elapsed funding cashflow to ``pos.accrued_funding_usd``.

        Convention: ``market.funding_rate > 0`` means longs pay shorts. A
        long position with positive funding accrues a negative cashflow
        (we pay); a short accrues a positive one (we receive). The
        cashflow is scaled by ``elapsed_minutes / period_minutes``.

        Also updates ``pos.last_mark`` so a single call to this method
        keeps mark-to-market and funding accrual aligned.
        """
        pos.last_mark = market.mark_price
        period_min = self._schedule.funding_period_hours * 60.0
        if period_min <= _EPS:
            return
        elapsed_min = (market.timestamp - pos.last_funding_ts).total_seconds() / 60.0
        if elapsed_min <= 0.0:
            return
        notional = pos.qty * market.mark_price
        sign = -1.0 if pos.side == "long" else 1.0
        cashflow = sign * market.funding_rate * notional * (elapsed_min / period_min)
        pos.accrued_funding_usd += cashflow
        pos.last_funding_ts = market.timestamp

    def _side_slippage_bps(self, notional: float, market: MarketData) -> float:
        """One-side slippage in bps via sqrt-law, with flat fallback.

        Identical formula to ``CostModel._side_slippage_bps`` so the paper
        simulator never produces better fills than the cost model predicted.
        """
        depth = market.book_depth
        if depth is None or depth <= _EPS:
            return self._schedule.flat_slippage_bps
        ratio = notional / depth
        return self._schedule.slippage_k * math.sqrt(ratio) * _BPS
