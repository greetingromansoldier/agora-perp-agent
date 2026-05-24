"""WebSocket-backed Hyperliquid data source via ccxt.pro.

`HyperliquidWsSource` implements the `DataSource` protocol but
populates its `MarketData` from a thread-safe cache that a background
asyncio loop keeps fresh over a WebSocket connection. The on-tick
`fetch()` call therefore does no network I/O.

Three streams populate the cache:

1. **L2 order book** — pushed over WebSocket
   (`exchange.watch_order_book`). Sub-second latency, no rate limit.
2. **OHLCV history** — refreshed periodically over REST
   (`exchange.fetch_ohlcv`). Closed candles never change; the live one
   updates within its interval. We refresh every ``ohlcv_refresh_s``
   seconds (default 30 s) — at 5-minute candles that easily catches each
   bar's close while costing ~0.3 REST calls per coin per second.
3. **Mark + funding** — taken from `metaAndAssetCtxs` via the existing
   `HyperliquidSource` (and its 2 s locked cache). One call for the whole
   universe, shared across all coins.

Combined request rate at 10 coins, 1-second dashboard refresh, 5-minute
candles: ~1 REST call per second (vs ~20 RPS for the all-REST pattern),
plus a single long-lived WS connection. Comfortably inside HL's published
limits.
"""

from __future__ import annotations

import asyncio
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

from core.contracts import MarketData, OhlcBar
from core.data import HyperliquidSource


@dataclass(slots=True)
class _CoinCache:
    """Per-coin live cache filled by the background WS task."""

    bars: tuple[OhlcBar, ...] = field(default_factory=tuple)
    bids: tuple[tuple[float, float], ...] = field(default_factory=tuple)
    asks: tuple[tuple[float, float], ...] = field(default_factory=tuple)
    bars_fetched_at: float = 0.0  # monotonic clock

    def has_book(self) -> bool:
        return bool(self.bids) and bool(self.asks)


class HyperliquidWsSource:
    """`DataSource` that streams L2 over WebSocket and polls the rest.

    A daemon thread runs an asyncio event loop with one ccxt.pro client
    feeding the local cache. `fetch()` is a pure cache read.

    Args:
        coins: HL coin symbols to subscribe to.
        interval: candle interval for the OHLCV stream.
        band_bps: depth aggregation band (passed to the existing
            `HyperliquidSource._compute_depth`).
        ohlcv_refresh_s: REST refresh cadence for OHLCV history.
        book_levels: levels per side requested from the WS feed.
    """

    def __init__(
        self,
        coins: list[str],
        *,
        interval: str = "5m",
        band_bps: float = 50.0,
        ohlcv_refresh_s: float = 30.0,
        book_levels: int = 20,
    ) -> None:
        self._coins = list(coins)
        self._interval = interval
        self._band_bps = band_bps
        self._ohlcv_refresh_s = ohlcv_refresh_s
        self._book_levels = book_levels
        self._caches: dict[str, _CoinCache] = {c: _CoinCache() for c in coins}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._rest = HyperliquidSource(interval=interval, band_bps=band_bps)
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="hl-ws"
        )
        self._thread.start()

    def fetch(self, asset: str, window: int) -> MarketData:
        """Return the latest snapshot for `asset` from the local cache.

        Args:
            asset: HL coin symbol.
            window: trailing candle count to include.

        Returns:
            A `MarketData` built from the cached candles, the cached
            order book, and mark/funding from the REST meta cache.

        Raises:
            ValueError: if the coin is not subscribed or the WS task
                hasn't populated its cache yet.
        """
        with self._lock:
            cache = self._caches.get(asset)
            if cache is None:
                raise ValueError(
                    f"asset {asset!r} not in WS subscription list"
                )
            if not cache.bars:
                raise ValueError(
                    f"no candle data yet for {asset!r}; WS warming up"
                )
            if not cache.has_book():
                raise ValueError(
                    f"no orderbook yet for {asset!r}; WS warming up"
                )
            bars = cache.bars[-window:]
            bids = cache.bids
            asks = cache.asks

        # Mark + funding via the existing locked meta cache.
        mark, funding = HyperliquidSource._parse_ctx(self._rest._meta(), asset)

        mid = 0.5 * (bids[0][0] + asks[0][0])
        depth = HyperliquidSource._compute_depth(
            bids, asks, mid, self._band_bps
        )

        return MarketData(
            asset=asset,
            timestamp=bars[-1].timestamp,
            mark_price=mark,
            funding_rate=funding,
            bars=bars,
            book_depth=depth,
        )

    def wait_ready(self, timeout: float = 10.0) -> bool:
        """Block until every coin has at least one bar + book, or timeout."""
        return self._ready.wait(timeout=timeout)

    def close(self) -> None:
        """Signal the background loop to stop and wait for the thread."""
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=3.0)

    # ----------------------------------------------------------- internals

    def _run(self) -> None:
        try:
            asyncio.run(self._async_main())
        except Exception as e:  # noqa: BLE001
            print(f"[ws thread crashed] {e}", file=sys.stderr)

    async def _async_main(self) -> None:
        try:
            import ccxt.pro as ccxtpro
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "HyperliquidWsSource requires ccxt. "
                "Install via `uv add ccxt` or `pip install ccxt`."
            ) from e

        exchange = ccxtpro.hyperliquid(
            {"enableRateLimit": True, "options": {"defaultType": "swap"}}
        )
        try:
            await self._load_markets_with_retry(exchange)

            tasks: list[asyncio.Task] = []
            symbols: dict[str, str] = {}
            for coin in self._coins:
                symbol = self._resolve_symbol(exchange, coin)
                symbols[coin] = symbol
                tasks.append(
                    asyncio.create_task(self._watch_book_loop(exchange, symbol, coin))
                )
                tasks.append(
                    asyncio.create_task(self._poll_ohlcv_loop(exchange, symbol, coin))
                )

            tasks.append(asyncio.create_task(self._readiness_watcher()))

            # Idle until asked to stop.
            while not self._stop.is_set():
                await asyncio.sleep(0.2)

            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            await exchange.close()

    async def _load_markets_with_retry(
        self, exchange, attempts: int = 6, initial_delay_s: float = 2.0
    ) -> None:
        """Call `exchange.load_markets()` with backoff on rate-limits.

        HL throttles aggressively after any heavy REST polling; the very
        first WS-source bootstrap can land mid-throttle. We retry with
        an expanding delay so the cool-off window can pass.
        """
        delay = initial_delay_s
        last_err: Exception | None = None
        for i in range(attempts):
            try:
                await exchange.load_markets()
                return
            except Exception as e:  # noqa: BLE001
                last_err = e
                if "429" in str(e) or "Too Many Requests" in str(e):
                    print(
                        f"[ws bootstrap] load_markets 429, retry {i + 1}/{attempts} "
                        f"in {delay:.1f}s",
                        file=sys.stderr,
                    )
                    await asyncio.sleep(delay)
                    delay = min(delay * 1.6, 30.0)
                    continue
                raise
        if last_err is not None:
            raise last_err

    async def _readiness_watcher(self) -> None:
        """Set the `ready` event once every coin cache has bars + book."""
        while not self._stop.is_set():
            await asyncio.sleep(0.2)
            with self._lock:
                all_ready = all(
                    bool(c.bars) and c.has_book() for c in self._caches.values()
                )
            if all_ready:
                self._ready.set()
                return

    @staticmethod
    def _resolve_symbol(exchange, coin: str) -> str:
        """Map a HL coin shortcode (``"BTC"``) to a ccxt swap symbol."""
        candidates = [
            f"{coin}/USDC:USDC",
            f"{coin}/USDT:USDT",
        ]
        for s in candidates:
            if s in exchange.markets:
                return s
        for s in exchange.markets:
            if s.startswith(f"{coin}/") and exchange.markets[s].get("swap"):
                return s
        raise ValueError(f"no ccxt swap symbol found for HL coin {coin!r}")

    async def _watch_book_loop(self, exchange, symbol: str, coin: str) -> None:
        while not self._stop.is_set():
            try:
                ob = await exchange.watch_order_book(symbol, limit=self._book_levels)
                bids = tuple(
                    (float(b[0]), float(b[1])) for b in ob.get("bids", [])
                )
                asks = tuple(
                    (float(a[0]), float(a[1])) for a in ob.get("asks", [])
                )
                with self._lock:
                    self._caches[coin].bids = bids
                    self._caches[coin].asks = asks
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001
                print(f"[ws book {coin}] {e}", file=sys.stderr)
                await asyncio.sleep(1.0)

    async def _poll_ohlcv_loop(self, exchange, symbol: str, coin: str) -> None:
        await self._poll_ohlcv_once(exchange, symbol, coin)
        while not self._stop.is_set():
            try:
                await asyncio.wait_for(
                    asyncio.shield(asyncio.sleep(self._ohlcv_refresh_s)),
                    timeout=self._ohlcv_refresh_s + 1.0,
                )
            except (asyncio.TimeoutError, asyncio.CancelledError):
                if self._stop.is_set():
                    return
            await self._poll_ohlcv_once(exchange, symbol, coin)

    async def _poll_ohlcv_once(self, exchange, symbol: str, coin: str) -> None:
        try:
            ohlcv = await exchange.fetch_ohlcv(
                symbol, self._interval, limit=50
            )
        except Exception as e:  # noqa: BLE001
            print(f"[rest ohlcv {coin}] {e}", file=sys.stderr)
            return
        bars = tuple(
            OhlcBar(
                timestamp=datetime.fromtimestamp(b[0] / 1000, tz=timezone.utc),
                open=float(b[1]),
                high=float(b[2]),
                low=float(b[3]),
                close=float(b[4]),
                volume=float(b[5]),
            )
            for b in ohlcv
        )
        with self._lock:
            self._caches[coin].bars = bars
            self._caches[coin].bars_fetched_at = time.monotonic()
