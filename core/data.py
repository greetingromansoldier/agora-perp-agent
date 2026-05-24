"""market data sources.

one `DataSource` protocol with two implementations: a live Hyperliquid
public feed (no auth) and a CSV replay used for backtests. The engine
consumes the protocol and never knows which one it has, so the same loop
runs live and offline.
"""

from __future__ import annotations

import csv as _csv
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from core.contracts import MarketData, OhlcBar

#: Public Hyperliquid info endpoint. No auth, no key, no money.
HYPERLIQUID_INFO_URL = "https://api.hyperliquid.xyz/info"

#: Candle interval -> milliseconds, for sizing the lookback window.
_INTERVAL_MS: dict[str, int] = {
    "1m": 60_000,
    "5m": 300_000,
    "15m": 900_000,
    "1h": 3_600_000,
}


class DataSource(Protocol):
    """Returns a `MarketData` snapshot for one asset."""

    def fetch(self, asset: str, window: int) -> MarketData:
        """Return a market snapshot for one asset.

        Args:
            asset: market symbol, e.g. "BTC".
            window: number of recent candles to include.

        Returns:
            A `MarketData` snapshot.
        """
        ...


def fetch_board(
    source: DataSource,
    assets: list[str],
    window: int,
    max_workers: int = 16,
) -> dict[str, MarketData]:
    """Fetch a snapshot for every asset in the universe, in parallel.

    Per-asset fetches go through a thread pool so wall-time is the slowest
    single fetch, not the sum. Each per-coin failure is **isolated**: a
    rate-limit or timeout on one asset does not poison the whole batch.
    The caller sees the failed asset as a missing key in the result dict.

    Thread safety: `HyperliquidSource` keeps an internal lock around its
    `metaAndAssetCtxs` cache, so concurrent threads serialise the refresh
    on a cache miss instead of stampeding the network.

    Args:
        source: any `DataSource` implementation.
        assets: market symbols to fetch.
        window: number of recent candles per asset.
        max_workers: thread-pool size.

    Returns:
        Mapping of asset symbol to its `MarketData` snapshot. Coins whose
        fetch failed are absent from the result; the caller can detect
        them by checking which requested assets are missing.
    """
    from concurrent.futures import ThreadPoolExecutor

    def _safe_fetch(asset: str) -> tuple[str, MarketData | None]:
        try:
            return asset, source.fetch(asset, window)
        except Exception:  # noqa: BLE001
            return asset, None

    out: dict[str, MarketData] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for asset, md in pool.map(_safe_fetch, assets):
            if md is not None:
                out[asset] = md
    return out


# ---------------------------------------------------------------------------
# CSV source (backtest / offline)
# ---------------------------------------------------------------------------


class CsvSource:
    """Replays candles from local CSV files, one file per asset.

    Expected CSV header: ``timestamp,open,high,low,close,volume`` with an
    optional trailing ``funding`` column. ``timestamp`` is ISO-8601.
    """

    def __init__(self, paths: dict[str, str | Path]) -> None:
        """Store the per-asset CSV paths.

        Args:
            paths: mapping of asset symbol to its CSV file path.
        """
        self._paths = {asset: Path(p) for asset, p in paths.items()}

    def fetch(self, asset: str, window: int) -> MarketData:
        """Read the asset's CSV and return its latest snapshot.

        Args:
            asset: market symbol to read.
            window: number of most-recent candles to keep.

        Returns:
            A `MarketData` snapshot. Mark price is the last close; funding
            is the last non-empty ``funding`` cell (0.0 if absent).

        Raises:
            ValueError: if the asset has no configured path, the file is
                missing, or it contains no rows.
        """
        path = self._paths.get(asset)
        if path is None:
            raise ValueError(f"No CSV path configured for asset `{asset}`.")
        if not path.exists():
            raise ValueError(f"CSV not found for `{asset}`: {path}")

        bars: list[OhlcBar] = []
        funding = 0.0
        with path.open(newline="") as fh:
            for row in _csv.DictReader(fh):
                bars.append(
                    OhlcBar(
                        timestamp=datetime.fromisoformat(row["timestamp"]),
                        open=float(row["open"]),
                        high=float(row["high"]),
                        low=float(row["low"]),
                        close=float(row["close"]),
                        volume=float(row["volume"]),
                    )
                )
                if row.get("funding") not in (None, ""):
                    funding = float(row["funding"])

        if not bars:
            raise ValueError(f"CSV for `{asset}` has no rows: {path}")

        bars = bars[-window:]
        last = bars[-1]
        return MarketData(
            asset=asset,
            timestamp=last.timestamp,
            mark_price=last.close,
            funding_rate=funding,
            bars=tuple(bars),
        )


# ---------------------------------------------------------------------------
# Hyperliquid source (live, public, no auth)
# ---------------------------------------------------------------------------


class HyperliquidSource:
    """Live market data from the public Hyperliquid info endpoint.

    Reads candles plus the current mark price and funding rate. Never
    places orders and never authenticates — it only quotes against the
    public feed so paper PnL stays honest.
    """

    def __init__(
        self,
        interval: str = "1m",
        timeout_s: float = 10.0,
        band_bps: float = 50.0,
        meta_ttl_s: float = 2.0,
    ) -> None:
        """Configure the candle interval, request timeout, and depth band.

        Args:
            interval: candle interval; one of the supported keys.
            timeout_s: per-request timeout in seconds.
            band_bps: half-width of the price band (in bps of mid) used to
                aggregate L2 book notional into ``MarketData.book_depth``.
            meta_ttl_s: how long to reuse the cached ``metaAndAssetCtxs``
                payload across calls. The payload covers the whole HL
                universe, so caching it amortises one POST across an
                N-coin board fetch. Set to ``0`` to disable caching.

        Raises:
            ValueError: if ``interval`` is not supported.
        """
        if interval not in _INTERVAL_MS:
            raise ValueError(
                f"Unsupported interval `{interval}`. "
                f"Supported: {sorted(_INTERVAL_MS)}."
            )
        self._interval = interval
        self._timeout_s = timeout_s
        self._band_bps = band_bps
        self._meta_ttl_s = meta_ttl_s
        self._meta_cache: tuple[float, object] | None = None
        self._meta_lock = threading.Lock()

    def fetch(self, asset: str, window: int) -> MarketData:
        """Fetch live candles, mark, funding, and book depth for one asset.

        Args:
            asset: market symbol, e.g. "BTC".
            window: number of most-recent candles to keep.

        Returns:
            A `MarketData` snapshot timestamped at the latest candle, with
            ``book_depth`` set to the average per-side notional within
            ±``band_bps`` of book mid.

        Raises:
            ValueError: if no candles are returned or the asset is absent
                from the Hyperliquid universe.
        """
        now_ms = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
        span_ms = _INTERVAL_MS[self._interval] * (window + 1)
        candles_raw = self._post(
            {
                "type": "candleSnapshot",
                "req": {
                    "coin": asset,
                    "interval": self._interval,
                    "startTime": now_ms - span_ms,
                    "endTime": now_ms,
                },
            }
        )
        bars = self._parse_candles(candles_raw)[-window:]
        if not bars:
            raise ValueError(f"No candles returned for `{asset}`.")

        mark, funding = self._parse_ctx(self._meta(), asset)

        bids, asks = self._parse_l2book(self._post({"type": "l2Book", "coin": asset}))
        mid = 0.5 * (bids[0][0] + asks[0][0]) if bids and asks else mark
        depth = self._compute_depth(bids, asks, mid, self._band_bps)

        return MarketData(
            asset=asset,
            timestamp=bars[-1].timestamp,
            mark_price=mark,
            funding_rate=funding,
            bars=bars,
            book_depth=depth,
        )

    def _post(self, body: dict) -> object:
        """POST to HL `/info` with exponential backoff on 429 rate-limits.

        We retry up to twice on HTTP 429 with 0.5s and 1.0s sleeps; any
        other error (or persistent 429) propagates. This is enough to
        ride out HL's typical bursty throttling without hiding genuine
        outages.
        """
        # Lazy import so the module (and CSV path / parsing tests) does not
        # require httpx to be installed.
        import httpx

        for attempt in range(3):
            resp = httpx.post(
                HYPERLIQUID_INFO_URL, json=body, timeout=self._timeout_s
            )
            if resp.status_code == 429 and attempt < 2:
                time.sleep(0.5 * (attempt + 1))
                continue
            resp.raise_for_status()
            return resp.json()
        # Unreachable — the loop either returns or raises above.
        raise RuntimeError("unreachable")

    def _meta(self) -> object:
        """Return the latest ``metaAndAssetCtxs`` payload, cached briefly.

        Cached behind a lock so concurrent ``fetch_board`` threads do not
        all hit the network on a cache miss (the payload is ~70 KB and
        each call is ~1 s — a thundering herd of 10 parallel coins would
        cost 10× the necessary load). Double-checked locking: fast path
        when the cache is warm, lock only when refreshing.
        """
        now = time.monotonic()
        cache = self._meta_cache
        if (
            cache is not None
            and self._meta_ttl_s > 0.0
            and now - cache[0] < self._meta_ttl_s
        ):
            return cache[1]
        with self._meta_lock:
            # Re-check; another thread may have refreshed while we waited.
            now = time.monotonic()
            cache = self._meta_cache
            if (
                cache is not None
                and self._meta_ttl_s > 0.0
                and now - cache[0] < self._meta_ttl_s
            ):
                return cache[1]
            payload = self._post({"type": "metaAndAssetCtxs"})
            self._meta_cache = (time.monotonic(), payload)
            return payload

    @staticmethod
    def _parse_candles(raw: object) -> tuple[OhlcBar, ...]:
        """Parse a Hyperliquid candleSnapshot payload into bars."""
        if not isinstance(raw, list):
            raise ValueError("Unexpected candle payload (expected a list).")
        bars = [
            OhlcBar(
                timestamp=datetime.fromtimestamp(c["t"] / 1000, tz=timezone.utc),
                open=float(c["o"]),
                high=float(c["h"]),
                low=float(c["l"]),
                close=float(c["c"]),
                volume=float(c["v"]),
            )
            for c in raw
        ]
        return tuple(bars)

    @staticmethod
    def _parse_ctx(raw: object, asset: str) -> tuple[float, float]:
        """Extract (mark_price, funding) for an asset from metaAndAssetCtxs.

        The payload is ``[meta, ctxs]`` aligned by universe index.
        """
        if not isinstance(raw, list) or len(raw) != 2:
            raise ValueError("Unexpected metaAndAssetCtxs payload.")
        meta, ctxs = raw
        universe = meta["universe"]
        for idx, entry in enumerate(universe):
            if entry["name"] == asset:
                ctx = ctxs[idx]
                return float(ctx["markPx"]), float(ctx["funding"])
        raise ValueError(f"Asset `{asset}` not found in Hyperliquid universe.")

    @staticmethod
    def _parse_l2book(raw: object) -> tuple[
        tuple[tuple[float, float], ...],
        tuple[tuple[float, float], ...],
    ]:
        """Parse a Hyperliquid l2Book payload into (bids, asks) levels.

        Each side is a tuple of ``(price, size)`` pairs. Hyperliquid returns
        ``{"coin": str, "time": int, "levels": [bids, asks]}`` where bids are
        sorted highest-first and asks lowest-first.
        """
        if not isinstance(raw, dict) or "levels" not in raw:
            raise ValueError("Unexpected l2Book payload (missing `levels`).")
        levels = raw["levels"]
        if not isinstance(levels, list) or len(levels) != 2:
            raise ValueError("Unexpected l2Book `levels` shape (need [bids, asks]).")
        bids_raw, asks_raw = levels
        bids = tuple((float(lv["px"]), float(lv["sz"])) for lv in bids_raw)
        asks = tuple((float(lv["px"]), float(lv["sz"])) for lv in asks_raw)
        return bids, asks

    @staticmethod
    def _compute_depth(
        bids: tuple[tuple[float, float], ...],
        asks: tuple[tuple[float, float], ...],
        mid: float,
        band_bps: float,
    ) -> float:
        """Average per-side notional inside ±``band_bps`` of ``mid``.

        Sums ``price × size`` for every level whose distance from ``mid`` is
        within the band, separately on each side, and returns the average.
        Averaging keeps the sqrt-law denominator a single number while
        smoothing slight bid/ask asymmetry; the conservative ``slippage_k``
        absorbs the residual.
        """
        if mid <= 0.0 or band_bps <= 0.0:
            return 0.0
        band = mid * band_bps / 10_000.0
        bid_notional = sum(px * sz for px, sz in bids if mid - px <= band)
        ask_notional = sum(px * sz for px, sz in asks if px - mid <= band)
        return 0.5 * (bid_notional + ask_notional)
