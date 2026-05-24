"""market data sources.

one `DataSource` protocol with two implementations: a live Hyperliquid
public feed (no auth) and a CSV replay used for backtests. The engine
consumes the protocol and never knows which one it has, so the same loop
runs live and offline.
"""

from __future__ import annotations

import csv as _csv
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
    source: DataSource, assets: list[str], window: int
) -> dict[str, MarketData]:
    """Fetch a snapshot for every asset in the universe.

    Args:
        source: any `DataSource` implementation.
        assets: market symbols to fetch.
        window: number of recent candles per asset.

    Returns:
        Mapping of asset symbol to its `MarketData` snapshot.
    """
    return {asset: source.fetch(asset, window) for asset in assets}


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

    def __init__(self, interval: str = "1m", timeout_s: float = 10.0) -> None:
        """Configure the candle interval and request timeout.

        Args:
            interval: candle interval; one of the supported keys.
            timeout_s: per-request timeout in seconds.

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

    def fetch(self, asset: str, window: int) -> MarketData:
        """Fetch live candles, mark price, and funding for one asset.

        Args:
            asset: market symbol, e.g. "BTC".
            window: number of most-recent candles to keep.

        Returns:
            A `MarketData` snapshot timestamped at the latest candle.

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

        mark, funding = self._parse_ctx(self._post({"type": "metaAndAssetCtxs"}), asset)
        return MarketData(
            asset=asset,
            timestamp=bars[-1].timestamp,
            mark_price=mark,
            funding_rate=funding,
            bars=bars,
        )

    def _post(self, body: dict) -> object:
        # Lazy import so the module (and CSV path / parsing tests) does not
        # require httpx to be installed.
        import httpx

        resp = httpx.post(HYPERLIQUID_INFO_URL, json=body, timeout=self._timeout_s)
        resp.raise_for_status()
        return resp.json()

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
