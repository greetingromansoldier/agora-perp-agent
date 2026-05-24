from __future__ import annotations

from datetime import datetime, timezone

import pytest

from core.contracts import MarketData, OhlcBar
from core.data import CsvSource, HyperliquidSource, fetch_board

_CSV = """timestamp,open,high,low,close,volume,funding
2026-05-21T00:00:00,100,101,99,100.5,10,0.00001
2026-05-21T00:01:00,100.5,102,100,101.5,12,0.00002
2026-05-21T00:02:00,101.5,103,101,102.0,8,0.00003
"""


def _write_csv(tmp_path, name="BTC.csv"):
    path = tmp_path / name
    path.write_text(_CSV)
    return path


def test_csv_source_builds_market_data(tmp_path):
    src = CsvSource({"BTC": _write_csv(tmp_path)})
    md = src.fetch("BTC", window=10)
    assert isinstance(md, MarketData)
    assert md.asset == "BTC"
    assert len(md.bars) == 3
    assert md.mark_price == 102.0          # last close
    assert md.funding_rate == 0.00003      # last funding row
    assert md.bars[0].timestamp == datetime(2026, 5, 21, 0, 0)


def test_csv_source_respects_window(tmp_path):
    src = CsvSource({"BTC": _write_csv(tmp_path)})
    md = src.fetch("BTC", window=2)
    assert len(md.bars) == 2
    assert md.bars[0].close == 101.5       # oldest kept bar


def test_csv_source_unknown_asset(tmp_path):
    src = CsvSource({"BTC": _write_csv(tmp_path)})
    with pytest.raises(ValueError, match="ETH"):
        src.fetch("ETH", window=10)


def test_fetch_board(tmp_path):
    src = CsvSource(
        {"BTC": _write_csv(tmp_path, "BTC.csv"), "ETH": _write_csv(tmp_path, "ETH.csv")}
    )
    board = fetch_board(src, ["BTC", "ETH"], window=3)
    assert set(board) == {"BTC", "ETH"}
    assert all(isinstance(md, MarketData) for md in board.values())


# --- Hyperliquid parsing (pure, no network) ---

_CANDLES = [
    {"t": 1747785600000, "o": "100", "h": "101", "l": "99", "c": "100.5", "v": "10"},
    {"t": 1747785660000, "o": "100.5", "h": "102", "l": "100", "c": "101.5", "v": "12"},
]

_META_CTX = [
    {"universe": [{"name": "BTC"}, {"name": "ETH"}]},
    [
        {"markPx": "101.5", "funding": "0.0000125"},
        {"markPx": "3200.0", "funding": "-0.00001"},
    ],
]


def test_parse_candles():
    bars = HyperliquidSource._parse_candles(_CANDLES)
    assert len(bars) == 2
    assert bars[0].open == 100.0
    assert bars[1].close == 101.5
    assert bars[0].timestamp.tzinfo == timezone.utc


def test_parse_ctx_maps_by_universe_index():
    mark, funding = HyperliquidSource._parse_ctx(_META_CTX, "ETH")
    assert mark == 3200.0
    assert funding == -0.00001


def test_parse_ctx_unknown_asset():
    with pytest.raises(ValueError, match="DOGE"):
        HyperliquidSource._parse_ctx(_META_CTX, "DOGE")


# --- L2 book parsing and depth ---

_L2 = {
    "coin": "BTC",
    "time": 1747785660000,
    "levels": [
        # bids (highest first): each 0.5 BTC at 99.5 / 99.0 / 98.0
        [{"px": "99.5", "sz": "0.5"}, {"px": "99.0", "sz": "0.5"}, {"px": "98.0", "sz": "0.5"}],
        # asks (lowest first): each 0.5 BTC at 100.5 / 101.0 / 102.0
        [{"px": "100.5", "sz": "0.5"}, {"px": "101.0", "sz": "0.5"}, {"px": "102.0", "sz": "0.5"}],
    ],
}


def test_parse_l2book_returns_typed_levels():
    bids, asks = HyperliquidSource._parse_l2book(_L2)
    assert bids[0] == (99.5, 0.5)
    assert asks[-1] == (102.0, 0.5)
    assert all(isinstance(px, float) and isinstance(sz, float) for px, sz in bids + asks)


def test_parse_l2book_rejects_bad_shape():
    with pytest.raises(ValueError, match="levels"):
        HyperliquidSource._parse_l2book({"coin": "BTC"})
    with pytest.raises(ValueError, match="levels"):
        HyperliquidSource._parse_l2book({"levels": [[{"px": "1", "sz": "1"}]]})


def test_compute_depth_sums_within_band_and_averages_sides():
    # mid = 100; band ±100 bps = ±1.0
    bids, asks = HyperliquidSource._parse_l2book(_L2)
    depth = HyperliquidSource._compute_depth(bids, asks, mid=100.0, band_bps=100.0)
    # within ±1 of mid: bids @ 99.5 (0.5 BTC) and 99.0 (just on the edge: 100-99=1<=1)
    bid_notional = 99.5 * 0.5 + 99.0 * 0.5
    ask_notional = 100.5 * 0.5 + 101.0 * 0.5
    assert depth == pytest.approx(0.5 * (bid_notional + ask_notional), rel=1e-12)


def test_compute_depth_excludes_levels_outside_band():
    bids, asks = HyperliquidSource._parse_l2book(_L2)
    # tight band ±50 bps = ±0.5 of mid 100; only 99.5 and 100.5 qualify
    depth = HyperliquidSource._compute_depth(bids, asks, mid=100.0, band_bps=50.0)
    assert depth == pytest.approx(0.5 * (99.5 * 0.5 + 100.5 * 0.5), rel=1e-12)


def test_compute_depth_returns_zero_for_invalid_mid_or_band():
    bids, asks = HyperliquidSource._parse_l2book(_L2)
    assert HyperliquidSource._compute_depth(bids, asks, mid=0.0, band_bps=50.0) == 0.0
    assert HyperliquidSource._compute_depth(bids, asks, mid=100.0, band_bps=0.0) == 0.0


def test_compute_depth_handles_empty_sides():
    empty: tuple[tuple[float, float], ...] = ()
    asks = ((100.5, 0.5), (101.0, 0.5))
    depth = HyperliquidSource._compute_depth(empty, asks, mid=100.0, band_bps=100.0)
    # bid side contributes 0, ask side contributes (100.5 + 101.0) × 0.5; avg / 2
    assert depth == pytest.approx(0.5 * (100.5 * 0.5 + 101.0 * 0.5), rel=1e-12)


def test_meta_cache_amortises_repeated_calls():
    src = HyperliquidSource(meta_ttl_s=60.0)
    calls = {"n": 0}

    def fake_post(body):
        calls["n"] += 1
        return _META_CTX

    src._post = fake_post  # type: ignore[method-assign]
    src._meta()
    src._meta()
    src._meta()
    assert calls["n"] == 1  # cached after first call


def test_meta_cache_disabled_when_ttl_zero():
    src = HyperliquidSource(meta_ttl_s=0.0)
    calls = {"n": 0}

    def fake_post(body):
        calls["n"] += 1
        return _META_CTX

    src._post = fake_post  # type: ignore[method-assign]
    src._meta()
    src._meta()
    assert calls["n"] == 2  # no caching with ttl=0
