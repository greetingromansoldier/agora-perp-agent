from __future__ import annotations

from datetime import datetime, timezone

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
    try:
        src.fetch("ETH", window=10)
    except ValueError as exc:
        assert "ETH" in str(exc)
    else:
        raise AssertionError("expected ValueError for unknown asset")


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
    try:
        HyperliquidSource._parse_ctx(_META_CTX, "DOGE")
    except ValueError as exc:
        assert "DOGE" in str(exc)
    else:
        raise AssertionError("expected ValueError for unknown asset")
