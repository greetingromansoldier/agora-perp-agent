from __future__ import annotations

from datetime import datetime, timedelta

from core.contracts import Forecast, MarketData, OhlcBar
from core.forecast import BaselineForecast

_T0 = datetime(2026, 5, 21)


def _market(closes, asset="BTC"):
    bars = tuple(
        OhlcBar(
            timestamp=_T0 + timedelta(minutes=i),
            open=c,
            high=c + 1,
            low=c - 1,
            close=c,
            volume=1.0,
        )
        for i, c in enumerate(closes)
    )
    return MarketData(
        asset=asset,
        timestamp=bars[-1].timestamp,
        mark_price=bars[-1].close,
        funding_rate=0.0,
        bars=bars,
    )


def test_baseline_returns_valid_forecast():
    fc = BaselineForecast().forecast(_market([float(i) for i in range(1, 60)]))
    assert isinstance(fc, Forecast)
    assert 0.0 <= fc.p_up <= 1.0
    assert 0.0 <= fc.p_down <= 1.0
    assert 0.0 <= fc.confidence <= 1.0
    assert fc.p_up + fc.p_down == 1.0
    assert fc.expected_move_bps == fc.expected_move_bps  # finite (not NaN)


def test_uptrend_is_bullish():
    fc = BaselineForecast().forecast(_market([float(i) for i in range(1, 60)]))
    assert fc.p_up > 0.5
    assert fc.expected_move_bps > 0.0


def test_downtrend_is_bearish():
    fc = BaselineForecast().forecast(_market([float(i) for i in range(60, 1, -1)]))
    assert fc.p_up < 0.5
    assert fc.expected_move_bps < 0.0
