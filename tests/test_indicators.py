from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from core.contracts import OhlcBar
from core.indicators import adx, atr, realized_vol, roc, rsi, sma

_T0 = datetime(2026, 5, 21)


def _bar(close, high=None, low=None, i=0):
    return OhlcBar(
        timestamp=_T0 + timedelta(minutes=i),
        open=close,
        high=high if high is not None else close + 1,
        low=low if low is not None else close - 1,
        close=close,
        volume=1.0,
    )


def _series(closes):
    return tuple(_bar(c, i=i) for i, c in enumerate(closes))


def test_sma_exact():
    bars = _series([1, 2, 3, 4, 5])
    assert sma(bars, 5) == 3.0
    assert sma(bars, 2) == 4.5


def test_sma_insufficient():
    with pytest.raises(ValueError, match="sma needs"):
        sma(_series([1, 2]), 5)


def test_roc_exact():
    bars = _series([100, 110, 121])
    assert roc(bars, 2) == pytest.approx(21.0)  # (121-100)/100 * 100


def test_realized_vol_constant_returns_is_zero():
    # constant +1% step → equal log returns → zero dispersion
    closes = [100 * (1.01**i) for i in range(10)]
    assert realized_vol(_series(closes), 5) == pytest.approx(0.0, abs=1e-9)


def test_atr_flat_range():
    # every bar has high-low = 2, no gaps → ATR converges to 2
    bars = _series([100] * 20)
    assert atr(bars, 14) == pytest.approx(2.0, abs=1e-6)


def test_rsi_all_gains_is_100():
    bars = _series([float(i) for i in range(1, 20)])  # strictly rising
    assert rsi(bars, 14) == pytest.approx(100.0)


def test_rsi_in_range():
    bars = _series([1, 2, 1.5, 2.5, 2, 3, 2.5, 3.5, 3, 4, 3.5, 4.5, 4, 5, 4.5, 5.5])
    val = rsi(bars, 14)
    assert 0.0 <= val <= 100.0


def test_adx_uptrend_properties():
    bars = _series([float(i) for i in range(1, 40)])  # clean uptrend
    adx_val, plus_di, minus_di = adx(bars, 14)
    assert 0.0 <= adx_val <= 100.0
    assert plus_di > minus_di  # up pressure dominant


def test_adx_insufficient():
    with pytest.raises(ValueError, match="adx needs"):
        adx(_series([1, 2, 3]), 14)
