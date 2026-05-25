"""technical indicators over OHLC bars.

Pure functions on tuples of `OhlcBar`. Textbook formulas (see
docs research `07-indicator-reference.md`) — public, no alpha here. The
edge is *how* features combine and calibrate, which lives in the private
forecast model, not in these primitives.
"""

from __future__ import annotations

import math

from core.contracts import OhlcBar

_EPS = 1e-12


def _closes(bars: tuple[OhlcBar, ...]) -> list[float]:
    return [b.close for b in bars]


def sma(bars: tuple[OhlcBar, ...], n: int) -> float:
    """Simple moving average of the last ``n`` closes.

    Raises:
        ValueError: if fewer than ``n`` bars are available.
    """
    closes = _closes(bars)
    if len(closes) < n:
        raise ValueError(f"sma needs >= {n} bars, got {len(closes)}.")
    return sum(closes[-n:]) / n


def ema(bars: tuple[OhlcBar, ...], n: int) -> float:
    """Exponential moving average with ``α = 2 / (n + 1)``.

    Seeded with the SMA of the first ``n`` closes, then folded forward
    over the remaining bars. With exactly ``n`` bars the result equals
    the SMA — accept this degenerate case and warm up with more history
    for a properly-smoothed value.

    Raises:
        ValueError: if fewer than ``n`` bars are available.
    """
    closes = _closes(bars)
    if len(closes) < n:
        raise ValueError(f"ema needs >= {n} bars, got {len(closes)}.")
    alpha = 2.0 / (n + 1.0)
    value = sum(closes[:n]) / n
    for close in closes[n:]:
        value = alpha * close + (1.0 - alpha) * value
    return value


def roc(bars: tuple[OhlcBar, ...], n: int) -> float:
    """Rate of change in percent over ``n`` bars.

    Raises:
        ValueError: if fewer than ``n + 1`` bars are available.
    """
    closes = _closes(bars)
    if len(closes) < n + 1:
        raise ValueError(f"roc needs >= {n + 1} bars, got {len(closes)}.")
    prev = closes[-1 - n]
    if abs(prev) < _EPS:
        return 0.0
    return 100.0 * (closes[-1] - prev) / prev


def realized_vol(bars: tuple[OhlcBar, ...], n: int) -> float:
    """Sample standard deviation of the last ``n`` log returns (per bar).

    Raises:
        ValueError: if fewer than ``n + 1`` bars are available.
    """
    closes = _closes(bars)
    if len(closes) < n + 1:
        raise ValueError(f"realized_vol needs >= {n + 1} bars, got {len(closes)}.")
    rets = [
        math.log(closes[i] / closes[i - 1])
        for i in range(len(closes) - n, len(closes))
    ]
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
    return math.sqrt(var)


def atr(bars: tuple[OhlcBar, ...], n: int) -> float:
    """Average true range (Wilder-smoothed) in price units.

    Raises:
        ValueError: if fewer than ``n + 1`` bars are available.
    """
    if len(bars) < n + 1:
        raise ValueError(f"atr needs >= {n + 1} bars, got {len(bars)}.")
    trs = [
        max(
            bars[i].high - bars[i].low,
            abs(bars[i].high - bars[i - 1].close),
            abs(bars[i].low - bars[i - 1].close),
        )
        for i in range(1, len(bars))
    ]
    value = sum(trs[:n]) / n
    for tr in trs[n:]:
        value = (value * (n - 1) + tr) / n
    return value


def rsi(bars: tuple[OhlcBar, ...], n: int) -> float:
    """Relative strength index (Wilder), range 0..100.

    Raises:
        ValueError: if fewer than ``n + 1`` bars are available.
    """
    closes = _closes(bars)
    if len(closes) < n + 1:
        raise ValueError(f"rsi needs >= {n + 1} bars, got {len(closes)}.")
    gains, losses = [], []
    for i in range(1, len(closes)):
        change = closes[i] - closes[i - 1]
        gains.append(max(change, 0.0))
        losses.append(max(-change, 0.0))
    avg_gain = sum(gains[:n]) / n
    avg_loss = sum(losses[:n]) / n
    for i in range(n, len(gains)):
        avg_gain = (avg_gain * (n - 1) + gains[i]) / n
        avg_loss = (avg_loss * (n - 1) + losses[i]) / n
    if avg_loss < _EPS:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


def adx(bars: tuple[OhlcBar, ...], n: int = 14) -> tuple[float, float, float]:
    """Average directional index with +DI/-DI (Wilder).

    Returns:
        ``(adx, plus_di, minus_di)``; all in 0..100. ADX measures trend
        strength regardless of direction; the DI pair shows up vs down
        pressure.

    Raises:
        ValueError: if fewer than ``2 * n + 1`` bars are available.
    """
    if len(bars) < 2 * n + 1:
        raise ValueError(f"adx needs >= {2 * n + 1} bars, got {len(bars)}.")

    trs, plus_dm, minus_dm = [], [], []
    for i in range(1, len(bars)):
        up = bars[i].high - bars[i - 1].high
        down = bars[i - 1].low - bars[i].low
        plus_dm.append(up if (up > down and up > 0) else 0.0)
        minus_dm.append(down if (down > up and down > 0) else 0.0)
        trs.append(
            max(
                bars[i].high - bars[i].low,
                abs(bars[i].high - bars[i - 1].close),
                abs(bars[i].low - bars[i - 1].close),
            )
        )

    def _wilder(series: list[float]) -> list[float]:
        smoothed = [sum(series[:n])]
        for x in series[n:]:
            smoothed.append(smoothed[-1] - smoothed[-1] / n + x)
        return smoothed

    tr_s, plus_s, minus_s = _wilder(trs), _wilder(plus_dm), _wilder(minus_dm)

    dx_vals: list[float] = []
    plus_di = minus_di = 0.0
    for tr_v, p_v, m_v in zip(tr_s, plus_s, minus_s):
        if tr_v < _EPS:
            continue
        plus_di = 100.0 * p_v / tr_v
        minus_di = 100.0 * m_v / tr_v
        denom = plus_di + minus_di
        dx_vals.append(100.0 * abs(plus_di - minus_di) / denom if denom > _EPS else 0.0)

    adx_value = sum(dx_vals[:n]) / n
    for dx in dx_vals[n:]:
        adx_value = (adx_value * (n - 1) + dx) / n
    return adx_value, plus_di, minus_di
