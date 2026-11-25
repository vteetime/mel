"""Indicators — clean implementations, no external TA dependency.

Every function takes a list of floats (closes, or highs/lows) oldest-first and returns
the LATEST value (or None when there is not enough history). None-in / None-out rather
than exceptions, because a thin window is a normal condition at startup, not an error.
"""

from __future__ import annotations

import math


def sma(values: list[float], period: int) -> float | None:
    if period <= 0 or len(values) < period:
        return None
    return sum(values[-period:]) / period


def ema_series(values: list[float], period: int) -> list[float]:
    """Full EMA series. Seeded with the SMA of the first `period` values."""
    if period <= 0 or len(values) < period:
        return []
    k = 2.0 / (period + 1.0)
    out = [sum(values[:period]) / period]
    for v in values[period:]:
        out.append(v * k + out[-1] * (1 - k))
    return out


def ema(values: list[float], period: int) -> float | None:
    s = ema_series(values, period)
    return s[-1] if s else None


def rsi(closes: list[float], period: int = 14) -> float | None:
    """Wilder's RSI."""
    if len(closes) < period + 1:
        return None
    gains, losses = [], []
    for i in range(1, period + 1):
        d = closes[i] - closes[i - 1]
        gains.append(max(d, 0.0))
        losses.append(max(-d, 0.0))
    avg_g = sum(gains) / period
    avg_l = sum(losses) / period
    for i in range(period + 1, len(closes)):
        d = closes[i] - closes[i - 1]
        avg_g = (avg_g * (period - 1) + max(d, 0.0)) / period
        avg_l = (avg_l * (period - 1) + max(-d, 0.0)) / period
    if avg_l == 0:
        return 100.0
    rs = avg_g / avg_l
    return 100.0 - (100.0 / (1.0 + rs))


def true_ranges(highs: list[float], lows: list[float], closes: list[float]) -> list[float]:
    out = []
    for i in range(1, len(closes)):
        out.append(max(highs[i] - lows[i],
                       abs(highs[i] - closes[i - 1]),
                       abs(lows[i] - closes[i - 1])))
    return out


def atr(highs: list[float], lows: list[float], closes: list[float],
        period: int = 14) -> float | None:
    """Wilder's ATR (smoothed true range)."""
    tr = true_ranges(highs, lows, closes)
    if len(tr) < period:
        return None
    a = sum(tr[:period]) / period
    for x in tr[period:]:
        a = (a * (period - 1) + x) / period
    return a


def bollinger(closes: list[float], period: int = 20, num_std: float = 2.0):
    """Returns (lower, middle, upper, pct_b). pct_b in 0..100."""
    if len(closes) < period:
        return None, None, None, None
    window = closes[-period:]
    mid = sum(window) / period
    var = sum((c - mid) ** 2 for c in window) / period
    sd = math.sqrt(var)
    upper = mid + num_std * sd
    lower = mid - num_std * sd
    last = closes[-1]
    pct_b = 50.0 if upper == lower else (last - lower) / (upper - lower) * 100.0
    return lower, mid, upper, pct_b


def macd_histogram(closes: list[float], fast: int = 12, slow: int = 26,
                   signal: int = 9) -> float | None:
    """MACD histogram (macd line - signal line)."""
    if len(closes) < slow + signal:
        return None
    ef = ema_series(closes, fast)
    es = ema_series(closes, slow)
    if not ef or not es:
        return None
    # align tails
    n = min(len(ef), len(es))
    macd_line = [a - b for a, b in zip(ef[-n:], es[-n:])]
    if len(macd_line) < signal:
        return None
    sig = ema_series(macd_line, signal)
    if not sig:
        return None
    return macd_line[-1] - sig[-1]


def htf_trend(htf_closes: list[float], fast: int = 20, slow: int = 50) -> dict:
    """Direction + strength of the higher-timeframe trend.

    strength is the EMA separation as a PERCENT of price — the same units bx-bot's
    guard G3 uses, so the thresholds are comparable.
    """
    if len(htf_closes) < slow:
        return {"direction": "unknown", "strength": 0.0}
    ef = ema(htf_closes, fast)
    es = ema(htf_closes, slow)
    if ef is None or es is None or es == 0:
        return {"direction": "unknown", "strength": 0.0}
    strength = abs(ef - es) / es * 100.0
    direction = "up" if ef > es else "down"
    return {"direction": direction, "strength": strength}


def volume_trend(volumes: list[float], lookback: int = 20) -> str:
    """'surging' | 'elevated' | 'normal' | 'quiet' vs the trailing average."""
    if len(volumes) < lookback + 1:
        return "normal"
    avg = sum(volumes[-lookback - 1:-1]) / lookback
    if avg <= 0:
        return "normal"
    ratio = volumes[-1] / avg
    if ratio >= 2.0:
        return "surging"
    if ratio >= 1.3:
        return "elevated"
    if ratio <= 0.5:
        return "quiet"
    return "normal"
