"""Trip's scoreboard strategy — clean port (Mel: Kraken ETH, single TP + single SL).

Faithful to the LOGIC of bx-bot's `rule_based_decision()`, reimplemented without its
accumulated debt. The scoring weights, thresholds and the >=5 entry rule are the
strategy as it actually runs live; do not "improve" them here without a separate
decision, or this stops being the same strategy.

KNOWN WEAKNESS (carried over deliberately, documented in README):
the scoreboard mixes trend-following points (EMA cross, HTF trend) with
mean-reverting ones (RSI, Bollinger %B). They routinely cancel, so entries fire only
when unrelated features coincidentally sum to the threshold. That is the live
behaviour, and any change to it must be a deliberate strategy change.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from indicators import (atr, bollinger, ema, htf_trend, macd_histogram, rsi,
                        volume_trend)


@dataclass
class MarketState:
    price: float
    atr: float | None
    atr_pct: float
    rsi: float | None
    bb_pct_b: float | None
    macd_hist: float | None
    ema_fast: float | None
    ema_slow: float | None
    ema_cross: str
    htf: dict
    volume_trend: str
    indicators: dict = field(default_factory=dict)


@dataclass
class Decision:
    action: str                 # LONG | SHORT | WAIT
    reason: str
    score_long: int = 0
    score_short: int = 0
    confidence: int = 0
    entry: float = 0.0
    sl: float = 0.0
    tp: float = 0.0
    risk_reward: float = 0.0
    regime: str = ""

    @property
    def is_entry(self) -> bool:
        return self.action in ("LONG", "SHORT")


def build_market_state(base: list, htf: list, ind_cfg: dict) -> MarketState:
    """base: 15m OHLCV rows [ts, o, h, l, c, v]; htf: 1h rows. Oldest first."""
    closes = [float(r[4]) for r in base]
    highs = [float(r[2]) for r in base]
    lows = [float(r[3]) for r in base]
    vols = [float(r[5]) for r in base]
    htf_closes = [float(r[4]) for r in htf]

    price = closes[-1] if closes else 0.0
    a = atr(highs, lows, closes, int(ind_cfg["atr_period"]))
    atr_pct = (a / price * 100.0) if (a and price > 0) else 0.0
    r = rsi(closes, int(ind_cfg["rsi_period"]))
    _lo, _mid, _up, pct_b = bollinger(closes, int(ind_cfg["bb_period"]),
                                      float(ind_cfg["bb_std"]))
    mh = macd_histogram(closes, int(ind_cfg["macd_fast"]), int(ind_cfg["macd_slow"]),
                        int(ind_cfg["macd_signal"]))
    ef = ema(closes, int(ind_cfg["ema_fast"]))
    es = ema(closes, int(ind_cfg["ema_slow"]))
    cross = "neutral"
    if ef is not None and es is not None:
        cross = "bullish" if ef > es else "bearish"
    htf_t = htf_trend(htf_closes, int(ind_cfg["htf_ema_fast"]), int(ind_cfg["htf_ema_slow"]))
    vt = volume_trend(vols, int(ind_cfg["volume_lookback"]))

    return MarketState(
        price=price, atr=a, atr_pct=atr_pct, rsi=r, bb_pct_b=pct_b, macd_hist=mh,
        ema_fast=ef, ema_slow=es, ema_cross=cross, htf=htf_t, volume_trend=vt,
        indicators={"atr": a, "atr_pct": atr_pct, "rsi": r, "bb_pct_b": pct_b,
                    "macd_hist": mh, "ema_cross": cross},
    )


def score_from_state(st: MarketState, cfg: dict) -> tuple[int, int]:
    """Return (long_score, short_score). Weights come from config, not literals."""
    sc = cfg["score"]
    lo = sh = 0

    if st.ema_cross == "bullish":
        lo += int(sc["ema_cross"])
    elif st.ema_cross == "bearish":
        sh += int(sc["ema_cross"])

    r = st.rsi
    if r is not None:
        if r < 35:
            lo += int(sc["rsi_extreme"])
        elif r > 65:
            sh += int(sc["rsi_extreme"])
        elif r < 45:
            lo += int(sc["rsi_lean"])
        elif r > 55:
            sh += int(sc["rsi_lean"])

    b = st.bb_pct_b
    if b is not None:
        if b < 15:
            lo += int(sc["bb_extreme"])
        elif b > 85:
            sh += int(sc["bb_extreme"])
        elif b < 35:
            lo += int(sc["bb_lean"])
        elif b > 65:
            sh += int(sc["bb_lean"])

    m = st.macd_hist
    if m is not None:
        if m > 0:
            lo += int(sc["macd_hist"])
        elif m < 0:
            sh += int(sc["macd_hist"])

    d = st.htf.get("direction")
    if d == "up":
        lo += int(sc["htf_trend"])
    elif d == "down":
        sh += int(sc["htf_trend"])

    if st.volume_trend in ("surging", "elevated"):
        lo += int(sc["volume"])
        sh += int(sc["volume"])

    return lo, sh


def _sl_distance(st: MarketState, cfg: dict) -> float:
    """max(ATR% x mult, floor%) as a fraction of price. The FLOOR is load-bearing."""
    ex = cfg["exits"]
    atr_component = st.atr_pct * float(ex["sl_atr_mult"]) / 100.0
    return max(atr_component, float(ex["sl_pct_floor"]))


def decide(st: MarketState, cfg: dict) -> Decision:
    """The scoreboard decision. Prices are derived from `price`, never from a ticker call."""
    lo, sh = score_from_state(st, cfg["strategy"])
    thr = int(cfg["strategy"]["entry_threshold"])
    ex = cfg["exits"]

    if lo > sh and lo >= thr:
        action = "LONG"
    elif sh > lo and sh >= thr:
        action = "SHORT"
    else:
        return Decision(action="WAIT", score_long=lo, score_short=sh,
                        reason=f"no clear signal (long={lo}, short={sh}, need >={thr})")

    conf = min(lo if action == "LONG" else sh, 10)
    price = st.price
    sl_dist = _sl_distance(st, cfg) * price
    tp_dist = float(ex["tp_pct"]) * price

    # ONE take-profit, not a ladder: Mel's exit is a single 4% TP at full size, so the
    # risk/reward the guard checks is the TP distance over the stop distance (4% against
    # the 2.5% floor = 1.6), and no partial level sits in between.
    if action == "LONG":
        sl = price - sl_dist
        tp = price + tp_dist
        regime = "trending_up" if st.htf.get("direction") == "up" else "ranging"
    else:
        sl = price + sl_dist
        tp = price - tp_dist
        regime = "trending_down" if st.htf.get("direction") == "down" else "ranging"

    rr = (tp_dist / sl_dist) if sl_dist > 0 else 0.0
    return Decision(action=action, score_long=lo, score_short=sh, confidence=conf,
                    entry=price, sl=sl, tp=tp, risk_reward=rr, regime=regime,
                    reason=f"{action} score {conf} (long={lo}, short={sh})")
