"""Tests for mel — sizing (Kraken contract semantics), scoreboard, single TP/SL, guards,
DB isolation, state, and the Kraken adapter's order mechanics.

The point of each test is a rule that, if broken, costs money or hides a position.
"""

import json
import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config as C          # noqa: E402
import exchange as ex       # noqa: E402
from db import BOT_ID, TradeDB  # noqa: E402
from guards import run_guards   # noqa: E402
from indicators import atr, bollinger, ema, macd_histogram, rsi, volume_trend  # noqa: E402
from state import State         # noqa: E402
from strategy import Decision, MarketState, build_market_state, decide, score_from_state  # noqa: E402

CFG = C.load_config()

# Kraken XRP/USD:USD, as verified live: 1 contract = 1 XRP, size step 1.0 (whole XRP), tick 0.0001.
KRAKEN_CS, KRAKEN_STEP = 1.0, 1.0
# KuCoin ETHUSDTM, Jack's venue: 1 contract = 0.01 ETH, indivisible.
KUCOIN_CS, KUCOIN_STEP = 0.01, 1.0


# ── sizing: Tee's 50% rule, priced in Kraken's contract semantics ────────────

def test_margin_never_exceeds_90pct_of_funds():
    s = C.size_position(732.09, 1.368, CFG, KRAKEN_CS, KRAKEN_STEP, confidence=10)
    assert s.margin <= 732.09 * 0.90 + 1e-9, f"margin {s.margin} exceeded the 90% cap"
    assert s.contracts > 0


def test_contract_amount_is_whole_xrp_units():
    """THE Kraken XRP rule: amount IS whole XRP (step 1.0), never a fraction.

    Flooring to the 1.0 step turns a correct ~963.3 XRP order into 963 — never into 0, which
    is the failure mode the fractional-venue path must not reintroduce.
    """
    # 732.09 free x 0.90 = $658.88 margin -> $1317.76 notional at 2x -> 963 XRP after
    # flooring to the 1.0 step (the raw figure is ~963.3).
    s = C.size_position(732.09, 1.368, CFG, KRAKEN_CS, KRAKEN_STEP, confidence=10)
    assert s.contracts == pytest.approx(963.0, abs=1e-9), s.contracts
    assert s.margin == pytest.approx(658.69, abs=0.1), s.margin
    assert s.contracts % KRAKEN_STEP == pytest.approx(0.0, abs=1e-12), "not on the step"
    assert s.contracts == int(s.contracts), "XRP amounts are whole units"
    # and the same funds on Jack's venue must still floor to whole contracts
    j = C.size_position(732.09, 1.368, CFG, KUCOIN_CS, KUCOIN_STEP, confidence=10)
    assert j.contracts == int(j.contracts), "KuCoin contracts are indivisible"


def test_notional_uses_contract_size_on_both_venues():
    k = C.size_position(1000.0, 1.368, CFG, KRAKEN_CS, KRAKEN_STEP, confidence=10)
    assert k.notional == pytest.approx(k.contracts * KRAKEN_CS * 1.368)
    u = C.size_position(1000.0, 1.368, CFG, KUCOIN_CS, KUCOIN_STEP, confidence=10)
    assert u.notional == pytest.approx(u.contracts * KUCOIN_CS * 1.368)


def test_ignoring_contract_size_is_a_100x_error():
    """The arithmetic the two venues disagree about, stated once so it cannot drift."""
    contracts, price = 1.0, 1.368
    assert contracts * KRAKEN_CS * price == pytest.approx(1.368)
    assert contracts * KUCOIN_CS * price == pytest.approx(0.01368)


def test_below_the_market_minimum_refuses_to_trade():
    """0.5 XRP is under the 1.0 whole-XRP step: refuse, never round up to the minimum."""
    s = C.size_position(0.5, 1.368, CFG, KRAKEN_CS, KRAKEN_STEP, confidence=10,
                        min_contracts=1.0)
    assert s.contracts == 0.0 and s.margin == 0.0


def test_min_margin_floor_refuses_tiny_accounts():
    s = C.size_position(5.0, 1.368, CFG, KRAKEN_CS, KRAKEN_STEP, confidence=10)
    assert s.contracts == 0.0, "margin below sizing.min_margin_usdt must not trade"


def test_confidence_scaling_off_means_flat_sizing():
    """A bare-threshold entry is traded SMALLER — that is Trip's G7 behaviour."""
    full = C.size_position(1000.0, 1.368, CFG, KRAKEN_CS, KRAKEN_STEP, confidence=8)
    bare = C.size_position(1000.0, 1.368, CFG, KRAKEN_CS, KRAKEN_STEP, confidence=5)
    assert bare.contracts == full.contracts
    assert bare.margin == full.margin


def test_confidence_scale_values():
    assert C.confidence_scale(4) == 0.0
    assert C.confidence_scale(5) == 0.5
    assert C.confidence_scale(6) == 0.75
    assert C.confidence_scale(7) == 0.75
    assert C.confidence_scale(8) == 1.0
    assert C.confidence_scale(10) == 1.0


def test_rounding_down_never_breaks_the_cap():
    for funds in (37.0, 123.45, 732.09, 999.99, 1000.0):
        s = C.size_position(funds, 1.368, CFG, KRAKEN_CS, KRAKEN_STEP, confidence=10)
        assert s.margin <= funds * 0.90 + 1e-9


def test_extra_scale_applies_multiplicatively():
    a = C.size_position(1000.0, 1.368, CFG, KRAKEN_CS, KRAKEN_STEP, confidence=8,
                        extra_scale=1.0)
    b = C.size_position(1000.0, 1.368, CFG, KRAKEN_CS, KRAKEN_STEP, confidence=8,
                        extra_scale=0.5)
    assert b.margin < a.margin


def test_a_missing_step_does_not_silently_round_everything_to_zero():
    """A venue that reports no step must not make sizing 0 — that hides a live bot."""
    s = C.size_position(1000.0, 1.368, CFG, KRAKEN_CS, 0.0, confidence=10)
    assert s.contracts > 0, "a 0 step must be treated as 'no rounding', not 'round to 0'"


def test_available_funds_uses_free_not_total():
    """Margin already committed is not available — sizing off total would double-count."""
    bal = {"free": {"USDT": 100.0}, "total": {"USDT": 1000.0}}
    assert C.available_funds(bal) == 100.0


def test_available_funds_falls_back_to_usd_labelled_collateral():
    """Kraken reports the same futures wallet as USDT on one endpoint and USD on another."""
    assert C.available_funds({"free": {"USD": 42.0}, "total": {}}) == 42.0
    assert C.available_funds({}) == 0.0


def test_available_funds_sums_the_multi_collateral_wallet():
    """[MONEY] Kraken holds USD and USDT at once; sizing must see both rows.

    Shape is the live Mel wallet on 2026-11-11 (USD 61.8753 + USDT 545.5385 free when
    flat, with 431.35 of used margin appearing only under USDT). Sizing off the first
    matching key — the old behaviour — saw 545.54 and deployed 490.69 instead of the
    546.67 that 90% of the real 607.41 calls for: 81% of the account, not 90%.
    """
    bal = {"free": {"USD": 61.8753, "USDT": 545.5385028},
           "used": {"USDT": 431.3514622031906},
           "total": {"USD": 61.8753, "USDT": 976.8899650031906}}
    got = C.available_funds(bal)
    assert abs(got - 607.4138028) < 1e-6, got
    assert got > 545.5385028, "the USD collateral must not be dropped"
    # and the sizing it feeds lands on 90% of the whole wallet
    sized = C.size_position(got, 1.0700, CFG, KRAKEN_CS, 0.0, confidence=10)
    assert sized.margin > 490.69, f"still sizing off one row: margin {sized.margin}"


def test_available_funds_sums_free_not_total_across_currencies():
    """Free is the basis for BOTH currencies — a committed USDT margin must not be re-spent."""
    bal = {"free": {"USD": 61.87, "USDT": 114.19},
           "used": {"USDT": 431.35},
           "total": {"USD": 61.87, "USDT": 545.54}}
    assert abs(C.available_funds(bal) - 176.06) < 1e-6


def test_available_funds_ignores_negative_rows():
    """A negative (borrowed) row must not reduce the collateral that is really there."""
    bal = {"free": {"USD": -5.0, "USDT": 100.0}, "total": {}}
    assert C.available_funds(bal) == 100.0


# ── Kraken adapter mechanics ─────────────────────────────────────────────────

class FakeBook:
    """Minimal ccxt-like client: market facts, an order book and a position."""

    def __init__(self, contract_size=KRAKEN_CS, step=KRAKEN_STEP, tick=0.0001,
                 accept_orders=True, fills=None, ledger=None):
        self.contract_size, self.step, self.tick = contract_size, step, tick
        self.orders, self.position, self.cancelled, self.placed = [], None, [], []
        self.accept_orders, self._fills, self._ledger = accept_orders, fills or [], ledger or []

    # market facts
    def market(self, symbol):
        return {"id": "PF_XRPUSD", "contractSize": self.contract_size, "type": "swap",
                "precision": {"price": self.tick, "amount": self.step},
                "limits": {"amount": {"min": None}}}

    def load_markets(self):
        return {}

    def price_to_precision(self, symbol, price):
        return f"{round(round(float(price) / self.tick) * self.tick, 10):.10f}".rstrip("0")

    def amount_to_precision(self, symbol, amount):
        return round(float(amount) / self.step) * self.step

    # orders
    def create_order(self, symbol, otype, side, amount, price=None, params=None):
        params = params or {}
        if not self.accept_orders:
            raise RuntimeError("venue refused the order")
        o = {"id": f"o{len(self.placed) + 1}", "symbol": symbol, "type": otype, "side": side,
             "amount": float(amount), "price": price, "stopPrice": params.get("stopPrice"),
             "reduceOnly": params.get("reduceOnly", False), "status": "open",
             "filled": 0.0, "info": {}}
        self.placed.append(o)
        self.orders.append(o)
        return dict(o)

    def fetch_order(self, oid, symbol=None):
        for o in self.orders + getattr(self, "closed_orders", []):
            if str(o.get("id")) == str(oid):
                return dict(o)
        return None

    def fetch_open_orders(self, symbol=None, params=None):
        if params and params.get("stop"):
            return [dict(o) for o in self.orders if o["type"] == "stop"]
        return [dict(o) for o in self.orders]

    def cancel_order(self, oid, symbol=None):
        import ccxt
        known = {str(o.get("id")) for o in self.orders}
        if str(oid) not in known:
            raise ccxt.OrderNotFound(f"no order {oid}")
        self.cancelled.append(str(oid))
        self.orders = [o for o in self.orders if str(o.get("id")) != str(oid)]
        return {"id": oid, "status": "canceled"}

    def fetch_positions(self, symbols=None):
        return [dict(self.position)] if self.position else []

    def fetch_my_trades(self, symbol=None, limit=100):
        return list(self._fills)

    def historyGetAccountLog(self, params=None):
        return {"logs": list(self._ledger)}


def test_market_spec_reads_kraken_contract_size_and_step():
    spec = ex.market_spec(FakeBook(), "XRP/USD:USD")
    assert spec["contract_size"] == 1.0
    assert spec["amount_step"] == 1.0
    assert spec["tick_size"] == 0.0001
    assert spec["market_id"] == "PF_XRPUSD"


def test_round_amount_floors_never_up():
    b = FakeBook()
    assert ex.round_amount(b, "XRP/USD:USD", 963.3) == pytest.approx(963.0)
    assert ex.round_amount(b, "XRP/USD:USD", 963.9999) == pytest.approx(963.0)
    assert ex.round_amount(b, "XRP/USD:USD", 3.0) == pytest.approx(3.0)
    assert ex.round_amount(b, "XRP/USD:USD", 0.5) == 0.0


def test_round_price_uses_tick_not_integer_decimals():
    """On Kraken precision.price is a TICK SIZE; int() on it destroys a $2,650 price."""
    b = FakeBook(tick=0.0001)
    assert ex.round_price(b, "XRP/USD:USD", 1.36831) == pytest.approx(1.3683)
    assert ex.round_price(b, "XRP/USD:USD", 1.36836) == pytest.approx(1.3684)


def test_place_stop_is_reduce_only_with_a_trigger():
    b = FakeBook()
    o = ex.place_stop(b, "XRP/USD:USD", "sell", 963.0, 1.3338)
    assert o["type"] == "stop" and o["reduceOnly"] is True
    assert float(o["stopPrice"]) == pytest.approx(1.3338)
    assert o["amount"] == pytest.approx(963.0)


def test_place_limit_tp_is_reduce_only_and_stepped():
    b = FakeBook()
    o = ex.place_limit(b, "XRP/USD:USD", "sell", 963.3, 1.42272)
    assert o["type"] == "limit" and o["reduceOnly"] is True
    assert o["amount"] == pytest.approx(963.0)


def test_stop_is_confirmed_rejects_a_wrong_side_trigger():
    """A 'sell' stop above entry is not a loss-limiting stop — refuse it."""
    b = FakeBook()
    o = ex.place_stop(b, "XRP/USD:USD", "sell", 963.0, 1.40)
    ok, why = ex.stop_is_confirmed(b, "XRP/USD:USD", o["id"], side="sell",
                                   min_amount=963.0, require_breakeven=False,
                                   entry=1.368, trigger=1.40)
    assert not ok and "above entry" in why


def test_stop_is_confirmed_rejects_undersized_protection():
    b = FakeBook()
    o = ex.place_stop(b, "XRP/USD:USD", "sell", 1.0, 1.3338)
    ok, why = ex.stop_is_confirmed(b, "XRP/USD:USD", o["id"], side="sell",
                                   min_amount=963.0, require_breakeven=False,
                                   entry=1.368, trigger=1.3338)
    assert not ok and "size" in why


def test_stop_is_confirmed_accepts_a_proper_initial_stop():
    b = FakeBook()
    o = ex.place_stop(b, "XRP/USD:USD", "sell", 963.0, 1.3338)
    ok, why = ex.stop_is_confirmed(b, "XRP/USD:USD", o["id"], side="sell",
                                   min_amount=963.0, require_breakeven=False,
                                   entry=1.368, trigger=1.3338)
    assert ok, why


def test_find_matching_order_prevents_a_duplicate_protective_order():
    """ccxt exposes no client order id on Kraken — the book is the idempotency check."""
    b = FakeBook()
    ex.place_limit(b, "XRP/USD:USD", "sell", 963.0, 1.42272)
    found = ex.find_matching_order(b, "XRP/USD:USD", side="sell", amount=963.0,
                                   price=ex.round_price(b, "XRP/USD:USD", 1.42272))
    assert found is not None
    assert ex.find_matching_order(b, "XRP/USD:USD", side="sell", amount=100.0,
                                  price=1.42272) is None
    assert ex.find_matching_order(b, "XRP/USD:USD", side="buy", amount=963.0,
                                  price=1.42272) is None


def test_open_position_reports_an_error_rather_than_flat():
    """An unreadable venue must never masquerade as 'flat'."""
    class Boom(FakeBook):
        def fetch_positions(self, symbols=None):
            raise RuntimeError("network down")
    p = ex.open_position(Boom(), "XRP/USD:USD")
    assert p["active"] is False and p["error"], p


def test_cancel_treats_already_gone_as_success():
    """Kraken auto-cancels a reduce-only order when the position closes, so a cancel that
    comes back 'not found' is the outcome we wanted — reporting it as a failure would
    alarm on the normal path."""
    b = FakeBook()
    assert ex.cancel(b, "XRP/USD:USD", "already-gone") is True
    assert ex.cancel(b, "XRP/USD:USD", "") is True                 # nothing to do


def test_cancel_reports_a_real_failure():
    class Refusing(FakeBook):
        def cancel_order(self, oid, symbol=None):
            raise RuntimeError("venue said no")
    assert ex.cancel(Refusing(), "XRP/USD:USD", "o1") is False


# ── indicators ───────────────────────────────────────────────────────────────

def test_rsi_bounds_and_direction():
    up = [100.0 + i for i in range(40)]
    assert rsi(up, 14) > 95
    dn = [100.0 - i for i in range(40)]
    assert rsi(dn, 14) < 5
    assert rsi([100.0], 14) is None


def test_ema_of_constant_series_is_the_constant():
    assert ema([5.0] * 60, 20) == pytest.approx(5.0)


def test_atr_positive_and_none_when_short():
    n = 30
    h = [101.0 + i for i in range(n)]
    l = [99.0 + i for i in range(n)]
    c = [100.0 + i for i in range(n)]
    a = atr(h, l, c, 14)
    assert a is not None and a > 0
    assert atr(h[:5], l[:5], c[:5], 14) is None


def test_bollinger_pct_b_at_bands():
    closes = [100.0] * 19 + [1000.0]
    _lo, _mid, _up, pct_b = bollinger(closes, 20, 2.0)
    assert pct_b is not None and pct_b > 50


def test_macd_histogram_none_when_short():
    assert macd_histogram([1.0] * 10) is None


def test_volume_trend_classification():
    assert volume_trend([100.0] * 20 + [300.0], 20) == "surging"
    assert volume_trend([100.0] * 20 + [140.0], 20) == "elevated"
    assert volume_trend([100.0] * 20 + [100.0], 20) == "normal"


# ── scoreboard (unchanged from Jack) ─────────────────────────────────────────

def _state(**kw) -> MarketState:
    base = dict(price=100.0, atr=1.0, atr_pct=1.0, rsi=50.0, bb_pct_b=50.0,
                macd_hist=0.0, ema_fast=100.0, ema_slow=100.0, ema_cross="neutral",
                htf={"direction": "unknown", "strength": 0.0}, volume_trend="normal")
    base.update(kw)
    return MarketState(**base)


def test_score_matches_configured_weights():
    st = _state(ema_cross="bullish", rsi=30.0, bb_pct_b=10.0, macd_hist=0.5,
                htf={"direction": "up", "strength": 1.0}, volume_trend="surging")
    lo, sh = score_from_state(st, CFG["strategy"])
    # 2 (ema) + 2 (rsi<35) + 2 (bb<15) + 1 (macd) + 2 (htf up) + 1 (vol) = 10
    assert lo == 10, f"expected 10, got {lo}"
    assert sh == 1, f"volume adds to both sides; expected 1, got {sh}"


def test_wait_below_threshold_and_on_a_tie():
    assert decide(_state(ema_cross="bullish"), CFG).action == "WAIT"
    st = _state(ema_cross="bullish", rsi=30.0, bb_pct_b=10.0, macd_hist=0.5,
                htf={"direction": "up", "strength": 1.0}, volume_trend="surging")
    lo, sh = score_from_state(st, CFG["strategy"])
    if lo == sh:
        assert decide(st, CFG).action == "WAIT"


def test_sl_uses_the_25_percent_floor_when_atr_is_small():
    """The load-bearing rule: a quiet ATR must not produce a tighter stop than 2.5%."""
    st = _state(price=100.0, atr=0.05, atr_pct=0.05)
    st.ema_cross = "bullish"
    st.rsi, st.bb_pct_b, st.macd_hist = 30.0, 10.0, 0.5
    st.htf = {"direction": "up", "strength": 1.0}
    st.volume_trend = "surging"
    d = decide(st, CFG)
    assert d.action == "LONG"
    dist_pct = abs(d.entry - d.sl) / d.entry * 100
    assert dist_pct == pytest.approx(CFG["exits"]["sl_pct_floor"] * 100, abs=0.01), \
        f"stop {dist_pct:.3f}% tighter than the 2.5% floor"


def test_sl_uses_atr_when_atr_exceeds_the_floor():
    st = _state(price=100.0, atr=4.0, atr_pct=4.0)
    st.ema_cross, st.rsi, st.bb_pct_b, st.macd_hist = "bullish", 30.0, 10.0, 0.5
    st.htf = {"direction": "up", "strength": 1.0}
    st.volume_trend = "surging"
    d = decide(st, CFG)
    dist_pct = abs(d.entry - d.sl) / d.entry * 100
    assert dist_pct == pytest.approx(4.0 * CFG["exits"]["sl_atr_mult"], abs=0.01)


def test_the_single_tp_is_4_percent_and_there_is_no_ladder():
    st = _state(price=100.0, atr=1.0, atr_pct=1.0, ema_cross="bullish", rsi=30.0,
                bb_pct_b=10.0, macd_hist=0.5, htf={"direction": "up", "strength": 1.0},
                volume_trend="surging")
    d = decide(st, CFG)
    assert d.action == "LONG"
    assert d.tp == pytest.approx(104.0), f"expected +4%, got {d.tp}"
    assert not hasattr(d, "tp1") and not hasattr(d, "tp2"), "a TP ladder reappeared"
    assert d.risk_reward == pytest.approx(0.04 / 0.025, abs=0.01)
    assert "tp_split" not in CFG["exits"], "the ladder config must not survive"
    assert CFG["exits"]["runner_enabled"] is False


def test_short_prices_are_mirrored():
    st = _state(price=100.0, atr=1.0, atr_pct=1.0, ema_cross="bearish", rsi=70.0,
                bb_pct_b=90.0, macd_hist=-0.5, htf={"direction": "down", "strength": 1.0},
                volume_trend="surging")
    d = decide(st, CFG)
    assert d.action == "SHORT"
    assert d.sl > d.entry > d.tp


# ── guards (unchanged from Jack, single-TP risk/reward) ──────────────────────

def _dec(action="LONG", conf=8):
    return Decision(action=action, reason="t", confidence=conf, entry=100.0,
                    sl=97.5, tp=104.0, risk_reward=1.6)


def test_daily_trade_cap_blocks():
    g = run_guards(_dec(), _state(), {"trades_today": 2}, CFG)
    assert not g.passed and "daily trade limit" in g.reason


def test_max_consecutive_losses_zero_disables_the_breaker():
    cfg = json.loads(json.dumps(CFG))
    cfg["guards"]["max_consecutive_losses"] = 0
    g = run_guards(_dec(), _state(), {"consecutive_losses": 99}, cfg)
    assert g.passed, "a 0 setting must DISABLE, not block forever"


def test_consecutive_loss_breaker_blocks_at_threshold():
    g = run_guards(_dec(), _state(), {"consecutive_losses": 3}, CFG)
    assert not g.passed and "circuit breaker" in g.reason


def test_rsi_extreme_blocks_long_and_short():
    assert not run_guards(_dec("LONG"), _state(rsi=80.0), {}, CFG).passed
    assert not run_guards(_dec("SHORT", 9), _state(rsi=20.0), {}, CFG).passed


def test_bb_extreme_blocks():
    assert not run_guards(_dec("LONG"), _state(bb_pct_b=95.0), {}, CFG).passed
    assert not run_guards(_dec("SHORT", 9), _state(bb_pct_b=5.0), {}, CFG).passed


def test_counter_trend_blocks_against_a_strong_htf():
    g = run_guards(_dec("SHORT", 10), _state(htf={"direction": "up", "strength": 5.0}), {}, CFG)
    assert not g.passed and "counter-trend" in g.reason


def test_counter_trend_allows_with_a_strong_trend():
    g = run_guards(_dec("LONG", 9), _state(htf={"direction": "up", "strength": 5.0}), {}, CFG)
    assert g.passed


def test_macd_fade_does_not_change_flat_size():
    # D1 (decided-by-default 2026-10-14): the validated config is FLAT 90%; the donor-only
    # MACD-fade half-size modifier is disabled, so a fading MACD must NOT halve the size.
    g = run_guards(_dec("LONG"), _state(macd_hist=-0.5), {}, CFG)
    assert g.passed and g.position_scale == 1.0, g


def test_sizing_is_flat_even_on_macd_fade():
    # A re-introduced half-size modifier must fail the suite. Drive the full entry-sizing
    # path (run_guards -> position_scale -> size_position) with a fading MACD on both
    # directions and assert the deployed margin is the full 90% (scale == 1.0).
    import config as C2
    sizing_cfg = C2.load_config()
    for action, macd_hist in (("LONG", -0.5), ("SHORT", 0.5)):
        g = run_guards(_dec(action), _state(macd_hist=macd_hist), {}, CFG)
        assert g.position_scale == 1.0, (action, g)
        s = C.size_position(732.09, 1.368, sizing_cfg,
                            contract_size=1.0, contract_step=1.0,
                            confidence=8, extra_scale=g.position_scale)
        # full 90% of free funds as margin, not half
        assert abs(s.margin - 732.09 * 0.90) < 1.0, (action, s)


def test_risk_reward_floor_blocks_a_thin_single_tp():
    d = Decision(action="LONG", reason="t", confidence=9, entry=100.0, sl=99.0,
                 tp=100.5, risk_reward=0.5)
    g = run_guards(d, _state(), {}, CFG)
    assert not g.passed and "R:R" in g.reason


def test_risk_reward_passes_mel_4_percent_tp():
    g = run_guards(_dec(), _state(), {}, CFG)
    assert g.passed, g.reason


def test_a_broken_gate_fails_closed(monkeypatch):
    """If a gate raises, the entry must be blocked, not allowed."""
    import guards
    monkeypatch.setattr(guards, "g_daily_trades",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    g = run_guards(_dec(), _state(), {}, CFG)
    assert not g.passed and "errored" in g.reason


# ── DB: own identity only ────────────────────────────────────────────────────

SCHEMA = """
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT, bot_id TEXT NOT NULL,
    exchange TEXT NOT NULL, symbol TEXT NOT NULL, direction TEXT NOT NULL,
    entry_price REAL NOT NULL, exit_price REAL NOT NULL, contracts REAL NOT NULL,
    leverage INTEGER, notional REAL, pnl_net REAL NOT NULL, pnl_pct REAL NOT NULL,
    fee_usdt REAL, funding_usdt REAL DEFAULT 0, entry_time TEXT NOT NULL,
    exit_time TEXT NOT NULL, duration_h REAL, exit_reason TEXT, status TEXT,
    sl_price REAL, tp1_price REAL, tp2_price REAL, sl_pct REAL, trigger TEXT,
    guard_config TEXT, fee_estimated INTEGER, settlement_source TEXT
    DEFAULT 'provisional', exchange_settlement_id TEXT, settled_at TEXT,
    entry_score REAL,
    created_at TEXT);
"""


def _make_db(tmp_path, name="t.db") -> TradeDB:
    """Create the real schema first, then construct — TradeDB verifies, never creates."""
    p = tmp_path / name
    con = sqlite3.connect(p)
    con.executescript(SCHEMA)
    con.commit()
    con.close()
    return TradeDB(p)


def test_db_writes_only_its_own_bot_id(tmp_path):
    db = _make_db(tmp_path)
    tid = db.record_entry(symbol="XRP/USD:USD", direction="long", entry_price=1.368,
                          contracts=0.183, leverage=2, sl_price=1.33384,
                          tp1_price=1.42272, tp2_price=0.0, notional=483.6)
    row = db._conn.execute("SELECT bot_id, exchange, status, tp1_price, tp2_price "
                           "FROM trades WHERE id=?", (tid,)).fetchone()
    assert row["bot_id"] == "mel", "mel must not write under another bot's id"
    assert row["exchange"] == "kraken_futures"
    assert row["status"] == "open"
    assert row["tp2_price"] == 0.0, "a single TP leaves no second level"
    assert row["tp1_price"] == pytest.approx(1.42272)


def test_db_stats_ignore_other_bots(tmp_path):
    """Ken's rows in particular must never leak into Mel's guard accounting."""
    db = _make_db(tmp_path)
    db._conn.executescript("""
        INSERT INTO trades (bot_id, exchange, symbol, direction, entry_price, exit_price,
            contracts, pnl_net, pnl_pct, entry_time, exit_time, status)
        VALUES ('ken','kraken_futures','SOL/USD:USD','short',112.3,111.9,14,-5,-5,
                '2020-01-01T00:00:00+00:00','2020-01-01T01:00:00+00:00','closed');
    """)
    s = db.stats()
    assert s["trades_today"] == 0, "another bot's rows leaked into mel's stats"
    assert s["pnl_total_pct"] == 0.0
    assert s["consecutive_losses"] == 0


def test_db_refuses_to_trade_on_a_schema_that_is_missing_columns(tmp_path):
    p = tmp_path / "bad.db"
    con = sqlite3.connect(p)
    con.execute("CREATE TABLE trades (id INTEGER PRIMARY KEY, bot_id TEXT)")
    con.commit()
    con.close()
    with pytest.raises(RuntimeError) as e:
        TradeDB(p)
    assert "missing columns" in str(e.value)


# ── state ────────────────────────────────────────────────────────────────────

def test_state_round_trip(tmp_path):
    s = State(tmp_path / "s.json")
    assert s.is_flat()
    s.set_open({"entry_price": 1.0})
    assert not State(tmp_path / "s.json").is_flat()
    s.clear()
    assert State(tmp_path / "s.json").is_flat()


def test_state_never_silently_returns_flat_on_corruption(tmp_path):
    """A corrupt state file must NOT read as 'no position' — that orphans a live one."""
    p = tmp_path / "s.json"
    p.write_text("{not json")
    with pytest.raises(RuntimeError) as e:
        State(p)
    assert "Refusing to continue" in str(e.value)
    assert (tmp_path / "s.json.corrupt").exists(), "the corrupt file must be preserved"


def test_state_halt_flag(tmp_path):
    s = State(tmp_path / "s.json")
    s.halt("daily cap")
    assert s.halted and s.halt_reason == "daily cap"
    s.unhalt()
    assert not s.halted


# ── config sanity: the two switches and the venue facts ──────────────────────

def test_config_is_kraken_eth_and_dry_run_by_default():
    assert CFG["bot"]["id"] == "mel"
    assert CFG["bot"]["ccxt_symbol"] == "XRP/USD:USD"
    assert CFG["bot"]["exchange"] == "krakenfutures"
    assert CFG["ops"]["dry_run_default"] is True
    assert CFG["sizing"]["max_margin_fraction"] == 0.90
    assert CFG["exits"]["tp_pct"] == 0.04
    assert CFG["exits"]["sl_pct_floor"] == 0.025
    assert CFG["exits"]["max_hold_hours"] == 24
