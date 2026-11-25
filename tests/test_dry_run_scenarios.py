"""Scenario tests for Mel's tick and exit paths against a fake Kraken.

Each scenario is a branch of a real tick. The ones marked [MONEY] are those that would
leave a naked position, double a position, or corrupt guard accounting if wrong.

What is deliberately NOT here: Jack's 1,281-line dry-run harness. Its scenarios assume a
three-tier ladder (TP1 fill -> breakeven move -> runner trail), so porting it wholesale
would have tested a strategy Mel does not run. Jack's harness is kept in the jack repo as
reference; these are Mel's branches, written for one TP and one SL.
"""

import sqlite3
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config as C           # noqa: E402
import exchange as ex        # noqa: E402
import sync as sync_mod      # noqa: E402
from db import TradeDB       # noqa: E402
from engine import Engine    # noqa: E402
from state import State      # noqa: E402
from strategy import Decision  # noqa: E402

SYM = "XRP/USD:USD"
RAW = "XRP/USD:USD"
CFG = C.load_config()
CFG["ops"] = dict(CFG["ops"], db_path=":memory:")

SCHEMA = """
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT, bot_id TEXT NOT NULL,
    exchange TEXT NOT NULL, symbol TEXT NOT NULL, direction TEXT NOT NULL,
    entry_price REAL NOT NULL, exit_price REAL NOT NULL, contracts REAL NOT NULL,
    leverage INTEGER, notional REAL, pnl_net REAL NOT NULL, pnl_pct REAL NOT NULL,
    fee_usdt REAL, funding_usdt REAL DEFAULT 0, funding_verified INTEGER DEFAULT 0,
    entry_time TEXT NOT NULL,
    exit_time TEXT NOT NULL, duration_h REAL, exit_reason TEXT, status TEXT,
    sl_price REAL, tp1_price REAL, tp2_price REAL, sl_pct REAL, trigger TEXT,
    guard_config TEXT, fee_estimated INTEGER, settlement_source TEXT
    DEFAULT 'provisional', exchange_settlement_id TEXT, settled_at TEXT,
    entry_score REAL, exit_detail TEXT,
    created_at TEXT);
"""


class FakeKraken:
    """ccxt-shaped fake: one-way position, resting orders, fills the test can trigger."""

    def __init__(self, price=1.368, fail_orders=()):
        self.price = price
        self.contract_size, self.step, self.tick = 1.0, 1.0, 0.0001
        self.position = None            # {"contracts", "side", "entryPrice"}
        self.orders = []                # resting
        self.fills = []
        self.placed = []
        self.cancelled = []
        self.fail_orders = set(fail_orders)   # order types to refuse
        self.unreadable = False
        self.ledger_fee = None

    # ── market facts ──
    def market(self, symbol):
        return {"id": "PF_XRPUSD", "contractSize": self.contract_size, "type": "swap",
                "precision": {"price": self.tick, "amount": self.step},
                "limits": {"amount": {"min": None}}}

    def load_markets(self):
        return {}

    def price_to_precision(self, symbol, price):
        return f"{round(round(float(price) / self.tick) * self.tick, 6):.6f}".rstrip("0")

    def amount_to_precision(self, symbol, amount):
        return round(float(amount) / self.step) * self.step

    def fetch_ticker(self, symbol):
        return {"last": self.price}

    # ── reads ──
    def fetch_positions(self, symbols=None):
        if self.unreadable:
            raise RuntimeError("exchange unreadable")
        return [dict(self.position)] if self.position else []

    def fetch_open_orders(self, symbol=None, params=None):
        if params and params.get("stop"):
            return [dict(o) for o in self.orders if o["type"] == "stop"]
        return [dict(o) for o in self.orders]

    def privateGetOpenorders(self, params=None):
        """Kraken's authoritative open-order list (the endpoint that works with this key)."""
        return {"result": "success", "serverTime": "2026-09-19T00:00:00Z",
                "openOrders": [{"order_id": o.get("id"), "orderType": o.get("type"),
                                "side": o.get("side"), "quantity": o.get("amount"),
                                "limitPrice": o.get("price"), "stopPrice": o.get("stopPrice"),
                                "status": o.get("status", "open"),
                                "reduceOnly": o.get("reduceOnly")}
                               for o in self.orders]}

    def fetch_order(self, oid, symbol=None):
        for o in self.orders + self.placed:
            if str(o.get("id")) == str(oid):
                return dict(o)
        # The VENUE RAISES for an id it does not know (proven live: ccxt OrderNotFound for an
        # unknown id, while privateGetOrdersStatus answers with an empty list). It does not
        # return None - and that difference decides whether a vanished stop gets restored or a
        # live order gets forgotten. Returning None here made the fake kinder than the venue.
        import ccxt
        raise ccxt.OrderNotFound(f"no order {oid}")

    def fetch_my_trades(self, symbol=None, limit=100):
        return list(self.fills)

    def fetch_balance(self):
        return {"free": {"USDT": 732.09}, "total": {"USDT": 732.09}}

    def historyGetAccountLog(self, params=None):
        if self.ledger_fee is None:
            return {"logs": []}
        return {"logs": [{"contract": "pf_xrpusd", "info": "futures trade",
                          "fee": self.ledger_fee}]}

    # ── writes ──
    def create_order(self, symbol, otype, side, amount, price=None, params=None):
        params = params or {}
        if otype in self.fail_orders:
            raise RuntimeError(f"venue refused {otype}")
        self._seq = getattr(self, "_seq", 0) + 1
        oid = f"o{self._seq}"
        o = {"id": oid, "symbol": symbol, "type": otype, "side": side,
             "amount": round(float(amount) / self.step) * self.step,
             "price": price, "stopPrice": params.get("stopPrice"),
             "reduceOnly": params.get("reduceOnly", False), "status": "open",
             "filled": 0.0, "average": None, "info": {}}
        self.placed.append(o)
        if otype == "market":
            self._fill_market(o)
        else:
            self.orders.append(o)
        return dict(o)

    def cancel_order(self, oid, symbol=None):
        """A real venue removes the order AND reports it as canceled by id afterwards."""
        import ccxt
        known = {str(o.get("id")) for o in self.orders}
        if str(oid) not in known:
            raise ccxt.OrderNotFound(f"no order {oid}")
        self.cancelled.append(str(oid))
        self.orders = [o for o in self.orders if str(o.get("id")) != str(oid)]
        for rec in self.placed:                     # keep the by-id record truthful
            if str(rec.get("id")) == str(oid):
                rec["status"] = "canceled"
        return {"id": oid, "status": "canceled"}

    # ── test helpers ──
    def _fill_market(self, o):
        n = float(o["amount"])
        px = self.price
        o["status"], o["filled"], o["average"] = "closed", n, px
        if o["reduceOnly"]:
            if self.position:
                left = round(self.position["contracts"] - n, 10)
                self.position = (dict(self.position, contracts=left) if left > 1e-9
                                 else None)
        else:
            self.position = {"contracts": n, "side": "long" if o["side"] == "buy" else "short",
                             "entryPrice": px}
        self.fills.append({"info": {"orderId": o["id"], "fee": round(px * n * 0.0005, 6)},
                           "order": o["id"], "amount": n, "price": px,
                           "fee": {"cost": round(px * n * 0.0005, 6)},
                           "timestamp": int(time.time() * 1000)})
        self.ledger_fee = (self.ledger_fee or 0.0) + round(px * n * 0.0005, 6)

    def trigger(self, oid):
        """Fire a resting protective order: fill it and move the position accordingly.

        The filled order is removed from the resting book but remains retrievable by id
        with status 'closed' — which is how a real venue answers "did it fill?".
        """
        o = next((x for x in self.orders if str(x["id"]) == str(oid)), None)
        assert o is not None, f"no resting order {oid}"
        o = dict(o)
        o["status"], o["filled"] = "closed", float(o["amount"])
        o["average"] = float(o.get("stopPrice") or o.get("price") or self.price)
        for rec in self.placed:                      # keep the by-id record in sync
            if str(rec.get("id")) == str(oid):
                rec.update({"status": "closed", "filled": o["filled"],
                            "average": o["average"]})
        self.orders = [x for x in self.orders if str(x["id"]) != str(oid)]
        px, n = o["average"], float(o["amount"])
        if self.position:
            left = round(self.position["contracts"] - n, 10)
            self.position = dict(self.position, contracts=left) if left > 1e-9 else None
        self.fills.append({"info": {"orderId": o["id"], "fee": round(px * n * 0.0005, 6)},
                           "order": o["id"], "amount": n, "price": px,
                           "fee": {"cost": round(px * n * 0.0005, 6)},
                           "timestamp": int(time.time() * 1000)})
        self.ledger_fee = (self.ledger_fee or 0.0) + round(px * n * 0.0005, 6)

    def order_of_type(self, otype):
        return next((o for o in self.orders if o["type"] == otype), None)


@pytest.fixture()
def env(tmp_path):
    """(client, engine, state, db, alerts) wired like a real tick."""
    client = FakeKraken()
    state = State(tmp_path / "state.json")
    p = tmp_path / "t.db"
    con = sqlite3.connect(p)
    con.executescript(SCHEMA)
    con.commit()
    con.close()
    db = TradeDB(p)
    alerts: list[str] = []

    def make(dry_run=False, cfg=None):
        return Engine(cfg or CFG, client, state, db, alerts.append, dry_run=dry_run)

    yield make, client, state, db, alerts
    db.close()


def long_decision(entry=1.368, tp_pct=0.04, sl_pct=0.025):
    return Decision(action="LONG", reason="score 8 (long=8, short=0)", confidence=8,
                    entry=entry, sl=entry * (1 - sl_pct), tp=entry * (1 + tp_pct),
                    risk_reward=tp_pct / sl_pct, regime="trending_up")


def size_for(engine, funds=732.09, confidence=8, scale=1.0):
    return C.size_position(funds, engine.spec and 1.368, CFG,
                           engine.spec["contract_size"], engine.spec["amount_step"],
                           confidence=confidence, extra_scale=scale)


# ── dry run ──────────────────────────────────────────────────────────────────

def test_dry_run_places_nothing(env):
    make, client, state, db, alerts = env
    eng = make(dry_run=True)
    out = eng.try_enter(long_decision(), size_for(eng))
    assert out["status"] == "dry_run"
    assert client.placed == [], "a dry run must not touch the order endpoints"
    assert state.is_flat() and db.stats()["trades_today"] == 0


# ── entry ────────────────────────────────────────────────────────────────────

def test_entry_places_one_sl_and_one_tp_at_full_size(env):
    """[MONEY] A single TP means the WHOLE position leaves at +4%, and the SL covers the
    whole position too. Placing either at a partial size would leave naked exposure."""
    make, client, state, db, alerts = env
    eng = make()
    sizing = size_for(eng)
    out = eng.try_enter(long_decision(), sizing)
    assert out["status"] == "ok", out

    pos = state.get_open()
    assert pos["contracts"] == pytest.approx(sizing.contracts)
    assert pos["tp_order_id"] and pos["sl_order_id"]
    assert "tp1_order_id" not in pos and "tp2_order_id" not in pos

    kinds = [(o["type"], o["reduceOnly"], o["amount"]) for o in client.orders]
    assert kinds == [("stop", True, pytest.approx(sizing.contracts)),
                     ("limit", True, pytest.approx(sizing.contracts))], kinds
    # the venue's tick is 0.1, so the levels are the decision's levels ROUNDED to it
    assert client.orders[0]["stopPrice"] == pytest.approx(
        ex.round_price(client, SYM, long_decision().sl), rel=1e-6)
    assert float(client.orders[1]["price"]) == pytest.approx(
        ex.round_price(client, SYM, long_decision().tp), rel=1e-6)
    assert client.orders[0]["stopPrice"] == pytest.approx(1.3338, abs=0.005)

    row = db._conn.execute("SELECT bot_id, exchange, status, tp1_price, tp2_price, "
                           "sl_price, contracts FROM trades").fetchone()
    assert row["bot_id"] == "mel" and row["exchange"] == "kraken_futures"
    assert row["status"] == "open" and row["tp2_price"] == 0.0
    assert row["tp1_price"] == pytest.approx(ex.round_price(client, SYM,
                                                            long_decision().tp), rel=1e-6)
    assert any("🐝 Mel — ENTRY" in a for a in alerts)


def test_a_failed_tp_leaves_the_position_protected_and_says_so(env):
    """The SL is the money-critical order. If only the TP fails, the position stays
    protected on the loss side and the next tick puts the TP back."""
    make, client, state, db, alerts = env
    client.fail_orders = {"limit"}
    eng = make()
    sizing = size_for(eng)
    out = eng.try_enter(long_decision(), sizing)
    assert out["status"] == "ok"
    assert client.order_of_type("stop") is not None, "the SL must still be resting"
    assert state.get_open()["tp_order_id"] == ""
    assert any("TP could not be placed" in a for a in alerts)

    # the next tick re-places it, once, with no duplicate
    client.fail_orders = set()
    eng.manage(None)
    assert client.order_of_type("limit") is not None
    assert state.get_open()["tp_order_id"]
    assert len([o for o in client.orders if o["type"] == "limit"]) == 1


def test_sl_failure_forces_the_position_closed_and_records_it(env):
    """[MONEY] No stop means no position. The abort must flatten and the row must exist
    (exit_reason sl_placement_failed) or the daily-loss cap cannot see the failure mode."""
    make, client, state, db, alerts = env
    client.fail_orders = {"stop"}
    eng = make()
    sizing = size_for(eng)
    out = eng.try_enter(long_decision(), sizing)
    assert out["status"] == "error"
    assert client.position is None, "an unprotected position must not be left open"
    assert state.is_flat()
    row = db._conn.execute("SELECT exit_reason, status FROM trades").fetchone()
    assert row["exit_reason"] == "sl_placement_failed"
    assert row["status"] == "closed"
    assert any("ABORTED" in a for a in alerts), \
        "an aborted entry must be reported, not just logged"


# ── exits ────────────────────────────────────────────────────────────────────

def test_tp_fill_closes_the_trade_and_retires_the_sibling_stop(env):
    """[MONEY] The order that did NOT fill is a dangling reduce-only stop: left on the
    book it can close the NEXT position."""
    make, client, state, db, alerts = env
    eng = make()
    eng.try_enter(long_decision(), size_for(eng))
    pos = state.get_open()
    tp_id, sl_id = pos["tp_order_id"], pos["sl_order_id"]

    client.trigger(tp_id)
    out = eng.manage(None)
    assert out["status"] == "closed" and out["reason"] == "tp"
    assert state.is_flat()
    assert sl_id in client.cancelled, "the sibling SL must be cancelled"
    assert client.orders == [], "nothing may be left resting"

    row = db._conn.execute("SELECT exit_reason, exit_price, status, pnl_net, "
                           "settlement_source FROM trades").fetchone()
    assert row["status"] == "closed" and row["exit_reason"] == "tp"
    assert row["exit_price"] == pytest.approx(ex.round_price(client, SYM, long_decision().tp), abs=1e-9)
    assert row["pnl_net"] > 0, "a 4% TP on a long must be a winning row"


def test_stop_fill_closes_the_trade_and_retires_the_sibling_tp(env):
    make, client, state, db, alerts = env
    eng = make()
    eng.try_enter(long_decision(), size_for(eng))
    pos = state.get_open()
    sl_id, tp_id = pos["sl_order_id"], pos["tp_order_id"]

    client.trigger(sl_id)
    out = eng.manage(None)
    assert out["status"] == "closed" and out["reason"] == "sl"
    assert tp_id in client.cancelled
    assert client.orders == []
    row = db._conn.execute("SELECT exit_reason, pnl_net FROM trades").fetchone()
    assert row["exit_reason"] == "sl" and row["pnl_net"] < 0


def test_hold_cap_closes_with_a_market_order_and_records_hold_cap(env):
    make, client, state, db, alerts = env
    eng = make()
    eng.try_enter(long_decision(), size_for(eng))
    pos = state.get_open()
    pos["entry_time"] = (datetime.now(timezone.utc)
                         - timedelta(hours=25)).isoformat()
    state.set_open(pos)

    out = eng.manage(None)
    assert out["status"] == "closed" and out["reason"] == "hold_cap"
    assert state.is_flat() and client.position is None
    for oid in (pos["sl_order_id"], pos["tp_order_id"]):
        assert oid in client.cancelled
    row = db._conn.execute("SELECT exit_reason FROM trades").fetchone()
    assert row["exit_reason"] == "hold_cap"


def test_an_unreadable_exchange_leaves_state_untouched(env):
    """[MONEY] Reading an outage as 'the position closed' would clear state and write a
    phantom close row for a position that is still live."""
    make, client, state, db, alerts = env
    eng = make()
    eng.try_enter(long_decision(), size_for(eng))
    client.unreadable = True
    out = eng.manage(None)
    assert out["status"] == "error"
    assert not state.is_flat(), "state must survive an unreadable venue"
    assert db._conn.execute("SELECT status FROM trades").fetchone()["status"] == "open"


# ── orphan adoption ──────────────────────────────────────────────────────────

def test_orphan_adoption_takes_the_position_and_places_one_tp_and_one_sl(env):
    """[MONEY] State flat + venue holding a position must ADOPT, never open a second one."""
    make, client, state, db, alerts = env
    client.position = {"contracts": 100.0, "side": "long", "entryPrice": 1.35}
    eng = make()
    out = eng.adopt_orphan(ex.open_position(client, SYM))
    assert out["status"] == "adopted"
    assert client.position["contracts"] == pytest.approx(100.0), "no new entry order"
    assert len([o for o in client.placed if not o["reduceOnly"]]) == 0

    pos = state.get_open()
    assert pos["direction"] == "long" and pos["contracts"] == pytest.approx(100.0)
    assert pos["sl_order_id"], "an orphan must be protected"
    assert "tp1_order_id" not in pos
    row = db._conn.execute("SELECT trigger, status FROM trades").fetchone()
    assert row["trigger"] == "adopted_orphan" and row["status"] == "open"


def test_orphan_with_no_stop_gets_one_immediately(env):
    make, client, state, db, alerts = env
    client.position = {"contracts": 100.0, "side": "long", "entryPrice": 1.35}
    eng = make()
    eng.adopt_orphan(ex.open_position(client, SYM))
    stop = client.order_of_type("stop")
    assert stop is not None and stop["reduceOnly"] is True
    assert float(stop["stopPrice"]) == pytest.approx(ex.round_price(client, SYM, 1.35 * 0.975), abs=1e-9)
    assert any("ORPHAN ADOPTED" in a for a in alerts)


# ── reconciliation against the venue ─────────────────────────────────────────

def test_sync_detects_an_orphan(env):
    make, client, state, db, alerts = env
    client.position = {"contracts": 100.0, "side": "long", "entryPrice": 1.35}
    snap = sync_mod.snapshot(client, SYM)
    v = sync_mod.reconcile(snap, state.get_open())
    assert v["action"] == "adopt"
    assert any("ORPHAN" in f for f in v["findings"])
    report = sync_mod.format_report(snap, v, CFG)
    assert "VERDICT: ADOPT" in report and "contract size" in report


def test_sync_detects_stale_state(env):
    make, client, state, db, alerts = env
    state.set_open({"contracts": 100.0, "direction": "long", "entry_price": 1.35})
    snap = sync_mod.snapshot(client, SYM)
    v = sync_mod.reconcile(snap, state.get_open())
    assert v["action"] == "reconcile_closed"
    assert any("STALE STATE" in f for f in v["findings"])


def test_sync_flags_a_live_position_with_no_stop(env):
    make, client, state, db, alerts = env
    client.position = {"contracts": 100.0, "side": "long", "entryPrice": 1.35}
    state.set_open({"contracts": 100.0, "direction": "long", "entry_price": 1.35})
    v = sync_mod.reconcile(sync_mod.snapshot(client, SYM), state.get_open())
    assert v["action"] == "protect"
    assert any("UNPROTECTED" in f for f in v["findings"])


def test_sync_flags_dangling_orders_when_flat_and_refuses_to_strip_protection(env):
    """[MONEY] A leftover stop while flat can close a FUTURE position — but cancelling
    orders is only safe when there is nothing to protect."""
    make, client, state, db, alerts = env
    client.orders = [{"id": "leftover", "type": "stop", "side": "sell", "amount": 100.0,
                      "stopPrice": 1.30, "reduceOnly": True, "status": "open",
                      "info": {}}]
    snap = sync_mod.snapshot(client, SYM)
    v = sync_mod.reconcile(snap, None)
    assert v["action"] == "cancel_dangling"
    assert any("DANGLING" in f for f in v["findings"])

    # WITHOUT owned ids it refuses: cancelling an order the bot cannot prove it placed is
    # the mistake the ownership rule exists to prevent.
    refused = sync_mod.cancel_dangling(client, snap, SYM)
    assert "refused" in refused and client.orders, refused
    # WITH the persisted id it cancels exactly that order.
    out = sync_mod.cancel_dangling(client, snap, SYM, owned_ids=["leftover"])
    assert out["cancelled"] == {"leftover": True} and client.orders == []

    # with a live position, the same call must refuse
    client.position = {"contracts": 100.0, "side": "long", "entryPrice": 1.35}
    client.orders = [{"id": "stop1", "type": "stop", "side": "sell", "amount": 100.0,
                      "stopPrice": 1.30, "reduceOnly": True, "status": "open",
                      "info": {}}]
    snap2 = sync_mod.snapshot(client, SYM)
    out2 = sync_mod.cancel_dangling(client, snap2, SYM, owned_ids=["stop1"])
    assert "refused" in out2 and client.orders, "protection must never be stripped"


def test_sync_reports_no_estimate_as_a_fact(env):
    make, client, state, db, alerts = env
    report = sync_mod.format_report(sync_mod.snapshot(client, SYM),
                                    sync_mod.reconcile(sync_mod.snapshot(client, SYM), None),
                                    CFG)
    assert "732.09" in report
    assert "flat" in report
    assert "not published" in report, "an unpublished venue minimum must say so"
