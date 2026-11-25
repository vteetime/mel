"""Regression tests for the Terra round-4 review findings.

Findings: 1 an empty but unverified listing still permitted entry · 2 a TP falsely certified
stop visibility · 3 ambiguous submissions outside the entry path · 4 an unreadable venue read
as flat at the recovery bound · 5 TP adoption was structural inference · 6 the opt-in tick
cleanup cancelled unproven orders · 7 the test alert guard could still send with a sink.
"""

import json
import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as C           # noqa: E402
import exchange as ex        # noqa: E402
import notify as N           # noqa: E402
import run_bot               # noqa: E402
from db import TradeDB       # noqa: E402
from engine import Engine    # noqa: E402
from state import State      # noqa: E402
from test_dry_run_scenarios import SCHEMA, SYM, FakeKraken, long_decision  # noqa: E402

CFG = C.load_config()


def _rig(tmp_path, client=None, dry_run=False, db=None, cfg=None):
    client = client or FakeKraken()
    state = State(tmp_path / "state.json")
    if db is None:
        con = sqlite3.connect(tmp_path / "t.db")
        con.executescript(SCHEMA)
        con.commit()
        con.close()
        db = TradeDB(tmp_path / "t.db")
    alerts = []
    eng = Engine(cfg or CFG, client, state, db, alerts.append, dry_run=dry_run)
    return eng, client, state, db, alerts


def _size(eng, funds=732.09, price=1.368, confidence=8, scale=1.0):
    return C.size_position(funds, price, CFG, eng.spec["contract_size"],
                           eng.spec["amount_step"], confidence=confidence,
                           extra_scale=scale)


class BlindTriggers(FakeKraken):
    """This venue's real shape: the account listing answers, triggers are refused."""
    def __init__(self, **kw):
        super().__init__(**kw)
        self.visible = []          # what the account listing actually shows

    def privateGetOpenorders(self, params=None):
        return {"openOrders": list(self.visible)}

    def fetch_open_orders(self, symbol=None, params=None):
        if params and params.get("stop"):
            raise RuntimeError("authenticationError")
        return list(self.visible)

    def fetch_ohlcv(self, symbol=None, timeframe="15m", since=None, limit=200):
        n = int(limit or 200)
        base = 1.35
        rows = []
        for i in range(n):
            px = base + (i % 7) * 1.5
            rows.append([1700000000000 + i * 900000, px, px + 2, px - 2, px + 0.5, 10.0])
        return rows


# ── finding 1: an empty listing is not a clear book ─────────────────────────

def test_an_outstanding_placed_id_makes_the_book_unclear(tmp_path):
    """An id we placed whose state cannot be READ is not clear. (An id the venue definitively
    does not know IS clear — that distinction is the round-7 fix, and it is what stops the flag
    wedging for ever.)"""
    class Unreadable(BlindTriggers):
        def fetch_order(self, oid, symbol=None):
            raise RuntimeError("API outage")

        def fetch_open_orders(self, symbol=None, params=None):
            raise RuntimeError("API outage")

    client = Unreadable()
    eng, client, state, db, alerts = _rig(tmp_path, client=client)
    assert eng.protection_all_clear() is True          # nothing ever placed
    placed = client.create_order("XRP/USD:USD", "stop", "sell", 0.277, None,
                                 {"reduceOnly": True, "stopPrice": 1.3338})
    state.set_flag("placed_ids", [placed["id"]])
    assert eng.protection_all_clear() is False, \
        "an id this bot placed whose state cannot be read must not read as clear"


def test_a_flat_tick_refuses_entry_when_the_book_cannot_be_proved_clear(tmp_path,
                                                                       monkeypatch):
    monkeypatch.setattr(run_bot, "ROOT", tmp_path)
    cfg = json.loads(json.dumps(CFG))
    cfg["ops"].update({"state_file": "state.json", "db_path": str(tmp_path / "t.db"),
                       "lock_file": str(tmp_path / "mel.lock")})
    (tmp_path / "cfg.yaml").write_text(json.dumps(cfg))
    con = sqlite3.connect(tmp_path / "t.db")
    con.executescript(SCHEMA)
    con.commit()
    con.close()
    (tmp_path / "LIVE").write_text("")

    class Unreadable(BlindTriggers):
        """Our order IS on the venue, but its state cannot be read right now."""
        def fetch_order(self, oid, symbol=None):
            raise RuntimeError("API outage")

        def fetch_open_orders(self, symbol=None, params=None):
            raise RuntimeError("API outage")

    client = Unreadable()                               # empty listing, triggers refused
    placed = client.create_order("XRP/USD:USD", "stop", "sell", 0.277, None,
                                 {"reduceOnly": True, "stopPrice": 1.3338})
    st = State(tmp_path / "state.json")
    st.set_flag("placed_ids", [placed["id"]])           # an id we really did place
    monkeypatch.setattr(run_bot.ex, "new_client", lambda *a, **k: client)
    monkeypatch.setenv("KRAKEN_API_KEY", "k")
    monkeypatch.setenv("KRAKEN_API_SECRET", "s")
    monkeypatch.setattr(sys, "argv", ["run_bot.py", "--live", "--config",
                                      str(tmp_path / "cfg.yaml")])
    baseline = len(client.placed)          # the stop placed during setup, nothing more
    rc = run_bot.main()
    assert rc == 1, "an unprovable book must block entry"
    assert client.position is None, "an entry was opened on a book that cannot be proven clear"
    assert len(client.placed) == baseline, "the tick placed an order anyway"


def test_a_virgin_account_may_trade_even_with_invisible_triggers(tmp_path):
    """No id was ever placed, so there is nothing invisible to worry about."""
    client = BlindTriggers()
    eng, client, state, db, alerts = _rig(tmp_path, client=client)
    assert eng.protection_all_clear() is True


# ── finding 2: only the SL certifies stop visibility ────────────────────────

def test_a_visible_tp_does_not_certify_stop_visibility(tmp_path):
    client = BlindTriggers()
    eng, client, state, db, alerts = _rig(tmp_path, client=client)
    eng.try_enter(long_decision(), _size(eng))
    pos = state.get_open()
    # the venue shows the TP but NOT the stop: exactly the shape that fooled the old check
    client.visible = [{"order_id": pos["tp_order_id"], "orderType": "lmt", "side": "sell",
                       "quantity": pos["contracts"], "limitPrice": pos["tp_price"],
                       "status": "open", "reduceOnly": True}]
    state.clear_flag("stop_listing_covers_triggers")
    eng._learn_stop_visibility()
    assert state.get_flag("stop_listing_covers_triggers") is False, \
        "a visible TP certified that stops are listable"


def test_a_visible_sl_does_certify_stop_visibility(tmp_path):
    client = BlindTriggers()
    eng, client, state, db, alerts = _rig(tmp_path, client=client)
    eng.try_enter(long_decision(), _size(eng))
    pos = state.get_open()
    sl_px = [o["price"] for o in client.placed if o["type"] == "stop"][0]
    client.visible = [{"order_id": pos["sl_order_id"], "orderType": "stp", "side": "sell",
                       "quantity": pos["contracts"], "stopPrice": sl_px,
                       "status": "open", "reduceOnly": True}]
    state.clear_flag("stop_listing_covers_triggers")
    eng._learn_stop_visibility()
    assert state.get_flag("stop_listing_covers_triggers") is True


# ── finding 3: ambiguity outside the entry path ─────────────────────────────

class StopTimesOut(FakeKraken):
    """The protective stop is accepted but never answered."""
    def create_order(self, symbol, otype, side, amount, price=None, params=None):
        import ccxt
        if otype == "stop":
            raise ccxt.RequestTimeout("no answer")
        return super().create_order(symbol, otype, side, amount, price, params)


def test_an_ambiguous_stop_is_recorded_as_possibly_resting(tmp_path):
    """[MONEY] The stop may have landed and we have no id: say so, block entries, and do not
    walk away as if the position were protected."""
    client = StopTimesOut()
    eng, client, state, db, alerts = _rig(tmp_path, client=client)
    eng.try_enter(long_decision(), _size(eng))
    pend = state.get_flag("possible_orphan_stop")
    assert pend and pend["trigger"] > 0 and pend["contracts"] > 0, pend
    assert client.position is None, "the unprotected entry must be flattened"
    assert any("may have been accepted" in a for a in alerts)


def test_the_ambiguous_stop_is_adopted_when_it_can_be_found(tmp_path):
    client = StopTimesOut()
    eng, client, state, db, alerts = _rig(tmp_path, client=client)
    eng.try_enter(long_decision(), _size(eng))
    pend = state.get_flag("possible_orphan_stop")
    # it turns out the stop DID land, and the listing can see it
    landed = {"id": "stop-that-landed", "type": "stop", "side": pend["side"],
              "amount": pend["contracts"], "stopPrice": pend["trigger"], "reduceOnly": True,
              "status": "open", "info": {}}
    client.orders = [landed]
    # NOT adopted: a matching shape does not prove the venue's order is ours (round-5
    # finding 5). It is surfaced as a candidate for an explicit human decision.
    assert eng.reconcile_possible_orphan_stop() == "candidate"
    assert "stop-that-landed" not in (state.get_flag("placed_ids") or []), \
        "a shape match must never be remembered as ours"
    lock = state.get_flag("possible_orphan_stop") or {}
    assert lock.get("candidate_id") == "stop-that-landed", lock


def test_an_ambiguous_stop_stays_unresolved_when_nothing_can_be_seen(tmp_path):
    class BlindStopTimeout(StopTimesOut):
        def fetch_open_orders(self, symbol=None, params=None):
            if params and params.get("stop"):
                raise RuntimeError("authenticationError")
            return list(self.visible)

    client = BlindStopTimeout()
    eng, client, state, db, alerts = _rig(tmp_path, client=client)
    eng.try_enter(long_decision(), _size(eng))
    client.orders, client.visible = [], []
    assert eng.reconcile_possible_orphan_stop() == "unresolved_blind"


class CloseTimesOutOnce(FakeKraken):
    """The reduce-only close is accepted once without an answer, then answers."""
    def __init__(self, **kw):
        super().__init__(**kw)
        self.silent = 1

    def create_order(self, symbol, otype, side, amount, price=None, params=None):
        import ccxt
        if otype == "market" and (params or {}).get("reduceOnly") and self.silent:
            self.silent -= 1
            raise ccxt.RequestTimeout("no answer")
        return super().create_order(symbol, otype, side, amount, price, params)


def test_an_ambiguous_force_close_reads_the_venue_and_never_assumes(tmp_path):
    """The close may have filled. State must not be cleared on a guess."""
    class StopFails(CloseTimesOutOnce):
        def create_order(self, symbol, otype, side, amount, price=None, params=None):
            if otype == "stop":
                raise RuntimeError("venue refused the stop")
            return super().create_order(symbol, otype, side, amount, price, params)

    client = StopFails()
    eng, client, state, db, alerts = _rig(tmp_path, client=client)
    out = eng.try_enter(long_decision(), _size(eng))
    # the position is still open when the retry runs, so the retry closes it for real
    assert client.position is None, "the retried close must leave the venue flat"
    assert state.is_flat()


def test_an_ambiguous_hold_cap_close_keeps_state_and_flags_it(tmp_path):
    client = CloseTimesOutOnce()
    eng, client, state, db, alerts = _rig(tmp_path, client=client)
    eng.try_enter(long_decision(), _size(eng))
    client.silent = 1                      # the hold-cap close will go unanswered
    pos = state.get_open()
    pos["entry_time"] = "2026-09-18T00:00:00+00:00"
    state.set_open(pos)
    out = eng.manage(None)
    assert out["status"] == "closing", out
    assert not state.is_flat(), "state must survive an unconfirmed close"
    assert state.get_flag("close_in_flight")


# ── finding 4: an unreadable venue is not flat ──────────────────────────────

def test_the_recovery_bound_requires_positive_flatness(tmp_path):
    """[MONEY] `open_position` reports a failed read as active=False, so the bound could
    record an approximate exit and clear state while the position was still live."""
    client = FakeKraken()
    eng, client, state, db, alerts = _rig(tmp_path, client=client)
    pos = eng.try_enter(long_decision(), _size(eng)) and state.get_open()
    client.trigger(pos["tp_order_id"])
    for f in client.fills:                       # no exit evidence ever appears
        if f["info"]["orderId"] == pos["tp_order_id"]:
            f["amount"] = 0.0
    pos = state.get_open()
    pos["unpriced_ticks"] = 5                    # past the bound
    pos["entry_time"] = "2026-09-18T00:00:00+00:00"
    state.set_open(pos)

    calls = {"n": 0}
    real = client.fetch_positions

    def flaky(symbols=None):
        calls["n"] += 1
        if calls["n"] >= 2:                      # the read used for the flatness check fails
            raise RuntimeError("venue unreadable")
        return real(symbols)

    client.fetch_positions = flaky
    out = eng.manage(None)
    assert out["status"] != "closed", out
    assert not state.is_flat(), "an unreadable venue must not be treated as flat"


def test_flatten_all_never_claims_flat_from_an_outage():
    class Unreadable(FakeKraken):
        def fetch_positions(self, symbols=None):
            raise RuntimeError("outage")

    out = ex.flatten_all(Unreadable(), SYM, attempts=1)
    assert out["flat"] is False and out["error"], out


# ── finding 5: TP adoption is not ownership ─────────────────────────────────

def test_a_matching_limit_is_not_adopted_as_the_take_profit(tmp_path):
    """[MONEY] Adopting it means cancelling an order this bot never placed."""
    client = FakeKraken()
    eng, client, state, db, alerts = _rig(tmp_path, client=client)
    sizing = _size(eng)
    decision = long_decision()
    client.orders = [{"id": "manual-tp", "type": "limit", "side": "sell",
                      "amount": sizing.contracts,
                      "price": ex.round_price(client, SYM, decision.tp),
                      "reduceOnly": True, "status": "open", "info": {}}]
    eng.try_enter(decision, sizing)
    pos = state.get_open() or {}
    assert pos.get("tp_order_id") in ("", None), "a foreign limit was adopted as our TP"
    assert "manual-tp" not in (client.cancelled or [])
    assert len([o for o in client.placed if o["type"] == "limit"]) == 0, \
        "a duplicate TP was stacked next to the matching order"
    assert state.get_flag("tp_state_uncertain") is True
    assert any("not placed by this bot" in a for a in alerts)


def test_the_re_placed_tp_is_not_adopted_either(tmp_path):
    client = FakeKraken()
    eng, client, state, db, alerts = _rig(tmp_path, client=client)
    eng.try_enter(long_decision(), _size(eng))
    pos = state.get_open()
    pos["tp_order_id"] = ""                       # as if placement had failed
    state.set_open(pos)
    client.orders.append({"id": "manual-tp", "type": "limit", "side": "sell",
                          "amount": pos["contracts"], "price": pos["tp_price"],
                          "reduceOnly": True, "status": "open", "info": {}})
    eng._ensure_tp(state.get_open(), pos["contracts"])
    assert state.get_open()["tp_order_id"] in ("", None)


# ── finding 6: the opt-in sweep no longer exists ────────────────────────────

def test_the_ownership_violating_sweep_is_gone():
    assert "cancel_dangling_on_flat" not in CFG["ops"], \
        "the whole-symbol cleanup option must not exist"


# ── finding 7: a test can never send, sink or no sink ───────────────────────

def test_a_test_process_cannot_send_even_with_a_sink(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "chat")
    sent = []
    monkeypatch.setattr(N.urllib.request, "urlopen",
                        lambda *a, **k: sent.append(a) or pytest.fail("sent from a test"))
    sink = []
    N.make_notifier(CFG, enabled=True, sink=sink)("🛑 Mel — must not leave the test")
    assert sent == [] and len(sink) == 1, "a sink must still RECORD the message"


# ── the field-drift guard (the sl_price bug) ────────────────────────────────

REQUIRED_OPEN_FIELDS = {
    # every key the exit path reads back. When the entry dict was rewritten in an earlier
    # round, sl_price silently disappeared, and a stop-out that could not be attributed by
    # order id was then priced at the take-profit level — a fabricated win.
    "trade_id", "direction", "entry_price", "contracts", "remaining", "entry_order_id",
    "opened_ms", "sl_price", "sl_order_id", "tp_price", "tp_order_id", "tp_filled",
    "entry_time", "leverage", "contract_size", "margin_usdt", "funds_usdt",
}


def test_the_entry_state_has_every_field_the_exit_path_reads(tmp_path):
    client = FakeKraken()
    eng, client, state, db, alerts = _rig(tmp_path, client=client)
    eng.try_enter(long_decision(), _size(eng))
    pos = state.get_open()
    assert REQUIRED_OPEN_FIELDS <= set(pos), \
        f"missing from the entry state: {sorted(REQUIRED_OPEN_FIELDS - set(pos))}"
    assert float(pos["sl_price"]) > 0 and float(pos["tp_price"]) > 0


def test_no_exit_reader_references_a_field_the_entry_state_never_sets(tmp_path):
    """Grep the exit-relevant readers for pos['...'] accesses and check them against the set."""
    import re
    src = (ROOT / "engine.py").read_text()
    read = set(re.findall(r'pos\["([a-z_]+)"\]', src))
    read |= set(re.findall(r'pos\.get\("([a-z_]+)"\)', src))
    optional = {
        # from the venue's position read, not from our state
        "active", "error", "last_price",
        # written on demand by the alert dedupe and the recovery path
        "exit_price", "level_fallback", "unpriced_ticks", "tp_state_uncertain",
        "partial_alerted", "priced_alerted", "unpriced_alerted", "flags", "status",
        # written by the protective-stop restore path
        "sl_restore_attempts", "sl_restored_at", "tp_submit_ambiguous",
    }
    missing = {k for k in read if k not in REQUIRED_OPEN_FIELDS and k not in optional}
    assert not missing, f"exit code reads fields the entry state does not set: {sorted(missing)}"
