"""Regression tests for the Terra round-2 review findings.

Findings: 1 unreadable book treated as clean · 2 dry-run sync could cancel live orders ·
3 symbol-wide cleanup could cancel foreign orders · 4 a force-close loss could vanish ·
5 partial exit fills finalised as a complete trade · 6 orphan adoption took foreign
protection and placed TP before SL · 7 exit ROE drifted with a post-entry leverage edit.
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
import run_bot               # noqa: E402
import settlement            # noqa: E402
from db import TradeDB       # noqa: E402
from engine import Engine    # noqa: E402
from state import State      # noqa: E402
from test_dry_run_scenarios import SCHEMA, SYM, FakeKraken, long_decision  # noqa: E402

CFG = C.load_config()


def _db(path):
    con = sqlite3.connect(path)
    con.executescript(SCHEMA)
    con.commit()
    con.close()
    return TradeDB(path)


def _rig(tmp_path, client=None, dry_run=False, db=None):
    client = client or FakeKraken()
    state = State(tmp_path / "state.json")
    db = db or _db(tmp_path / "t.db")
    alerts = []
    eng = Engine(CFG, client, state, db, alerts.append, dry_run=dry_run)
    return eng, client, state, db, alerts


def _size(eng, funds=732.09, price=1.368, confidence=8, scale=1.0):
    return C.size_position(funds, price, CFG, eng.spec["contract_size"],
                           eng.spec["amount_step"], confidence=confidence,
                           extra_scale=scale)


def _enter(eng, state, client):
    eng.try_enter(long_decision(), _size(eng))
    return state.get_open()


# ── finding 1: an unreadable book is not a clean book ────────────────────────

def test_a_failed_book_read_raises_in_strict_mode_and_is_empty_in_reporting_mode():
    class Blind(FakeKraken):
        """Every order source is down — that is what an outage looks like."""
        def privateGetOpenorders(self, params=None):
            raise RuntimeError("API outage")

        def fetch_open_orders(self, symbol=None, params=None):
            raise RuntimeError("API outage")

    b = Blind()
    assert ex.open_orders(b, SYM) == [], "reporting mode stays tolerant"
    with pytest.raises(ex.ExchangeError):
        ex.open_orders(b, SYM, strict=True)


def test_retirement_keeps_its_block_when_the_book_cannot_be_read(tmp_path):
    """[MONEY] Clearing the flag on an outage would re-open trading with a reduce-only
    survivor possibly still resting."""
    class BlindAfterPlace(FakeKraken):
        """The orders EXIST on the venue, but every read now fails (an outage)."""
        def fetch_order(self, oid, symbol=None):
            raise RuntimeError("API outage")

        def fetch_open_orders(self, symbol=None, params=None):
            raise RuntimeError("API outage")

    client = BlindAfterPlace()
    eng, client, state, db, alerts = _rig(tmp_path, client=client)
    # the ids must be REAL (present on the venue) or their absence would be definitive - which is
    # the whole point of the round-7 fix: an unknown id is disprovable, an unreadable one is not
    sl = client.create_order("XRP/USD:USD", "stop", "sell", 0.2, None,
                             {"reduceOnly": True, "stopPrice": 2535.0})
    tp = client.create_order("XRP/USD:USD", "limit", "sell", 0.2, 1.40, {"reduceOnly": True})
    state.set_open({"trade_id": 1, "direction": "long", "entry_price": 1.35,
                    "contracts": 0.2, "sl_order_id": sl["id"], "tp_order_id": tp["id"],
                    "tp_filled": False, "sl_price": 2535.0, "tp_price": 1.40,
                    "entry_time": "2026-09-19T00:00:00+00:00"})
    residual = eng._retire_protection(state.get_open(), "a test close")
    assert residual == [sl["id"], tp["id"]], residual
    assert state.get_flag("dangling_orders") == [sl["id"], tp["id"]]


# ── finding 2: cancellation requires an armed bot ────────────────────────────

def test_dry_run_sync_refuses_to_cancel_orders(tmp_path, monkeypatch, capsys):
    client = FakeKraken()
    client.orders = [{"id": "ours", "type": "stop", "side": "sell", "amount": 0.2,
                      "stopPrice": 1.30, "reduceOnly": True, "status": "open",
                      "info": {}}]
    monkeypatch.setattr(run_bot.ex, "new_client", lambda *a, **k: client)
    rc = run_bot.do_sync(CFG, {"api_key": "k", "api_secret": "s"},
                         dry_run=True, cancel_dangling=True)
    out = capsys.readouterr().out
    assert rc == 1 and "REFUSED" in out
    assert client.cancelled == [], "a dry-run sync cancelled a live order"
    assert [o["id"] for o in client.orders] == ["ours"]


def test_armed_sync_cancels_only_mel_owned_ids(tmp_path, monkeypatch):
    """Armed cancellation acts on the ids the bot persisted — nothing else."""
    client = FakeKraken()
    client.orders = [
        {"id": "ours", "type": "stop", "side": "sell", "amount": 0.2,
         "stopPrice": 1.30, "reduceOnly": True, "status": "open", "info": {}},
        {"id": "manual", "type": "stop", "side": "sell", "amount": 0.2,
         "stopPrice": 2400.0, "reduceOnly": True, "status": "open", "info": {}},
    ]
    monkeypatch.setattr(run_bot, "ROOT", tmp_path)
    monkeypatch.setattr(run_bot.ex, "new_client", lambda *a, **k: client)
    cfg = json.loads(json.dumps(CFG))
    cfg["ops"]["state_file"] = "state.json"
    st = State(tmp_path / "state.json")
    st.set_flag("dangling_orders", ["ours"])

    rc = run_bot.do_sync(cfg, {"api_key": "k", "api_secret": "s"},
                         dry_run=False, cancel_dangling=True)
    assert rc == 0
    assert client.cancelled == ["ours"], client.cancelled
    assert [o["id"] for o in client.orders] == ["manual"], \
        "an order this bot did not record was cancelled"


def test_armed_sync_refuses_without_persisted_ownership(tmp_path, monkeypatch, capsys):
    client = FakeKraken()
    client.orders = [{"id": "manual", "type": "stop", "side": "sell", "amount": 0.2,
                      "stopPrice": 2400.0, "reduceOnly": True, "status": "open",
                      "info": {}}]
    monkeypatch.setattr(run_bot, "ROOT", tmp_path)
    monkeypatch.setattr(run_bot.ex, "new_client", lambda *a, **k: client)
    cfg = json.loads(json.dumps(CFG))
    cfg["ops"]["state_file"] = "state.json"
    run_bot.do_sync(cfg, {"api_key": "k", "api_secret": "s"},
                    dry_run=False, cancel_dangling=True)
    assert client.cancelled == []
    assert client.orders, "nothing may be cancelled without proof of ownership"


# ── finding 3: never cancel what the bot cannot prove it placed ──────────────

def test_retirement_leaves_foreign_orders_alone_and_does_not_remember_them(tmp_path):
    client = FakeKraken()
    eng, client, state, db, alerts = _rig(tmp_path, client=client)
    pos = _enter(eng, state, client)
    client.orders.append({"id": "manual", "type": "stop", "side": "sell", "amount": 0.2,
                          "stopPrice": 1.30, "reduceOnly": True, "status": "open",
                          "info": {}})
    client.trigger(pos["tp_order_id"])
    eng.manage(None)
    assert "manual" not in client.cancelled, "a foreign order was cancelled"
    assert state.get_flag("dangling_orders") in (None, []), \
        "a foreign order was remembered for cancellation"
    assert any("were NOT placed by this bot" in a for a in alerts)


def test_force_close_retires_only_the_id_it_submitted(tmp_path):
    client = FakeKraken(fail_orders={"stop"})
    eng, client, state, db, alerts = _rig(tmp_path, client=client)
    client.orders.append({"id": "manual", "type": "stop", "side": "sell", "amount": 0.1,
                          "stopPrice": 2000.0, "reduceOnly": True, "status": "open",
                          "info": {}})
    eng.try_enter(long_decision(), _size(eng))
    assert "manual" not in client.cancelled
    assert client.position is None, "the aborted entry must be flattened"


def test_a_flat_book_with_unexplained_protection_blocks_and_is_not_cancelled(tmp_path,
                                                                             monkeypatch):
    """The tick must not sweep orders it cannot prove it placed; it reports and refuses."""
    monkeypatch.setattr(run_bot, "ROOT", tmp_path)
    cfg = json.loads(json.dumps(CFG))
    cfg["ops"].update({"state_file": "state.json", "db_path": str(tmp_path / "t.db"),
                       "lock_file": str(tmp_path / "mel.lock")})
    (tmp_path / "cfg.yaml").write_text(json.dumps(cfg))
    db = _db(tmp_path / "t.db")
    db.close()
    (tmp_path / "LIVE").write_text("")            # armed

    client = FakeKraken()
    client.orders = [{"id": "leftover", "type": "stop", "side": "sell", "amount": 0.2,
                      "stopPrice": 1.30, "reduceOnly": True, "status": "open",
                      "info": {}}]
    monkeypatch.setattr(run_bot.ex, "new_client", lambda *a, **k: client)
    monkeypatch.setenv("KRAKEN_API_KEY", "k")
    monkeypatch.setenv("KRAKEN_API_SECRET", "s")
    monkeypatch.setattr(sys, "argv", ["run_bot.py", "--live", "--config",
                                      str(tmp_path / "cfg.yaml")])
    rc = run_bot.main()
    assert rc == 1, "an unexplained protective order must stop new entries"
    assert client.cancelled == [] and client.orders, "the order was cancelled anyway"


# ── finding 4: a confirmed force-close must reach the ledger ─────────────────

def test_force_close_that_cannot_be_written_is_queued_and_retried(tmp_path):
    class FlakyOnce(TradeDB):
        """Fails the first exit write, like a locked or mis-schema'd ledger."""
        fails = 1

        def record_exit(self, *a, **k):
            if type(self).fails:
                type(self).fails -= 1
                return False
            return super().record_exit(*a, **k)

    _db(tmp_path / "t.db").close()                 # schema first
    db = FlakyOnce(tmp_path / "t.db")
    client = FakeKraken(fail_orders={"stop"})
    state = State(tmp_path / "state.json")
    alerts = []
    eng = Engine(CFG, client, state, db, alerts.append, dry_run=False)
    eng.try_enter(long_decision(), _size(eng))

    queued = state.get_flag("pending_close") or []
    assert queued, "a force-closed loss must not be forgotten"
    assert any("queued" in a.lower() for a in alerts)

    out = eng.retry_pending_close()
    assert out["status"] == "ok" and out["written"] == 1
    assert not (state.get_flag("pending_close") or [])
    row = db._conn.execute("SELECT exit_reason, status FROM trades WHERE status='closed'"
                           ).fetchone()
    assert row["exit_reason"] == "sl_placement_failed"


def test_pending_close_records_the_margin_fraction_it_deployed(tmp_path):
    class NoWrite(TradeDB):
        def record_exit(self, *a, **k):
            return False

    _db(tmp_path / "t.db").close()                 # schema first
    db = NoWrite(tmp_path / "t.db")
    client = FakeKraken(fail_orders={"stop"})
    state = State(tmp_path / "state.json")
    eng = Engine(CFG, client, state, db, lambda m: None, dry_run=False)
    sizing = _size(eng)
    eng.try_enter(long_decision(), sizing)
    queued = (state.get_flag("pending_close") or [])[0]
    assert queued["margin_fraction"] == pytest.approx(sizing.margin_fraction, abs=1e-6)


# ── finding 5: a partially-evidenced exit is not recordable ──────────────────

def test_partial_fill_evidence_blocks_finalisation(tmp_path):
    client = FakeKraken()
    eng, client, state, db, alerts = _rig(tmp_path, client=client)
    pos = _enter(eng, state, client)
    client.trigger(pos["tp_order_id"])
    for f in client.fills:                      # only part of the exit is recovered
        if f["info"]["orderId"] == pos["tp_order_id"]:
            f["amount"] = round(float(f["amount"]) / 2, 6)
    out = eng.manage(None)
    assert out["status"] == "partial_settlement", out
    assert not state.is_flat(), "state must stay open so a real exit can be recorded later"
    assert db._conn.execute("SELECT COUNT(*) c FROM trades WHERE status='closed'"
                            ).fetchone()["c"] == 0
    assert any("does not yet account for the exit" in a for a in alerts)


def test_settlement_re_queries_wider_before_calling_fills_partial():
    """The first read is a small tail; a busy account needs the wider second ask."""
    calls = []

    class Wider(FakeKraken):
        def fetch_my_trades(self, symbol=None, limit=100, **kw):
            calls.append(limit)
            if limit >= 1000:
                return [{"info": {"orderId": "o9", "fee": 0.1}, "order": "o9",
                         "amount": 0.2, "price": 1.40, "fee": {"cost": 0.1},
                         "timestamp": 1}]
            return [{"info": {"orderId": "o9", "fee": 0.1}, "order": "o9",
                     "amount": 0.05, "price": 1.40, "fee": {"cost": 0.1},
                     "timestamp": 1}]

    b = Wider()
    res = settlement.settled_close(b, SYM, close_order_id="o9", entry_price=1.35,
                                   contracts=0.2, contract_size=1.0, fallback_price=0.0,
                                   opened_ms=1_700_000_000_000)
    assert 1000 in calls, "the wider re-query never ran"
    assert res["fills_complete"] is True and res["exit_price"] == pytest.approx(1.40)


# ── finding 6: adoption places OUR protection, SL first ──────────────────────

def test_orphan_adoption_adopts_nothing_and_places_sl_before_tp(tmp_path):
    client = FakeKraken()
    eng, client, state, db, alerts = _rig(tmp_path, client=client)
    client.position = {"contracts": 2.0, "side": "long", "entryPrice": 1.35}
    client.orders = [
        {"id": "foreign-sl", "type": "stop", "side": "sell", "amount": 2.0,
         "stopPrice": 2535.0, "reduceOnly": True, "status": "open", "info": {}},
        {"id": "foreign-tp", "type": "limit", "side": "sell", "amount": 2.0,
         "price": 1.40, "reduceOnly": True, "status": "open", "info": {}},
    ]
    out = eng.adopt_orphan(ex.open_position(client, SYM))
    assert out["status"] == "adopted"

    ours = [o for o in client.placed if o["type"] in ("stop", "limit")]
    assert [o["type"] for o in ours] == ["stop", "limit"], \
        "the stop must be placed and confirmed BEFORE the take-profit"
    pos = state.get_open()
    assert pos["sl_order_id"] == ours[0]["id"] and pos["tp_order_id"] == ours[1]["id"]
    assert pos["sl_order_id"] not in ("foreign-sl", "foreign-tp")
    assert pos["tp_order_id"] != "foreign-tp", "a foreign order was adopted"
    assert "foreign-sl" not in client.cancelled
    assert any("were NOT placed by this bot" in a for a in alerts)


def test_adoption_records_the_venue_leverage(tmp_path):
    client = FakeKraken()
    eng, client, state, db, alerts = _rig(tmp_path, client=client)
    client.position = {"contracts": 2.0, "side": "long", "entryPrice": 1.35,
                       "leverage": 5}
    eng.adopt_orphan(ex.open_position(client, SYM))
    assert state.get_open()["leverage"] == 5


# ── finding 7: the exit is costed with the entry's terms ─────────────────────

def test_exit_roe_is_frozen_at_entry_leverage(tmp_path):
    """A config edit after entry must not restate a closed trade's risk."""
    client = FakeKraken()
    eng, client, state, db, alerts = _rig(tmp_path, client=client)
    pos = _enter(eng, state, client)
    entry_margin = pos["margin_usdt"]

    cfg2 = json.loads(json.dumps(CFG))
    cfg2["sizing"]["leverage"] = 10          # the operator "increases leverage" on paper
    eng2 = Engine(cfg2, client, state, db, alerts.append, dry_run=False)
    client.trigger(pos["tp_order_id"])
    out = eng2.manage(None)
    assert out["status"] == "closed"
    row = db._conn.execute("SELECT pnl_net, pnl_pct, leverage FROM trades "
                           "WHERE status='closed'").fetchone()
    assert row["leverage"] == 2, "the entry's leverage is what the trade carries"
    expected = out["pnl_usdt"] / entry_margin * 100.0
    assert row["pnl_pct"] == pytest.approx(expected, rel=1e-3), \
        "ROE was recomputed with the NEW leverage instead of the entry's"


# ── venue quirk surfaced while fixing finding 1 ─────────────────────────────

def test_a_refused_trigger_listing_is_not_an_outage():
    """Kraken rejects the trigger-order query with `authenticationError` for this key.

    Ken swallowed that silently for months (bare `except: pass`), which is a very plausible
    source of his duplicate protective stops. The MAIN open-orders call failing is an
    outage and must raise; the trigger supplement failing is a venue permission quirk and
    must not block trading, because stop ownership is established by ID, not by listing.
    """

    class NoTriggerList(FakeKraken):
        def __init__(self):
            super().__init__()
            self.orders = [{"id": "limit-1", "type": "limit", "side": "sell",
                            "amount": 0.1, "price": 2700.0, "reduceOnly": True,
                            "status": "open", "info": {}}]

        def fetch_open_orders(self, symbol=None, params=None):
            if params and params.get("stop"):
                raise RuntimeError("authenticationError")
            return [dict(o) for o in self.orders]

    b = NoTriggerList()
    got = ex.open_orders(b, SYM, strict=True)          # must not raise
    assert [o["id"] for o in got] == ["limit-1"]
    assert ex.order_gone(b, SYM, "limit-1") is False, "a resting order is not gone"
    assert ex.order_gone(b, SYM, "never-existed") is False, \
        "absence from a listing that cannot be complete is not proof of a cancel"

    class FullView(FakeKraken):
        """A venue that DOES enumerate triggers: absence from its listing means gone."""
        def fetch_open_orders(self, symbol=None, params=None):
            return []

    assert ex.order_gone(FullView(), SYM, "never-existed") is True

    class MainCallFails(FakeKraken):
        """Both sources down = an outage, and an outage must raise in strict mode."""
        def privateGetOpenorders(self, params=None):
            raise RuntimeError("network down")

        def fetch_open_orders(self, symbol=None, params=None):
            raise RuntimeError("network down")

    with pytest.raises(ex.ExchangeError):
        ex.open_orders(MainCallFails(), SYM, strict=True)


def test_order_gone_reads_the_venue_not_a_cached_opinion(tmp_path):
    client = FakeKraken()
    eng, client, state, db, alerts = _rig(tmp_path, client=client)
    pos = _enter(eng, state, client)
    assert ex.order_gone(client, SYM, pos["sl_order_id"]) is False   # still resting
    client.trigger(pos["tp_order_id"])
    eng.manage(None)
    assert ex.order_gone(client, SYM, pos["sl_order_id"]) is True     # retired


def test_partial_evidence_is_bounded_and_ends_in_an_honest_row(tmp_path):
    """Refusing forever is its own failure: a trade that can never be priced keeps state
    open, blocks every future entry and stays invisible to the guard rails. With the venue
    confirmed FLAT, the fallback is the position's own level, flagged as unverified."""
    client = FakeKraken()
    eng, client, state, db, alerts = _rig(tmp_path, client=client)
    pos = _enter(eng, state, client)
    client.trigger(pos["tp_order_id"])
    for f in client.fills:                      # the exit fill never materialises
        if f["info"]["orderId"] == pos["tp_order_id"]:
            f["amount"] = 0.0
    limit = 3
    for i in range(limit - 1):
        out = eng.manage(None)
        assert out["status"] == "partial_settlement", (i, out)
        assert not state.is_flat()
    out = eng.manage(None)
    assert out["status"] == "closed", out
    assert state.is_flat()
    row = db._conn.execute("SELECT exit_reason, settlement_source FROM trades "
                           "WHERE status='closed'").fetchone()
    assert row["settlement_source"] == "level_fallback"
    assert any("from the" in a and "LEVEL" in a for a in alerts)
