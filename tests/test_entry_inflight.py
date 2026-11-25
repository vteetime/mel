"""Mel round-14: a fill whose ledger/state write fails must never leave a naked UNTRACKED position.

Terra R2 MUST-FIX-NOW: `try_enter` recorded the position only AFTER the fill, so when the fill
landed but `db.record_entry()` (or `state.set_open()`) raised, the position was left naked AND
untracked — the venue held XRP with no `state.open`, no `pending_close` and no reservation, so the
next tick could treat the bot as flat and open a second position. The fix writes a durable
`entry_inflight` reservation BEFORE the market submit, blocks any second entry while it is set, and
reconciles it at the top of the next tick (re-protecting the live position with its own SL/TP and
re-recording the row).

Pinned here:
  (a) db.record_entry raising  => the venue position stays live, `entry_inflight` is durable, and a
      second entry is refused before any order is submitted;
  (b) state.set_open raising   => same, and the ledger row id is preserved in the reservation;
  (c) reconcile_entry_inflight() re-protects + re-records the live position and retires the
      reservation exactly once.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_dry_run_scenarios import FakeKraken, long_decision  # noqa: E402
from test_trip_class import _rig, _size                       # noqa: E402

import exchange as ex                                         # noqa: E402


def _fail_db_record_entry_always(eng):
    def boom(**kwargs):
        raise RuntimeError("ledger unwritable")
    eng.db.record_entry = boom


def _fail_db_record_entry_once(eng):
    orig = eng.db.record_entry
    state = {"n": 0}

    def flaky(**kwargs):
        state["n"] += 1
        if state["n"] == 1:
            raise RuntimeError("ledger unwritable")
        return orig(**kwargs)

    eng.db.record_entry = flaky
    return state


def test_record_entry_failure_keeps_reservation_and_blocks_a_second_entry(tmp_path):
    """(a) record_entry raising after the fill must leave a durable reservation, not a bare venue."""
    eng, client, state, db, alerts = _rig(tmp_path)
    _fail_db_record_entry_always(eng)

    out = eng.try_enter(long_decision(), _size(eng))
    assert out.get("status") == "error", out
    assert client.position, "the venue still holds the filled position"
    assert state.get_open() is None, "state must stay flat (set_open never ran)"

    r = state.get_flag("entry_inflight")
    assert r, "a durable reservation must survive the ledger-write failure"
    assert r["side"] == "buy" and r["contracts"] > 0 and r["sl_price"] > 0

    # A second entry attempt is refused BEFORE any order is submitted.
    fills_before = len(client.fills)
    out2 = eng.try_enter(long_decision(), _size(eng))
    assert out2.get("status") == "error", out2
    assert "entry_inflight" in out2.get("message", ""), out2
    assert len(client.fills) == fills_before, "the second attempt must not submit a market order"


def test_set_open_failure_keeps_reservation_and_blocks_a_second_entry(tmp_path):
    """(b) set_open raising after record_entry must keep the reservation and the ledger row id."""
    eng, client, state, db, alerts = _rig(tmp_path)
    state.set_open = lambda pos: (_ for _ in ()).throw(RuntimeError("state disk full"))

    out = eng.try_enter(long_decision(), _size(eng))
    assert out.get("status") == "error", out
    assert client.position, "the venue still holds the filled position"

    r = state.get_flag("entry_inflight")
    assert r, "a durable reservation must survive the state-write failure"
    assert r.get("trade_id") is not None, "the ledger row id must be preserved in the reservation"

    fills_before = len(client.fills)
    out2 = eng.try_enter(long_decision(), _size(eng))
    assert out2.get("status") == "error" and "entry_inflight" in out2.get("message", ""), out2
    assert len(client.fills) == fills_before, "the second attempt must not submit a market order"


def test_reservation_is_written_before_the_market_submit(tmp_path):
    """The durable marker must precede the market order, not only the caught-exception handler.

    If the process crashes between the fill and the persistence write (or record_entry raises a
    non-Exception), the except handler never runs — so the only thing that can leave a recovery
    marker is a reservation written BEFORE the submit. Pin that ordering directly.
    """
    eng, client, state, db, alerts = _rig(tmp_path)
    reserved_at_submit = {}
    orig = client.create_order

    def spy(symbol, otype, side, amount, price=None, params=None):
        if otype == "market" and not (params or {}).get("reduceOnly"):
            reserved_at_submit["flag"] = bool(state.get_flag("entry_inflight"))
        return orig(symbol, otype, side, amount, price, params)

    client.create_order = spy
    eng.try_enter(long_decision(), _size(eng))
    assert reserved_at_submit.get("flag") is True, \
        "the entry_inflight reservation must be durable BEFORE the market order is submitted"


def test_reconcile_recovers_and_retires_the_reservation(tmp_path):
    """(c) the next tick reconciles the reserved position: protect + re-record + retire once."""
    eng, client, state, db, alerts = _rig(tmp_path)
    _fail_db_record_entry_once(eng)

    out = eng.try_enter(long_decision(), _size(eng))
    assert out.get("status") == "error", out
    assert state.get_flag("entry_inflight"), "reservation must be set before reconciliation"

    res = eng.reconcile_entry_inflight()
    assert res.get("status") == "recovered", res
    assert state.get_flag("entry_inflight") is None, "the reservation is retired after recovery"

    pos = state.get_open()
    assert pos, "the recovered position must be tracked"
    assert pos["trade_id"] and pos["sl_order_id"], "tracked + protected"
    assert any(o["type"] == "stop" for o in client.orders), "a protective stop must rest on the venue"

    # A second reconcile is a no-op: it must not stack a second stop.
    stops_before = len([o for o in client.orders if o["type"] == "stop"])
    res2 = eng.reconcile_entry_inflight()
    assert res2.get("status") in ("none", "already_tracked"), res2
    stops_after = len([o for o in client.orders if o["type"] == "stop"])
    assert stops_after == stops_before, "re-reconciling must not stack a second stop"


class AmbiguousEntrySubmit(FakeKraken):
    """The market entry FILLS on the venue, but the submit reply is lost (AmbiguousSubmission)."""

    def create_order(self, symbol, otype, side, amount, price=None, params=None):
        params = params or {}
        if otype == "market" and not params.get("reduceOnly"):
            super().create_order(symbol, otype, side, amount, price, params)
            raise ex.AmbiguousSubmission("timeout after an accepted entry submit")
        return super().create_order(symbol, otype, side, amount, price, params)


def test_ambiguous_submit_adoption_failure_keeps_the_reservation(tmp_path):
    """Terra R3: clearing the reservation BEFORE the ambiguous-submit adoption is the bug.

    The venue holds the fill; the adoption then fails. The reservation must survive so the next
    tick reconciles the still-live position instead of treating the bot as flat (and opening a
    second one).
    """
    eng, client, state, db, alerts = _rig(tmp_path, client=AmbiguousEntrySubmit())

    def boom():
        raise RuntimeError("adoption crashed")

    eng.reconcile_entry_inflight = boom

    out = eng.try_enter(long_decision(), _size(eng))
    assert client.position, "precondition: the venue accepted and holds the entry"
    assert out.get("status") == "error", out
    assert client.position, "the venue still holds the filled position"
    assert state.get_open() is None, "no state was recorded"
    assert state.get_flag("entry_inflight"), \
        "the reservation MUST survive a failed adoption (this was the R3 defect)"

    fills_before = len(client.fills)
    out2 = eng.try_enter(long_decision(), _size(eng))
    assert out2.get("status") == "error" and "entry_inflight" in out2.get("message", ""), out2
    assert len(client.fills) == fills_before, "the second attempt must not submit a market order"


def test_ambiguous_submit_adoption_uses_the_reservations_own_terms(tmp_path):
    """The adoption records the position with THIS entry's SL, not a config-floor stop."""
    eng, client, state, db, alerts = _rig(tmp_path, client=AmbiguousEntrySubmit())
    dec = long_decision(entry=1.368, tp_pct=0.04, sl_pct=0.04)   # strategy SL != the 2.5% floor
    out = eng.try_enter(dec, _size(eng))
    assert out.get("status") == "adopted", out
    pos = state.get_open()
    assert pos and pos["sl_order_id"], "the adopted position must be tracked + protected"
    assert state.get_flag("entry_inflight") is None, "the reservation is retired after adoption"
    floor_sl = round(1.368 * (1 - 0.025), 4)
    assert abs(pos["sl_price"] - floor_sl) > 0.001, \
        "the adopted stop must use the entry's own SL, not the config-floor stop"
    assert any(o["type"] == "stop" for o in client.orders), "a protective stop must rest"
