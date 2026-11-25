"""Mel round-7 fixes: the two venue-shape assumptions that no test could see.

FINDING 1 — `order_gone` could never prove an order gone on this venue (it required a trigger
listing this key refuses), so `dangling_orders` could be set once and never clear, and
`--sync --cancel-dangling` never touched the flag: the README promised a way out that the code did
not implement. Reproduced live: for an id that certainly does not exist,
`privateGetOrdersStatus` answers `orders: []` and ccxt raises `OrderNotFound`, yet
`order_gone(bogus)` returned False.

The fix splits two questions that were sharing one conservative rule:
  * "may I CANCEL this id?"         -> `order_gone` (unchanged: absence must come from a book that
                                       can be shown to be COMPLETE)
  * "may I stop TRACKING this id?"  -> `order_provably_finished` (a definitive by-id answer is
                                       enough — the venue saying "no such order" is a fact about
                                       the id, not a claim about a listing's completeness)

FINDING 2 — `stop_is_confirmed` read the trigger from ccxt's `fetch_order` and consulted our own
listing only when that call returned NOTHING. If ccxt returned an order whose trigger it had not
mapped, a good stop read as "no trigger": at entry that force-closes every position, per-tick it
can never confirm protection.
"""

import json
import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import ccxt                 # noqa: E402
import exchange as ex       # noqa: E402
import run_bot              # noqa: E402
from state import State     # noqa: E402
from test_dry_run_scenarios import SCHEMA, SYM, FakeKraken, CFG   # noqa: E402


# ── 1. the two questions, and the asymmetry between them ───────────────────

def test_a_definitively_unknown_id_is_provably_finished():
    """The clearance question must be answerable without a trigger listing."""
    class Venue(FakeKraken):
        def fetch_order(self, oid, symbol=None):
            raise ccxt.OrderNotFound("no such order")     # exactly what the live venue answers

    assert ex.order_provably_finished(Venue(), SYM, "never-existed") is True


def test_a_definitively_unknown_id_is_still_not_cancellable():
    """…while the CANCEL question stays conservative on a venue whose trigger listing is refused,
    which is the situation that made order_gone unable to answer at all."""
    class Venue(FakeKraken):
        def fetch_order(self, oid, symbol=None):
            raise ccxt.OrderNotFound("no such order")

        def fetch_open_orders(self, symbol=None, params=None):
            if params and params.get("stop"):
                raise RuntimeError("authenticationError")      # this key cannot list stops
            return []

    assert ex.order_gone(Venue(), SYM, "never-existed") is False
    assert ex.order_provably_finished(Venue(), SYM, "never-existed") is True


def test_an_unreadable_id_is_not_provably_finished():
    """A read failure is not proof of absence — this is what keeps the block in place."""
    class Outage(FakeKraken):
        def fetch_order(self, oid, symbol=None):
            raise RuntimeError("API outage")

        def fetch_open_orders(self, symbol=None, params=None):
            raise RuntimeError("API outage")

    assert ex.order_provably_finished(Outage(), SYM, "ours") is False


def test_a_none_return_is_not_proof_of_absence():
    """ccxt RAISES for an unknown id; a bare None is anomalous and must not be read as gone."""
    class NoneVenue(FakeKraken):
        def fetch_order(self, oid, symbol=None):
            return None

    assert ex.order_provably_finished(NoneVenue(), SYM, "ours") is False


def test_a_resting_order_is_never_provably_finished():
    """THE safety-critical direction: a survivor must not be forgotten.

    If someone later widened the terminal-status set (or the venue answered oddly), this is the
    assertion that catches a reduce-only order being dropped while it still rests.
    """
    class Resting(FakeKraken):
        def fetch_order(self, oid, symbol=None):
            return {"id": oid, "status": "open", "side": "sell", "amount": 0.277,
                    "stopPrice": 1.3338, "info": {"reduceOnly": True}}

    assert ex.order_provably_finished(Resting(), SYM, "ours") is False
    # and a status the venue might introduce tomorrow must default to "still resting"
    class OddStatus(FakeKraken):
        def fetch_order(self, oid, symbol=None):
            return {"id": oid, "status": "some_new_status", "side": "sell", "amount": 0.277,
                    "stopPrice": 1.3338, "info": {"reduceOnly": True}}

    assert ex.order_provably_finished(OddStatus(), SYM, "ours") is False


def test_a_filled_id_is_provably_finished():
    class Venue(FakeKraken):
        def fetch_order(self, oid, symbol=None):
            return {"id": oid, "status": "closed", "side": "sell", "amount": 1.0,
                    "stopPrice": 1.3338, "info": {}}

    assert ex.order_provably_finished(Venue(), SYM, "ours") is True


# ── 2. the trigger, read from wherever the venue reports it ────────────────

def test_the_trigger_is_read_from_our_listing_when_ccxt_does_not_map_it():
    """The live-proven listing carries stopPrice; a good stop must not read as unverifiable."""
    class CcxtWithoutTrigger(FakeKraken):
        """fetch_order returns the order but without the trigger ccxt failed to map."""
        def fetch_order(self, oid, symbol=None):
            return {"id": oid, "status": "open", "side": "sell", "amount": 0.277,
                    "stopPrice": None, "info": {"reduceOnly": True}}

        def privateGetOpenorders(self, params=None):
            return {"openOrders": [{"order_id": "sl-1", "orderType": "stp", "side": "sell",
                                    "quantity": 0.277, "stopPrice": 1.3338,
                                    "reduceOnly": True, "status": "open"}]}

    ok, why = ex.stop_is_confirmed(CcxtWithoutTrigger(), SYM, "sl-1", side="sell",
                                   min_amount=0.277, require_breakeven=False,
                                   entry=1.368, trigger=1.3338)
    assert ok, why


def test_the_cross_check_does_not_weaken_the_level_comparison():
    """Reading the trigger elsewhere must still COMPARE it to the intended level."""
    class WrongLevel(FakeKraken):
        def fetch_order(self, oid, symbol=None):
            return {"id": oid, "status": "open", "side": "sell", "amount": 0.277,
                    "stopPrice": None, "info": {"reduceOnly": True}}

        def privateGetOpenorders(self, params=None):
            return {"openOrders": [{"order_id": "sl-1", "orderType": "stp", "side": "sell",
                                    "quantity": 0.277, "stopPrice": 1.30,
                                    "reduceOnly": True, "status": "open"}]}

    ok, why = ex.stop_is_confirmed(WrongLevel(), SYM, "sl-1", side="sell", min_amount=0.277,
                                   require_breakeven=False, entry=1.368, trigger=1.3338)
    assert not ok and "not the expected" in why, why


def test_a_stop_with_no_trigger_anywhere_is_still_rejected():
    class NoTriggerAnywhere(FakeKraken):
        def fetch_order(self, oid, symbol=None):
            return {"id": oid, "status": "open", "side": "sell", "amount": 0.277,
                    "stopPrice": None, "info": {"reduceOnly": True}}

        def privateGetOpenorders(self, params=None):
            return {"openOrders": []}

    ok, why = ex.stop_is_confirmed(NoTriggerAnywhere(), SYM, "sl-1", side="sell",
                                   min_amount=0.277, require_breakeven=False,
                                   entry=1.368, trigger=1.3338)
    assert not ok and "no trigger" in why, why


# ── 3. the dangling flag clears when the ids are disprovable ───────────────

def test_the_dangling_flag_clears_for_a_disprovable_id(tmp_path, monkeypatch):
    """The wedge: a flag set by an id the venue does not know must not block entries for ever."""
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
    st = State(tmp_path / "state.json")
    st.set_flag("dangling_orders", ["never-existed-1", "never-existed-2"])

    class Venue(FakeKraken):
        def fetch_order(self, oid, symbol=None):
            raise ccxt.OrderNotFound("no such order")

        def fetch_ohlcv(self, symbol=None, timeframe="15m", since=None, limit=200):
            n = int(limit or 200)
            return [[1700000000000 + i * 900000, 1.35, 2602, 2598, 2600.5, 10.0]
                    for i in range(n)]

    monkeypatch.setattr(run_bot.ex, "new_client", lambda *a, **k: Venue())
    monkeypatch.setenv("KRAKEN_API_KEY", "k")
    monkeypatch.setenv("KRAKEN_API_SECRET", "s")
    monkeypatch.setattr(sys, "argv", ["run_bot.py", "--live", "--config",
                                      str(tmp_path / "cfg.yaml")])
    rc = run_bot.main()
    st.load()          # the tick wrote the file; this object is a cached copy
    assert not st.get_flag("dangling_orders"), \
        "the dangling flag survived ids the venue definitively does not know"


# ── 4. the operator's way out ──────────────────────────────────────────────

def test_clear_dangling_needs_arming_and_a_note(tmp_path, monkeypatch):
    """Behaviour, not a source-substring match: both refusals must actually happen."""
    import contextlib
    import io
    cfg = json.loads(json.dumps(CFG))
    cfg["ops"]["state_file"] = "state.json"
    monkeypatch.setattr(run_bot, "ROOT", tmp_path)
    # NO network: the venue is a fake (an unreachable Kraken used to fail these spuriously)
    monkeypatch.setattr(run_bot.ex, "new_client", lambda *a, **k: FakeKraken())
    st = State(tmp_path / "state.json")
    st.set_flag("dangling_orders", ["id-a"])

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = run_bot.do_sync(cfg, {"apiKey": "k", "secret": "s"}, dry_run=True,
                             clear_dangling=True, note="x")
    assert rc == 1 and "ARMED" in buf.getvalue(), buf.getvalue()
    st.load()
    assert st.get_flag("dangling_orders") == ["id-a"], "an unarmed call cleared the flag"

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = run_bot.do_sync(cfg, {"apiKey": "k", "secret": "s"}, dry_run=False,
                             clear_dangling=True, note="")
    assert rc == 1 and "note" in buf.getvalue().lower(), buf.getvalue()
    st.load()
    assert st.get_flag("dangling_orders") == ["id-a"], "an un-noted call cleared the flag"


def test_clear_dangling_also_resolves_the_placed_id_history(tmp_path, monkeypatch):
    """The flat-state gate checks placed_ids: clearing only the dangling flag would leave the
    operator blocked after following the documented procedure.

    `id-b` is UNANSWERABLE (a read failure) — a genuinely disprovable id would not need the
    operator's word at all, so the fake makes the venue refuse to answer for it.
    """
    import contextlib
    import io

    class HalfBlind(FakeKraken):
        def fetch_order(self, oid, symbol=None):
            if str(oid) == "id-b":
                raise RuntimeError("the venue would not answer for this id")
            raise ccxt.OrderNotFound(f"no order {oid}")

    cfg = json.loads(json.dumps(CFG))
    cfg["ops"]["state_file"] = "state.json"
    monkeypatch.setattr(run_bot, "ROOT", tmp_path)
    monkeypatch.setattr(run_bot.ex, "new_client", lambda *a, **k: HalfBlind())
    st = State(tmp_path / "state.json")
    st.set_flag("dangling_orders", ["id-a"])
    st.set_flag("placed_ids", ["id-a", "id-b"])

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = run_bot.do_sync(cfg, {"apiKey": "k", "secret": "s"}, dry_run=False,
                             clear_dangling=True, note="checked the UI: nothing resting")
    assert rc == 0, buf.getvalue()
    st.load()
    assert not st.get_flag("dangling_orders")
    assert "id-b" not in (st.get_flag("placed_ids") or []), \
        "an unanswerable placed id survived, so the flat-state gate still blocks"
    audit = st.get_flag("cleared_dangling") or {}
    assert "id-b" in (audit.get("placed_ids_asserted_gone") or []), \
        "the assertion must record which ids it removed"


def test_clear_dangling_records_what_was_asserted(tmp_path, monkeypatch):
    cfg = json.loads(json.dumps(CFG))
    cfg["ops"]["state_file"] = "state.json"
    monkeypatch.setattr(run_bot, "ROOT", tmp_path)
    monkeypatch.setattr(run_bot.ex, "new_client", lambda *a, **k: FakeKraken())
    st = State(tmp_path / "state.json")
    st.set_flag("dangling_orders", ["id-a"])
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = run_bot.do_sync(cfg, {"apiKey": "k", "secret": "s"}, dry_run=False,
                             clear_dangling=True, note="checked the Kraken UI: nothing resting")
    assert rc == 0, buf.getvalue()
    st.load()          # do_sync wrote the file; this object is a cached copy
    assert not st.get_flag("dangling_orders")
    audit = st.get_flag("cleared_dangling") or {}
    assert "checked the Kraken UI" in json.dumps(audit) and audit.get("ids") == ["id-a"]


# ── round 9: the escape must refuse when a position is live, label ids honestly,
#    and keep more than one clearance on record ─────────────────────────────


def test_clear_dangling_refuses_while_a_position_is_open(tmp_path, monkeypatch):
    """An open position's resting ids ARE its protection: recording them as asserted gone would
    corrupt the history that proves what this bot ever placed."""
    import contextlib
    import io
    c = FakeKraken()
    c.position = {"contracts": 0.2, "side": "long", "entryPrice": 1.35}
    cfg = json.loads(json.dumps(CFG))
    cfg["ops"]["state_file"] = "state.json"
    monkeypatch.setattr(run_bot, "ROOT", tmp_path)
    monkeypatch.setattr(run_bot.ex, "new_client", lambda *a, **k: c)
    st = State(tmp_path / "state.json")
    st.set_flag("dangling_orders", ["id-a"])
    st.set_flag("placed_ids", ["id-a"])

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = run_bot.do_sync(cfg, {"apiKey": "k", "secret": "s"}, dry_run=False,
                             clear_dangling=True, note="looked at the UI")
    assert rc == 1 and "OPEN" in buf.getvalue(), buf.getvalue()
    st.load()
    assert st.get_flag("dangling_orders") == ["id-a"], "the flag was cleared on a live position"
    assert "id-a" in (st.get_flag("placed_ids") or []), "live ids were dropped from the history"


def test_a_still_resting_id_is_named_as_such_not_as_unanswerable(tmp_path, monkeypatch):
    """The operator must be told which ids the venue reports as LIVE: that is the whole risk of
    this assertion, and "unanswerable" hid it."""
    import contextlib
    import io

    class Resting(FakeKraken):
        def fetch_order(self, oid, symbol=None):
            return {"id": oid, "status": "open", "side": "sell", "amount": 0.277,
                    "stopPrice": 1.3338, "info": {"reduceOnly": True}}

    cfg = json.loads(json.dumps(CFG))
    cfg["ops"]["state_file"] = "state.json"
    monkeypatch.setattr(run_bot, "ROOT", tmp_path)
    monkeypatch.setattr(run_bot.ex, "new_client", lambda *a, **k: Resting())
    st = State(tmp_path / "state.json")
    st.set_flag("dangling_orders", ["still-open-1"])

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = run_bot.do_sync(cfg, {"apiKey": "k", "secret": "s"}, dry_run=False,
                             clear_dangling=True, note="the UI showed nothing")
    out = buf.getvalue()
    assert rc == 0, out
    assert "STILL RESTING" in out, out
    assert "could not answer" not in out, "a resting order is not an unanswered read: " + out
    st.load()
    rec = st.get_flag("cleared_dangling") or {}
    assert "still-open-1" in (rec.get("still_resting") or []), rec


def test_each_clearance_is_kept_not_overwritten(tmp_path, monkeypatch):
    """Last-write-wins would destroy the first assertion's evidence on the second clearance."""
    import contextlib
    import io
    cfg = json.loads(json.dumps(CFG))
    cfg["ops"]["state_file"] = "state.json"
    monkeypatch.setattr(run_bot, "ROOT", tmp_path)
    monkeypatch.setattr(run_bot.ex, "new_client", lambda *a, **k: FakeKraken())
    st = State(tmp_path / "state.json")

    for n, i in ((1, "first-id"), (2, "second-id")):
        st.set_flag("dangling_orders", [i])
        with contextlib.redirect_stdout(io.StringIO()):
            rc = run_bot.do_sync(cfg, {"apiKey": "k", "secret": "s"}, dry_run=False,
                                 clear_dangling=True, note=f"clearance {n}")
        assert rc == 0
        st.load()
    hist = st.get_flag("cleared_dangling_history") or []
    assert len(hist) == 2, hist
    assert [h["note"] for h in hist] == ["clearance 1", "clearance 2"], hist
    assert hist[0]["ids"] == ["first-id"] and hist[1]["ids"] == ["second-id"], hist


# ── round 10: flatness must be PROVEN, and ids counted once ────────────────


def test_clear_dangling_refuses_when_the_position_cannot_be_READ(tmp_path, monkeypatch):
    """An unreadable position is not a flat position.

    `open_position` reports a failed read as {"active": False, "error": ...}, so a guard that
    checked only `active` would clear ids during an outage — while a position might be live.
    """
    import contextlib
    import io

    class Blind(FakeKraken):
        def fetch_positions(self, symbols=None):
            raise RuntimeError("the venue would not answer for positions")

    cfg = json.loads(json.dumps(CFG))
    cfg["ops"]["state_file"] = "state.json"
    monkeypatch.setattr(run_bot, "ROOT", tmp_path)
    monkeypatch.setattr(run_bot.ex, "new_client", lambda *a, **k: Blind())
    st = State(tmp_path / "state.json")
    st.set_flag("dangling_orders", ["id-a"])

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = run_bot.do_sync(cfg, {"apiKey": "k", "secret": "s"}, dry_run=False,
                             clear_dangling=True, note="looked at the UI")
    assert rc == 1, buf.getvalue()
    assert "cannot prove the account is flat" in buf.getvalue(), buf.getvalue()
    st.load()
    assert st.get_flag("dangling_orders") == ["id-a"], "a failed read cleared the flag"


def test_an_id_in_both_lists_is_counted_once(tmp_path, monkeypatch):
    """Entry residuals are remembered in BOTH dangling_orders and placed_ids; the audit must not
    report the same id twice, or the operator is told a false count of what they are asserting."""
    import contextlib
    import io

    class Resting(FakeKraken):
        def fetch_order(self, oid, symbol=None):
            return {"id": oid, "status": "open", "side": "sell", "amount": 0.277,
                    "stopPrice": 1.3338, "info": {"reduceOnly": True}}

    cfg = json.loads(json.dumps(CFG))
    cfg["ops"]["state_file"] = "state.json"
    monkeypatch.setattr(run_bot, "ROOT", tmp_path)
    monkeypatch.setattr(run_bot.ex, "new_client", lambda *a, **k: Resting())
    st = State(tmp_path / "state.json")
    st.set_flag("dangling_orders", ["same-id"])
    st.set_flag("placed_ids", ["same-id"])

    with contextlib.redirect_stdout(io.StringIO()):
        rc = run_bot.do_sync(cfg, {"apiKey": "k", "secret": "s"}, dry_run=False,
                             clear_dangling=True, note="UI showed one order, gone")
    assert rc == 0
    st.load()
    rec = st.get_flag("cleared_dangling") or {}
    assert rec.get("still_resting") == ["same-id"], rec
    assert len(rec.get("still_resting") or []) == 1, "the id was double-counted"
    assert len(rec.get("placed_ids_asserted_gone") or []) == 1, rec
