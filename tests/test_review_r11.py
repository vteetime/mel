"""Mel round-11: an unconfirmed force-close must stay visible to the breaker.

The reviewer found the path with no test: `_force_close` recorded nothing when the venue could not
confirm the close, so `sl_failure_streak` never incremented and the protection-failure breaker that
exists to stop Trip's blind re-entry never fired (Trip's churn cost ~$226 in fees).

The fix has two halves, and the second one is a hazard the first would have introduced:
  * the record is queued even when the close is unconfirmed, marked `close_unverified`, so the
    failure reaches the guards instead of vanishing with the position;
  * the ledger backlog blocks ENTRIES only — it used to return from the whole tick, which for an
    unconfirmed close (a position that may still be live) would have skipped the orphan guard and
    `manage()`, leaving a real position unmanaged and naked.

Both halves are pinned here.
"""

import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_bot                                          # noqa: E402
from state import State                                 # noqa: E402
from test_dry_run_scenarios import (                     # noqa: E402
    SCHEMA, FakeKraken, long_decision,
)
from test_trip_class import CFG, _rig, _size             # noqa: E402


class StopFailsAndCloseDoesNotFill(FakeKraken):
    """Entry-time stop is refused; the force-close is then ACCEPTED but reduces nothing.

    That combination is what the old code dropped: no confirmed close, so no record, no queue.

    The force-close is identified by SHAPE (a reduce-only market order while the venue holds a
    position), not by a clientOid: `ex.place_market` sets none — unlike `place_stop`, which tags
    every order with an id we can search for. An entry or a force-close whose submit reply is
    lost therefore has no client-side id to reconcile by; the engine reads the position instead.
    """

    def create_order(self, symbol, otype, side, amount, price=None, params=None):
        if params and params.get("stopPrice"):           # the protective stop at entry
            raise RuntimeError("venue refused the protective stop")
        before = dict(self.position) if self.position else None
        o = super().create_order(symbol, otype, side, amount, price, params)
        if otype == "market" and (params or {}).get("reduceOnly") and before:
            self.position = before                       # accepted, but nothing was reduced
        return o


def _entry_abort(tmp_path):
    eng, client, state, db, alerts = _rig(tmp_path, client=StopFailsAndCloseDoesNotFill())
    out = eng.try_enter(long_decision(), _size(eng))
    return eng, client, state, db, alerts, out


def _rows(tmp_path):
    with sqlite3.connect(tmp_path / "t.db") as con:
        return con.execute("SELECT exit_reason FROM trades WHERE status='closed'").fetchall()


def test_an_unconfirmed_force_close_is_queued_not_dropped(tmp_path):
    """Trip's churn was invisible because the failure path returned before recording."""
    eng, client, state, db, alerts, out = _entry_abort(tmp_path)
    assert out.get("status") == "error", out
    queued = state.get_flag("pending_close") or []
    assert queued, "an unconfirmed close must be queued, not dropped"
    assert queued[0].get("close_unverified") is True, queued
    assert queued[0].get("reason") == "sl_placement_failed", queued
    assert any("NOT confirmed" in str(a) or "could NOT be confirmed" in str(a) for a in alerts), \
        alerts


def test_an_unconfirmed_close_is_not_written_while_the_position_may_be_live(tmp_path):
    """A 'closed' row for a position that is still open invents P&L and hides a naked position.

    The position itself is now RECORDED (written before the stop was submitted), so the fake
    venue holding a position is matched by a tracked state record — the ticket a later tick
    uses to re-protect it. Only the CLOSE is withheld while it may be live.
    """
    eng, client, state, db, alerts, _out = _entry_abort(tmp_path)
    assert client.position, "the fake venue still holds the position"
    assert state.get_open() is not None, \
        "the possibly-live position must be RECORDED and tracked, not left untracked"
    before = _rows(tmp_path)
    out = eng.retry_pending_close()
    assert out.get("written", 0) == 0, out
    assert _rows(tmp_path) == before, "a closed row was written for a possibly-live position"
    assert state.get_flag("pending_close"), "the record must still be queued"
    assert any("UNCONFIRMED" in str(a) for a in alerts), alerts
    assert db.stats()["sl_failure_streak"] == 0, \
        "an unproven close must not be counted as a settled protection failure"


def test_an_unconfirmed_close_is_written_once_the_venue_proves_flat(tmp_path):
    """The other half: when the venue confirms the position is gone, the failure must land."""
    eng, client, state, db, alerts, _out = _entry_abort(tmp_path)
    client.position = None                    # the venue now answers: nothing open
    out = eng.retry_pending_close()
    assert out.get("written", 0) >= 1, out
    reasons = [r[0] for r in _rows(tmp_path)]
    assert "sl_placement_failed" in reasons, reasons
    assert not state.get_flag("pending_close"), "the queue must be drained"
    assert db.stats()["sl_failure_streak"] >= 1, \
        "the protection-failure breaker must see this trade"


def test_a_queued_close_blocks_entries_but_never_management(tmp_path, monkeypatch):
    """The hazard the fix itself could have introduced.

    The ledger backlog used to return from the whole tick. With a queued UNCONFIRMED close the
    venue may still hold the position, so returning early skipped the orphan guard and `manage()`
    — leaving a live position unmanaged. Entries must be blocked; management must still run.
    """
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

    client = FakeKraken()
    client.position = {"contracts": 0.277, "side": "long", "entryPrice": 1.368}
    st = State(tmp_path / "state.json")
    st.set_open({"trade_id": 1, "direction": "long", "entry_price": 1.368, "contracts": 0.277,
                 "remaining": 0.277, "sl_price": 1.3338, "sl_order_id": "sl-x",
                 "tp_price": 1.42272, "tp_order_id": "tp-x",
                 "entry_time": "2026-09-20T00:00:00+00:00"})
    st.set_flag("pending_close", [{
        "trade_id": 2, "close_unverified": True, "needs_price": False, "exit_price": 1.35,
        "entry_price": 1.368, "contracts": 0.277, "reason": "sl_placement_failed",
        "side": "buy", "leverage": 2, "margin_fraction": 0.25, "settlement_source": "provisional"}])

    calls = {"manage": 0, "enter": 0}
    monkeypatch.setattr(run_bot.Engine, "manage",
                        lambda self, _x=None: (calls.__setitem__("manage", calls["manage"] + 1)
                                               or {"status": "holding"}))
    monkeypatch.setattr(run_bot.Engine, "try_enter",
                        lambda self, _d, _s: (calls.__setitem__("enter", calls["enter"] + 1)
                                              or {"status": "ok"}))
    monkeypatch.setattr(run_bot.ex, "new_client", lambda *a, **k: client)
    monkeypatch.setenv("KRAKEN_API_KEY", "k")
    monkeypatch.setenv("KRAKEN_API_SECRET", "s")
    monkeypatch.setattr(sys, "argv", ["run_bot.py", "--live", "--config",
                                      str(tmp_path / "cfg.yaml")])
    rc = run_bot.main()
    assert calls["manage"] == 1, \
        "the tick never reached manage(): an unconfirmed close blocked POSITION management"
    assert calls["enter"] == 0, "entries must stay blocked while the ledger has a close outstanding"
    # rc 0 is correct: `manage()` handled the open position and the tick ended there, so the
    # entries-only block was never reached. The point of the test is that management RAN.
    assert rc == 0, rc
