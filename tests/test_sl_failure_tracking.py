"""Mel round-13: a failed entry stop must never leave a naked UNTRACKED position.

Terra's MUST-FIX-NOW on the renamed tree: `try_enter` persisted the position (state + ledger)
LAST — after the stop was placed — so when the stop failed and the force-close could not be
confirmed, `state.open` was still null and a venue-held position had NO record for the per-tick
path (protection re-verification, hold cap, orphan adoption) to manage. The fix records the
position BEFORE the stop is submitted and keeps the record when the close is unconfirmed.

Pinned here:
  (a) SL failure + unconfirmed close => the position IS recorded and IS re-protected/tracked, and
      the record survives;
  (b) the venue later proves flat => the failure settles EXACTLY ONCE — no invented second row;
  (c) deleting the provisional-record write must fail these tests (mutation, run by hand).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_dry_run_scenarios import FakeKraken, long_decision  # noqa: E402
from test_trip_class import _rig, _size                       # noqa: E402


class StopFailsOnceThenRecovers(FakeKraken):
    """Entry stop refused; the force-close reduces nothing; a LATER stop (re-protection) succeeds."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.stop_attempts = 0

    def create_order(self, symbol, otype, side, amount, price=None, params=None):
        params = params or {}
        if params.get("stopPrice"):
            self.stop_attempts += 1
            if self.stop_attempts == 1:
                raise RuntimeError("venue refused the protective stop")
        before = dict(self.position) if self.position else None
        o = super().create_order(symbol, otype, side, amount, price, params)
        if otype == "market" and params.get("reduceOnly") and before:
            self.position = before          # accepted, but nothing was reduced (unconfirmed close)
        return o


def _entry_abort(tmp_path, client=None):
    eng, client, state, db, alerts = _rig(tmp_path, client=client or StopFailsOnceThenRecovers())
    out = eng.try_enter(long_decision(), _size(eng))
    return eng, client, state, db, alerts, out


def _row_count(db):
    return db._conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]


def test_sl_failure_unconfirmed_close_is_recorded_and_reprotected(tmp_path):
    """(a) the possibly-live position must be RECORDED before the stop, and manage() re-protects."""
    eng, client, state, db, alerts, out = _entry_abort(tmp_path)
    assert out.get("status") == "error", out
    assert client.position, "the venue still holds the position (unconfirmed close)"

    pos = state.get_open()
    assert pos is not None, "the possibly-live position MUST be recorded (not left untracked)"
    assert pos["contracts"] > 0 and pos["entry_price"] > 0
    assert pos["trade_id"], "the record must carry its ledger row id"
    assert pos["sl_order_id"] == "", "no stop was placed — it failed at entry"
    assert state.get_flag("pending_close"), "the failed close stays queued and visible"

    # The next tick's manage() re-protects it and supersedes the stale unverified close.
    eng.manage(None)
    pos = state.get_open()
    assert pos and pos["sl_order_id"], "manage() must re-place protection for the tracked position"
    assert not state.get_flag("pending_close"), \
        "the superseded unverified close must be consumed (no second row later)"
    assert _row_count(db) == 1, "still exactly one lifecycle row"


def test_sl_failure_unconfirmed_close_settles_exactly_once_when_flat(tmp_path):
    """(b) once the venue proves flat, the failure settles once — the pre-created row, not a new one."""
    eng, client, state, db, alerts, out = _entry_abort(tmp_path)
    assert client.position
    assert _row_count(db) == 1, "one entry row, created before the stop"

    client.position = None                    # the venue now proves flat
    res = eng.retry_pending_close()
    assert res.get("written", 0) >= 1, res

    rows = db._conn.execute("SELECT id, status, exit_reason FROM trades").fetchall()
    assert len(rows) == 1, f"exactly one lifecycle row, got {rows}"
    assert rows[0]["status"] == "closed" and rows[0]["exit_reason"] == "sl_placement_failed"
    assert not state.get_flag("pending_close"), "the queue must drain"
    assert db.stats()["sl_failure_streak"] >= 1, "the protection-failure breaker must see it"
