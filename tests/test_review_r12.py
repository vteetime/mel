"""Mel round-12: two regressions from the round-11 fix, pinned here.

The round-11 fix queued an unconfirmed force-close so the protection-failure breaker would see it.
Round-12 found it left two holes:

  * F2 — the three `AmbiguousSubmission` branches inside `_force_close` still returned BEFORE the
    queue, so an AMBIGUOUS force-close (as opposed to a merely-unconfirmed one) never reached the
    breaker and never blocked entries. They now fall through to the queue like every other failure.

  * F1 — a queued `close_unverified` record carries `trade_id=None`. When the orphan guard then
    ADOPTED the still-live position, the adoption wrote one lifecycle row and, once flatness was
    proven, `retry_pending_close` wrote a SECOND one for the same contracts — double-counting the
    trade and its P&L. Adoption now consumes the queued unverified close.
"""

import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import exchange as ex                                    # noqa: E402
from test_dry_run_scenarios import (                     # noqa: E402
    FakeKraken, long_decision,
)
from test_trip_class import _rig, _size                  # noqa: E402


class CloseSubmissionAmbiguous(FakeKraken):
    """Entry-time stop is refused; the force-close submission then times out (ambiguous).

    The FIRST stop is the entry stop, so it fails and drives `_force_close`. Every later stop
    (the adoption stop) is allowed, so an orphan adoption can still protect the position.
    """

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.stop_attempts = 0

    def create_order(self, symbol, otype, side, amount, price=None, params=None):
        params = params or {}
        if params.get("stopPrice"):
            self.stop_attempts += 1
            if self.stop_attempts == 1:
                raise RuntimeError("venue refused the protective stop")
        if otype == "market" and params.get("reduceOnly"):
            # the close is accepted-or-not, the venue never answers: an ambiguous submission
            raise ex.AmbiguousSubmission("venue did not confirm the force-close")
        return super().create_order(symbol, otype, side, amount, price, params)


def _ambiguous_close_abort(tmp_path):
    eng, client, state, db, alerts = _rig(tmp_path, client=CloseSubmissionAmbiguous())
    out = eng.try_enter(long_decision(), _size(eng))
    return eng, client, state, db, alerts, out


def _row_count(db):
    return db._conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]


def test_an_ambiguous_force_close_is_queued_not_dropped(tmp_path):
    """F2: an AMBIGUOUS force-close must reach the breaker, not vanish on an early return."""
    eng, client, state, db, alerts, out = _ambiguous_close_abort(tmp_path)
    assert out.get("status") == "error", out
    queued = state.get_flag("pending_close") or []
    assert queued, "an ambiguous force-close must be queued, not dropped"
    assert queued[0].get("close_unverified") is True, queued
    assert queued[0].get("reason") == "sl_placement_failed", queued
    # The entry row is now created BEFORE the stop is submitted (the SL-tracking fix), so the
    # queued close references that real row instead of None — which is what lets the retry close
    # the right lifecycle rather than invent a second one.
    assert queued[0].get("trade_id") == 1, queued
    assert any("NOT confirmed" in str(a) for a in alerts), alerts


def test_orphan_adoption_consumes_an_unverified_pending_close(tmp_path):
    """F1: adoption must consume the queued unverified close — else a second row is invented."""
    eng, client, state, db, alerts, _out = _ambiguous_close_abort(tmp_path)
    assert state.get_flag("pending_close"), "precondition: unverified close queued"
    assert client.position, "precondition: the venue still holds the position"

    res = eng.adopt_orphan(ex.open_position(client, eng.sym))
    assert res.get("status") == "adopted", res
    assert not state.get_flag("pending_close"), \
        "adoption must consume the unverified pending close (else a second row is invented)"
    row = db._conn.execute("SELECT trigger, status FROM trades").fetchone()
    assert row is not None and row["status"] == "open"
    # The entry row pre-exists now (created before the stop was submitted), so adoption reuses it
    # rather than minting a fresh "sl_failure_forceclose" row. The one-row invariant is the point;
    # the scoreboard trigger is that pre-existing row's provenance.
    assert row["trigger"] == "scoreboard", row["trigger"]

    # the adopted position later exits and its row is closed by the normal path; prove
    # retry_pending_close has nothing left to invent
    client.position = None
    out = eng.retry_pending_close()
    assert out.get("written", 0) == 0, "a consumed close must not write a phantom row"
    assert _row_count(db) == 1, f"exactly one lifecycle row, got {_row_count(db)}"
