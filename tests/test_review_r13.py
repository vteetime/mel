"""Mel round-13: the naked-position stranding on a double-ambiguous failure.

Round-13's verdict was PASS, but it flagged one pre-arming blocker (d): when an entry's SL
submission is AMBIGUOUS (``possible_orphan_stop`` set) and the follow-up force-close is ALSO
ambiguous (F2 queues a ``close_unverified`` pending close), the venue may still hold the
position while state is flat. ``run_bot.main()`` used to return from the WHOLE tick on the
unresolved stop — before ``retry_pending_close`` and the orphan guard — so a live leveraged
position sat naked with no auto-recovery until a human ran ``--sync --resolve-ambiguous-stop``.

The unresolved stop must gate ENTRIES only (it may be resting untracked and could close a
FUTURE position); the orphan guard must still run and adopt/re-protect the live position.
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
from test_dry_run_scenarios import SCHEMA, FakeKraken   # noqa: E402
from test_trip_class import CFG                         # noqa: E402


def _tick_state(tmp_path, monkeypatch, client, queue_close=True):
    """Wire a flat-state tick: LIVE sentinel, flat state, unresolved orphan stop, optional
    queued unverified close."""
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
    # FLAT state: the entry aborted inside `_force_close` before `set_open`, so no open row.
    st.set_flag("possible_orphan_stop", {"contracts": 0.277, "trigger": 1.3338,
                                         "side": "sell", "at": 0, "where": "entry"})
    if queue_close:
        st.set_flag("pending_close", [{
            "trade_id": None, "close_unverified": True, "needs_price": False,
            "exit_price": 1.35, "entry_price": 1.368, "contracts": 0.277,
            "reason": "sl_placement_failed", "side": "buy", "leverage": 2,
            "margin_fraction": 0.25, "settlement_source": "provisional"}])

    monkeypatch.setattr(run_bot.ex, "new_client", lambda *a, **k: client)
    monkeypatch.setenv("KRAKEN_API_KEY", "k")
    monkeypatch.setenv("KRAKEN_API_SECRET", "s")
    monkeypatch.setattr(sys, "argv", ["run_bot.py", "--live", "--config",
                                      str(tmp_path / "cfg.yaml")])
    return cfg


def test_an_unresolved_orphan_stop_does_not_strand_a_live_position(tmp_path, monkeypatch):
    """(d) round-13 pre-arming blocker: the stop must gate entries, never the whole tick.

    With `possible_orphan_stop` unresolved AND a live position on the venue, the tick must
    still reach the orphan guard and ADOPT the position — the old code returned 1 first and
    left it naked.
    """
    client = FakeKraken()
    client.position = {"contracts": 0.277, "side": "long", "entryPrice": 1.368}
    _tick_state(tmp_path, monkeypatch, client)

    calls = {"adopt": 0}
    monkeypatch.setattr(run_bot.Engine, "retry_pending_close", lambda self: {"written": 0})
    monkeypatch.setattr(run_bot.Engine, "adopt_orphan",
                        lambda self, live: (calls.__setitem__("adopt", calls["adopt"] + 1)
                                            or {"status": "adopted"}))

    rc = run_bot.main()
    assert calls["adopt"] == 1, \
        "an unresolved orphan stop must gate ENTRIES only, not strand the live position"
    assert rc == 0, rc


def test_an_unresolved_orphan_stop_still_blocks_entries_when_flat(tmp_path, monkeypatch):
    """The other half of (d): when the venue is FLAT, the stop must still block the entry search.

    The orphan guard has nothing to adopt, so the tick falls through to the entry gates — the
    stop's entries-only block must fire there, not let the tick open a position over a stop
    that may be resting untracked. No close is queued, so the ONLY thing blocking entry is the
    stop gate itself.
    """
    client = FakeKraken()            # venue is flat: no position
    _tick_state(tmp_path, monkeypatch, client, queue_close=False)

    calls = {"enter": 0}
    monkeypatch.setattr(run_bot.Engine, "try_enter",
                        lambda self, _d, _s: (calls.__setitem__("enter", calls["enter"] + 1)
                                              or {"status": "ok"}))

    rc = run_bot.main()
    assert calls["enter"] == 0, \
        "an unresolved orphan stop must still block the entry search while flat"
    assert rc == 1, rc
