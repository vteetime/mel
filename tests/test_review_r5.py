"""Regression tests for the Terra round-5 review findings.

1 an accepted force-close RETRY treated as a fill · 2 an untracked survivor from an ambiguous
SL in adoption/restoration · 3 a duplicated TP after an ambiguous submit · 4 estimated P&L
steering the live guard rails · 5 a shape-matched stop adopted as ours.
"""

import json
import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as C          # noqa: E402
import guards as G          # noqa: E402
import run_bot              # noqa: E402
from db import TradeDB      # noqa: E402
from engine import Engine   # noqa: E402
from state import State     # noqa: E402
from test_dry_run_scenarios import SCHEMA, FakeKraken, long_decision  # noqa: E402

CFG = C.load_config()


def _rig(tmp_path, client=None, cfg=None):
    client = client or FakeKraken()
    state = State(tmp_path / "state.json")
    con = sqlite3.connect(tmp_path / "t.db")
    con.executescript(SCHEMA)
    con.commit()
    con.close()
    db = TradeDB(tmp_path / "t.db")
    alerts = []
    eng = Engine(cfg or CFG, client, state, db, alerts.append, dry_run=False)
    return eng, client, state, db, alerts


def _size(eng, confidence=8):
    return C.size_position(732.09, 1.368, CFG, eng.spec["contract_size"],
                           eng.spec["amount_step"], confidence=confidence, extra_scale=1.0)


# ── 1. acceptance is not a fill ─────────────────────────────────────────────

def test_an_accepted_but_unfilled_close_retry_keeps_state(tmp_path):
    """[MONEY] The venue takes the retry, nothing fills: state must NOT be cleared, and the
    position must not be abandoned while it is still open."""
    class StopFailsCloseAccepted(FakeKraken):
        def create_order(self, symbol, otype, side, amount, price=None, params=None):
            import ccxt
            if otype == "stop":
                raise RuntimeError("venue refused the stop")
            if otype == "market" and (params or {}).get("reduceOnly"):
                # accepted, never filled — the exact shape the old code called close_ok
                raise ccxt.RequestTimeout("no answer, order may exist")
            return super().create_order(symbol, otype, side, amount, price, params)

    client = StopFailsCloseAccepted()
    eng, client, state, db, alerts = _rig(tmp_path, client=client)
    out = eng.try_enter(long_decision(), _size(eng))
    # The entry's stop was refused, so a close was attempted; both the close and its retry
    # were accepted without filling. The bot must NOT claim the position is gone.
    assert client.position is not None, \
        "the venue still holds the position — nothing may report it as closed"
    assert out.get("status") == "error", out
    # round-12: an unconfirmed close is now QUEUED (not silently dropped) so the
    # protection-failure breaker sees it; state stays flat and the position is not claimed closed.
    assert state.get_flag("pending_close"), \
        "the unconfirmed close must be queued so the breaker sees it"
    assert any("MANUAL" in a or "not confirmed" in a for a in alerts), alerts


# ── 2. ambiguous SL in adoption / restoration ───────────────────────────────

def test_an_ambiguous_adopted_stop_raises_the_durable_lock(tmp_path):
    class AdoptStopTimesOut(FakeKraken):
        def create_order(self, symbol, otype, side, amount, price=None, params=None):
            import ccxt
            if otype == "stop" and (params or {}).get("stopPrice") and self.position:
                raise ccxt.RequestTimeout("no answer")
            return super().create_order(symbol, otype, side, amount, price, params)

    client = AdoptStopTimesOut()
    eng, client, state, db, alerts = _rig(tmp_path, client=client)
    # an orphan exists on the venue with no state
    orphan = {"contracts": 0.2, "side": "long", "entry_price": 1.35}
    client.position = orphan
    out = eng.adopt_orphan(orphan)
    lock = state.get_flag("possible_orphan_stop")
    assert lock and lock.get("where") == "sl-adopt", \
        "an ambiguous adopted stop must leave a durable, findable marker"
    assert out.get("status") in ("orphan_unprotected", "adopted")
    assert any("ADOPTED" in a and "untracked" in a for a in alerts)


def test_an_ambiguous_restore_locks_and_fails_closed(tmp_path):
    """Restoring a vanished stop that goes unanswered must not silently continue."""
    class RestoreTimesOut(FakeKraken):
        def __init__(self, **kw):
            super().__init__(**kw)
            self.stops = 0

        def create_order(self, symbol, otype, side, amount, price=None, params=None):
            import ccxt
            if otype == "stop":
                self.stops += 1
                if self.stops > 1:          # the entry stop lands; the restore never answers
                    raise ccxt.RequestTimeout("no answer")
            return super().create_order(symbol, otype, side, amount, price, params)

    client = RestoreTimesOut()
    eng, client, state, db, alerts = _rig(tmp_path, client=client)
    eng.try_enter(long_decision(), _size(eng))
    pos = state.get_open()
    oid = str(pos["sl_order_id"])
    client.orders = [o for o in client.orders if str(o["id"]) != oid]
    client.placed = [o for o in client.placed if str(o["id"]) != oid]
    eng.manage(None)
    lock = state.get_flag("possible_orphan_stop")
    assert lock and lock.get("where") == "sl-restore", lock
    assert state.is_flat(), "an unconfirmable restore must fail closed, not hold naked"


# ── 3. an ambiguous TP is never duplicated ──────────────────────────────────

def test_an_ambiguous_tp_is_not_re_sent_on_a_guess(tmp_path):
    class TpTimesOut(FakeKraken):
        def create_order(self, symbol, otype, side, amount, price=None, params=None):
            import ccxt
            if otype == "limit":
                raise ccxt.RequestTimeout("no answer")
            return super().create_order(symbol, otype, side, amount, price, params)

    client = TpTimesOut()
    eng, client, state, db, alerts = _rig(tmp_path, client=client)
    eng.try_enter(long_decision(), _size(eng))
    assert state.get_flag("tp_submit_ambiguous"), "the ambiguity must be remembered"
    before = len([o for o in client.placed if o["type"] == "limit"])
    for _ in range(4):
        eng.manage(None)
    after = len([o for o in client.placed if o["type"] == "limit"])
    assert after == before, "the TP was re-sent blindly after an ambiguous submit"
    assert any("will NOT be re-sent" in a or "NOT be re-sent" in a for a in alerts)


def test_an_ambiguous_tp_stops_after_the_bounded_retries(tmp_path):
    """If the book keeps showing nothing and the venue keeps not answering, it gives up."""
    class FlakyTp(FakeKraken):
        def __init__(self, **kw):
            super().__init__(**kw)
            self.tp_calls = 0

        def create_order(self, symbol, otype, side, amount, price=None, params=None):
            import ccxt
            if otype == "limit":
                self.tp_calls += 1
                if self.tp_calls > 3:
                    raise ccxt.RequestTimeout("no answer")
            return super().create_order(symbol, otype, side, amount, price, params)

    client = FlakyTp()
    eng, client, state, db, alerts = _rig(tmp_path, client=client)
    eng.try_enter(long_decision(), _size(eng))
    for _ in range(8):
        eng.manage(None)
    assert client.tp_calls <= 4, f"unbounded TP attempts: {client.tp_calls}"


# ── 4. estimated P&L must not steer the guards ──────────────────────────────

def test_an_estimated_settlement_halts_entries_until_reconciled(tmp_path):
    eng, client, state, db, alerts = _rig(tmp_path)
    eng.try_enter(long_decision(), _size(eng))
    pos = state.get_open()
    db.record_exit(int(pos["trade_id"]), exit_price=1.35, pnl_net=5.0, pnl_pct=1.4,
                   exit_reason="unverified_exit", settlement_source="level_fallback")
    stats = db.stats()
    assert stats["unverified_settlements"] == 1, stats
    ok, why = G.g_unverified_settlements(stats["unverified_settlements"], CFG)
    assert not ok and "estimated price" in why


def test_an_estimated_row_does_not_invent_a_loss_streak(tmp_path):
    """A guessed exit must not manufacture (or hide) consecutive losses for the breaker."""
    eng, client, state, db, alerts = _rig(tmp_path)
    eng.try_enter(long_decision(), _size(eng))
    pos = state.get_open()
    db.record_exit(int(pos["trade_id"]), exit_price=1.30, pnl_net=-40.0, pnl_pct=-11.0,
                   exit_reason="unverified_exit", settlement_source="level_fallback")
    assert db.stats()["consecutive_losses"] == 0, \
        "an estimated loss must not count as a real one in the streak"


def test_the_unknown_exit_is_not_labelled_as_a_stop_or_target(tmp_path):
    src = (ROOT / "engine.py").read_text()
    assert 'reason = "unverified_exit"' in src, \
        "an unattributed exit must not be recorded as a stop-out or a take-profit"
    assert 'settle["settlement_source"] = "level_fallback"' in src


# ── 5. a shape match is a candidate, never ownership ───────────────────────

def test_a_shape_matched_stop_is_never_recorded_as_ours(tmp_path):
    eng, client, state, db, alerts = _rig(tmp_path)
    state.set_flag("possible_orphan_stop", {"contracts": 0.277, "trigger": 1.3338,
                                            "side": "sell", "at": 0})
    client.orders = [{"id": "maybe-mine", "type": "stop", "side": "sell", "amount": 0.277,
                      "stopPrice": 1.3338, "reduceOnly": True, "status": "open", "info": {}}]
    assert eng.reconcile_possible_orphan_stop() == "candidate"
    assert "maybe-mine" not in (state.get_flag("placed_ids") or []), \
        "a candidate must never be remembered as Mel-owned"
    assert (state.get_flag("possible_orphan_stop") or {}).get("candidate_id") == "maybe-mine"


def test_the_tick_keeps_blocking_on_a_candidate(tmp_path):
    """A candidate is not a resolution: entries stay blocked until a human decides."""
    src = (ROOT / "run_bot.py").read_text()
    assert '"unresolved", "unresolved_blind", "candidate"' in src


def test_the_explicit_adoption_path_exists_and_is_armed_only(tmp_path):
    src = (ROOT / "run_bot.py").read_text()
    assert "--adopt-orphan-stop" in src
    assert "needs an ARMED bot" in src
