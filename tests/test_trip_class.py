"""The Trip failure class — regression suite.

Trip (Binance XRPUSDT) lost profits to protective-stop failures in four distinct ways. Mel must
be immune to all four, and each one gets a test here:

  1. a stop that is ACCEPTED and then silently expires  → entry confirmation + per-tick re-check
  2. a stop that is gone and could NOT be restored      → fail closed, and the row is RECORDED
  3. re-entering into a broken protection path          → a stop-failure streak halts entries
  4. a force-close that bypasses the guard rails        → the failure row must reach the stats

Every test asserts the position is never left naked without the bot saying so.
"""

import json
import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as C            # noqa: E402
import exchange as ex         # noqa: E402
import guards as G            # noqa: E402
import run_bot                # noqa: E402
from db import TradeDB        # noqa: E402
from engine import Engine     # noqa: E402
from state import State       # noqa: E402
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


def _enter(eng):
    eng.try_enter(long_decision(), _size(eng))
    return eng.state.get_open()


def _vanish_stop(client, oid):
    """The venue drops the order: it is no longer resting and no longer answerable by id."""
    client.orders = [o for o in client.orders if str(o.get("id")) != str(oid)]
    client.placed = [o for o in client.placed if str(o.get("id")) != str(oid)]


# ── 1. the stop vanishes mid-position ───────────────────────────────────────

def test_a_stop_that_vanishes_mid_position_is_restored(tmp_path):
    """Trip's profit drain: accepted at entry, gone later, position naked and unwatched."""
    eng, client, state, db, alerts = _rig(tmp_path)
    pos = _enter(eng)
    old_id = pos["sl_order_id"]
    _vanish_stop(client, old_id)
    eng.manage(None)
    pos = state.get_open()
    assert pos, "the position must still be managed"
    assert pos["sl_order_id"] and pos["sl_order_id"] != old_id, \
        "a NEW stop id must be tracked after restoration"
    assert any(o["type"] == "stop" for o in client.orders), "no resting stop on the book"
    assert pos["sl_order_id"] in (state.get_flag("placed_ids") or [])
    assert any("NO LONGER RESTING" in a for a in alerts)


def test_the_restored_stop_is_verified_before_the_bot_believes_it(tmp_path):
    """A re-place that the venue accepts but that does not verify must NOT be recorded."""
    class StopDoesNotStick(FakeKraken):
        def create_order(self, symbol, otype, side, amount, price=None, params=None):
            o = super().create_order(symbol, otype, side, amount, price, params)
            if otype == "stop":
                # accepted, then immediately expired by the venue (Trip #2 exactly: the order
                # is acknowledged and silently stops being a stop)
                self.orders = [x for x in self.orders if x["id"] != o["id"]]
                for rec in self.placed:
                    if rec["id"] == o["id"]:
                        rec["status"] = "expired"
            return o

    eng, client, state, db, alerts = _rig(tmp_path, client=StopDoesNotStick())
    pos = _enter(eng)
    assert pos is None, "an unverifiable stop must abort the entry, not hold the position"
    assert state.is_flat(), "an unverifiable stop must not leave a live position"
    assert client.position is None, "the venue must not be left holding anything"


# ── 2. it cannot be restored → fail closed, and record it ──────────────────

def test_a_stop_that_cannot_be_restored_fails_closed_and_is_recorded(tmp_path):
    """[MONEY] Trip #11: the force-close path returned before recording, so the churn was
    invisible to every DB-backed guard. The failure path must not bypass the caps."""
    class RefusesRestore(FakeKraken):
        def create_order(self, symbol, otype, side, amount, price=None, params=None):
            if otype == "stop" and len(self.placed) > 1:
                raise RuntimeError("venue refused the replacement stop")
            return super().create_order(symbol, otype, side, amount, price, params)

    eng, client, state, db, alerts = _rig(tmp_path, client=RefusesRestore())
    pos = _enter(eng)
    _vanish_stop(client, pos["sl_order_id"])
    out = eng.manage(None)
    assert out.get("status") in ("closed", "closing", "error"), out
    assert client.position is None, "fail closed means the position is closed"
    assert state.is_flat()
    row = db.stats()
    assert row["trades_today"] >= 1, "the failure close must be visible to the daily cap"
    assert row["sl_failure_streak"] >= 1, \
        "a protective-stop failure must be visible to the protection breaker"
    with sqlite3.connect(tmp_path / "t.db") as con:
        reason = con.execute("SELECT exit_reason FROM trades WHERE status='closed' "
                             "ORDER BY id DESC LIMIT 1").fetchone()[0]
    assert reason == "sl_missing_unprotected", reason


def test_restores_are_bounded_then_the_position_is_closed(tmp_path):
    """A venue that keeps dropping our stops is not protection — stop holding it naked."""
    eng, client, state, db, alerts = _rig(tmp_path)
    pos = _enter(eng)
    for _ in range(4):
        pos = state.get_open()
        if not pos:
            break
        _vanish_stop(client, pos["sl_order_id"])
        eng.manage(None)
    assert state.is_flat(), "after the restore bound the position must be closed"
    assert any("lost its protective stop" in a for a in alerts)


# ── 3. an unaskable venue must not produce a duplicate ─────────────────────

def test_an_unaskable_venue_never_stacks_a_duplicate_stop(tmp_path):
    """'I could not ask' is not 'it is gone'. The safe direction here is to hold and say so."""
    class Blind(FakeKraken):
        def fetch_order(self, oid, symbol=None):
            raise RuntimeError("network unreachable")

        def fetch_open_orders(self, symbol=None, params=None):
            raise RuntimeError("network unreachable")

    eng, client, state, db, alerts = _rig(tmp_path, client=Blind())
    pos = _enter(eng)
    before = len([o for o in client.orders if o["type"] == "stop"])
    out = eng.manage(None)
    assert out["status"] == "holding", out
    after = len([o for o in client.orders if o["type"] == "stop"])
    assert after == before, "a stop was stacked while the venue could not be asked"
    assert any("cannot be verified" in a for a in alerts)


def test_a_resting_but_invalid_stop_is_replaced(tmp_path):
    """Exists but does not protect (wrong size/side/reduce-only) is as bad as missing."""
    eng, client, state, db, alerts = _rig(tmp_path)
    pos = _enter(eng)
    for o in client.orders:
        if str(o["id"]) == str(pos["sl_order_id"]):
            o["amount"] = 1.0           # shrunk below the open size (one whole XRP)
    eng.manage(None)
    pos = state.get_open()
    assert any(o["type"] == "stop" and o["amount"] >= pos["remaining"] - 1e-9
               for o in client.orders), "the stop was not restored to the right size"


# ── 4. the churn breaker ────────────────────────────────────────────────────

def test_a_protection_failure_streak_halts_entries():
    cfg = json.loads(json.dumps(CFG))
    cfg["guards"]["max_consecutive_sl_failures"] = 2
    ok, why = G.g_sl_failure_streak(0, cfg)
    assert ok
    ok, why = G.g_sl_failure_streak(1, cfg)
    assert ok
    ok, why = G.g_sl_failure_streak(2, cfg)
    assert not ok and "protective-stop failure streak" in why


def test_a_protection_failure_outranks_a_perfect_signal():
    """Trip burned ~$226 re-entering. A broken safety path must veto any market reason."""
    cfg = json.loads(json.dumps(CFG))
    from types import SimpleNamespace
    dec = long_decision()
    market = SimpleNamespace(htf={"trend": "up"}, rsi=52.0, bb_pct_b=48.0, macd_hist=0.5)
    res = G.run_guards(dec, market,
                       {"trades_today": 0, "consecutive_losses": 0, "pnl_today_pct": 0.0,
                        "pnl_total_pct": 0.0, "pnl_today_equity_pct": 0.0,
                        "pnl_total_equity_pct": 0.0, "candles_since_close": 99,
                        "sl_failure_streak": 3}, cfg)
    assert not res.passed, res.reason
    assert "protective-stop" in res.reason


def test_a_clean_stop_is_re_verified_without_side_effects(tmp_path):
    """The per-tick check must be quiet on a healthy position (no churn from the check)."""
    eng, client, state, db, alerts = _rig(tmp_path)
    pos = _enter(eng)
    before = list(client.placed)
    alerts.clear()                      # the entry alert is expected; the TICKS must be silent
    for _ in range(3):
        eng.manage(None)
    assert client.placed == before, "the tick placed something on a healthy position"
    assert state.get_open()["sl_order_id"] == pos["sl_order_id"]
    assert alerts == [], alerts
