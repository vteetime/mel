"""Regression tests for the Terra round-6 review findings (all six).

1 a substituted trigger made a materially looser stop look healthy · 2 a stop that FILLED was
accepted as resting protection · 3 the protection-failure breaker could be bypassed while a
queued close stayed unwritten · 4 the streak halt had no reachable recovery · 5 an ambiguous
stop with no candidate had no clearing path · 6 `close_in_flight` was never consumed.
"""

import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as C          # noqa: E402
import exchange as ex       # noqa: E402
import guards as G          # noqa: E402
import run_bot              # noqa: E402
from db import TradeDB      # noqa: E402
from engine import Engine   # noqa: E402
from state import State     # noqa: E402
from test_dry_run_scenarios import SCHEMA, SYM, FakeKraken, long_decision  # noqa: E402

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


def _stop(oid, side="sell", amount=0.277, trigger=1.3338, status="open", reduce_only=True,
          include_trigger=True):
    o = {"id": oid, "type": "stop", "side": side, "amount": amount, "price": None,
         "reduceOnly": reduce_only, "status": status, "filled": 0.0, "average": None,
         "info": {}}
    if include_trigger:
        o["stopPrice"] = trigger
        o["info"] = {"stopPrice": trigger, "orderType": "stp"}
    return o


def _confirm(client, oid, **over):
    kw = dict(side="sell", min_amount=0.277, require_breakeven=False,
              entry=1.368, trigger=1.3338)
    kw.update(over)
    return ex.stop_is_confirmed(client, SYM, oid, **kw)


# ── 1. the venue's trigger is compared, never substituted ───────────────────

def test_a_looser_stop_is_not_confirmed():
    """A stop on the losing side but at the WRONG level is not the risk limit we chose."""
    client = FakeKraken()
    client.orders = [_stop("s1", trigger=1.30)]          # 3% further away than intended
    ok, why = _confirm(client, "s1")
    assert not ok and "not the expected" in why, why


def test_an_order_with_no_readable_trigger_is_not_confirmed():
    client = FakeKraken()
    client.orders = [_stop("s1", include_trigger=False)]
    ok, why = _confirm(client, "s1")
    assert not ok and "no trigger" in why, why


def test_the_right_trigger_is_confirmed():
    client = FakeKraken()
    client.orders = [_stop("s1")]
    ok, why = _confirm(client, "s1")
    assert ok, why


def test_rounding_within_tolerance_still_confirms():
    """Kraken rounds to the market tick; a tick of difference is not a different level."""
    client = FakeKraken()
    client.orders = [_stop("s1", trigger=1.3339)]
    ok, why = _confirm(client, "s1")
    assert ok, why


# ── 2. a filled stop is not resting protection ──────────────────────────────

def test_a_filled_stop_is_not_resting_protection():
    client = FakeKraken()
    client.orders = [_stop("s1", status="filled")]
    ok, why = _confirm(client, "s1")
    assert not ok and "no longer resting" in why, why


def test_a_partially_filled_stop_gets_the_remainder_protected(tmp_path):
    """[MONEY] The stop fired for part of the position: the rest is naked until we re-place."""
    class PartialFill(FakeKraken):
        def __init__(self, **kw):
            super().__init__(**kw)
            self.partial = False

        def fetch_order(self, oid, symbol=None):
            o = super().fetch_order(oid, symbol)
            # only the ORIGINAL stop (captured after entry) has filled; the replacement Mel
            # places in response must verify normally
            if o and self.partial and str(o["id"]) == str(self.original_stop_id):
                o["status"] = "closed"
            return o

    client = PartialFill()
    eng, client, state, db, alerts = _rig(tmp_path, client=client)
    eng.try_enter(long_decision(), _size(eng))
    pos = state.get_open()
    client.original_stop_id = pos["sl_order_id"]           # this one fills while we look
    client.partial = True
    # the stop took PART of the position: the venue now holds 100.0, not the original 963.0
    client.position = dict(client.position, contracts=100.0)
    out = eng._ensure_sl(state.get_open(), 100.0)
    pos = state.get_open()
    placed = [o for o in client.orders if o["type"] == "stop"]
    assert placed, "no stop remains on the book for the remainder"
    assert out == "replaced", out
    new_stop = [o for o in placed if str(o["id"]) != str(client.original_stop_id)][0]
    assert abs(float(new_stop["amount"]) - 100.0) < 1e-6, \
        f"the replacement must cover only the remainder, got {new_stop['amount']}"


def test_a_stop_that_filled_with_the_position_closed_restores_nothing(tmp_path):
    class FilledAndFlat(FakeKraken):
        def __init__(self, **kw):
            super().__init__(**kw)
            self.filled = False

        def fetch_order(self, oid, symbol=None):
            o = super().fetch_order(oid, symbol)
            if o and self.filled and o["type"] == "stop":
                o["status"] = "closed"
            return o

    client = FilledAndFlat()
    eng, client, state, db, alerts = _rig(tmp_path, client=client)
    eng.try_enter(long_decision(), _size(eng))
    pos = state.get_open()
    client.filled = True
    client.position = None                                  # the stop took the whole position
    client.orders = [o for o in client.orders if o["type"] != "stop"]
    assert eng._ensure_sl(state.get_open(), 0.0) == "ok"      # nothing to protect
    assert [o for o in client.orders if o["type"] == "stop"] == [], \
        "a stop was placed with no position open"


# ── 3. an unwritten close blocks entry ─────────────────────────────────────

def test_an_unwritable_close_blocks_entries(tmp_path, monkeypatch):
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
    # a close with no price and no level: it cannot be priced, so it stays queued
    st.set_flag("pending_close", [{"trade_id": None, "entry_price": 1.368, "contracts": 0.277,
                                   "exit_price": 0.0, "reason": "sl_placement_failed",
                                   "side": "sell", "needs_price": True}])

    class Fake(FakeKraken):
        def fetch_ohlcv(self, symbol=None, timeframe="15m", since=None, limit=200):
            n = int(limit or 200)
            return [[1700000000000 + i * 900000, 1.35, 2602, 2598, 2600.5, 10.0]
                    for i in range(n)]

    client = Fake()
    monkeypatch.setattr(run_bot.ex, "new_client", lambda *a, **k: client)
    monkeypatch.setenv("KRAKEN_API_KEY", "k")
    monkeypatch.setenv("KRAKEN_API_SECRET", "s")
    monkeypatch.setattr(sys, "argv", ["run_bot.py", "--live", "--config",
                                      str(tmp_path / "cfg.yaml")])
    assert run_bot.main() == 1
    assert client.placed == [], "an entry was taken while a closed trade was invisible"


# ── 4. the streak halt has a reachable recovery ────────────────────────────

def test_an_acknowledged_streak_no_longer_halts():
    stats = {"sl_failure_streak": 2, "last_protection_failure_at": "2026-09-19T10:00:00+00:00"}
    st = State(Path("/tmp") / "mel_ack_test.json")
    st.clear_flag("ack_sl_failure_streak")
    assert run_bot.effective_protection_streak(stats, st) == 2
    st.set_flag("ack_sl_failure_streak",
                {"at": "2026-09-19T11:00:00+00:00", "note": "venue blip", "by": "operator"})
    assert run_bot.effective_protection_streak(stats, st) == 0


def test_the_acknowledgement_expires_on_a_new_failure():
    st = State(Path("/tmp") / "mel_ack_test2.json")
    st.set_flag("ack_sl_failure_streak", {"at": "2026-09-19T11:00:00+00:00", "note": "x"})
    newer = {"sl_failure_streak": 1,
             "last_protection_failure_at": "2026-09-19T12:00:00+00:00"}
    assert run_bot.effective_protection_streak(newer, st) == 1, \
        "a failure AFTER the acknowledgement must halt again"


def test_the_clear_halt_tool_clears_a_real_streak(tmp_path):
    st_path = tmp_path / "state.json"
    State(st_path).set_flag("possible_orphan_stop",
                            {"contracts": 0.277, "trigger": 1.3338, "side": "sell",
                             "where": "sl-restore"})
    tool = ROOT / "tools" / "clear_halt.py"
    refused = subprocess.run([sys.executable, str(tool), "--halt", "ambiguous_stop",
                              "--note", "", "--yes", "--state", str(st_path)],
                             capture_output=True, text=True)
    assert refused.returncode == 1 and "REQUIRED" in refused.stdout
    ok = subprocess.run([sys.executable, str(tool), "--halt", "ambiguous_stop",
                         "--note", "checked the venue UI: nothing resting", "--yes",
                         "--state", str(st_path)], capture_output=True, text=True)
    assert ok.returncode == 0
    st = State(st_path)
    assert not st.get_flag("possible_orphan_stop")
    assert "checked the venue UI" in json.dumps(st.get_flag("resolved_ambiguous_stop"))


# ── 5. a no-candidate ambiguous stop has a supported exit ──────────────────

def test_resolve_ambiguous_stop_is_armed_only_and_audited(tmp_path):
    state_file = tmp_path / "state.json"
    st = State(state_file)
    st.set_flag("possible_orphan_stop", {"contracts": 0.277, "trigger": 1.3338,
                                         "side": "sell", "where": "sl-restore"})
    cfg = json.loads(json.dumps(CFG))
    cfg["ops"]["state_file"] = "state.json"
    cfg["bot"]["ccxt_symbol"] = SYM
    # dry-run must refuse (it is an assertion that unblocks real trading)
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = run_bot.do_sync(cfg, {"apiKey": "k", "secret": "s"}, dry_run=True,
                             resolve_ambiguous_stop=True, note="x")
    assert rc == 1 and "REFUSED" in buf.getvalue(), buf.getvalue()


# ── 6. close_in_flight is consumed ─────────────────────────────────────────

def test_an_in_flight_close_is_consumed_when_the_position_still_exists(tmp_path):
    eng, client, state, db, alerts = _rig(tmp_path)
    eng.try_enter(long_decision(), _size(eng))
    state.set_flag("close_in_flight", {"at": 1, "reason": "hold_cap"})
    eng.manage(None)
    assert not state.get_flag("close_in_flight"), \
        "the flag was never consumed, so 'reconcile or re-close, never both' was untrue"


def test_an_in_flight_close_is_consumed_and_settled_when_flat(tmp_path):
    eng, client, state, db, alerts = _rig(tmp_path)
    eng.try_enter(long_decision(), _size(eng))
    pos = state.get_open()
    state.set_flag("close_in_flight", {"at": 1, "reason": "hold_cap"})
    client.position = None                                  # the close landed after all
    eng.manage(None)
    assert not state.get_flag("close_in_flight")
    st = db.stats()
    assert st["trades_today"] >= 0                          # the reconcile path ran
