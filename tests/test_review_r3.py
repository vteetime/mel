"""Regression tests for the Terra round-3 review findings.

Findings: 1 incomplete trigger visibility treated as clean · 2 mutations blindly retried
after ambiguous transport failures · 3 armed sync cleanup cancelled unproven orders ·
4 the force-close queue created a SECOND lifecycle · 5 zero-fill settlement could deadlock.
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
import sync as sync_mod      # noqa: E402
from db import TradeDB       # noqa: E402
from engine import Engine    # noqa: E402
from state import State      # noqa: E402
from test_dry_run_scenarios import SCHEMA, SYM, FakeKraken, long_decision  # noqa: E402

CFG = C.load_config()


def _rig(tmp_path, client=None, dry_run=False, db=None, cfg=None):
    client = client or FakeKraken()
    state = State(tmp_path / "state.json")
    if db is None:
        con = sqlite3.connect(tmp_path / "t.db")
        con.executescript(SCHEMA)
        con.commit()
        con.close()
        db = TradeDB(tmp_path / "t.db")
    alerts = []
    eng = Engine(cfg or CFG, client, state, db, alerts.append, dry_run=dry_run)
    return eng, client, state, db, alerts


def _size(eng, funds=732.09, price=1.368, confidence=8, scale=1.0):
    return C.size_position(funds, price, CFG, eng.spec["contract_size"],
                           eng.spec["amount_step"], confidence=confidence,
                           extra_scale=scale)


# ── finding 2: never blind-retry a mutation ─────────────────────────────────

class TimeoutThenAccept(FakeKraken):
    """A venue that accepts the order but does not answer — twice, if asked twice."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.attempts = 0

    def create_order(self, symbol, otype, side, amount, price=None, params=None):
        self.attempts += 1
        if self.attempts == 1 or True:
            import ccxt
            raise ccxt.RequestTimeout("no answer (the order may have been accepted)")
        return super().create_order(symbol, otype, side, amount, price, params)


def test_a_mutation_is_submitted_once_and_ambiguity_is_raised(tmp_path):
    b = TimeoutThenAccept()
    with pytest.raises(ex.AmbiguousSubmission):
        ex.place_market(b, SYM, "buy", 0.2)
    assert b.attempts == 1, "the order was submitted more than once"
    assert b.placed == [], "a retry actually placed an order"


def test_reads_still_retry(tmp_path, monkeypatch):
    calls = {"n": 0}
    import ccxt

    def flaky():
        calls["n"] += 1
        if calls["n"] == 1:
            raise ccxt.NetworkError("blip")
        return "ok"

    monkeypatch.setattr(ex.time, "sleep", lambda *_: None)
    assert ex.retry(flaky, attempts=3) == "ok" and calls["n"] == 2


def test_ambiguous_entry_adopts_the_fill_instead_of_doubling_it(tmp_path):
    """[MONEY] The submit timed out but the venue holds the position: protect it, do not
    send a second order."""
    class LandedButSilent(FakeKraken):
        def create_order(self, symbol, otype, side, amount, price=None, params=None):
            import ccxt
            if otype == "market":
                # the order lands at the venue, then the response is lost
                super().create_order(symbol, otype, side, amount, price, params or {})
                raise ccxt.RequestTimeout("no answer")
            return super().create_order(symbol, otype, side, amount, price, params)

    client = LandedButSilent()
    eng, client, state, db, alerts = _rig(tmp_path, client=client)
    out = eng.try_enter(long_decision(), _size(eng))
    assert out["status"] == "adopted", out
    assert client.position is not None, "the fill was thrown away"
    assert len([o for o in client.placed if o["type"] == "market" and not o["reduceOnly"]]) == 1, \
        "a second entry order was submitted"
    assert state.get_open()["sl_order_id"], "the adopted fill must be protected"


def test_ambiguous_entry_that_did_not_land_is_reported_not_resubmitted(tmp_path):
    class NeverLanded(FakeKraken):
        def create_order(self, symbol, otype, side, amount, price=None, params=None):
            import ccxt
            raise ccxt.RequestTimeout("no answer")

    client = NeverLanded()
    eng, client, state, db, alerts = _rig(tmp_path, client=client)
    out = eng.try_enter(long_decision(), _size(eng))
    assert out["status"] == "error" and "not confirmed" in out["message"]
    assert client.placed == [] and state.is_flat()


# ── finding 1: an incomplete order view is never reported as clean ──────────

def test_sync_reports_an_incomplete_stop_view_as_unverified(tmp_path):
    class NoTriggers(FakeKraken):
        def fetch_open_orders(self, symbol=None, params=None):
            if params and params.get("stop"):
                raise RuntimeError("authenticationError")
            return [dict(o) for o in self.orders]

    b = NoTriggers()
    snap = sync_mod.snapshot(b, SYM)
    assert snap["triggers_listable"] is False
    verdict = sync_mod.reconcile(snap, None)
    assert verdict["stop_coverage_verified"] is False
    assert verdict["action"] == "none_unverified"
    report = sync_mod.format_report(snap, verdict, CFG)
    assert "REFUSED by this venue/key" in report
    assert "VERDICT: NONE_UNVERIFIED" in report
    assert "UNVERIFIED" in report


def test_stop_visibility_is_learned_from_our_own_resting_stop(tmp_path):
    """The listing's trigger coverage is a question only evidence can answer, and the
    evidence is our own stop: it is asked for the moment one is resting."""
    client = FakeKraken()

    class NoTriggers(FakeKraken):
        """Like this venue: the account listing answers, the trigger query is refused, and
        the stop does NOT appear in what can be read."""
        def privateGetOpenorders(self, params=None):
            return {"openOrders": []}

        def fetch_open_orders(self, symbol=None, params=None):
            if params and params.get("stop"):
                raise RuntimeError("authenticationError")
            return []

    client = NoTriggers()
    eng, client, state, db, alerts = _rig(tmp_path, client=client)
    eng.try_enter(long_decision(), _size(eng))
    assert state.get_flag("stop_listing_covers_triggers") is False
    assert any("does NOT show our own" in a for a in alerts)


def test_stop_visibility_confirmed_when_the_listing_carries_it(tmp_path):
    client = FakeKraken()
    eng, client, state, db, alerts = _rig(tmp_path, client=client)
    eng.try_enter(long_decision(), _size(eng))
    assert state.get_flag("stop_listing_covers_triggers") is True
    assert not any("does NOT show" in a for a in alerts)


# ── finding 4: one force-close, one lifecycle ───────────────────────────────

def test_force_close_queue_completes_the_original_row(tmp_path):
    """[MONEY] An earlier queue carried `trade_id: None`, so the retry created a SECOND
    entry row and closed that one — leaving the real entry open forever."""
    class FailsExitOnce(TradeDB):
        fails = 1

        def record_exit(self, *a, **k):
            if type(self).fails:
                type(self).fails -= 1
                return False
            return super().record_exit(*a, **k)

    con = sqlite3.connect(tmp_path / "t.db")
    con.executescript(SCHEMA)
    con.commit()
    con.close()
    db = FailsExitOnce(tmp_path / "t.db")
    client = FakeKraken(fail_orders={"stop"})
    state = State(tmp_path / "state.json")
    eng = Engine(CFG, client, state, db, lambda m: None, dry_run=False)
    eng.try_enter(long_decision(), _size(eng))

    queued = state.get_flag("pending_close") or []
    assert len(queued) == 1 and queued[0]["trade_id"], queued

    out = eng.retry_pending_close()
    assert out["written"] == 1
    rows = db._conn.execute("SELECT id, status, exit_reason, pnl_net FROM trades").fetchall()
    assert len(rows) == 1, f"expected exactly one lifecycle, got {len(rows)}: " \
                           f"{[(r['id'], r['status']) for r in rows]}"
    assert rows[0]["status"] == "closed"
    assert rows[0]["exit_reason"] == "sl_placement_failed"
    assert db._conn.execute("SELECT COUNT(*) c FROM trades WHERE status='open'"
                            ).fetchone()["c"] == 0


def test_retry_writes_the_settlement_it_was_given_not_a_recomputation(tmp_path):
    class FailsExitOnce(TradeDB):
        fails = 1

        def record_exit(self, *a, **k):
            if type(self).fails:
                type(self).fails -= 1
                return False
            return super().record_exit(*a, **k)

    con = sqlite3.connect(tmp_path / "t.db")
    con.executescript(SCHEMA)
    con.commit()
    con.close()
    db = FailsExitOnce(tmp_path / "t.db")
    client = FakeKraken(fail_orders={"stop"})
    state = State(tmp_path / "state.json")
    eng = Engine(CFG, client, state, db, lambda m: None, dry_run=False)
    eng.try_enter(long_decision(), _size(eng))
    rec = (state.get_flag("pending_close") or [])[0]
    eng.retry_pending_close()
    row = db._conn.execute("SELECT exit_price, exit_reason, settlement_source FROM trades"
                           ).fetchone()
    assert row["exit_price"] == pytest.approx(float(rec["exit_price"])), \
        "the retry recomputed the exit price instead of using the settlement it was given"


# ── finding 3: destructive sync cleanup is ownership-scoped ─────────────────

def test_sync_cleanup_cancels_only_persisted_ids(tmp_path):
    client = FakeKraken()
    client.orders = [
        {"id": "ours", "type": "stop", "side": "sell", "amount": 0.2,
         "stopPrice": 1.30, "reduceOnly": True, "status": "open", "info": {}},
        {"id": "foreign", "type": "stop", "side": "sell", "amount": 0.2,
         "stopPrice": 2400.0, "reduceOnly": True, "status": "open", "info": {}},
    ]
    snap = sync_mod.snapshot(client, SYM)
    out = sync_mod.cancel_dangling(client, snap, SYM, owned_ids=["ours"])
    assert out["cancelled"] == {"ours": True}
    assert "foreign" in out.get("left_alone", {})
    assert [o["id"] for o in client.orders] == ["foreign"]

    refused = sync_mod.cancel_dangling(client, snap, SYM)
    assert "refused" in refused and refused["cancelled"] == {}
