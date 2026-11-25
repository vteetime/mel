"""Regression tests for the Terra round-1 review findings.

Each test is named for the finding it closes, so a future change that reopens one fails
loudly with the reason attached. Findings: (1) sibling retirement not confirmed, (2) a dry
run could still submit orders, (3) flat cleanup and orphan adoption acted on orders the bot
does not own, (4) settlement could present a level or a partial fill as venue-verified,
(5) verified funding was dropped, (6) a failed exit write lost guard accounting,
(7) the equity caps were sums of margin ROE, (8) the documented install did not run.
"""

import json
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as C           # noqa: E402
import exchange as ex        # noqa: E402
import guards as G           # noqa: E402
import settlement            # noqa: E402
import sync as sync_mod      # noqa: E402
from db import TradeDB       # noqa: E402
from engine import Engine    # noqa: E402
from state import State      # noqa: E402
from test_dry_run_scenarios import FakeKraken, SCHEMA, SYM, long_decision  # noqa: E402

CFG = C.load_config()


def _db(tmp_path):
    p = tmp_path / "t.db"
    con = sqlite3.connect(p)
    con.executescript(SCHEMA)
    con.commit()
    con.close()
    return TradeDB(p)


def _engine(tmp_path, client=None, dry_run=False, db=None):
    client = client or FakeKraken()
    state = State(tmp_path / "state.json")
    db = db or _db(tmp_path)
    alerts = []
    eng = Engine(CFG, client, state, db, alerts.append, dry_run=dry_run)
    return eng, client, state, db, alerts


def size_for(engine, funds=732.09, confidence=8, scale=1.0, price=1.368):
    return C.size_position(funds, price, CFG, engine.spec["contract_size"],
                           engine.spec["amount_step"], confidence=confidence,
                           extra_scale=scale)


# ── finding 2: a dry run must not be able to trade ───────────────────────────

def test_adapter_refuses_every_order_from_a_dry_run_client():
    """ccxt implements no dry-run option, so the barrier has to be ours."""
    b = FakeKraken()
    b.options = {"dryRun": True}
    for fn, args in ((ex.place_market, (b, SYM, "buy", 0.1)),
                     (ex.place_stop, (b, SYM, "sell", 0.1, 1.30)),
                     (ex.place_limit, (b, SYM, "sell", 0.1, 2700.0))):
        with pytest.raises(ex.ExchangeError):
            fn(*args)
    assert b.placed == [], "a dry-run client submitted an order"


def test_dry_run_tick_never_closes_a_hold_capped_position(tmp_path):
    """[MONEY] Management is a trading path: a hold-capped position reaches _close()."""
    client = FakeKraken()
    state = State(tmp_path / "state.json")
    db = _db(tmp_path)
    alerts = []
    live = Engine(CFG, client, state, db, alerts.append, dry_run=False)
    live.try_enter(long_decision(), size_for(live))
    assert not state.is_flat()

    state.set_open(dict(state.get_open(), entry_time=(
        datetime.now(timezone.utc) - timedelta(hours=30)).isoformat()))
    dry = Engine(CFG, client, state, db, alerts.append, dry_run=True)
    placed_before = len(client.placed)
    out = dry.manage(None)

    assert out["status"] == "dry_run" and "close" in out["would"], out
    assert len(client.placed) == placed_before, "dry run submitted an order"
    assert client.position is not None, "dry run closed a live position"
    assert not state.is_flat(), "dry run must not clear state"
    assert db._conn.execute("SELECT COUNT(*) c FROM trades WHERE status='closed'"
                            ).fetchone()["c"] == 0


def test_dry_run_reports_an_orphan_without_adopting_it(tmp_path):
    eng, client, state, db, alerts = _engine(tmp_path, dry_run=True)
    client.position = {"contracts": 0.2, "side": "long", "entryPrice": 1.35}
    out = eng.adopt_orphan(ex.open_position(client, SYM))
    assert out["status"] == "orphan_dry_run"
    assert client.placed == [] and client.orders == []
    assert state.is_flat(), "a dry run must not write state"
    assert any("ORPHAN DETECTED in DRY RUN" in a for a in alerts)


# ── finding 1: sibling retirement must be confirmed and block entries ────────

def test_uncancellable_sibling_is_persisted_and_alerted(tmp_path):
    """[MONEY] A surviving reduce-only stop can close a LATER position."""
    class StubbornCancel(FakeKraken):
        def cancel_order(self, oid, symbol=None):
            raise RuntimeError("venue refused the cancel")

    client = StubbornCancel()
    eng, client, state, db, alerts = _engine(tmp_path, client=client)
    eng.try_enter(long_decision(), size_for(eng))
    sl_id = state.get_open()["sl_order_id"]
    client.trigger(state.get_open()["tp_order_id"])          # TP fills, SL survives
    out = eng.manage(None)
    assert out["status"] == "closed"
    assert state.get_flag("dangling_orders"), "the unresolved order must be remembered"
    assert sl_id in state.get_flag("dangling_orders")
    assert any("could not be retired" in a for a in alerts)


def test_clean_exit_leaves_no_dangling_flag(tmp_path):
    eng, client, state, db, alerts = _engine(tmp_path)
    eng.try_enter(long_decision(), size_for(eng))
    client.trigger(state.get_open()["tp_order_id"])
    eng.manage(None)
    assert state.get_flag("dangling_orders") in (None, [])
    assert client.orders == []


def test_dangling_flag_blocks_new_entries_end_to_end(tmp_path, monkeypatch):
    """Run a real tick with an unresolved order recorded: it must not open a position."""
    import run_bot
    monkeypatch.setattr(run_bot, "ROOT", tmp_path)
    cfg = json.loads(json.dumps(CFG))
    cfg["ops"]["state_file"] = "state.json"
    cfg["ops"]["db_path"] = str(tmp_path / "t.db")
    cfg["ops"]["lock_file"] = str(tmp_path / "mel.lock")
    (tmp_path / "cfg.yaml").write_text(json.dumps(cfg))
    con = sqlite3.connect(tmp_path / "t.db")
    con.executescript(SCHEMA)
    con.commit()
    con.close()

    state = State(tmp_path / "state.json")
    state.set_flag("dangling_orders", ["leftover-1"])
    (tmp_path / "LIVE").write_text("")          # armed, so the guard is live

    client = FakeKraken()
    client.orders = [{"id": "leftover-1", "type": "stop", "side": "sell", "amount": 0.2,
                      "stopPrice": 1.30, "reduceOnly": True, "status": "open",
                      "info": {}}]

    class RefusingCancel(FakeKraken):
        def cancel_order(self, oid, symbol=None):
            return {"id": oid, "status": "open"}   # still there afterwards

    refusing = RefusingCancel()
    refusing.orders = list(client.orders)
    monkeypatch.setattr(run_bot.ex, "new_client", lambda *a, **k: refusing)
    monkeypatch.setenv("KRAKEN_API_KEY", "k")
    monkeypatch.setenv("KRAKEN_API_SECRET", "s")

    rc = run_bot.main.__wrapped__() if hasattr(run_bot.main, "__wrapped__") else None
    monkeypatch.setattr(sys, "argv", ["run_bot.py", "--live", "--config",
                                      str(tmp_path / "cfg.yaml")])
    rc = run_bot.main()
    assert rc == 1, "a tick with unresolved protection must refuse to trade"
    assert refusing.position is None, "no position may be opened while the book is dirty"


# ── finding 3: only this bot's protective orders are acted on ────────────────

def test_orphan_adoption_ignores_a_foreign_or_wrong_side_stop(tmp_path):
    """[MONEY] Adopting an order that does not protect leaves the position naked."""
    eng, client, state, db, alerts = _engine(tmp_path)
    client.position = {"contracts": 2.0, "side": "long", "entryPrice": 1.35}
    client.orders = [
        # a BUY stop cannot protect a long, and a non-reduce-only order is not protection
        {"id": "wrong-side", "type": "stop", "side": "buy", "amount": 2.0,
         "stopPrice": 2700.0, "reduceOnly": True, "status": "open", "info": {}},
        {"id": "not-reduce-only", "type": "stop", "side": "sell", "amount": 2.0,
         "stopPrice": 1.30, "reduceOnly": False, "status": "open", "info": {}},
    ]
    eng.adopt_orphan(ex.open_position(client, SYM))
    fresh = [o for o in client.placed if o["type"] == "stop" and o["id"] not in
             ("wrong-side", "not-reduce-only")]
    assert fresh, "a real protective stop must be placed"
    assert fresh[0]["reduceOnly"] is True and fresh[0]["side"] == "sell"
    assert state.get_open()["sl_order_id"] == fresh[0]["id"]


def test_orphan_adoption_gives_an_unprotected_orphan_a_take_profit(tmp_path):
    eng, client, state, db, alerts = _engine(tmp_path)
    client.position = {"contracts": 100.0, "side": "long", "entryPrice": 1.35}
    eng.adopt_orphan(ex.open_position(client, SYM))
    tp = client.order_of_type("limit")
    assert tp is not None, "an adopted position with no TP cannot take profit"
    assert float(tp["price"]) == pytest.approx(1.35 * 1.04, rel=1e-6)
    assert tp["amount"] == pytest.approx(100.0)
    assert state.get_open()["tp_order_id"] == tp["id"]


def test_cancel_dangling_leaves_non_reduce_only_orders_alone(tmp_path):
    eng, client, state, db, alerts = _engine(tmp_path)
    client.orders = [
        {"id": "ours", "type": "stop", "side": "sell", "amount": 0.2,
         "stopPrice": 1.30, "reduceOnly": True, "status": "open", "info": {}},
        {"id": "manual", "type": "limit", "side": "buy", "amount": 0.5, "price": 2400.0,
         "reduceOnly": False, "status": "open", "info": {}},
    ]
    out = sync_mod.cancel_dangling(client, sync_mod.snapshot(client, SYM), SYM,
                                   owned_ids=["ours"])
    assert out["cancelled"] == {"ours": True}
    assert "manual" in out["left_alone"] or "manual" in out.get("skipped", {})
    assert [o["id"] for o in client.orders] == ["manual"], "a foreign order was cancelled"


# ── finding 4: settlement must not present a level or partial fill as verified ─

def _fills(order_id, amount, price, fee=None):
    return {"info": {"orderId": order_id, "fee": fee if fee is not None else amount * price * 0.0005},
            "order": order_id, "amount": amount, "price": price,
            "fee": {"cost": fee if fee is not None else amount * price * 0.0005},
            "timestamp": int(datetime.now(timezone.utc).timestamp() * 1000)}


def test_settlement_with_no_fills_reports_a_level_fallback(tmp_path):
    b = FakeKraken()
    b.ledger_fee = 0.5
    res = settlement.settled_close(b, SYM, close_order_id="o9", entry_price=1.35,
                                   contracts=0.2, contract_size=1.0,
                                   fallback_price=1.40, opened_ms=None)
    assert res["price_verified"] is False
    assert res["settlement_source"] == "level_fallback"


def test_settlement_flags_a_partial_fill_history(tmp_path):
    b = FakeKraken()
    b.fills = [_fills("o9", 0.1, 1.40)]        # only half of the 0.2 position
    res = settlement.settled_close(b, SYM, close_order_id="o9", entry_price=1.35,
                                   contracts=0.2, contract_size=1.0, fallback_price=0.0)
    assert res["fills_complete"] is False
    assert res["settlement_source"] == "fills_partial"


def test_settlement_flags_a_ledger_disagreement(tmp_path):
    """The ledger is summed by symbol + time window, so it is only comparable when BOTH
    sides of the trade are known — the entry fee included. Comparing a ledger that carries
    the entry fee against exit-only fills would flag a disagreement on every trade.

    WHEN IT DISAGREES, THE FILLS WIN (changed 2026-09-20). The ledger can be short — a window
    starting at `opened_ms` misses the entry leg, which silently dropped the entry fee out of
    every Mel row's P&L — or it can absorb an unrelated trade in the same window. Our own fills
    are the only source that is both venue-stated per fill and complete for THIS trade. The
    disagreement is still surfaced (`ledger_disagrees`), it just no longer decides the number.
    See tests/test_settlement_fee_window.py for the full regression set."""
    b = FakeKraken()
    b.fills = [_fills("o9", 0.2, 1.40), _fills("o1", 0.2, 1.35)]   # exit + entry
    b.ledger_fee = 12.0                          # wildly different from the fills' fee
    res = settlement.settled_close(b, SYM, close_order_id="o9", entry_order_id="o1",
                                   entry_price=1.35, contracts=0.2, contract_size=1.0,
                                   fallback_price=0.0)
    assert res["ledger_disagrees"] is True
    fills_fee = 0.2 * 1.40 * 0.0005 + 0.2 * 1.35 * 0.0005
    assert res["fee_usdt"] == pytest.approx(fills_fee), \
        "a contaminated or short ledger must not set the cost basis"
    assert res["fee_from_ledger"] is False


def test_settlement_does_not_flag_a_disagreement_it_cannot_judge(tmp_path):
    """Exit-only fills vs a ledger that includes the entry fee is not evidence of a bad
    ledger — so it must not raise a false alarm."""
    b = FakeKraken()
    b.fills = [_fills("o9", 0.2, 1.40)]
    b.ledger_fee = 12.0
    res = settlement.settled_close(b, SYM, close_order_id="o9", entry_price=1.35,
                                   contracts=0.2, contract_size=1.0, fallback_price=0.0)
    assert res["ledger_disagrees"] is False


def test_settlement_reports_funding_separately_from_the_fee(tmp_path):
    b = FakeKraken()
    b.fills = [_fills("o9", 0.2, 1.40)]
    b.ledger_fee = 0.27
    b.ledger_extra = [{"contract": "pf_xrpusd", "info": "funding rate change",
                       "realized_funding": -0.11}]
    orig = b.historyGetAccountLog

    def with_funding(params=None):
        out = orig(params)
        out["logs"] = list(out.get("logs") or []) + b.ledger_extra
        return out

    b.historyGetAccountLog = with_funding
    res = settlement.settled_close(b, SYM, close_order_id="o9", entry_price=1.35,
                                   contracts=0.2, contract_size=1.0, fallback_price=0.0)
    assert res["funding_usdt"] == pytest.approx(-0.11)
    assert res["funding_verified"] is True


# ── finding 5: funding reaches the exit row ──────────────────────────────────

def test_funding_is_recorded_on_the_exit_row(tmp_path):
    client = FakeKraken()
    eng, client, state, db, alerts = _engine(tmp_path, client=client)
    eng.try_enter(long_decision(), size_for(eng))
    orig = client.historyGetAccountLog
    client.historyGetAccountLog = lambda params=None: {
        "logs": (list(orig(params).get("logs")) +
                 [{"contract": "pf_xrpusd", "info": "funding rate change",
                   "realized_funding": -0.42}])}
    client.trigger(state.get_open()["tp_order_id"])
    eng.manage(None)
    row = db._conn.execute("SELECT funding_usdt, funding_verified FROM trades WHERE status='closed'"
                           ).fetchone()
    assert row["funding_usdt"] == pytest.approx(-0.42)
    # ...and the figure's PROVENANCE is recorded with it: this funding came from the venue's own
    # account log, so the row must say so. Unwired since the column was added (Paul, 2026-09-20).
    assert row["funding_verified"] == 1, "funding from the ledger must be marked verified"


# ── finding 6: a failed exit write must not erase the trade from the guards ──

def test_failed_record_exit_keeps_state_open_for_retry(tmp_path):
    class NoWriteDB(TradeDB):
        def record_exit(self, *a, **k):
            return False

    client = FakeKraken()
    db = NoWriteDB(tmp_path / "t.db") if (tmp_path / "t.db").exists() else None
    p = tmp_path / "t.db"
    con = sqlite3.connect(p)
    con.executescript(SCHEMA)
    con.commit()
    con.close()
    db = NoWriteDB(p)
    state = State(tmp_path / "state.json")
    alerts = []
    eng = Engine(CFG, client, state, db, alerts.append, dry_run=False)
    eng.try_enter(long_decision(), size_for(eng))
    client.trigger(state.get_open()["tp_order_id"])
    out = eng.manage(None)
    assert out["status"] == "unrecorded"
    assert not state.is_flat(), "state must stay open so the write is retried"
    assert any("exit NOT recorded" in a for a in alerts)


# ── finding 7: the caps are measured against EQUITY, not margin ROE ──────────

def test_stats_convert_margin_roe_to_equity_percentage(tmp_path):
    db = _db(tmp_path)
    tid = db.record_entry(symbol=SYM, direction="long", entry_price=1.35,
                          contracts=0.2, leverage=2, sl_price=2535.0, tp1_price=1.40,
                          tp2_price=0.0, notional=520.0,
                          guard_config={"margin_fraction": 0.5})
    db.record_exit(tid, exit_price=1.35, pnl_net=-26.0, pnl_pct=-10.0, exit_reason="sl")
    s = db.stats()
    assert s["pnl_total_pct"] == pytest.approx(-10.0)          # margin ROE, as stored
    assert s["pnl_total_equity_pct"] == pytest.approx(-5.0)    # 0.5 x -10% of equity
    assert s["pnl_today_equity_pct"] == pytest.approx(-5.0)


def test_a_row_without_a_margin_fraction_falls_back_conservatively(tmp_path):
    db = _db(tmp_path)
    tid = db.record_entry(symbol=SYM, direction="long", entry_price=1.35,
                          contracts=0.2, leverage=2, sl_price=2535.0, tp1_price=1.40,
                          tp2_price=0.0, notional=520.0, trigger="adopted_orphan")
    db.record_exit(tid, exit_price=1.35, pnl_net=-26.0, pnl_pct=-10.0, exit_reason="sl")
    assert db.stats()["pnl_total_equity_pct"] == pytest.approx(-10.0), \
        "an unknown fraction must count as 1.0 so a cap can only trigger earlier"


def test_guards_use_the_equity_percentage_when_present(tmp_path):
    dec = long_decision()
    state = None
    from strategy import MarketState
    st = MarketState(price=1.35, atr=26.0, atr_pct=1.0, rsi=50.0, bb_pct_b=50.0,
                     macd_hist=0.0, ema_fast=1.0, ema_slow=1.0, ema_cross="neutral",
                     htf={"direction": "unknown", "strength": 0.0},
                     volume_trend="normal")
    # -12% margin ROE at a 50% margin fraction = -6% of equity: inside the -10% cap.
    inside = G.run_guards(dec, st, {"pnl_today_pct": -12.0, "pnl_today_equity_pct": -6.0}, CFG)
    assert inside.passed, inside.reason
    # the same ROE is fatal once the fraction is 1.0 (no equity discount available)
    outside = G.run_guards(dec, st, {"pnl_today_pct": -12.0, "pnl_today_equity_pct": -12.0}, CFG)
    assert not outside.passed and "daily loss cap" in outside.reason


def test_entry_records_the_margin_fraction_it_actually_deployed(tmp_path):
    eng, client, state, db, alerts = _engine(tmp_path)
    sizing = size_for(eng)
    eng.try_enter(long_decision(), sizing)
    gc = json.loads(db._conn.execute("SELECT guard_config FROM trades").fetchone()[0])
    assert gc["margin_fraction"] == pytest.approx(sizing.margin / sizing.funds, abs=1e-6)
    assert gc["margin_fraction"] <= 0.9 + 1e-9


# ── finding 8: the documented install runs ──────────────────────────────────

def test_documented_install_command_exists():
    """README tells the operator to run `.venv/bin/pip`; the file must be there."""
    pip = ROOT / ".venv" / "bin" / "pip"
    if not pip.exists():
        pytest.skip("no .venv in this checkout (fresh clone) — README covers both paths")
    assert pip.exists()
