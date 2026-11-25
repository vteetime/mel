"""Fixes from the independent review pass (DeepSeek v4 pro, 2026-09-19).

1 an UNPRICED force-close was queued as exit_price=0, and the retry then wrote a fabricated
  ~-200% ROE (about the whole notional on 2x) which trips the daily-loss/drawdown caps and
  blocks entries permanently · 2 `max_sl_restores` was read from the wrong config section, so
  the documented safety bound never bound · 3 an invalid-but-ours stop was replaced without
  being cancelled, leaving an unreachable survivor · 4 the leverage the venue applies was
  never verified although sizing and every cap assume it.
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
import exchange as ex       # noqa: E402
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


class UnpricedClose(FakeKraken):
    """The close DOES fill the position, but the order carries no price and no fills."""
    def create_order(self, symbol, otype, side, amount, price=None, params=None):
        if otype == "stop":
            raise RuntimeError("venue refused the stop")
        if otype == "market" and (params or {}).get("reduceOnly"):
            n = float(amount)
            self.position = None                      # filled: the position is gone
            o = {"id": f"o{len(self.placed) + 1}", "symbol": symbol, "type": "market",
                 "side": side, "amount": n, "price": None, "stopPrice": None,
                 "reduceOnly": True, "status": "closed", "filled": 0.0, "average": None,
                 "info": {}}
            self.placed.append(o)
            return dict(o)
        return super().create_order(symbol, otype, side, amount, price, params)

    def fetch_my_trades(self, symbol=None, limit=100):
        return []                                     # no evidence either


# ── 1. no fabricated loss from an unpriced close ────────────────────────────

def test_an_unpriced_force_close_queues_no_zero_price_exit(tmp_path):
    eng, client, state, db, alerts = _rig(tmp_path, client=UnpricedClose())
    eng.try_enter(long_decision(), _size(eng))
    queued = state.get_flag("pending_close") or []
    for q in queued:
        if float(q.get("exit_price") or 0) <= 0:
            assert q.get("needs_price") is True, "an unpriced exit must be marked as such"
    # The record is still written — Trip's lesson is that a closed trade must never be
    # invisible — but it must be PRICED, and priced from evidence or a flagged level.
    out = eng.retry_pending_close()
    assert out.get("written", 0) == 1, out
    with sqlite3.connect(tmp_path / "t.db") as con:
        row = con.execute("SELECT exit_price, pnl_pct, exit_reason, settlement_source "
                          "FROM trades ORDER BY id DESC LIMIT 1").fetchone()
    assert row[0] > 0, f"a zero price reached the ledger: {row}"
    assert row[1] > -50, f"a fabricated full-notional loss was recorded: {row}"
    with sqlite3.connect(tmp_path / "t.db") as con:
        bad = con.execute("SELECT COUNT(*) FROM trades WHERE exit_price <= 0").fetchone()[0]
    assert bad == 0, "a fabricated loss reached the ledger"


def test_the_retry_refuses_a_queued_zero_price(tmp_path):
    """Backstop: even if one gets queued, it must never be written as a full-notional loss."""
    eng, client, state, db, alerts = _rig(tmp_path)
    eng.try_enter(long_decision(), _size(eng))
    tid = state.get_open()["trade_id"]
    state.set_flag("pending_close", [{
        "trade_id": tid, "entry_price": 1.368, "contracts": 963.0, "exit_price": 0.0,
        "reason": "sl_placement_failed", "leverage": 2, "margin_fraction": 0.9,
        "sl_price": 1.3338, "close_side": "sell", "contract_size": 1.0,
        "side": "sell", "needs_price": True}])
    out = eng.retry_pending_close()
    with sqlite3.connect(tmp_path / "t.db") as con:
        row = con.execute("SELECT exit_price, pnl_pct, status FROM trades WHERE id=?",
                          (tid,)).fetchone()
    # never a zero price, never a full-notional loss: priced from the level and flagged
    assert row[0] > 0, f"the zero-price exit was written: {row}"
    assert row[1] > -50, f"a fabricated loss was recorded: {row}"
    assert row[2] == "closed", row
    with sqlite3.connect(tmp_path / "t.db") as con:
        src = con.execute("SELECT settlement_source FROM trades WHERE id=?", (tid,)).fetchone()[0]
    assert src in ("level_fallback", "fills", "exchange"), src


def test_the_recovery_tool_repairs_an_impossible_price(tmp_path):
    db_path = tmp_path / "tools.db"
    con = sqlite3.connect(db_path)
    con.executescript(SCHEMA)
    con.execute("""INSERT INTO trades (bot_id,exchange,symbol,direction,entry_price,
      exit_price,contracts,leverage,notional,pnl_net,pnl_pct,entry_time,exit_time,exit_reason,
      status,sl_price,guard_config,settlement_source)
      VALUES ('mel','krakenfutures','XRP/USD:USD','long',1.368,0.0,0.277,2,732.0,-732.0,-200.0,
      '2026-09-19T10:00:00+00:00','2026-09-19T12:00:00+00:00','sl_placement_failed','closed',
      1.3338,'{"margin_fraction": 0.5}','provisional')""")
    con.commit()
    con.close()
    import subprocess
    tool = ROOT / "tools" / "reconcile_settlement.py"
    listing = subprocess.run([sys.executable, str(tool), "--db", str(db_path), "--list"],
                             capture_output=True, text=True)
    assert "IMPOSSIBLE" in listing.stdout
    fixed = subprocess.run([sys.executable, str(tool), "--db", str(db_path),
                            "1", "1.33380", "--yes"], capture_output=True, text=True)
    assert fixed.returncode == 0 and "written" in fixed.stdout
    with sqlite3.connect(db_path) as con:
        pnl_pct = con.execute("SELECT pnl_pct FROM trades WHERE id=1").fetchone()[0]
    assert -6 < pnl_pct < -4, f"a stop-out should be a small loss, got {pnl_pct}%"


# ── 2. the restore bound must bind ──────────────────────────────────────────

def test_the_configured_restore_bound_is_the_one_that_applies(tmp_path):
    cfg = json.loads(json.dumps(CFG))
    cfg["guards"]["max_sl_restores"] = 1
    eng, client, state, db, alerts = _rig(tmp_path, cfg=cfg)
    eng.try_enter(long_decision(), _size(eng))
    losses = 0
    for _ in range(5):
        pos = state.get_open()
        if not pos:
            break
        oid = str(pos["sl_order_id"])
        client.orders = [o for o in client.orders if str(o["id"]) != oid]
        client.placed = [o for o in client.placed if str(o["id"]) != oid]
        eng.manage(None)
        losses += 1
    assert state.is_flat(), "the position must close once the bound is reached"
    assert losses == 2, f"with max_sl_restores=1 the position survives one restore, got {losses}"


def test_the_key_lives_where_the_code_reads_it():
    assert "max_sl_restores" in CFG["guards"], "the code reads it from guards"
    src = (ROOT / "engine.py").read_text()
    assert 'self.cfg["guards"].get("max_sl_restores"' in src
    assert (ROOT / "README.md").read_text().count("ops.max_sl_restores") == 0


# ── 3. an invalid stop of ours is cancelled before replacement ──────────────

def test_an_invalid_stop_is_cancelled_before_the_replacement(tmp_path):
    eng, client, state, db, alerts = _rig(tmp_path)
    eng.try_enter(long_decision(), _size(eng))
    pos = state.get_open()
    oid = str(pos["sl_order_id"])
    for o in client.orders:
        if str(o["id"]) == oid:
            o["amount"] = 1.0               # exists, but does not protect the position
    eng.manage(None)
    assert oid in client.cancelled, "the invalid stop was replaced WITHOUT being cancelled"
    assert str(state.get_open()["sl_order_id"]) != oid


def test_a_failed_cancel_does_not_stack_a_second_stop(tmp_path):
    class CancelRefused(FakeKraken):
        def cancel_order(self, oid, symbol=None):
            raise RuntimeError("venue refused the cancel")

    eng, client, state, db, alerts = _rig(tmp_path, client=CancelRefused())
    eng.try_enter(long_decision(), _size(eng))
    pos = state.get_open()
    oid = str(pos["sl_order_id"])
    for o in client.orders:
        if str(o["id"]) == oid:
            o["amount"] = 0.001
    before = len([o for o in client.orders if o["type"] == "stop"])
    assert eng._ensure_sl(state.get_open(), float(pos["remaining"])) == "unknown"
    after = len([o for o in client.orders if o["type"] == "stop"])
    assert after == before, "a duplicate stop was stacked while the old one could not be cancelled"


# ── 4. leverage is verified, not assumed ────────────────────────────────────

def test_a_leverage_mismatch_is_reported(tmp_path):
    class Lev5(FakeKraken):
        def privateGetLeveragepreferences(self, params=None):
            return {"leveragePreferences": [{"symbol": "PF_XRPUSD", "maxLeverage": 5.0}]}

    ok, why = ex.ensure_leverage(Lev5(), "XRP/USD:USD", 2)
    assert not ok and "5" in why and "2" in why, why


def test_a_matching_leverage_is_confirmed(tmp_path):
    class Lev2(FakeKraken):
        def privateGetLeveragepreferences(self, params=None):
            return {"leveragePreferences": [{"symbol": "PF_XRPUSD", "maxLeverage": 2.0}]}

    ok, why = ex.ensure_leverage(Lev2(), "XRP/USD:USD", 2)
    assert ok and "confirmed" in why, why


def test_an_unlisted_leverage_is_unverified_not_a_match(tmp_path):
    class NoPrefs(FakeKraken):
        def privateGetLeveragepreferences(self, params=None):
            return {"leveragePreferences": []}

    ok, why = ex.ensure_leverage(NoPrefs(), "XRP/USD:USD", 2)
    assert ok and "unverified" in why, why


def test_a_leverage_mismatch_stops_the_tick_before_any_order(tmp_path, monkeypatch):
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

    class WrongLeverage(FakeKraken):
        def privateGetLeveragepreferences(self, params=None):
            return {"leveragePreferences": [{"symbol": "PF_XRPUSD", "maxLeverage": 5.0}]}

        def fetch_ohlcv(self, symbol=None, timeframe="15m", since=None, limit=200):
            n = int(limit or 200)
            return [[1700000000000 + i * 900000, 1.35 + i % 7, 2602, 2598, 2600.5, 10.0]
                    for i in range(n)]

    client = WrongLeverage()
    monkeypatch.setattr(run_bot.ex, "new_client", lambda *a, **k: client)
    monkeypatch.setenv("KRAKEN_API_KEY", "k")
    monkeypatch.setenv("KRAKEN_API_SECRET", "s")
    monkeypatch.setattr(sys, "argv", ["run_bot.py", "--live", "--config",
                                      str(tmp_path / "cfg.yaml")])
    assert run_bot.main() == 1
    assert client.placed == [], "an order was placed despite a leverage mismatch"
