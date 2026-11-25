"""`--manage-only` must never evaluate — let alone open — an entry.

WHY THIS FILE EXISTS
Mel's hold cap (and every other protective check) is evaluated once per tick, so an
hourly-only tick lets a cap that lands seconds before a tick overshoot by a whole hour.
Measured on the live ledger: 11 of 13 hold-cap exits closed at 24.99-25.00h against a 24h
cap. The fix is a second cron (`13,28,43,58 * * * *`) that runs the SAME management path
more often — and the whole value of it rests on one property: it must be incapable of
opening a position.

Why not simply run the whole tick four times an hour? Because the scoreboard is built from
the FORMING 1h candle (`build_market_state` reads `closes[-1]`, and Kraken returns the
in-progress bar). A 4x-per-hour decision tick would score that same bar at four different
ages — 2, 17, 32 and 47 minutes old — which is an unbacktested change to entry timing, not
just a tighter cap. So the flag has to stop the tick BEFORE the candle read, and these
tests pin exactly that boundary:

  * manage-only + flat  -> returns 0 WITHOUT fetching candles (no score, no entry path);
  * the same tick, flag removed -> DOES fetch candles, so the first test is discriminating
    rather than passing because of some other early return;
  * manage-only + open position -> still runs `manage()` (the hold cap must fire on these
    ticks; that is the entire point) and still never fetches candles.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config as C            # noqa: E402
import exchange as ex         # noqa: E402
import run_bot                # noqa: E402
from db import TradeDB        # noqa: E402
from state import State       # noqa: E402

SYM = "XRP/USD:USD"


class FakeKraken:
    """Just enough ccxt for a dry-run tick; records whether candles were ever read."""

    def __init__(self):
        self.ohlcv_calls = 0
        self.position = None

    def market(self, symbol):
        return {"id": "PF_XRPUSD", "contractSize": 1.0, "type": "swap",
                "precision": {"price": 0.0001, "amount": 1.0},
                "limits": {"amount": {"min": None}}}

    def load_markets(self):
        return {}

    def fetch_positions(self, symbols=None):
        return [] if self.position is None else [dict(self.position, symbol=SYM)]

    def fetch_open_orders(self, symbol=None, params=None):
        return []

    def fetch_ticker(self, symbol):
        return {"last": 0.95, "info": {"price": 0.95}}

    def fetch_ohlcv(self, symbol, timeframe=None, limit=None):
        self.ohlcv_calls += 1
        raise AssertionError(
            "manage-only tick read candles — it must stop before the entry path")


def _cfg(tmp_path):
    """A config pointed entirely at tmp_path: no live state, no live ledger, no live lock."""
    cfg = C.load_config()
    cfg["ops"] = dict(cfg["ops"],
                      state_file=str(tmp_path / "state.json"),
                      log_dir=str(tmp_path / "logs"),
                      lock_file=str(tmp_path / "mel.lock"),
                      db_path=str(tmp_path / "trades.db"),
                      dry_run_default=True)
    (tmp_path / "logs").mkdir(exist_ok=True)
    p = tmp_path / "config.yaml"
    import yaml
    p.write_text(yaml.safe_dump(cfg))
    return p


def _make_db(path):
    """Create a temp ledger with the REAL trades schema.

    TradeDB refuses to start on a missing/partial schema (by design), so the test clones the
    DDL from the live ledger rather than hand-maintaining a second copy that can drift.
    """
    import sqlite3
    if path.exists():
        return
    src = sqlite3.connect("/opt/data/cic/trades.db")
    ddl = [r[0] for r in src.execute(
        "select sql from sqlite_master where type='table' and name='trades'")]
    src.close()
    if not ddl:
        pytest.skip("live trades schema unavailable; cannot build a temp ledger")
    dst = sqlite3.connect(str(path))
    dst.execute(ddl[0])
    dst.commit()
    dst.close()


@pytest.fixture
def harness(tmp_path, monkeypatch):
    fake = FakeKraken()
    cfg_path = _cfg(tmp_path)
    _make_db(tmp_path / "trades.db")
    monkeypatch.setattr(run_bot.ex, "new_client", lambda *a, **k: fake)
    monkeypatch.setattr(run_bot, "load_keys", lambda: {"api_key": "k", "api_secret": "s"})
    return fake, cfg_path, tmp_path


def _hold_lock(path):
    """Grab the tick lock the way a competing process would (separate fd -> real conflict)."""
    import fcntl
    fh = open(str(path), "w")
    fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return fh


def _run(monkeypatch, cfg_path, *argv):
    monkeypatch.setattr(sys, "argv", ["run_bot.py", "--dry-run", *argv,
                                      "--config", str(cfg_path)])
    return run_bot.main()


def test_manage_only_flat_never_reads_candles(harness, monkeypatch):
    """The load-bearing property: no candles -> no score -> no entry."""
    fake, cfg_path, tmp = harness
    assert _run(monkeypatch, cfg_path, "--manage-only") == 0
    assert fake.ohlcv_calls == 0, "manage-only tick reached the entry path"
    assert not State(tmp / "state.json").get_open(), "manage-only tick opened a position"


def test_the_test_discriminates_without_the_flag(harness, monkeypatch):
    """Same tick, flag removed: it DOES read candles (so the test above proves the flag)."""
    fake, cfg_path, tmp = harness
    with pytest.raises(AssertionError, match="read candles"):
        _run(monkeypatch, cfg_path)
    assert fake.ohlcv_calls == 1


def test_manage_only_with_an_open_position_still_manages(harness, monkeypatch):
    """The point of the extra ticks: management (incl. the hold cap) must still run."""
    fake, cfg_path, tmp = harness
    calls = {}

    def fake_manage(self, decision=None):
        calls["managed"] = True
        return {"status": "holding", "remaining": 1.0, "held_h": 23.9}

    monkeypatch.setattr(run_bot.Engine, "manage", fake_manage)
    st = State(tmp / "state.json")
    st.set_open({"trade_id": 1, "direction": "short", "entry_price": 0.96,
                 "contracts": 10.0, "remaining": 10.0, "entry_time": "2026-11-15T09:03:26+00:00",
                 "sl_price": 0.99, "sl_order_id": "x", "tp_price": 0.92, "tp_order_id": "y",
                 "tp_filled": False})
    assert _run(monkeypatch, cfg_path, "--manage-only") == 0
    assert calls.get("managed") is True, "manage-only tick skipped management"
    assert fake.ohlcv_calls == 0


# ── Terra round-1 must-fix: the lock must not silently swallow the pass ──────────────────────
# The entry tick exits 0 on a taken lock (a duplicate is worthless). For a manage-only tick that
# behaviour would discard exactly the hold-cap check the job exists for, and say nothing — the
# promised <=15-minute bound would be false. So it waits, then runs the deferred pass; and if the
# lock never frees it exits NON-ZERO so the gap is visible instead of green.

def test_manage_only_waits_for_the_lock_then_runs_the_deferred_pass(harness, monkeypatch):
    fake, cfg_path, tmp = harness
    calls = {}

    def fake_manage(self, decision=None):
        calls["managed"] = True
        return {"status": "holding", "remaining": 1.0, "held_h": 24.1}

    monkeypatch.setattr(run_bot.Engine, "manage", fake_manage)
    monkeypatch.setattr(run_bot, "MANAGE_ONLY_LOCK_WAIT_SECONDS", 5.0)
    st = State(tmp / "state.json")
    st.set_open({"trade_id": 2, "direction": "short", "entry_price": 0.96,
                 "contracts": 10.0, "remaining": 10.0,
                 "entry_time": "2026-11-15T09:03:26+00:00",
                 "sl_price": 0.99, "sl_order_id": "x", "tp_price": 0.92,
                 "tp_order_id": "y", "tp_filled": False})

    held = _hold_lock(tmp / "mel.lock")

    import threading
    import fcntl
    def release_later():
        import time as _t
        _t.sleep(0.6)
        fcntl.flock(held, fcntl.LOCK_UN)
        held.close()
    th = threading.Thread(target=release_later)
    th.start()
    try:
        assert _run(monkeypatch, cfg_path, "--manage-only") == 0
    finally:
        th.join()
    assert calls.get("managed") is True, (
        "the manage-only tick gave up instead of running the deferred management pass")


def test_manage_only_exits_nonzero_when_the_lock_never_frees(harness, monkeypatch):
    fake, cfg_path, tmp = harness
    calls = {}

    def fake_manage(self, decision=None):
        calls["managed"] = True
        return {"status": "holding"}

    monkeypatch.setattr(run_bot.Engine, "manage", fake_manage)
    monkeypatch.setattr(run_bot, "MANAGE_ONLY_LOCK_WAIT_SECONDS", 0.5)
    held = _hold_lock(tmp / "mel.lock")      # 0.5s bound, never released
    try:
        rc = _run(monkeypatch, cfg_path, "--manage-only")
        assert rc == 3, f"a swallowed lock must not exit 0 (got {rc})"
        assert not calls.get("managed"), "ran management without the lock"
    finally:
        import fcntl
        fcntl.flock(held, fcntl.LOCK_UN)
        held.close()
