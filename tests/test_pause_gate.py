"""The fleet pause control really stops Mel opening positions — Tier 1 and Tier 2, verified not assumed.

WHY THIS FILE EXISTS
The dashboard's "Trading Paused" control once wrote a flag no trading code read (measured on the real
fleet: 381 ticks ran after the flag was set), and the flag was keyed by exchange id, so it was inherited
by whatever bot traded there next. Mel was the last bot still unwired: her runner never called the
guard and her engine never consulted the control, so a pause shown on her card changed nothing.

Two lessons from wiring her are pinned here, because both are silent:

  * THE CONTROL MUST ACTUALLY LOAD. The import is deliberately defensive (a missing control must not
    stop a bot that is managing a live position), and that defensive `except` also swallows a mistake in
    the wiring itself. On the first attempt `engine.py` used `sys.path` without importing `sys`, the
    NameError was swallowed, PAUSE became None, and the fail-closed gate would have refused EVERY entry
    for ever — a live bot that had silently stopped trading. `test_the_control_is_actually_importable`
    is the check that catches that, and it is the first test in this file for that reason.

  * NOTHING IN THE ENTRY GATE MAY TOUCH AN OPEN POSITION. The gate is the first statement of
    `try_enter` — the only path that can open a position — and management (stop re-confirmation, the
    time stop, the exits) runs before it in `tick()`. `test_pausing_does_not_gate_management` holds that
    line: a pause must never be able to leave a position unmanaged.
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config as C            # noqa: E402
import engine as E            # noqa: E402
import exchange as ex         # noqa: E402
import pause_state as PAUSE   # noqa: E402


class _FakeClient:
    """Enough client for construction; every order call is a failure by design."""

    def fetch_positions(self, symbols=None):
        return []

    def fetch_open_orders(self, symbol=None, params=None):
        return []


def _store(tmp_path, body):
    p = tmp_path / "bot_pause_state.json"
    p.write_text(body if isinstance(body, str) else json.dumps(body))
    return str(p)


@pytest.fixture
def engine(monkeypatch):
    """A real Engine with dry_run=False and order placement booby-trapped.

    dry_run=False matters: it removes the early return that would otherwise place no order whatever the
    gate did, so the tests below fail if the gate lets an entry through.
    """
    monkeypatch.setattr(ex, "market_spec", lambda *a, **k: {"contracts": 1.0})
    orders = []
    monkeypatch.setattr(ex, "place_market", lambda *a, **k: orders.append(a) or {"id": "fake"})
    monkeypatch.setattr(ex, "place_stop", lambda *a, **k: orders.append(a) or {"id": "fake"})
    monkeypatch.setattr(ex, "place_limit", lambda *a, **k: orders.append(a) or {"id": "fake"})
    cfg = C.load_config(ROOT / "config.yaml")
    eng = E.Engine(cfg, _FakeClient(), state={}, db=None, notify=None, dry_run=False)
    eng.orders = orders        # so a test can prove nothing was placed
    return eng


class _Decision:
    action, entry, sl, tp = "LONG", 3000.0, 2950.0, 3120.0


class _Sizing:
    contracts, margin, notional, scale = 1.0, 100.0, 100.0, 1.0


def test_the_control_is_actually_importable():
    """The wiring itself, not its intent: a swallowed NameError here stops the bot trading for ever."""
    assert E.PAUSE is not None, (
        "engine.PAUSE is None — the defensive import failed. The gate will refuse EVERY entry, so Mel "
        "would silently stop trading rather than pause. Check the import block, including that every "
        "module it uses (`sys`!) is imported."
    )
    assert hasattr(E.PAUSE, "blocks_new_entries")


def test_a_paused_bot_refuses_to_open_and_places_no_order(engine, tmp_path, monkeypatch):
    monkeypatch.setattr(PAUSE, "STORE_PATH",
                        _store(tmp_path, {"mel": {"paused": True, "reason": "operator"}}))
    out = engine.try_enter(_Decision(), _Sizing())
    assert out.get("status") == "paused", out
    assert "paused" in out.get("reason", "").lower()
    assert engine.orders == [], f"a paused bot placed {engine.orders}"


def test_a_not_paused_store_does_not_block_the_entry(engine, tmp_path, monkeypatch):
    """The other direction, which a refuse-everything gate would pass by accident: an explicit
    `paused: false` must let the tick past the gate. Past it the entry needs a real venue, so a failure
    there is the fake client's, not the gate's — what is asserted is that the gate did not answer
    "paused", and that the refusal is therefore coming from the pause control rather than from
    something that simply refuses everything."""
    monkeypatch.setattr(PAUSE, "STORE_PATH", _store(tmp_path, {"mel": {"paused": False}}))
    try:
        out = engine.try_enter(_Decision(), _Sizing())
    except Exception:
        out = {"status": "reached the venue (fake client failed, as expected)"}
    assert (out or {}).get("status") != "paused", out


def test_a_corrupt_store_fails_closed(engine, tmp_path, monkeypatch):
    monkeypatch.setattr(PAUSE, "STORE_PATH", _store(tmp_path, "{ not json"))
    assert engine.try_enter(_Decision(), _Sizing()).get("status") == "paused"


def test_an_unavailable_control_fails_closed(engine, monkeypatch):
    """A control that cannot be imported is 'cannot establish intent', never 'no pause'."""
    monkeypatch.setattr(E, "PAUSE", None)
    assert engine.try_enter(_Decision(), _Sizing()).get("status") == "paused"


def test_the_gate_asks_about_mel_own_bot_id(engine, monkeypatch):
    """Keyed by bot id: an exchange-keyed flag is how a pause is inherited by the next bot there."""
    seen = []

    class _Spy:
        @staticmethod
        def blocks_new_entries(bot_id, *a, **k):
            seen.append(bot_id)
            return True

    monkeypatch.setattr(E, "PAUSE", _Spy)
    engine.try_enter(_Decision(), _Sizing())
    assert seen == ["mel"], seen


def test_pausing_does_not_gate_management(engine, tmp_path, monkeypatch):
    """Management must run while paused: a pause may never leave a position unmanaged."""
    consulted = []

    class _Spy:
        @staticmethod
        def blocks_new_entries(bot_id, *a, **k):
            consulted.append(bot_id)
            return True

    monkeypatch.setattr(E, "PAUSE", _Spy)
    try:
        engine.manage(_Decision())
    except Exception:
        pass          # a fake client cannot get far; what matters is whether the gate was consulted
    assert consulted == [], "the pause gate was consulted from the management path"


def test_the_runner_runs_the_guard_before_the_tick():
    """Tier 1: the wrapper must consult the guard FIRST, and honour its exit codes."""
    src = (ROOT / "run_mel.sh").read_text()
    guard_at = src.index("pause_guard.py --bot mel")
    tick_at = src.index(".venv/bin/python3 run_bot.py")   # the invocation, not a mention in the header
    assert guard_at < tick_at, "the guard must run BEFORE the tick"
    assert "--venue" in src, "the guard needs the venue's own view, not the state file's"
    assert "tools/venue_state.py" in src, "the venue value must come from the exchange"
    for code in ("10)", "11)"):
        assert code in src, f"the runner does not handle guard exit {code}"
    assert "/usr/bin/python3" in src, "the guard must run on the stdlib system python, not the venv"
