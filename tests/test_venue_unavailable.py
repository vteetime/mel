"""Mel — degrade gracefully on a venue outage instead of crashing.

Design: /opt/data/DESIGN_MEL_VENUE_UNAVAILABLE.md.

During Kraken Futures' maintenance overrun (2026-10-15 07:00–07:52 UTC) Mel's 07:02 tick
died with an unhandled `ccxt.ExchangeNotAvailable` at `new_client() -> load_markets()` and
her own alerting stayed silent for the whole window (the SL/TP rest on Kraken, so the
position was never naked — the defect was operational, not monetary).

Fix, two parts:
 1. `exchange.new_client` wraps `load_markets()` in the existing reads-only `retry()` and
    translates a persistent availability failure into a dedicated `VenueUnavailable` error;
    non-retryable errors (e.g. auth) still propagate unchanged.
 2. `run_bot.main` catches `VenueUnavailable` at client creation, logs, fires ONE deduped
    alert, and returns 0 (a healthy "not trading this tick", consistent with the
    no-collateral / halted paths).
"""

import json
import sqlite3
import sys
from pathlib import Path

import pytest
import ccxt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_bot                                          # noqa: E402
import exchange as ex                                   # noqa: E402
from test_dry_run_scenarios import SCHEMA               # noqa: E402
from test_trip_class import CFG                         # noqa: E402


class _FakeClient:
    """A ccxt-shaped client whose load_markets() we control."""

    def __init__(self, load_markets=None):
        self.options = {}
        self._load_markets = load_markets or (lambda: None)

    def load_markets(self):
        return self._load_markets()


def _patch_ccxt(monkeypatch, fake):
    monkeypatch.setattr(ex.ccxt, "krakenfutures", lambda *a, **k: fake)


# ── new_client translation ───────────────────────────────────────────────────

def test_new_client_recovers_and_returns_a_client(monkeypatch):
    fake = _FakeClient()
    _patch_ccxt(monkeypatch, fake)
    calls = []
    monkeypatch.setattr(ex, "retry",
                        lambda fn, attempts=4, base_delay=1.0: calls.append(fn) or None)
    out = ex.new_client({}, {"api_key": "k", "api_secret": "s"}, dry_run=True)
    assert out is fake
    assert len(calls) == 1 and callable(calls[0]), \
        "load_markets must be routed through the reads-only retry wrapper"


def test_new_client_retries_a_transient_outage_and_recovers(monkeypatch):
    """A real one-shot 503 must be retried (via the REAL retry), not crash the tick."""
    monkeypatch.setattr(ex.time, "sleep", lambda _s: None)   # no real backoff in tests
    attempts = {"n": 0}

    def flaky():
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise ccxt.ExchangeNotAvailable("krakenfutures GET .../instruments 503")
        return None

    fake = _FakeClient(flaky)
    _patch_ccxt(monkeypatch, fake)
    out = ex.new_client({}, {"api_key": "k", "api_secret": "s"}, dry_run=True)
    assert out is fake
    assert attempts["n"] == 2, "a transient 503 must be retried, not propagated"


def test_new_client_raises_venue_unavailable_on_persistent_failure(monkeypatch):
    _patch_ccxt(monkeypatch, _FakeClient())

    def _final_fail(fn, attempts=4, base_delay=1.0):
        raise ex.ExchangeError("exchange call failed after 4 attempts: 503")

    monkeypatch.setattr(ex, "retry", _final_fail)
    with pytest.raises(ex.VenueUnavailable) as ei:
        ex.new_client({}, {"api_key": "k", "api_secret": "s"}, dry_run=True)
    assert "load_markets" in str(ei.value), \
        "the translated error must say what could not be loaded"
    assert ei.value.__cause__ is not None, \
        "the VenueUnavailable must chain the underlying ExchangeError"


def test_new_client_does_not_swallow_a_non_retryable_error(monkeypatch):
    """Bad credentials are NOT 'venue down' — they must stay loud, untranslated."""
    _patch_ccxt(monkeypatch, _FakeClient())

    def _auth_fail(fn, attempts=4, base_delay=1.0):
        raise ccxt.AuthenticationError("invalid api key")

    monkeypatch.setattr(ex, "retry", _auth_fail)
    with pytest.raises(ccxt.AuthenticationError):
        ex.new_client({}, {"api_key": "k", "api_secret": "s"}, dry_run=True)


# ── main() degrades cleanly ──────────────────────────────────────────────────

def _setup_tick(tmp_path, monkeypatch):
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
    monkeypatch.setenv("KRAKEN_API_KEY", "k")
    monkeypatch.setenv("KRAKEN_API_SECRET", "s")
    monkeypatch.setattr(sys, "argv", ["run_bot.py", "--live", "--config",
                                      str(tmp_path / "cfg.yaml")])


def test_a_tick_degrades_cleanly_when_the_venue_is_unreachable(tmp_path, monkeypatch):
    _setup_tick(tmp_path, monkeypatch)

    def _raise(*a, **k):
        raise ex.VenueUnavailable(
            "Kraken Futures unreachable (load_markets failed): test 503")

    monkeypatch.setattr(run_bot.ex, "new_client", _raise)
    alerts = []
    monkeypatch.setattr(
        run_bot, "alert_once",
        lambda state, key, notify_fn, message, cooldown_h=6.0: alerts.append((key, message)) or True)

    rc = run_bot.main()
    assert rc == 0, "a venue outage must be a clean skip (0), not a crash (1/traceback)"
    assert len(alerts) == 1 and alerts[0][0] == "venue_unavailable", \
        "exactly one deduped 'venue unavailable' alert must fire"
    assert "Kraken" in alerts[0][1] and "resting" in alerts[0][1], \
        "the alert must say the position stays protected by its resting stop"
