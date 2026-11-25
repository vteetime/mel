"""Alert-safety tests — an incident class, not a review finding.

On 2026-09-19 the Mel test suite sent Tee eight real Telegram alerts. Two tests drive
`run_bot.main()` end-to-end with a temp `LIVE` sentinel (so the tick runs in live mode), and
the shell running pytest had Telegram credentials in its environment — so the notifier was
live and the alerts about a FAKE venue's orders went out for real.

Two rules now, and both are tested here:

1. **A test process can never send an alert.** `notify.under_test()` (pytest sets
   `PYTEST_CURRENT_TEST`) suppresses every network send.
2. **A stuck condition alerts once per cooldown, not once per tick.** A blocked entry is
   re-evaluated every 15 minutes; repeating the same sentence is noise, not information.
"""

import json
import os
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import notify as N          # noqa: E402
from state import State     # noqa: E402

CFG = {"ops": {"alert_prefix": "🐝 Mel"}}


def test_a_test_process_cannot_send(monkeypatch):
    """The exact failure that happened: live notifier + credentials + a test."""
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "token-that-would-work")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "12345")
    monkeypatch.delenv("MEL_UNDER_TEST", raising=False)

    sent = []

    def bomb(*a, **k):
        sent.append(a)
        raise AssertionError("a test process tried to send a real Telegram alert")

    monkeypatch.setattr(N.urllib.request, "urlopen", bomb)
    n = N.make_notifier(CFG, enabled=True)           # live-looking, as in the incident
    n("🛑 Mel — this must never reach a human from a test")
    assert sent == [], "the test suite sent a real alert"
    assert N.under_test() is True, "pytest must be detected as a test process"


def test_an_unset_flag_still_sends_when_not_under_test(monkeypatch):
    """The guard must not disarm the bot in production."""
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.delenv("MEL_UNDER_TEST", raising=False)
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "t")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "c")
    calls = []
    monkeypatch.setattr(N.urllib.request, "urlopen",
                        lambda req, *a, **k: calls.append(req) or _Resp())
    N.make_notifier(CFG, enabled=True)("🐝 Mel — hello")
    assert len(calls) == 1, "a production alert was suppressed by mistake"


class _Resp:
    def read(self):
        return b"{}"

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_a_stuck_condition_alerts_once_per_cooldown(tmp_path):
    state = State(tmp_path / "state.json")
    sent = []
    fn = sent.append

    assert N.alert_once(state, "flat_with_protection", fn, "first") is True
    assert N.alert_once(state, "flat_with_protection", fn, "again 15 minutes later") is False
    assert sent == ["first"], sent

    # after the cooldown the condition is worth saying again
    state.set_flag("alerted:flat_with_protection", time.time() - 7 * 3600)
    assert N.alert_once(state, "flat_with_protection", fn, "still stuck") is True
    assert sent == ["first", "still stuck"]

    # and a DIFFERENT condition is not suppressed by the first one's cooldown
    assert N.alert_once(state, "dangling_unresolved", fn, "other") is True
    assert len(sent) == 3


def test_alert_credentials_are_read_at_call_time_not_build_time(monkeypatch):
    """Jill's incident, pinned for Mel.

    A notifier that captures the Telegram credentials when it is BUILT drops every alert in silence if
    they were absent then: `notify()` returns without sending and only logs. `run_mel.sh` sources .env,
    but a hand-run tick — or a scheduler that supplies the environment later in startup — does not, which
    is exactly how Jill's entry and exit alerts were written to a log and delivered to nobody. So the
    credentials are read when the alert is SENT.
    """
    import config as C
    import notify as N

    for var in ("TRIPKC_TELEGRAM_TOKEN", "TELEGRAM_BOT_TOKEN",
                "TRIPKC_TELEGRAM_CHAT", "TELEGRAM_CHAT_ID"):
        monkeypatch.delenv(var, raising=False)

    cfg = C.load_config(ROOT / "config.yaml")
    notify = N.make_notifier(cfg, enabled=True)          # built with NO credentials in the environment

    monkeypatch.setenv("TRIPKC_TELEGRAM_TOKEN", "TOKEN-X")   # supplied only afterwards
    monkeypatch.setenv("TRIPKC_TELEGRAM_CHAT", "CHAT-Y")
    monkeypatch.setattr(N, "under_test", lambda: False)      # this file must observe a real send

    class _Resp:
        def read(self):
            return b"{}"

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    sent = []
    monkeypatch.setattr(N.urllib.request, "urlopen",
                        lambda req, timeout=None: sent.append(req.full_url) or _Resp())

    notify("hello")
    assert sent == ["https://api.telegram.org/botTOKEN-X/sendMessage"], sent
