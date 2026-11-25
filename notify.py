"""Alerts.

One rule: alert delivery must NEVER be able to break the trading path. Every send is
wrapped, so a dead Telegram token or a network blip cannot raise into the engine and
abort an exit. This is the lesson from bx-bot, where an unwrapped send_alert on a
failure path is a hazard in its own right.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.request

log = logging.getLogger("mel.notify")

# ── alert gate (Jev triage, fail-open, OFF unless ALERT_GATE=on) ──────────────
try:
    import sys as _sys
    _sys.path.insert(0, "/opt/data/jev-spike/triage")
    import alert_gate as _alert_gate
except Exception:  # a missing gate module must never break the alert path
    _alert_gate = None


def gated_send(text: str) -> bool:
    """True => deliver. False => SUPPRESS (digest path). Fail-open: never swallow."""
    if _alert_gate is None:
        return True
    try:
        return _alert_gate.gate(text)["action"] in ("SEND", "ESCALATE")
    except Exception:
        return True


def under_test() -> bool:
    """True when this process is a test run (pytest sets PYTEST_CURRENT_TEST).

    [INCIDENT] Without this guard, the tests that drive `run_bot.main()` end-to-end ran a
    LIVE-mode tick with a temp sentinel, and because the test shell had the Telegram token in
    its environment the notifier was live — so the suite sent Tee eight real alerts about a
    FAKE venue's orders. A test must never be able to message a human: the guard is here, at
    the only place that talks to the network, so it holds for every future test.
    """
    return bool(os.environ.get("PYTEST_CURRENT_TEST") or os.environ.get("MEL_UNDER_TEST"))


def alert_once(state, key: str, notify_fn, message: str, cooldown_h: float = 6.0) -> bool:
    """Send an alert for a STUCK condition at most once per cooldown.

    A condition that persists (unresolved protection, unexplained orders, a blocked entry)
    is re-evaluated every tick, and 15-minute repetition is noise, not information. The
    watchdog on this host uses the same shape: remember when it was last said, say it again
    only after the cooldown. Returns True when the message was sent.
    """
    import time
    if state is None:
        notify_fn(message)
        return True
    now = time.time()
    stamp = state.get_flag(f"alerted:{key}")
    try:
        last = float(stamp or 0.0)
    except (TypeError, ValueError):
        last = 0.0
    if last and (now - last) < cooldown_h * 3600:
        log.info("ALERT SUPPRESSED (%s, said %.1fh ago): %s", key, (now - last) / 3600, message)
        return False
    state.set_flag(f"alerted:{key}", now)
    notify_fn(message)
    return True


def _creds() -> tuple[str, str]:
    """The Telegram credentials, read from the environment EVERY time they are needed.

    Deliberately not cached: see the note in `notify` below. Trip's and Jill's tokens are both accepted
    so one machine can carry the fleet's alerting without per-bot config.
    """
    return (os.environ.get("TRIPKC_TELEGRAM_TOKEN") or os.environ.get("TELEGRAM_BOT_TOKEN", ""),
            os.environ.get("TRIPKC_TELEGRAM_CHAT") or os.environ.get("TELEGRAM_CHAT_ID", ""))


def make_notifier(cfg: dict, enabled: bool = True, sink: list | None = None):
    """Return a callable(str). Records to `sink` when provided (tests)."""
    prefix = cfg.get("ops", {}).get("alert_prefix", "Mel")

    def notify(msg: str) -> None:
        if sink is not None:
            sink.append(msg)
        log.info("ALERT: %s", msg)
        # Read the credentials HERE, at call time, not when the notifier was built. The runner sources
        # .env in the shell, but a hand-run tick (or a scheduler that supplies the environment later in
        # startup) leaves the build-time values empty — and an empty token does not fail loudly, it
        # returns below and drops the alert in silence. That is exactly what happened to Jill, whose
        # entry and exit alerts were logged and delivered to nobody; this is the same fix applied to
        # Mel before she can repeat it. (Mel review, alert path.)
        token, chat = _creds()
        if not enabled or not token or not chat:
            return
        if under_test():
            # Sink or no sink: a test may RECORD a message, never TELEGRAPH one. The earlier
            # form of this guard only held when no sink was supplied, which left the same
            # incident reproducible by any test that wanted to assert on alerts.
            log.info("ALERT SUPPRESSED (test process): %s", msg)
            return
        if not gated_send(msg):
            log.info("ALERT GATED (not sent): %s", msg)
            return
        try:
            body = json.dumps({"chat_id": chat, "text": msg,
                               "disable_web_page_preview": True}).encode()
            req = urllib.request.Request(
                f"https://api.telegram.org/bot{token}/sendMessage", data=body,
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=10) as r:
                r.read()
        except Exception as exc:                      # never propagate
            log.warning("alert delivery failed (ignored): %s", exc)

    return notify


def null_notifier(msg: str) -> None:
    log.info("ALERT (suppressed): %s", msg)
