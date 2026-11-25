"""The consecutive-loss circuit breaker must be able to RELEASE.

Found by Mel's own backtest on 2026-09-20: `g_consecutive_losses` blocked entries at 3 losses and
its only reset was a winning close — which the block prevents. The out-of-sample window produced
three losses and then eighteen months of no entries at all. The same code and config are live in
Jack, neither bot had reached 3 yet, so this was fixed before it fired.
"""

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import guards as G                     # noqa: E402
import run_bot                         # noqa: E402
from state import State                # noqa: E402
from test_dry_run_scenarios import CFG  # noqa: E402


def _iso(**kw):
    return (datetime.now(timezone.utc) - timedelta(**kw)).isoformat()


def test_a_streak_from_today_blocks_entries():
    ok, why = G.g_consecutive_losses(3, CFG, _iso(minutes=5))
    assert ok is False
    assert "circuit breaker" in why


def test_a_streak_from_YESTERDAY_does_not_block():
    """THE anti-wedge assertion: this is the test that fails against the old win-only reset."""
    ok, why = G.g_consecutive_losses(3, CFG, _iso(days=1, minutes=1))
    assert ok is True, "a stale streak must not block a new day's trading"
    assert why == ""


def test_an_unknown_last_loss_time_blocks_fail_closed():
    ok, _why = G.g_consecutive_losses(3, CFG, None)
    assert ok is False, "an unreadable history is not evidence that the streak is stale"


def test_the_breaker_is_disabled_by_zero():
    cfg = {**CFG, "guards": {**CFG["guards"], "max_consecutive_losses": 0}}
    assert G.g_consecutive_losses(9, cfg, _iso(minutes=1))[0] is True


def test_the_message_tells_the_operator_how_to_clear_it():
    ok, why = G.g_consecutive_losses(3, CFG, _iso(minutes=1))
    assert ok is False
    assert "UTC day boundary" in why and "clear_halt.py --halt loss_streak" in why, why


def test_an_acknowledgement_clears_the_breaker(tmp_path):
    st = State(tmp_path / "state.json")
    st.set_flag("ack_loss_streak", {"at": _iso(minutes=1), "note": "reviewed", "by": "operator"})
    stats = {"consecutive_losses": 3, "last_loss_at": _iso(minutes=5)}
    assert run_bot.effective_loss_streak(stats, st) == 0


def test_a_new_loss_after_the_acknowledgement_re_arms_it(tmp_path):
    """The ack expires BY CONSTRUCTION — otherwise it would be a permanent bypass of the breaker."""
    st = State(tmp_path / "state.json")
    st.set_flag("ack_loss_streak", {"at": _iso(hours=2), "note": "reviewed", "by": "operator"})
    stats = {"consecutive_losses": 4, "last_loss_at": _iso(minutes=1)}   # a loss LANDED after it
    assert run_bot.effective_loss_streak(stats, st) == 4


def test_an_acknowledgement_without_a_streak_changes_nothing(tmp_path):
    st = State(tmp_path / "state.json")
    st.set_flag("ack_loss_streak", {"at": _iso(), "note": "x", "by": "operator"})
    assert run_bot.effective_loss_streak({"consecutive_losses": 0, "last_loss_at": None}, st) == 0
