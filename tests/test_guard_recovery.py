"""Guard-recovery tests: every halt Mel can enter must have a working, documented way out.

A fail-closed guard that cannot be cleared is a wedge — the bot stops trading and nobody can tell
whether it is protecting them or broken. These tests drive the FULL loop, including running the
operator tool as a subprocess against a real database, because "the README says so" is not
evidence that the escape hatch works.
"""

import json
import sqlite3
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as C          # noqa: E402
import guards as G          # noqa: E402
from db import TradeDB      # noqa: E402
from test_dry_run_scenarios import SCHEMA, long_decision  # noqa: E402

CFG = C.load_config()


def _market():
    return SimpleNamespace(htf={"trend": "up"}, rsi=52.0, bb_pct_b=48.0, macd_hist=0.5)


def _ctx(**over):
    ctx = {"trades_today": 0, "consecutive_losses": 0, "pnl_today_pct": 0.0,
           "pnl_total_pct": 0.0, "pnl_today_equity_pct": 0.0, "pnl_total_equity_pct": 0.0,
           "candles_since_close": 99, "sl_failure_streak": 0, "unverified_settlements": 0}
    ctx.update(over)
    return ctx


def _db(tmp_path) -> TradeDB:
    con = sqlite3.connect(tmp_path / "guard.db")
    con.executescript(SCHEMA)
    con.commit()
    con.close()
    return TradeDB(tmp_path / "guard.db")


def _closed(db, *, exit_price, pnl_pct, pnl_net, reason, source):
    tid = db.record_entry(symbol="XRP/USD:USD", direction="long", entry_price=1.368,
                          contracts=0.277, leverage=2, sl_price=1.3338, tp1_price=1.42272,
                          tp2_price=0.0, notional=732.0, sl_pct=0.025,
                          guard_config={"margin_fraction": 0.5})
    db.record_exit(tid, exit_price=exit_price, pnl_net=pnl_net, pnl_pct=pnl_pct,
                   exit_reason=reason, settlement_source=source)
    return tid


def _allowed(ctx):
    return G.run_guards(long_decision(), _market(), ctx, CFG).passed


# ── the unverified-settlement halt ─────────────────────────────────────────

def test_the_unverified_settlement_halt_blocks_then_the_tool_clears_it(tmp_path):
    db = _db(tmp_path)
    _closed(db, exit_price=1.35, pnl_pct=-11.5, pnl_net=-42.0,
            reason="unverified_exit", source="level_fallback")
    ctx = _ctx(**db.stats())
    assert ctx["unverified_settlements"] == 1
    assert not _allowed(ctx), "an estimated settlement must halt new risk"

    # the operator tool, run the way the README documents it
    tool = ROOT / "tools" / "reconcile_settlement.py"
    listing = subprocess.run([sys.executable, str(tool), "--db", str(tmp_path / "guard.db"),
                              "--list"], capture_output=True, text=True)
    assert "ESTIMATED or IMPOSSIBLE" in listing.stdout
    done = subprocess.run([sys.executable, str(tool), "--db", str(tmp_path / "guard.db"),
                           "1", "2547.30", "--yes"], capture_output=True, text=True)
    assert done.returncode == 0 and "written" in done.stdout, done.stdout

    ctx = _ctx(**db.stats())
    assert ctx["unverified_settlements"] == 0
    assert _allowed(_ctx(**{**db.stats(), "trades_today": 0})), \
        "trading must resume once the row is reconciled"


def test_the_estimated_row_is_not_rewritable_twice(tmp_path):
    """The tool must refuse a row that is no longer an estimate (it could clobber a real one)."""
    db = _db(tmp_path)
    _closed(db, exit_price=1.35, pnl_pct=-11.5, pnl_net=-42.0,
            reason="unverified_exit", source="level_fallback")
    tool = ROOT / "tools" / "reconcile_settlement.py"
    subprocess.run([sys.executable, str(tool), "--db", str(tmp_path / "guard.db"),
                    "1", "2547.30", "--yes"], capture_output=True, text=True)
    again = subprocess.run([sys.executable, str(tool), "--db", str(tmp_path / "guard.db"),
                            "1", "1.300", "--yes"], capture_output=True, text=True)
    assert again.returncode == 1 and "refusing to rewrite" in again.stdout, again.stdout


def test_a_measured_settlement_never_halts_anything(tmp_path):
    db = _db(tmp_path)
    _closed(db, exit_price=1.42272, pnl_pct=8.0, pnl_net=29.3, reason="tp", source="fills")
    ctx = _ctx(**db.stats())
    assert ctx["unverified_settlements"] == 0
    assert _allowed(ctx)


# ── the protection-failure streak ──────────────────────────────────────────

def test_the_protection_failure_streak_clears_after_a_clean_close(tmp_path):
    db = _db(tmp_path)
    _closed(db, exit_price=2640.0, pnl_pct=-0.2, pnl_net=-0.7,
            reason="sl_placement_failed", source="provisional")
    assert db.stats()["sl_failure_streak"] == 1
    _closed(db, exit_price=1.42272, pnl_pct=8.0, pnl_net=29.3, reason="tp", source="fills")
    assert db.stats()["sl_failure_streak"] == 0, \
        "a normal close must end the protection-failure streak"
    # isolate the protection guard from the daily-trade cap (two closed rows is the cap itself)
    assert _allowed(_ctx(**{**db.stats(), "trades_today": 0}))
    assert G.g_sl_failure_streak(db.stats()["sl_failure_streak"], CFG)[0] is True


def test_the_streak_counts_only_CONCUTIVE_failures(tmp_path):
    db = _db(tmp_path)
    _closed(db, exit_price=1.35, pnl_pct=-3.0, pnl_net=-11.0,
            reason="sl_placement_failed", source="provisional")
    _closed(db, exit_price=2610.0, pnl_pct=-2.0, pnl_net=-7.0,
            reason="sl_missing_unprotected", source="provisional")
    assert db.stats()["sl_failure_streak"] == 2, \
        "both protection-failure reasons are the same safety fault"
    cfg = json.loads(json.dumps(CFG))
    assert not G.g_sl_failure_streak(2, cfg)[0]
    _closed(db, exit_price=1.42272, pnl_pct=8.0, pnl_net=29.3, reason="tp", source="fills")
    assert db.stats()["sl_failure_streak"] == 0


def test_the_halts_are_reported_to_the_operator_not_silent(tmp_path, monkeypatch):
    """Every halt must be visible: the reason string has to name the condition and the exit."""
    cfg = json.loads(json.dumps(CFG))
    _, unverified = G.g_unverified_settlements(1, cfg)
    assert "estimated" in unverified and "reconcile" in unverified
    _, streak = G.g_sl_failure_streak(2, cfg)
    assert "protective-stop" in streak
    # and the README documents the clearing command
    readme = (ROOT / "README.md").read_text()
    assert "tools/reconcile_settlement.py" in readme
    assert "--adopt-orphan-stop" in readme
