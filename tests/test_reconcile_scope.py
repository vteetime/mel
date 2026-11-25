"""The reconcile tool must only ever touch THIS bot's CLOSED rows.

Found by review (Paul, 2026-09-20) against Mel: `tools/reconcile_settlement.py` queried the
SHARED ledger (`/opt/data/cic/trades.db`) with neither a `bot_id` nor a `status='closed'` filter.
Live, that made `--list` announce "new entries are halted until reconciled" while listing FOUR OPEN
POSITIONS belonging to THREE OTHER BOTS — Jack's ETH long, SOL's short, BX's XRP long and Mel's own
open #355 — none of which was a settlement at all, and none of which Mel's guard was counting
(Mel's own status reported `unverified_settlements: 0`).

The dangerous direction is the write: an operator following the tool's own instructions
(`reconcile_settlement.py <id> <price> --yes`) would have written a fabricated `exit_price` and
`pnl_net` into another bot's OPEN row, corrupting that bot's guard accounting while leaving the row
open. These tests pin the scope in both directions: what it shows, and what it will write.
"""
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SCHEMA = """
CREATE TABLE trades (
  id INTEGER PRIMARY KEY AUTOINCREMENT, bot_id TEXT NOT NULL, exchange TEXT, symbol TEXT NOT NULL,
  direction TEXT NOT NULL, entry_price REAL NOT NULL, exit_price REAL, contracts REAL,
  leverage INTEGER, notional REAL, pnl_net REAL, pnl_pct REAL, fee_usdt REAL,
  entry_time TEXT, exit_time TEXT, exit_reason TEXT, status TEXT, settlement_source TEXT)
"""

BOT = "mel"


def _row(**kw) -> dict:
    base = dict(bot_id=BOT, exchange="kraken_futures", symbol="XRP/USD:USD", direction="long",
                entry_price=2573.6, exit_price=0.0, contracts=0.142, leverage=2, notional=365.4,
                pnl_net=0.0, pnl_pct=0.0, fee_usdt=0.0, entry_time="2026-09-01T00:00:00+00:00",
                exit_time="2026-09-01T01:00:00+00:00", exit_reason="level_fallback",
                status="closed", settlement_source="level_fallback")
    base.update(kw)
    return base


def _db(tmp_path: Path, rows: list[dict]) -> Path:
    p = tmp_path / "trades.db"
    con = sqlite3.connect(p)
    con.executescript(SCHEMA)
    for r in rows:
        con.execute(f"INSERT INTO trades ({','.join(r)}) VALUES ({','.join('?' * len(r))})",
                    list(r.values()))
    con.commit()
    con.close()
    return p


def _run(db: Path, *args):
    return subprocess.run([sys.executable, "tools/reconcile_settlement.py", "--db", str(db), *args],
                          capture_output=True, text=True, cwd=str(ROOT), timeout=120)


def _cell(db: Path, tid: int, col: str):
    con = sqlite3.connect(db)
    try:
        return con.execute(f"SELECT {col} FROM trades WHERE id=?", (tid,)).fetchone()[0]
    finally:
        con.close()


def test_another_bots_broken_row_is_invisible(tmp_path):
    """Jack's level-fallback row is Jack's problem: Mel's tool must not even see it."""
    db = _db(tmp_path, [_row(bot_id="jack", symbol="ETH/USDT:USDT")])
    r = _run(db, "--list")
    assert r.returncode == 0
    assert "nothing to reconcile" in r.stdout, r.stdout
    assert "jack" not in r.stdout


def test_an_open_position_is_invisible(tmp_path):
    """The live false alarm: Mel's own OPEN #355 is not a settlement and cannot be repriced."""
    db = _db(tmp_path, [_row(status="open", exit_price=0.0, settlement_source=None,
                             exit_reason=None, exit_time=None)])
    r = _run(db, "--list")
    assert "nothing to reconcile" in r.stdout, r.stdout


def test_it_refuses_to_rewrite_another_bots_row(tmp_path):
    """The money-path assertion: no write, and the message names whose row it is."""
    db = _db(tmp_path, [_row(bot_id="jack", symbol="ETH/USDT:USDT")])
    r = _run(db, "1", "2690.0", "--yes")
    assert r.returncode == 1, r.stdout + r.stderr
    assert "not 'mel'" in r.stdout and "another bot" in r.stdout, r.stdout
    assert _cell(db, 1, "exit_price") == 0.0, "another bot's row was rewritten"
    assert _cell(db, 1, "settlement_source") == "level_fallback"


def test_it_refuses_to_reprice_an_open_position(tmp_path):
    db = _db(tmp_path, [_row(status="open")])
    r = _run(db, "1", "2690.0", "--yes")
    assert r.returncode == 1
    assert "still open" in r.stdout, r.stdout
    assert _cell(db, 1, "settlement_source") == "level_fallback"


def test_a_real_mel_settlement_is_listed_then_corrected(tmp_path):
    """The happy path still works — and touches only its own row."""
    other = _row(bot_id="jack", symbol="ETH/USDT:USDT")
    db = _db(tmp_path, [_row(), other])
    listed = _run(db, "--list")
    assert "#1" in listed.stdout and "jack" not in listed.stdout, listed.stdout
    assert "not listed" in listed.stdout, "the operator should know other bots' rows are out of scope"

    dry = _run(db, "1", "2676.3")
    assert dry.returncode == 0 and "dry run" in dry.stdout, dry.stdout
    assert _cell(db, 1, "exit_price") == 0.0, "a dry run wrote"

    written = _run(db, "1", "2676.3", "--fee", "0.26", "--yes")
    assert written.returncode == 0, written.stdout + written.stderr
    assert _cell(db, 1, "exit_price") == 2676.3
    assert _cell(db, 1, "settlement_source") == "manual_reconcile"
    # the corrected P&L is the real fill, not the estimated level
    pnl = _cell(db, 1, "pnl_net")
    assert pnl == pytest.approx((2676.3 - 2573.6) * 0.142 - 0.26, rel=1e-3), pnl
    assert _cell(db, 2, "exit_price") == 0.0, "the other bot's row moved"

    assert "nothing to reconcile" in _run(db, "--list").stdout


def test_a_measured_row_is_still_refused(tmp_path):
    """Unchanged behaviour: a row with real evidence is not up for rewriting."""
    db = _db(tmp_path, [_row(settlement_source="exchange", exit_price=1.35)])
    r = _run(db, "1", "2690.0", "--yes")
    assert r.returncode == 1 and "refusing to rewrite a measured row" in r.stdout, r.stdout
    assert _cell(db, 1, "exit_price") == 1.35


def test_the_tool_scopes_to_the_id_that_actually_writes_the_rows():
    """Two sources of truth is a trap, and this is the one that bites silently.

    Ledger rows carry `db.BOT_ID`; `config.yaml`'s `bot.id` is a second, independent value. The
    first version of this fix read config.yaml, so a rebrand that edited one and not the other
    would scope the tool to a phantom id: it would report "nothing to reconcile" while a real
    level_fallback row kept the guard halting, with no working escape. (Flagged by review.)
    """
    import importlib.util

    import db as dbmod

    spec = importlib.util.spec_from_file_location("reconcile_under_test",
                                                  ROOT / "tools" / "reconcile_settlement.py")
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    assert tool._bot_id() == dbmod.BOT_ID, (
        "the tool must scope to the id the ledger actually carries, not to config.yaml")


def test_an_empty_or_null_bot_id_row_is_invisible(tmp_path):
    """`bot_id=?` never matches NULL or '' — an unowned row cannot be listed or repriced."""
    db = _db(tmp_path, [_row(bot_id="")])
    assert "nothing to reconcile" in _run(db, "--list").stdout
    assert _run(db, "1", "2690.0", "--yes").returncode == 1
    assert _cell(db, 1, "exit_price") == 0.0
