"""Run tools/clear_halt.py for real — the documented way OUT of every halt must work for each one.

WHY THIS FILE EXISTS: a review (Paul, 2026-09-20, on commit aa3ed57) found that adding the
`loss_streak` branch to clear_halt.py had left the `sl_failure_streak` body INSIDE it. Two
consequences, both silent:

  * `--halt loss_streak --yes` wrote the ack for the consecutive-loss breaker AND the ack for the
    PROTECTION-FAILURE breaker — lifting the guard that exists because a venue stopped honouring
    stops (Trip: ~$226 in taker fees re-entering), without a human checking the venue.
  * `--halt sl_failure_streak --yes` had no branch at all: it fell through to the ambiguous-stop
    path, printed "nothing to clear", exited 0, and left the protection breaker WEDGED — the exact
    class of defect the tool exists to be the way out of.

225 tests passed while that was true, because nothing imported or ran this file. These tests run the
real CLI as a subprocess and assert that each halt touches EXACTLY its own flag, so a swallowed or
missing branch fails here rather than on a live bot.
"""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _halts() -> tuple:
    """`tools/` is not a package, so load the tuple from the file itself."""
    spec = importlib.util.spec_from_file_location("clear_halt_under_test",
                                                  ROOT / "tools" / "clear_halt.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return tuple(mod.HALTS)


HALTS = _halts()

ACK_FOR = {"sl_failure_streak": "ack_sl_failure_streak", "loss_streak": "ack_loss_streak"}
PENDING_LOCK = {"candidate_id": "abc123", "where": "tp", "contracts": 13, "trigger": 1.30}


def _run(state_path: Path, halt: str, *extra: str):
    return subprocess.run(
        [sys.executable, "tools/clear_halt.py", "--halt", halt, "--note", "reviewed the venue",
         "--yes", "--state", str(state_path), *extra],
        capture_output=True, text=True, cwd=str(ROOT), timeout=120)


def _flags(state_path: Path) -> dict:
    if not state_path.exists():
        return {}
    return json.loads(state_path.read_text()).get("flags", {})


@pytest.mark.parametrize("halt", sorted(ACK_FOR))
def test_each_acknowledgement_writes_only_its_own_flag(tmp_path, halt):
    """Clearing one breaker must never quietly clear another."""
    sp = tmp_path / "state.json"
    r = _run(sp, halt)
    assert r.returncode == 0, f"exit {r.returncode}: {r.stdout}{r.stderr}"
    flags = _flags(sp)
    expected = ACK_FOR[halt]
    assert expected in flags, f"--halt {halt} must write {expected}; flags are {sorted(flags)}"
    for other in ACK_FOR.values():
        if other != expected:
            assert other not in flags, (
                f"--halt {halt} ALSO cleared {other}. Clearing the protection-failure breaker "
                f"requires its own explicit assertion — that guard exists because a venue dropped "
                f"stops, and Trip's blind re-entry cost ~$226 in fees.")


@pytest.mark.parametrize("halt", sorted(ACK_FOR))
def test_the_note_is_recorded_as_the_audit_trail(tmp_path, halt):
    sp = tmp_path / "state.json"
    _run(sp, halt)
    assert "reviewed the venue" in json.dumps(_flags(sp)), "the note is the audit trail"


@pytest.mark.parametrize("halt", sorted(ACK_FOR))
def test_it_refuses_without_a_note_or_without_yes(tmp_path, halt):
    sp = tmp_path / "state.json"
    no_note = subprocess.run(
        [sys.executable, "tools/clear_halt.py", "--halt", halt, "--yes", "--state", str(sp)],
        capture_output=True, text=True, cwd=str(ROOT), timeout=120)
    assert no_note.returncode == 1 and "note" in (no_note.stdout + no_note.stderr).lower()
    dry = subprocess.run(
        [sys.executable, "tools/clear_halt.py", "--halt", halt, "--note", "x", "--state", str(sp)],
        capture_output=True, text=True, cwd=str(ROOT), timeout=120)
    assert dry.returncode == 0 and "dry run" in dry.stdout
    assert not _flags(sp), "a dry run must write nothing"


def test_no_halt_falls_through_to_the_ambiguous_stop_path(tmp_path):
    """A halt with no branch prints 'nothing to clear' and exits 0 — a silent no-op that leaves the
    breaker wedged. Every name in HALTS must do its own work."""
    for halt in HALTS:
        sp = tmp_path / f"{halt}.json"
        sp.write_text(json.dumps({"flags": {"possible_orphan_stop": dict(PENDING_LOCK)}}))
        r = _run(sp, halt)
        assert r.returncode == 0, f"--halt {halt}: exit {r.returncode} {r.stdout}{r.stderr}"
        assert "nothing to clear" not in r.stdout, (
            f"--halt {halt} fell through to the ambiguous-stop branch: no branch handles it, so "
            f"following the documented command leaves the breaker wedged")
        if halt == "ambiguous_stop":
            assert "resolved_ambiguous_stop" in _flags(sp)
        else:
            assert ACK_FOR[halt] in _flags(sp), f"--halt {halt} wrote nothing"
            assert "resolved_ambiguous_stop" not in _flags(sp), (
                f"--halt {halt} also cleared the ambiguous-stop lock — wrong-halt clear")


def test_an_ambiguous_stop_clear_needs_the_lock_to_exist(tmp_path):
    sp = tmp_path / "state.json"
    r = _run(sp, "ambiguous_stop")
    assert r.returncode == 0 and "nothing to clear" in r.stdout


def test_list_shows_every_breaker_and_its_ack(tmp_path):
    sp = tmp_path / "state.json"
    r = subprocess.run([sys.executable, "tools/clear_halt.py", "--list", "--state", str(sp)],
                       capture_output=True, text=True, cwd=str(ROOT), timeout=120)
    out = r.stdout + r.stderr
    assert r.returncode == 0
    for name in HALTS:
        assert name in out, f"--list must mention {name}: an operator cannot see what is set"
