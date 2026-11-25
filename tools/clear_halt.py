#!/usr/bin/env python3
"""Clear a safety HALT that would otherwise be permanent, with an audited acknowledgement.

Mel's halts are fail-closed, which is right — but a halt whose only natural reset is an event
the halt itself prevents is a wedge. Two of them had no way out:

  * `sl_failure_streak` — blocks entries after consecutive protective-stop failures, and its
    documented reset was "the next clean close", which a halted bot cannot produce.
  * an ambiguous protective stop that can be neither found nor disproved (`possible_orphan_stop`).
  * `loss_streak` — the consecutive-loss circuit breaker. Its reset is now the UTC day boundary
    (2026-09-20), because a win-only reset was a permanent wedge: the block prevents the very trade
    that would produce the reset. This override is for an operator who does not want to wait for
    midnight; it expires by construction when a new loss lands.

Both are cleared by a human ASSERTION about the venue, so both are recorded with what was
asserted, when, and why. The acknowledgement expires by construction: any NEW protection failure
moves the bar past it.

    python3 tools/clear_halt.py --list
    python3 tools/clear_halt.py --halt sl_failure_streak --note "venue blip; venue shows no stops" --yes
    python3 tools/clear_halt.py --halt ambiguous_stop --note "checked Kraken UI: no stop resting" --yes
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

HALTS = ("sl_failure_streak", "ambiguous_stop", "loss_streak")


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _state_path(cfg: dict, override: str | None) -> Path:
    return Path(override) if override else ROOT / str(cfg["ops"]["state_file"])


def _cfg() -> dict:
    import yaml
    return yaml.safe_load((ROOT / "config.yaml").read_text())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--halt", choices=HALTS)
    ap.add_argument("--note", default="", help="what you checked and concluded (recorded)")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--yes", action="store_true", help="actually write the acknowledgement")
    ap.add_argument("--state", default=None)
    a = ap.parse_args()

    from state import State
    st = State(_state_path(_cfg(), a.state))

    if a.list or not a.halt:
        lock = st.get_flag("possible_orphan_stop") or {}
        ack = st.get_flag("ack_sl_failure_streak") or {}
        print("halts that need a human:")
        print(f"  ambiguous_stop    : {'PENDING  ' + json.dumps(lock)[:110] if lock else 'not set'}")
        print(f"  sl_failure_streak : ack {ack.get('at') or 'none'}"
              f"{' — ' + str(ack.get('note')) if ack.get('note') else ''}")
        lack = st.get_flag("ack_loss_streak") or {}
        print(f"  loss_streak       : ack {lack.get('at') or 'none'}"
              f"{' — ' + str(lack.get('note')) if lack.get('note') else ''}"
              f"  (the breaker also resets by itself at the next UTC day boundary)")
        if lock or not ack:
            print("\nrun:  python3 tools/clear_halt.py --halt <name> --note \"what you checked\" --yes")
        return 0

    if not a.note.strip():
        print("a --note is REQUIRED: this is a human assertion about the venue, and the note is "
              "the audit trail")
        return 1
    if not a.yes:
        print(f"dry run — re-run with --yes to clear '{a.halt}'")
        return 0

    if a.halt == "sl_failure_streak":
        st.set_flag("ack_sl_failure_streak",
                    {"at": _iso_now(), "note": a.note.strip(), "by": "operator"})
        print(f"acknowledged at {_iso_now()} — the protection-failure streak is cleared until a NEW "
              f"failure\n  note: {a.note.strip()}")
        return 0

    if a.halt == "loss_streak":
        st.set_flag("ack_loss_streak",
                    {"at": _iso_now(), "note": a.note.strip(), "by": "operator"})
        print(f"acknowledged at {_iso_now()} — the consecutive-loss breaker is cleared until a "
              f"NEW loss\n  note: {a.note.strip()}\n  (it would also have reset at the next UTC "
              f"day boundary)")
        return 0

    if a.halt != "ambiguous_stop":
        # Unreachable while --halt is limited to HALTS. It exists because a halt name added to that
        # tuple WITHOUT a branch here would otherwise fall through and silently print "nothing to
        # clear" — the exact wedge this tool is supposed to be the way OUT of.
        print(f"unknown halt {a.halt!r} — nothing cleared. Known: {', '.join(HALTS)}")
        return 2

    lock = st.get_flag("possible_orphan_stop")
    if not lock:
        print("nothing to clear: no ambiguous stop is pending")
        return 0
    st.set_flag("resolved_ambiguous_stop",
                {"at": _iso_now(), "note": a.note.strip(), "had_candidate":
                 str(lock.get("candidate_id") or ""), "lock": lock})
    st.clear_flag("possible_orphan_stop")
    print(f"cleared the ambiguous-stop lock ({lock.get('where')}, {lock.get('contracts')} @ "
          f"{lock.get('trigger')})\n  note: {a.note.strip()}\n  the resolution is recorded in "
          f"state.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
