#!/usr/bin/env python3
"""What the VENUE says about Mel's position: `flat`, `open`, or `unprovable`. Read-only.

`run_mel.sh` passes this to the fleet pause guard (Tier 1) as `--venue`. The value must come from the
EXCHANGE, never from Mel's state file: at the moment we decide whether to SKIP a tick, the state file is
exactly the thing we do not trust, and a stale "flat" in it would skip a tick while the venue held a real
position with a resting stop (Kraken files stops separately from limit orders, so a state-file check can
also disagree with the venue about protection).

Prints ONE token and nothing else, because the runner captures stdout:

    flat        the venue reports no position
    open        the venue holds a position
    unprovable  the venue could not be read — never report this as `flat`

Always exits 0. The TOKEN is the output: a non-zero exit would be indistinguishable from a crashed
guard, and the runner's contract for a guard failure differs from its contract for an unreadable venue.

Writes nothing. Places, amends and cancels nothing — `dry_run=True` on the client means even a
mislabelled call cannot submit an order. `exchange.open_position()` reports a failed read as
`{"active": False, "error": ...}`, and this treats that as `unprovable`, never as flat; the same
distinction the engine makes internally when it must not reconcile a live position as closed.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def venue_state() -> str:
    try:
        import config as C
        import exchange as ex
        import run_bot as R

        cfg = C.load_config(os.environ.get("MEL_CONFIG") or (ROOT / "config.yaml"))
        sym = cfg["bot"]["symbol"]
        cl = ex.new_client(cfg, R.load_keys(), dry_run=True)
        pos = ex.open_position(cl, sym)
    except Exception as exc:            # noqa: BLE001 — any failure means "cannot see the venue"
        print(f"[pause] venue probe failed: {exc}", file=sys.stderr)
        return "unprovable"

    if pos.get("error"):
        print(f"[pause] venue read incomplete: {pos['error']}", file=sys.stderr)
        return "unprovable"
    return "open" if pos.get("active") else "flat"


if __name__ == "__main__":
    print(venue_state())
    sys.exit(0)
