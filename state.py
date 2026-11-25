"""State — the bot's own position state, persisted atomically.

Separate file from every other bot. Writes are atomic (tempfile + os.replace) so a
crash mid-write can never leave a truncated JSON that the next tick reads as "flat"
while a real position is open.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path

log = logging.getLogger("mel.state")


class State:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._d: dict = {"open": None, "halted": False, "halt_reason": "", "flags": {}}
        self.load()

    def load(self) -> dict:
        if self.path.exists():
            try:
                self._d = json.loads(self.path.read_text())
            except (json.JSONDecodeError, OSError) as exc:
                # Never silently treat a corrupt file as "no position" — that would
                # orphan a live position. Keep the corrupt copy and fail loudly.
                bad = self.path.with_suffix(self.path.suffix + ".corrupt")
                try:
                    self.path.replace(bad)
                except OSError:
                    pass
                raise RuntimeError(
                    f"state file {self.path} was unreadable ({exc}); moved to {bad}. "
                    "Refusing to continue — check for an open position on the exchange."
                ) from exc
        return self._d

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(self.path.parent), prefix=".state-")
        try:
            with os.fdopen(fd, "w") as fh:
                json.dump(self._d, fh, indent=1)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, self.path)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    # ── accessors ────────────────────────────────────────────────────────────
    def get_open(self) -> dict | None:
        return self._d.get("open")

    def set_open(self, pos: dict) -> None:
        self._d["open"] = pos
        self.save()

    def clear(self) -> None:
        self._d["open"] = None
        self.save()

    def is_flat(self) -> bool:
        return not self._d.get("open")

    def halt(self, reason: str) -> None:
        self._d["halted"] = True
        self._d["halt_reason"] = reason
        self.save()

    def unhalt(self) -> None:
        self._d["halted"] = False
        self._d["halt_reason"] = ""
        self.save()

    # ── flags ────────────────────────────────────────────────────────────────
    # Small named facts that must outlive a tick: an unresolved dangling protective
    # order, for instance. Kept in the same atomic file as the position so a crash
    # cannot separate the two.
    def set_flag(self, key: str, value) -> None:
        """Store a flag. `None` (or a bare empty container) clears it.

        `False` is a VALUE, not an absence: "the venue does not show our stops" and "we do
        not know yet" are different facts, and collapsing them made the bot re-announce an
        already-learned limitation on every tick. Use clear_flag() to remove a flag.
        """
        flags = self._d.setdefault("flags", {})
        if value is None or value == "" or value == [] or value == {}:
            flags.pop(key, None)
        else:
            flags[key] = value
        self.save()

    def get_flag(self, key: str, default=None):
        return (self._d.get("flags") or {}).get(key, default)

    def clear_flag(self, key: str) -> None:
        self.set_flag(key, None)

    # ── the guard-rail gate (dashboard pause card, reporting-only) ──────────────
    def set_gate(self, gate: dict | None) -> None:
        """Publish (or clear) the guard-rail gate the dashboard card reads.

        Derived from the SAME guard functions that blocked the entry (see
        Engine.rail_state), so the card can never disagree with the rail that fired.
        ``None`` clears it. A reporting write must never be able to stop a tick from
        saving its position state, so callers wrap this defensively.
        """
        if gate:
            self._d["gate"] = gate
        else:
            self._d.pop("gate", None)
        self.save()

    @property
    def flags(self) -> dict:
        return dict(self._d.get("flags") or {})

    @property
    def halted(self) -> bool:
        return bool(self._d.get("halted"))

    @property
    def halt_reason(self) -> str:
        return str(self._d.get("halt_reason") or "")
