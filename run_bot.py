#!/usr/bin/env python3
"""mel — one tick. Cron-driven, every 1 hour.

    python run_bot.py --dry-run   # DEFAULT: decide and log, place NOTHING
    python run_bot.py --live      # real orders (also requires the LIVE sentinel)
    python run_bot.py --sync      # read-only: what does Kraken actually hold?
    python run_bot.py --status    # state + guard stats

Dry-run is the default so a misconfigured cron cannot trade by accident. Going live
requires the explicit flag, and the flag alone is not enough: the LIVE sentinel file must
also exist. Two independent switches, both deliberate.

Exit codes: 0 ok/no-op, 1 error. A silence-preserving tick prints nothing on HOLD/WAIT so
the cron log stays readable.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import logging
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import config as C            # noqa: E402
import exchange as ex         # noqa: E402
import sync as sync_mod       # noqa: E402
from db import TradeDB        # noqa: E402
from engine import Engine, read_override_until  # noqa: E402
from guards import run_guards  # noqa: E402
from notify import alert_once, make_notifier  # noqa: E402
from state import State       # noqa: E402
from strategy import build_market_state, decide  # noqa: E402

log = logging.getLogger("mel")


def setup_logging(cfg: dict) -> None:
    d = ROOT / cfg["ops"].get("log_dir", "logs")
    d.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(d / "mel.log")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    logging.basicConfig(level=logging.INFO, handlers=[fh, sh])


def load_env_file(path: Path) -> dict:
    """Minimal .env reader — no python-dotenv dependency, no shell involved."""
    out: dict = {}
    try:
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip().strip('"').strip("'")
    except OSError:
        pass
    return out


def load_keys() -> dict:
    """Kraken futures credentials.

    Resolution: process environment first, then THIS directory's `.env`. Unlike Jack,
    there is no fallback to another project's `.env` — Mel is standalone, and silently
    borrowing a different bot's credentials is how a trade lands on the wrong account.
    Secrets are never logged, never written to state, never put in an alert.
    """
    keys = {"api_key": os.environ.get("KRAKEN_API_KEY", ""),
            "api_secret": os.environ.get("KRAKEN_API_SECRET", "")}
    if all(keys.values()):
        return keys
    local = load_env_file(ROOT / ".env")
    keys["api_key"] = keys["api_key"] or local.get("KRAKEN_API_KEY", "")
    keys["api_secret"] = keys["api_secret"] or local.get("KRAKEN_API_SECRET", "")
    return keys


def resolve_dry_run(args_live: bool, root=None) -> bool:
    """[MONEY] Decide whether this invocation may place REAL orders.

    Two deliberate acts are required: the `--live` flag AND the LIVE sentinel file. The
    cron wrapper gates on the sentinel, but a manual `run_bot.py --live` would bypass
    that and be one flag away from real money. Requiring the sentinel in both paths means
    an accidental or copy-pasted flag cannot arm the bot.
    """
    live_sentinel = (root or ROOT) / "LIVE"
    if args_live and not live_sentinel.exists():
        log.error("--live given but %s does not exist — running DRY. "
                  "Create the sentinel to arm real orders.", live_sentinel)
    return not (args_live and live_sentinel.exists())


# A manage-only tick WAITS this long for a competing tick before giving up loudly. It has to
# wait at all: the pass it runs is the hold-cap check, and silently skipping that pass (the
# entry-tick behaviour) would make the tighter cap bound false with nothing saying so.
MANAGE_ONLY_LOCK_WAIT_SECONDS = 240.0


class LockBusy(RuntimeError):
    """Another Mel tick holds the single-tick lock (see acquire_lock)."""


def acquire_lock(path: str, wait_seconds: float = 0.0):
    """Take the single-tick lock, or raise LockBusy.

    A normal (entry) tick never waits: another tick is already doing the same work, so a
    duplicate is worthless and skipping it is correct.

    A MANAGE-ONLY tick is different, and waiting is the point of it. Its job is to evaluate
    the hold cap more often than hourly, and a silent "skip because busy" would discard
    exactly that pass whenever the hourly tick overran its slot — the promised <=15-minute
    cap bound would be false and nothing would say so (Terra round-1 must-fix). So it WAITS
    (bounded), then runs the management pass it was created for; if the lock is still held at
    the bound the caller fails loudly rather than pretending the work was done.
    """
    fh = open(path, "w")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return fh
    except OSError:
        if wait_seconds <= 0:
            fh.close()
            raise LockBusy("another mel tick is running")

    deadline = time.monotonic() + float(wait_seconds)
    warned = False
    while True:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if warned:
                log.warning("manage-only tick: lock acquired after waiting — running the "
                            "deferred management pass now")
            return fh
        except OSError:
            if not warned:
                log.warning("manage-only tick: another mel tick still holds %s — waiting up "
                            "to %.0fs for it (skipping would drop this cap check silently)",
                            path, float(wait_seconds))
                warned = True
            if time.monotonic() >= deadline:
                fh.close()
                raise LockBusy(f"still held after {float(wait_seconds):.0f}s")
            time.sleep(0.25)



def effective_loss_streak(stats: dict, state) -> int:
    """The consecutive-loss streak a HUMAN has already cleared, if any.

    The guard's natural reset is now the UTC day boundary; this adds an operator override for the
    case where waiting is not acceptable. It expires BY CONSTRUCTION: any new loss moves
    `last_loss_at` past the acknowledgement, so the breaker re-arms itself.
    """
    streak = int(stats.get("consecutive_losses") or 0)
    ack = state.get_flag("ack_loss_streak") or {}
    ack_at = str(ack.get("at") or "")
    last_loss = str(stats.get("last_loss_at") or "")
    if streak and ack_at and (not last_loss or ack_at > last_loss):
        log.warning("consecutive-loss streak of %d acknowledged by %s at %s (%s) — cleared",
                    streak, ack.get("by") or "operator", ack_at, ack.get("note") or "")
        return 0
    return streak


def effective_protection_streak(stats: dict, state) -> int:
    """The protection-failure streak a HUMAN has already dealt with, if any.

    The guard's natural reset is a later clean close — which a halted bot cannot produce, so
    without this the halt was permanent (Terra round-6 finding 4). An operator
    acknowledgement (`tools/clear_halt.py`) clears it, and expires BY CONSTRUCTION: any new
    protection failure moves `last_protection_failure_at` past the acknowledgement.
    """
    streak = int(stats.get("sl_failure_streak") or 0)
    ack = state.get_flag("ack_sl_failure_streak") or {}
    ack_at = str(ack.get("at") or "")
    last_fail = str(stats.get("last_protection_failure_at") or "")
    if streak and ack_at and (not last_fail or ack_at > last_fail):
        log.warning("protection-failure streak of %d acknowledged by %s at %s (%s) — cleared",
                    streak, ack.get("by") or "operator", ack_at, ack.get("note") or "")
        return 0
    return streak


def _iso_now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def do_sync(cfg: dict, keys: dict, dry_run: bool = True, cancel_dangling: bool = False,
            adopt_orphan_stop: bool = False, resolve_ambiguous_stop: bool = False,
            clear_dangling: bool = False, note: str = "") -> int:
    """Read-only reconciliation — 'what is actually left on the exchange?'

    Deliberately uses a LIVE-capable client, not the dry-run one: every call here is a
    read, and a dry-run client cannot see the real account.

    [MONEY] `--cancel-dangling` is the ONE destructive action in this path, and it is only
    reachable when the bot is ARMED (`--live` + the LIVE sentinel). It used to be reachable
    under `--dry-run`, which meant the documented "a dry run places and cancels nothing"
    guarantee was false for a command that strips real protective orders off the venue.
    """
    sym = cfg["bot"]["ccxt_symbol"]
    state = State(ROOT / cfg["ops"]["state_file"])
    notify = make_notifier(cfg, enabled=not dry_run)   # sync reports are operator-facing
    client = ex.new_client(cfg, keys, dry_run=False)
    snap = sync_mod.snapshot(client, sym)
    verdict = sync_mod.reconcile(snap, state.get_open())
    print(sync_mod.format_report(snap, verdict, cfg))

    if resolve_ambiguous_stop:
        # [MONEY] The last resort for an ambiguous stop that CANNOT be found OR disproved: the
        # bot cannot clear it, and without a path out it blocks entries for ever. Only a human
        # can assert "I looked at the venue and there is no such stop". Armed, audited, and it
        # says plainly in the log and the alert that a person made that call.
        if dry_run:
            print("\nREFUSED: --resolve-ambiguous-stop is an operator assertion about the "
                  "venue, so it needs an ARMED bot (--live AND the LIVE sentinel).")
            return 1
        lock = state.get_flag("possible_orphan_stop")
        if not lock:
            print("\n--resolve-ambiguous-stop: nothing to resolve (no ambiguous stop pending).")
            return 1
        st = State(ROOT / cfg["ops"]["state_file"])
        st.set_flag("resolved_ambiguous_stop",
                    {"at": _iso_now(), "note": note,
                     "had_candidate": str(lock.get("candidate_id") or ""),
                     "lock": lock})
        st.clear_flag("possible_orphan_stop")
        print(f"\nresolved and cleared the ambiguous-stop lock "
              f"({lock.get('where')}, {lock.get('contracts')} @ {lock.get('trigger')})."
              f"\n  NOTE: {note or '(none given)'}"
              f"\n  Entries are unblocked; the resolution is recorded in state.json.")
        return 0

    if clear_dangling:
        # [MONEY] The last resort for a dangling flag the venue will not resolve by itself. A
        # human asserts "I have looked at the exchange and these orders are gone"; the assertion
        # is recorded with the note and the per-id verdict, so it is auditable rather than a
        # silent reset. Without it, a venue that cannot answer by id blocks entries for ever.
        if dry_run:
            print("\nREFUSED: --clear-dangling overrides a safety block, so it needs an ARMED "
                  "bot (--live AND the LIVE sentinel).")
            return 1
        if not str(note or "").strip():
            print("\nREFUSED: --clear-dangling requires --note with what you checked: the note "
                  "IS the audit trail for overriding a protection block.")
            return 1
        # [MONEY] REFUSE WHILE A POSITION IS OPEN. `stale` below includes ids the venue reports
        # as STILL RESTING, and while a position is live those are the orders protecting it right
        # now: clearing would drop them from the tracking history and file them as "asserted
        # gone". Nothing is gained by allowing it — the flag gates ENTRIES, not exits, so an open
        # position is not even the case this command exists for. `--cancel-dangling` refuses on an
        # open position for the same reason.
        _pos = snap.get("position") or {}
        # [MONEY] A FAILED position read is not evidence of flatness. `open_position` returns
        # {"active": False, "error": ...} when the venue does not answer, so checking `active`
        # alone let this command run during an outage with a position possibly live — precisely
        # the condition the dangling flag exists for. Refuse: flatness has to be proven, never
        # assumed from a failure to look.
        if _pos.get("error"):
            print(f"\nREFUSED: --clear-dangling cannot prove the account is flat: "
                  f"{str(_pos.get('error'))[:120]}"
                  "\n  An unreadable position is not a flat position. Retry when the venue"
                  "\n  answers, or use the flat-state gate's own path once it does.")
            return 1
        if _pos.get("active") and float(_pos.get("contracts") or 0) > 0:
            print("\nREFUSED: --clear-dangling while a position is OPEN."
                  "\n  This command records ids as asserted gone, and an open position's"
                  "\n  resting SL/TP ids are exactly the ids it would record."
                  "\n  Nothing is blocked by waiting: the flag gates ENTRIES, not exits.")
            return 1
        ids = [str(i) for i in (state.get_flag("dangling_orders") or [])]
        # [MONEY] The flat-state gate checks `placed_ids`, not just `dangling_orders`, so
        # clearing one and not the other left the operator's documented escape half-implemented:
        # they would follow the procedure and the bot would stay blocked. Both are resolved here,
        # and both sets are recorded so the assertion is reconstructable.
        hist = [str(i) for i in (state.get_flag("placed_ids") or []) if i]
        # [MONEY] Label every id by what the VENUE says, one at a time. Reporting them all as
        # "unanswerable" hid the dangerous half: an id the venue reports as `open` is STILL
        # RESTING, and the whole safety of this escape is that the operator knows which ids they
        # are asserting away. `unknown` (a read that failed) is the safe direction.
        states = {i: ex.fetch_order_state(client, sym, i)[0]
                  for i in sorted(set(ids) | set(hist))}
        stale = [i for i in hist if states.get(i) not in ("finished", "missing")]
        if not ids and not stale:
            print("\n--clear-dangling: nothing to clear (no dangling protective orders, and"
                  "\n  every id this bot placed is already provably finished).")
            return 0
        # dedupe: an id usually appears in BOTH lists (entry residuals are remembered in both),
        # and counting it twice inflated the still-resting/unanswerable numbers and the alert
        touched = list(dict.fromkeys(ids + stale))
        still_resting = [i for i in touched if states.get(i) == "open"]
        unanswerable = [i for i in touched if states.get(i) == "unknown"]
        provable = [i for i in ids if states.get(i) in ("finished", "missing")]
        record = {"at": _iso_now(), "note": str(note).strip(), "ids": ids,
                  "provably_finished": {i: states.get(i) in ("finished", "missing") for i in ids},
                  "placed_ids_asserted_gone": stale, "still_resting": still_resting,
                  "unanswerable": unanswerable, "by": "operator"}
        # [AUDIT] Append-only. Last-write-wins means a second clearance destroys the first one's
        # evidence, and "reconstructable months later" has to survive more than one.
        history = list(state.get_flag("cleared_dangling_history") or [])
        history.append(record)
        del history[:-20]
        state.set_flag("cleared_dangling_history", history)
        state.set_flag("cleared_dangling", record)
        state.clear_flag("dangling_orders")
        if stale:
            state.set_flag("placed_ids", [i for i in hist if i not in stale])
        out = [f"\ncleared {len(ids)} dangling id(s) after an operator assertion.",
               f"\n  note: {str(note).strip()}",
               f"\n  the venue could PROVE only {len(provable)} of them finished; "
               f"{len(ids) - len(provable)} rest on your word alone."]
        if still_resting:
            out.append(f"\n  🛑 {len(still_resting)} of them the venue reports as STILL RESTING "
                       f"({', '.join(i[:12] for i in still_resting[:3])}) — you have asserted "
                       f"that orders which look live are gone.")
        if unanswerable:
            out.append(f"\n  {len(unanswerable)} could not be answered at all (a failed read, "
                       f"not a 'no such order').")
        if stale:
            out.append(f"\n  {len(stale)} id(s) this bot placed were not provably finished and "
                       f"are now asserted gone too: "
                       f"{', '.join(i[:12] for i in stale[:3])}")
        print("".join(out))
        notify(f"🧹 Mel — an operator asserted that {len(ids)} dangling protective order(s) are "
               f"gone ({len(ids) - len(provable)} of them the venue could not prove"
               + (f", {len(still_resting)} of which it reports as STILL RESTING"
                  if still_resting else "")
               + f"), plus {len(stale)} non-finished id(s) from the placed-id history. Entries "
                 f"unblock; the assertion is recorded in state.json.")
        return 0

    if adopt_orphan_stop:
        # [MONEY] The ONE way a shape-matched stop becomes ours. Mel never infers ownership
        # from submission intent + a matching side/size/trigger — a manual stop with the same
        # shape is indistinguishable after a timeout, and recording its id would mean this bot
        # later cancels an order it never placed. A human looks at the venue and decides here.
        if dry_run:
            print("\nREFUSED: --adopt-orphan-stop declares an order to be OURS (its id is "
                  "then tracked and cancellable), so it needs an ARMED bot "
                  "(--live AND the LIVE sentinel).")
            return 1
        lock = state.get_flag("possible_orphan_stop") or {}
        cid = str(lock.get("candidate_id") or "")
        if not cid:
            print("\n--adopt-orphan-stop: nothing to adopt (no ambiguous stop is pending).")
            return 1
        st = State(ROOT / cfg["ops"]["state_file"])
        hist = [str(x) for x in (st.get_flag("placed_ids") or []) if x]
        if cid not in hist:
            hist.append(cid)
        st.set_flag("placed_ids", hist[-40:])
        st.clear_flag("possible_orphan_stop")
        try:
            remaining = [o for o in ex.open_orders(client, sym, strict=True)
                         if str(o.get("id") or "") == cid]
        except Exception as exc:
            print(f"\ncannot re-read the book to confirm {cid[:12]}… still rests ({exc}) — "
                  f"adopted it as ours anyway; it is now cancellable.")
            return 0
        print(f"\nadopted {cid[:12]}… as OURS"
              f"{' (confirmed still resting)' if remaining else ' (no longer resting)'}."
              f"\n  It is now tracked in placed_ids and can be cancelled by a normal"
              f" retirement or `--sync --cancel-dangling`.")
        return 0

    if cancel_dangling:
        if dry_run:
            print("\nREFUSED: --cancel-dangling needs an ARMED bot (--live AND the LIVE "
                  "sentinel).\n  Cancelling is a real action on the venue, so it is not "
                  "available in dry-run mode.\n  To inspect only, drop the flag; to clear "
                  "the book deliberately, arm the bot first.")
            return 1
        if verdict.get("action") == "cancel_dangling":
            # Only ids THIS BOT persisted as unresolved are candidates. Without them the
            # command refuses: cancelling an order we cannot prove we placed is exactly the
            # mistake the ownership rule exists to prevent.
            out = sync_mod.cancel_dangling(
                client, snap, sym,
                owned_ids=State(ROOT / cfg["ops"]["state_file"]).get_flag("dangling_orders"))
            print(f"\ncancelled: {out}")
            return 0
        print(f"\n--cancel-dangling ignored: verdict is "
              f"{verdict.get('action')} (a live position keeps its protection)")
    if not verdict.get("readable", True):
        return 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="mel — Kraken XRP/USD:USD (single TP + SL)")
    ap.add_argument("--live", action="store_true", help="place real orders")
    ap.add_argument("--dry-run", action="store_true", help="decide only (default)")
    ap.add_argument("--sync", action="store_true",
                    help="read-only: reconcile against Kraken and print what it holds")
    ap.add_argument("--clear-dangling", action="store_true", dest="clear_dangling",
                    help="with --sync: assert (armed, audited, needs --note) that the dangling "
                         "protective orders are gone, unblocking entries")
    ap.add_argument("--resolve-ambiguous-stop", action="store_true",
                    dest="resolve_ambiguous_stop",
                    help="with --sync: assert (armed, audited) that no ambiguous stop is "
                         "resting, clearing a lock the bot cannot clear by itself")
    ap.add_argument("--note", default="", help="with --resolve-ambiguous-stop: why")
    ap.add_argument("--adopt-orphan-stop", action="store_true", dest="adopt_orphan_stop",
                    help="with --sync: declare an ambiguous candidate stop to be OURS "
                         "(armed only; it becomes tracked and cancellable)")
    ap.add_argument("--cancel-dangling", action="store_true",
                    help="with --sync: cancel resting orders while truly flat")
    ap.add_argument("--status", action="store_true", help="print state and exit")
    ap.add_argument("--manage-only", action="store_true", dest="manage_only",
                    help="manage an open position only: never evaluate or open an entry")
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    args = ap.parse_args()

    cfg = C.load_config(args.config)
    setup_logging(cfg)
    dry_run = resolve_dry_run(args.live)
    manage_only = bool(args.manage_only)

    sym = cfg["bot"]["ccxt_symbol"]
    state = State(ROOT / cfg["ops"]["state_file"])
    db = TradeDB(cfg["ops"]["db_path"])
    notify = make_notifier(cfg, enabled=not dry_run)

    if args.status:
        stats = db.stats()
        print(json.dumps({"open": state.get_open(), "halted": state.halted,
                          "stats": stats, "mode": "dry_run" if dry_run else "LIVE",
                          "bot": cfg["bot"]["id"], "symbol": sym}, indent=2, default=str))
        return 0

    try:
        lock = acquire_lock(cfg["ops"]["lock_file"],
                            wait_seconds=(MANAGE_ONLY_LOCK_WAIT_SECONDS if manage_only else 0.0))
    except LockBusy as exc:
        if not manage_only:
            print("another mel tick is running — skipping")
            return 0
        # [MONEY] This pass did NOT evaluate the hold cap or re-verify the stop. Say so in the
        # log, on Telegram (once per alert cooldown, not 4x/hour) and in the exit code — a
        # green tick that silently skipped the cap is the exact defect this job exists to fix.
        log.error("manage-only tick: lock not acquired (%s) — management NOT run", exc)
        try:
            alert_once(state, "manage_only_lock_stuck", notify,
                       f"🛑 Mel — a manage-only tick could not take the tick lock ({exc}), so "
                       f"the hold cap and the stop re-check were NOT evaluated this pass. "
                       f"Another tick is holding it too long. Inspect with `--sync`.")
        except Exception:
            log.warning("could not send the stuck-lock alert", exc_info=True)
        print(f"manage-only tick could not acquire {cfg['ops']['lock_file']} ({exc}) — "
              f"management pass NOT run")
        return 3
    try:
        keys = load_keys()
        if not keys["api_key"] or not keys["api_secret"]:
            log.error("KRAKEN_API_KEY / KRAKEN_API_SECRET are not set (env or %s/.env)",
                      ROOT)
            return 1

        if args.clear_dangling and not args.sync:
            log.error("--clear-dangling only means something with --sync")
            return 1
        if args.resolve_ambiguous_stop and not args.sync:
            log.error("--resolve-ambiguous-stop only means something with --sync")
            return 1
        if args.adopt_orphan_stop and not args.sync:
            log.error("--adopt-orphan-stop only means something with --sync; refusing to "
                      "start a trading tick with it")
            return 1
        if args.cancel_dangling and not args.sync:
            log.error("--cancel-dangling only means something with --sync; refusing to "
                      "start a trading tick with it")
            return 1
        if args.sync:
            return do_sync(cfg, keys, dry_run=dry_run,
                           cancel_dangling=args.cancel_dangling,
                           adopt_orphan_stop=args.adopt_orphan_stop,
                           resolve_ambiguous_stop=args.resolve_ambiguous_stop,
                           clear_dangling=args.clear_dangling,
                           note=args.note)

        try:
            client = ex.new_client(cfg, keys, dry_run=dry_run)
        except ex.VenueUnavailable as exc:
            log.error("venue unreachable: %s — not trading this tick", exc)
            alert_once(state, "venue_unavailable", notify,
                       f"🟠 Mel — can't reach Kraken Futures this tick ({exc}). "
                       f"Not trading; the position stays protected by its resting "
                       f"stop on the venue. Retrying next tick.")
            return 0
        # [MONEY] Record the leverage the venue is told, so it is auditable against the
        # sizing that produced the order. Kraken one-way futures takes no hedged /
        # marginMode param, so those Jack-era settings are gone rather than sent blind.
        ex.configure_orders(leverage=cfg["sizing"]["leverage"])
        if not dry_run:
            # [MONEY] Verify, don't assume: every size, margin fraction and equity cap is
            # built on this number, and `set_leverage` only logs its failures.
            lev_ok, lev_why = ex.ensure_leverage(client, sym,
                                                 int(cfg["sizing"]["leverage"]))
            if not lev_ok:
                log.critical("leverage mismatch: %s — refusing to trade", lev_why)
                alert_once(state, "leverage_mismatch", notify,
                           f"🛑 Mel — the venue will apply a different leverage than the "
                           f"configured one ({lev_why}). Every position size and loss cap "
                           f"assumes the configured value, so NO entry is taken until this "
                           f"matches. Check the account's leverage settings.")
                return 1
            log.info("leverage: %s", lev_why)
        engine = Engine(cfg, client, state, db, notify, dry_run=dry_run)

        # ── 0. AN UNRESOLVED AMBIGUOUS STOP (we submitted it, we never got its id) ──
        # [MONEY] This gates ENTRIES only, never the whole tick. An ambiguous stop can coexist
        # with an ambiguous force-close: the venue may still hold the position while state is
        # flat, and returning here would skip the orphan guard and `manage()` below — leaving a
        # live position naked until a human runs `--sync`. The block is applied AFTER management
        # (see "1b") exactly like the ledger backlog.
        entries_blocked_by_stop = False
        pending_stop = state.get_flag("possible_orphan_stop")
        if pending_stop and not dry_run:
            verdict_stop = engine.reconcile_possible_orphan_stop()
            if verdict_stop in ("unresolved", "unresolved_blind", "candidate"):
                log.error("an ambiguous protective stop cannot be located or disproved "
                          "(%s) — blocking new entries", verdict_stop)
                alert_once(state, "ambiguous_stop_unresolved", notify,
                           f"🛑 Mel — a stop whose submission went unanswered cannot be "
                           f"found OR disproved on {cfg['bot']['symbol']} "
                           f"({verdict_stop}). NOT opening new positions: it may be resting "
                           f"untracked and could close a future one. Inspect with `--sync`.")
                entries_blocked_by_stop = True

        # ── 0. LEDGER BACKLOG — a close that could not be written is retried, never lost ──
        # A force-closed trade whose row failed to write would otherwise be invisible to
        # every guard rail: trade count, cooldown, the loss streak, the daily cap and the
        # drawdown halt all read closed rows.
        if state.get_flag("pending_close"):
            try:
                res = engine.retry_pending_close()
                log.info("pending close retry: %s", res)
            except Exception as exc:
                log.exception("pending close retry failed: %s", exc)
        entries_blocked_by_backlog, backlog_left = False, 0
        if state.get_flag("pending_close") and not dry_run:
            # [MONEY] A close that still cannot be written means a REAL closed trade is
            # invisible to every guard rail — including the protection-failure breaker that
            # exists to stop re-entry into broken protection. Entering now recreates Trip's
            # blind churn, so entries are blocked until the ledger catches up.
            #
            # [MONEY] This used to RETURN from the whole tick. That was safe only while every
            # queued close implied a CONFIRMED-flat position. An unconfirmed force-close
            # (`close_unverified`) can be queued while the venue still holds the position, so
            # returning here would skip the orphan guard and `manage()` below and leave a real
            # position unmanaged and naked. The block is now an ENTRIES-only gate, applied
            # after management (see "1b").
            backlog_left = len(state.get_flag("pending_close") or [])
            log.error("%d close(s) still unwritten — entries will be blocked", backlog_left)
            alert_once(state, "unwritten_close", notify,
                       f"🛑 Mel — {backlog_left} closed trade(s) cannot be written to the "
                       f"ledger, so the guard rails cannot see them (this is exactly how Trip "
                       f"re-entered into failing protection). NOT opening new positions until "
                       f"the ledger accepts them; the tick retries every run.")
            entries_blocked_by_backlog = True

        # ── 0. ORPHAN GUARD — the exchange must never hold an untracked position ──
        # [MONEY] A reduce-only order that survived an exit can close a LATER position.
        # While one is unresolved the bot opens nothing: the alternative is a stop from a
        # finished trade silently closing a fresh one.
        dangling = state.get_flag("dangling_orders") or []
        if dangling and not dry_run:
            pos_now = ex.open_position(client, sym)
            if pos_now.get("error"):
                log.error("cannot check the venue for dangling protection (%s) — refusing "
                          "to trade this tick", pos_now["error"])
                return 1
            if pos_now.get("active"):
                log.warning("dangling protection was recorded but a position is open — "
                            "leaving the resting orders alone")
            else:
                for oid in dangling:
                    ex.cancel(client, sym, str(oid))
                # VERIFY BY ID: the question is "is THIS order gone?", and a venue that
                # refuses to LIST trigger orders can still answer it. A read we cannot make
                # returns "not gone" (order_gone), so an outage keeps the block in place.
                mine = [str(x) for x in dangling]
                # CLEARANCE, not cancellation. `order_gone` keeps its conservative rule for
                # cancelling, but an id the venue definitively does not know must not pin this
                # flag for ever — reproduced live: a certainly-nonexistent id returned False
                # from order_gone purely because this key cannot list trigger orders.
                left = [i for i in mine if not ex.order_provably_finished(client, sym, i)]
                try:
                    still = {str(o.get("id") or "") for o in
                             ex.open_orders(client, sym, strict=True)}
                except Exception as exc:
                    log.info("could not list the book while checking dangling protection "
                             "(%s) — per-id verification stands", str(exc)[:80])
                    still = set()
                if left:
                    state.set_flag("dangling_orders", left)
                    log.error("refusing to trade: %d protective order(s) still unresolved "
                              "(%s)", len(left), ", ".join(left[:3]))
                    # A stuck condition is re-checked every tick: say it once, not every
                    # 15 minutes (the alert is repeated only after the cooldown).
                    alert_once(state, "dangling_unresolved", notify,
                               f"🛑 Mel — {len(left)} protective order(s) from a finished "
                               f"trade could not be cancelled; NOT opening new positions "
                               f"until the book is clear. Run `--sync` to inspect.")
                    return 1
                others = sorted(i for i in still if i and i not in mine)
                state.clear_flag("dangling_orders")
                notify("🧹 Mel — the previously dangling protective order(s) are confirmed "
                       "cancelled; trading resumes on the next tick.")
                if others:
                    log.warning("%d order(s) on %s are not this bot's: %s",
                                len(others), cfg["bot"]["symbol"], ", ".join(others[:3]))
                    notify(f"⚠️ Mel — {len(others)} other order(s) rest on "
                           f"{cfg['bot']['symbol']} that this bot did not place; left "
                           f"untouched. Run `--sync` to inspect.")

        # ── 0. ENTRY-INFLIGHT RECONCILIATION (a fill that landed but failed to persist) ──
        # The reservation is written BEFORE the market submit (engine.try_enter), so a ledger/state
        # write that fails AFTER the fill leaves this durable marker. Reconcile it here — BEFORE
        # the flat-state orphan guard — so a position THIS bot opened is recovered with its own
        # exact SL/TP, not adopted as an unknown orphan with config-floor stops. If it cannot be
        # reconciled, block the tick: the position is reserved (not an orphan), so the orphan
        # guard must not double-adopt it, and the next tick re-runs the reconcile.
        if state.get_flag("entry_inflight"):
            if dry_run:
                log.error("entry_inflight set in a dry run — refusing to trade this tick")
                return 1
            try:
                res = engine.reconcile_entry_inflight()
                log.info("entry-inflight reconcile: %s", res)
            except Exception as exc:
                log.exception("entry-inflight reconcile failed: %s", exc)
                res = {"status": "blocked"}
            if res.get("status") in ("blocked", "unprotectable"):
                log.error("an in-flight entry could not be reconciled — blocking this tick")
                return 1

        # ── 0. ORPHAN GUARD — the exchange must never hold an untracked position ──
        # [MONEY] Runs BEFORE anything else. With a flat state but a live position on the
        # exchange (lost state file, interrupted entry, manual order), falling through to
        # the entry search opens a SECOND position on top of the first. Reproduced live on
        # Jack: a missing state.json turned a 28-contract short into 52.
        if state.is_flat():
            try:
                live = ex.open_position(client, sym)
            except Exception as exc:
                log.error("orphan check failed (%s) — refusing to trade this tick", exc)
                return 1
            # open_position reports a failed fetch as active=False WITH an error, so the
            # except above never fires for an exchange outage. Without this check the guard
            # reads "no orphan" from an UNREADABLE exchange and falls through to the entry
            # search — opening a second position on top of one it simply could not see.
            if live.get("error"):
                log.error("orphan check could not read the exchange (%s) — "
                          "refusing to trade this tick", live["error"])
                return 1
            if live.get("active") and float(live.get("contracts") or 0.0) > 0:
                log.critical("ORPHAN: exchange holds %s %s but state is flat — adopting",
                             live.get("contracts"), live.get("side"))
                try:
                    res = engine.adopt_orphan(live)
                    status = res.get("status")
                    if status == "adopted":
                        return 0
                    if status == "orphan_dry_run":
                        # Reported by the engine with an alert; a dry run is not a failure.
                        log.warning("orphan present and this tick is a DRY RUN — nothing "
                                    "adopted, nothing placed")
                        return 0
                    return 1
                except Exception as exc:
                    log.exception("orphan adoption failed: %s", exc)
                    return 1
            # Flat AND nothing of ours tracked: any resting order left on the symbol is a
            # dangling protective order from a finished trade — the one class of order
            # that can close a position Mel opens LATER.
            try:
                triggers_listable = ex._stops_view_available(client, sym)
            except Exception:
                triggers_listable = False
            try:
                resting = ex.open_orders(client, sym, strict=True)
            except Exception as exc:
                # An unreadable book is not an empty book. Trading on that assumption is
                # how a leftover reduce-only stop meets a brand-new position.
                log.error("cannot read the order book (%s) — refusing to open a position "
                          "this tick", exc)
                return 1
            if not resting and not triggers_listable and not engine.protection_all_clear():
                # [MONEY] AN EMPTY LISTING IS NOT A CLEAR BOOK. This venue cannot show stop
                # orders through the normal listing, so "nothing is resting" is only
                # believable when the trigger view works, or when nothing this bot placed is
                # still unaccounted for. This is the case the whole guard exists for, and it
                # used to sit inside `if resting:` — where an empty listing could never reach
                # it (Terra round-4 finding 1).
                log.error("empty listing cannot be shown to include stop orders and a "
                          "previously placed order id is outstanding — refusing entry")
                alert_once(state, "book_not_provable", notify,
                           f"🛑 Mel — the order listing is empty but this venue cannot show "
                           f"stop orders through it, and an order this bot placed is not "
                           f"provably finished. It could still be resting and could close a "
                           f"position opened now. NOT opening a position; run `--sync`.")
                return 1
            if resting:
                # OWNERSHIP: only PROTECTIVE (reduce-only) orders are cancelled. A
                # non-reduce-only order is not this bot's protection — it may be a manual
                # order or another tool's entry — so it is reported and left alone rather
                # than cancelled out from under whoever placed it.
                protective = [o for o in resting if ex.is_reduce_only(o) is not False]
                foreign = [o for o in resting if ex.is_reduce_only(o) is False]
                if foreign:
                    log.warning("%d resting order(s) are not reduce-only — leaving them "
                                "alone (not this bot's protection)", len(foreign))
                    if not dry_run:
                        notify(f"⚠️ Mel — {len(foreign)} resting order(s) on "
                               f"{cfg['bot']['symbol']} are NOT reduce-only; left untouched. "
                               f"Run `--sync` to inspect.")
                if not protective and not triggers_listable:
                    # The listing is empty but it CANNOT be shown to include stop orders, so
                    # an invisible leftover stop is still conceivable. That is not a reason
                    # to wedge the bot (there is a runner: the ids persisted in
                    # `dangling_orders` are verified individually, and every protection Mel
                    # places is verified by id at exit) — but it IS a reason to keep saying
                    # so, once, until the venue proves otherwise.
                    log.warning("order listing cannot be shown to include stop orders on this "
                                "venue/key (trigger view refused) — an empty listing is not "
                                "proof the book is clear; protection is verified per id")
                    if state.get_flag("stop_listing_covers_triggers") is None:
                        notify("🔎 Mel — this venue refuses its trigger-order listing, so an "
                               "empty order list cannot be shown to mean 'no stop resting'. "
                               "Mel verifies protection per id instead; the answer is "
                               "recorded automatically the first time one of our stops rests.")
                if protective:
                    ids = ", ".join(str(o.get("id") or "?")[:10] for o in protective)
                    log.warning("flat but %d protective order(s) still resting (%s)",
                                len(protective), ids)
                    # [MONEY] Mel NEVER CANCELS WHAT IT CANNOT PROVE IT PLACED — not even
                    # here, and not behind an opt-in flag. The flag that used to authorise a
                    # whole-symbol sweep was removed: with no id history a "leftover" order is
                    # indistinguishable from a manual one, and the sweep also stored whatever
                    # remained as this bot's to cancel later. Clearing stays a deliberate,
                    # armed, id-scoped action (`--sync --cancel-dangling`).
                    if not dry_run:
                        alert_once(state, "flat_with_protection", notify,
                                   f"🛑 Mel — flat but {len(protective)} protective order(s) "
                                   f"still rest on {cfg['bot']['symbol']} ({ids}). This bot "
                                   f"did not place them in this state, so it will NOT cancel "
                                   f"them; new entries are blocked meanwhile. Inspect with "
                                   f"`--sync`, or arm the bot and run `--sync "
                                   f"--cancel-dangling` to clear them deliberately.")
                        log.error("refusing to open a position while unexplained protective "
                                  "order(s) rest on the symbol")
                        return 1

        # ── 1. manage an open position first, always ──
        if not state.is_flat():
            try:
                res = engine.manage(None)
                if res.get("status") == "holding":
                    log.info("holding %.3f (%sh)", res.get("remaining", 0), res.get("held_h"))
                return 0
            except Exception as exc:
                log.exception("manage failed: %s", exc)
                return 1

        # ── 1a. MANAGE-ONLY TICK — management is done, entries are NOT evaluated ──
        # [MONEY] The hold cap, the stop re-verification and the orphans above are all
        # evaluated ONCE PER TICK, so on an hourly cron a cap that lands seconds before a
        # tick waits a full extra hour. Mel's answer is a second, 15-minute cron that runs
        # this same management path and stops here — never scoring the market, never
        # opening a position.
        #
        # Deliberately placed AFTER the manage block and BEFORE the entry gates (1b/1c/2/3):
        # everything above is protection and must run on every tick; everything below exists
        # only to decide an entry, and a manage-only tick must not even read candles. Skipping
        # the 1h decision path here also keeps the strategy honest — the score is built from
        # the FORMING 1h candle, so a 4x-per-hour decision tick would sample the same bar at
        # four different ages and change entry timing (unbacktested). Entries stay on the
        # hourly tick.
        if manage_only:
            log.info("manage-only tick: flat, nothing to manage — entries not evaluated (by design)")
            return 0

        # ── 1b. the ledger backlog blocks ENTRIES, never management ──
        # Placed after `manage()` on purpose: a queued close that could not be CONFIRMED
        # (`close_unverified`) can coexist with a live position, and the orphan guard and
        # `manage()` above are exactly what keep that position protected. Blocking the tick
        # earlier — as it used to — skipped both and left the position naked.
        if entries_blocked_by_backlog:
            log.error("entries blocked: %d close(s) still unwritten", backlog_left)
            return 1

        # ── 1c. an unresolved ambiguous stop blocks ENTRIES, never management ──
        # Same placement rule as 1b: the orphan guard and `manage()` above are what keep a
        # possibly-live position protected, so the stop's entries-only block fires only after
        # they have run.
        if entries_blocked_by_stop:
            log.error("entries blocked: an ambiguous protective stop cannot be located or "
                      "disproved")
            return 1

        # ── 2. halted? ──
        if state.halted:
            log.info("halted: %s", state.halt_reason)
            try:
                state.set_gate(engine.rail_state())
            except Exception:
                pass
            return 0

        # ── 2b. publish the guard-rail gate for the dashboard card (reporting only) ──
        # Derived from the SAME guard functions that block the entry (Engine.rail_state), so
        # the card can never disagree with the rail that actually fired. A reporting failure
        # must not stop the tick, so it is swallowed here.
        try:
            state.set_gate(engine.rail_state())
        except Exception:
            pass

        # ── 3. look for an entry ──
        base = ex.fetch_ohlcv(client, sym, cfg["strategy"]["base_timeframe"],
                              int(cfg["strategy"]["base_window"]))
        htf = ex.fetch_ohlcv(client, sym, cfg["strategy"]["htf_timeframe"],
                             int(cfg["strategy"]["htf_window"]))
        if len(base) < 60 or len(htf) < 30:
            log.warning("not enough candles (base=%d htf=%d) — waiting", len(base), len(htf))
            return 0

        st = build_market_state(base, htf, cfg["indicators"])
        dec = decide(st, cfg)
        if not dec.is_entry:
            log.info("WAIT — %s", dec.reason)
            return 0

        stats = db.stats()
        ctx = dict(stats)
        ctx["sl_failure_streak"] = effective_protection_streak(stats, state)
        # [MONEY] The consecutive-loss breaker is day-bounded (guards.g_consecutive_losses): an
        # operator acknowledgement can clear it early, and the real stats + the last loss time let
        # the guard tell a streak from TODAY (block) from one from days ago (already stale).
        ctx["consecutive_losses"] = effective_loss_streak(stats, state)
        ctx["last_loss_at"] = stats.get("last_loss_at")
        if stats.get("last_exit_time"):
            try:
                from datetime import datetime, timezone
                t0 = datetime.fromisoformat(str(stats["last_exit_time"]))
                if t0.tzinfo is None:
                    t0 = t0.replace(tzinfo=timezone.utc)
                mins = (datetime.now(timezone.utc) - t0).total_seconds() / 60.0
                ctx["candles_since_close"] = int(mins / 60)
            except (TypeError, ValueError):
                ctx["candles_since_close"] = None

        override_until = read_override_until(cfg["bot"]["id"], time.time())
        if override_until:
            log.info("guard-rail override active (until %s) — entering past the rails",
                     override_until)
        g = run_guards(dec, st, ctx, cfg, overridden=bool(override_until))
        if not g.passed:
            log.info("blocked — %s", g.reason)
            return 0
        for w in g.warnings:
            log.info("warning — %s", w)

        # Size off the REAL free collateral even in dry-run: a dry-run that sizes itself
        # from an invented equity figure cannot tell you whether the real order would be
        # too small to place, which is exactly what a dry run is for.
        try:
            funds = ex.free_quote(client, cfg["bot"]["quote"])
        except Exception as exc:
            funds = float(cfg["sizing"].get("dry_run_equity", 1000.0)) if dry_run else 0.0
            log.warning("balance read failed (%s) — sizing against %.2f", exc, funds)
        if funds <= 0:
            log.info("no free collateral — not trading")
            return 0

        spec = engine.spec
        sizing = C.size_position(funds, dec.entry, cfg,
                                 spec["contract_size"], spec["amount_step"],
                                 confidence=dec.confidence,
                                 extra_scale=g.position_scale,
                                 min_contracts=(spec.get("min_amount")
                                                or cfg["sizing"].get("min_contracts", 0.001)))
        if sizing.contracts <= 0:
            log.info("size too small (funds=%.2f, scale=%.2f, step=%s, min=%s) — not trading",
                     funds, sizing.scale, spec["amount_step"], sizing.min_contracts)
            return 0

        log.info("%s score=%d margin=$%.2f notional=$%.2f contracts=%s (=%s XRP) scale=%.2f",
                 dec.action, dec.confidence, sizing.margin, sizing.notional,
                 f"{sizing.contracts:g}",
                 f"{sizing.contracts * spec['contract_size']:g}", sizing.scale)
        out = engine.try_enter(dec, sizing)
        if out.get("status") == "error":
            log.error("entry failed: %s", out.get("message"))
            return 1
        return 0
    finally:
        try:
            db.close()
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
            lock.close()


if __name__ == "__main__":
    sys.exit(main())
