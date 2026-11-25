#!/usr/bin/env python3
"""Operator tool: record a MANUAL (or venue-side) close that the bot could not price.

WHY THIS EXISTS
`Engine._finish` prices an exit from fills of the orders WE recorded — SL, TP, TP1/TP2. When the
position was closed by something this bot did NOT place (an operator's manual market close, a
venue-side liquidation, a human in the Kraken UI), those ids have no fills, so `settled_close`
returns "no exit fill price and no fallback — caller must not record" and `_finish` correctly
REFUSES to write a row rather than invent a price. State stays open and every tick retries.

Nothing is wrong with the bot there — fabricating a price is worse than an unpriced row. But the
venue's own fill history DOES hold the evidence; it is simply keyed to an order id the bot never
learned. This tool finds it, insists that it COVERS the position, and then hands it to the engine's
own `_finish` so the arithmetic, the provenance flags and the row are all the reviewed ones.

    python3 tools/record_manual_close.py           # show the evidence it found (writes nothing)
    python3 tools/record_manual_close.py --yes     # record the row and clear the stale state

It REFUSES when: the venue still holds the position (never record a close for an open position), the
state is already flat, the entry fill cannot be read, the discovered fills do not cover the full
position (partial evidence is not a recordable exit), or the close spans more than one order id
(that needs a human to look, not a guess).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config as C           # noqa: E402
import db as DB              # noqa: E402
import exchange as ex        # noqa: E402
import run_bot as RB         # noqa: E402
import state as ST           # noqa: E402
from engine import Engine    # noqa: E402


def _backup(cfg: dict) -> Path:
    dest = Path("/opt/data/backups") / f"mel_record_manual_close_{time.strftime('%Y%m%d_%H%M%S')}"
    dest.mkdir(parents=True, exist_ok=True)
    for rel in (cfg["ops"]["state_file"], cfg["ops"]["db_path"]):
        src = ROOT / rel
        if src.exists():
            shutil.copy2(src, dest / Path(rel).name)
    return dest


def _retire_our_protection(client, sym: str, pos: dict) -> tuple[bool, str]:
    """Make sure no order of OURS is still resting before a close is recorded.

    `Engine._finish` retires protection in the tick path (`_reconcile_closed` calls
    `_retire_protection` first); calling `_finish` directly bypasses that, which matters when the
    position died somewhere we did not act — a venue liquidation or a human closing in the Kraken
    UI leaves the reduce-only SL/TP resting, and a survivor can close the NEXT position.

    Ownership is an id we recorded. Foreign ids are reported, never cancelled.
    """
    ours = {str(pos.get(k)) for k in ("sl_order_id", "tp_order_id") if pos.get(k)}
    ours |= {str(i) for i in (pos.get("placed_ids") or []) if i}
    for oid in sorted(ours):
        if not ex.order_provably_finished(client, sym, oid):
            ex.cancel(client, sym, oid)
    try:
        resting = ex.open_orders(client, sym, strict=True)
    except Exception as exc:
        # This venue refuses the trigger listing for this key, so the book cannot always be read
        # completely. Check our own ids individually instead of assuming the book is clear.
        left = [o for o in sorted(ours) if not ex.order_provably_finished(client, sym, o)]
        if left:
            return False, f"our order(s) {left} cannot be proven gone ({exc})"
        return True, "per-id verification only (this venue refuses the trigger listing)"
    mine, theirs = _ours(resting, ours)
    if mine:
        return False, f"our order(s) {mine} are still resting"
    # A book read is NOT proof on its own: this venue refuses the trigger listing for some keys,
    # so one of our stops can be absent from the listing simply by being invisible. Verify every
    # id DEFINITIVELY, exactly as the engine's own _retire_protection does — otherwise a survivor
    # that the listing hides gets reported as "book clear" and can close the NEXT position
    # (Paul, round 2).
    not_proven = [o for o in sorted(ours) if not ex.order_provably_finished(client, sym, o)]
    if not_proven:
        return False, (f"our order(s) {not_proven} cannot be proven finished — a book read is not "
                       f"proof where the listing may be incomplete")
    if theirs:
        return False, (f"FOREIGN order(s) {theirs} rest on {sym} — not ours, not touched; "
                       f"verify by hand before recording a close or opening anything")
    return True, "book clear of our orders (every id proven finished)"


def _ours(resting: list, ours: set) -> tuple[list, list]:
    """Split a resting book into (ours, foreign) by recorded id."""
    mine, theirs = [], []
    for o in resting:
        oid = str((o or {}).get("id") or "")
        if not oid:
            continue
        (mine if oid in ours else theirs).append(oid)
    return mine, theirs


def _validate_evidence(fills: list, contracts: float) -> tuple[bool, str, str]:
    """(ok, order_id, reason) — the coverage rules, venue-free so they can be TESTED.

    Extracted on Paul's round-2 recommendation: the over-coverage guard ("attributing an oversized
    sell to this row would fabricate its exit") is the same wrong-number class as the rest of this
    round, and it lived only inside `main()` where no test could reach it.
    """
    if not fills:
        return False, "", "no closing fills found on the venue since this position opened"
    by_order: dict[str, float] = {}
    total = 0.0
    for f in fills:
        oid = str(f.get("order") or "")
        amt = float(f.get("amount") or 0.0)
        by_order[oid] = by_order.get(oid, 0.0) + amt
        total += amt
    if total < contracts * 0.999:
        return False, "", (f"the fills cover only {total:g} of {contracts:g} — partial evidence is "
                           f"not a recordable exit")
    if total > contracts * 1.001:
        return False, "", (f"the fills cover {total:g} against a position of {contracts:g} — MORE "
                           f"than this trade, so attributing it here would fabricate the exit")
    ids = [k for k, v in by_order.items() if v > 0 and k]
    if len(ids) != 1:
        return False, "", f"the close spans {len(ids)} order ids {ids} — that needs a human"
    return True, ids[0], f"one closing order {ids[0]} covering {total:g}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--yes", action="store_true", help="required: actually record the row")
    args = ap.parse_args()

    cfg = C.load_config(str(ROOT / "config.yaml"))
    keys = RB.load_keys()
    sym = cfg["bot"]["ccxt_symbol"]
    state = ST.State(ROOT / cfg["ops"]["state_file"])
    pos = state.get_open()
    if not pos:
        print("state is already flat — nothing to record.")
        return 0

    client = ex.new_client(cfg, keys, dry_run=False)
    live = ex.open_position(client, sym)
    if live.get("error"):
        print(f"ABORT: cannot read the venue ({live['error']}) — 'cannot read' is not 'flat'.")
        return 1
    held = float(live.get("contracts") or 0.0) if live.get("active") else 0.0
    print(f"=== state: #{pos.get('trade_id')} {pos.get('direction')} {pos.get('contracts')} @ "
          f"{pos.get('entry_price')} (opened {pos.get('entry_time')})")
    print(f"=== venue: {'FLAT' if held <= 0 else f'{held} STILL OPEN'}")
    if held > 0:
        print("ABORT: the venue still holds the position — refusing to record a close for it.")
        return 1

    contracts = float(pos.get("contracts") or 0.0)
    opened_ms = int(pos.get("opened_ms") or 0)
    if opened_ms <= 0:
        print("ABORT: this position carries no opened_ms, so a fill search has no window to bound "
              "it — with since=0 the whole account history is in scope and a stale sell could be "
              "attributed to this trade.")
        return 1
    closing_side = "sell" if str(pos.get("direction")).lower() == "long" else "buy"

    print(f"\n=== looking for {closing_side} fills since the position opened (need {contracts} covered) ===")
    try:
        fills = [f for f in (client.fetch_my_trades(sym, limit=200) or [])
                 if str(f.get("side") or "").lower() == closing_side
                 and int(f.get("timestamp") or 0) >= opened_ms]
    except Exception as exc:
        print(f"ABORT: fill history unreadable ({exc})")
        return 1
    if not fills:
        print("ABORT: no closing fills found on the venue since this position opened — the venue "
              "cannot price this close, so nothing is recorded (that is the honest outcome).")
        return 1

    by_order_total = 0.0
    for f in fills:
        amt = float(f.get("amount") or 0.0)
        by_order_total += amt
        fee = (f.get("fee") or {}).get("cost") if isinstance(f.get("fee"), dict) else f.get("fee")
        print(f"    {f.get('datetime')} {f.get('side')} {amt} @ {f.get('price')} "
              f"fee {fee} order {f.get('order')}")
    print(f"    covered: {by_order_total:g} of {contracts:g}")
    ok, oid, why = _validate_evidence(fills, contracts)
    if not ok:
        print(f"ABORT: {why}.")
        return 1
    print(f"\n=== evidence: {why}")

    dest = _backup(cfg)
    print(f"=== backup: {dest}")
    if not args.yes:
        print("\n(dry: nothing recorded — re-run with --yes to record the row)")
        return 0

    # Retire OUR protection before recording. `_finish` normally runs after `_reconcile_closed`'s
    # `_retire_protection`; calling it directly skips that, and a reduce-only survivor from a close
    # this bot did not perform could close the NEXT position.
    clean, why = _retire_our_protection(client, sym, pos)
    print(f"=== protection: {'clear' if clean else 'NOT clear'} — {why}")
    if not clean:
        print("ABORT: refusing to record a close while protective orders of ours (or unknown orders "
              "on this symbol) may still be resting.")
        return 1

    # Hand the discovered order id to the ENGINE's own settle+record path: same arithmetic,
    # same provenance flags, same row, same state clearing. We only supply the evidence.
    notify = RB.make_notifier(cfg, enabled=True)
    db_path = ROOT / cfg["ops"]["db_path"]
    engine = Engine(cfg, client, state, DB.TradeDB(str(db_path)), notify, dry_run=False)
    res = engine._finish(pos, 0.0, "manual_close", close_order={"id": oid})
    print(f"\n=== _finish -> {res}")

    # Verify by re-reading: state must be flat and the row must carry a real price.
    st2 = json.loads((ROOT / cfg["ops"]["state_file"]).read_text())
    print(f"=== state now: {'FLAT' if not st2.get('open') else 'STILL OPEN'}")
    import sqlite3
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM trades WHERE id = ?", (pos.get("trade_id"),)).fetchone()
    if row:
        keys = ("trade_id", "direction", "entry_price", "exit_price", "pnl", "pnl_net", "fee",
                "fees", "settlement_source", "price_verified", "reason", "exit_reason", "status")
        print("    row:", {k: row[k] for k in keys if k in row.keys()})
    else:
        print("    (no row found for that trade_id — check the ledger by hand)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
