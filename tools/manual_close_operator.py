#!/usr/bin/env python3
"""Operator script: MANUALLY CLOSE Mel's open position (Tee's instruction).

Uses ONLY Mel's own reviewed helpers — no hand-rolled orders:
  * `exchange.open_position`  - read what the venue actually holds
  * `exchange.flatten_all`    - reduce-only market close, SIZED OFF THE LIVE POSITION, verified by re-read
  * `exchange.cancel`         - by-id cancel that reads the venue's answer instead of assuming
  * `exchange.order_provably_finished` - per-id proof before claiming an order of ours is gone
    (our orders are cancelled by recorded id only; anything else resting is REPORTED, never
    touched: it may be a human's order on the same account)

SEQUENCE, and why this order:
  1. READ the venue and REQUIRE the live position to match `state.json` before touching anything.
     A mismatch means state and venue disagree, which is a report-and-stop, not a fix-and-hope.
  2. FLATTEN FIRST, cancel the resting exits SECOND. Keeping protection in place until the position
     is genuinely closed is the safe direction: if the stop fires during the flatten, the position
     closes and the reduce-only close order is then rejected for zero size (reduce-only cannot flip
     a position), and `flatten_all` re-reads the position at the top of every attempt, so the race is
     handled rather than merely survived.
  3. CANCEL the recorded exit ids anyway. A leftover reduce-only order can close a FUTURE position
     (the duplicate-stop class), so leaving them is not a harmless extra.
  4. VERIFY BY RE-READING. `accepted` is not `filled`; the script refuses to claim flat from an order
     response. If the venue cannot be read, it reports UNKNOWN and exits non-zero.

It does NOT touch the bot's ledger. Mel's own `sync.py` treats "state says open, exchange says
flat" as STALE STATE to be reconciled as closed, so the next tick writes the honest close row from
venue fill evidence. That is the house path, and this script deliberately leaves it alone.

    python3 tools/manual_close_operator.py --yes
"""
from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import exchange as ex        # noqa: E402
import config as C           # noqa: E402
import run_bot as RB         # noqa: E402  load_keys() lives here: env first, then Mel's own .env


def _backup(cfg: dict) -> Path:
    ts = time.strftime("%Y%m%d_%H%M%S")
    dest = Path("/opt/data/backups") / f"mel_manual_close_{ts}"
    dest.mkdir(parents=True, exist_ok=True)
    for rel in (cfg["ops"]["state_file"], cfg["ops"]["db_path"]):
        src = ROOT / rel
        if src.exists():
            shutil.copy2(src, dest / Path(rel).name)
    return dest


def _ours(resting: list, ours: set) -> tuple[list, list]:
    """Split a resting book into (ours, foreign) by id.

    Ownership is an id we recorded, never a shape match (order-mutation-safety rule 5): an order
    that merely looks like ours may belong to a human on the same account, and cancelling it is
    not recoverable. Foreign ids are REPORTED, never touched — `cancel_all_for_symbol` hid this
    by sweeping everything.
    """
    mine, theirs = [], []
    for o in resting:
        oid = str((o or {}).get("id") or "")
        if not oid:
            continue
        (mine if oid in ours else theirs).append(oid)
    return mine, theirs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--yes", action="store_true", help="required: actually submit the close")
    args = ap.parse_args()

    cfg = C.load_config(str(ROOT / "config.yaml"))
    keys = RB.load_keys()
    if not keys.get("api_key"):
        print(f"ABORT: no API key (env or {ROOT}/.env)")
        return 1
    sym = cfg["bot"]["ccxt_symbol"]
    state_path = ROOT / cfg["ops"]["state_file"]

    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    pos_state = dict(state.get("open") or {})
    exit_ids = list((state.get("flags") or {}).get("placed_ids") or [])
    print(f"=== state.json: {'POSITION OPEN' if pos_state else 'flat'} | recorded exits: {exit_ids}")
    if pos_state:
        print(f"    {pos_state.get('direction')} {pos_state.get('contracts')} @ {pos_state.get('entry_price')} "
              f"(trade #{pos_state.get('trade_id')}, SL {pos_state.get('sl_price')}, TP {pos_state.get('tp_price')})")

    dest = _backup(cfg)
    print(f"=== backup written: {dest}")

    client = ex.new_client(cfg, keys, dry_run=False)
    print(f"=== dry-run client? {ex.is_dry_run(client)}  (must be False)")

    # 1. READ the venue and REQUIRE agreement with our own notes.
    v = ex.open_position(client, sym)
    if v.get("error"):
        print(f"ABORT: cannot read the venue: {v['error']}")
        return 1
    vc = float(v.get("contracts") or 0.0) if v.get("active") else 0.0
    print(f"=== venue position: {'active' if v.get('active') else 'flat'} {vc} {v.get('side')}")
    if vc <= 0:
        print("== venue is already flat — nothing to close. (Ledger may still need its stale state reconciled.)")
        return 0
    if pos_state and abs(vc - float(pos_state.get("contracts") or 0.0)) > 1e-9:
        print(f"ABORT: venue holds {vc} but state.json records {pos_state.get('contracts')}. "
              f"Reconcile by hand before closing.")
        return 1
    if not args.yes:
        print("\n(dry: nothing submitted — re-run with --yes to close)")
        return 0

    # 2. FLATTEN FIRST (protection stays in place until the position is actually gone).
    print("\n=== flattening (reduce-only market, sized off the LIVE position) ===")
    flat = ex.flatten_all(client, sym, attempts=3)
    print(f"    {flat}")
    if flat.get("error"):
        print("ABORT: flatten UNVERIFIED — the venue could not be read. Do not assume it is closed.")
        return 1
    if not flat.get("flat"):
        print(f"ABORT: flatten FAILED — {flat.get('residual')} still open. Exits left in place on purpose.")
        return 1

    # 3. CANCEL the recorded exits — ours ONLY, and report anything else.
    print("\n=== cancelling recorded exits (ours only) ===")
    ours = set(str(i) for i in exit_ids if i)
    if pos_state:
        ours |= {str(pos_state.get(k)) for k in ("sl_order_id", "tp_order_id") if pos_state.get(k)}
    for oid in exit_ids:
        if ex.order_provably_finished(client, sym, str(oid)):
            print(f"    {oid}: already finished (provable by id)")
            continue
        ok = ex.cancel(client, sym, str(oid))
        print(f"    {oid}: cancel -> {'gone' if ok else 'UNCONFIRMED (may still be alive)'}")
    try:
        resting = ex.open_orders(client, sym, strict=True)
    except Exception as exc:
        # This venue refuses the trigger listing for this key, so a complete book read may be
        # impossible. Fall back to per-id proof rather than assuming the book is clear.
        print(f"    book read not possible ({exc}) — relying on per-id verification above")
        resting = []
    mine, theirs = _ours(resting, ours)
    for oid in mine:
        print(f"    {oid}: ours, still resting -> cancel -> "
              f"{'gone' if ex.cancel(client, sym, oid) else 'UNCONFIRMED'}")
    if theirs:
        # [MONEY] NOT OURS. Never cancel: it may be a human's order on the same account. Report it
        # so an operator can look, and refuse to claim the book is clean.
        print(f"    FOREIGN orders still resting, NOT touched: {theirs}")
        print("    -> verify by hand before opening anything on this symbol")

    # 4. VERIFY BY RE-READING.
    print("\n=== verification (re-read, not assumed) ===")
    after = ex.open_position(client, sym)
    if after.get("error"):
        print(f"    position read FAILED: {after['error']} -> UNKNOWN, verify manually")
        return 1
    left = float(after.get("contracts") or 0.0) if after.get("active") else 0.0
    print(f"    venue position now: {'FLAT' if left <= 0 else f'{left} STILL OPEN'}")
    try:
        resting = ex.open_orders(client, sym)
        print(f"    resting orders: {[str(o.get('id')) for o in resting] or 'none'}")
    except Exception as exc:
        print(f"    resting-order read failed ({exc}) — note this venue refuses the trigger listing "
              f"for this key, so verify stops by id")

    # 5. Realised result, from the venue's own fills.
    print("\n=== fills for this close ===")
    total_fee = 0.0
    try:
        for f in client.fetch_my_trades(sym, limit=10) or []:
            px, amt = f.get("price"), f.get("amount")
            fee = (f.get("fee") or {}).get("cost") if isinstance(f.get("fee"), dict) else f.get("fee")
            total_fee += float(fee or 0.0)
            print(f"    {f.get('datetime')} {f.get('side')} {amt} @ {px}  fee {fee}")
    except Exception as exc:
        print(f"    fill read failed: {exc}")

    entry = float(pos_state.get("entry_price") or 0.0)
    if entry and pos_state.get("direction") == "long":
        print(f"\n    entry {entry} -> exit (see fills above), {pos_state.get('contracts')} ETH")
        print(f"    fees on the recorded fills: ~{total_fee:.4f} USDT")
    print("\n=== ledger NOT touched (by design): Mel's next tick reconciles the stale state as closed ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
