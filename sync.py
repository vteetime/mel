"""Exchange reconciliation — ask Kraken what it actually holds, and believe the answer.

`python run_bot.py --sync` prints this. The same snapshot is consulted at the top of
every tick (as the orphan guard), because the exchange is authoritative and our
state.json / trades.db / dashboard are all just our own notes.

What it answers, in order of importance:

  1. **Is there a position?** If yes and state is flat, that is an ORPHAN — the tick
     adopts it rather than opening a second one on top. If state says open and the
     exchange says flat, the state is STALE and must be reconciled as closed.
  2. **Is the position protected?** A live position with no reduce-only stop covering it
     is the dangerous case: it gets a stop, or it gets flattened.
  3. **Is anything left resting that should not be?** A leftover stop after a flat exit
     can close a FUTURE position. This is the duplicate-stop class of bug (see Ken's
     cleanup_ken_duplicate_stops.py) and it is reported explicitly.
  4. **What is actually left to trade with?** Free collateral, not `total` — margin
     already committed is not available.

Read-only by default. `cancel_dangling` is the single destructive option and is only
reachable from the explicit CLI flag.
"""

from __future__ import annotations

import logging
import time

import exchange as ex

log = logging.getLogger("mel.sync")


def snapshot(client, symbol: str, fills_limit: int = 20) -> dict:
    """One read-only pass over the venue. Never raises; unreadable parts are flagged."""
    snap: dict = {"ts": int(time.time()), "symbol": symbol}

    try:
        snap["balance"] = ex.balance_report(client)
    except Exception as exc:
        snap["balance"] = {"free": 0.0, "total": 0.0, "error": str(exc)}

    snap["position"] = ex.open_position(client, symbol)

    try:
        snap["orders"] = ex.open_orders(client, symbol)
    except Exception as exc:
        snap["orders"] = []
        snap["orders_error"] = str(exc)

    try:
        snap["recent_fills"] = list(client.fetch_my_trades(symbol, limit=fills_limit) or [])
    except Exception as exc:
        snap["recent_fills"] = []
        snap["fills_error"] = str(exc)

    try:
        spec = ex.market_spec(client, symbol)
    except Exception as exc:
        spec = {"error": str(exc)}
    snap["spec"] = spec

    snap["stops"] = [o for o in snap["orders"]
                     if str(o.get("type") or "").lower() in ("stop", "stopmarket",
                                                             "stop-loss", "stop_loss")
                     or ex.order_trigger(o) > 0]
    snap["limits"] = [o for o in snap["orders"] if o not in snap["stops"]]
    # CAN THIS VENUE/KEY EVEN ENUMERATE TRIGGER ORDERS? Kraken refuses the trigger-specific
    # query for the key in use. The listing read above comes from the account's own
    # open-orders endpoint (which does answer), but whether IT carries stop orders is only
    # established by evidence — see Engine._learn_stop_visibility, which records the answer
    # the first time Mel has a stop of its own resting.
    try:
        snap["triggers_listable"] = ex._stops_view_available(client, symbol)
    except Exception:
        snap["triggers_listable"] = False
    return snap


def reconcile(snap: dict, state_open: dict | None) -> dict:
    """Compare what the venue holds against what our state believes. Returns a verdict."""
    pos = snap.get("position") or {}
    held = float(pos.get("contracts") or 0.0) if pos.get("active") else 0.0
    readable = not pos.get("error")

    stops = snap.get("stops") or []
    protecting = [o for o in stops if ex.is_reduce_only(o) is not False]

    verdict: dict = {
        "venue_position": held,
        "venue_side": pos.get("side") or "",
        "venue_entry": float(pos.get("entry_price") or 0.0),
        "state_position": float((state_open or {}).get("contracts") or 0.0),
        "resting_orders": len(snap.get("orders") or []),
        "resting_stops": len(stops),
        "readable": readable,
        "findings": [],
        "action": "none",
    }
    f = verdict["findings"]

    if not readable:
        f.append(f"exchange position UNREADABLE ({pos.get('error')}) — refusing to act "
                 f"on this read")
        verdict["action"] = "refuse"
        return verdict

    # Coverage is the sum of what the surviving stops can STILL FILL (`order_amount` answers
    # the OUTSTANDING size), compared against what is still held — like for like, so a
    # partially filled stop shows up as the shortfall it is.
    covered = sum(ex.order_amount(o) for o in protecting)
    verdict["stop_coverage"] = covered

    if held > 0 and not state_open:
        f.append(f"ORPHAN: venue holds {held:g} {verdict['venue_side']} with no bot state")
        verdict["action"] = "adopt"
    elif held <= 0 and state_open:
        f.append(f"STALE STATE: state believes {verdict['state_position']:g} is open, "
                 f"venue holds nothing")
        verdict["action"] = "reconcile_closed"
    elif held > 0 and state_open:
        if abs(held - verdict["state_position"]) > 1e-9:
            f.append(f"SIZE DRIFT: venue {held:g} vs state "
                     f"{verdict['state_position']:g} — the venue is authoritative")
        if not protecting:
            f.append("UNPROTECTED: live position with no reduce-only stop resting")
            verdict["action"] = "protect"
        elif covered + 1e-9 < held:
            f.append(f"UNDER-PROTECTED: stops cover {covered:g} of {held:g}")
            verdict["action"] = "protect"
    elif held <= 0 and not state_open:
        dangling = [o for o in (snap.get("orders") or [])]
        if dangling:
            kinds = ", ".join(sorted({str(o.get('type') or 'order') for o in dangling}))
            f.append(f"DANGLING ORDERS: flat, but {len(dangling)} order(s) still resting "
                     f"({kinds}) — a leftover reduce-only stop can close a LATER position")
            verdict["action"] = "cancel_dangling"

    unverifiable = (not verdict["resting_orders"]) and not snap.get("triggers_listable")
    if unverifiable:
        # An empty listing from a source that cannot be shown to include stop orders is not
        # proof of an empty book. Say so rather than printing "clean".
        f.append("NOTE: this venue refuses its trigger-order listing for the key in use, so "
                 "the order list above could be missing stop orders. Stop coverage is "
                 "UNVERIFIED — protection is verified per id instead (see --sync output "
                 "and the engine's stop-visibility flag).")
        verdict["stop_coverage_verified"] = False
    else:
        verdict["stop_coverage_verified"] = True
    if not f:
        f.append("clean: venue and state agree")
    elif unverifiable and verdict["action"] == "none":
        verdict["action"] = "none_unverified"
    return verdict


def format_report(snap: dict, verdict: dict, cfg: dict) -> str:
    """Operator-facing text. Numbers only — no estimates dressed as facts."""
    b = snap.get("balance") or {}
    spec = snap.get("spec") or {}
    pos = snap.get("position") or {}
    lines = []
    sym = cfg["bot"]["symbol"]
    lines.append(f"KRAKEN SYNC — {sym} ({snap.get('symbol')})")
    lines.append("=" * 62)
    lines.append(f"collateral      : {b.get('free', 0.0):,.2f} free / "
                 f"{b.get('total', 0.0):,.2f} total {b.get('currency', '')}")
    if b.get("error"):
        lines.append(f"                  (balance read FAILED: {b['error']})")
    if pos.get("active"):
        lines.append(f"position        : {pos.get('contracts'):g} "
                     f"{str(pos.get('side') or '').upper()} @ "
                     f"{float(pos.get('entry_price') or 0):,.2f}  "
                     f"(unrealized {float(pos.get('unrealized') or 0):+,.2f})")
    elif pos.get("error"):
        lines.append(f"position        : UNREADABLE ({pos['error']})")
    else:
        lines.append("position        : flat")
    lines.append(f"resting orders  : {verdict.get('resting_orders', 0)} "
                 f"({verdict.get('resting_stops', 0)} stop-type)")
    for o in snap.get("orders") or []:
        trig = ex.order_trigger(o)
        lines.append(f"   - {str(o.get('type') or '?'):<8} {str(o.get('side') or ''):<4} "
                     f"amt {ex.order_amount(o):g}"
                     + (f" trigger {trig:,.2f}" if trig else
                        f" price {float(o.get('price') or 0):,.2f}")
                     + f" reduceOnly={ex.is_reduce_only(o)} id={str(o.get('id') or '')[:12]}")
    lines.append(f"contract size   : {spec.get('contract_size')} "
                 f"(1 contract = {spec.get('contract_size')} of the base asset)")
    lines.append(f"amount step     : {spec.get('amount_step')}  |  tick size "
                 f"{spec.get('tick_size')}  |  venue min "
                 f"{spec.get('min_amount') if spec.get('min_amount') else 'not published'}")
    if snap.get("recent_fills"):
        last = snap["recent_fills"][-1]
        lines.append(f"last fill       : {float(last.get('amount') or 0):g} @ "
                     f"{float(last.get('price') or 0):,.2f}")
    lines.append(f"trigger listing : "
                 f"{'available' if snap.get('triggers_listable') else 'REFUSED by this venue/key'}")
    lines.append("-" * 62)
    lines.append(f"VERDICT: {verdict.get('action', 'none').upper()}")
    for item in verdict.get("findings", []):
        lines.append(f"  * {item}")
    return "\n".join(lines)


def cancel_dangling(client, snap: dict, symbol: str, owned_ids=None) -> dict:
    """Cancel PROTECTIVE orders **this bot recorded**, while flat. Armed CLI only.

    [MONEY] OWNERSHIP IS AN ID, EVEN HERE. An earlier version of this function cancelled
    every visible order that merely looked protective (reduce-only), which is inference, not
    ownership: a manual stop or another tool's limit is reduce-only too, and cancelling it
    is acting outside this bot's remit. It now cancels exactly the ids persisted in
    `state.dangling_orders` — orders Mel itself recorded as unresolved. Everything else in
    the book is reported and left alone, and with no persisted ids this refuses outright.

    Other refusals, both deliberate:
      * a live position keeps its protection;
      * an order that is NOT reduce-only is never a candidate anyway.
    """
    pos = snap.get("position") or {}
    if pos.get("active") and float(pos.get("contracts") or 0) > 0:
        return {"cancelled": {}, "refused": "venue holds a position — refusing to strip "
                                            "its protection"}
    owned = {str(i) for i in (owned_ids or []) if i}
    visible = {str(o.get("id") or ""): o for o in (snap.get("orders") or [])}
    if not owned:
        return {"cancelled": {}, "owned_ids": [],
                "refused": "no persisted mel-owned order ids — refusing to cancel orders "
                           "this bot cannot prove it placed",
                "visible": sorted(visible)}
    cancelled, skipped = {}, {}
    for oid in owned:
        if oid not in visible:
            # Already gone (or never visible): nothing to cancel, and that is fine.
            cancelled[oid] = True
            continue
        if ex.is_reduce_only(visible[oid]) is False:
            skipped[oid] = "not reduce-only — left alone"
            continue
        cancelled[oid] = ex.cancel(client, symbol, oid)
    out = {"cancelled": cancelled, "owned_ids": sorted(owned)}
    others = sorted(set(visible) - owned)
    if others:
        out["left_alone"] = {oid: "not a mel-owned id — reported, not cancelled"
                             for oid in others}
    if skipped:
        out["skipped"] = skipped
    return out
