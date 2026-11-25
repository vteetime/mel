#!/usr/bin/env python3
"""Backfill Mel's two recording gaps: the `exchange_close` causes and the entry score.

WHY THE CAUSE WAS LOST. `exchange_close` meant "the position vanished and the bot
could not tell which protective order took it". On Kraken that was routine: a filled
order is dropped from the order book, so `_order_filled_state()` got ordernotfound
for BOTH the take-profit and the stop, and the price heuristic failed too because the
hourly tick runs minutes after the fill. The evidence was never missing — the exit
price the bot recorded is the VWAP of the closing order's own fills.

EVIDENCE, strongest first:
  1. the venue fill for that order id (fillType maker = resting limit = the TP;
     taker = market stop), read from Kraken's fill ledger;
  2. where the ledger no longer reaches, the recorded exit price against the
     recorded TP/SL levels.

Every attribution records WHICH of the two it used in `exit_detail.attributed_from`,
plus the matched order id / level. Nothing is attributed that neither test supports.

Read-only until --apply.

    .venv/bin/python _backfill_mel_history.py            # dry run
    .venv/bin/python _backfill_mel_history.py --apply    # write
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from datetime import datetime, timedelta

DB = "/opt/data/cic/trades.db"
LOG = "/opt/data/mel/logs/mel.log"
BOT = "mel"
SYMBOL = "XRP/USD:USD"

TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
SCORE_RE = re.compile(r"\b(?:LONG|SHORT)\s+score=(\d+)")
FINISHED_RE = re.compile(r"order ([0-9a-f]{8}-[0-9a-f]{3}) is provably finished")
# An exit price this close to a recorded level is that level's fill: a limit TP fills
# AT its price, and a stop fills within a tick or two of its trigger.
LEVEL_TOL = 0.0005


def _dt(value):
    if not value:
        return None
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).replace(tzinfo=None)


def log_events():
    out = []
    with open(LOG, errors="replace") as fh:
        for line in fh:
            m = TS_RE.match(line)
            if m:
                out.append((datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S"), line.rstrip()))
    return out


def venue_fills_by_order():
    """{order_id: [fills]} from the venue. Empty on any failure — the level check then
    carries the attribution alone and says so."""
    try:
        sys.path.insert(0, "/opt/data/mel")
        import ccxt
        creds = {}
        for line in open("/opt/data/shared/credentials/kraken.env"):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                creds[k.strip()] = v.strip().strip('"').strip("'")
        client = ccxt.krakenfutures({"apiKey": creds["KRAKEN_API_KEY"],
                                     "secret": creds["KRAKEN_API_SECRET"],
                                     "enableRateLimit": True})
        out = {}
        for f in client.fetch_my_trades(SYMBOL, limit=1000) or []:
            oid = str((f.get("info") or {}).get("order_id") or f.get("order") or "")
            if oid:
                out.setdefault(oid, []).append(f)
        return out
    except Exception as exc:                                   # pragma: no cover
        print(f"# venue fills unavailable ({exc}) — level attribution only", file=sys.stderr)
        return {}


def attribute(trade, fills_by_order, short_ids):
    """(reason, detail) for one row, or (None, detail) when no evidence supports a call."""
    exit_px = float(trade["exit_price"] or 0)
    tp = float(trade["tp1_price"] or 0)
    sl = float(trade["sl_price"] or 0)

    # 1. a venue fill on one of the two protection orders
    for short in short_ids:
        for oid, fs in fills_by_order.items():
            if not oid.startswith(short):
                continue
            fill = fs[0]
            px = float(fill["price"])
            if tp and abs(px - tp) / tp <= LEVEL_TOL:
                return "tp", {"attributed_from": "venue_fill", "order_id": oid,
                              "fill_price": px, "fill_type": (fill.get("info") or {}).get("fillType")}
            if sl and abs(px - sl) / sl <= LEVEL_TOL:
                return "sl", {"attributed_from": "venue_fill", "order_id": oid,
                              "fill_price": px, "fill_type": (fill.get("info") or {}).get("fillType")}

    # 2. the recorded exit price against the recorded levels. The exit price IS the
    #    VWAP of the closing order's fills, so a match is still fill evidence — just
    #    not tied to a named order id any more.
    if tp and abs(exit_px - tp) / tp <= LEVEL_TOL:
        return "tp", {"attributed_from": "exit_price_vs_level", "level": "tp", "exit_price": exit_px}
    if sl and abs(exit_px - sl) / sl <= LEVEL_TOL:
        return "sl", {"attributed_from": "exit_price_vs_level", "level": "sl", "exit_price": exit_px}
    return None, {}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args(argv)

    events = log_events()
    fills_by_order = venue_fills_by_order()

    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        f"SELECT * FROM trades WHERE bot_id='{BOT}' ORDER BY entry_time").fetchall()

    plan = []
    for t in rows:
        et = _dt(t["entry_time"])
        if not et:
            continue
        # ── entry score, from the entry tick's own log line ──
        score = None
        for ts, line in events:
            if et - timedelta(minutes=3) <= ts <= et + timedelta(minutes=8):
                m = SCORE_RE.search(line)
                if m:
                    score = float(m.group(1))
                    break

        # ── exit attribution ──
        new_reason = t["exit_reason"]
        detail = {}
        if (t["exit_reason"] or "") == "exchange_close":
            xt = _dt(t["exit_time"])
            short_ids = []
            if xt:
                for ts, line in events:
                    if xt - timedelta(minutes=6) <= ts <= xt + timedelta(minutes=2):
                        m = FINISHED_RE.search(line)
                        if m:
                            short_ids.append(m.group(1))
            reason, detail = attribute(t, fills_by_order, short_ids)
            if reason:
                detail = {"source": "reconstructed", **detail}
                new_reason = reason

        plan.append({"id": t["id"], "entry_time": t["entry_time"],
                     "exit_reason_old": t["exit_reason"], "exit_reason_new": new_reason,
                     "score_old": t["entry_score"], "score_new": score,
                     "detail": detail})

    changed_reason = [p for p in plan if p["exit_reason_new"] != p["exit_reason_old"]]
    scored = [p for p in plan if p["score_new"] is not None]
    print(f"# {len(plan)} mel trades | reasons to fix: {len(changed_reason)} | scores recovered: {len(scored)}")
    for p in plan:
        if p["exit_reason_new"] != p["exit_reason_old"] or p["score_new"] is not None:
            print(json.dumps(p))

    if not args.apply:
        print("# dry run — pass --apply to write", file=sys.stderr)
        return 0

    with conn:
        for p in plan:
            fields, params = [], []
            if p["exit_reason_new"] != p["exit_reason_old"]:
                fields.append("exit_reason = ?")
                params.append(p["exit_reason_new"])
                fields.append("exit_detail = ?")
                params.append(json.dumps(p["detail"]))
            if p["score_new"] is not None:
                fields.append("entry_score = ?")
                params.append(p["score_new"])
            if not fields:
                continue
            params.append(p["id"])
            conn.execute(f"UPDATE trades SET {', '.join(fields)} WHERE id = ?", params)
    conn.close()
    print(f"# wrote {len(changed_reason)} reason fix(es) and {len(scored)} score(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
