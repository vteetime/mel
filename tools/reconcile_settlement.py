#!/usr/bin/env python3
"""Reconcile a settlement that was recorded from a LEVEL instead of from evidence.

A row with `settlement_source='level_fallback'` exists because the venue never produced fill
evidence for the exit: its price is a level the venue *should* have executed at, so its P&L is an
estimate. This tool also repairs a row with an IMPOSSIBLE price (`exit_price <= 0`), which would
otherwise compute a loss of the position's whole notional and wedge the caps. Mel keeps that row in the exposure caps (an invisible loss would be worse) but halts
NEW entries while it stands (`guards.max_unverified_settlements`), and excludes it from the loss
streak. This is the operator's tool to close the loop: give it the real exit price you read off
the venue, and it recomputes the row honestly and clears the halt.

    python3 tools/reconcile_settlement.py --list
    python3 tools/reconcile_settlement.py 12 2547.30
    python3 tools/reconcile_settlement.py 12 2547.30 --fee 0.41 --yes

It refuses to touch a row that was NOT a level fallback (there is nothing to fix there), prints
the before/after, and only writes with `--yes` (or when the row is unambiguous and you pass the
price — dry by default is friendlier, but a stuck bot needs a one-liner, so `--yes` is explicit).
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _cfg_db_path() -> Path:
    import yaml
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    return ROOT / str(cfg["ops"]["db_path"])


def _bot_id() -> str:
    """WHO this tool may touch. The ledger is SHARED by every bot on this host, so an unscoped
    query does not merely show the wrong rows — it lets an operator write a fabricated exit price
    into another bot's OPEN row and corrupt that bot's guard accounting. (Found by review: the
    tool used to report "new entries are halted" while listing four open positions belonging to
    three other bots.)

    The id comes from the module that WRITES the rows, not from config.yaml. `db.BOT_ID` is what
    every ledger row actually carries; config.yaml's `bot.id` is a second source of truth that can
    drift from it. If they disagreed, scoping to the config value would hide this bot's own rows and
    leave a real level_fallback row with no working escape — a silent wedge in the tool that exists
    to be the way out of one. (Second source of truth flagged by review, 2026-09-20.)
    """
    try:
        import db as _db
        return str(_db.BOT_ID)
    except Exception:                     # never leave the operator with no tool at all
        import yaml
        return str(yaml.safe_load((ROOT / "config.yaml").read_text())["bot"]["id"])


def _rows(con) -> list[dict]:
    """Only THIS bot's CLOSED rows: an open row has no settlement to correct, and another bot's
    row is not this tool's business."""
    cur = con.execute(
        """SELECT id, symbol, direction, entry_price, exit_price, contracts, leverage,
                  notional, pnl_pct, pnl_net, fee_usdt, exit_reason, settlement_source,
                  exit_time
             FROM trades WHERE bot_id=? AND status='closed'
               AND (settlement_source='level_fallback' OR exit_price <= 0)
             ORDER BY id DESC""", (_bot_id(),))
    return [dict(r) for r in cur.fetchall()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("trade_id", nargs="?", type=int)
    ap.add_argument("exit_price", nargs="?", type=float)
    ap.add_argument("--fee", type=float, default=None, help="total fee in USDT, if known")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--yes", action="store_true", help="actually write the correction")
    ap.add_argument("--db", default=None)
    a = ap.parse_args()

    db = Path(a.db) if a.db else _cfg_db_path()
    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row

    pending = _rows(con)
    if a.list or a.trade_id is None:
        if not pending:
            print("no settlement rests on an estimated price — nothing to reconcile")
            return 0
        print(f"{len(pending)} of {_bot_id()}'s own settlement(s) are ESTIMATED or IMPOSSIBLE "
              f"(its new entries are halted until reconciled). Other bots' rows are not listed "
              f"and cannot be touched by this tool:\n")
        for r in pending:
            print(f"  trade #{r['id']}  {r['symbol']}  {str(r['direction']).upper()}  "
                  f"entry {r['entry_price']}  booked exit {r['exit_price']} "
                  f"({r['exit_reason']})  pnl_pct {r['pnl_pct']:.2f}  "
                  f"pnl ${r['pnl_net']:.2f}  at {r['exit_time']}")
        print("\nrun:  python3 tools/reconcile_settlement.py <trade_id> <real_exit_price> "
              "[--fee X] --yes")
        return 0

    row = con.execute("SELECT * FROM trades WHERE id=? AND bot_id=?",
                      (a.trade_id, _bot_id())).fetchone()
    if row is None:
        foreign = con.execute("SELECT bot_id, status FROM trades WHERE id=?",
                              (a.trade_id,)).fetchone()
        if foreign is not None:
            print(f"trade #{a.trade_id} belongs to {foreign['bot_id']!r}, not {_bot_id()!r} — "
                  f"refusing to touch another bot's row")
            return 1
        print(f"no trade #{a.trade_id} for {_bot_id()!r}")
        return 1
    if str(row["status"]) != "closed":
        print(f"trade #{a.trade_id} is still {row['status']} — it has no settlement to correct; "
              f"an open position must be closed on the venue, not repriced here")
        return 1
    corrupt = float(row["exit_price"] or 0) <= 0
    if str(row["settlement_source"] or "") != "level_fallback" and not corrupt:
        print(f"trade #{a.trade_id} is '{row['settlement_source']}' with a real price — "
              f"not an estimated settlement; refusing to rewrite a measured row")
        return 1
    if corrupt:
        print(f"trade #{a.trade_id} carries an IMPOSSIBLE exit price ({row['exit_price']}) — "
              f"this row cannot be trusted by any guard until it is repriced")
    if not a.exit_price or a.exit_price <= 0:
        print("a real exit price is required")
        return 1

    # The table stores `notional`, not contract_size/margin: contract size is recovered from
    # notional / (entry * contracts) and the margin committed is notional / leverage, which is
    # exactly the base the row's own pnl_pct is expressed against (ROE on margin).
    entry = float(row["entry_price"])
    lev = float(row["leverage"] or 1) or 1.0
    contracts = float(row["contracts"] or 0)
    notional = float(row["notional"] or 0.0)
    if notional <= 0 and entry > 0:
        notional = entry * contracts
    if entry <= 0 or contracts <= 0 or notional <= 0:
        print("refusing to write: the row lacks the entry figures needed to reprice it "
              "(entry_price/contracts/notional)")
        return 1
    csize = notional / (entry * contracts)
    d = 1.0 if str(row["direction"]).lower() == "long" else -1.0
    gross = (float(a.exit_price) - entry) * d * contracts * csize
    fee = float(a.fee) if a.fee is not None else float(row["fee_usdt"] or 0.0)
    pnl_net = gross - fee
    margin = notional / lev
    pnl_pct = (pnl_net / margin * 100.0) if margin else 0.0

    print(f"trade #{a.trade_id}  booked: exit {row['exit_price']} -> "
          f"pnl_pct {row['pnl_pct']:.2f}  pnl ${row['pnl_net']:.2f}")
    print(f"trade #{a.trade_id}  real  : exit {a.exit_price} -> "
          f"pnl_pct {pnl_pct:.2f}  pnl ${pnl_net:.2f}  (fee ${fee:.2f})")
    if not a.yes:
        print("\ndry run — re-run with --yes to write this correction")
        return 0
    con.execute(
        """UPDATE trades SET exit_price=?, pnl_net=?, pnl_pct=?, fee_usdt=?,
                            settlement_source='manual_reconcile'
           WHERE id=? AND bot_id=? AND status='closed'
             AND (settlement_source='level_fallback' OR exit_price <= 0)""",
        (float(a.exit_price), round(pnl_net, 6), round(pnl_pct, 6), round(fee, 6),
         a.trade_id, _bot_id()))
    con.commit()
    left = len(_rows(con))
    print(f"\nwritten. {left} estimated settlement(s) remain "
          f"(new entries resume when this reaches 0).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
