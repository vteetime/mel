"""Regression tests for the fee-window defect found on Mel's #355 manual close (2026-09-20).

THE BUG: `settled_close` queried the venue ledger from `opened_ms`, but that stamp is written only
once the ENTRY is confirmed (engine.py:404, ~3 s after the entry fill). A window starting there
excludes the entry leg's own `futures trade` ledger row, so the ledger returned the EXIT fee alone
— and because the ledger was preferred over the fills, `fee_usdt` recorded only the exit fee and
`pnl_net` was overstated by the entry fee on every Mel close. On #355 that is 0.1827 USDT, ~4.8%
of the trade's P&L.

TWO INDEPENDENT FIXES, both asserted here:
  1. the ledger window is backed off from the earliest fill we hold (so the entry leg is included);
  2. when the ledger still disagrees materially with the fees attached to OUR OWN fills, the fills
     win — they are the only source that is both venue-stated per fill and complete for this trade,
     whereas the ledger is summed by contract + window and can miss a leg or absorb another trade.

Each test is named for the behaviour it locks in, so a future change that reopens this fails loudly.
"""
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as C            # noqa: E402
import settlement            # noqa: E402
from db import TradeDB        # noqa: E402
from engine import Engine     # noqa: E402
from state import State       # noqa: E402
from test_dry_run_scenarios import FakeKraken, SCHEMA, SYM, long_decision  # noqa: E402

CFG = C.load_config()

# The real #355 numbers, so the arithmetic in these tests is the arithmetic that was wrong.
ENTRY_TS = 1789901126646          # 2026-09-20T10:45:26.646Z, the entry fill
OPENED_MS = 1789901129719         # ...+3.073s: when the engine stamped opened_ms, AFTER the fill
EXIT_TS = 1789917381682           # 2026-09-20T15:16:21.682Z, the exit fill
ENTRY_PX, EXIT_PX, QTY = 2573.6, 2601.9, 0.142
ENTRY_FEE = ENTRY_PX * QTY * 0.0005      # 0.18272
EXIT_FEE = EXIT_PX * QTY * 0.0005        # 0.18473
GROSS = (EXIT_PX - ENTRY_PX) * QTY       # 4.0186


def _fill(order_id, amount, price, ts, fee):
    return {"info": {"orderId": order_id, "fee": fee}, "order": order_id, "amount": amount,
            "price": price, "fee": {"cost": fee}, "timestamp": ts}


def _client(ledger_fee):
    b = FakeKraken()
    b.fills = [_fill("ex", QTY, EXIT_PX, EXIT_TS, EXIT_FEE),
               _fill("en", QTY, ENTRY_PX, ENTRY_TS, ENTRY_FEE)]
    b.ledger_fee = ledger_fee
    return b


def _settle(client, **kw):
    return settlement.settled_close(
        client, SYM, close_order_id="ex", entry_order_id="en", entry_price=ENTRY_PX,
        contracts=QTY, contract_size=1.0, fallback_price=0.0, opened_ms=OPENED_MS, **kw)


# ── fix 1: the window includes the entry leg ─────────────────────────────────

def test_ledger_window_is_backed_off_to_before_the_entry_fill():
    """`opened_ms` is stamped AFTER the entry fill, so a window starting there loses the entry
    leg's own ledger row. The query must therefore start from the earliest fill we hold."""
    b = _client(ledger_fee=ENTRY_FEE + EXIT_FEE)
    seen = {}

    def spy(params=None):
        seen.update(params or {})
        return {"logs": []}

    b.historyGetAccountLog = spy
    _settle(b)
    assert "since" in seen, "the ledger must be queried with an explicit window"
    assert int(seen["since"]) <= ENTRY_TS - 60_000, (
        f"window starts at {seen['since']}, which is after the entry fill {ENTRY_TS}: the entry "
        f"fee would fall outside it (this is the #355 defect)")
    assert int(seen["since"]) < OPENED_MS, "the window must reach back past opened_ms"


# ── fix 2: a short/false ledger cannot undercharge ───────────────────────────

def test_a_window_short_ledger_cannot_undercharge_the_fee():
    """The live failure: the ledger returned the exit fee alone (0.1847) while our fills carried
    both legs (0.36746). The fills must win, and the row must carry the full cost."""
    res = _settle(_client(ledger_fee=EXIT_FEE))
    assert res["ledger_disagrees"] is True, "a partial ledger must be surfaced, not silently used"
    assert res["fee_usdt"] == pytest.approx(ENTRY_FEE + EXIT_FEE, abs=1e-9)
    assert res["fee_from_ledger"] is False
    assert res["fee_estimated"] is False, "the fills are the venue's own numbers, not an estimate"
    assert res["price_verified"] is True
    assert res["exit_price"] == pytest.approx(EXIT_PX)
    # the figure the row's P&L is built from
    assert GROSS - res["fee_usdt"] == pytest.approx(3.6511, abs=1e-3), (
        "net must be gross minus BOTH legs; the old behaviour gave 3.8339, overstating by 0.1827")


def test_a_matching_ledger_is_still_preferred():
    """A ledger that agrees remains the authoritative source — the fix must not demote it."""
    b = _client(ledger_fee=ENTRY_FEE + EXIT_FEE)
    res = _settle(b)
    assert res["ledger_disagrees"] is False
    assert res["fee_from_ledger"] is True
    assert res["fee_usdt"] == pytest.approx(ENTRY_FEE + EXIT_FEE, abs=1e-9)


def test_a_ledger_absorbing_another_trade_is_rejected():
    """The opposite contamination: an unrelated ETH trade inside the window inflates the ledger.
    Our own fills are complete for THIS trade, so they win."""
    res = _settle(_client(ledger_fee=12.0))
    assert res["ledger_disagrees"] is True
    assert res["fee_usdt"] == pytest.approx(ENTRY_FEE + EXIT_FEE, abs=1e-9)


def test_incomplete_fills_do_not_beat_the_ledger():
    """A fill that omits its fee is represented as 0, so our own figure would be missing a leg.
    The ledger must stay authoritative — an undercount is not evidence against it (Paul r2)."""
    b = FakeKraken()
    b.fills = [_fill("ex", QTY, EXIT_PX, EXIT_TS, EXIT_FEE),
               _fill("en", QTY, ENTRY_PX, ENTRY_TS, 0.0)]      # entry fee missing
    b.ledger_fee = ENTRY_FEE + EXIT_FEE
    res = _settle(b)
    assert res["ledger_disagrees"] is False, "no false alarm when we cannot judge"
    assert res["fee_usdt"] == pytest.approx(ENTRY_FEE + EXIT_FEE, abs=1e-9)
    assert res["fee_from_ledger"] is True


def test_real_fills_are_not_labelled_estimated_when_the_ledger_is_unreadable():
    """With no ledger, the fee is the sum of OUR OWN fills' venue-stated fees: measured, not
    estimated. Labelling a real fee 'estimated' is the same provenance defect in reverse."""
    res = _settle(_client(ledger_fee=None))          # historyGetAccountLog returns no rows
    assert res["fee_usdt"] == pytest.approx(ENTRY_FEE + EXIT_FEE, abs=1e-9)
    assert res["fee_estimated"] is False
    assert res["settlement_source"] == "fills"


# ── end to end: the closed row carries both legs ─────────────────────────────

def test_engine_close_records_both_leg_fees(tmp_path):
    """The reported bug, end to end: a closed row must net the exit fee AND the entry fee."""
    p = tmp_path / "t.db"
    con = sqlite3.connect(p)
    con.executescript(SCHEMA)
    con.commit()
    con.close()
    client = FakeKraken()
    state = State(tmp_path / "state.json")
    db = TradeDB(p)
    eng = Engine(CFG, client, state, db, lambda *a, **k: None, dry_run=False)
    from test_review_fixes import size_for
    eng.try_enter(long_decision(), size_for(eng))
    # In production the ledger carries only the exit leg; reproduce exactly that.
    exit_side_fee = sum(float((f.get("fee") or {}).get("cost") or 0.0)
                        for f in client.fills if str(f.get("side") or "") == "sell")
    client.ledger_fee = exit_side_fee or None
    client.trigger(state.get_open()["tp_order_id"])  # type: ignore[index]
    eng.manage(None)
    row = db._conn.execute(
        "SELECT fee_usdt, pnl_net, entry_price, exit_price, contracts, status FROM trades "
        "WHERE status='closed'").fetchone()
    assert row is not None, "the close must have written a row"
    both = sum(float((f.get("fee") or {}).get("cost") or 0.0) for f in client.fills)
    assert row["fee_usdt"] == pytest.approx(both, abs=1e-6), (
        f"row fee {row['fee_usdt']} must be BOTH legs ({both}); recording only the exit fee is the "
        f"#355 defect")
    gross = (float(row["exit_price"]) - float(row["entry_price"])) * float(row["contracts"])
    assert row["pnl_net"] == pytest.approx(gross - both, abs=1e-3)   # rows store 4dp


# ── round-3: the account log's realized_pnl is an independent P&L figure ─────

def test_the_account_log_realized_pnl_is_surfaced():
    """Kraken's account log states a per-fill realized P&L GROSS of fee (the fee is a separate
    field on the same `futures trade` row). It is the only independent P&L figure the venue
    publishes through ccxt — without it a `fills`/`exchange_ledger` row is checked against
    nothing but its own inputs (Paul r3)."""
    b = _client(ledger_fee=ENTRY_FEE + EXIT_FEE)
    orig = b.historyGetAccountLog

    def with_rpnl(params=None):
        out = orig(params)
        out["logs"] = [dict(r, realized_pnl=GROSS) if r.get("info") == "futures trade" else r
                       for r in (out.get("logs") or [])]
        return out

    b.historyGetAccountLog = with_rpnl
    res = _settle(b)
    assert res["pnl_realized"] == pytest.approx(GROSS, abs=1e-9), (
        "the account log's realized P&L must be surfaced for the cross-check")


def test_no_realized_pnl_figure_is_not_misread_as_zero():
    """A ledger without the realized_pnl field must yield None, not 0.0 — a false zero would
    read as 'the venue says the trade made nothing' and mis-fire the cross-check."""
    res = _settle(_client(ledger_fee=ENTRY_FEE + EXIT_FEE))
    assert res.get("pnl_realized") is None, (
        "an absent realized_pnl field must be None, not 0.0")


def test_a_gross_pnl_disagreement_with_the_account_log_alerts(tmp_path):
    """The engine must alert when our gross P&L diverges materially from the account log's
    realized P&L (the independent figure), instead of recording the number silently."""
    p = tmp_path / "t.db"
    con = sqlite3.connect(p)
    con.executescript(SCHEMA)
    con.commit()
    con.close()
    client = FakeKraken()
    state = State(tmp_path / "state.json")
    db = TradeDB(p)
    alerts = []
    eng = Engine(CFG, client, state, db, alerts.append, dry_run=False)
    from test_review_fixes import size_for
    eng.try_enter(long_decision(), size_for(eng))
    # The account log reports a realized P&L materially different from our fills' gross.
    def wrong_rpnl(params=None):
        return {"logs": [{"contract": "pf_xrpusd", "info": "futures trade",
                          "fee": 0.27, "realized_pnl": -123.45}]}
    client.historyGetAccountLog = wrong_rpnl
    client.trigger(state.get_open()["tp_order_id"])
    eng.manage(None)
    joined = " ".join(alerts)
    assert "realized P&L" in joined, (
        f"a material gross-P&L gap must alert, got: {alerts!r}")


def test_the_cross_check_does_not_false_alarm_on_a_healthy_close(tmp_path):
    """Pin GROSS-vs-GROSS (Paul r4): a healthy close whose fee is large must NOT alert on the
    account-log cross-check, because the comparison is gross-to-gross. If the check subtracted
    the fee (net), a fee above the threshold would open a fee-sized gap and false-alarm — the
    mutation Paul's round-4 review showed no test would catch."""
    p = tmp_path / "t.db"
    con = sqlite3.connect(p)
    con.executescript(SCHEMA)
    con.commit()
    con.close()
    client = FakeKraken()
    state = State(tmp_path / "state.json")
    db = TradeDB(p)
    alerts = []
    eng = Engine(CFG, client, state, db, alerts.append, dry_run=False)
    from test_review_fixes import size_for
    eng.try_enter(long_decision(), size_for(eng))
    pos = state.get_open()
    entry = float(pos["entry_price"])
    contracts = float(pos["contracts"])
    # The exit level is the resting TP; the cross-check's `px` is that fill's VWAP.
    exit_px = float(client.order_of_type("limit")["price"])
    gross = (exit_px - entry) * contracts
    client.trigger(pos["tp_order_id"])
    # Inflate the fee far above the alert threshold. A GROSS comparison (gap 0) stays silent;
    # a NET comparison would open a gap of exactly this fee and false-alarm.
    big = 10.0
    for f in client.fills:
        f["fee"]["cost"] = big
        f["info"]["fee"] = big
    client.ledger_fee = big * 2        # agrees with the inflated fills -> no disagreement
    client.historyGetAccountLog = lambda params=None: {
        "logs": [{"contract": "pf_xrpusd", "info": "futures trade",
                  "fee": big * 2, "realized_pnl": gross}]}
    eng.manage(None)
    joined = " ".join(alerts)
    assert "realized P&L" not in joined, (
        f"a gross-to-gross match must not false-alarm, got: {alerts!r}")


def test_a_force_close_gross_gap_alerts(tmp_path):
    """[MONEY] The aborted-entry (force-close) path must run the same account-log realized-P&L
    cross-check as _finish: its P&L was previously self-derived with no independent check
    (Paul r4 follow-up)."""
    p = tmp_path / "t.db"
    con = sqlite3.connect(p)
    con.executescript(SCHEMA)
    con.commit()
    con.close()
    client = FakeKraken()
    state = State(tmp_path / "state.json")
    db = TradeDB(p)
    alerts = []
    eng = Engine(CFG, client, state, db, alerts.append, dry_run=False)
    client.fail_orders = {"stop"}
    client.historyGetAccountLog = lambda params=None: {
        "logs": [{"contract": "pf_xrpusd", "info": "futures trade",
                  "fee": 0.0, "realized_pnl": -123.45}]}
    from test_review_fixes import size_for
    eng.try_enter(long_decision(), size_for(eng))
    joined = " ".join(alerts)
    assert "realized P&L" in joined, (
        f"an aborted entry with a material gross gap must alert, got: {alerts!r}")
