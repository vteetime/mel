"""Two recording defects on Mel's history, pinned so they cannot return.

1. `exchange_close` — nine of Mel's 26 closed trades were labelled with it. That
   label means "the position vanished and the bot could not tell which order did
   it". On Kraken that was the NORMAL case, not an exception: a filled order is
   dropped from the order book, so `_order_filled_state()` gets ordernotfound for
   BOTH the take-profit and the stop, and the price heuristic then fails too (the
   hourly tick runs minutes after the fill). The answer was already in hand: the
   fills `settled_close` reads for the exit price each name the order id that
   produced them. Five of the nine were take-profits, four were stop-outs.

2. The entry `trigger` said only "scoreboard", which is the same word for a
   marginal score-5 entry and a 9. The score is what a review actually needs.

Nothing here asserts a GUESS: attribution comes from a fill on a specific order
id, and a trade whose fills name neither protection order must stay unlabelled.
"""

import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as C                      # noqa: E402
import settlement                       # noqa: E402
from db import TradeDB                  # noqa: E402
from engine import Engine               # noqa: E402
from state import State                 # noqa: E402
from test_dry_run_scenarios import SCHEMA, FakeKraken   # noqa: E402

CFG = C.load_config()
SYM = "XRP/USD:USD"

# The real #556 numbers (2026-11-14): short 1235 XRP, TP filled as a MAKER limit at
# exactly the TP price, and the bot recorded `exchange_close`.
ENTRY_PX, TP_PX, SL_PX, QTY = 0.9908, 0.9512, 1.0156, 1235.0


def _fill(order_id, amount, price, ts=1794674798680, fee=0.0):
    return {"info": {"orderId": order_id, "fee": fee}, "order": order_id,
            "amount": amount, "price": price, "fee": {"cost": fee}, "timestamp": ts}


def _rig(tmp_path, client, db_path=None):
    state = State(tmp_path / "state.json")
    path = db_path or (tmp_path / "t.db")
    con = sqlite3.connect(path)
    con.executescript(SCHEMA)
    con.commit()
    con.close()
    db = TradeDB(path)
    eng = Engine(CFG, client, state, db, lambda m: None, dry_run=False)
    return eng, db, path


def _row(path, trade_id):
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    try:
        return con.execute("SELECT * FROM trades WHERE id=?", (trade_id,)).fetchone()
    finally:
        con.close()


def _pos(trade_id, tp_order="tp-1", sl_order="sl-1"):
    return {
        "trade_id": trade_id, "direction": "short", "entry_price": ENTRY_PX,
        "contracts": QTY, "leverage": 2, "contract_size": 1.0, "margin_usdt": 611.8,
        "entry_order_id": "en", "tp_order_id": tp_order, "sl_order_id": sl_order,
        "tp_price": TP_PX, "sl_price": SL_PX, "opened_ms": 1794645000000,
        "entry_time": "2026-11-14T10:03:23+00:00",
    }


def _recorded(db, entry_score=None):
    return db.record_entry(
        symbol=SYM, direction="short", entry_price=ENTRY_PX, contracts=QTY,
        leverage=2, sl_price=SL_PX, tp1_price=TP_PX, tp2_price=0.0,
        notional=ENTRY_PX * QTY, sl_pct=0.025, trigger="scoreboard",
        guard_config={}, entry_score=entry_score,
    )


# ── settlement surfaces which order ids actually filled ─────────────────────

def test_settled_close_reports_the_order_ids_that_produced_fills():
    b = FakeKraken()
    b.fills = [_fill("tp-1", QTY, TP_PX), _fill("en", QTY, ENTRY_PX)]
    res = settlement.settled_close(
        b, SYM, close_order_id="", entry_order_id="en",
        extra_order_ids=["tp-1", "sl-1"],
        entry_price=ENTRY_PX, contracts=QTY, contract_size=1.0,
        fallback_price=0.0, opened_ms=1794645000000)
    assert "tp-1" in res["exit_order_ids"]
    assert "sl-1" not in res["exit_order_ids"], (
        "the stop never filled — listing it as an exit order would attribute the "
        "trade to an order that did nothing")


# ── _finish attributes from that evidence ───────────────────────────────────

def test_exchange_close_is_attributed_to_the_take_profit_from_its_fill(tmp_path):
    client = FakeKraken()
    client.fills = [_fill("tp-1", QTY, TP_PX), _fill("en", QTY, ENTRY_PX)]
    eng, db, path = _rig(tmp_path, client)
    tid = _recorded(db)

    eng._finish(_pos(tid), 0.0, "exchange_close")

    assert _row(path, tid)["exit_reason"] == "tp"


def test_exchange_close_is_attributed_to_the_stop_from_its_fill(tmp_path):
    client = FakeKraken()
    client.fills = [_fill("sl-1", QTY, SL_PX), _fill("en", QTY, ENTRY_PX)]
    eng, db, path = _rig(tmp_path, client)
    tid = _recorded(db)

    eng._finish(_pos(tid), 0.0, "exchange_close")

    assert _row(path, tid)["exit_reason"] == "sl"


def test_an_unattributable_close_is_not_given_a_reason_it_did_not_earn(tmp_path):
    """No fill names either protection order -> the label must stay honest.

    A fabricated 'sl' here would corrupt the loss streak the consecutive-loss
    breaker counts, which is exactly why the code refuses to guess.
    """
    client = FakeKraken()
    client.fills = [_fill("en", QTY, ENTRY_PX)]     # entry only, no exit fill
    eng, db, path = _rig(tmp_path, client)
    tid = _recorded(db)

    eng._finish(_pos(tid), 0.0, "exchange_close")

    reason = _row(path, tid)["exit_reason"]
    assert reason in (None, "", "exchange_close", "unverified_exit"), (
        f"unattributable close was recorded as {reason!r}")


def test_an_explicit_reason_is_never_overwritten_by_the_fills(tmp_path):
    """The hold cap and a manual close know why they closed; fills must not relabel them."""
    client = FakeKraken()
    client.fills = [_fill("tp-1", QTY, TP_PX), _fill("en", QTY, ENTRY_PX)]
    eng, db, path = _rig(tmp_path, client)
    tid = _recorded(db)

    eng._finish(_pos(tid), TP_PX + 0.01, "hold_cap")

    assert _row(path, tid)["exit_reason"] == "hold_cap"


# ── the entry score is persisted ────────────────────────────────────────────

def test_record_entry_persists_the_score(tmp_path):
    _eng, db, path = _rig(tmp_path, FakeKraken())
    tid = _recorded(db, entry_score=7.0)
    row = _row(path, tid)
    assert row["entry_score"] == pytest.approx(7.0)


def test_record_entry_without_a_score_stays_null_not_zero(tmp_path):
    """Absent must read as "unknown", never as a score of 0."""
    _eng, db, path = _rig(tmp_path, FakeKraken())
    tid = _recorded(db)
    assert _row(path, tid)["entry_score"] is None


def test_try_enter_writes_the_decisions_score():
    """Source contract: the entry path passes decision.confidence, not a constant."""
    import inspect
    src = inspect.getsource(Engine.try_enter)
    assert 'entry_score=float(getattr(decision, "confidence", 0) or 0)' in src
    assert '"decision"' in src and "score_long" in src, (
        "the losing side's score is what says how close the call was")


# ── exit detail: the structure behind the reason ─────────────────────────────

def test_exit_detail_names_the_clock_and_the_leg(tmp_path):
    eng, _db, _path = _rig(tmp_path, FakeKraken())
    pos = {"entry_time": "2026-11-14T10:00:00+00:00", "contracts": 1235.0}
    d = eng._exit_detail(pos, 0.9512, "tp", 48.12, None)
    assert d["cap"]["kind"] == "stop"
    assert d["cap"]["hours"] == 24, "the configured cap must be named"
    assert d["legs"][0]["leg"] == "tp"
    assert d["legs"][0]["price"] == pytest.approx(0.9512)
    assert "attributed_from" not in d, "no fills -> no attribution claim"


def test_exit_detail_marks_a_fill_attributed_exit(tmp_path):
    eng, _db, _path = _rig(tmp_path, FakeKraken())
    detail = eng._exit_detail({"entry_time": "2026-11-14T10:00:00+00:00", "contracts": 1235.0},
                              0.9512, "tp", 48.12, {"exit_order_ids": ["a338f730-45c2"]})
    assert detail["attributed_from"] == "venue_fill"
    assert detail["exit_order_ids"] == ["a338f730-45c2"]


def test_exit_detail_never_raises_on_hostile_position_fields(tmp_path):
    """The close path must not lose the row (or the state clear) to a bad field."""
    eng, _db, _path = _rig(tmp_path, FakeKraken())
    assert eng._exit_detail({"entry_time": "not-a-date", "contracts": "not-a-number"},
                            "bad", "hold_cap", "bad", None)["legs"][0]["contracts"] == 0.0
