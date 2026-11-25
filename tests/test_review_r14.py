"""Mel read every RESTING order as size zero, so protection looked absent.

Terra's verified verdict (2026-10-15, INBOX 20260715T075922Z-terra-mel-order-amount-verdict) reproduced
this on the LIVE book: Mel held 991 SHORT XRP @ 1.3172 with a reduce-only stop (a2fc321e…, trigger
1.3504) and a reduce-only take-profit (a2fc3221…, price 1.2648), BOTH resting with `unfilledSize=991`,
and the bot read both as size 0:

    $ .venv/bin/python3 run_bot.py --sync
    resting orders  : 2 (1 stop-type)
       - lmt      buy  amt 0 price 1.26 reduceOnly=True id=a2fc3221-d72
       - stop     buy  amt 0 trigger 1.35 reduceOnly=True id=a2fc321e-2bd
    VERDICT: PROTECT
      * UNDER-PROTECTED: stops cover 0 of 991

Root cause is Mel's OWN raw-order wrapper (not ccxt): Kraken's `/derivatives/api/v3/openorders` rows
carry `unfilledSize` / `filledSize` and no `quantity`, and `exchange.open_orders` built each wrapper
from `o.get("quantity")`. Every size comparison in the bot then compared the real 991 against 0:
the coverage sum in `sync.reconcile`, the duplicate search in `find_matching_order`, the ambiguous-TP
adoption test and the ambiguous-stop candidate test in the engine.

These tests pin the size accessor, the duplicate search, the coverage verdict, the partially-filled
case (which is where a total-size read over-states coverage) and the fail-closed handling of a size
that cannot be read at all. They fail on the pre-fix tree.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import exchange as ex   # noqa: E402
import sync             # noqa: E402

SYM = "XRP/USD:USD"
STOP_ID = "a2fc321e-2bdf-4b20-bab6-e5d2ea500107"
TP_ID = "a2fc3221-d727-4ee1-b191-49c6273bb5dd"
POSITION = 991.0


class KrakenRawBook:
    """Mel's PRIMARY read: the account's own open-orders endpoint, no ccxt mapping involved.

    `fetch_open_orders` answers the trigger-specific query with the authenticationError this
    key actually returns (see exchange.open_orders), so nothing here can quietly get its size
    from ccxt instead of from the raw rows under test.
    """

    options = {"dryRun": True}

    def __init__(self, rows):
        self.rows = rows

    def privateGetOpenorders(self):
        return {"openOrders": [dict(r) for r in self.rows]}

    def fetch_open_orders(self, symbol=None, params=None):
        if params and params.get("stop"):
            raise RuntimeError("authenticationError")
        return []


def _row(order_id, *, otype="lmt", side="buy", unfilled=991, filled=0, limit=None, stop=None,
         reduce_only=True, status="untouched"):
    """A row shaped exactly like Kraken's live `/derivatives/api/v3/openorders` payload."""
    r = {"order_id": order_id, "symbol": "PF_XRPUSD", "side": side, "orderType": otype,
         "limitPrice": limit, "stopPrice": stop, "reduceOnly": reduce_only, "status": status}
    if unfilled is not None:
        r["unfilledSize"] = unfilled
    if filled is not None:
        r["filledSize"] = filled
    return r


def live_book():
    """The live book as Terra read it: two untouched reduce-only 991 orders, no `quantity`."""
    return KrakenRawBook([_row(TP_ID, limit=1.2648),
                          _row(STOP_ID, otype="stop", stop=1.3504)])


def test_the_raw_wrapper_carries_the_venues_own_size_not_a_missing_quantity():
    """`unfilledSize`/`filledSize` ARE the size on this endpoint; `quantity` never appears."""
    book = live_book()
    orders = {o["id"]: o for o in ex.open_orders(book, SYM, strict=True)}
    assert set(orders) == {STOP_ID, TP_ID}, orders
    assert ex.order_amount(orders[STOP_ID]) == pytest.approx(POSITION)
    assert ex.order_amount(orders[TP_ID]) == pytest.approx(POSITION)


def test_find_matching_order_hits_both_live_resting_orders():
    """The duplicate search must see the orders that are demonstrably on the book.

    Both misses were live: MATCH_STOP=None and MATCH_LIMIT=None while both orders rested.
    """
    book = live_book()
    stop = ex.find_matching_order(book, SYM, side="buy", amount=POSITION, trigger=1.3504,
                                  kind="stop")
    limit = ex.find_matching_order(book, SYM, side="buy", amount=POSITION, price=1.2648,
                                   kind="limit")
    assert str((stop or {}).get("id")) == STOP_ID, stop
    assert str((limit or {}).get("id")) == TP_ID, limit


def test_a_partially_filled_order_reports_what_is_still_outstanding():
    """A half-filled TP (495 done, 496 left) is 496 of protection, never 991.

    This is the trap the fix has to get right: a total-size read of a partially filled order
    over-states coverage, and the engine's duplicate tests compare against what is LEFT.
    """
    book = KrakenRawBook([_row(TP_ID, limit=1.2648, unfilled=496, filled=495)])
    o = ex.open_orders(book, SYM)[0]
    assert ex.order_amount(o) == pytest.approx(496.0), ex.order_amount(o)
    # the 496 that is left matches a 496-sized remainder ...
    assert ex.find_matching_order(book, SYM, side="buy", amount=496.0, price=1.2648,
                                  kind="limit") is not None
    # ... and the 991 total must NOT be reported as outstanding protection for it
    assert ex.find_matching_order(book, SYM, side="buy", amount=991.0, price=1.2648,
                                  kind="limit") is None


def test_a_quantity_shaped_row_still_reads():
    """Compatibility: a venue/shape that sends `quantity` instead of `unfilledSize`."""
    book = KrakenRawBook([{"order_id": STOP_ID, "orderType": "stop", "side": "buy",
                           "stopPrice": 1.3504, "reduceOnly": True, "status": "open",
                           "quantity": 991}])
    o = ex.open_orders(book, SYM)[0]
    assert ex.order_amount(o) == pytest.approx(POSITION)


def test_the_live_book_is_reported_as_fully_covered():
    """`--sync`'s verdict for this book: 991 of 991, and no UNDER-PROTECTED line.

    `stops`/`limits` are classified exactly as `sync.snapshot()` classifies them.
    """
    book = live_book()
    orders = ex.open_orders(book, SYM)
    snap = {
        "ts": 0, "symbol": SYM,
        "balance": {"free": 100.0, "total": 200.0, "currency": "USD"},
        "position": {"active": True, "contracts": POSITION, "side": "short",
                     "entry_price": 1.3172, "unrealized": 4.1, "error": None},
        "orders": orders,
        "recent_fills": [],
        "spec": {"contract_size": 1.0, "amount_step": 1.0, "tick_size": 0.0001,
                 "min_amount": None},
        "triggers_listable": False,
    }
    snap["stops"] = [o for o in orders
                     if str(o.get("type") or "").lower() in ("stop", "stopmarket", "stop-loss",
                                                             "stop_loss")
                     or ex.order_trigger(o) > 0]
    snap["limits"] = [o for o in orders if o not in snap["stops"]]

    verdict = sync.reconcile(snap, {"contracts": POSITION})
    assert verdict["stop_coverage"] == pytest.approx(POSITION), verdict
    report = sync.format_report(snap, verdict, {"bot": {"symbol": SYM}})
    assert "UNDER-PROTECTED" not in report, report
    assert "amt 0 " not in report, report
    assert report.count("amt 991") == 2, report


def test_an_order_with_no_size_field_at_all_is_unknown_and_is_not_accepted_as_protection():
    """Fail closed on an unreadable size — the shape change must not read as "protected".

    `order_amount` still answers 0.0 for a size that is absent everywhere, and
    `stop_is_confirmed` must NOT accept a stop whose size it cannot read. The reason carries
    the word "unreadable" because engine.py's per-tick verifier keys on it to avoid a
    cancel-and-replace on a read it could not perform.
    """
    book = KrakenRawBook([_row(STOP_ID, otype="stop", stop=1.3504, unfilled=None, filled=None)])
    o = ex.open_orders(book, SYM)[0]
    assert ex.order_amount(o) == 0.0, ex.order_amount(o)

    ok, why = ex.stop_is_confirmed(book, SYM, STOP_ID, side="buy", min_amount=POSITION,
                                   require_breakeven=False, entry=1.3172, trigger=1.3504)
    assert not ok, "a stop whose size cannot be read was accepted as protection"
    assert "unreadable" in why, why


def test_a_size_the_venue_states_as_zero_is_a_read_not_an_unreadable_field():
    """The boundary of the fail-closed branch above, pinned deliberately and on the record.

    An explicit `0` from the venue is a READ (the order can fill nothing more), so it takes the
    SIZE-specific rejection (`size 0 < required 991`) and never the `unreadable` reason — the
    per-tick verifier keys on that word to leave an order alone, and a stated zero is not a
    mapping failure. Both branches refuse the stop; only the routing differs.
    """
    book = KrakenRawBook([_row(STOP_ID, otype="stop", stop=1.3504, unfilled=0, filled=991)])
    o = ex.open_orders(book, SYM)[0]
    assert ex.order_amount(o) == 0.0
    assert ex._size_readable(o) is True

    ok, why = ex.stop_is_confirmed(book, SYM, STOP_ID, side="buy", min_amount=POSITION,
                                   require_breakeven=False, entry=1.3172, trigger=1.3504)
    assert not ok, "an order that can fill nothing was accepted as protection"
    assert "unreadable" not in why, why
    assert "size 0" in why, why
