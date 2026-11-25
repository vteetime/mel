"""Terra round-2 must-fix regressions on the size accessor (2026-10-15).

Two narrow defects were reproduced against the round-1 revision (INBOX
20261015T082435Z-terra-mel-order-size-review-r2):

 1. A stop the venue states as OUTSTANDING 0 was still accepted as protection. The
    round-1 form kept the old fail-open (`if amt and amt + 1e-9 < min_amount`) for a
    readable zero, so an open reduce-only stop with `remaining=0` on a 10-contract
    position returned (True, '') and `_ensure_sl` answered "ok". An order that can fill
    nothing protects nothing; the undersize rejection is now unconditional for
    min_amount > 0 and the UNREADABLE reason stays a separate, distinctly worded branch
    because engine.py keys on the word "unreadable" to choose the non-replacement path.

 2. `order_amount` ignored ccxt's unified top-level `filled`. On the mixed-provenance
    path (`open_orders` concatenates raw-wrapper dicts with ccxt dicts when the raw
    endpoint is down) an order shaped amount=991, filled=495, remaining=None reported
    991 instead of the outstanding 496, so the duplicate search missed the partially
    filled order and re-opened the duplicate-reduce-only path.
"""

import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as C           # noqa: E402
import exchange as ex        # noqa: E402
from db import TradeDB       # noqa: E402
from engine import Engine    # noqa: E402
from state import State      # noqa: E402
from test_dry_run_scenarios import CFG, SCHEMA, SYM, FakeKraken  # noqa: E402

POSITION = 10.0
SL_PRICE = 90.0
ENTRY = 100.0


def _db(tmp_path):
    p = tmp_path / "t.db"
    con = sqlite3.connect(p)
    con.executescript(SCHEMA)
    con.commit()
    con.close()
    return TradeDB(p)


def _rig(tmp_path, client):
    state = State(tmp_path / "state.json")
    eng = Engine(CFG, client, state, _db(tmp_path), lambda m: None, dry_run=False)
    return eng, state


class ZeroStopVenue(FakeKraken):
    """The venue states an OPEN, reduce-only stop at the right trigger with 0 outstanding."""

    def __init__(self, stop):
        super().__init__()
        self.orders = [stop]


class NoSizeVenue(FakeKraken):
    """The venue answers with an order carrying NO size field anywhere at all."""

    def __init__(self, stop):
        super().__init__()
        self.orders = [stop]


def _zero_stop(order_id="zero-sl"):
    return {"id": order_id, "type": "stop", "side": "sell", "amount": 0.0, "remaining": 0.0,
            "stopPrice": SL_PRICE, "reduceOnly": True, "status": "open", "info": {}}


def _pos(order_id):
    return {"direction": "long", "entry_price": ENTRY, "sl_price": SL_PRICE,
            "sl_order_id": order_id}


# ── must-fix 1: a zero-outstanding stop is not protection ────────────────────

def test_a_zero_outstanding_stop_is_not_confirmed_as_protection():
    """`remaining=0` on an open stop: nothing can fill, so nothing is protected."""
    venue = ZeroStopVenue(_zero_stop())
    ok, why = ex.stop_is_confirmed(venue, SYM, "zero-sl", side="sell", min_amount=POSITION,
                                   require_breakeven=False, entry=ENTRY, trigger=SL_PRICE)
    assert not ok, f"a zero-outstanding stop was accepted as protection ({why!r})"
    assert "unreadable" not in why, (
        "a zero the venue STATES is a readable size and must take the size-specific "
        f"reason, not the unreadable path: {why!r}")
    assert "size 0" in why and "required 10" in why, why


def test_ensure_sl_replaces_a_zero_outstanding_stop(tmp_path):
    """A stop the venue states as 0 must be cancelled and replaced, not left standing."""
    eng, _ = _rig(tmp_path, ZeroStopVenue(_zero_stop()))
    out = eng._ensure_sl(_pos("zero-sl"), POSITION)
    assert out == "replaced", (
        "a zero-outstanding stop must be cancelled and re-placed as a valid one; "
        f"got {out!r} — the non-replacement 'unknown' path is only for unreadable sizes")


def test_a_size_that_cannot_be_read_at_all_still_routes_to_the_non_replacement_path(tmp_path):
    """The distinct `unreadable` wording is load-bearing: engine.py keys on it.

    engine.py:1494-1498 returns "unknown" (leave it alone, do not cancel and re-place) for a
    reason containing "unreadable". Pinned here because must-fix 1 touches the neighbouring
    branch, and a zero that the venue STATES must not be confused with a size that failed to
    map.
    """
    nosize = {"id": "nosize-sl", "type": "stop", "side": "sell", "stopPrice": SL_PRICE,
              "reduceOnly": True, "status": "open", "info": {}}
    venue = NoSizeVenue(nosize)
    ok, why = ex.stop_is_confirmed(venue, SYM, "nosize-sl", side="sell", min_amount=POSITION,
                                   require_breakeven=False, entry=ENTRY, trigger=SL_PRICE)
    assert not ok and "unreadable" in why, why
    eng, _ = _rig(tmp_path, NoSizeVenue(dict(nosize)))
    assert eng._ensure_sl(_pos("nosize-sl"), POSITION) == "unknown", \
        "an unreadable size must not become a cancel-and-replace"


# ── must-fix 2: the ccxt fallback must honour the unified top-level `filled` ──

class RawEndpointDown:
    """The raw open-orders endpoint fails, so `open_orders` uses the ccxt listing.

    This is the mixed-provenance path the review named: `open_orders` concatenates raw
    wrapper dicts and ccxt dicts into one list once the raw endpoint is unavailable.
    """

    options = {"dryRun": True}

    def __init__(self, orders):
        self._orders = orders

    def privateGetOpenorders(self):
        raise RuntimeError("raw open-orders endpoint down")

    def fetch_open_orders(self, symbol=None, params=None):
        if params and params.get("stop"):
            raise RuntimeError("authenticationError")   # the venue's real quirk for this key
        return [dict(o) for o in self._orders]


def test_a_ccxt_partial_fill_reports_the_outstanding_size_and_the_search_sees_it():
    """amount=991, filled=495, remaining=None -> 496 outstanding, never the 991 total."""
    tp_id = "tp-partial"
    total, done = 991.0, 495.0
    outstanding = total - done
    venue = RawEndpointDown([
        {"id": tp_id, "type": "limit", "side": "buy", "amount": total, "filled": done,
         "remaining": None, "price": 1.2648, "status": "open", "reduceOnly": True, "info": {}},
    ])

    o = ex.open_orders(venue, SYM)[0]
    assert ex.order_amount(o) == pytest.approx(outstanding), (
        f"order_amount read {ex.order_amount(o)} for a partially filled ccxt order that has "
        f"{outstanding} outstanding")

    assert ex.find_matching_order(venue, SYM, side="buy", amount=outstanding, price=1.2648,
                                  kind="limit") is not None, \
        "the outstanding reduce-only order was invisible to the duplicate search"
    assert ex.find_matching_order(venue, SYM, side="buy", amount=total, price=1.2648,
                                  kind="limit") is None, \
        "the original TOTAL was reported as outstanding protection"


def test_the_unified_filled_wins_over_a_stale_info_filled_size():
    """ccxt's top-level `filled` is preferred when both exist, and never goes below 0."""
    o = {"id": "x", "amount": 991.0, "filled": 495, "remaining": None,
         "info": {"filledSize": 0, "quantity": 991}}
    assert ex.order_amount(o) == pytest.approx(496.0)
    over = {"id": "y", "amount": 100.0, "filled": 150, "remaining": None, "info": {}}
    assert ex.order_amount(over) == pytest.approx(0.0), \
        "a fill larger than the total must clamp at 0, never go negative"
    info_only = {"id": "z", "amount": 991.0, "remaining": None,
                 "info": {"filledSize": 495}}
    assert ex.order_amount(info_only) == pytest.approx(496.0), \
        "info.filledSize must still be honoured when the unified field is absent"
