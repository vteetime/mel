"""Guards added after Paul's review of the manual-close tooling (2026-09-20).

Two of his findings were about the OPERATOR TOOLS rather than the bot:
  * `manual_close_operator` swept the whole book with `cancel_all_for_symbol`, which would cancel
    an order a human placed on the same account. Ownership is a recorded id, never a shape.
  * `record_manual_close` called `Engine._finish` directly, skipping the protection retirement the
    tick path performs — fine when the operator tool had already swept its own ids, dangerous
    standalone (a venue liquidation or a UI close leaves the reduce-only SL/TP resting, and a
    survivor can close the NEXT position).

These tests exercise the split/retirement logic directly; the tools' own `main()` paths need a
venue and are covered by the dry runs recorded in the CHANGELOG.
"""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_dry_run_scenarios import FakeKraken, SYM  # noqa: E402


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rmc = _load("record_manual_close")
mco = _load("manual_close_operator")

RESTING = [{"id": "ours-1"}, {"id": "human-7"}, {"id": "ours-2"}]


def test_owns_splits_by_recorded_id_only():
    mine, theirs = rmc._ours(RESTING, {"ours-1", "ours-2"})
    assert mine == ["ours-1", "ours-2"]
    assert theirs == ["human-7"], "an id we did not record is never ours to cancel"


def test_both_tools_share_the_same_ownership_rule():
    a = mco._ours(RESTING, {"ours-1"})
    b = rmc._ours(RESTING, {"ours-1"})
    assert a == b == (["ours-1"], ["human-7", "ours-2"])


def test_retire_our_protection_is_clean_when_nothing_rests():
    client = FakeKraken()
    ok, why = rmc._retire_our_protection(client, SYM, {"sl_order_id": "s1", "tp_order_id": "t1"})
    assert ok is True, why


def test_retire_our_protection_refuses_while_a_foreign_order_rests():
    """A human's order on the same account must block the record, not get cancelled."""
    client = FakeKraken()
    client.orders = [{"id": "human-7", "status": "open", "reduceOnly": True}]
    ok, why = rmc._retire_our_protection(client, SYM, {"sl_order_id": "s1", "tp_order_id": "t1"})
    assert ok is False
    assert "human-7" in why and "FOREIGN" in why
    assert not any(str(o.get("id")) == "human-7" for o in client.placed), "must not cancel it"


def test_retire_our_protection_cancels_only_its_own_resting_order():
    client = FakeKraken()
    client.orders = [{"id": "s1", "status": "open", "reduceOnly": True},
                     {"id": "human-7", "status": "open", "reduceOnly": True}]
    ok, why = rmc._retire_our_protection(client, SYM, {"sl_order_id": "s1", "tp_order_id": ""})
    assert ok is False, "the foreign order still blocks the record"
    assert "human-7" in why
    cancelled = [str(c) for c in getattr(client, "cancelled", [])]
    assert "s1" in cancelled, "our own resting order must be cancelled"
    assert "human-7" not in cancelled, "a foreign order is never cancelled"


# ── the extracted evidence rules (Paul round 2: the over-coverage guard was unreachable) ──

def _f(order, amount, price, fee=0.1):
    return {"order": order, "amount": amount, "price": price, "fee": {"cost": fee}}


def test_validate_evidence_accepts_one_covering_order():
    ok, oid, why = rmc._validate_evidence([_f("a", 8, 2600), _f("a", 5, 2600)], 13.0)
    assert ok is True and oid == "a", why


def test_validate_evidence_rejects_partial_coverage():
    ok, _, why = rmc._validate_evidence([_f("a", 12.9, 2600)], 13.0)
    assert ok is False and "partial" in why


def test_validate_evidence_rejects_oversized_coverage():
    """Attributing an oversized sell to this row would fabricate its exit."""
    ok, _, why = rmc._validate_evidence([_f("a", 14, 2600)], 13.0)
    assert ok is False and "MORE" in why


def test_validate_evidence_rejects_a_multi_order_close():
    ok, _, why = rmc._validate_evidence([_f("a", 6, 2600), _f("b", 7, 2600)], 13.0)
    assert ok is False and "order ids" in why


def test_validate_evidence_rejects_no_fills():
    ok, _, why = rmc._validate_evidence([], 13.0)
    assert ok is False and "no closing fills" in why
