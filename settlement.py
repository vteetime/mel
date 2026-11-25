"""Exchange settlement — reconcile a close against Kraken BEFORE writing the row.

The rule (Tee's): when closing a position, do not trust our own arithmetic. Ask the
exchange what actually happened and record THAT.

What Kraken can be asked, in descending order of authority:

  1. **Account log** (`historyGetAccountLog`) — one `futures trade` entry per fill with
     the fee the venue actually charged, plus `funding rate change` entries. This is the
     only place Kraken states real costs; a schedule-based guess is not a substitute.
  2. **Fills** (`fetch_my_trades`) — real prices and sizes per order. The exit price is
     the VWAP of the exit orders' fills, never the last chunk's price or a ticker.
  3. **Our own arithmetic** — last resort, and the row is flagged
     `settlement_source='provisional'` so downstream reporting knows how well the number
     was verified.

Kraken differences from Jack's KuCoin settlement, and why the code looks different:
  * KuCoin reports a POSITION-level `realizedPnl` and a per-fill `fillPnl`. Kraken
    publishes neither through ccxt, so both of those sources are gone — not skipped.
  * CONTRACT SIZE. Kraken ETH/USD:USD has contractSize 1.0: the order amount IS ETH.
    KuCoin's ETHUSDTM is 0.01, where omitting contract_size overstates every money
    figure 100x. The multiplication is kept in one place (`base_qty`) because that
    factor is the single most expensive thing to get wrong on either venue.

Never raises into the trading path: any failure degrades to the next source. A close
must always be recordable, but it must be HONEST about how well it was verified.
"""

from __future__ import annotations

import logging

log = logging.getLogger("mel.settlement")

FEE_PROVISIONAL_RATE = 0.0005      # 0.05% taker per side, only when the venue is silent


def _f(v, default=0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def base_qty(contracts: float, contract_size: float) -> float:
    """Contracts -> base-asset quantity. THE conversion; never inline it.

    Kraken ETH: contracts ARE ETH (size 1.0). KuCoin ETHUSDTM: 1 contract = 0.01 ETH.
    """
    return max(0.0, _f(contracts)) * (_f(contract_size, 1.0) or 1.0)


def fills_for_order(client, symbol: str, order_id: str, limit: int = 500,
                    since_ms: int | None = None) -> list[dict]:
    """All fills belonging to one order id, normalised.

    Matches on `info.orderId` (Kraken's own id) and then on ccxt's `order` field —
    neither is guaranteed on every response, so both are tried before giving up. A
    missed match here silently costs the fee and the real exit price.
    """
    if not order_id:
        return []
    try:
        trades = client.fetch_my_trades(symbol, limit=limit, **({"since": since_ms}
                                                               if since_ms else {})) or []
    except TypeError:
        try:                                    # venue ignores `since`
            trades = client.fetch_my_trades(symbol, limit=limit) or []
        except Exception as exc:
            log.warning("fetch_my_trades failed: %s", exc)
            return []
    except Exception as exc:
        log.warning("fetch_my_trades failed: %s", exc)
        return []
    out = []
    for t in trades:
        info = t.get("info") or {}
        oid = str(info.get("orderId") or info.get("order_id") or t.get("order") or "")
        if oid != str(order_id):
            continue
        amt, px = _f(t.get("amount")), _f(t.get("price"))
        if amt <= 0 or px <= 0:
            continue
        fee_obj = t.get("fee")
        fee = _f(fee_obj.get("cost")) if isinstance(fee_obj, dict) else _f(fee_obj)
        if fee == 0.0:
            fee = _f(info.get("fee"))
        out.append({"amount": amt, "price": px, "fee": abs(fee),
                    "order_id": oid,
                    "ts": t.get("timestamp") or info.get("fillTime")})
    return out


def vwap_fee(fills: list[dict]) -> tuple[float, float, float]:
    """(vwap, total_fee, total_amount) from normalised fills."""
    num = den = fee = 0.0
    for f in fills:
        num += f["amount"] * f["price"]
        den += f["amount"]
        fee += f["fee"]
    return ((num / den) if den > 0 else 0.0, fee, den)


def ledger_costs(client, symbol: str, since_ms: int | None) -> tuple[float, float, float | None, bool]:
    """(fee, funding, realized_pnl, verified) for this symbol from the venue's own account log.

    `realized_pnl` is Kraken's own per-fill realized P&L, summed over the window, GROSS of fee
    (the fee is a separate field on the same `futures trade` row). It is None when no row
    carried the field, so the caller can tell "no independent figure" from "zero P&L".

    `verified` is False whenever the ledger could not be read or contained no matching
    `futures trade` row — the caller must then record the fee as ESTIMATED rather than
    present a guess as a fact.
    """
    market_id = ""
    try:
        market_id = str((client.market(symbol).get("id") or "")).lower()
    except Exception:
        pass
    fee = funding = realized_pnl = 0.0
    got = False
    rpnl_seen = False
    try:
        raw = client.historyGetAccountLog({"since": since_ms} if since_ms else {})
        logs = raw.get("logs", []) if isinstance(raw, dict) else (raw or [])
    except Exception as exc:
        log.warning("account log unavailable: %s", exc)
        return 0.0, 0.0, None, False
    for row in logs or []:
        try:
            contract = str(row.get("contract") or "").lower()
            if market_id and contract and contract != market_id:
                continue
            info = str(row.get("info") or "")
            if info == "futures trade":
                f = row.get("fee")
                if f is not None:
                    fee += float(f)
                    got = True
                rp = row.get("realized_pnl")
                if rp is not None:
                    realized_pnl += float(rp)
                    rpnl_seen = True
            elif info == "funding rate change":
                rf = row.get("realized_funding")
                if rf is not None:
                    funding += float(rf)
        except (TypeError, ValueError, AttributeError):
            continue
    return fee, funding, (realized_pnl if rpnl_seen else None), got


def settled_close(client, symbol: str, *, close_order_id: str, entry_order_id: str = "",
                  extra_order_ids: list[str] | None = None,
                  entry_fills: list[dict] | None = None,
                  entry_price: float = 0.0, contracts: float = 0.0,
                  contract_size: float = 1.0,
                  fallback_price: float = 0.0,
                  opened_ms: int | None = None,
                  net_pnl_of=lambda pnl: pnl) -> dict:
    """Resolve the authoritative close data. Returns a dict, never raises.

    Keys: exit_price, fee_usdt, pnl_net, settlement_source, exchange_settlement_id,
          fee_estimated, vwap_from_fills, funding_usdt, funding_verified, _gross_abs.
    """
    res = {"exit_price": _f(fallback_price), "fee_usdt": 0.0, "pnl_net": None,
           "settlement_source": "provisional", "exchange_settlement_id": "",
           "fee_estimated": True, "vwap_from_fills": False,
           "funding_usdt": 0.0, "funding_verified": False,
           # PROVENANCE FLAGS. `settlement_source` alone would let a configured LEVEL
           # masquerade as a verified fill, so price and quantity are tracked separately
           # from the fee. Both must be true for a row to claim venue verification.
           "price_verified": False, "fills_complete": False,
           "fee_from_ledger": False, "ledger_disagrees": False,
           "exit_order_ids": [],
           # Kraken's account-log realized P&L (gross of fee), when the ledger states it —
           # the only independent P&L figure the venue publishes through ccxt.
           "pnl_realized": None}

    # ── 1. exit fills: every exit order, not just the final close ──
    # A Mel exit is a single order, but the SL and the TP can both have fired, and an
    # exit can be split between them. Aggregating is what keeps the exit price honest.
    exit_fills = fills_for_order(client, symbol, close_order_id)
    for _oid in (extra_order_ids or []):
        if _oid and str(_oid) != str(close_order_id):
            exit_fills = exit_fills + fills_for_order(client, symbol, _oid)
    exit_vwap, exit_fee, exit_amt = vwap_fee(exit_fills)
    if exit_vwap > 0:
        res["exit_price"] = exit_vwap
        res["vwap_from_fills"] = True
        res["price_verified"] = True
    elif res["exit_price"] <= 0:
        log.warning("no exit fill price and no fallback — caller must not record")
        return res
    res["exchange_settlement_id"] = str(close_order_id or "")

    # QUANTITY COMPLETENESS. `fetch_my_trades` returns a bounded history, so a long-lived
    # or busy account can yield only PART of an exit. Pricing the whole trade from a
    # truncated set silently mis-states the exit, so the row is downgraded — and because a
    # PARTIAL figure is not recordable at all (the engine refuses it), the wider re-query
    # below runs before any such verdict.
    def _covers(amt: float) -> bool:
        return not contracts or bool(amt) and amt + 1e-9 >= _f(contracts) * 0.999

    if contracts and exit_amt and not _covers(exit_amt):
        # ASK AGAIN, WIDER: the first read is a small tail by default and a busy account
        # can push this trade's fills out of it.
        wider: list[dict] = []
        for _oid in [close_order_id, *(extra_order_ids or [])]:
            if _oid:
                wider += fills_for_order(client, symbol, _oid, limit=1000,
                                         since_ms=opened_ms)
        w_vwap, w_fee, w_amt = vwap_fee(wider)
        if w_amt > exit_amt:
            exit_fills, exit_vwap, exit_fee, exit_amt = wider, w_vwap, w_fee, w_amt
            if w_vwap > 0:
                res["exit_price"] = w_vwap

    res["fills_complete"] = _covers(exit_amt)
    if not res["fills_complete"]:
        log.warning("exit fills cover only %s of %s contracts — the exit price is partial",
                    exit_amt, contracts)

    # WHICH ORDER ACTUALLY CLOSED THE POSITION. Every exit fill is tagged with the
    # order id that produced it, so the question "did the TP or the stop take this
    # trade out?" is answerable from evidence the caller already paid for — as long
    # as the answer is not thrown away here. (The order BOOK cannot answer it on this
    # venue: a filled order is dropped, and fetch_order answers ordernotfound.)
    res["exit_order_ids"] = sorted({str(f.get("order_id"))
                                    for f in exit_fills if f.get("order_id")})

    # ── 2. entry fills: the real entry VWAP and the entry-side fee ──
    entry_fee = 0.0
    entry_vwap = _f(entry_price)
    # Keep whatever entry evidence we actually resolve: the ledger window below must reach back
    # past the ENTRY fill, and `entry_fills` (the caller's) is usually absent — `_finish` does not
    # pass it. Without this the window still started at `opened_ms` and the fix was inert.
    entry_evidence: list[dict] = list(entry_fills or [])
    if entry_fills:
        entry_vwap, entry_fee, _ea = vwap_fee(entry_fills)
    elif entry_order_id:
        ef = fills_for_order(client, symbol, entry_order_id)
        if ef:
            entry_vwap, entry_fee, _ea = vwap_fee(ef)
            entry_evidence = list(ef)

    # ── 3. the venue's ledger: the fees and funding it actually charged ──
    # The window is backed off from the EARLIEST evidence we hold, not from `opened_ms`. That
    # stamp is written only once the entry is CONFIRMED (engine sets it after the order fills),
    # so a window starting there excludes the entry leg's own ledger row — the ledger then
    # returns the exit fee alone, and preferring it silently dropped the entry fee out of every
    # P&L. Found on trade #355 (2026-09-20): pnl_net overstated by 0.1827, ~4.8% of that trade,
    # and systematically on every Mel close.
    stamps = [f.get("ts") for f in (list(exit_fills) + entry_evidence) if f.get("ts")]
    floor_ms = (int(min(stamps)) - 60_000) if stamps else None
    if floor_ms is None and entry_evidence:
        # Degradation must be OBSERVABLE: with no timestamps the window falls back to
        # `opened_ms`, which is exactly the short window that dropped the entry fee — the
        # fills-vs-ledger backstop below still applies, but the operator should be able to see
        # that the primary fix was bypassed (Paul, round 2).
        log.warning("no fill timestamps in the evidence — the ledger window falls back to "
                    "opened_ms, so the entry leg's fee row may fall outside it")
    starts = [int(x) for x in (opened_ms, floor_ms) if x]
    since = min(starts) if starts else None
    ledger_fee, funding, ledger_rpnl, ledger_ok = ledger_costs(client, symbol, since)
    fills_fee = exit_fee + entry_fee
    if ledger_ok:
        # A fills fee is only comparable when BOTH legs carry one. `fills_for_order` represents
        # a missing fee field as 0, and without this check the ledger could lose a disagreement
        # to a figure that is simply missing a leg — trusting an undercount is how costs get
        # dropped (Paul, round 2).
        legs_complete = entry_fee > 0 and exit_fee > 0
        if not legs_complete:
            log.info("fills are fee-incomplete (entry %.6f, exit %.6f) — the ledger stays "
                     "authoritative and no disagreement is claimed",
                     entry_fee, exit_fee)
            res["fee_usdt"] = ledger_fee
            res["fee_estimated"] = False
            res["fee_from_ledger"] = True
        elif abs(ledger_fee - fills_fee) > max(0.05, 0.2 * fills_fee):
            res["ledger_disagrees"] = True
            log.warning("ledger fee %.6f vs fills fee %.6f for the same window",
                        ledger_fee, fills_fee)
            # A DISAGREEING LEDGER LOSES. Our fills are the only source that is both stated
            # by the venue per fill AND complete for THIS trade; the ledger is summed by
            # contract + window, so it can miss a leg (exactly the case above, when the
            # window was short) or absorb somebody else's trade. Recording the partial
            # figure understated costs and overstated every P&L built from the row.
            res["fee_usdt"] = fills_fee
            res["fee_estimated"] = False
            res["fee_from_ledger"] = False
        else:
            res["fee_usdt"] = ledger_fee
            res["fee_estimated"] = False
            res["fee_from_ledger"] = True
    res["funding_usdt"] = funding
    res["funding_verified"] = ledger_ok
    if ledger_rpnl is not None:
        # The account log's own realized P&L, gross of fee. Exposed so the engine can
        # cross-check its computed P&L against an independent figure (Paul r3).
        res["pnl_realized"] = ledger_rpnl

    # THE SOURCE NAMES WHAT WAS ACTUALLY VERIFIED. A level used as the exit price is not
    # venue-verified, so it must not be labelled `exchange_ledger` just because the FEE
    # came from the ledger.
    if res["price_verified"]:
        res["settlement_source"] = "exchange_ledger" if res["fee_from_ledger"] else "fills"
        if not res["fills_complete"]:
            res["settlement_source"] = "fills_partial"
    else:
        res["settlement_source"] = "level_fallback"

    size = exit_amt or _f(contracts)
    cs = _f(contract_size, 1.0) or 1.0
    if entry_vwap > 0:
        entry_price = entry_vwap

    # ── 4. built from venue-reported parts: real VWAPs and real fees ──
    if entry_price > 0 and size > 0:
        fee = res["fee_usdt"] or fills_fee
        if fee <= 0:
            fee = round((entry_price + res["exit_price"]) * base_qty(size, cs)
                        * FEE_PROVISIONAL_RATE, 6)
            res["fee_estimated"] = True
            res["settlement_source"] = ("fills" if res["vwap_from_fills"]
                                        else "provisional")
        elif not ledger_ok and fills_fee > 0:
            # The ledger could not be read, so the fee is the sum of OUR OWN fills' venue-stated
            # fees. That is measured, not estimated — labelling a real fee "estimated" is the same
            # provenance-honesty defect as calling an estimate verified, in the other direction.
            res["fee_estimated"] = False
        res["fee_usdt"] = fee
        res["contracts"] = size
        res["contract_size"] = cs
        res["entry_price"] = entry_price
        res["base_qty"] = base_qty(size, cs)
        # Signed by the caller with the trade direction.
        res["_gross_abs"] = (res["exit_price"] - entry_price) * res["base_qty"]
        res["pnl_net"] = None
    return res
