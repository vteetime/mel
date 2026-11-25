"""Kraken Futures adapter (ccxt `krakenfutures`) for Mel.

Everything here is a thin, explicit wrapper around ccxt. What it establishes, and why
each piece exists, is in the comments — the mechanics were learned on Ken (the previous
Kraken bot) and are not free to change:

  * POSITION MODE. Kraken futures is ONE-WAY: one position per symbol, no `positionSide`,
    and no `hedged` / `marginMode` order params. Jack's adapter sent both because KuCoin
    demanded them; sending them here would be meaningless at best. `reduceOnly` is what
    keeps an exit from opening the opposite side, and it is the ONLY thing that does.

  * CONTRACT SIZE. Kraken ETH/USD:USD has contractSize 1.0, i.e. the order amount IS the
    ETH amount (0.138 = 0.138 ETH) and it is quoted to 3 decimals. The amount is
    therefore NOT a whole number. Every money figure goes through
    `contracts * contract_size`; `market_spec()` publishes both so no caller can guess.

  * PRICE PRECISION. On Kraken, ccxt's `precision['price']` is a TICK SIZE (0.1 on ETH),
    not a count of decimals. `int(precision)` on it — the naive read — rounds a $2,650
    price to $2650 at best and to 2 at worst. Always go through `price_to_precision()`.

  * STOP ORDERS. Kraken takes a protective stop as order type `stop` with a flat
    `stopPrice` and `reduceOnly: True`. `place_stop()` is the only way this module
    submits one, and every placement is re-read and validated by `stop_is_confirmed()`
    before it is trusted.

  * IDEMPOTENCY. ccxt does not expose Kraken's `clOrdId`, so a retry after a timeout
    cannot be de-duplicated by client id the way KuCoin's `clientOid` allowed. The
    substitute is `find_matching_order()`: before re-placing a protective order after a
    failure, look at what is actually resting. A duplicate protective stop is not
    harmless — it is a second order that can close a LATER position.

Nothing here places an order unless the caller has already passed guards and sizing.

DRY-RUN IS ENFORCED HERE, NOT BY ccxt. Jack's adapter set `client.options["dryRun"]` and
trusted it; **ccxt implements no such option** (grep the package: zero references), so a
dry-run client is still a fully capable trading client. Every order-submitting function in
this module therefore refuses when the client is flagged dry-run. The barrier is at the
adapter so a future code path cannot forget it.
"""

from __future__ import annotations

import logging
import math
import time

import ccxt

log = logging.getLogger("mel.exchange")

RETRYABLE = (ccxt.NetworkError, ccxt.RequestTimeout, ccxt.ExchangeNotAvailable,
             ccxt.DDoSProtection, ccxt.RateLimitExceeded)

# Kraken futures has no position-mode switch to configure. `configure_orders` exists so
# the leverage actually sent to the venue is explicit, storable and auditable.
ORDER_PARAMS: dict = {"reduceOnly": False}


class ExchangeError(RuntimeError):
    pass


class VenueUnavailable(ExchangeError):
    """The venue is unreachable or erroring (outage, maintenance, 503s)."""


def configure_orders(*, leverage: int) -> dict:
    """Record the leverage every order is submitted under.

    [MONEY] Kept explicit rather than implicit: sizing assumes `cfg.sizing.leverage`, so
    the number the venue is told must be the same one the sizing used. Kraken one-way
    futures has no `hedged` / `marginMode` to send, so those Jack-era params are gone.
    """
    global ORDER_PARAMS
    lev = int(leverage)
    if lev < 1:
        raise ExchangeError(f"refusing leverage {lev}")
    ORDER_PARAMS = {"reduceOnly": False, "leverage": lev}
    log.info("order defaults: %s", ORDER_PARAMS)
    return dict(ORDER_PARAMS)


def is_dry_run(client) -> bool:
    """True when this client was created in dry-run mode (see new_client)."""
    try:
        return bool((client.options or {}).get("dryRun"))
    except Exception:
        return False


def _refuse_if_dry_run(client, what: str) -> None:
    """[MONEY] The last line of defence for 'a dry run places NOTHING'.

    ccxt has no dry-run mode of its own, so setting the option only documents intent. A
    read-only tick that decided to close a position or protect an orphan would otherwise
    send a real order while every operator-facing document says it cannot.
    """
    if is_dry_run(client):
        raise ExchangeError(
            f"dry-run client: refusing to submit {what}. A dry run places nothing — "
            f"remove the flag deliberately (arm with --live + the LIVE sentinel) to trade.")


def new_client(cfg: dict, keys: dict, dry_run: bool = True):
    ex = ccxt.krakenfutures({
        "apiKey": keys.get("api_key", ""),
        "secret": keys.get("api_secret", ""),
        "enableRateLimit": True,
    })
    if dry_run:
        # A dry-run client must never be able to reach the order endpoints.
        ex.options["dryRun"] = True
    try:
        retry(ex.load_markets)
    except ExchangeError as exc:
        raise VenueUnavailable(
            f"Kraken Futures unreachable (load_markets failed): {exc}") from exc
    return ex


class AmbiguousSubmission(ExchangeError):
    """A mutating call whose outcome is UNKNOWN: it may or may not have landed.

    Timeouts and network errors after a submit are not "it failed" — Kraken may have
    accepted the order and simply not answered. Re-sending is how one entry becomes two, or
    one stop becomes a pair (and a spare stop can close a LATER position). Callers must
    resolve the ambiguity by reading state, or flatten and alert.
    """


def retry(fn, attempts: int = 4, base_delay: float = 1.0):
    """Retry for READS ONLY. Never wrap a mutating call in this — see submit_once()."""
    last = None
    for i in range(attempts):
        try:
            return fn()
        except RETRYABLE as exc:
            last = exc
            log.warning("retryable exchange error (%d/%d): %s", i + 1, attempts, exc)
            time.sleep(base_delay * (2 ** i))
    raise ExchangeError(f"exchange call failed after {attempts} attempts: {last}")


def submit_once(fn, what: str):
    """Submit a MUTATION exactly once. Ambiguous failures raise AmbiguousSubmission.

    Why this is not a retry: Kraken exposes no client order id through ccxt, so a
    timeout-after-accept is indistinguishable from a failure, and a re-send duplicates a
    real order. The caller resolves it by reading positions/orders (see the engine), or
    fails closed by flattening.
    """
    try:
        return fn()
    except RETRYABLE as exc:
        raise AmbiguousSubmission(
            f"{what}: venue did not confirm the outcome ({exc}). It may have been accepted. "
            f"Resolve by reading positions/orders before submitting anything again.") from exc


# ── market facts ─────────────────────────────────────────────────────────────

def market_spec(client, symbol: str) -> dict:
    """Contract size, tick size, amount step and the venue's minimum, if it states one.

    These four numbers are the difference between a correct order and a 100x error, so
    they come from the market rather than from config. `min_amount` falls back to None
    when the venue does not publish a minimum (Kraken does not, via ccxt) — the caller
    then uses the configured `sizing.min_contracts` instead of inventing a number here.
    """
    m = client.market(symbol)
    info = m.get("info") or {}
    prec = (m.get("precision") or {})
    limits = ((m.get("limits") or {}).get("amount") or {})
    tick = prec.get("price")
    if not isinstance(tick, (int, float)) or tick <= 0:
        tick = info.get("tickSize")
    step = prec.get("amount")
    if not isinstance(step, (int, float)) or step <= 0:
        step = 0.001
    return {
        # One contract = contract_size of the base asset. Kraken ETH: 1.0 (amount IS ETH).
        "contract_size": float(m.get("contractSize") or 1.0),
        "tick_size": float(tick) if tick else 0.1,
        "amount_step": float(step),
        "min_amount": (float(limits["min"]) if limits.get("min") else None),
        "market_type": str(m.get("type") or ""),
        "market_id": str(m.get("id") or ""),
    }


def round_price(client, symbol: str, price: float) -> float:
    """Round to the venue's tick. Never `int(precision)` — on Kraken it is a tick size."""
    try:
        return float(client.price_to_precision(symbol, price))
    except Exception as exc:
        log.warning("price_to_precision failed (%s) — falling back to a rounded price", exc)
        try:
            tick = market_spec(client, symbol)["tick_size"]
            return round(round(float(price) / tick) * tick, 10)
        except Exception:
            return round(float(price), 2)


def round_amount(client, symbol: str, amount: float, mode: str = "down") -> float:
    """Round an order amount to the market's step — Kraken ETH is 3 decimals, NOT whole.

    Flooring to a whole contract here is the silent-no-trade bug: Jack's sizing floored
    to int because KuCoin contracts are indivisible, and on Kraken that turns a
    correct 0.138 ETH order into 0.
    """
    amount = float(amount)
    try:
        prec = market_spec(client, symbol)["amount_step"]
    except Exception:
        prec = 0.001
    if not prec or prec <= 0:
        return amount
    n = amount / prec
    n = int(n + 1e-9) if mode == "down" else round(n)
    return float(round(n * prec, 10))


def fetch_price(client, symbol: str) -> float:
    t = retry(lambda: client.fetch_ticker(symbol))
    px = float(t.get("last") or (t.get("info") or {}).get("price") or 0.0)
    if px <= 0:
        raise ExchangeError(f"no usable price for {symbol}")
    return px


def fetch_ohlcv(client, symbol: str, timeframe: str, limit: int) -> list:
    """Oldest-first OHLCV rows."""
    rows = retry(lambda: client.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit))
    return [list(r) for r in rows or []]


def free_quote(client, quote: str = "USDT") -> float:
    """Free collateral available to size a NEW position.

    Kraken futures is a MULTI-COLLATERAL wallet: USD and USDT can both sit in it at once
    and the venue converts as margin is required. `config.available_funds` sums the free
    rows rather than returning the first one it finds.
    """
    import config as C
    bal = retry(lambda: client.fetch_balance())
    return C.available_funds(bal, quote)


def balance_report(client) -> dict:
    """Balance snapshot for the reconciliation report. Read-only.

    Sums across Kraken's multi-collateral currencies (USD + USDT) so the operator report
    and the sizing basis agree — see config.available_funds for why one row is not the
    whole wallet.
    """
    import config as C
    bal = retry(lambda: client.fetch_balance())
    free = bal.get("free") or {}
    total = bal.get("total") or {}

    def pick(src):
        s = 0.0
        found = False
        for key in C.COLLATERAL_CURRENCIES:
            v = src.get(key)
            if v is None:
                continue
            try:
                s += float(v)
            except (TypeError, ValueError):
                continue
            found = True
        return s if found else 0.0

    keys = tuple(k for k in C.COLLATERAL_CURRENCIES if k in free or k in total)
    return {"free": pick(free), "total": pick(total),
            "currency": "+".join(keys) if keys else "USD"}


def set_leverage(client, leverage: int, symbol: str) -> bool:
    """Best-effort: Kraken applies the account's leverage tier. Failure is not fatal.

    NOTE: best-effort is not good enough on its own — sizing, margin_fraction and every equity
    cap assume this number, so the caller should verify with `venue_leverage()`.
    """
    try:
        _refuse_if_dry_run(client, "set leverage")
        client.set_leverage(int(leverage), symbol)
        return True
    except Exception as exc:
        log.info("set_leverage(%s, %s) not applied: %s", leverage, symbol, exc)
        return False


def venue_leverage(client, symbol: str) -> float | None:
    """The leverage preference Kraken reports for this symbol, or None if it lists none.

    Read-only (`privateGetLeveragepreferences`). None means "not explicitly set", not "1x": the
    venue's default applies and ccxt exposes no way to read it, so the caller treats None as
    UNVERIFIED rather than as a match.
    """
    try:
        r = client.privateGetLeveragepreferences() or {}
    except Exception as exc:
        log.warning("cannot read the venue's leverage preferences: %s", exc)
        return None
    for pref in (r.get("leveragePreferences") or []):
        if str(pref.get("symbol") or "").upper() in _symbol_aliases(symbol):
            try:
                return float(pref.get("maxLeverage"))
            except (TypeError, ValueError):
                return None
    return None


def ensure_leverage(client, symbol: str, want: int) -> tuple[bool, str]:
    """Set the leverage and VERIFY what the venue will apply. -> (ok, detail)

    `ok=False` means a proven mismatch (the venue reports a different number): sizing, the
    margin fraction and the equity caps are all built on `want`, so trading on a different
    number is trading a different risk. An unreadable preference list is reported as
    unverified-but-allowed, because the set call itself succeeded.
    """
    set_leverage(client, int(want), symbol)
    got = venue_leverage(client, symbol)
    if got is None:
        return True, "unverified (the venue lists no explicit preference for this symbol)"
    if abs(got - float(want)) > 1e-9:
        return False, f"the venue will apply {got:g}x, not the configured {int(want)}x"
    return True, f"{got:g}x confirmed"


def _symbol_aliases(symbol: str) -> set[str]:
    """Kraken reports futures symbols as PF_ETHUSD; ccxt uses ETH/USD:USD."""
    s = str(symbol or "")
    base = s.split("/")[0].replace("-", "").upper()
    return {s.upper(), f"PF_{base}USD", f"PF_{base}USDT", base}


# ── positions ────────────────────────────────────────────────────────────────

def open_position(client, symbol: str) -> dict:
    """Return {'active', 'contracts', 'side', 'entry_price'} for the exchange position.

    A FAILED read is reported as {'active': False, 'error': ...} rather than raised, and
    every caller must treat an error as "unknown", never as "flat" — reading an outage
    as flat is how a live position gets reconciled as closed, or doubled.
    """
    try:
        poss = client.fetch_positions([symbol])
    except Exception as exc:
        log.error("fetch_positions failed: %s", exc)
        return {"active": False, "contracts": 0.0, "side": "", "entry_price": 0.0,
                "error": str(exc)}
    for p in poss or []:
        contracts = abs(float(p.get("contracts") or 0.0))
        if contracts <= 0:
            continue
        return {"active": True, "contracts": contracts,
                "side": (p.get("side") or "").lower(),
                "entry_price": float(p.get("entryPrice") or 0.0),
                "notional": float(p.get("notional") or 0.0),
                "unrealized": float(p.get("unrealizedPnl") or 0.0),
                "leverage": p.get("leverage"), "error": None}
    return {"active": False, "contracts": 0.0, "side": "", "entry_price": 0.0,
            "error": None}


def open_orders(client, symbol: str, include_stops: bool = True,
                strict: bool = False) -> list[dict]:
    """Every resting order for the symbol — limits AND stops.

    Kraken files stop orders separately from limit orders on some endpoints, so a check
    that only reads `fetch_open_orders()` will report "no protection" while a stop is
    resting, and a re-place then adds a second one.

    [MONEY] `strict=True` turns any read failure into an ExchangeError. WITHOUT IT AN
    OUTAGE LOOKS LIKE AN EMPTY BOOK: the tolerant form returns `[]` on error, and callers
    that ask "is anything still resting?" then conclude the book is clean and lift the
    block that protects against a surviving reduce-only order. Every safety decision
    (dangling retry, flat cleanup, post-exit verification) must use strict reads; only
    reporting paths may tolerate a failure.
    """
    failed: list[str] = []
    out: list[dict] = []
    res_stops_unavailable = False
    # KRAKEN'S OWN ENDPOINT FIRST. `/derivatives/api/v3/openorders` is the account's open
    # order list and it answers for this key, while the ccxt trigger-specific query
    # (`params={"stop": True}`, a different request shape) is refused with
    # `authenticationError`. Reading the authoritative list first means the primary view is
    # not silently missing whatever the refused query would have added.
    try:
        raw = client.privateGetOpenorders()
        rows = (raw or {}).get("openOrders") if isinstance(raw, dict) else raw
        for o in rows or []:
            # [MONEY] THE SIZE, FROM THE FIELDS THIS ENDPOINT ACTUALLY SENDS. These rows carry
            # `unfilledSize` / `filledSize` and NO `quantity` (verified live: both resting
            # orders read back as None here while the venue reported `unfilledSize: 991`), so
            # taking the size from `quantity` handed every order back with amount=None and
            # everything downstream then compared the real 991 against 0 — `--sync` printed
            # "UNDER-PROTECTED: stops cover 0 of 991" over a fully protected position,
            # `find_matching_order` missed both resting orders, and the engine could not
            # recognise a TP that had actually landed. `remaining` is the OUTSTANDING size
            # (what the order can still fill) and `amount` the TOTAL, mirroring ccxt's unified
            # pair; `quantity` is still honoured for any venue/shape that sends it.
            unfilled = _size(o.get("unfilledSize"))
            if unfilled is None:
                unfilled = _size(o.get("unfilled"))
            filled = _size(o.get("filledSize"))
            if unfilled is not None or filled is not None:
                total = (unfilled or 0.0) + (filled or 0.0)
            else:
                total = _size(o.get("quantity"))
            out.append({
                "id": str(o.get("order_id") or o.get("orderId") or ""),
                "type": str(o.get("orderType") or "").lower(),
                "side": "buy" if str(o.get("side") or "").lower() in ("buy", "b") else "sell",
                "amount": total, "remaining": unfilled,
                "price": o.get("limitPrice"),
                "stopPrice": o.get("stopPrice"), "status": str(o.get("status") or "open"),
                "reduceOnly": o.get("reduceOnly"),
                "timestamp": o.get("receivedTime") or o.get("lastUpdateTime"),
                "info": o,
            })
    except Exception as exc:
        # Only a failure of EVERY source is a failure of the view: a missing or refused
        # primary endpoint with a working ccxt fallback is a degraded read, not an outage.
        log.info("kraken open-orders endpoint unavailable (%s) — using the ccxt listing",
                 str(exc)[:80])
        try:
            out += list(client.fetch_open_orders(symbol) or [])
        except Exception as exc2:
            log.warning("fetch_open_orders failed too: %s", exc2)
            failed.append(f"orders: {exc2}")
    if include_stops:
        try:
            out += list(client.fetch_open_orders(symbol, params={"stop": True}) or [])
        except Exception as exc:
            # [VENUE QUIRK, not an outage] Kraken rejects the trigger-order query with
            # `authenticationError` for this key. Ken wrapped the same call in a bare
            # `except: pass` for months, so KItt's ancestor never saw stop orders listed
            # either — which is very plausibly where Ken's duplicate protective stops came
            # from (see cleanup_ken_duplicate_stops.py). It is not fatal here because stop
            # ownership is established by ID (see order_gone), not by listing: a query the
            # venue refuses is not evidence that the account is unreadable. The MAIN call
            # failing still raises — that IS an outage.
            log.info("stop-order listing unavailable on this venue/key (%s) — relying on "
                     "per-id verification", str(exc)[:90])
            res_stops_unavailable = True
    if failed and strict:
        raise ExchangeError("cannot read the order book (" + "; ".join(failed) + ")")
    seen, uniq = set(), []
    for o in out:
        oid = str(o.get("id") or "")
        if oid and oid in seen:
            continue
        seen.add(oid)
        uniq.append(o)
    return uniq


def find_matching_order(client, symbol: str, *, side: str, amount: float,
                        trigger: float | None = None, price: float | None = None,
                        kind: str | None = None, tolerance: float = 1e-9) -> dict | None:
    """Look for an already-resting order that matches what we are about to place.

    This is Mel's idempotency check — ccxt gives us no client order id on Kraken, so
    "did my timed-out stop actually land?" is answered by reading the book, not by
    re-submitting and hoping.

    KIND MATTERS. A protective stop and a take-profit can share side AND size (both are
    reduce-only sell orders for the full position), and a stop carries no `price` field
    to compare against — so a search for the TP that ignores `kind` happily matches the
    SL. Reproduced here: the entry tick concluded "a matching TP already rests", adopted
    the STOP as its take-profit, and left the position with no TP at all.
    """
    for o in open_orders(client, symbol):
        if str(o.get("side") or "").lower() != str(side).lower():
            continue
        o_type = str(o.get("type") or "").lower()
        o_trigger = order_trigger(o)
        if kind:
            o_is_stop = o_type in ("stop", "stopmarket", "stop-loss", "stop_loss") \
                or o_trigger > 0
            if kind == "stop" and not o_is_stop:
                continue
            if kind == "limit" and o_is_stop:
                continue
        # [MONEY] THE SAME ACCESSOR AS EVERYTHING ELSE. This compared `o["amount"]` while the
        # coverage sum and both engine comparisons used `order_amount(o)`, so one definition of
        # size did not govern the dedupe: on a Kraken raw row (which carries no `quantity`) it
        # compared 0 against the requested size and matched NOTHING while a full-size order was
        # resting — reproduced live as MATCH_STOP=None and MATCH_LIMIT=None with both orders
        # demonstrably on the book.
        amt = order_amount(o)
        if abs(amt - float(amount)) > max(tolerance, 1e-9):
            continue
        if trigger is not None:
            if o_trigger <= 0:
                continue
            if abs(o_trigger - float(trigger)) > 1e-6:
                continue
        elif o_trigger > 0:
            # A triggered order is not the plain limit we are looking for.
            continue
        o_price = o.get("price")
        if price is not None and o_price is not None and trigger is None:
            try:
                if abs(float(o_price) - float(price)) > 1e-6:
                    continue
            except (TypeError, ValueError):
                continue
        return o
    return None


# ── orders ───────────────────────────────────────────────────────────────────

def place_market(client, symbol: str, side: str, contracts: float,
                 reduce_only: bool = False, tag: str = "entry") -> dict:
    _refuse_if_dry_run(client, f"a market {tag} order")
    amount = round_amount(client, symbol, contracts)
    params = {"reduceOnly": bool(reduce_only)}
    return submit_once(
        lambda: client.create_order(symbol, "market", side, amount, None, params),
        f"market {tag} order ({side} {amount:g})")


def place_stop(client, symbol: str, side: str, contracts: float, stop_price: float,
               tag: str = "sl") -> dict:
    """Protective stop. Kraken form: type `stop`, flat `stopPrice`, reduceOnly.

    `reduce_only=True` is what makes this CLOSE rather than open the opposite side, and
    it is not optional — on a one-way venue an unreduced stop in the wrong direction is
    a brand-new position.
    """
    _refuse_if_dry_run(client, f"a {tag} stop order")
    amount = round_amount(client, symbol, contracts)
    px = round_price(client, symbol, stop_price)
    params = {"reduceOnly": True, "stopPrice": px}
    return submit_once(
        lambda: client.create_order(symbol, "stop", side, amount, None, params),
        f"{tag} stop ({side} {amount:g} @ {px:g})")


def place_limit(client, symbol: str, side: str, contracts: float, price: float,
                tag: str = "tp") -> dict:
    _refuse_if_dry_run(client, f"a {tag} limit order")
    amount = round_amount(client, symbol, contracts)
    px = round_price(client, symbol, price)
    params = {"reduceOnly": True}
    return submit_once(
        lambda: client.create_order(symbol, "limit", side, amount, px, params),
        f"{tag} limit ({side} {amount:g} @ {px:g})")


def resolve_fill(client, symbol: str, order: dict | None,
                 expected: float | None = None) -> tuple[float, float]:
    """Return (contracts_filled, average_entry_price) for a just-placed market order.

    An accepted order is not a filled order. Resolution order, most authoritative first:
      1. the order's own filled/average (Kraken usually populates these)
      2. a re-read of the order by id
      3. the exchange's POSITION record — ground truth for what is actually held
    Never returns a ticker price as an entry price: an unverifiable fill is reported as
    (0, 0) so the caller flattens instead of protecting a position it cannot size.
    """
    filled = 0.0
    price = 0.0
    oid = (order or {}).get("id")
    if order:
        filled = float(order.get("filled") or 0.0)
        price = float(order.get("average") or order.get("price") or 0.0)

    for _ in range(3):
        if filled > 0 and price > 0:
            return filled, price
        if not oid:
            break
        try:
            o = retry(lambda: client.fetch_order(oid, symbol), attempts=2, base_delay=0.6)
        except Exception:
            o = None
        if o:
            f2 = float(o.get("filled") or 0.0)
            p2 = float(o.get("average") or o.get("price") or 0.0)
            if f2 > 0:
                filled = f2 or filled
            if p2 > 0:
                price = p2 or price
        if filled > 0 and price > 0:
            return filled, price
        time.sleep(1.0)

    # 3. the position itself
    pos = open_position(client, symbol)
    if pos.get("active"):
        pc = abs(float(pos.get("contracts") or 0.0))
        pp = float(pos.get("entry_price") or 0.0)
        if pc > 0:
            filled = pc
        if pp > 0:
            price = pp
    if filled <= 0 or price <= 0:
        log.error("could not resolve fill for %s (expected %s)", oid, expected)
        return 0.0, 0.0
    return filled, price


def flatten_all(client, symbol: str, attempts: int = 3) -> dict:
    """Close whatever the exchange actually holds, and verify it is flat.

    Sized off the LIVE position, never off a reported fill: an abort that sizes itself
    from an order response of 0 would try to close nothing and leave the real position
    running unprotected.
    """
    last = {"active": False, "contracts": 0.0}
    for i in range(attempts):
        pos = open_position(client, symbol)
        last = pos
        if pos.get("error"):
            log.error("flatten: cannot read the exchange (%s)", pos["error"])
            return {"flat": False, "attempts": i + 1, "residual": None,
                    "error": pos["error"]}
        if not pos.get("active") or float(pos.get("contracts") or 0.0) <= 0:
            return {"flat": True, "attempts": i + 1, "residual": 0.0}
        n = float(pos["contracts"])
        close_side = "buy" if str(pos.get("side")).lower() == "short" else "sell"
        log.critical("flattening %s of %s (attempt %d)", n, pos.get("side"), i + 1)
        try:
            place_market(client, symbol, close_side, n, reduce_only=True, tag="flatten")
        except Exception as exc:
            log.error("flatten order failed: %s", exc)
        time.sleep(2.0)
    pos = open_position(client, symbol)
    if pos.get("error"):
        # [MONEY] "I could not read the venue" is not "the venue is flat". The old form read
        # an outage as residual 0 and reported a successful flatten, which callers treat as
        # authorisation to stop managing the position.
        log.critical("flatten UNVERIFIED: cannot read %s (%s)", symbol, pos["error"])
        return {"flat": False, "attempts": attempts, "residual": None,
                "error": pos["error"]}
    left = float(pos.get("contracts") or 0.0) if pos.get("active") else 0.0
    if left > 0:
        log.critical("FLATTEN FAILED — %s still open on %s", left, symbol)
    return {"flat": left <= 0, "attempts": attempts, "residual": left}


def order_gone(client, symbol: str, order_id: str) -> bool:
    """Is this order provably finished? Ownership is an ID, and an ID is checkable.

    Returns True only for a state that means "no longer resting": canceled, expired,
    rejected, filled, or absent from a book that could be read COMPLETELY. Everything else
    — still open, or a read that cannot establish the answer — returns False, because
    "I could not tell" and "it is gone" are different answers and only the second is safe
    to act on.
    """
    if not order_id:
        return True
    # NOTE: this answers "may I CANCEL this id?", where the dangerous direction is touching
    # something that might still be alive, so absence must come from a book that can be shown
    # to be COMPLETE. Protecting an OPEN position asks the opposite question with the opposite
    # asymmetry and uses `fetch_order_state` directly — see Engine._ensure_sl.
    o = fetch_order(client, symbol, str(order_id))
    if o is None:
        # `fetch_order` returns None on failure as well as for a genuinely unknown id, so a
        # failed read is NOT proof of a cancel. Ask the book — strictly, because the
        # tolerant form reports an outage as an empty book.
        try:
            live = {str(x.get("id") or "") for x in open_orders(client, symbol, strict=True)}
        except Exception:
            return False
        if not _stops_view_available(client, symbol):
            # This venue refuses the trigger-order listing for the key in use, so the list
            # above can be missing stop orders by construction. Absence from a listing that
            # cannot be complete proves nothing.
            log.info("cannot prove order %s is gone: the trigger listing is unavailable "
                     "here, so the book cannot be shown to be complete", order_id)
            return False
        return str(order_id) not in live
    status = str(o.get("status") or "").lower()
    if status in ("canceled", "cancelled", "rejected", "expired", "closed", "filled"):
        return True
    return False


def order_provably_finished(client, symbol: str, order_id: str) -> bool:
    """May we STOP TRACKING this id? -> True for a definitive answer from the venue.

    Distinct from `order_gone`, which answers "may I CANCEL it?" and must not act on absence from
    a listing that cannot be shown to be complete. The dangerous direction here is the opposite:
    an id that can never be proven finished wedges the bot for ever.

    Live evidence (read-only, this account): for an id that does not exist,
    `privateGetOrdersStatus` answers `orders: []` and ccxt raises `OrderNotFound`, while
    `order_gone` returns False purely because this key cannot list trigger orders.
    """
    if not order_id:
        return True
    verdict, why = fetch_order_state(client, symbol, str(order_id))
    if verdict in ("finished", "missing"):
        log.info("order %s is provably finished (%s: %s)", str(order_id)[:12], verdict, why)
        return True
    return False


def _stops_view_available(client, symbol: str) -> bool:
    """Can this venue/key enumerate trigger (stop) orders at all?

    Asked, not assumed: Kraken answers `authenticationError` for the trigger query with the
    key in use here, which means a plain listing may not be the whole book.
    """
    try:
        client.fetch_open_orders(symbol, params={"stop": True})
        return True
    except Exception:
        return False


def cancel(client, symbol: str, order_id: str) -> bool:
    """Cancel one order and read the VENUE'S ANSWER, don't just avoid an exception.

    `canceled`, `filled` and `notFound` all mean "no longer resting" and count as success —
    a stop the venue already retired is the outcome we wanted. 'notFound' in particular is
    what Kraken returns when a reduce-only order was auto-cancelled on a flat position.
    Anything ambiguous is reported as a failure, because the caller treats False as "this
    order may still be alive".
    """
    if not order_id:
        return True
    if is_dry_run(client):
        log.info("dry-run client: not cancelling %s", order_id)
        return False
    try:
        # A cancel is idempotent on the venue, but it is still submitted once per id here:
        # a blind retry would be harmless for the SAME id, yet silence about the first
        # attempt is not something to paper over.
        o = client.cancel_order(str(order_id), symbol)
    except ccxt.OrderNotFound:
        return True
    except RETRYABLE as exc:
        log.warning("cancel %s not confirmed (%s) — treating as still open", order_id, exc)
        return False
    except Exception as exc:
        log.warning("cancel %s failed: %s", order_id, exc)
        return False
    status = str((o or {}).get("status") or "").lower()
    if status in ("canceled", "cancelled", "notfound", "not_found", "filled", "closed"):
        return True
    if status in ("", "open"):
        # The venue answered without a decisive status: ask the order itself.
        return order_gone(client, symbol, str(order_id))
    log.warning("cancel %s returned an unrecognised status %r", order_id, status)
    return False


def cancel_all_for_symbol(client, symbol: str) -> dict:
    """Cancel every resting order for the symbol. Used by the sync/flatten paths."""
    results = {}
    for o in open_orders(client, symbol):
        oid = str(o.get("id") or "")
        if not oid:
            continue
        results[oid] = cancel(client, symbol, oid)
    return results


def fetch_order(client, symbol: str, order_id: str) -> dict | None:
    if not order_id:
        return None
    try:
        return retry(lambda: client.fetch_order(str(order_id), symbol))
    except Exception as exc:
        log.warning("fetch_order %s failed: %s", order_id, exc)
        return None


def fetch_order_state(client, symbol: str, order_id: str) -> tuple[str, str]:
    """What does the venue say about this id? -> ("finished"|"open"|"missing"|"unknown", why)

    `fetch_order` collapses every failure into None, which cannot distinguish "the venue has no
    such order" from "I could not ask the venue". For an id this bot placed, the first is a
    DEFINITIVE answer and the second is not, and that difference decides whether a naked
    position gets its stop back or a duplicate gets stacked on a healthy one.
    """
    if not order_id:
        return "missing", "no id"
    try:
        o = retry(lambda: client.fetch_order(str(order_id), symbol))
    except Exception as exc:
        # ccxt raises OrderNotFound for an id the venue does not know. Kraken does not always
        # use that class, so the error TEXT is checked too — a "not found"-shaped error is a
        # definitive answer; anything else (timeout, auth, outage) is not.
        text = f"{type(exc).__name__}: {exc}".lower()
        if "notfound" in text or "not found" in text or "unknown order" in text \
                or "no such order" in text or "does not exist" in text:
            return "missing", text[:120]
        return "unknown", text[:120]
    if o is None:
        # NOT definitive. ccxt RAISES OrderNotFound for an id the venue does not know (proven
        # live on this account); a bare None is anomalous, so it must not be read as "gone" —
        # that is the direction that lets a live order be forgotten.
        return "unknown", "the venue returned no order for this id (anomalous, not a 404)"
    status = str(o.get("status") or "").lower()
    if status in ("canceled", "cancelled", "rejected", "expired", "closed", "filled"):
        return "finished", f"status={status}"
    return "open", ""


def order_trigger(order: dict | None) -> float:
    """The trigger price of an order, from wherever this venue put it."""
    o = order or {}
    info = o.get("info") or {}
    for v in (o.get("stopPrice"), info.get("stopPrice"), o.get("triggerPrice"),
              info.get("triggerPrice")):
        try:
            f = float(v)
            if f > 0:
                return f
        except (TypeError, ValueError):
            continue
    return 0.0


def _size(v) -> float | None:
    """One size field from a raw venue payload or a unified order, as a non-negative float.

    None means "this field is not there / not a number", and the caller moves on to the next
    source. It deliberately does NOT mean zero: reporting an unreadable field as 0.0 is what
    made a payload shape change read as "the venue holds no protection".
    """
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return abs(f)


def _size_readable(order: dict | None) -> bool:
    """Did this order carry a size the bot could actually READ — even a zero one?

    False means every source `order_amount` knows about is absent, empty or unparseable: the
    "silently zero" case, where a payload shape change turns a REAL size into 0. That is a
    mapping failure, not a report of zero, and `stop_is_confirmed` refuses to treat it as
    protection. A venue that explicitly states 0 (nothing left to fill) is a read, not a
    failure: it is refused as protection too, but by the size-specific branch with the number
    in the reason, NOT by this one.

    `filled` is deliberately NOT listed here: it is a quantity DEDUCTED from a total, not a
    source of outstanding size, so an order carrying only `filled` has no readable outstanding
    size and belongs in the unreadable branch.
    """
    o = order or {}
    info = o.get("info") or {}
    for v in (o.get("remaining"), info.get("unfilledSize"), info.get("unfilled"),
              o.get("amount"), info.get("quantity")):
        if _size(v) is not None:
            return True
    return False


def order_amount(order: dict | None) -> float:
    """How much of this order is still OUTSTANDING — the size it can still fill.

    [MONEY] ONE DEFINITION OF SIZE FOR THE WHOLE BOT. Every caller compares this against what
    is LEFT of the position: the coverage sum in `sync.reconcile` (do the stops cover the held
    contracts?), the duplicate search in `find_matching_order`, the ambiguous-TP adoption test
    in `engine._ensure_tp` and the ambiguous-stop candidate test. A partially filled order is
    where one wrong choice costs money: a half-filled TP reports `filledSize=495,
    unfilledSize=496` while the position's remainder is 496, so counting its TOTAL (991) would
    over-state coverage by 495 lots, and reporting the filled part — or zero — would make a
    resting order invisible and invite a duplicate reduce-only order. Outstanding is what this
    returns, and it is what a like-for-like comparison needs.

    Read in this order; the first readable value wins:
      1. `remaining` — ccxt's unified "still to fill" (and our wrapper's mirror of it);
      2. `info.unfilledSize` / `info.unfilled` — the venue's own field on Kraken's
         `/derivatives/api/v3/openorders`;
      3. `amount - filled` — a unified TOTAL with the FILLED part removed. The unified top-level
         `filled` is preferred and the venue-shaped `info.filledSize` is the fallback, so a ccxt
         order that reports `filled` but not `remaining` still reads as OUTSTANDING rather than
         total (this is the shape `open_orders` admits when the raw endpoint is down and it
         falls back to `fetch_open_orders`). A fill larger than the total clamps at 0, never
         negative;
      4. `amount` / `info.quantity` — a total with no fill information at all. This is the shape
         the per-id `fetch_order` path returns here; for an order the venue has not filled,
         total and outstanding are the same number, so the best available read is the total.

    A readable zero is a real "nothing left to fill" and is returned as 0.0 — as is "no size
    field is present or readable anywhere" (see `_size`). Callers must treat 0.0 as UNKNOWN
    coverage, never as "protected": `stop_is_confirmed` refuses such a stop outright.
    """
    o = order or {}
    info = o.get("info") or {}
    for v in (o.get("remaining"), info.get("unfilledSize"), info.get("unfilled")):
        f = _size(v)
        if f is not None:
            return f
    filled = _size(o.get("filled"))
    if filled is None:
        filled = _size(info.get("filledSize"))
    for v in (o.get("amount"), info.get("quantity")):
        f = _size(v)
        if f is not None:
            return max(f - filled, 0.0) if filled is not None else f
    return 0.0


def is_reduce_only(order: dict | None) -> bool:
    o = order or {}
    info = o.get("info") or {}
    for v in (o.get("reduceOnly"), info.get("reduceOnly")):
        if v is None:
            continue
        if isinstance(v, str):
            return v.strip().lower() in ("true", "1", "yes")
        return bool(v)
    # Kraken's own payload does not always echo it; a stop that would OPEN a position
    # would carry a direction matching the current bias, which the caller checks.
    return None  # type: ignore[return-value]


def price_tick(client, symbol: str) -> float:
    """The market's price step, for tolerance decisions. Falls back to a cent."""
    try:
        m = client.market(symbol) or {}
        return float((m.get("precision") or {}).get("price") or 0.0) or 0.01
    except Exception:
        return 0.01


def stop_is_confirmed(client, symbol: str, order_id: str, *, side: str,
                      min_amount: float, require_breakeven: bool,
                      entry: float, trigger: float) -> tuple[bool, str]:
    """Verify a protective stop actually exists, protects the right way, at the right size.

    Mirrors Jack's gate, which exists because an earlier bot demanded breakeven from an
    INITIAL stop (which sits adversely to entry by design) and produced a 106-cycle
    force-close churn. `require_breakeven` is explicit at every call site and is False
    for initial / orphan / any loss-limiting stop.

    Two reads, because Kraken lists stops on a separate query: fetch by id, then fall
    back to the open-order lists. Only a stop found and validated is trusted.
    """
    o = fetch_order(client, symbol, order_id)
    if not o:
        for cand in open_orders(client, symbol):
            if str(cand.get("id") or "") == str(order_id):
                o = cand
                break
    if not o:
        return False, "order not found"

    status = str(o.get("status") or "").lower()
    if status in ("canceled", "cancelled", "rejected", "expired"):
        return False, f"status={status}"
    if status in ("closed", "filled"):
        # A stop that has ALREADY FILLED is not resting protection. Accepting it as "confirmed"
        # is how a partially filled stop left the remainder of a position unprotected: the
        # caller was told it was fine because the order was found (Terra round-6 finding 2).
        return False, f"status={status} (no longer resting)"

    o_side = str(o.get("side") or "").lower()
    if o_side and o_side != str(side).lower():
        return False, f"side={o_side} (expected {side})"

    # [MONEY] AN UNREADABLE SIZE IS NOT EVIDENCE OF ADEQUATE PROTECTION — AND NEITHER IS A READ
    # OF ZERO. The old form (`if amt and amt + 1e-9 < min_amount`) skipped the undersize check
    # whenever the read came back 0, so a payload shape that stopped carrying the size silently
    # turned every undersize test into a pass, AND an order the venue genuinely reported as
    # "nothing left to fill" was accepted as full protection for an open position. Both are the
    # same defect class: an order that can fill zero contracts protects zero contracts, whether
    # the zero was STATED or produced by amount-step rounding. The two reasons stay DISTINCT,
    # because engine.py matches the word "unreadable" to leave an order alone rather than cancel
    # and re-place it; a stated zero is a readable size and takes the size-specific reason.
    amt = order_amount(o)
    if amt <= 0 and not _size_readable(o):
        return False, (f"size unreadable (order {str(o.get('id') or '')[:12]} reports no size "
                       f"at all; required {float(min_amount):g})")
    if float(min_amount) > 0 and amt + 1e-9 < float(min_amount):
        return False, f"size {amt:g} < required {float(min_amount):g}"

    ro = is_reduce_only(o)
    if ro is False:
        return False, "order is not reduce-only"

    # [MONEY] The venue's OWN trigger, not a substitute for it. `order_trigger(o) or trigger`
    # used to fall back to the EXPECTED value whenever the order reported none, so a stop at
    # any price at all passed the trigger checks — a materially looser stop silently replaced
    # the risk limit the strategy chose (Terra round-6 finding 1).
    trig = order_trigger(o)
    if trig <= 0:
        # ccxt may return the order WITHOUT the trigger it did not map (it reads
        # `order.priceTriggerOptions.triggerPrice`). Our own listing carries `stopPrice` and is
        # live-proven to work here, so consult it before declaring the level unverifiable — a
        # false "no trigger" at entry force-closes every position, and per-tick it can never
        # confirm protection.
        try:
            for cand in open_orders(client, symbol):
                if str(cand.get("id") or "") == str(order_id):
                    trig = order_trigger(cand)
                    if trig > 0:
                        log.info("trigger for %s read from our own order listing (%.4f)",
                                 str(order_id)[:12], trig)
                        break
        except Exception as exc:
            log.warning("could not cross-check the trigger for %s: %s", str(order_id)[:12], exc)
    if trig <= 0:
        return False, ("the venue reports no trigger price for this order, so its level "
                       "cannot be verified")
    want = float(trigger or 0.0)
    if want > 0:
        tol = max(2.0 * price_tick(client, symbol), want * 0.0005)
        if abs(trig - want) > tol:
            return False, (f"trigger {trig:g} is not the expected {want:g} "
                           f"(tolerance {tol:g})")
    if entry > 0:
        if require_breakeven:
            if str(side).lower() == "sell" and trig < entry:
                return False, f"breakeven stop is below entry ({trig:g} < {entry:g})"
            if str(side).lower() == "buy" and trig > entry:
                return False, f"breakeven stop is above entry ({trig:g} > {entry:g})"
        else:
            # A loss-limiting stop MUST sit on the losing side of entry. A stop on the
            # profitable side would lock in a loss at a price the market has not reached,
            # and would be rejected by the venue anyway.
            if str(side).lower() == "sell" and trig > entry:
                return False, f"stop trigger {trig:g} is above entry {entry:g} for a sell"
            if str(side).lower() == "buy" and trig < entry:
                return False, f"stop trigger {trig:g} is below entry {entry:g} for a buy"
    return True, ""


def account_log(client, since_ms: int | None = None) -> list[dict]:
    """Kraken futures account log — authoritative fees and funding, per fill.

    Ken reconciled fees and funding from this endpoint for months; it is the only place
    the venue states what it actually charged. Returns [] on any failure so a reporting
    path can never break trading.
    """
    try:
        raw = client.historyGetAccountLog({"since": since_ms} if since_ms else {})
    except Exception as exc:
        log.warning("account log unavailable: %s", exc)
        return []
    if isinstance(raw, dict):
        return list(raw.get("logs") or [])
    return []
