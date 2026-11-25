"""Engine — one tick of the bot: manage an open position, or look for an entry.

Deliberately small and explicit. Mel's exit stack is ONE take-profit and ONE stop:
  SL      = entry -/+ max(ATR% x 1.5, 2.5%)   confirmed reduce-only stop, FULL size
  TP      = +/-4.0%                           reduce-only limit, FULL size
  hold cap= 24h -> market close at the cap

There is no partial exit, no breakeven move and no trailing runner — that is the whole
difference from Jack. Both protective orders rest on the venue from the entry tick, so
protection never depends on this process being alive between ticks.

Because both orders cover the FULL position, exactly one of them will fill and the other
is left dangling. Retiring the survivor is therefore not optional housekeeping: a
leftover reduce-only stop can close a LATER position. `_close`, `_finish` and the TP/SL
reconcile paths all clear the sibling, and `sync.py` reports dangling orders from the
venue's own book.

Protective-order rules, learned expensively:
  * An INITIAL stop is a loss-limiting stop and must NOT be validated as a breakeven
    stop. Demanding breakeven from it caused a 106-cycle force-close churn in the
    previous bot. `require_breakeven=False` everywhere in Mel — there is no breakeven
    stop to require it of.
  * Every protective order is confirmed open, correct side, reduce-only and
    sufficient size BEFORE it is trusted. If the SL cannot be confirmed, the
    position is closed immediately and the row is recorded (see _force_close).
  * If a close cannot be confirmed flat, NO closed row is written — a false closed
    row would tell the guards a live position is finished.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import exchange as ex
import settlement
from exchange import ExchangeError
from notify import alert_once

# The fleet pause control — TIER 2 (see /opt/data/DESIGN_PAUSE_FLEET.md, /opt/data/shared/PAUSE.md).
#
# Imported defensively: if the shared control is absent or unimportable the bot must STILL START and keep
# MANAGING a live position. Losing the ability to exit is far worse than losing the ability to enter, so
# a hard failure here would trade a safety feature for a stuck position. The entry gate in try_enter then
# fails CLOSED, which is the direction that cannot open new risk.
try:
    # APPEND, not insert(0): a foreign directory at the front of sys.path would silently shadow one of
    # the bot's own modules the day /opt/data/shared gains a file with the same name. `pause_state` is
    # only provided there, so appending still finds it and can never take priority over Mel's own code.
    sys.path.append("/opt/data/shared")
    import pause_state as PAUSE
except Exception:                      # noqa: BLE001 — a missing control must not stop the bot
    PAUSE = None

log = logging.getLogger("mel.engine")


def _now_ms() -> int:
    return int(time.time() * 1000)


def _hours_since(iso_ts: str) -> float:
    try:
        t0 = datetime.fromisoformat(str(iso_ts))
        if t0.tzinfo is None:
            t0 = t0.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - t0).total_seconds() / 3600.0
    except (TypeError, ValueError):
        return 0.0


def _next_utc_midnight() -> str:
    """The ISO timestamp of the next UTC day boundary — when the day-scoped rails reopen."""
    n = datetime.now(timezone.utc)
    nxt = (n + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return nxt.isoformat()


# ── the CIC guard-rail OVERRIDE ("resume despite the rail") ──────────────────────────
# Written by the dashboard (POST /api/bot/{id}/override) when an operator decides a rail
# should not hold. Same file, same `until` field, same parse as Jill / sol-v2 / BX, so the
# card's "guard rails bypassed" read can never disagree with what this bot actually does.
OVERRIDE_FILE = "/opt/data/cic/bot_override_state.json"


def read_override_until(bot_id: str, now_epoch: float) -> str | None:
    """The `until` of an ACTIVE override for `bot_id`, or None when there is not one.

    FAIL-CLOSED in the only direction that matters: anything unreadable (absent, malformed,
    no entry, empty/garbage/naive `until`) means the override is NOT active, so the rails keep
    their authority. The cost of being wrong that way is a rail the operator must re-grant;
    the other way it is a money-bot trading past a halt nobody authorised.
    """
    try:
        data = json.loads(Path(OVERRIDE_FILE).read_text())
    except Exception:                              # noqa: BLE001 — absent/malformed = not overridden
        return None
    entry = data.get(bot_id) if isinstance(data, dict) else None
    if not isinstance(entry, dict):
        return None
    until = entry.get("until")
    if not until:
        return None
    try:
        dt = datetime.fromisoformat(str(until).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if dt.tzinfo is None:
        return None
    if dt > datetime.fromtimestamp(float(now_epoch), tz=timezone.utc):
        return str(until)
    return None


class Engine:
    def __init__(self, cfg, client, state, db, notify, dry_run=True):
        self.cfg = cfg
        self.client = client
        self.state = state
        self.db = db
        self.notify = notify
        self.dry_run = dry_run
        self.sym = cfg["bot"]["ccxt_symbol"]
        self.raw_sym = cfg["bot"]["symbol"]
        self.spec = ex.market_spec(client, self.sym)

    # ── the guard-rail gate for the dashboard pause card ──────────────────────
    def rail_state(self) -> dict | None:
        """The rail currently blocking NEW ENTRIES, as the card should show it, or None.

        ONE DEFINITION PER RAIL: each condition is the SAME ``guards.g_*`` function the
        entry path calls (``run_guards``), so the card can never disagree with the gate that
        actually blocked the entry. The only reason text not taken from a guard is the halted
        latch, which already carries its own operator-facing string.

        DELIBERATELY EXCLUDED: the signal-shaped filters (g_risk_reward, g_counter_trend,
        g_rsi_extreme, g_bb_extreme) — they reject ONE candidate signal, not the bot — and the
        safety latches (g_sl_failure_streak, g_unverified_settlements, g_cooldown), which an
        override must never skip. A rail here means "this bot cannot open a position at all
        right now", which is a stronger and truer claim.

        Never raises: it runs on the commit path, and a reporting failure must not be able to
        stop a tick's state from being saved. On any error it logs and returns None — showing
        no pause rather than inventing one.
        """
        try:
            import guards as G
            g = self.cfg            # the guard functions index cfg["guards"] themselves
            if self.state.halted:
                return {"paused": True, "reason": self.state.halt_reason or "halted",
                        "reset_at": None, "source": "guard_rail"}

            stats = self.db.stats()

            def eq(key_eq: str, key_ro: str) -> float:
                v = stats.get(key_eq)
                if v is None:
                    v = stats.get(key_ro)
                try:
                    return float(v or 0.0)
                except (TypeError, ValueError):
                    return 0.0

            # Drawdown is read READ-ONLY here, on purpose: it blocks the entry path every tick
            # while breached, so publishing it keeps the card honest with no money-path effect.
            ok, why = G.g_drawdown(eq("pnl_total_equity_pct", "pnl_total_pct"), g)
            if not ok:
                return {"paused": True, "reason": why, "reset_at": None,
                        "source": "guard_rail"}

            reset_at = _next_utc_midnight()
            # Replicate effective_loss_streak's operator acknowledgement (run_bot.py) so the
            # card and the entry gate read the SAME streak.
            streak = int(stats.get("consecutive_losses") or 0)
            ack = self.state.get_flag("ack_loss_streak") or {}
            ack_at = str(ack.get("at") or "")
            last_loss = str(stats.get("last_loss_at") or "")
            if streak and ack_at and (not last_loss or ack_at > last_loss):
                streak = 0

            for ok, why in (
                G.g_daily_loss_cap(eq("pnl_today_equity_pct", "pnl_today_pct"), g),
                G.g_consecutive_losses(streak, g, stats.get("last_loss_at"),
                                       datetime.now(timezone.utc).isoformat()),
                G.g_daily_trades(stats.get("trades_today", 0), g),
            ):
                if not ok:
                    return {"paused": True, "reason": why, "reset_at": reset_at,
                            "source": "guard_rail"}
            return None
        except Exception as exc:                       # noqa: BLE001 — reporting must never break a tick
            log.warning("rail_state failed (%s: %s) — publishing no gate this tick",
                        type(exc).__name__, exc)
            return None

    # ── entry ────────────────────────────────────────────────────────────────
    def adopt_orphan(self, live: dict) -> dict:
        """State is flat but the exchange HOLDS a position — adopt it, never double it.

        [MONEY] Without this check the tick sees a flat state, falls through to the entry
        search, and opens a SECOND position on top of the live one. Reproduced live:
        a missing state.json turned a 28-contract short into 52. Every guard rail
        (max_daily_trades, daily-loss cap, drawdown halt) is state/DB driven, so an
        unadopted orphan is also invisible to all of them.

        Adopt from the exchange, which is authoritative, then make sure the position is
        actually protected — an orphan may have no stop at all.
        """
        is_long = str(live.get("side")).lower() == "long"
        direction = "long" if is_long else "short"
        contracts = float(live.get("contracts") or 0.0)
        entry = float(live.get("entry_price") or 0.0)
        stop_side = "sell" if is_long else "buy"

        # [MONEY] Adoption places a stop (and may place a TP), so a dry run must not do it.
        if self.dry_run:
            log.critical("ORPHAN DETECTED (dry run): the venue holds %s %s on %s with no "
                         "bot state. Nothing placed, nothing written — this tick is a dry "
                         "run.", f"{contracts:g}", direction.upper(), self.raw_sym)
            self.notify(f"🛟 Mel — ORPHAN DETECTED in DRY RUN: the venue holds {contracts:g} "
                        f"{direction.upper()} {self.raw_sym} with no bot state, and a dry run "
                        f"places nothing. Arm the bot (or protect it by hand) to have it "
                        f"adopted and stopped.")
            return {"status": "orphan_dry_run", "contracts": contracts,
                    "direction": direction}

        # REPORT, NEVER ADOPT, orders that were already resting. An orphan exists because
        # the state file that would prove ownership was lost, so a structural match — right
        # side, full size, reduce-only — is a guess, not proof. Adopting one stores its id
        # and this bot later cancels an order it never placed (Terra finding 6).
        preexisting: list[dict] = []
        try:
            preexisting = ex.open_orders(self.client, self.sym)
        except Exception as exc:
            log.warning("orphan: could not list the resting book: %s", exc)

        # Leverage as the VENUE reports it for this position, not the config: an orphan was
        # opened by something else, and a later config edit must not rewrite its risk.
        adopt_lev = int(live.get("leverage") or self.cfg["sizing"]["leverage"] or 1)

        sl = tp = None

        # 1. OUR stop FIRST, placed and CONFIRMED, before a take-profit is attempted. An
        #    orphan that stays naked through a TP/DB delay is the failure this path exists
        #    to prevent.
        pct = float(self.cfg["exits"].get("sl_pct_floor", 0.025))
        stop_px = ex.round_price(self.client, self.sym,
                                 entry * (1 - pct) if is_long else entry * (1 + pct))
        try:
            o = ex.place_stop(self.client, self.sym, stop_side, contracts, stop_px,
                              tag="sl-adopt")
            ok, why = ex.stop_is_confirmed(
                self.client, self.sym, o.get("id"), side=stop_side,
                min_amount=contracts, require_breakeven=False,
                entry=entry, trigger=stop_px)
            if ok:
                sl = o
                self.notify(f"🛟 Mel — ORPHAN ADOPTED. Placed OUR SL "
                            f"{contracts:g} @ {stop_px:,.2f} on {self.raw_sym}.")
            else:
                log.critical("orphan stop not confirmed: %s", why)
        except ex.AmbiguousSubmission as amb:
            self._lock_orphan_stop(stop_side, contracts, stop_px, "sl-adopt")
            self.notify(f"💀 Mel — CRITICAL: the stop for an ADOPTED {self.raw_sym} orphan "
                        f"may have been accepted without a reply and this bot has NO id for "
                        f"it ({amb}). It may be resting untracked; new entries stay blocked "
                        f"until the book is reconciled (`--sync`).")
        except Exception as exc:
            log.critical("orphan stop placement FAILED: %s", exc)

        if sl is None:
            flat = ex.flatten_all(self.client, self.sym)
            self.notify(f"💀 Mel — ORPHAN could not be protected; "
                        f"{'flattened' if flat.get('flat') else 'FLATTEN FAILED, residual ' + str(flat.get('residual'))}. "
                        f"MANUAL ACTION REQUIRED on {self.raw_sym}.")
            return {"status": "orphan_unprotected", "flatten": flat}

        # 2. then OUR take-profit, at the configured rate: an adopted position with no TP
        #    cannot exit for a profit at all.
        tp_pct = float(self.cfg["exits"]["tp_pct"])
        tp_px = ex.round_price(self.client, self.sym,
                               entry * (1 + tp_pct) if is_long else entry * (1 - tp_pct))
        try:
            tp = ex.place_limit(self.client, self.sym, stop_side, contracts, tp_px,
                                tag="tp-adopt")
        except Exception as exc:
            log.critical("orphan TP placement failed: %s", exc)
            self.notify(f"⚠️ Mel — adopted orphan has NO take-profit and placing one "
                        f"failed ({exc}). It is protected on the loss side only.")

        # [MONEY] Adoption supersedes a queued UNVERIFIED force-close (round-12 F1). The close
        # that could not be confirmed queued a `close_unverified` record with no trade id so the
        # protection failure stayed visible and entries stayed blocked. Adopting this position
        # re-books it under a tracked row whose exit is recorded through the normal path; leaving
        # the record queued would make `retry_pending_close` invent a SECOND row for the same
        # contracts and double-count the trade and its P&L. Consume it, and mark the row so the
        # cause stays auditable.
        from_failed_entry = self._consume_unverified_pending_close()

        # 3. the ledger row, so the guard rails can see the position
        trade_id = (self.state.get_open() or {}).get("trade_id")
        if not trade_id:
            row = self.db.open_trade()
            trade_id = row["id"] if row else None
        if not trade_id:
            # Nothing in the DB either: create the row so guard-rail accounting sees it.
            trade_id = self.db.record_entry(
                symbol=self.raw_sym, direction=direction, entry_price=entry,
                contracts=contracts, leverage=adopt_lev,
                notional=entry * contracts * self.spec["contract_size"],
                sl_price=entry * (0.975 if is_long else 1.025), sl_pct=0.025,
                tp1_price=0.0, tp2_price=0.0,
                trigger="sl_failure_forceclose" if from_failed_entry else "adopted_orphan",
                guard_config={"adopted_orphan": True,
                              # Deployment is unknowable for an orphan; leaving the field
                              # out makes the guards count it at 1.0 (fail-closed) rather
                              # than at a fraction this bot would be inventing.
                              "leverage_from_venue": adopt_lev})

        if preexisting:
            ids = ", ".join(str(o.get("id") or "?")[:10] for o in preexisting[:3])
            log.warning("orphan: %d order(s) were already resting (%s) — left untouched",
                        len(preexisting), ids)
            self.notify(f"⚠️ Mel — {len(preexisting)} order(s) were already resting on "
                        f"{self.raw_sym} ({ids}) and were NOT placed by this bot. They were "
                        f"left untouched; they can still close a future position, so run "
                        f"`--sync` to inspect them.")

        pos = {
            "trade_id": trade_id, "direction": direction, "entry_price": entry,
            "contracts": contracts, "remaining": contracts,
            "entry_order_id": "", "opened_ms": None,
            "sl_price": ex.order_trigger(sl) or float(sl.get("stopPrice") or 0.0),
            "sl_order_id": str(sl.get("id")),
            "tp_price": float((tp or {}).get("price") or 0.0),
            "tp_order_id": str((tp or {}).get("id") or ""), "tp_filled": False,
            "entry_time": datetime.now(timezone.utc).isoformat(), "adopted": True,
            # Persisted so the EXIT is costed with the terms the position was taken on
            # under (Terra finding 7).
            "leverage": adopt_lev,
            "contract_size": float(self.spec.get("contract_size") or 1.0),
            "margin_usdt": None,      # unknown for an orphan; falls back to notional / lev
        }
        self._remember_ids(pos.get("sl_order_id"), pos.get("tp_order_id"))
        self.state.set_open(pos)
        self.notify(f"🛟 Mel — ADOPTED an existing {direction.upper()} of {contracts:g} "
                    f"{self.raw_sym} @ {entry:,.2f} that had no bot state (lost state file "
                    f"or interrupted entry). SL {pos['sl_price']:,.2f}. No new position opened.")
        return {"status": "adopted", "contracts": contracts, "direction": direction}

    def _note_entry_persistence_failure(self, side, stop_side, filled, entry_price,
                                        sizing, sl_price, tp_price, trade_id, order, exc):
        """A fill landed but the ledger/state write failed: refresh the durable reservation.

        The reservation was written BEFORE the market submit, so the position is already
        recoverable; this only refreshes it with the post-fill facts (actual size/price, the
        entry order id, and the ledger id when record_entry succeeded but set_open did not).
        It is never cleared here — the next tick's reconcile_entry_inflight() reads the venue,
        re-protects the live position with these terms, and retires the reservation only after
        the position is tracked or proven flat.
        """
        try:
            resv = dict(self.state.get_flag("entry_inflight") or {})
            resv.update({
                "side": side, "stop_side": stop_side,
                "entry_price": entry_price, "contracts": filled,
                "sl_price": sl_price, "tp_price": tp_price,
                "entry_order_id": str((order or {}).get("id") or ""),
                "trade_id": trade_id,
                "leverage": int(self.cfg["sizing"]["leverage"]),
                "margin_fraction": round(sizing.margin_fraction, 6),
                "funds": round(sizing.funds, 2),
                "margin_usdt": round(sizing.margin, 6),
            })
            self.state.set_flag("entry_inflight", resv)
        except Exception as e:
            log.critical("could not refresh the entry reservation (%s) — the pre-submit "
                         "marker still stands", e)
        self.notify(
            f"💀 Mel — CRITICAL: the entry FILLED on {self.raw_sym} ({filled:g} @ "
            f"{entry_price:,.2f}) but its record could not be written ({exc}). The position "
            f"is RESERVED and the next tick will re-protect and re-record it with these "
            f"terms; new entries are blocked until it is reconciled or proven flat.")

    def reconcile_entry_inflight(self) -> dict:
        """Recover an entry whose fill landed but whose ledger/state write failed.

        Runs at the top of the tick (before the orphan guard) while `entry_inflight` is set.
        The reservation carries the exact side/size/SL/TP the entry was opened under, so the
        position is re-protected and re-recorded with ITS terms — not adopted as an unknown
        orphan with config-floor stops. The reservation is retired ONLY after the position is
        tracked (state + ledger + a resting stop) or the venue proves flat; an unreadable venue
        is never treated as flat.
        """
        resv = self.state.get_flag("entry_inflight") or {}
        if not resv:
            return {"status": "none"}

        # Already tracked (a prior reconcile got as far as set_open before a crash): retire the
        # reservation and let manage() re-verify protection. Re-protecting here would stack a
        # second stop on a position that already has one.
        if self.state.get_open():
            self.state.clear_flag("entry_inflight")
            log.warning("in-flight entry was already tracked — reservation retired")
            return {"status": "already_tracked"}

        try:
            live = ex.open_position(self.client, self.sym)
        except Exception as exc:
            live = {"active": False, "error": str(exc)}
        if live.get("error"):
            self.notify("💀 Mel — CRITICAL: an in-flight entry is RESERVED but the venue "
                        "cannot be read; the reservation stays until flatness is proven.")
            return {"status": "blocked", "reason": "venue unreadable"}

        active = bool(live.get("active")) and float(live.get("contracts") or 0.0) > 0
        if not active:
            # The fill never landed (rejected/timeout with nothing placed) or was already
            # closed. Nothing to track or protect: retire the reservation and record nothing.
            self.state.clear_flag("entry_inflight")
            log.warning("in-flight entry resolved FLAT — reservation retired, nothing recorded")
            return {"status": "flat"}

        side = str(resv.get("side") or "buy")
        stop_side = str(resv.get("stop_side") or ("sell" if side == "buy" else "buy"))
        direction = "long" if side == "buy" else "short"
        contracts = float(live.get("contracts") or resv.get("contracts") or 0.0)
        entry_price = float(live.get("entry_price") or resv.get("entry_price") or 0.0)
        sl_price = float(resv.get("sl_price") or 0.0)
        tp_price = float(resv.get("tp_price") or 0.0)
        leverage = int(resv.get("leverage") or self.cfg["sizing"]["leverage"] or 1)

        # 1. attach an existing open ledger row, else create one with the reservation's terms.
        trade_id = resv.get("trade_id") or None
        try:
            if not trade_id:
                row = self.db.open_trade()
                trade_id = row["id"] if row else None
            if not trade_id:
                trade_id = self.db.record_entry(
                    symbol=self.raw_sym, direction=direction, entry_price=entry_price,
                    contracts=contracts, leverage=leverage, sl_price=sl_price,
                    tp1_price=tp_price, tp2_price=0.0,
                    notional=entry_price * contracts * float(self.spec.get("contract_size") or 1.0),
                    sl_pct=(abs(entry_price - sl_price) / entry_price) if entry_price else 0.0,
                    trigger="entry_inflight_recovered",
                    guard_config={"margin_fraction": float(resv.get("margin_fraction") or 0.0),
                                  "funds_at_entry": float(resv.get("funds") or 0.0),
                                  "exits": "single_tp_sl", "tp_pct": float(resv.get("tp_pct") or 0.0),
                                  "entry_inflight": True})
        except Exception as exc:
            log.critical("could not create/attach the recovered entry's ledger row: %s", exc)
            trade_id = None

        # 2. protect FIRST (the stop is venue-side, independent of the ledger). If a prior
        # partial reconcile already placed one, verify it rather than stacking a second — a
        # duplicate reduce-only stop can close a LATER position (round-4 finding 5).
        sl_order = None
        prior_sl_id = str(resv.get("sl_order_id") or "")
        if prior_sl_id:
            try:
                if ex.order_provably_finished(self.client, self.sym, prior_sl_id):
                    log.warning("the recovered stop %s is gone — re-placing", prior_sl_id[:12])
                else:
                    sl_order = {"id": prior_sl_id}
            except Exception:
                sl_order = {"id": prior_sl_id}   # cannot tell: trust the id we already hold
        if not sl_order:
            try:
                sl_order = ex.place_stop(self.client, self.sym, stop_side, contracts, sl_price,
                                         tag="sl")
                ok, why = ex.stop_is_confirmed(self.client, self.sym, sl_order.get("id"),
                                               side=stop_side, min_amount=contracts,
                                               require_breakeven=False, entry=entry_price,
                                               trigger=sl_price)
                if not ok:
                    raise ExchangeError(f"SL not confirmed: {why}")
            except ex.AmbiguousSubmission as amb:
                self._lock_orphan_stop(stop_side, contracts, sl_price, "sl-recover")
                self.notify(f"💀 Mel — CRITICAL: the recovered entry's stop may be resting "
                            f"without an id ({amb}); new entries stay blocked until `--sync`.")
                sl_order = None
            except Exception as exc:
                log.critical("recovered-entry SL placement failed (%s)", exc)
                sl_order = None

        if sl_order is None:
            # Cannot protect the live position: fail closed — flatten, and keep the reservation
            # if flatness cannot be proven.
            flat = ex.flatten_all(self.client, self.sym)
            if flat.get("flat"):
                self.state.clear_flag("entry_inflight")
                self.notify("💀 Mel — the in-flight entry could not be protected and was "
                            "flattened; nothing recorded.")
                return {"status": "flattened"}
            self.notify("💀 Mel — CRITICAL: the in-flight entry could not be protected AND "
                        "could not be flattened. MANUAL ACTION REQUIRED on " + self.raw_sym + ".")
            return {"status": "blocked", "reason": "unprotectable"}

        # 3. TP, best effort (the SL already covers the loss side).
        tp_id = ""
        try:
            tp = ex.place_limit(self.client, self.sym, stop_side, contracts, tp_price, tag="tp")
            tp_id = str(tp.get("id") or "")
        except Exception as exc:
            log.critical("recovered-entry TP placement failed: %s", exc)

        # Record the stop/TP ids into the reservation BEFORE any further persistence, so a later
        # failure (ledger or state) cannot cause the next tick to place a SECOND stop on a
        # position that already has one.
        try:
            resv = dict(self.state.get_flag("entry_inflight") or {})
            resv["sl_order_id"] = str(sl_order.get("id") or "")
            resv["tp_order_id"] = tp_id
            self.state.set_flag("entry_inflight", resv)
        except Exception as exc:
            log.critical("could not persist the recovered stop id into the reservation: %s", exc)

        if not trade_id:
            # The ledger row still did not land. The stop IS placed and its id is recorded in
            # the reservation above, so the position is protected; keep the reservation and let
            # the next tick retry the record.
            self.notify("🧾 Mel — the recovered entry is protected (SL placed) but its ledger "
                        "row could not be written; the reservation stays and the next tick "
                        "retries the record. New entries are blocked.")
            return {"status": "blocked", "reason": "ledger unwritable"}

        pos = {
            "trade_id": trade_id, "direction": direction, "entry_price": entry_price,
            "contracts": contracts, "remaining": contracts,
            "entry_order_id": str(resv.get("entry_order_id") or ""),
            "opened_ms": resv.get("opened_ms"),
            "sl_price": sl_price, "sl_order_id": str(sl_order.get("id") or ""),
            "tp_price": tp_price, "tp_order_id": tp_id, "tp_filled": False,
            "entry_time": datetime.now(timezone.utc).isoformat(), "entry_inflight": True,
            "leverage": leverage,
            "contract_size": float(self.spec.get("contract_size") or 1.0),
            "margin_usdt": resv.get("margin_usdt"), "funds_usdt": resv.get("funds"),
        }
        try:
            self._remember_ids(pos.get("sl_order_id"), pos.get("tp_order_id"))
            self.state.set_open(pos)
            self.state.clear_flag("entry_inflight")
        except Exception as exc:
            # set_open failed (the same class that triggered this recovery). The stop is placed
            # and its id is in the reservation, so the position is protected; keep the reservation
            # and let the next tick re-run the record via the already_tracked path.
            log.critical("recovered entry could not be persisted (%s) — reservation kept", exc)
            return {"status": "blocked", "reason": "state unwritable"}

        self.notify(f"🛟 Mel — RECOVERED an in-flight entry: {direction.upper()} {contracts:g} "
                    f"{self.raw_sym} @ {entry_price:,.2f} (its record had failed after the "
                    f"fill). SL {sl_price:,.2f} / TP {tp_price:,.2f}. Reservation retired.")
        return {"status": "recovered", "contracts": contracts, "direction": direction}

    def try_enter(self, decision, sizing) -> dict:
        # ── TIER 2: the pause gate ────────────────────────────────────────────────────────────
        # This is the ONLY code path that can open a position (run_bot.py:771 calls it, nowhere else),
        # so the gate belongs here. Tier 1 in the runner is not enough on its own: a hand-run
        # `.venv/bin/python3 run_bot.py --live` never passes through the wrapper.
        #
        # It CANNOT disturb a live position: `manage()` — stop re-confirmation, the time stop and the
        # exits — runs from tick() BEFORE this function, and nothing here cancels, amends or replaces
        # an order. A pause means "no NEW positions", nothing more.
        #
        # Fails CLOSED in the direction that matters: an unreadable/corrupt store returns True from
        # blocks_new_entries(), and a control that cannot be imported at all is treated as "cannot
        # establish intent" rather than as "no pause". A missed entry is bounded; a safety control that
        # silently stops existing is not. [FLAGGED FOR REVIEW: the alternative reading is "absent
        # control == no pause == trade normally". The conservative direction is chosen and logged
        # loudly; it is a one-line change if the reviewer rules the other way.]
        if PAUSE is None:
            log.error("pause control unavailable (/opt/data/shared not importable) — "
                      "refusing to open; managing only")
            return {"status": "paused", "reason": "pause control unavailable — refusing to open"}
        if PAUSE.blocks_new_entries(self.cfg["bot"]["id"]):
            log.warning("paused — refusing to open a new position this tick")
            return {"status": "paused", "reason": "paused: new entries refused"}

        # ── an ENTRY already in flight (reserved, unreconciled) blocks a second ──────────────
        # [MONEY] The `entry_inflight` reservation is written BEFORE the market submit and is
        # retired only once the position is tracked (ledger + state) or proven flat. If it is
        # still set here, a previous entry's fill landed but its persistence failed, and this is
        # the ONLY code path that can open a position — so refusing here is what stops one entry
        # from becoming two. The reservation is reconciled at the top of the next tick (run_bot),
        # where the venue is read and the live position is re-protected or flattened.
        if self.state.get_flag("entry_inflight"):
            log.error("an entry is still in-flight (reserved) — refusing to open a second")
            return {"status": "error",
                    "message": "entry_inflight: reconcile the reserved entry first"}

        if self.dry_run:
            return {"status": "dry_run", "action": decision.action,
                    "entry": decision.entry, "sl": decision.sl, "tp": decision.tp,
                    "contracts": sizing.contracts, "margin": sizing.margin,
                    "notional": sizing.notional, "scale": sizing.scale}

        side = "buy" if decision.action == "LONG" else "sell"
        stop_side = "sell" if side == "buy" else "buy"

        sl_price = ex.round_price(self.client, self.sym, decision.sl)
        tp_price = ex.round_price(self.client, self.sym, decision.tp)

        # ── [MONEY] RESERVE BEFORE SUBMIT ────────────────────────────────────────────────────
        # A durable marker of the entry this bot is ABOUT to place, written before the market
        # order. If the fill lands but the ledger/state write below fails (SQLite I/O, lock,
        # disk full), this reservation is what the next tick reconciles: it knows the exact
        # side, size and SL/TP the entry was opened under, so the position is re-protected and
        # re-recorded with ITS terms rather than adopted as an unknown orphan with config-floor
        # stops. It is retired ONLY after the position is tracked (ledger + state) or proven flat.
        self.state.set_flag("entry_inflight", {
            "side": side, "stop_side": stop_side,
            "contracts": sizing.contracts,
            "sl_price": sl_price, "tp_price": tp_price,
            "leverage": int(self.cfg["sizing"]["leverage"]),
            "margin_fraction": round(sizing.margin_fraction, 6),
            "funds": round(sizing.funds, 2),
            "margin_usdt": round(sizing.margin, 6),
            "opened_ms": _now_ms(),
            "entry_order_id": "", "trade_id": None,
            "tp_pct": float(self.cfg["exits"]["tp_pct"]),
            "max_hold_hours": float(self.cfg["exits"]["max_hold_hours"]),
        })

        try:
            order = ex.place_market(self.client, self.sym, side, sizing.contracts,
                                    tag="entry")
        except ex.AmbiguousSubmission as amb:
            # [MONEY] THE ORDER MAY HAVE LANDED. Kraken exposes no client order id through
            # ccxt, so a timeout after an accepted submit is indistinguishable from a
            # failure — and re-submitting is how one entry becomes two. Ask the only source
            # that cannot lie: the position itself.
            log.critical("entry submission AMBIGUOUS (%s) — checking the venue, NOT resubmitting",
                         amb)
            pos_now = ex.open_position(self.client, self.sym)
            if pos_now.get("error"):
                self.notify(f"💀 Mel — CRITICAL: entry submit was not confirmed ({amb}) and "
                            f"the position cannot be read ({pos_now['error']}). NOT "
                            f"resubmitting; MANUAL CHECK REQUIRED on {self.raw_sym}.")
                return {"status": "error", "message": "ambiguous submit, venue unreadable"}
            if not pos_now.get("active") or float(pos_now.get("contracts") or 0) <= 0:
                log.warning("ambiguous entry submit but the venue holds no position — "
                            "treating it as not landed")
                self.state.clear_flag("entry_inflight")
                return {"status": "error", "message": f"submit not confirmed: {amb}"}
            log.critical("ambiguous entry submit: the venue DOES hold %.4f %s — adopting the "
                         "fill instead of sending a second order",
                         float(pos_now.get("contracts") or 0), pos_now.get("side"))
            self.notify(f"⚠️ Mel — entry submit timed out, but the venue holds "
                        f"{float(pos_now.get('contracts') or 0):g} "
                        f"{str(pos_now.get('side')).upper()} on {self.raw_sym}. Adopting it "
                        f"and protecting it rather than sending a second order.")
            # [MONEY] RETAIN the reservation through the adoption (Terra round 3). Clearing it
            # here — before the adoption has durably recorded state.open + protection — left a
            # live Kraken position with NO state and NO reservation if the adoption failed or the
            # process died between the two lines, so the next tick could treat the bot as flat
            # and open a second position. reconcile_entry_inflight() re-reads the venue, records
            # the position with THIS entry's own SL/TP (not a config-floor stop), and retires the
            # reservation only once state.open is durable — or the venue proves flat. If it
            # raises, the reservation is still set (it is never cleared before this point).
            try:
                res = self.reconcile_entry_inflight()
            except Exception as exc2:
                log.exception("ambiguous-submit adoption failed: %s", exc2)
                return {"status": "error",
                        "message": "ambiguous entry: adoption failed, reserved for recovery"}
            if res.get("status") in ("blocked", "unprotectable"):
                return {"status": "error",
                        "message": "ambiguous entry: adoption is reserved for recovery"}
            if res.get("status") == "recovered":
                # try_enter's established contract for this branch is "adopted" (the fill was
                # adopted into tracked state); keep that wording for callers while the work moves
                # to reconcile_entry_inflight(), which uses this entry's own SL/TP.
                return {"status": "adopted", "contracts": res.get("contracts"),
                        "direction": res.get("direction")}
            return res
        except ex.ExchangeError as exc:
            # A venue REJECTION (size below the minimum, insufficient margin, a bad
            # precision) is not a network blip: nothing was placed and nothing must be
            # recorded. Kraken does not publish its minimum order size through the API, so
            # this is the path that reports it — loudly, with the venue's own words, and
            # without a traceback in the cron log.
            log.error("entry rejected by the venue: %s", exc)
            self.notify(f"⚠️ Mel — entry REJECTED by Kraken for {sizing.contracts:g} "
                        f"{self.raw_sym} ({sizing.notional:,.2f} notional): {exc}. "
                        f"Nothing was placed.")
            self.state.clear_flag("entry_inflight")
            return {"status": "error", "message": f"entry rejected: {exc}"}
        # [MONEY] Never read the fill off the order response alone: KuCoin returns
        # filled=None for a market order that did fill. resolve_fill falls back to a
        # re-read, then to the exchange's own position record.
        filled, entry_price = ex.resolve_fill(self.client, self.sym, order,
                                              expected=sizing.contracts)
        # [MONEY] `filled < 1` was Jack's test, and it is WRONG on Kraken: ETH/USD:USD
        # has contract_size 1.0, so a correct order is a FRACTION of a contract (0.277
        # ETH). Carrying that comparison over made the engine treat every good fill as
        # unresolvable and force-flatten it — a bot that opens and instantly closes
        # position after position. The test is positivity, not one whole contract.
        if filled <= 0 or entry_price <= 0:
            # Nothing coherent to protect; flatten whatever is ACTUALLY held and do not
            # record a phantom entry. Sizing the abort off `filled` would leave the real
            # remainder running unprotected.
            log.critical("entry fill unresolvable (filled=%s px=%s) — flattening live position",
                         filled, entry_price)
            flat = ex.flatten_all(self.client, self.sym)
            if not flat.get("flat"):
                self.notify(f"💀 Mel — CRITICAL: entry unconfirmed AND "
                            f"{flat.get('residual')} contracts still open on "
                            f"{self.raw_sym}. MANUAL ACTION REQUIRED.")
                # residual remains: keep the reservation so the next tick reconciles it
            else:
                self.state.clear_flag("entry_inflight")
            return {"status": "error", "message": "unusable entry fill",
                    "flatten": flat}

        # ── [MONEY] Record the exposure BEFORE the stop is submitted ──────────────────────────
        # The old order was fill -> SL -> DB -> TP -> set_open, so when the SL failed and the
        # force-close could not be confirmed, state.open was still null and the venue-held
        # position had NO record for the per-tick path (protection re-verification, hold cap,
        # orphan adoption) to manage — naked AND untracked. Writing the record first means every
        # failure after this point leaves a tracked position: the order ids start empty and are
        # filled in below, so a record with an empty sl_order_id is exactly the signal manage()
        # reads to re-place protection, and it already holds the ledger row and the immutable
        # entry terms (entry_price / contracts / leverage / margin) a later exit is costed on.
        trade_id = None
        try:
            trade_id = self.db.record_entry(
                symbol=self.raw_sym, direction="long" if side == "buy" else "short",
                entry_price=entry_price, contracts=filled,
                leverage=int(self.cfg["sizing"]["leverage"]), sl_price=sl_price,
                # ONE TP: it lives in tp1_price (the column every consumer already reads),
                # and tp2_price is written as 0.0 because there is no second level.
                tp1_price=tp_price, tp2_price=0.0,
                notional=entry_price * filled * self.spec["contract_size"],
                sl_pct=abs(entry_price - sl_price) / entry_price,
                trigger="scoreboard",
                # The score that justified the entry. "scoreboard" alone says nothing a
                # later review can use: a 5 and a 9 are the same word in that column, and
                # Trip's own history showed high scores performing WORSE (0-for-7 at 7+).
                # Keep the winning side in its own field for querying, and the full
                # breakdown beside it — the losing side's score is what says how close
                # the call was.
                entry_score=float(getattr(decision, "confidence", 0) or 0),
                guard_config={"max_hold_hours": self.cfg["exits"]["max_hold_hours"],
                              "max_margin_fraction": self.cfg["sizing"]["max_margin_fraction"],
                              # The unit the equity caps are measured in: margin deployed /
                              # free collateral at entry, from the executed size.
                              "margin_fraction": round(sizing.margin_fraction, 6),
                              "funds_at_entry": round(sizing.funds, 2),
                              "exits": "single_tp_sl", "tp_pct": self.cfg["exits"]["tp_pct"],
                              "decision": {
                                  "score_long": int(getattr(decision, "score_long", 0) or 0),
                                  "score_short": int(getattr(decision, "score_short", 0) or 0),
                                  "threshold": int(self.cfg["strategy"]["entry_threshold"]),
                                  "reason": str(getattr(decision, "reason", "") or ""),
                              }})
            self.state.set_open({
                "trade_id": trade_id, "direction": "long" if side == "buy" else "short",
                "entry_price": entry_price, "contracts": filled, "remaining": filled,
                "entry_order_id": order.get("id", ""), "opened_ms": _now_ms(),
                "sl_price": sl_price, "sl_order_id": "",
                "tp_price": tp_price, "tp_order_id": "", "tp_filled": False,
                "entry_time": datetime.now(timezone.utc).isoformat(),
                "leverage": int(self.cfg["sizing"]["leverage"]),
                "contract_size": float(self.spec.get("contract_size") or 1.0),
                "margin_usdt": round(sizing.margin, 6),
                "funds_usdt": round(sizing.funds, 2),
            })
        except Exception as exc:
            # [MONEY] The fill LANDED but the ledger/state write failed. The reservation was
            # written BEFORE the market submit, so the position is recoverable: keep it (do not
            # clear), refresh it with the post-fill facts, and let the next tick reconcile it —
            # re-protecting the live position with these exact terms and re-recording the row.
            # Do NOT crash the tick past a durable recovery marker, and do NOT abandon the
            # position to the orphan path (which would substitute config-floor stops for the
            # strategy's own SL/TP).
            self._note_entry_persistence_failure(side, stop_side, filled, entry_price,
                                                 sizing, sl_price, tp_price, trade_id,
                                                 order, exc)
            return {"status": "error",
                    "message": f"entry persistence failed after fill: {exc}"}

        # NOW tracked: a ledger row AND state.open (with an empty sl_order_id) exist. The
        # reservation's tracking duty is done — protection is next, and if THAT fails
        # _force_close() + manage() own the recovery. Retire the reservation here so a
        # failed stop cannot resurrect it.
        self.state.clear_flag("entry_inflight")

        # ── primary SL: confirm before trusting ──
        sl_order = None
        try:
            sl_order = ex.place_stop(self.client, self.sym, stop_side, filled, sl_price, tag="sl")
            ok, why = ex.stop_is_confirmed(
                self.client, self.sym, sl_order.get("id"), side=stop_side,
                min_amount=filled, require_breakeven=False,
                entry=entry_price, trigger=sl_price)
            if not ok:
                raise ExchangeError(f"SL not confirmed: {why}")
        except ex.AmbiguousSubmission as amb:
            # The stop may exist and we have NO id for it. Record the possibility rather
            # than walking away blind: the expected size and trigger are enough for the next
            # tick to look for it, and the flag blocks new entries until the book is clear.
            log.critical("SL submission AMBIGUOUS (%s) — a stop matching %.4f @ %.4f may be "
                         "resting untracked", amb, filled, sl_price)
            self.state.set_flag("possible_orphan_stop",
                                {"contracts": filled, "trigger": sl_price,
                                 "side": stop_side, "at": _now_ms()})
            self.notify(f"💀 Mel — CRITICAL: the stop for a new {self.raw_sym} position may "
                        f"have been accepted without a reply, and this bot has NO id for it. "
                        f"It may be resting untracked ({filled:g} @ {sl_price:,.2f}). New "
                        f"entries are blocked until the book is reconciled — run `--sync`.")
            return self._force_close(side, stop_side, filled, entry_price, decision,
                                     sl_price, str(amb), sl_order_id="", sizing=sizing)
        except Exception as exc:
            log.critical("SL placement/confirmation failed (%s) — force-closing", exc)
            return self._force_close(side, stop_side, filled, entry_price, decision,
                                     sl_price, str(exc),
                                     sl_order_id=str((sl_order or {}).get("id") or ""),
                                     sizing=sizing)

        # ── the single TP, at FULL size ──
        # Full size is what makes it a single TP and not a partial: the whole position
        # leaves at +4%. The SL placed above already covers the full position too, so if
        # this TP fails to place the position is still protected on the loss side.
        tp_id = ""
        tp_ambiguous = None
        try:
            # Counting on a timeout: if a TP is somehow already resting (a retried tick),
            # adopt it instead of stacking a second one.
            # NO STRUCTURAL ADOPTION. An order that merely matches side/size/price is not
            # ours: adopting its id means this bot later cancels something it never placed
            # (Terra round-4 finding 5). If one is resting anyway, say so and do not stack a
            # second identical reduce-only order — a duplicate is its own hazard, because the
            # spare can close a LATER position.
            existing = ex.find_matching_order(self.client, self.sym, side=stop_side,
                                              amount=filled, price=tp_price, kind="limit")
            if existing:
                log.warning("a matching limit order already rests (%s) at the TP price — NOT "
                            "adopting it and NOT stacking a duplicate",
                            str(existing.get("id") or "")[:12])
                self.notify(f"⚠️ Mel — a limit order already rests at this position's "
                            f"take-profit price but was not placed by this bot, so it was NOT "
                            f"adopted (its id would later be cancelled as ours) and no "
                            f"duplicate was placed. This position has NO tracked "
                            f"take-profit; `--sync` shows the book.")
                self.state.set_flag("tp_state_uncertain", True)
            else:
                try:
                    o = ex.place_limit(self.client, self.sym, stop_side, filled, tp_price,
                                       tag="tp")
                    tp_id = str(o.get("id") or "")
                except ex.AmbiguousSubmission as amb:
                    # It may be resting. Do NOT let a later tick place a second full-size
                    # reduce-only TP next to a survivor.
                    tp_ambiguous = {"price": float(tp_price), "contracts": float(filled),
                                    "at": _now_ms(), "where": "tp", "attempts": 0}
                    log.critical("TP submission AMBIGUOUS (%s) — no duplicate will be sent",
                                 amb)
                    self.notify(f"⚠️ Mel — the take-profit for the new {self.raw_sym} "
                                f"position may have been accepted without a reply ({amb}). "
                                f"It will NOT be re-sent on a guess: a duplicate reduce-only "
                                f"limit could close a LATER position. The SL covers the loss "
                                f"side; `--sync` shows the book.")
        except Exception as exc:
            log.warning("TP placement failed: %s — position is protected by its SL only", exc)
            self.notify(f"⚠️ Mel — TP could not be placed ({exc}). The position is "
                        f"protected by its SL; the next tick retries the TP.")

        if tp_ambiguous:
            self.state.set_flag("tp_submit_ambiguous", tp_ambiguous)
        self.state.set_open({
            "trade_id": trade_id, "direction": "long" if side == "buy" else "short",
            "entry_price": entry_price, "contracts": filled, "remaining": filled,
            "entry_order_id": order.get("id", ""), "opened_ms": _now_ms(),
            # [MONEY] Read back by _reconcile_closed to tell a stop-out from a TP exit, and
            # used as the last-resort exit price at the recovery bound. It went missing when
            # this dict was rewritten, so a stop-out whose order id could not be attributed
            # was recorded at the TAKE-PROFIT price — a fabricated win.
            "sl_price": sl_price, "sl_order_id": sl_order.get("id", ""),
            "tp_price": tp_price, "tp_order_id": tp_id, "tp_filled": False,
            "entry_time": datetime.now(timezone.utc).isoformat(),
            # [MONEY] The terms this trade was OPENED under, frozen in state. _finish()
            # used to re-read leverage from config, so raising leverage after entry
            # silently lowered the recorded ROE at exit and weakened the equity-based loss
            # caps (Terra finding 7). The exit is costed with the entry's own numbers.
            "leverage": int(self.cfg["sizing"]["leverage"]),
            "contract_size": float(self.spec.get("contract_size") or 1.0),
            "margin_usdt": round(sizing.margin, 6),
            "funds_usdt": round(sizing.funds, 2),
        })
        self._remember_ids(sl_order.get("id", ""), tp_id)
        self._learn_stop_visibility()

        self.notify(f"🐝 Mel — ENTRY {decision.action} {filled:g} {self.raw_sym} @ "
                    f"{entry_price:,.2f}\nSL {sl_price:,.2f} | TP {tp_price:,.2f} (+4%)\n"
                    f"score {decision.confidence} | margin ${sizing.margin:,.2f} "
                    f"({sizing.scale:.2f}x)")
        return {"status": "ok", "trade_id": trade_id, "entry": entry_price,
                "contracts": filled, "sl": sl_price, "tp": tp_price}

    # ── force close (SL could not be established) ────────────────────────────
    def _force_close(self, side, stop_side, filled, entry_price, decision, sl_price, why,
                     sl_order_id: str = "", sizing=None):
        """Flatten, then record ONLY if flat is confirmed.

        Writes nothing when the close cannot be verified: a 'closed' row for a
        position that may still be open would make the guards act on invented P&L.
        """
        close_ok, close_order = False, None
        recorded, px, tid = False, 0.0, None
        settle: dict = {}
        # [MONEY] try_enter wrote the entry row BEFORE the stop was submitted, so a force-close —
        # confirmed or not — belongs to THAT row, never a fresh one. Reading the trade id here
        # (rather than only on the confirmed path) is what lets the queued pending_close and the
        # settlement reference the same lifecycle instead of a second row for the same contracts.
        tid = (self.state.get_open() or {}).get("trade_id") or None
        try:
            close_order = ex.place_market(self.client, self.sym, stop_side, filled,
                                          reduce_only=True, tag="forceclose")
            pos = ex.open_position(self.client, self.sym)
            if pos.get("error") or (pos.get("active") and pos.get("contracts", 0) > 0):
                self.notify(f"💀 Mel — CRITICAL: SL failed and close left "
                            f"{pos.get('contracts', '?')} contracts open. MANUAL INTERVENTION REQUIRED")
            else:
                close_ok = True
        except ex.AmbiguousSubmission as amb:
            # [MONEY] The close may have FILLED. `state.clear()` here would abandon the trade:
            # no lifecycle row, no retry record, nothing for the guards. Ask the venue.
            live = ex.open_position(self.client, self.sym)
            if live.get("error"):
                self.notify(f"💀 Mel — CRITICAL: close submission was not confirmed ({amb}) "
                            f"and the position cannot be read ({live['error']}). NOT assuming "
                            f"anything; state kept. MANUAL CHECK REQUIRED on {self.raw_sym}.")
                # [MONEY] Fall through to the queue (round-12 F2): an unconfirmed close whose
                # position cannot even be read must still reach the protection-failure breaker
                # and block entries. Returning here left sl_failure_streak blind.
            elif live.get("active"):
                log.critical("ambiguous close but the venue still holds %.4f — retrying the "
                             "close ONCE now that the position is confirmed open",
                             float(live.get("contracts") or 0))
                try:
                    close_order = ex.place_market(self.client, self.sym, stop_side,
                                                  float(live.get("contracts") or filled),
                                                  reduce_only=True, tag="forceclose-retry")
                except Exception as exc2:
                    self.notify(f"💀 Mel — CRITICAL: close could not be confirmed and the "
                                f"retry failed ({exc2}). Exposure remains; MANUAL "
                                f"INTERVENTION REQUIRED on {self.raw_sym}.")
                    # [MONEY] Fall through to the queue (round-12 F2): the failure must be
                    # recorded so the breaker fires and entries stay blocked.
                else:
                    # [MONEY] ACCEPTANCE IS NOT A FILL. This retry used to set close_ok the
                    # moment the venue took the order, and everything after it — retiring
                    # protection, clearing state, writing the row — assumed the position was
                    # gone. An accepted-but-unfilled (or partially filled) retry would then
                    # leave live exposure with nothing managing it.
                    after = ex.open_position(self.client, self.sym)
                    if after.get("error") or after.get("active"):
                        self.notify(f"💀 Mel — CRITICAL: the retried close was accepted but "
                                    f"the venue still shows "
                                    f"{after.get('contracts', '?') if not after.get('error') else 'an unreadable position'} "
                                    f"on {self.raw_sym}. NOT clearing state; MANUAL CHECK "
                                    f"REQUIRED.")
                        # [MONEY] Fall through to the queue (round-12 F2).
                    else:
                        close_ok, px = True, self._fill_price(close_order) or px
            else:
                log.warning("ambiguous close: the venue is FLAT, so the close landed — "
                            "recording it normally")
                close_ok = True
        except Exception as exc:
            self.notify(f"💀 Mel — CRITICAL: SL failed AND close failed ({exc}). "
                        f"Position may be naked. MANUAL INTERVENTION REQUIRED")

        if close_ok:
            # A stop may still have LANDED despite the error we saw (a timeout is not a
            # rejection). Retire only the id we actually submitted — an earlier version
            # swept the whole symbol here, which would cancel an order this bot never
            # placed and remember it for cancellation on the next tick.
            self._retire_protection({}, "an aborted entry",
                                    known_ids=[sl_order_id] if sl_order_id else [])
            # The exchange refused the stop, so the entry was abandoned and flattened.
            # That is a real event on the money path and it is logged nowhere else —
            # without this the next thing heard about it is a missing row.
            self.notify(f"⚠️ Mel — entry ABORTED on {self.raw_sym}: the stop could not "
                        f"be placed ({why}). The position was flattened to stay safe and "
                        f"no new position was opened.")

        if close_ok and close_order:
            px = self._fill_price(close_order)
            if px <= 0:
                self.notify("💀 Mel — CRITICAL: force-close confirmed flat but its fill "
                            "price could not be established, so it was NOT recorded. "
                            "Guard rails cannot see this trade.")
            else:
                try:
                    # reconcile against the exchange like any other close
                    settle = settlement.settled_close(
                        self.client, self.sym,
                        close_order_id=(close_order or {}).get("id", ""),
                        entry_price=entry_price, contracts=filled,
                        fallback_price=px, opened_ms=None)
                    px = float(settle.get("exit_price") or px)
                    fee = float(settle.get("fee_usdt") or 0.0)
                    fee_estimated = bool(settle.get("fee_estimated", True))
                    source = str(settle.get("settlement_source") or "provisional")
                    settle_id = str(settle.get("exchange_settlement_id") or "")
                    if fee <= 0 and settle.get("pnl_net") is None:
                        # Quantity, not contracts: Kraken ETH has contract_size 1.0 (the
                        # amount IS ETH) while KuCoin's is 0.01. Using `filled` raw would
                        # misprice the fee by that factor.
                        csize_f = float(self.spec.get("contract_size") or 1.0) or 1.0
                        fee = round((entry_price + px) * filled * csize_f * 0.0005, 6)

                    fx = 1.0 if side == "buy" else -1.0
                    csize_f = float(self.spec.get("contract_size") or 1.0) or 1.0
                    notional = entry_price * filled * csize_f
                    lev = int(self.cfg["sizing"]["leverage"])
                    margin = notional / lev if lev else notional
                    if settle.get("pnl_net") is not None:
                        pnl = float(settle["pnl_net"])
                    else:
                        pnl = (px - entry_price) * filled * csize_f * fx - fee

                    # The margin fraction this size actually deployed, so the equity caps
                    # see the real deployment rather than the fail-closed 1.0 fallback.
                    mf = float(getattr(sizing, "margin_fraction", 0.0) or 0.0)
                    # [MONEY] Reuse the row try_enter created before the stop: creating a fresh
                    # 'aborted' row here would leave that pre-existing scoreboard row OPEN forever
                    # and double-count the one trade. The exit_reason below carries the failure.
                    if not tid:
                        # Defensive: the pre-SL row was never written (an entry row that failed to
                        # land). Fall back to the historical aborted row so the loss stays visible.
                        tid = self.db.record_entry(
                            symbol=self.raw_sym, direction="long" if side == "buy" else "short",
                            entry_price=entry_price, contracts=filled, leverage=lev,
                            sl_price=sl_price, tp1_price=decision.tp, tp2_price=0.0,
                            notional=notional, sl_pct=abs(entry_price - sl_price) / entry_price,
                            trigger="sl_failure_forceclose",
                            guard_config={"margin_fraction": round(mf, 6),
                                          "funds_at_entry": round(float(getattr(sizing, "funds", 0.0) or 0.0), 2),
                                          "aborted": True})
                    if not tid:
                        raise ExchangeError("no trade id")
                    # The force-close (aborted-entry) path records P&L without the account-log
                    # cross-check that _finish runs — extend it here so an aborted entry's P&L
                    # is checked against the venue's own realized figure too (Paul r4 follow-up).
                    self._alert_on_pnl_gap(settle, px, entry_price, filled * csize_f, fx,
                                           notional, tid, source, pnl)
                    if not self.db.record_exit(tid, exit_price=px, pnl_net=round(pnl, 4),
                                               pnl_pct=round(pnl / margin * 100, 4) if margin else 0.0,
                                               exit_reason="sl_placement_failed",
                                               fee_usdt=round(fee, 6), fee_estimated=fee_estimated,
                                               settlement_source=source,
                                               exchange_settlement_id=settle_id):
                        raise ExchangeError("record_exit returned False")
                    recorded = True
                except Exception as rec_e:
                    self.notify(f"💀 Mel — CRITICAL: force-close was NOT recorded ({rec_e}). "
                                f"Guard rails cannot see this trade.")
        # [MONEY] Clearing state alone would erase a REAL loss from every guard rail
        # (trade count, cooldown, the loss streak, the daily cap, the drawdown halt) — the
        # exposure is gone but the money is not. A pending record is left behind for the
        # next tick to retry, exactly like a normal close that failed to write.
        if not recorded:
            # [MONEY] It IS queued even when `px` is 0, but flagged `needs_price`: a closed
            # trade must never be invisible to the guard rails (Trip's churn was invisible
            # because the failure path returned before writing a row). What must never happen
            # is writing the 0 AS a price — that fabricates a loss of the whole notional
            # (≈ -200% ROE on 2x), which trips the daily-loss and drawdown caps and blocks
            # entries permanently. `retry_pending_close` prices it from evidence or from a
            # level, and flags the row when it has to guess.
            # THE ROW ID, THE NUMBERS AND NOTHING THAT CAN BE RECOMPUTED. An earlier version
            # queued `trade_id: None`, so the retry created a SECOND entry row and closed
            # that one — leaving the real entry `open` forever and writing a P&L computed
            # from scratch. The queue now carries the identity of the row that already
            # exists plus the settlement the caller derived, so the retry completes THIS
            # lifecycle instead of inventing another.
            self._queue_pending_close({
                "trade_id": tid,
                "entry_price": entry_price, "contracts": filled,
                # px may be 0 (filled but unpriced). It is queued as a PRICE TO FIND, never as
                # a price: `retry_pending_close` prices it from evidence or from a level and
                # flags the row, because writing 0 fabricates a loss of the whole notional.
                "exit_price": px, "needs_price": float(px or 0.0) <= 0,
                # [MONEY] True when the venue could NOT confirm the close (position still
                # active, or unreadable). The retry must then prove flatness before it writes a
                # closed row; the row itself stays queued so the protection failure is visible
                # to the breaker instead of vanishing with the position.
                "close_unverified": not close_ok,
                "close_order_id": str((close_order or {}).get("id") or ""),
                "reason": "sl_placement_failed",
                "side": "buy" if side == "buy" else "sell",
                "sl_price": sl_price, "tp_price": decision.tp,
                "leverage": int(self.cfg["sizing"]["leverage"]),
                "margin_fraction": float(getattr(sizing, "margin_fraction", 0.0) or 0.0),
                "fee_usdt": float(settle.get("fee_usdt") or 0.0),
                "fee_estimated": bool(settle.get("fee_estimated", True)),
                "settlement_source": str(settle.get("settlement_source") or "provisional"),
                "exchange_settlement_id": str(settle.get("exchange_settlement_id") or ""),
                "note": f"force-close after: {why}"})
            self.notify("🧾 Mel — the force-closed trade could not be written to the "
                        "ledger; it is queued and the next tick will retry, so the loss "
                        "stays visible to the guard rails."
                        + ("" if close_ok else " The close itself was NOT confirmed by the "
                                              "venue — it may not have filled."))
        if not recorded and not close_ok:
            # [MONEY] Do NOT clear. The close is unproven, so the position may still be live and
            # NAKED (its stop never placed). The position record was written BEFORE the stop was
            # submitted (try_enter), so state.open IS set here: the next tick's manage() re-verifies
            # and re-places protection, and the queued close keeps the failure visible to the guard
            # rails. Entries stay blocked by the queued close itself (run_bot refuses to enter while
            # `pending_close` is non-empty) and by the unverified-settlement guard once the row
            # lands, so keeping the state cannot produce a second entry.
            self.notify("🛑 Mel — the force-close could NOT be confirmed, so the position may "
                        "still be OPEN on the venue with NO verified stop. A position record was "
                        "written before the stop was submitted, so the next tick re-verifies and "
                        "re-places protection; the failed close stays queued in the ledger and new "
                        "entries stay blocked until the venue confirms flat. If the venue still "
                        "shows the position next tick, inspect it by hand.")
        else:
            self.state.clear()
        return {"status": "error", "message": f"SL failed: {why}"}

    def _queue_pending_close(self, record: dict) -> None:
        """Remember a close that could not be written, so it is retried, not lost."""
        try:
            cur = self.state.get_flag("pending_close") or []
            cur.append(record)
            self.state.set_flag("pending_close", cur)
        except Exception as exc:
            log.critical("could not persist a pending close (%s) — the loss may be "
                         "invisible to the guard rails: %s", exc, record)

    def _consume_unverified_pending_close(self) -> bool:
        """Drop queued UNVERIFIED force-closes that an orphan adoption supersedes.

        A force-close whose fill could not be confirmed queues a `close_unverified` record with
        no trade id, so the failure stays visible and entries stay blocked. When the orphan
        guard then ADOPTS that still-live position, the adoption re-books it under a tracked row
        whose exit is recorded through the normal path. Leaving the queued record in place would
        make `retry_pending_close` later invent a SECOND lifecycle row for the same contracts —
        double-counting the trade and its P&L (round-12 F1). Adoption consumes it instead.
        """
        queued = self.state.get_flag("pending_close") or []
        kept = [r for r in queued if not r.get("close_unverified")]
        consumed = len(kept) != len(queued)
        if consumed:
            self.state.set_flag("pending_close", kept)
            log.warning("orphan adoption superseded %d unverified pending close(s)",
                        len(queued) - len(kept))
        return consumed

    def retry_pending_close(self) -> dict:
        """Try to write queued closes. Called at the top of every tick.

        Idempotent by construction: an entry is removed only once its row is written.
        """
        queued = self.state.get_flag("pending_close") or []
        if not queued:
            return {"status": "none"}
        left, written = [], 0
        for rec in queued:
            if rec.get("close_unverified"):
                # [MONEY] Only PROVEN flatness closes an unverified force-close. A closed row
                # for a position that is still open would make every guard act on invented P&L
                # (and hide a naked position), so the record stays queued and the alert says
                # what the venue reports. This is the same rule as `--clear-dangling`: a failure
                # to look is not evidence of anything.
                live = ex.open_position(self.client, self.sym)
                if live.get("error") or (live.get("active")
                                         and float(live.get("contracts") or 0) > 0):
                    why_live = ("the position cannot be read"
                                if live.get("error")
                                else f"the position is still open ({live.get('contracts')} "
                                     f"contracts)")
                    rec["needs_operator"] = True
                    left.append(rec)
                    alert_once(self.state, f"unverified_close:{rec.get('trade_id')}",
                               self.notify,
                               f"🛑 Mel — the force-close for trade #{rec.get('trade_id')} is "
                               f"still UNCONFIRMED: {why_live}. No closed row is written while a "
                               f"position may be live (it would invent P&L and hide a naked "
                               f"position). Entries stay blocked; retried every tick.")
                    continue
            if float(rec.get("exit_price") or 0.0) <= 0:
                # A queued 0 is not a price: writing it fabricates a loss of the WHOLE
                # notional (≈ -200% ROE on 2x), which trips the daily-loss and drawdown caps
                # and blocks entries for good. Try to price it honestly instead: the venue's
                # own fills first, then the level the position was protected at, flagged as
                # unverified. Only if neither exists does it stay queued for an operator.
                priced = False
                try:
                    st = settlement.settled_close(
                        self.client, self.sym,
                        close_order_id=str(rec.get("close_order_id") or ""),
                        entry_order_id=str(rec.get("entry_order_id") or ""),
                        entry_price=float(rec.get("entry_price") or 0.0),
                        contracts=float(rec.get("contracts") or 0.0),
                        contract_size=float(rec.get("contract_size") or 1.0),
                        fallback_price=float(rec.get("sl_price") or 0.0),
                        opened_ms=int(rec.get("opened_ms") or 0) or None)
                except Exception as exc:
                    st, priced = {}, False
                    log.warning("could not price the queued close for #%s: %s",
                                rec.get("trade_id"), exc)
                else:
                    if float(st.get("exit_price") or 0) > 0:
                        rec["exit_price"] = float(st["exit_price"])
                        rec["fee_usdt"] = float(st.get("fee_usdt") or 0.0)
                        rec["fee_estimated"] = bool(st.get("fee_estimated", True))
                        rec["settlement_source"] = str(st.get("settlement_source")
                                                       or "provisional")
                        priced = True
                if not priced:
                    lvl = float(rec.get("sl_price") or rec.get("tp_price") or 0.0)
                    if lvl > 0:
                        rec["exit_price"] = lvl
                        rec["reason"] = "unverified_exit"
                        rec["settlement_source"] = "level_fallback"
                        rec["fee_estimated"] = True
                        priced = True
                        self.notify(f"⚠️ Mel — the force-closed trade #{rec.get('trade_id')} "
                                    f"had no fill evidence, so it was recorded from its "
                                    f"protective level ({lvl:,.2f}) and flagged "
                                    f"`level_fallback`. New entries pause until it is "
                                    f"reconciled (`tools/reconcile_settlement.py --list`).")
                if not priced:
                    rec["needs_operator"] = True
                    left.append(rec)
                    alert_once(self.state, f"bad_pending_close:{rec.get('trade_id')}",
                               self.notify,
                               f"🛑 Mel — a queued close for trade #{rec.get('trade_id')} has "
                               f"no price and no level to price it from, so it was NOT "
                               f"written (writing it would fabricate a full-notional loss).")
                    continue
            try:
                created = False
                tid = rec.get("trade_id")
                if not tid:
                    # No row was ever created for this close (the entry write itself
                    # failed), so this is the one case where creating a lifecycle is right.
                    tid = self.db.record_entry(
                        symbol=self.raw_sym,
                        direction=("long" if str(rec.get("side") or "sell") == "buy"
                                   else "short"),
                        entry_price=float(rec["entry_price"]), contracts=float(rec["contracts"]),
                        leverage=int(rec.get("leverage") or 1),
                        sl_price=float(rec.get("sl_price") or 0.0),
                        tp1_price=float(rec.get("tp_price") or 0.0), tp2_price=0.0,
                        notional=float(rec["entry_price"]) * float(rec["contracts"])
                        * float(self.spec.get("contract_size") or 1.0),
                        trigger="sl_failure_forceclose",
                        guard_config={"margin_fraction": float(rec.get("margin_fraction") or 0.0)})
                    created = True
                if not tid:
                    left.append(rec)
                    continue
                csize = float(self.spec.get("contract_size") or 1.0) or 1.0
                lev = int(rec.get("leverage") or 1) or 1
                notional = float(rec["entry_price"]) * float(rec["contracts"]) * csize
                margin = notional / lev if lev else notional
                fx = 1.0 if str(rec.get("side") or "sell") == "buy" else -1.0
                # The caller's settlement numbers, written as they were derived — a retry
                # that recomputes can drift from what actually happened at the venue.
                fee = float(rec.get("fee_usdt") or 0.0)
                pnl = (float(rec["exit_price"]) - float(rec["entry_price"])) \
                    * float(rec["contracts"]) * csize * fx - fee
                ok = self.db.record_exit(int(tid), exit_price=float(rec["exit_price"]),
                                         pnl_net=round(pnl, 4),
                                         pnl_pct=round((pnl / margin * 100.0) if margin else 0.0, 4),
                                         exit_reason=rec.get("reason") or "sl_placement_failed",
                                         fee_usdt=round(fee, 6), funding_usdt=0.0,
                                         fee_estimated=bool(rec.get("fee_estimated", True)),
                                         settlement_source=str(rec.get("settlement_source")
                                                                 or "provisional"),
                                         exchange_settlement_id=str(
                                             rec.get("exchange_settlement_id") or ""))
                if ok:
                    written += 1
                    if rec.get("close_unverified"):
                        # proven flat and the row has landed: the trade is genuinely over
                        self.state.clear()
                    if created:
                        log.warning("pending close needed a fresh lifecycle row (#%s)", tid)
                else:
                    left.append(rec)
            except Exception as exc:
                log.warning("pending close still unwritable: %s", exc)
                left.append(rec)
        self.state.set_flag("pending_close", left)
        if written:
            self.notify(f"🧾 Mel — {written} previously unwritten close(s) are now in the "
                        f"ledger; the guard rails can see them.")
        if left:
            log.critical("%d close(s) still cannot be written", len(left))
        return {"status": "ok", "written": written, "left": len(left)}

    def _fill_price(self, order: dict) -> float:
        for k in ("average", "price"):
            try:
                v = float((order or {}).get(k) or 0.0)
            except (TypeError, ValueError):
                v = 0.0
            if v > 0:
                return v
        oid = str((order or {}).get("id") or "")
        if not oid:
            return 0.0
        try:
            trades = self.client.fetch_my_trades(self.sym, limit=50) or []
        except Exception:
            return 0.0
        num = den = 0.0
        for t in trades:
            info = t.get("info") or {}
            if str(info.get("orderId") or t.get("order") or "") != oid:
                continue
            try:
                a, p = float(t.get("amount") or 0), float(t.get("price") or 0)
            except (TypeError, ValueError):
                continue
            if a > 0 and p > 0:
                num += a * p
                den += a
        return (num / den) if den > 0 else 0.0

    # ── manage an open position ──────────────────────────────────────────────
    def manage(self, decision) -> dict:
        pos = self.state.get_open()
        if not pos:
            return {"status": "flat"}

        live = ex.open_position(self.client, self.sym)
        # An UNVERIFIABLE read is not "flat". open_position reports a failed fetch as
        # {"active": False, "error": ...} rather than raising, so treating that as "the
        # position closed" would reconcile a LIVE position as closed — state cleared, a
        # phantom close row written, and the bot no longer managing a position that still
        # exists on the exchange. Refuse to act until the exchange can be read.
        if live.get("error"):
            log.error("position read failed (%s) — leaving state untouched", live["error"])
            return {"status": "error", "message": f"exchange unreadable: {live['error']}"}
        price = ex.fetch_price(self.client, self.sym)
        remaining = float(live.get("contracts") or 0.0)

        # ── CONSUME AN IN-FLIGHT CLOSE (Terra round-6 finding 6) ──────────────
        # `close_in_flight` was set by an ambiguous close and NEVER read, so the alert's
        # promise ("the next tick reads the venue and will reconcile or re-close, never both")
        # was not true of the code. This is that read: the venue has now been consulted, so
        # the flag is spent. Flat means the unknown close landed (settle it by the normal
        # path); still-open means it did not (a fresh close is legitimate).
        if self.state.get_flag("close_in_flight"):
            self.state.clear_flag("close_in_flight")
            if not live.get("active") or remaining <= 0:
                log.warning("an ambiguous close is RESOLVED: the venue is flat, so it landed")
            else:
                log.warning("an ambiguous close did NOT land (%s still open) — a fresh close "
                            "may be submitted now", f"{remaining:g}")

        # ── DRY RUN NEVER SUBMITS ─────────────────────────────────────────────
        # [MONEY] A dry run must not be able to trade, and management IS a trading path: a
        # hold-capped position reaches `_close()`, which sends a reduce-only market order.
        # ccxt has NO dry-run mode of its own (the option Jack relied on is inert — see
        # exchange.py), so the guard has to be here. Report what would happen instead.
        if self.dry_run:
            held = _hours_since(pos.get("entry_time"))
            cap = float(self.cfg["exits"]["max_hold_hours"])
            would = "close at the hold cap" if held >= cap else "hold"
            log.info("DRY RUN — would %s (%s open, %.2fh held); no order submitted",
                     would, f"{remaining:g}", held)
            pos["last_price"] = price
            if not would.startswith("close"):
                self.state.set_open(pos)
            return {"status": "dry_run", "would": would, "remaining": remaining,
                    "held_h": round(held, 2), "price": price}

        # position gone from the exchange -> it was stopped or TP'd; reconcile
        if not live.get("active") or remaining <= 0:
            return self._reconcile_closed(pos, price)

        # The exchange is authoritative on size. After a TP fills, the protective stop
        # must be resized to what is ACTUALLY left — sizing it from the original entry
        # size would try to close more than is open. Keep this in state so every
        # downstream protective action uses the true remainder.
        pos["remaining"] = remaining

        # [MONEY] A provisional record (written before the entry stop was submitted) has an empty
        # sl_order_id. The venue has just answered "active", which PROVES the earlier force-close
        # did NOT land — so the queued UNVERIFIED close for this trade is superseded: the position
        # is being managed normally and its exit will be recorded through the normal path. Leaving
        # the record queued would make retry_pending_close invent a SECOND row for the same
        # contracts once the venue turns flat (round-12 F1, same rule as orphan adoption).
        if not pos.get("sl_order_id") and (self.state.get_flag("pending_close") or []):
            if self._consume_unverified_pending_close():
                self.notify("🧾 Mel — a previously unconfirmed force-close was superseded by "
                            "re-protecting the open position; its queued record is dropped so the "
                            "trade is not double-counted.")

        # ── the single TP ──
        # If it filled, the position is gone or nearly gone; the SL is the leftover that
        # must be retired (the venue may already have auto-cancelled it — treat that as
        # success). If the TP is MISSING (a placement that failed at entry), put it back:
        # the SL protects the loss side, but a position with no TP is a strategy that
        # cannot take profit at all.
        if pos.get("tp_order_id"):
            if not pos.get("tp_filled"):
                o = ex.fetch_order(self.client, self.sym, pos["tp_order_id"])
                if o and str(o.get("status") or "").lower() in ("closed", "filled"):
                    pos["tp_filled"] = True
                    self.state.set_open(pos)
                    self.notify(f"🎯 Mel — TP filled @ {float(pos.get('tp_price') or 0):,.2f} "
                                f"(position should now be flat; SL retired on the next read)")
        elif not pos.get("tp_filled"):
            self._ensure_tp(pos, remaining)

        # Position still open with the TP filled and size left over? That is a partial
        # fill, not a design feature: log it loudly, because a single TP is supposed to
        # take the WHOLE position.
        if pos.get("tp_filled") and remaining > 1e-9:
            log.warning("TP reports filled but %s still open — partial TP fill, "
                        "SL still covers the remainder", remaining)
            self.notify(f"⚠️ Mel — TP reported filled but {remaining:g} of the position is "
                        f"still open. The SL still covers it; check the venue.")

        # ── the single SL, re-verified EVERY tick ──
        # [MONEY] This is the Trip failure class. A stop that is accepted at entry can still
        # expire, be cancelled by the venue, or be reduced by a partial fill — and a position
        # whose stop quietly vanished is naked until something else closes it. Confirming once,
        # at entry, cannot see any of that.
        sl_state = self._ensure_sl(pos, remaining)
        if sl_state == "closed":
            # the position closed while we were checking its protection (a stop that filled)
            return self._reconcile_closed(pos, price)
        if sl_state == "failed":
            log.critical("the protective stop is missing and could not be restored — closing")
            return self._close(pos, price, remaining, "sl_missing_unprotected")
        if sl_state == "unknown":
            alert_once(self.state, "sl_unverifiable", self.notify,
                       f"🛑 Mel — the protective stop on the open {self.raw_sym} position "
                       f"cannot be verified right now (the venue read failed) and this bot "
                       f"will NOT stack a duplicate order on a guess. It re-checks every tick. "
                       f"If this persists, inspect the venue by hand.")

        # hold cap
        held = _hours_since(pos.get("entry_time"))
        if held >= float(self.cfg["exits"]["max_hold_hours"]):
            return self._close(pos, price, remaining, "hold_cap")

        pos["last_price"] = price
        self.state.set_open(pos)
        return {"status": "holding", "price": price, "remaining": remaining, "held_h": round(held, 2)}

    def _ensure_sl(self, pos, remaining: float) -> str:
        """Re-verify the protective stop while a position is open; restore it if it is gone.

        Returns "ok" (verified resting) · "replaced" (was gone, restored) · "failed" (gone and
        could not be restored — the caller must fail closed) · "unknown" (cannot tell: do
        nothing, say so, and never stack a duplicate).

        Trip's profits were drained by a stop that was ACCEPTED and then silently expired, so
        entry-time confirmation is not enough: the check has to run every tick, and it has to
        re-place for what is ACTUALLY open (a partially filled stop leaves the rest exposed).
        """
        if not pos:
            return "ok"                      # nothing tracked, nothing to protect
        sl_price = float(pos.get("sl_price") or 0.0)
        side = "sell" if pos["direction"] == "long" else "buy"
        entry = float(pos.get("entry_price") or 0.0)
        if sl_price <= 0 or remaining <= 0 or entry <= 0:
            return "ok"

        oid = str(pos.get("sl_order_id") or "")
        if oid:
            # Ask about the ID. For protecting an open position the asymmetry is the reverse of
            # cancellation: holding a naked position is worse than briefly stacking a spare
            # reduce-only order, so a definitive "the venue has no such order" is acted on.
            verdict, why = ex.fetch_order_state(self.client, self.sym, oid)
            if verdict in ("finished", "missing"):
                # A stop can FILL between the two reads. Ask the position before restoring: if
                # it is gone there is nothing to protect (and manage() will settle it), and if
                # only part of the position left, the remainder is what needs a stop.
                live_now = ex.open_position(self.client, self.sym)
                if not live_now.get("error") and not live_now.get("active"):
                    log.warning("stop %s is %s but the position is CLOSED — nothing to "
                                "protect; settling by the normal path", oid[:12], verdict)
                    return "closed"
                if live_now.get("error"):
                    log.warning("cannot read the position while restoring a stop (%s) — "
                                "restoring for the last known size", live_now["error"])
                elif float(live_now.get("contracts") or 0.0) > 0:
                    remaining = float(live_now["contracts"])
                log.critical("the protective stop %s is NOT RESTING (%s: %s) while %s is open",
                             oid[:12], verdict, why, f"{remaining:g}")
                self.notify(f"🚨 Mel — the protective stop on the open {self.raw_sym} position "
                            f"is NO LONGER RESTING at the venue ({verdict}). Re-placing it now "
                            f"for {remaining:g} @ {sl_price:,.2f}.")
            elif verdict == "open":
                ok, why2 = ex.stop_is_confirmed(
                    self.client, self.sym, oid, side=side, min_amount=remaining,
                    require_breakeven=False, entry=entry, trigger=sl_price)
                if ok:
                    return "ok"
                if ("not found" in why2 or "unreadable" in why2 or "read failed" in why2
                        or "no trigger" in why2):
                    log.warning("cannot confirm the protective stop %s (%s) — not re-placing "
                                "on a guess", oid[:12], why2)
                    return "unknown"
                log.critical("the protective stop EXISTS BUT DOES NOT PROTECT (%s) — "
                             "cancelling it before re-placing", why2)
                self.notify(f"🚨 Mel — the resting stop on the open {self.raw_sym} position "
                            f"failed verification ({why2}). Cancelling it and re-placing.")
                # [MONEY] The bad stop is OURS (we hold its id), so cancel it — placing a new
                # one beside it would leave a reduce-only survivor that no id-scoped cleanup
                # can reach. If the cancel fails, do NOT place a duplicate: try again next tick.
                # `ex.cancel` ANSWERS with a bool (it reads the venue's reply); it does not
                # raise. Treating it as exception-only would sail past a failed cancel and
                # stack a duplicate.
                try:
                    cancelled = ex.cancel(self.client, self.sym, oid)
                except Exception as exc:
                    cancelled = False
                    log.critical("cancel of the invalid stop %s raised: %s", oid[:12], exc)
                if not cancelled:
                    log.critical("could not cancel the invalid stop %s — NOT placing a second "
                                 "stop beside it", oid[:12])
                    self.notify(f"⚠️ Mel — the invalid stop {oid[:12]}… could not be "
                                f"cancelled, so no replacement was placed (a duplicate "
                                f"reduce-only order could close a later position). Retrying "
                                f"next tick.")
                    return "unknown"
            else:
                # The venue could not be asked (timeout, auth, outage). That is NOT evidence
                # the stop is gone, and re-placing here would duplicate a healthy stop.
                log.warning("cannot ask the venue about the protective stop %s (%s) — holding, "
                            "not re-placing on a guess", oid[:12], why)
                return "unknown"
            # It is gone: clear the id FIRST so a failed re-place below cannot leave state
            # claiming a protection that does not exist.
            pos["sl_order_id"] = ""
            self.state.set_open(pos)

        # BOUNDED RESTORES. If the venue keeps dropping the stop we just placed, re-placing
        # forever is not protection — it is Trip's churn with a naked position in between.
        # After `guards.max_sl_restores` attempts on one position, fail closed instead.
        attempts = int(pos.get("sl_restore_attempts") or 0)
        if attempts >= int(self.cfg["guards"].get("max_sl_restores", 2)):
            log.critical("this position has lost its stop %d time(s) — the venue is not "
                         "honouring our protection; closing rather than holding naked",
                         attempts)
            self.notify(f"🚨 Mel — the {self.raw_sym} position has lost its protective stop "
                        f"{attempts} time(s) and could not be kept protected. CLOSING the "
                        f"position rather than holding it unprotected.")
            return "failed"
        pos["sl_restore_attempts"] = attempts + 1
        self.state.set_open(pos)
        try:
            o = ex.place_stop(self.client, self.sym, side, remaining, sl_price,
                              tag="sl-restore")
        except ex.AmbiguousSubmission as amb:
            # The restore may have landed. Fail closed AND leave a durable marker, so a
            # survivor can never be forgotten (it is not in any id we know).
            self._lock_orphan_stop(side, remaining, sl_price, "sl-restore")
            self.notify(f"💀 Mel — CRITICAL: the replacement stop for the open {self.raw_sym} "
                        f"position was not confirmed ({amb}) and no id is known for it. "
                        f"Closing the position rather than trusting it.")
            return "failed"
        except Exception as exc:
            log.critical("could not restore the protective stop (%s)", exc)
            return "failed"
        new_id = str(o.get("id") or "")
        ok, why = ex.stop_is_confirmed(self.client, self.sym, new_id, side=side,
                                       min_amount=remaining, require_breakeven=False,
                                       entry=entry, trigger=sl_price)
        if not ok:
            log.critical("the restored stop did not verify (%s)", why)
            return "failed"
        pos["sl_order_id"] = new_id
        pos["sl_restored_at"] = _now_ms()
        self._remember_ids(new_id)
        self.state.set_open(pos)
        log.warning("protective stop restored: %s for %s @ %.2f", new_id[:12], f"{remaining:g}",
                    sl_price)
        self.notify(f"🛟 Mel — the protective stop was missing and has been RESTORED "
                    f"({remaining:g} @ {sl_price:,.2f}). The position is protected again.")
        return "replaced"

    def _ensure_tp(self, pos, remaining: float) -> None:
        """Re-place the single TP when it is missing (entry-time placement failed).

        Idempotent by construction: the resting book is checked first, so a TP that
        exists but was not recorded cannot be duplicated. Placed at the price the entry
        decision computed, not at the current market — the take-profit is a level from
        the strategy, not a trailing target.
        """
        price = float(pos.get("tp_price") or 0.0)
        if price <= 0 or remaining <= 0:
            return
        side = "sell" if pos["direction"] == "long" else "buy"
        amb = pos.get("tp_submit_ambiguous") or self.state.get_flag("tp_submit_ambiguous")
        if amb:
            # A TP we submitted may already be resting with no id. Replacing it blindly is how
            # a second full-size reduce-only order ends up on the book. Look first — STRICTLY,
            # because the tolerant read reports an outage as an empty book.
            try:
                live = ex.open_orders(self.client, self.sym, strict=True)
            except Exception as exc:
                log.warning("cannot check for the ambiguous TP (%s) — not re-sending", exc)
                return
            if any(str(o.get("id") or "") and abs(ex.order_amount(o) - remaining) <= 1e-6
                   and abs(float(o.get("price") or 0) - price) <= 1e-6 for o in live):
                if not pos.get("tp_state_uncertain"):
                    pos["tp_state_uncertain"] = True
                    self.state.set_open(pos)
                    self.notify(f"⚠️ Mel — the take-profit whose submission went unanswered "
                                f"IS resting at the target price. It is not adopted (no id "
                                f"provenance) and no duplicate was placed.")
                return
            attempts = int(amb.get("attempts") or 0)
            if attempts >= 2:
                if not pos.get("tp_state_uncertain"):
                    pos["tp_state_uncertain"] = True
                    self.state.set_open(pos)
                    alert_once(self.state, "tp_ambiguous_gave_up", self.notify,
                               f"🛑 Mel — this position's take-profit could not be confirmed "
                               f"and will NOT be re-sent again (twice is enough): a duplicate "
                               f"reduce-only limit could close a later position. The SL "
                               f"protects the loss side and the hold cap will exit the trade.")
                return
            amb["attempts"] = attempts + 1
            pos["tp_submit_ambiguous"] = amb
            self.state.set_open(pos)
            log.warning("ambiguous TP: nothing matching is resting; attempt %d/2",
                        attempts + 1)
        try:
            existing = ex.find_matching_order(self.client, self.sym, side=side,
                                              amount=remaining, price=price, kind="limit")
            if existing:
                # Same rule as entry: never adopt. Report once and leave it alone.
                if not pos.get("tp_state_uncertain"):
                    pos["tp_state_uncertain"] = True
                    self.state.set_open(pos)
                    self.notify(f"⚠️ Mel — a limit order already rests at this position's "
                                f"take-profit price and was NOT placed by this bot; it was "
                                f"not adopted and no duplicate was placed.")
                return
            o = ex.place_limit(self.client, self.sym, side, remaining, price, tag="tp")
            pos["tp_order_id"] = str(o.get("id") or "")
            self.state.set_open(pos)
            self.notify(f"🐝 Mel — TP re-placed at {price:,.2f} for {remaining:g} "
                        f"(the entry tick could not place it)")
        except Exception as exc:
            log.warning("TP re-place failed: %s", exc)

    def _reconcile_closed(self, pos, price: float) -> dict:
        """The position vanished from the exchange — a stop or TP took it out."""
        entry = float(pos["entry_price"])
        d = 1.0 if pos["direction"] == "long" else -1.0

        # One of the two protective orders took the position out; the OTHER one is now a
        # dangling reduce-only order sitting on the book. Kraken usually auto-cancels a
        # reduce-only order when the position closes, but "usually" is not a risk control:
        # a stop left behind can close the NEXT position Mel opens. Retiring it is
        # CONFIRMED, not attempted — a residual is persisted and blocks new entries.
        self._retire_protection(pos, "an unattributed close")

        # ASK THE VENUE WHICH ORDER FILLED. Comparing the current ticker against the two
        # levels is a guess: a fast move leaves the price nowhere near the level by the
        # time the tick runs, so a stop-out gets recorded as an unexplained
        # "exchange_close" and the reason a trade died is lost. The orders themselves are
        # the evidence.
        filled_side = self._order_filled_state(pos)
        if filled_side == "tp":
            reason = "tp"
            exit_px = float(pos.get("tp_price") or 0.0)
        elif filled_side == "sl":
            reason = "sl"
            exit_px = float(pos.get("sl_price") or 0.0)
        elif pos.get("sl_price") and (
                (d > 0 and price <= float(pos["sl_price"]) * 1.002) or
                (d < 0 and price >= float(pos["sl_price"]) * 0.998)):
            reason = "sl"
            exit_px = float(pos["sl_price"])
        elif pos.get("tp_filled"):
            reason = "tp"
            exit_px = float(pos.get("tp_price") or 0.0)
        else:
            # The cause is UNKNOWN, so do NOT invent the exit price from the current ticker:
            # the position may have closed much earlier at a different price, and a fabricated
            # exit price is worse than an unpriced row. Pass 0 and let the exchange's own
            # fills/history supply it — _finish refuses to record without a real price.
            reason = "exchange_close"
            exit_px = 0.0
        return self._finish(pos, exit_px, reason)

    def _retire_protection(self, pos, context: str, known_ids=None) -> list:
        """Cancel THIS BOT's protective orders and confirm they are gone.

        [MONEY] A single TP and a single SL both cover the full position, so one of them
        always survives an exit. A surviving reduce-only order can close a LATER position,
        which is why a failed cancel is not a log line: the residual is persisted to state
        and the next tick refuses to open a position until it is gone.
        """
        if known_ids is None:
            known_ids = [str(pos.get(k) or "") for k in ("sl_order_id", "tp_order_id")]
        ids = [i for i in known_ids if i]
        for oid in ids:
            if not ex.cancel(self.client, self.sym, oid):
                log.error("could not cancel %s while retiring protection after %s",
                          oid, context)
        residual: list = []
        try:
            # CHECK BY ID. Ownership is an id, an id is checkable, and unlike listing the
            # trigger orders this works whatever the venue's permissions are.
            # CLEARANCE, not cancellation: an id the venue definitively does not know must not
            # pin this flag for ever (live-reproduced: order_gone() said False for an id that
            # certainly does not exist, because this key cannot list trigger orders).
            residual = [i for i in ids
                        if not ex.order_provably_finished(self.client, self.sym, i)]
            # A listing is still consulted for ORDERS WE DO NOT RECOGNISE — reported, never
            # touched. (On this venue a refused trigger listing means stops may be missing
            # from it; the code never treats absence from a listing as proof.)
            try:
                still = {str(o.get("id") or "") for o in
                         ex.open_orders(self.client, self.sym, strict=True)}
            except Exception as exc:
                log.info("could not list the book after %s (%s) — per-id verification "
                         "stands", context, str(exc)[:80])
                still = set()
            foreign = sorted(i for i in still if i and i not in ids)
            if foreign:
                # NOT OURS. Report it; never cancel or remember it. Cancelling an order we
                # did not place (a manual order, another tool's) is acting outside this
                # bot's remit, and remembering it as "dangling" made the NEXT tick cancel it.
                log.warning("%d resting order(s) on %s are not this bot's: %s",
                            len(foreign), self.raw_sym, ", ".join(foreign[:3]))
                self.notify(f"⚠️ Mel — {len(foreign)} resting order(s) on {self.raw_sym} "
                            f"were NOT placed by this bot and were left untouched. They can "
                            f"still close a future position; run `--sync` to inspect.")
        except Exception as exc:
            log.error("could not verify the book after %s (%s) — assuming the worst and "
                      "keeping the block in place", context, exc)
            residual = ids
        try:
            if residual:
                self.state.set_flag("dangling_orders", residual)
            else:
                self.state.clear_flag("dangling_orders")
        except Exception as exc:
            log.warning("could not persist the dangling-order flag: %s", exc)
        if residual:
            self.notify(f"💀 Mel — CRITICAL: {len(residual)} protective order(s) could not "
                        f"be retired after {context} ({', '.join(residual[:3])}). A "
                        f"reduce-only survivor can close a FUTURE position; new entries are "
                        f"blocked until the book is clear.")
        return residual

    def _lock_orphan_stop(self, side: str, contracts: float, trigger: float,
                          where: str) -> None:
        """Record that a protective stop we submitted may be resting with no id.

        Used by EVERY path that submits a stop (entry, orphan adoption, restoration). One of
        them used to swallow the ambiguity as a generic failure, which left a stop that no
        id-scoped retirement could ever reach — an untracked reduce-only order that can close
        a position opened later.
        """
        self.state.set_flag("possible_orphan_stop",
                            {"contracts": float(contracts), "trigger": float(trigger),
                             "side": str(side), "at": _now_ms(), "where": where})
        log.critical("AMBIGUOUS stop submission (%s): %s %s @ %.4f may be resting untracked",
                     where, side, f"{contracts:g}", trigger)

    def _note_tp_ambiguity(self, pos: dict, price: float, contracts: float,
                           where: str) -> None:
        """Record that a TP submission went unanswered, so it must never be re-sent blindly."""
        pos["tp_submit_ambiguous"] = {"price": float(price), "contracts": float(contracts),
                                      "at": _now_ms(), "where": where, "attempts": 0}
        log.critical("AMBIGUOUS TP submission (%s) for %s @ %.4f — no duplicate will be sent",
                     where, f"{contracts:g}", price)

    def _remember_ids(self, *ids) -> None:
        """Keep a bounded history of the order ids THIS bot placed.

        Ownership is an id, and an id is only useful if it is written down. The history is
        what lets the flat-state gate answer "is the book clear?" by verification instead of
        by trusting a listing this venue cannot promise is complete.
        """
        try:
            hist = [str(x) for x in (self.state.get_flag("placed_ids") or []) if x]
            for i in ids:
                i = str(i or "")
                if i and i not in hist:
                    hist.append(i)
            self.state.set_flag("placed_ids", hist[-40:])
        except Exception as exc:
            log.warning("could not persist the placed-id history: %s", exc)

    def protection_all_clear(self) -> bool:
        """True only when every id this bot ever placed is provably finished."""
        hist = [str(x) for x in (self.state.get_flag("placed_ids") or []) if x]
        if not hist:
            return True
        outstanding = [i for i in hist
                       if not ex.order_provably_finished(self.client, self.sym, i)]
        if outstanding:
            log.warning("%d previously placed order id(s) are not provably gone: %s",
                        len(outstanding), ", ".join(outstanding[:3]))
            return False
        return True

    def reconcile_possible_orphan_stop(self) -> str:
        """Resolve a stop whose submission was ambiguous (we know we sent it, we lack its id).

        This is the ONE case where matching on shape is legitimate: the bot has a record of
        the side, size and trigger it submitted. Finding it means adopting OUR order; not
        finding it while stop visibility is unavailable means the book cannot be cleared, so
        the tick keeps blocking rather than guessing.
        """
        pending = self.state.get_flag("possible_orphan_stop")
        if not pending:
            return "none"
        try:
            cands = [o for o in ex.open_orders(self.client, self.sym)
                     if str(o.get("side") or "").lower() == str(pending.get("side") or "")
                     and ex.order_trigger(o) > 0
                     and abs(ex.order_amount(o) - float(pending.get("contracts") or 0)) <= 1e-6
                     and abs(ex.order_trigger(o) - float(pending.get("trigger") or 0)) <= 1e-6]
        except Exception as exc:
            log.info("cannot search for the ambiguous stop yet (%s)", str(exc)[:80])
            cands = []
        if cands:
            # A CANDIDATE, not a conclusion. Submission intent plus a matching shape does not
            # prove the venue's order is ours — a manual stop with the same side/size/trigger
            # is indistinguishable after a timeout, and recording its id would mean this bot
            # later cancels an order it never placed. It is surfaced for a human decision;
            # `--sync --adopt-orphan-stop` is how an operator declares it ours.
            oid = str(cands[0].get("id") or "")
            pending["candidate_id"] = oid
            pending["candidate_seen_at"] = _now_ms()
            self.state.set_flag("possible_orphan_stop", pending)
            log.critical("a stop matching our ambiguous submission is resting (%s) — it is a "
                         "CANDIDATE, not proven ours; entries stay blocked until it is "
                         "resolved", oid[:12])
            return "candidate"
        if not ex._stops_view_available(self.client, self.sym):
            return "unresolved_blind"
        return "unresolved"

    def _learn_stop_visibility(self) -> None:
        """Record whether the venue's order listing actually shows a resting STOP.

        This venue refuses its trigger-specific order query for the key in use, so the
        question "can a stop be seen at all?" has no documentation answer — only evidence.
        The moment Mel has its own stop resting, the listing is asked. The answer is
        persisted, alerted ONCE, and it is what makes the flat-state gate honest: if stops
        are invisible in the listing, then an empty listing is not proof of an empty book.
        """
        try:
            live = {str(o.get("id") or "") for o in ex.open_orders(self.client, self.sym)}
            pos = self.state.get_open() or {}
            # THE STOP ALONE ANSWERS THIS. A visible limit TP says nothing about whether the
            # listing carries triggered orders, and certifying on it would let the bot claim
            # a completeness it has never observed.
            sl_id = str(pos.get("sl_order_id") or "")
            if not sl_id:
                return
            ours = {sl_id}
            seen = ours & live
            known = self.state.get_flag("stop_listing_covers_triggers")
            covers = bool(seen)
            if known is None or bool(known) != covers:
                self.state.set_flag("stop_listing_covers_triggers", covers)
                if covers:
                    log.info("stop visibility confirmed: our resting stop appears in the "
                             "venue's order listing")
                else:
                    log.warning("stop visibility LIMITED: our resting SL does NOT appear in "
                                "the venue's order listing")
                    self.notify("🔎 Mel — the venue's order listing does NOT show our own "
                                "resting stop. Empty listings therefore cannot prove the book "
                                "is clear; protection is verified by id from here on.")
        except Exception as exc:
            log.info("could not learn stop visibility yet (%s)", str(exc)[:80])

    def _order_filled_state(self, pos) -> str:
        """Which of the two protective orders the venue says FILLED: 'tp', 'sl' or ''.

        Empty means neither could be confirmed filled (usually a still-resting order, or
        a venue that will not say) — the caller then falls back to the price heuristic.
        """
        for key, tag in (("tp_order_id", "tp"), ("sl_order_id", "sl")):
            oid = pos.get(key)
            if not oid:
                continue
            o = ex.fetch_order(self.client, self.sym, oid)
            if o and str(o.get("status") or "").lower() in ("closed", "filled"):
                return tag
        return ""

    def _filled_amount(self, order: dict) -> float:
        """How much the venue says actually filled on an order."""
        o = order or {}
        info = o.get("info") or {}
        for v in (o.get("filled"), o.get("dealSize"),
                  info.get("dealSize"), info.get("filledSize")):
            if v is None:
                continue
            try:
                n = float(v)
                if n > 0:
                    return n
            except (TypeError, ValueError):
                continue
        return 0.0

    def _confirm_close(self, close_order) -> tuple[float, bool, float, str]:
        """Verify a submitted close ACTUALLY happened, and get its real fill price.

        place_market() returns the venue's ACCEPTANCE of the request, not a fill: the order
        can still be rejected, left unfilled, or only partly filled. Treating acceptance as
        "filled" retired every protective order and wrote a closed row for a position that was
        still open — the bare-position failure mode, plus a trade record describing something
        that never happened.

        Returns (fill_price, flat_confirmed, filled_amount, why). fill_price is 0.0 when no
        fill price could be established; the caller must NOT substitute a ticker price.
        """
        oid = str((close_order or {}).get("id") or "")
        fill = self._fill_price(close_order)
        filled = self._filled_amount(close_order)
        status = ""
        if oid:
            o = ex.fetch_order(self.client, self.sym, oid) or {}
            status = str(o.get("status") or "")
            fill = self._fill_price(o) or fill
            filled = self._filled_amount(o) or filled

        # The decisive check is the POSITION, not the order: the exchange's own answer to
        # "is there still exposure?" is the only thing that proves the close happened.
        live = ex.open_position(self.client, self.sym)
        if live.get("error"):
            return 0.0, False, filled, f"cannot verify flatness ({live['error']})"
        left = float(live.get("contracts") or 0.0)
        if live.get("active") and left > 0:
            return fill, False, filled, (f"still {left:g} contracts open after the close "
                                         f"(order status={status or 'unknown'})")
        log.info("close confirmed flat (status=%s, fill=%s, filled=%s)",
                 status or "unknown", f"{fill:.5f}" if fill else "unavailable",
                 f"{filled:g}" if filled else "unavailable")
        return fill, True, filled, ""

    def _close(self, pos, price: float, remaining: float, reason: str) -> dict:
        side = "sell" if pos["direction"] == "long" else "buy"
        close_order = None
        try:
            # SUBMIT THE CLOSE FIRST. Cancelling the stop and the exits before the market
            # order means a failed close leaves the position completely unprotected — the
            # exact situation the protection exists for. The protective orders are
            # reduceOnly, so they cannot fight the close.
            close_order = ex.place_market(self.client, self.sym, side, remaining,
                                          reduce_only=True, tag=reason)
        except ex.AmbiguousSubmission as amb:
            # The close may have filled. Keep the state (so the next tick reconciles rather
            # than opening a new position) and mark that a close is in flight.
            log.critical("close submission AMBIGUOUS (%s) — state kept for reconciliation",
                         amb)
            self.state.set_flag("close_in_flight", {"at": _now_ms(), "reason": reason})
            self.notify(f"⚠️ Mel — the {reason} close was submitted but not confirmed "
                        f"({amb}). State is kept; the next tick reads the venue and will "
                        f"reconcile or re-close, never both.")
            return {"status": "closing", "message": str(amb)}
        except Exception as exc:
            log.error("close failed: %s — protection left in place", exc)
            return {"status": "error", "message": f"close failed: {exc}"}

        # An ACCEPTED order is not a FILLED order. Do not retire protection, and do not write
        # a closed row, until the exchange confirms the exposure is gone.
        px, flat_ok, filled, why = self._confirm_close(close_order)
        if not flat_ok:
            log.error("close submitted but NOT confirmed flat (%s) — protection retained, "
                      "nothing recorded; the next tick reconciles", why)
            self.notify(f"⚠️ Mel — close submitted but not confirmed flat ({why}). "
                        f"Protection left in place and nothing recorded; "
                        f"the next tick will reconcile.")
            return {"status": "closing", "message": why}

        # A FLAT position is not proof that OUR order closed it. A protective stop or a TP can
        # flatten the position inside the same window, and booking that as e.g. a hold_cap
        # close states a reason that is false and can pull the wrong settlement inputs. If the
        # close order did not fill, reconcile by what actually happened instead of asserting
        # this reason — _reconcile_closed derives the real cause (sl / tp / exchange_close).
        if filled <= 0:
            log.warning("position is flat but the close order did NOT fill — a protective "
                        "order closed it; reconciling by actual cause, not '%s'", reason)
            return self._reconcile_closed(pos, price)

        # A PARTIAL fill is a SPLIT exit: our order took part of the position and something
        # else (a stop or a TP) took the rest. Recording the requested reason alone states a
        # cause that is only half true. The settlement already aggregates the protective
        # orders' fills so the MONEY is right — this is about the REASON being honest, because
        # downstream reads exit_reason to explain the trade.
        if remaining > 0 and filled < remaining * 0.999:
            log.warning("exit was SPLIT: the %s close filled %g of %g — a protective order "
                        "covered the remainder", reason, filled, remaining)
            reason = f"{reason}_split"

        # Our order did it: exposure is gone — now clear the orders no longer needed.
        # Best-effort: a failure here must not stop the exit from being recorded.
        # Both siblings, every time: whichever did not fill is a dangling reduce-only order
        # that could close a LATER position. Confirmed, not attempted.
        self._retire_protection(pos, f"the {reason} close")
        return self._finish(pos, px, reason, close_order=close_order)

    def _alert_on_pnl_gap(self, settle: dict, px: float, entry: float, eth: float, d: float,
                          notional: float, trade_id, source: str, pnl: float) -> None:
        """Alert when our GROSS P&L diverges materially from the account log's realized P&L.

        The account log's `realized_pnl` is GROSS of fee (the fee is a separate field on the
        same `futures trade` row), so the comparison is GROSS to GROSS: `gross` here must NOT
        subtract the fee. A fee-sized gap is noise — subtracting it would false-alarm on every
        healthy close whose fee exceeds the threshold, and the fee itself is already validated
        by the separate fee cross-check (Paul r4).
        """
        rpnl = settle.get("pnl_realized")
        if rpnl is None:
            return
        gross = (px - entry) * eth * d
        gap = abs(float(rpnl) - gross)
        if gap > max(0.50, 0.002 * abs(notional)):
            self.notify(f"⚠️ Mel — settlement disagreement on trade "
                        f"#{trade_id}: recorded net ${pnl:+,.2f} "
                        f"(source={source}) but the account log's realized P&L is "
                        f"${float(rpnl):+,.2f} gross vs our ${gross:+,.2f} gross "
                        f"(gap ${gap:,.2f}). Worth a look.")

    def _exit_detail(self, pos, px: float, reason: str, pnl: float,
                     settle: dict | None = None) -> dict:
        """The structure behind the one-token exit_reason (fleet-standard shape).

        Mel's reason vocabulary is already honest ('tp' / 'sl' / 'hold_cap'), but the word
        alone cannot say how long the cap clock ran, or that the 'tp' was attributed from a
        venue FILL rather than assumed. Same keys as Trip's and Jill's `exit_detail` so the
        history is comparable across the fleet.

        CANNOT RAISE: it is called on the close path just before the state is cleared.
        """
        def _n(v, default=0.0):
            try:
                return float(v)
            except (TypeError, ValueError):
                return default

        try:
            try:
                held = _hours_since(pos.get("entry_time"))
            except Exception:                                   # noqa: BLE001
                held = None
            cap_hours = None
            try:
                cap_hours = (self.cfg.get("exits") or {}).get("max_hold_hours")
            except Exception:                                   # noqa: BLE001
                cap_hours = None
            is_cap = reason in ("hold_cap", "time_stop")
            detail = {
                "cap": {
                    "kind": "full" if is_cap else "stop",
                    "hours": int(_n(cap_hours)) if cap_hours else None,
                    "measured_from": "entry",
                    "held_h": round(held, 2) if held is not None else None,
                },
                "legs": [{
                    "leg": str(reason),
                    "contracts": round(_n(pos.get("contracts")), 4),
                    "price": round(_n(px), 6),
                    "pnl": round(_n(pnl), 4),
                    "at": datetime.now(timezone.utc).isoformat(),
                }],
            }
            # Which protective order closed it, when the venue's fills answered that
            # question. Without this a `tp` and an `sl` look equally certain in the history.
            if settle and settle.get("exit_order_ids"):
                detail["attributed_from"] = "venue_fill"
                detail["exit_order_ids"] = list(settle["exit_order_ids"])[:4]
            return detail
        except Exception as exc:                                # noqa: BLE001
            log.warning("could not build exit_detail (%s) — recording the close without it", exc)
            return {}

    def _finish(self, pos, exit_price: float, reason: str, close_order=None) -> dict:
        """Reconcile the close against the EXCHANGE, then write what the exchange says.

        Our own arithmetic is the fallback, never the first choice — the row records
        `settlement_source` so downstream reporting can tell how well it was verified.
        """
        entry = float(pos["entry_price"])
        contracts = float(pos.get("contracts") or 0.0)
        d = 1.0 if pos["direction"] == "long" else -1.0
        # TERMS FROM ENTRY, not from today's config: leverage and margin are frozen in
        # state when the trade opens (Terra finding 7). A config edit afterwards cannot
        # retroactively restate a closed trade's ROE — and therefore cannot mask how much
        # of the account an old loss actually consumed.
        lev = int(pos.get("leverage") or self.cfg["sizing"]["leverage"] or 1)
        # [MONEY] contracts are NOT coins. The factor differs by venue: Kraken ETHUSD has
        # contract_size 1.0 (the amount IS ETH — 0.138 means 0.138 ETH), KuCoin ETHUSDTM
        # has 0.01, where omitting it overstates notional and P&L 100x. One conversion,
        # used for notional, P&L and P&L%.
        csize = float(pos.get("contract_size") or self.spec.get("contract_size") or 1.0) or 1.0
        eth = contracts * csize
        notional = entry * eth
        margin = float(pos.get("margin_usdt") or 0.0) or (notional / lev if lev else notional)

        settle = settlement.settled_close(
            self.client, self.sym,
            close_order_id=(close_order or {}).get("id", ""),
            entry_order_id=pos.get("entry_order_id", ""),
            # TP1/TP2 partial exits must be included: each books its own realized PnL
            # on the exchange, and a trade can exit in pieces. The protective STOPS are
            # included too: when a stop closed the position, its own fill is the exit — and
            # without these ids there was no exit fill to price it from, which is how an
            # earlier version came to fall back on the current ticker.
            extra_order_ids=[pos.get("tp_order_id", ""), pos.get("sl_order_id", "")],
            entry_price=entry, contracts=contracts, contract_size=csize,
            fallback_price=exit_price, opened_ms=pos.get("opened_ms"))

        # ── ATTRIBUTION: which protective order actually closed it ──
        # `exchange_close` is what the bot writes when it CANNOT tell. On this venue
        # that is the normal case, not the exception: a filled order disappears from
        # the order book, so _order_filled_state() gets ordernotfound for BOTH the TP
        # and the stop, and the price heuristic then fails too (the hourly tick runs
        # minutes after the fill, by which time price has left the level). The FILLS
        # settle it — settlement just read them, and each fill names its order id.
        # Nine historical exits were recorded as `exchange_close` this way; five were
        # take-profits and four were stop-outs, and every one of them was analysable
        # from the fill's own order id.
        if reason == "exchange_close":
            filled_ids = {str(i) for i in (settle.get("exit_order_ids") or [])}
            if pos.get("tp_order_id") and str(pos["tp_order_id"]) in filled_ids:
                reason = "tp"
                log.info("attributed exchange_close -> tp from the venue fill (order %s)",
                         str(pos["tp_order_id"])[:13])
            elif pos.get("sl_order_id") and str(pos["sl_order_id"]) in filled_ids:
                reason = "sl"
                log.info("attributed exchange_close -> sl from the venue fill (order %s)",
                         str(pos["sl_order_id"])[:13])

        px = float(settle.get("exit_price") or exit_price)
        if px <= 0:
            # No price could be established from the order's own fill, the fills VWAP, or the
            # caller's fallback. A close recorded at 0 corrupts every P&L figure and every
            # downstream report built from this row, so refuse to write it and say so rather
            # than record a number nobody verified.
            log.error("refusing to record trade #%s: no exit price could be established",
                      pos.get("trade_id"))
            # Alert ONCE and persist the marker. State stays OPEN, which keeps the tick in
            # reconciliation instead of opening a new trade on top of an undescribed close,
            # and makes the retry VISIBLE rather than a silent 15-minute loop. It clears
            # itself as soon as the venue can price the exit.
            if not pos.get("unpriced_alerted"):
                pos["unpriced_alerted"] = True
                try:
                    self.state.set_open(pos)
                except Exception as exc:
                    log.warning("could not persist the unpriced marker: %s", exc)
                self.notify(f"💀 Mel — CRITICAL: exit for trade #{pos.get('trade_id')} ({reason}) "
                            f"could not be priced from the exchange, so it was NOT recorded. "
                            f"State stays open and each tick retries until the venue can price it.")
            return {"status": "unpriced", "reason": reason,
                    "message": "no exit price could be established"}
        # [MONEY] PARTIAL EVIDENCE IS NOT A RECORDABLE EXIT. When the venue's fill history
        # covers only part of the position, the exit price — and so the P&L and the
        # equity-based caps built on it — describe a DIFFERENT trade. Keep the state, say
        # so, and retry; settlement.py's wider re-query has usually already resolved it.
        if settle.get("fills_complete") is False and contracts > 0:
            # BOUNDED, because "refuse forever" is its own failure: a trade that can never be
            # priced keeps state open, blocks every future entry, and stays invisible to the
            # guard rails. The venue being FLAT is the fact that makes a fallback honest —
            # the position is gone, only the evidence of its price is missing.
            ticks = int(pos.get("unpriced_ticks") or 0) + 1
            pos["unpriced_ticks"] = ticks
            limit = int(self.cfg["ops"].get("evidence_recovery_ticks", 3))
            _live = ex.open_position(self.client, self.sym) or {}
            # POSITIVE flatness only: an unreadable venue reports active=False WITH an error,
            # and clearing state on that would abandon a live position while writing an
            # approximate P&L for it.
            flat_confirmed = (not _live.get("error")) and (not _live.get("active"))
            if _live.get("error"):
                log.error("cannot confirm the venue is flat (%s) — not treating this as "
                          "flat", _live["error"])
            log.error("exit evidence is %s for trade #%s (%s) — attempt %d/%d",
                      "absent" if not settle.get("vwap_from_fills") else "partial",
                      pos.get("trade_id"), reason, ticks, limit)

            if not (flat_confirmed and ticks >= limit):
                if not pos.get("partial_alerted"):
                    pos["partial_alerted"] = True
                    try:
                        self.state.set_open(pos)
                    except Exception as exc:
                        log.warning("could not persist the partial marker: %s", exc)
                    self.notify(f"⚠️ Mel — trade #{pos.get('trade_id')}: the venue's fill "
                                f"history does not yet account for the exit, so it was NOT "
                                f"recorded. Retrying; after {limit} attempts with the venue "
                                f"confirmed flat it is recorded from the best available price.")
                else:
                    try:
                        self.state.set_open(pos)
                    except Exception as exc:
                        log.warning("could not persist the attempt counter: %s", exc)
                return {"status": "partial_settlement", "reason": reason,
                        "attempt": ticks,
                        "message": "exit evidence incomplete — retrying before recording"}

            # The bounded fallback. The price is the position's own stop/target (the level
            # the venue would have executed at), the row is flagged as unverified, and the
            # operator is told plainly — an honest approximate row beats an invisible loss.
            if px <= 0:
                px = float(pos.get("sl_price") or pos.get("tp_price") or 0.0)
            if reason in ("unknown", ""):
                # The cause was never established, so the row must not be labelled as a stop
                # or a target — a false attribution corrupts the loss streak the breaker uses.
                reason = "unverified_exit"
            settle["settlement_source"] = "level_fallback"
            if px <= 0:
                log.critical("no price at all for trade #%s — cannot record even a fallback",
                             pos.get("trade_id"))
                return {"status": "unpriced", "reason": reason,
                        "message": "no price could be established"}
            settle["exit_price"] = px
            settle["settlement_source"] = "level_fallback"
            settle["fee_estimated"] = True
            settle["fills_complete"] = True          # proceed deliberately, flagged below
            self.notify(f"🧾 Mel — trade #{pos.get('trade_id')} was recorded from the "
                        f"{reason.upper()} LEVEL, not from fills: after {ticks} attempts with "
                        f"the venue flat, no fill could be found for this exit. "
                        f"source=level_fallback — worth a look in the ledger.")

        fee = float(settle.get("fee_usdt") or 0.0)
        fee_estimated = bool(settle.get("fee_estimated", True))
        source = str(settle.get("settlement_source") or "provisional")
        settle_id = str(settle.get("exchange_settlement_id") or "")
        funding = float(settle.get("funding_usdt") or 0.0)

        # PROVENANCE IS REPORTED, NOT ASSUMED. `source` now names what was actually
        # verified: 'exchange_ledger' / 'fills' only when the exit price came from real
        # fills, 'fills_partial' when the venue's fill history covered only part of the
        # position, and 'level_fallback' when the price is the configured TP/SL level
        # rather than a fill. Each downgrade is worth saying out loud once.
        if not settle.get("price_verified", True) and not pos.get("priced_alerted"):
            pos["priced_alerted"] = True
            try:
                self.state.set_open(pos)
            except Exception as exc:
                log.warning("could not persist the priced marker: %s", exc)
            self.notify(f"⚠️ Mel — trade #{pos.get('trade_id')} was recorded with the "
                        f"configured {reason.upper()} level as its exit price, not a venue "
                        f"fill (no matching fill was found). source={source}")
        if settle.get("fills_complete") is False:
            self.notify(f"⚠️ Mel — trade #{pos.get('trade_id')}: the venue's fill history "
                        f"covered only part of the position, so the exit price is PARTIAL "
                        f"(source={source}). Worth a look before trusting the P&L.")
        if settle.get("ledger_disagrees"):
            self.notify(f"⚠️ Mel — trade #{pos.get('trade_id')}: the account ledger's fee "
                        f"does not match the fees on this trade's own fills "
                        f"(ledger ${fee:,.4f}). Recorded the ledger figure (venue truth), "
                        f"but the ledger is summed by symbol + window, so check for another "
                        f"trade in the same window.")

        if settle.get("pnl_net") is not None:
            # the exchange's own number — already net, do NOT subtract the fee again
            pnl = float(settle["pnl_net"])
        elif settle.get("_gross_abs") is not None:
            # built from the exchange's real entry/exit VWAPs and real charged fees
            pnl = float(settle["_gross_abs"]) * d - fee
        else:
            pnl = (px - entry) * eth * d - fee

        pnl_pct = (pnl / margin * 100.0) if margin else 0.0

        # Cross-check against independent figures. A material gap means one of them is
        # wrong — do not pick silently. Both checks are kept:
        #   * the venue's per-fill fillPnl (independent of our VWAP arithmetic)
        #   * our own arithmetic, for the positions-history path where fillPnl is absent
        # NOTE: source == "fills" is deliberately NOT in the list below. That label means the price
        # AND the fee both came from our own fills' venue-stated numbers, so `ours` is computed from
        # the very inputs that produced `pnl` — the "check" would compare a value with itself and
        # always pass. Kraken publishes no position-level realizedPnl or per-fill fillPnl through
        # ccxt, so a Kraken close has no independent figure to check against; the LABEL is what
        # carries that meaning. Do not add it here to make the lists look symmetrical.
        ours = (px - entry) * eth * d - fee
        fp = settle.get("pnl_fillpnl")
        if fp is not None and source != "exchange_fillpnl":
            gap = abs(float(fp) - pnl)
            if gap > max(0.50, 0.002 * abs(notional)):
                self.notify(f"⚠️ Mel — settlement disagreement on trade "
                            f"#{pos.get('trade_id')}: recorded ${pnl:+,.2f} "
                            f"(source={source}) vs the exchange's fillPnl ${float(fp):+,.2f} "
                            f"(gap ${gap:,.2f}). Worth a look.")
        elif source in ("exchange", "exchange_fillpnl"):
            gap = abs(pnl - ours)
            if gap > max(0.50, 0.002 * abs(notional)):
                self.notify(f"⚠️ Mel — settlement mismatch on trade "
                            f"#{pos.get('trade_id')}: exchange ${pnl:+,.2f} vs computed "
                            f"${ours:+,.2f} (gap ${gap:,.2f}). Recorded the exchange "
                            f"figure (source={source}).")

        # Kraken's account log states a per-fill realized P&L GROSS of fee (the fee is a
        # separate field on the same row), summed over the close window. It is the only
        # independent P&L figure the venue publishes through ccxt, so a `fills` or
        # `exchange_ledger` row finally has something to be checked against — not just its
        # own inputs (Paul r3). GROSS to GROSS — see _alert_on_pnl_gap, which must NOT
        # subtract the fee.
        self._alert_on_pnl_gap(settle, px, entry, eth, d, notional,
                               pos.get("trade_id"), source, pnl)

        ok = False
        if pos.get("trade_id"):
            # [MONEY] `funding_usdt` is recorded in its own column and is NOT folded into
            # `pnl_net`/`pnl_pct`: that matches Ken's convention on this venue, where the
            # dashboard reports funding separately from trading P&L. Dropping it entirely
            # (as an earlier version did) loses a real cost of holding across a funding
            # window, which is what the recorded figure exists to show.
            ok = self.db.record_exit(int(pos["trade_id"]), exit_price=px,
                                     pnl_net=round(pnl, 4), pnl_pct=round(pnl_pct, 4),
                                     exit_reason=reason, fee_usdt=round(fee, 6),
                                     funding_usdt=round(funding, 6),
                                     funding_verified=bool(settle.get("funding_verified")),
                                     fee_estimated=fee_estimated,
                                     duration_h=_hours_since(pos.get("entry_time")),
                                     settlement_source=source,
                                     exchange_settlement_id=settle_id,
                                     exit_detail=self._exit_detail(pos, px, reason, pnl, settle))
        if not ok:
            # [MONEY] DO NOT clear state. A closed trade that never reached the DB is
            # invisible to every guard rail — trade count, cooldown, the consecutive-loss
            # breaker, the daily-loss cap and the drawdown halt all read closed rows. State
            # stays open so the tick comes straight back to reconciliation and retries the
            # write, exactly like the unpriced path.
            log.critical("exit NOT recorded for trade #%s (%s) — state kept OPEN so the "
                         "next tick retries", pos.get("trade_id"), reason)
            self.notify(f"💀 Mel — CRITICAL: exit NOT recorded (trade "
                        f"#{pos.get('trade_id')}, {reason}). State stays open and the next "
                        f"tick retries; guard rails CANNOT see this trade until it lands.")
            return {"status": "unrecorded", "reason": reason,
                    "message": "record_exit failed"}
        self.state.clear()
        self.notify(f"🔷 Mel — EXIT {reason} @ {px:,.2f} | "
                    f"PnL ${pnl:+,.2f} ({pnl_pct:+.2f}%) | settled={source}"
                    f"{' (fee estimated)' if fee_estimated else ''}"
                    f"{f' | funding ${funding:+,.4f}' if funding else ''}")
        return {"status": "closed", "reason": reason, "pnl_usdt": round(pnl, 4),
                "pnl_pct": round(pnl_pct, 4), "settlement_source": source}
