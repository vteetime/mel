"""Guards — risk gates applied to every candidate entry.

Each guard is a pure function returning (ok, reason). `run_guards` applies them in
order and returns the first failure plus an aggregate position scale.

Deliberately independent of any exchange or DB so the whole gate stack is unit
testable and cannot be bypassed by a caller that forgot a step.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GuardResult:
    passed: bool
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    position_scale: float = 1.0

    @property
    def reason(self) -> str:
        return "; ".join(self.failures) if self.failures else "ok"


# ── individual gates ─────────────────────────────────────────────────────────

def g_daily_trades(trades_today: int, cfg: dict) -> tuple[bool, str]:
    maxd = int(cfg["guards"]["max_daily_trades"])
    if maxd > 0 and trades_today >= maxd:
        return False, f"daily trade limit reached ({trades_today}/{maxd})"
    return True, ""


def g_consecutive_losses(streak: int, cfg: dict, last_loss_at: str | None = None,
                         now_iso: str | None = None) -> tuple[bool, str]:
    """Block further entries for the REST OF THE UTC DAY after `max_consecutive_losses` losses.

    0 disables. The reset is the DAY BOUNDARY, not a win. A breaker whose only reset is a winning
    close can never reset, because it blocks the very trade that would produce the win — a one-way
    wedge, found by Mel's own backtest on 2026-09-20: three losses, then 18 months with no entries.
    The daily loss cap already brakes on loss with a natural daily reset; this matches it.

    `last_loss_at` is the most recent closed trade's exit time. Unknown (None) blocks — fail closed,
    since an unreadable history is not evidence that the streak is stale.
    """
    maxl = int(cfg["guards"]["max_consecutive_losses"])
    if maxl <= 0 or streak < maxl:
        return True, ""
    if last_loss_at:
        if now_iso is None:
            from datetime import datetime, timezone
            now_iso = datetime.now(timezone.utc).isoformat()
        if str(last_loss_at)[:10] != str(now_iso)[:10]:
            return True, ""                     # the streak is from an earlier day: today is fresh
    return False, (f"circuit breaker: {streak} consecutive losses (max {maxl}) — entries resume at "
                   f"the next UTC day boundary, or now with "
                   f"`tools/clear_halt.py --halt loss_streak --note \"...\" --yes`")


def _equity_pct(ctx: dict, equity_key: str, roe_key: str) -> float:
    """Prefer the equity-normalised figure; fall back to margin ROE (documented above)."""
    v = ctx.get(equity_key)
    if v is None:
        v = ctx.get(roe_key)
    try:
        return float(v or 0.0)
    except (TypeError, ValueError):
        return 0.0


def g_daily_loss_cap(pnl_today_pct: float, cfg: dict) -> tuple[bool, str]:
    cap = float(cfg["guards"]["daily_loss_cap_pct"])
    if cap != 0 and pnl_today_pct <= cap:
        return False, f"daily loss cap hit ({pnl_today_pct:.2f}% <= {cap:.2f}%)"
    return True, ""


def g_drawdown(pnl_total_pct: float, cfg: dict) -> tuple[bool, str]:
    cap = float(cfg["guards"]["max_drawdown_pct"])
    if cap != 0 and pnl_total_pct <= cap:
        return False, f"max drawdown hit ({pnl_total_pct:.2f}% <= {cap:.2f}%) — halted"
    return True, ""


def g_cooldown(candles_since_close: int | None, cfg: dict) -> tuple[bool, str]:
    need = int(cfg["guards"]["cooldown_candles"])
    if candles_since_close is not None and candles_since_close < need:
        return False, f"cooldown ({candles_since_close}/{need} candles since last close)"
    return True, ""


def g_unverified_settlements(count: int, cfg: dict) -> tuple[bool, str]:
    """Halt NEW entries while a settled trade's P&L rests on an estimate.

    A `level_fallback` row exists because the venue never produced evidence for the exit. Its
    price is a level the venue *should* have executed at, and its P&L therefore an estimate —
    it must not quietly stand in for a measurement inside the loss caps and the streak, which
    is the number the daily-loss and drawdown breakers act on. An operator reconciles it
    (`tools/reconcile_settlement.py`) and trading resumes.
    """
    limit = int(cfg["guards"].get("max_unverified_settlements", 1))
    if limit > 0 and int(count or 0) >= limit:
        return (False, f"{int(count)} settlement(s) rest on an estimated price "
                       f"(max {limit}) — reconcile before taking new risk")
    return True, ""


def g_sl_failure_streak(streak: int, cfg: dict) -> tuple[bool, str]:
    """Halt entries after repeated protective-stop failures.

    A stop that could not be placed (or that vanished and could not be restored) is a broken
    SAFETY path, not a bad strategy call. Re-entering into it is how the Trip churn loop spent
    ~$226 on taker fees while never holding a protected position.
    """
    limit = int(cfg["guards"].get("max_consecutive_sl_failures", 2))
    if limit > 0 and int(streak or 0) >= limit:
        return (False, f"protective-stop failure streak: {int(streak)} in a row "
                       f"(max {limit}) — the venue/adapter is not protecting positions")
    return True, ""


def g_risk_reward(tp: float, sl: float, entry: float, cfg: dict) -> tuple[bool, str]:
    # Mel has ONE take-profit, so this compares the trade's actual target against its
    # stop — 4% over the 2.5% floor = 1.6:1, and the gate cannot be passed by a partial
    # level the position would have to reach first.
    tp_d = abs(tp - entry)
    sl_d = abs(sl - entry)
    if sl_d <= 0:
        return False, "SL distance is zero"
    need = float(cfg["guards"]["min_risk_reward"])
    rr = tp_d / sl_d
    if need > 0 and rr < need:
        return False, f"TP R:R {rr:.2f}:1 below floor {need:.2f}:1"
    return True, ""


def g_counter_trend(action: str, htf: dict, confidence: int, cfg: dict) -> tuple[bool, str]:
    """No SHORT into a strong HTF uptrend, no LONG into a strong downtrend."""
    if not bool(cfg["guards"].get("block_counter_trend", True)):
        return True, ""
    direction = htf.get("direction", "unknown")
    strength = float(htf.get("strength", 0.0) or 0.0)
    if direction == "unknown":
        return True, ""
    strong = float(cfg["guards"]["counter_trend_strong_pct"])
    min_conf = int(cfg["guards"]["counter_trend_min_conf"])
    against = (direction == "up" and action == "SHORT") or \
              (direction == "down" and action == "LONG")
    if not against:
        return True, ""
    if strength > strong:
        return False, f"counter-trend {action} vs strong HTF {direction} ({strength:.1f}%)"
    if strength > 1.0 and confidence < min_conf:
        return False, (f"counter-trend {action} needs confidence >= {min_conf} "
                       f"(got {confidence}, HTF sep {strength:.1f}%)")
    return True, ""


def g_rsi_extreme(action: str, rsi: float | None, confidence: int, cfg: dict) -> tuple[bool, str]:
    """A stop whose job is to LIMIT loss must not be chased into an extreme."""
    if not bool(cfg["guards"].get("block_rsi_extreme", True)) or rsi is None:
        return True, ""
    if action == "LONG" and rsi > 75:
        return False, f"RSI {rsi:.1f} too overbought for LONG"
    if action == "SHORT" and rsi < 25:
        return False, f"RSI {rsi:.1f} too oversold for SHORT"
    if action == "LONG" and rsi > 70 and confidence < 7:
        return False, f"RSI {rsi:.1f} elevated — need confidence >= 7"
    if action == "SHORT" and rsi < 30 and confidence < 7:
        return False, f"RSI {rsi:.1f} depressed — need confidence >= 7"
    return True, ""


def g_bb_extreme(action: str, bb_pct_b: float | None, confidence: int, cfg: dict) -> tuple[bool, str]:
    if not bool(cfg["guards"].get("block_bb_extreme", True)) or bb_pct_b is None:
        return True, ""
    if action == "LONG" and bb_pct_b > 90:
        return False, f"BB%B {bb_pct_b:.0f} near upper band — wait for pullback"
    if action == "SHORT" and bb_pct_b < 10:
        return False, f"BB%B {bb_pct_b:.0f} near lower band — wait for bounce"
    if action == "LONG" and bb_pct_b > 80 and confidence < 7:
        return False, f"BB%B {bb_pct_b:.0f} near upper band — need confidence >= 7"
    if action == "SHORT" and bb_pct_b < 20 and confidence < 7:
        return False, f"BB%B {bb_pct_b:.0f} near lower band — need confidence >= 7"
    return True, ""


def g_macd_scale(action: str, macd_hist: float | None) -> tuple[float, str]:
    """DISABLED (D1, 2026-10-14 — decided-by-default, Tee can reverse in one line).

    The validated backtest assumed FLAT 90% sizing; a donor-only MACD-fade half-size
    modifier is not in the tested set and does not belong in a bot whose whole
    justification is that backtest. Full size IS the tested size, so this always
    returns 1.0. Re-enabling it requires restoring the half-size branch AND a test
    asserting flat sizing must then be updated to fail (see tests/test_kitt.py
    ``test_sizing_is_flat_even_on_macd_fade``).
    """
    return 1.0, ""


# ── aggregate ────────────────────────────────────────────────────────────────

def run_guards(decision, state, ctx: dict, cfg: dict, overridden: bool = False) -> GuardResult:
    """Apply every gate to a candidate entry.

    ctx keys: trades_today, consecutive_losses, pnl_today_pct, pnl_total_pct,
              candles_since_close, and (preferred) pnl_today_equity_pct /
              pnl_total_equity_pct.

    THE UNIT MATTERS. A closed row's `pnl_pct` is ROE on the margin committed, not a
    percentage of account equity — they are equal only when a position deploys the whole
    account. The caps below are configured as percentages of EQUITY, so the equity figures
    are used whenever the DB can supply them (it derives them from the margin fraction
    recorded on each trade). Falling back to the ROE sums would make a −10% daily cap
    mean −10% of a position's margin, which is a different (smaller) loss.

    ``overridden`` is the operator's guard-rail override ("resume despite the rails"): the four
    account-level rails below are NOT evaluated when it is set, so no latch is raised and no
    alert fires contradicting a decision the operator just made. The safety latches and the
    signal-shaped filters still apply — an override is about risk appetite, never about trading
    into a broken protective path or an unknown P&L.
    """
    failures: list[str] = []
    warnings: list[str] = []

    # ── the four account-level RAILS (override-able) ─────────────────────────
    if not overridden:
        for ok, why in (
            _r(g_daily_trades, ctx.get("trades_today", 0), cfg),
            _r(g_consecutive_losses, ctx.get("consecutive_losses", 0), cfg, ctx.get("last_loss_at"),
               ctx.get("now_iso")),
            _r(g_daily_loss_cap, _equity_pct(ctx, "pnl_today_equity_pct", "pnl_today_pct"), cfg),
            _r(g_drawdown, _equity_pct(ctx, "pnl_total_equity_pct", "pnl_total_pct"), cfg),
        ):
            if not ok:
                failures.append(why)

    for ok, why in (
        _r(g_cooldown, ctx.get("candles_since_close"), cfg),
        # A broken protective path outranks every market-based reason to enter.
        _r(g_sl_failure_streak, ctx.get("sl_failure_streak", 0), cfg),
        _r(g_unverified_settlements, ctx.get("unverified_settlements", 0), cfg),
        _r(g_risk_reward, decision.tp, decision.sl, decision.entry, cfg),
        _r(g_counter_trend, decision.action, state.htf, decision.confidence, cfg),
        _r(g_rsi_extreme, decision.action, state.rsi, decision.confidence, cfg),
        _r(g_bb_extreme, decision.action, state.bb_pct_b, decision.confidence, cfg),
    ):
        if not ok:
            failures.append(why)

    macd_scale, warn = g_macd_scale(decision.action, state.macd_hist)
    if warn:
        warnings.append(warn)

    return GuardResult(passed=not failures, failures=failures, warnings=warnings,
                       position_scale=macd_scale if not failures else 0.0)


def _r(fn, *args):
    try:
        return fn(*args)
    except Exception as exc:                     # a broken gate must FAIL CLOSED
        return False, f"{fn.__name__} errored: {exc}"
