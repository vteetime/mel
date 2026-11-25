"""Trade logging to the shared CIC trades.db — with mel's OWN bot_id.

Isolation matters: the shared guards filter on bot_id. Writing this bot's rows under
another bot's id would corrupt that bot's risk accounting — the exact bug class fixed
in bx-bot §11. Every row here carries bot_id = mel, and in particular a Mel row must
never land under `ken`: Ken's accounting is read by the dashboard and by Ken's guards.

SINGLE TP: `tp1_price` carries Mel's one take-profit, `tp2_price` is written as 0.0
(nullable, and 0.0 is what existing consumers already read as "no level"). `exit_reason`
is 'tp', 'sl', 'hold_cap', 'manual', 'sync_missing' or 'sl_placement_failed' — the
ladder reasons ('tp1', 'tp_runner') cannot occur here.

The real table already exists and is shared by bx / sol / xrp / ken / kraken-sol. This
module therefore does NOT create or reshape it; it verifies the expected columns are
present and fails loudly if not, so a schema drift is caught before an order is placed
rather than at write time.

NOT NULL columns that must be supplied even at ENTRY (verified against the live table):
  bot_id, exchange, symbol, direction, entry_price, exit_price, contracts,
  pnl_net, pnl_pct, entry_time, exit_time
So an entry row is written with exit_price=0.0, pnl_net=0.0, pnl_pct=0.0, exit_time=''
and closed later by record_exit().
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

BOT_ID = "mel"
EXCHANGE = "kraken_futures"

_REQUIRED = {
    "bot_id", "exchange", "symbol", "direction", "entry_price", "exit_price",
    "contracts", "pnl_net", "pnl_pct", "entry_time", "exit_time", "status",
    "sl_price", "tp1_price", "tp2_price", "sl_pct", "notional", "guard_config",
    "created_at", "fee_usdt", "duration_h", "trigger", "fee_estimated",
    # The scoreboard score at entry. It is in _REQUIRED for the same reason every
    # other written column is: this list is checked at STARTUP, so a missing column
    # refuses to trade instead of failing the INSERT after an order has already
    # filled on the venue.
    "entry_score",
}


PROTECTION_FAILURE_REASONS = ("sl_placement_failed", "sl_missing_unprotected")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class TradeDB:
    def __init__(self, path: str | Path):
        self.path = str(path)
        self._conn = sqlite3.connect(self.path, timeout=30)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA busy_timeout=30000")
        self._verify_schema()

    def _verify_schema(self) -> None:
        try:
            cols = {r[1] for r in self._conn.execute("PRAGMA table_info(trades)")}
        except sqlite3.Error as exc:
            raise RuntimeError(f"cannot read trades schema at {self.path}: {exc}") from exc
        if not cols:
            raise RuntimeError(f"trades table missing at {self.path}")
        missing = sorted(_REQUIRED - cols)
        if missing:
            raise RuntimeError(
                f"trades table is missing columns this bot writes: {missing}. "
                "Refusing to trade rather than fail at write time.")

    # ── writes ───────────────────────────────────────────────────────────────
    def record_entry(self, *, symbol: str, direction: str, entry_price: float,
                     contracts: float, leverage: int, sl_price: float, tp1_price: float,
                     tp2_price: float, notional: float, sl_pct: float = 0.0,
                     trigger: str = "scoreboard", guard_config: dict | None = None,
                     entry_time: str | None = None,
                     entry_score: float | None = None) -> int:
        cur = self._conn.execute(
            """INSERT INTO trades
                 (bot_id, exchange, symbol, direction, entry_price, exit_price,
                  contracts, leverage, notional, pnl_net, pnl_pct, fee_usdt,
                  entry_time, exit_time, exit_reason, status, sl_price, tp1_price,
                  tp2_price, sl_pct, trigger, guard_config, fee_estimated,
                  settlement_source, created_at, entry_score)
               VALUES (?,?,?,?,?,0,?,?,?,0,0,0,?,'',NULL,'open',?,?,?,?,?,?,0,
                       'provisional',?,?)""",
            (BOT_ID, EXCHANGE, symbol, direction, float(entry_price),
             float(contracts), int(leverage), float(notional),
             entry_time or _now(), float(sl_price), float(tp1_price), float(tp2_price),
             float(sl_pct), trigger, json.dumps(guard_config or {}), _now(),
             None if entry_score is None else float(entry_score)))
        self._conn.commit()
        return int(cur.lastrowid)

    def record_exit(self, trade_id: int, *, exit_price: float, pnl_net: float,
                    pnl_pct: float, exit_reason: str, fee_usdt: float = 0.0,
                    funding_usdt: float = 0.0, duration_h: float | None = None,
                    fee_estimated: bool = True, exit_time: str | None = None,
                    settlement_source: str = "provisional",
                    exchange_settlement_id: str = "",
                    funding_verified: bool = False,
                    exit_detail: dict | None = None) -> bool:
        """Close a trade. `settlement_source` records HOW WELL this was verified:
        'exchange' (the venue's own realizedPnl), 'fills' (real fills, computed), or
        'provisional' (our arithmetic). Never claim more certainty than we have.

        `funding_verified` is the same honesty question for funding: True only when the venue's own
        account log supplied the figure. It is written but deliberately NOT in `_REQUIRED` — it is
        pure provenance that no guard reads, and a startup gate for it would refuse to run against a
        perfectly good older ledger.
        """
        et = exit_time or _now()
        if duration_h is None:
            row = self._conn.execute(
                "SELECT entry_time FROM trades WHERE id=? AND bot_id=?",
                (int(trade_id), BOT_ID)).fetchone()
            duration_h = 0.0
            if row and row["entry_time"]:
                try:
                    t0 = datetime.fromisoformat(str(row["entry_time"]))
                    duration_h = (datetime.now(timezone.utc) - t0).total_seconds() / 3600.0
                except (TypeError, ValueError):
                    duration_h = 0.0
        cur = self._conn.execute(
            """UPDATE trades SET exit_price=?, pnl_net=?, pnl_pct=?, fee_usdt=?,
                   funding_usdt=?, funding_verified=?, exit_reason=?, exit_time=?, duration_h=?,
                   fee_estimated=?, status='closed', settlement_source=?,
                   exchange_settlement_id=?, settled_at=?, exit_detail=?
               WHERE id=? AND bot_id=? AND status='open'""",
            (float(exit_price), float(pnl_net), float(pnl_pct), float(fee_usdt),
             float(funding_usdt), 1 if funding_verified else 0, exit_reason, et, float(duration_h),
             1 if fee_estimated else 0, settlement_source,
             exchange_settlement_id or None, et,
             json.dumps(exit_detail) if exit_detail else None,
             int(trade_id), BOT_ID))
        self._conn.commit()
        return cur.rowcount == 1

    def update_sl(self, trade_id: int, sl_price: float) -> bool:
        cur = self._conn.execute(
            "UPDATE trades SET sl_price=? WHERE id=? AND bot_id=? AND status='open'",
            (float(sl_price), int(trade_id), BOT_ID))
        self._conn.commit()
        return cur.rowcount == 1

    # ── reads (scoped to mel only) ────────────────────────────────────────
    @staticmethod
    def _margin_fraction(guard_config) -> float:
        """The share of free collateral a trade deployed, from its entry-time record.

        Default 1.0 (fail-closed): treating ROE as if it were an equity percentage makes
        the loss caps trigger EARLIER, never later. Rows written before this field existed
        (orphan adoptions) therefore cannot silently weaken a cap.
        """
        try:
            data = json.loads(guard_config) if guard_config else {}
        except (TypeError, ValueError):
            data = {}
        try:
            mf = float(data.get("margin_fraction"))
        except (TypeError, ValueError):
            return 1.0
        if mf <= 0 or mf > 1.0:
            return 1.0
        return mf

    def stats(self) -> dict:
        """Everything the guards need. Scoped to BOT_ID — never global.

        `pnl_today_pct` / `pnl_total_pct` are margin ROE (what each row stores).
        `pnl_today_equity_pct` / `pnl_total_equity_pct` convert each row to a percentage
        of account equity using the margin fraction recorded at its entry, which is the
        unit the configured loss caps are expressed in. The guards prefer the equity
        figures; the ROE sums remain for reporting.
        """
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        row = self._conn.execute(
            """SELECT COUNT(*) AS c, COALESCE(SUM(pnl_pct),0) AS s
               FROM trades WHERE bot_id=? AND status='closed' AND date(exit_time)=?""",
            (BOT_ID, today)).fetchone()
        trades_today = int(row["c"] or 0)
        pnl_today = float(row["s"] or 0.0)

        rows = list(self._conn.execute(
            "SELECT pnl_pct, guard_config, date(exit_time) AS d FROM trades "
            "WHERE bot_id=? AND status='closed'", (BOT_ID,)))
        pnl_total = sum(float(r["pnl_pct"] or 0.0) for r in rows)
        pnl_total_eq = 0.0
        pnl_today_eq = 0.0
        for r in rows:
            eq = float(r["pnl_pct"] or 0.0) * self._margin_fraction(r["guard_config"])
            pnl_total_eq += eq
            if r["d"] == today:
                pnl_today_eq += eq

        # A settlement whose price came from a level rather than from evidence is an
        # ESTIMATE. It still counts toward the exposure caps (an invisible loss would be
        # worse), but it must not manufacture a loss streak or hide one, so it is excluded
        # from the streak and surfaced separately for the guard that halts new risk.
        unverified = int(self._conn.execute(
            """SELECT COUNT(*) AS c FROM trades WHERE bot_id=? AND status='closed'
               AND settlement_source='level_fallback'""", (BOT_ID,)).fetchone()["c"] or 0)

        streak = 0
        for r in self._conn.execute(
            """SELECT pnl_pct FROM trades WHERE bot_id=? AND status='closed'
               AND (settlement_source IS NULL OR settlement_source != 'level_fallback')
               ORDER BY exit_time DESC LIMIT 50""", (BOT_ID,)):
            if (r["pnl_pct"] or 0) <= 0:
                streak += 1
            else:
                break

        # Consecutive trailing closed rows whose exit was a PROTECTIVE-STOP FAILURE. This is
        # not a losing trade, it is a broken safety path, so it gets its own breaker: the
        # Trip churn loop burned ~$226 in fees re-entering after entry stops failed.
        sl_streak = 0
        for r in self._conn.execute(
                """SELECT exit_reason FROM trades WHERE bot_id=? AND status='closed'
                   ORDER BY exit_time DESC LIMIT 20""", (BOT_ID,)):
            if str(r["exit_reason"] or "") in PROTECTION_FAILURE_REASONS:
                sl_streak += 1
            else:
                break
        sl_today = int(self._conn.execute(
            """SELECT COUNT(*) AS c FROM trades WHERE bot_id=? AND status='closed'
               AND date(exit_time)=? AND exit_reason IN ('sl_placement_failed',
               'sl_missing_unprotected')""", (BOT_ID, today)).fetchone()["c"] or 0)

        last = self._conn.execute(
            """SELECT exit_time FROM trades WHERE bot_id=? AND status='closed'
               ORDER BY exit_time DESC LIMIT 1""", (BOT_ID,)).fetchone()
        last_fail = self._conn.execute(
            """SELECT MAX(exit_time) AS t FROM trades WHERE bot_id=? AND status='closed'
               AND exit_reason IN ('sl_placement_failed', 'sl_missing_unprotected')""",
            (BOT_ID,)).fetchone()["t"]

        last_loss = self._conn.execute(
            """SELECT exit_time, pnl_net FROM trades WHERE bot_id=? AND status='closed'
               ORDER BY exit_time DESC LIMIT 1""", (BOT_ID,)).fetchone()
        last_loss_at = (last_loss["exit_time"]
                        if last_loss and float(last_loss["pnl_net"] or 0.0) <= 0 else None)

        return {"last_loss_at": last_loss_at,
                "trades_today": trades_today,
                "last_protection_failure_at": last_fail,
                "sl_failure_streak": sl_streak, "sl_failures_today": sl_today,
                "unverified_settlements": unverified,
                "pnl_today_pct": pnl_today, "pnl_total_pct": pnl_total,
                "pnl_today_equity_pct": pnl_today_eq,
                "pnl_total_equity_pct": pnl_total_eq,
                "consecutive_losses": streak,
                "last_exit_time": last["exit_time"] if last else None}

    def open_trade(self) -> dict | None:
        r = self._conn.execute(
            "SELECT * FROM trades WHERE bot_id=? AND status='open' ORDER BY id DESC LIMIT 1",
            (BOT_ID,)).fetchone()
        return dict(r) if r else None

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass
