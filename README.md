# Mel — Kraken XRP/USD:USD, Trend-Aligned Pullback System (1h / 4h, one TP + one SL)

Successor to Kitt (Kraken ETH/USD:USD, retired). Same proven Kraken engine, rebuilt for the
**TPS** strategy on **XRP** — base **1h / HTF 4h**, **ONE take-profit (+4%)** and **ONE stop**
(`max(1.5×ATR%, 2.5%)`), flat 90% sizing at 2×. Dry-run by default. The full spec and the frozen
backtest evidence are in `/opt/data/KITT_XRP_TPS_SUCCESSION_DESIGN.md`.

## Install

```bash
cd /opt/data/mel
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python -r requirements.txt
ln -sfn /opt/data/shared/credentials/kraken.env .env   # canonical Kraken keys, never copied
```

## Run

```bash
.venv/bin/python3 run_bot.py --dry-run     # one tick: manage or decide, place nothing
.venv/bin/python3 run_bot.py --status      # print state + guard stats, exit
.venv/bin/python3 run_bot.py --sync        # read-only reconcile against Kraken
```

Arming requires **two** deliberate acts: the `LIVE` sentinel file in this directory **and** the
`--live` flag. Neither is present. A dry run places **nothing** (enforced in `exchange.py`, not by
ccxt).

## Halts and how to clear them

The bot blocks new entries (never management, never an exit) when a guard rail fires, and every
halt is reported to Telegram, not just logged. The operator-facing clear commands:

| condition | clears with |
|---|---|
| unresolved ambiguous protective stop (`possible_orphan_stop`) | `python3 tools/clear_halt.py --halt ambiguous_stop --note "..." --yes` |
| protection-failure streak (`sl_failure_streak`) | `python3 tools/clear_halt.py --halt sl_failure_streak --note "..." --yes` |
| consecutive-loss breaker (`loss_streak`) | `python3 tools/clear_halt.py --halt loss_streak --note "..." --yes` (also resets at the UTC day boundary) |
| a settled row resting on an estimated price | `python3 tools/reconcile_settlement.py` |

Venue-side reconciliation (read-only `--sync`, plus audited actions):

- `--adopt-orphan-stop` — declare an ambiguous candidate stop to be ours (armed only).
- `--resolve-ambiguous-stop` — assert no ambiguous stop is resting.
- `--clear-dangling` / `--cancel-dangling` — act on leftover protective orders.

## Guard rails (all under `guards:` in config.yaml)

- `max_daily_trades` (2) · `daily_loss_cap_pct` (-8.0) · `max_consecutive_losses` (3) ·
  `max_drawdown_pct` (-15.0) · `min_risk_reward` (1.0) · `max_hold_hours` (24)
- protective integrity: `max_sl_restores` (2), `max_unverified_settlements` (1),
  `max_consecutive_sl_failures` (2), `cooldown_candles` (3)
- sizing is **flat** — `confidence_scaling` is OFF.

## Layout

`engine.py` (one-tick engine) · `exchange.py` (Kraken adapter, contract-size-safe) · `guards.py`
(rails) · `strategy.py` (scoreboard) · `run_bot.py` (entry point) · `db.py` / `settlement.py`
(own-trade ledger) · `state.py` (atomic state) · `tools/` (operator tools) · `tests/` (suite).
