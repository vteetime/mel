# Changelog

All notable changes to **Mel** — the Kraken Futures XRP 1h strategy bot
(`/opt/data/mel`, account `kraken-main`, hold cap 24h), will be documented here.

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [Unreleased] — 2026-11-16 — the hold cap is checked every 15 minutes, and a trade records its exit detail

### Added
- **`run_bot.py --manage-only`** — a management-only tick that runs the full management path (hold cap, stops, guards) and returns **before** the candle read, so it cannot score or open a position. The untested entry path is left exactly as it was.
- **Cron pair:** the hourly full tick stays as it was, and `mel_manage.sh` runs `--manage-only` at 13, 28, 43 and 58 past the hour, so a 24h hold cap is enforced on a 15-minute bound instead of an hourly one.
- **`tests/test_manage_only.py`** — the flag cannot open a position, and the lock behaviour is asserted rather than assumed.
- **`exit_detail`** in the engine (`_exit_detail`), built from the venue FILL rather than assumed, with the same keys as Trip's and Jill's so the dashboards read one shape. `entry_score` is written from the decision's confidence at entry. `_backfill_mel_history.py` populates history.
- **`tests/test_exit_attribution.py`** — a close without exit detail is recorded and logged as a warning rather than failing the close.

### Changed
- **`MANAGE_ONLY_LOCK_WAIT_SECONDS = 240`.** A manage-only tick now waits for a competing tick before giving up; if it still cannot take the lock it fails **loudly** (exit 3 plus a Telegram alert) instead of exiting 0, because an overlapping hourly tick must never make a due management pass silently disappear.
- `exchange.py` / `config.py` carry the balance-read changes that make Mel size from the corrected Kraken figure.

### Verified
- **2026-11-25, by this job:** the flag, the 240s wait, the loud-failure path and the exit-detail builder are read from the committed source (`run_bot.py:102-105`, `engine.py:2039`). This file was created in this pass; the suites were **not** re-run here.

---
