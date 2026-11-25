#!/bin/bash
# Mel — cron wrapper (every 1 hour). Kraken XRP/USD:USD, 2x, one TP (4%) + one SL.
#
# LOCKING: run_bot.py owns the lock (ops.lock_file, non-blocking, exits 0 on collision).
# This wrapper deliberately does NOT take a second lock — on Jack an earlier version locked
# the same path in both places, the Python side always lost the race, and every tick exited
# 0 looking healthy while never trading. One lock, one owner.
#
# LIVE GATE: real orders only while /opt/data/mel/LIVE exists; otherwise every tick is a
# dry run. Funding the bot and arming it are two separate deliberate acts.
#
# ENVIRONMENT: run_bot.py reads KRAKEN_API_KEY / KRAKEN_API_SECRET from the process
# environment, falling back to .env NEXT TO THE CODE (load_keys). This wrapper sources it
# when present so a bare cron entry works; a scheduler that supplies the environment itself
# works equally.
#
# Empty stdout = silent tick by design. notify.py sends Telegram alerts for entry / exit /
# error / guard-rail events only, so a 15-minute cron cannot spam.

set -u

MEL_DIR="/opt/data/mel"
cd "$MEL_DIR" || exit 1
mkdir -p logs

if [ -f "$MEL_DIR/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    . "$MEL_DIR/.env"
    set +a
fi

MODE="--dry-run"
if [ -f "$MEL_DIR/LIVE" ]; then
    MODE="--live"
fi

# ── MANAGE-ONLY (the 15-minute ticks) ───────────────────────────────────────────────────────────
# Same wrapper, same pause guard, same lock, same LIVE gate — the ONLY difference is that the
# Python side stops after management and never scores an entry. Set by the mel-manage cron entry
# via MEL_MANAGE_ONLY=1. Entries therefore stay on the hourly decision tick, which is the only
# tick that reads the (still forming) 1h candle.
MANAGE_ONLY_FLAG=""
LABEL=""
if [ "${MEL_MANAGE_ONLY:-0}" = "1" ]; then
    MANAGE_ONLY_FLAG="--manage-only"
    LABEL=" manage-only"
fi

# ── TIER 1: the fleet pause guard ────────────────────────────────────────────────────────────────
# Runs BEFORE the tick. Design + contract: /opt/data/DESIGN_PAUSE_FLEET.md, /opt/data/shared/PAUSE.md.
#
# The venue value must come from the EXCHANGE (tools/venue_state.py), never from state.json: at the
# moment we decide whether to SKIP a tick, the state file is the thing we do not trust.
#
# Deliberately the SYSTEM python3, not the bot's venv: the guard is stdlib-only so that a broken or
# missing venv can never take the pause control down with it. An unexpected guard exit (anything but
# 0/10/11) ticks as before rather than stopping the bot — a crashed guard must not become an outage,
# and Tier 2 in engine.try_enter still refuses entries.
#
#   0  RUN          tick normally
#   10 SKIP         do not tick at all (venue holds the stop, so protection does not depend on us)
#   11 MANAGE_ONLY  tick, but refuse to open a new position in-engine
GUARD_PY="${PAUSE_GUARD_PY:-/usr/bin/python3}"
VENUE=$("$MEL_DIR/.venv/bin/python3" "$MEL_DIR/tools/venue_state.py" 2>> logs/mel_cron.log) || VENUE="unprovable"
case "$VENUE" in flat|open|unprovable) ;; *) VENUE="unprovable" ;; esac
"$GUARD_PY" /opt/data/shared/pause_guard.py --bot mel --venue "$VENUE" >> logs/mel_cron.log 2>&1
GUARD_RC=$?
case $GUARD_RC in
    0)  ;;
    10) echo "$(date -u +%FT%TZ): paused, venue $VENUE — tick SKIPPED" >> logs/mel_cron.log; exit 0 ;;
    11) echo "$(date -u +%FT%TZ): paused, venue $VENUE — manage-only tick (entries refused in-engine)" >> logs/mel_cron.log ;;
    *)  echo "$(date -u +%FT%TZ): pause guard exited $GUARD_RC (unexpected) — ticking as before" >> logs/mel_cron.log ;;
esac

echo "--- $(date -u +%FT%TZ) tick ($MODE${LABEL}) ---" >> logs/mel_cron.log
.venv/bin/python3 run_bot.py "$MODE" $MANAGE_ONLY_FLAG >> logs/mel_cron.log 2>&1
RC=$?

if [ $RC -ne 0 ]; then
    echo "$(date -u +%FT%TZ): tick exited $RC (mode $MODE)" >> logs/mel_cron.log
fi
exit $RC
