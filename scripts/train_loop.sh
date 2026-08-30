#!/usr/bin/env bash
# Restart-on-crash supervisor for a training run.
#
# The clean-460 run died once to `CUDA error: unspecified launch failure` (update
# 56539) — a driver/hardware hiccup, not OOM, so a plain relaunch fixes it. Resume
# is full-state from model_last.pt, which is written every last_per_updates (2500),
# so a crash costs at most ~2500 updates of work.
#
#   nohup scripts/train_loop.sh [config.yaml] >/dev/null 2>&1 &
#
# To stop training: kill this script FIRST, then the trainer — otherwise the
# supervisor sees the exit as a crash and restarts it.
#   pkill -f train_loop.sh
#   pkill -f '.venv/bin/python src/wavtts/train/train.py --config-name <config.yaml>'

set -u
cd "$(dirname "$0")/.."

CONFIG=${1:-WavTTS_clean.yaml}
LOG=logs/train_${CONFIG%.yaml}.log
TRAINER_CMDLINE=".venv/bin/python src/wavtts/train/train.py --config-name $CONFIG"
mkdir -p logs

# Is a trainer for this config already running? A bare `pgrep -f` is not enough: it also
# matches any shell whose command line happens to mention the config, and then the wait
# below never ends. Requiring the match to be an actual python process rules that out.
trainer_running() {
    local pid
    for pid in $(pgrep -f "$TRAINER_CMDLINE"); do
        case $(cat "/proc/$pid/comm" 2>/dev/null) in python*) return 0 ;; esac
    done
    return 1
}

while true; do
    # ponytail: adopt a run already in flight instead of racing it with a second one
    while trainer_running; do sleep 30; done

    echo "=== launching $CONFIG at $(date '+%F %T') ===" >>"$LOG"
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        .venv/bin/accelerate launch --mixed_precision bf16 \
        src/wavtts/train/train.py --config-name "$CONFIG" >>"$LOG" 2>&1
    rc=$?

    if [ $rc -eq 0 ]; then
        echo "=== training finished (exit 0) at $(date '+%F %T') ===" >>"$LOG"
        break
    fi
    echo "=== crashed (exit $rc), resuming from model_last.pt in 60s ===" >>"$LOG"
    sleep 60
done
