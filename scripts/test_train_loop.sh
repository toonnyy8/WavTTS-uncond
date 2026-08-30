#!/usr/bin/env bash
# Self-check for train_loop.sh: restarts on crash, stops on success, and waits
# for a run already in flight. Runs the real script with the trainer swapped for a stub
# and a fake config name, so it never touches the GPU or a checkpoint.
#   scripts/test_train_loop.sh
set -u
cd "$(dirname "$0")/.."

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"; kill $(jobs -p) 2>/dev/null' EXIT
mkdir -p "$WORK/scripts" "$WORK/logs" "$WORK/.venv/bin" "$WORK/src/wavtts/train"

python3 - "$WORK" <<'PY'
import sys
w = sys.argv[1]
s = open('scripts/train_loop.sh').read()
s = s.replace('CONFIG=${1:-WavTTS_clean.yaml}', 'CONFIG=FAKE_TEST.yaml')
s = s.replace('''    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
        .venv/bin/accelerate launch --mixed_precision bf16 \\
        src/wavtts/train/train.py --config-name "$CONFIG" >>"$LOG" 2>&1''',
              '''    bash -c 'sleep 1; exit $(cat rc)' >>"$LOG" 2>&1''')
s = s.replace('sleep 60', 'sleep 1').replace('sleep 30', 'sleep 1')
s = s.replace('LOG=logs/train_${CONFIG%.yaml}.log', 'LOG=logs/train_clean.log')
open(w + '/scripts/loop.sh', 'w').write(s)
PY
chmod +x "$WORK/scripts/loop.sh"
cd "$WORK"
fail=0
check() { if [ "$1" = pass ]; then echo "PASS $2"; else echo "FAIL $2"; fail=1; fi; }

# restarts on a non-zero exit
echo 1 >rc; : >logs/train_clean.log
./scripts/loop.sh & lp=$!; sleep 6; kill $lp 2>/dev/null; wait $lp 2>/dev/null
n=$(grep -c "crashed (exit 1), resuming" logs/train_clean.log)
[ "$n" -ge 2 ] && check pass "restarts on crash (n=$n)" || check fail "restarts on crash (n=$n)"

# stops on a clean exit instead of looping forever
echo 0 >rc; : >logs/train_clean.log
timeout 10 ./scripts/loop.sh; rc=$?
{ [ "$rc" -eq 0 ] && grep -q "training finished (exit 0)" logs/train_clean.log; } \
  && check pass "stops on success" || check fail "stops on success (rc=$rc)"

# adopts a run already in flight rather than starting a second trainer
echo 0 >rc; : >logs/train_clean.log
# a real python under the expected path/argv, so it matches both the cmdline and /proc/comm
ln -sf "$(readlink -f "$(command -v python3)")" .venv/bin/python
echo 'import time; time.sleep(6)' >src/wavtts/train/train.py
.venv/bin/python src/wavtts/train/train.py --config-name FAKE_TEST.yaml & dp=$!
sleep 1
./scripts/loop.sh & lp=$!; sleep 3
grep -q launching logs/train_clean.log && check fail "waits while a trainer is alive" \
                                       || check pass "waits while a trainer is alive"
wait $dp 2>/dev/null; sleep 4; kill $lp 2>/dev/null
grep -q launching logs/train_clean.log && check pass "launches once it exits" \
                                       || check fail "launches once it exits"
exit $fail
