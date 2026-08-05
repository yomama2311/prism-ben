#!/bin/bash
# Rebuttal §4: train 5-seed prior-head ensembles (pusht + cube).
# Same recipe as released heads (defaults), only the seed differs.
# Runs concurrently with the 10-seed eval sweep (heads are tiny).
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export STABLEWM_HOME="$SCRIPT_DIR/.stable-wm"
source "$SCRIPT_DIR/.venv/bin/activate"
LOG="$SCRIPT_DIR/logs/rebuttal_ensemble.log"
for task in pusht cube; do
  for seed in 0 1 2 3 4; do
    out="$SCRIPT_DIR/prior_head_${task}_ens${seed}.pt"
    if [[ -f "$out" ]]; then echo "[skip] $out exists" | tee -a "$LOG"; continue; fi
    echo "[$(date '+%F %T')] training $task seed=$seed" | tee -a "$LOG"
    python "$SCRIPT_DIR/train_prior_head.py" --task "$task" --seed "$seed" \
      --out "$out" >> "$LOG" 2>&1 || echo "[FAIL] $task seed=$seed" | tee -a "$LOG"
  done
done
echo "[$(date '+%F %T')] ENSEMBLE TRAINING COMPLETE" | tee -a "$LOG"
touch "$SCRIPT_DIR/logs/rebuttal_ensemble.DONE"
