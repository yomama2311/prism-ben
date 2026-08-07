#!/bin/bash
# Data-side test of the S2 explanation: train DP with goal distances covering
# what receding-horizon inference induces (k ~ U{1..25}), then evaluate at
#   T_a=25 : the cadence-matched setting (PRISM replans every 25 env-steps too)
#   T_a=8  : the mechanism test -- does the collapse disappear?
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
export STABLEWM_HOME="$SCRIPT_DIR/.stable-wm"
source "$SCRIPT_DIR/.venv/bin/activate"
LOG="$SCRIPT_DIR/logs/dp_randomgoal.log"
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
CK="$SCRIPT_DIR/dp_baseline_pusht_randomgoal.pt"

if [[ ! -f "$CK" ]]; then
  log "training DP pusht with random goal offset (100k steps, same budget) ..."
  python -m dp_baseline.train --task pusht --random-goal-offset \
    --num-workers 8 --out "$CK" >> "$LOG" 2>&1 || { log "TRAIN FAILED"; exit 1; }
fi
log "training done"

R="$SCRIPT_DIR/logs/rebuttal_dp_randomgoal.tsv"
[[ -f "$R" ]] || echo -e "task\ttrain\tTa\tseed\tsr" > "$R"
for ta in 25 8; do
  for seed in 0 1 42; do
    out="$SCRIPT_DIR/logs/dprg_ta${ta}_s${seed}.stdout"
    timeout 2400 python "$SCRIPT_DIR/eval_diffusion_policy.py" --config-name=pusht \
      policy=pusht/lewm solver=mppi +dp.ckpt="$CK" +dp.n_action_steps=$ta \
      eval.num_eval=50 seed=$seed > "$out" 2>&1
    sr=$(grep -oE "success_rate[^a-z]*[0-9]+\.[0-9]+" "$out" | head -1 | grep -oE "[0-9]+\.[0-9]+")
    echo -e "pusht\trandomgoal\t$ta\t$seed\t${sr:-NA}" >> "$R"
    log "  Ta=$ta seed=$seed sr=${sr:-NA}"
  done
done
log "DP RANDOM-GOAL COMPLETE"
touch "$SCRIPT_DIR/logs/dp_randomgoal.DONE"
