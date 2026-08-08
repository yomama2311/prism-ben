#!/bin/bash
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"; export STABLEWM_HOME="$SCRIPT_DIR/.stable-wm"
source "$SCRIPT_DIR/.venv/bin/activate"
LOG="$SCRIPT_DIR/logs/dp_randomgoal_cube.log"
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
CK="$SCRIPT_DIR/dp_baseline_cube_randomgoal.pt"
if [[ ! -f "$CK" ]]; then
  log "training cube DP with random goal offset (100k steps) ..."
  python -m dp_baseline.train --task cube --random-goal-offset \
    --num-workers 8 --out "$CK" >> "$LOG" 2>&1 || { log "TRAIN FAILED"; exit 1; }
fi
R="$SCRIPT_DIR/logs/rebuttal_dp_randomgoal.tsv"
for ta in 8 25; do
  for seed in 0 1 42; do
    out="$SCRIPT_DIR/logs/dprg_cube_ta${ta}_s${seed}.stdout"
    timeout 2400 python "$SCRIPT_DIR/eval_diffusion_policy.py" --config-name=cube \
      policy=cube/lewm solver=mppi +dp.ckpt="$CK" +dp.n_action_steps=$ta \
      eval.num_eval=50 seed=$seed > "$out" 2>&1
    sr=$(grep -oE "success_rate[^a-z]*[0-9]+\.[0-9]+" "$out" | head -1 | grep -oE "[0-9]+\.[0-9]+")
    echo -e "cube\trandomgoal\t$ta\t$seed\t${sr:-NA}" >> "$R"
    log "  Ta=$ta seed=$seed sr=${sr:-NA}"
  done
done
log "CUBE RANDOM-GOAL COMPLETE"
touch "$SCRIPT_DIR/logs/dp_randomgoal_cube.DONE"
