#!/bin/bash
# Fill the '---' cell in Table I: Cube DP trained with end-of-episode goals
# (the original pre-fix convention). Eval at T_a=8, matching every other cell
# in that column (PushT's 23.3 was measured the same way).
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"; export STABLEWM_HOME="$SCRIPT_DIR/.stable-wm"
source "$SCRIPT_DIR/.venv/bin/activate"
LOG="$SCRIPT_DIR/logs/dp_endgoal_cube.log"
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
CK="$SCRIPT_DIR/dp_baseline_cube_endgoal.pt"
if [[ ! -f "$CK" ]]; then
  log "training cube DP with end-of-episode goals (100k steps, same budget) ..."
  python -m dp_baseline.train --task cube --endframe-goal \
    --num-workers 8 --out "$CK" >> "$LOG" 2>&1 || { log "TRAIN FAILED"; exit 1; }
fi
log "training done"
R="$SCRIPT_DIR/logs/rebuttal_dp_endgoal.tsv"
[[ -f "$R" ]] || echo -e "task\tTa\tseed\tsr" > "$R"
for seed in 0 1 42; do
  out="$SCRIPT_DIR/logs/dpeg_cube_s${seed}.stdout"
  timeout 2400 python "$SCRIPT_DIR/eval_diffusion_policy.py" --config-name=cube \
    policy=cube/lewm solver=mppi +dp.ckpt="$CK" +dp.n_action_steps=8 \
    eval.num_eval=50 seed=$seed > "$out" 2>&1
  sr=$(grep -oE "success_rate[^a-z]*[0-9]+\.[0-9]+" "$out" | head -1 | grep -oE "[0-9]+\.[0-9]+")
  echo -e "cube\t8\t$seed\t${sr:-NA}" >> "$R"
  log "  seed=$seed sr=${sr:-NA}"
done
log "CUBE ENDGOAL COMPLETE"
touch "$SCRIPT_DIR/logs/dp_endgoal_cube.DONE"
