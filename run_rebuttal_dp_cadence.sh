#!/bin/bash
# Confirm the cadence diagnosis on the ORIGINAL DP eval path:
# same ckpt, same protocol, only n_action_steps 8 -> 25.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export STABLEWM_HOME="$SCRIPT_DIR/.stable-wm"
source "$SCRIPT_DIR/.venv/bin/activate"
RESULTS="$SCRIPT_DIR/logs/rebuttal_dp_cadence.tsv"
[[ -f "$RESULTS" ]] || echo -e "task\tn_action_steps\tseed\tsr" > "$RESULTS"
for task in pusht cube; do
  for seed in 0 1 42; do
    out="$SCRIPT_DIR/logs/dp_cadence25_${task}_s${seed}.stdout"
    timeout 2400 python "$SCRIPT_DIR/eval_diffusion_policy.py" --config-name="$task" \
      policy="$task/lewm" solver=mppi \
      +dp.ckpt="$SCRIPT_DIR/dp_baseline_${task}.pt" +dp.n_action_steps=25 \
      eval.num_eval=50 seed=$seed > "$out" 2>&1
    sr=$(grep -oE "success_rate[^a-z]*[0-9]+\.[0-9]+" "$out" | head -1 | grep -oE "[0-9]+\.[0-9]+")
    echo -e "$task\t25\t$seed\t${sr:-NA}" >> "$RESULTS"
    echo "[$(date '+%F %T')] $task seed=$seed sr=${sr:-NA}"
  done
done
touch "$SCRIPT_DIR/logs/rebuttal_dp_cadence.DONE"
