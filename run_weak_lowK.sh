#!/bin/bash
# Weak prior at a LOW sample budget: does PoG's precision weighting hold up
# better than moment replacement when the planner has few samples?
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export STABLEWM_HOME="$SCRIPT_DIR/.stable-wm"
source "$SCRIPT_DIR/.venv/bin/activate"
R="$SCRIPT_DIR/logs/rebuttal_weak_lowK.tsv"
[[ -f "$R" ]] || echo -e "task\tmode\tK\tseed\tsr" > "$R"
for task in pusht cube; do
  CK="$SCRIPT_DIR/dp_baseline_${task}_corrupt50.pt"
  [[ -f "$CK" ]] || { echo "missing $CK"; continue; }
  for K in 32; do
    for mode in pog replace; do
      for seed in 0 1 42; do
        grep -qP "^${task}\t${mode}\t${K}\t${seed}\t" "$R" && continue
        out="$SCRIPT_DIR/logs/weaklowK_${task}_${mode}_K${K}_s${seed}.stdout"
        EXTRA=""; [[ "$mode" == "replace" ]] && EXTRA="+dpprior.sigma_clamp_max=1.0"
        timeout 2400 python "$SCRIPT_DIR/eval_dp_prior.py" --config-name=$task \
          policy=$task/lewm solver=mppi solver.num_samples=$K \
          +dpprior.injection_mode=$mode +dpprior.ckpt="$CK" \
          +dpprior.n_dp_samples=16 +dpprior.sigma_scale=1.0 $EXTRA \
          eval.num_eval=50 seed=$seed > "$out" 2>&1
        sr=$(grep -oE "success_rate[^a-z]*[0-9]+\.[0-9]+" "$out" | head -1 | grep -oE "[0-9]+\.[0-9]+")
        echo -e "$task\t$mode\t$K\t$seed\t${sr:-NA}" >> "$R"
        echo "[$(date '+%F %T')] $task $mode K=$K seed=$seed sr=${sr:-NA}"
      done
    done
  done
done
touch "$SCRIPT_DIR/logs/weak_lowK.DONE"
