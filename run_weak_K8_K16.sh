#!/bin/bash
# Weak prior at K=16 and K=8, both rules + the prior-free floor.
# PRE-COMMITTED: all 36 cells reported regardless of outcome.
# PiJEPA-style keeps its specified clamp (sigma_max = sigma_pi); PoG has none.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"; export STABLEWM_HOME="$SCRIPT_DIR/.stable-wm"
source "$SCRIPT_DIR/.venv/bin/activate"
R="$SCRIPT_DIR/logs/rebuttal_weak_lowK.tsv"
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$SCRIPT_DIR/logs/weak_K8_K16.log"; }

for task in pusht cube; do
  CK="$SCRIPT_DIR/dp_baseline_${task}_corrupt50.pt"
  for K in 16 8; do
    # prior-free floor at this budget
    for seed in 0 1 42; do
      grep -qP "^${task}\tvanilla\t${K}\t${seed}\t" "$R" && continue
      out="$SCRIPT_DIR/logs/weaklowK_${task}_vanilla_K${K}_s${seed}.stdout"
      timeout 2400 python eval.py --config-name=$task policy=$task/lewm solver=mppi \
        solver.num_samples=$K solver.topk=$K eval.num_eval=50 seed=$seed > "$out" 2>&1
      sr=$(grep -oE "success_rate[^a-z]*[0-9]+\.[0-9]+" "$out" | head -1 | grep -oE "[0-9]+\.[0-9]+")
      echo -e "$task\tvanilla\t$K\t$seed\t${sr:-NA}" >> "$R"; log "$task vanilla K=$K s=$seed sr=${sr:-NA}"
    done
    # the two fusion rules on the weak prior
    for mode in pog replace; do
      for seed in 0 1 42; do
        grep -qP "^${task}\t${mode}\t${K}\t${seed}\t" "$R" && continue
        out="$SCRIPT_DIR/logs/weaklowK_${task}_${mode}_K${K}_s${seed}.stdout"
        EXTRA=""; [[ "$mode" == "replace" ]] && EXTRA="+dpprior.sigma_clamp_max=1.0"
        timeout 2400 python eval_dp_prior.py --config-name=$task \
          policy=$task/lewm solver=mppi solver.num_samples=$K \
          +dpprior.injection_mode=$mode +dpprior.ckpt="$CK" \
          +dpprior.n_dp_samples=16 +dpprior.sigma_scale=1.0 $EXTRA \
          eval.num_eval=50 seed=$seed > "$out" 2>&1
        sr=$(grep -oE "success_rate[^a-z]*[0-9]+\.[0-9]+" "$out" | head -1 | grep -oE "[0-9]+\.[0-9]+")
        echo -e "$task\t$mode\t$K\t$seed\t${sr:-NA}" >> "$R"; log "$task $mode K=$K s=$seed sr=${sr:-NA}"
      done
    done
  done
done
log "WEAK K=8/16 COMPLETE"
touch "$SCRIPT_DIR/logs/weak_K8_K16.DONE"
