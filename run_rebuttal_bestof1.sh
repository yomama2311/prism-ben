#!/bin/bash
# Decomposition control: best_of_1 = DP sample at 25-step cadence, NO WM selection.
# Separates "DP quality (incl. memorization)" from "WM verification" in the
# hybrid's win. 2 tasks x 3 seeds.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export STABLEWM_HOME="$SCRIPT_DIR/.stable-wm"
source "$SCRIPT_DIR/.venv/bin/activate"
RESULTS="$SCRIPT_DIR/logs/rebuttal_hybrid_results.tsv"
LOG="$SCRIPT_DIR/logs/rebuttal_bestof1.log"
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
for task in pusht cube; do
  for seed in 0 1 42; do
    if grep -qP "^${task}\tbest_of_1\t${seed}\t[0-9.]+\t" "$RESULTS"; then continue; fi
    out="$SCRIPT_DIR/logs/hybrid_${task}_best_of_1_s${seed}.stdout"
    t0=$(date +%s)
    log "$task | best_of_1 | seed=$seed ..."
    timeout 2400 python "$SCRIPT_DIR/eval_dp_prior.py" --config-name="$task" \
      policy="$task/lewm" solver=mppi solver.num_samples=128 \
      +dpprior.injection_mode=best_of_n \
      +dpprior.ckpt="$SCRIPT_DIR/dp_baseline_${task}.pt" \
      +dpprior.n_dp_samples=1 +dpprior.sigma_scale=1.0 \
      eval.num_eval=50 seed=$seed > "$out" 2>&1
    sr=$(grep -oE "success_rate[^a-z]*[0-9]+\.[0-9]+" "$out" | head -1 | grep -oE "[0-9]+\.[0-9]+")
    dpms=$(grep -oE "dp_sample_ms_per_replan: mean=[0-9.]+" "$out" | grep -oE "[0-9.]+$")
    t1=$(date +%s); [[ -z "$sr" ]] && sr="NA"
    echo -e "$task\tbest_of_1\t$seed\t$sr\t${dpms:-NA}\t$((t1-t0))" >> "$RESULTS"
    log "  sr=$sr wall=$((t1-t0))s"
  done
done
log "BEST_OF_1 DONE"
touch "$SCRIPT_DIR/logs/rebuttal_bestof1.DONE"
