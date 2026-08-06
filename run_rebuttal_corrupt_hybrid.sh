#!/bin/bash
# Does PoG add value when the POLICY prior is degraded?
# Uses the eps=50% corrupted DP (standalone SR 55.3) as the prior source.
# Headroom exists here, unlike the clean-data ceiling regime.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export STABLEWM_HOME="$SCRIPT_DIR/.stable-wm"
source "$SCRIPT_DIR/.venv/bin/activate"
R="$SCRIPT_DIR/logs/rebuttal_corrupt_hybrid.tsv"
LOG="$SCRIPT_DIR/logs/rebuttal_corrupt_hybrid.log"
[[ -f "$R" ]] || echo -e "task\tmode\tN\tseed\tsr\tdp_ms" > "$R"
CK="$SCRIPT_DIR/dp_baseline_pusht_corrupt50.pt"
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

run(){  # mode N seed
  local mode="$1" N="$2" seed="$3"
  grep -qP "^pusht\t${mode}\t${N}\t${seed}\t" "$R" && { log "skip $mode N=$N s=$seed"; return; }
  local out="$SCRIPT_DIR/logs/chyb_${mode}_N${N}_s${seed}.stdout"
  timeout 2400 python "$SCRIPT_DIR/eval_dp_prior.py" --config-name=pusht \
    policy=pusht/lewm solver=mppi solver.num_samples=128 \
    +dpprior.injection_mode=$mode +dpprior.ckpt="$CK" \
    +dpprior.n_dp_samples=$N +dpprior.sigma_scale=1.0 \
    eval.num_eval=50 seed=$seed > "$out" 2>&1
  local sr=$(grep -oE "success_rate[^a-z]*[0-9]+\.[0-9]+" "$out" | head -1 | grep -oE "[0-9]+\.[0-9]+")
  local ms=$(grep -oE "dp_sample_ms_per_replan: mean=[0-9.]+" "$out" | grep -oE "[0-9.]+$")
  echo -e "pusht\t$mode\t$N\t$seed\t${sr:-NA}\t${ms:-NA}" >> "$R"
  log "  $mode N=$N seed=$seed sr=${sr:-NA}"
}

for seed in 0 1 42 2 3 4 5 6 7 8; do
  run best_of_n 1  $seed     # corrupted DP, no fusion, no verification (~standalone)
  run pog       16 $seed     # PoG fusion of corrupted DP moments
  run replace   16 $seed     # PiJEPA-style init from corrupted DP
  run best_of_n 16 $seed     # WM verification only, no fusion
done
log "CORRUPT-HYBRID DONE"
touch "$SCRIPT_DIR/logs/rebuttal_corrupt_hybrid.DONE"
