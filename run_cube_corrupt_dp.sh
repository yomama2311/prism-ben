#!/bin/bash
# Cube eps=50% corrupted DP: training + standalone eval + hybrid decomposition.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
export STABLEWM_HOME="$SCRIPT_DIR/.stable-wm"
source "$SCRIPT_DIR/.venv/bin/activate"
LOG="$SCRIPT_DIR/logs/cube_corrupt_dp.log"
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
CK="$SCRIPT_DIR/dp_baseline_cube_corrupt50.pt"

if [[ ! -f "$CK" ]]; then
  log "training cube DP with corrupt-ep-frac 0.5 (~7h) ..."
  python -m dp_baseline.train --task cube --corrupt-ep-frac 0.5 \
    --num-workers 8 --out "$CK" >> "$LOG" 2>&1 || { log "TRAIN FAILED"; exit 1; }
fi
log "training done"

# standalone (T_a=25)
RA="$SCRIPT_DIR/logs/rebuttal_dp_corrupt.tsv"
for seed in 0 1 42; do
  out="$SCRIPT_DIR/logs/dp_corrupt50_cube_s${seed}.stdout"
  timeout 2400 python "$SCRIPT_DIR/eval_diffusion_policy.py" --config-name=cube \
    policy=cube/lewm solver=mppi +dp.ckpt="$CK" +dp.n_action_steps=25 \
    eval.num_eval=50 seed=$seed > "$out" 2>&1
  sr=$(grep -oE "success_rate[^a-z]*[0-9]+\.[0-9]+" "$out" | head -1 | grep -oE "[0-9]+\.[0-9]+")
  echo -e "cube\t50\t$seed\t${sr:-NA}" >> "$RA"
  log "standalone seed=$seed sr=${sr:-NA}"
done

# hybrid decomposition (matches the PushT protocol)
R="$SCRIPT_DIR/logs/rebuttal_corrupt_hybrid.tsv"
run(){ local mode="$1" N="$2" seed="$3"
  grep -qP "^cube\t${mode}\t${N}\t${seed}\t" "$R" && return
  local out="$SCRIPT_DIR/logs/chyb_cube_${mode}_N${N}_s${seed}.stdout"
  timeout 2400 python "$SCRIPT_DIR/eval_dp_prior.py" --config-name=cube \
    policy=cube/lewm solver=mppi solver.num_samples=128 \
    +dpprior.injection_mode=$mode +dpprior.ckpt="$CK" \
    +dpprior.n_dp_samples=$N +dpprior.sigma_scale=1.0 \
    eval.num_eval=50 seed=$seed > "$out" 2>&1
  local sr=$(grep -oE "success_rate[^a-z]*[0-9]+\.[0-9]+" "$out" | head -1 | grep -oE "[0-9]+\.[0-9]+")
  echo -e "cube\t$mode\t$N\t$seed\t${sr:-NA}\tNA" >> "$R"
  log "  $mode N=$N seed=$seed sr=${sr:-NA}"
}
for seed in 0 1 42 2 3 4 5 6 7 8; do
  run best_of_n 1 $seed; run pog 16 $seed; run replace 16 $seed; run best_of_n 16 $seed
done
log "CUBE CORRUPT DP COMPLETE"
touch "$SCRIPT_DIR/logs/cube_corrupt_dp.DONE"
