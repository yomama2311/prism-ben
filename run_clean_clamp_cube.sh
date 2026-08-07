#!/bin/bash
# Consistency: the clean-prior DP->init row must also use PiJEPA's clamp.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export STABLEWM_HOME="$SCRIPT_DIR/.stable-wm"
source "$SCRIPT_DIR/.venv/bin/activate"
R="$SCRIPT_DIR/logs/rebuttal_pijepa_clamp.tsv"
for seed in 0 1 42; do
  grep -qP "^cube\tclean\t1.0\t${seed}\t" "$R" && continue
  out="$SCRIPT_DIR/logs/pijclamp_cube_clean_c1.0_s${seed}.stdout"
  timeout 2400 python "$SCRIPT_DIR/eval_dp_prior.py" --config-name=cube \
    policy=cube/lewm solver=mppi solver.num_samples=128 \
    +dpprior.injection_mode=replace +dpprior.ckpt="$SCRIPT_DIR/dp_baseline_cube.pt" \
    +dpprior.n_dp_samples=16 +dpprior.sigma_scale=1.0 +dpprior.sigma_clamp_max=1.0 \
    eval.num_eval=50 seed=$seed > "$out" 2>&1
  sr=$(grep -oE "success_rate[^a-z]*[0-9]+\.[0-9]+" "$out" | head -1 | grep -oE "[0-9]+\.[0-9]+")
  echo -e "cube\tclean\t1.0\t$seed\t${sr:-NA}" >> "$R"
  echo "[$(date '+%F %T')] cube clean clamp=1.0 seed=$seed sr=${sr:-NA}"
done
touch "$SCRIPT_DIR/logs/clean_clamp_cube.DONE"
