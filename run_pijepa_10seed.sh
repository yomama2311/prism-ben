#!/bin/bash
# Match the seed count of the PoG arm (10) for the as-specified comparison.
# PRE-COMMITTED: exactly 10 seeds (0,1,42,2..8), reported whatever the outcome.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export STABLEWM_HOME="$SCRIPT_DIR/.stable-wm"
source "$SCRIPT_DIR/.venv/bin/activate"
R="$SCRIPT_DIR/logs/rebuttal_pijepa_clamp.tsv"
for seed in 2 3 4 5 6 7 8; do
  grep -qP "^pusht\tcorrupt50\t1.0\t${seed}\t" "$R" && continue
  out="$SCRIPT_DIR/logs/pijclamp_corrupt50_c1.0_s${seed}.stdout"
  timeout 2400 python "$SCRIPT_DIR/eval_dp_prior.py" --config-name=pusht \
    policy=pusht/lewm solver=mppi solver.num_samples=128 \
    +dpprior.injection_mode=replace \
    +dpprior.ckpt="$SCRIPT_DIR/dp_baseline_pusht_corrupt50.pt" \
    +dpprior.n_dp_samples=16 +dpprior.sigma_scale=1.0 \
    +dpprior.sigma_clamp_max=1.0 \
    eval.num_eval=50 seed=$seed > "$out" 2>&1
  sr=$(grep -oE "success_rate[^a-z]*[0-9]+\.[0-9]+" "$out" | head -1 | grep -oE "[0-9]+\.[0-9]+")
  echo -e "pusht\tcorrupt50\t1.0\t$seed\t${sr:-NA}" >> "$R"
  echo "[$(date '+%F %T')] clamp1.0 seed=$seed sr=${sr:-NA}"
done
touch "$SCRIPT_DIR/logs/rebuttal_pijepa_10seed.DONE"
