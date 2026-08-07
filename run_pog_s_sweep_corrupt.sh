#!/bin/bash
# Symmetric treatment: sweep PoG's sigma_scale in the same degraded-prior regime
# where we swept PiJEPA-style's clamp. Both methods get their one knob tuned.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export STABLEWM_HOME="$SCRIPT_DIR/.stable-wm"
source "$SCRIPT_DIR/.venv/bin/activate"
R="$SCRIPT_DIR/logs/rebuttal_pog_s_corrupt.tsv"
LOG="$SCRIPT_DIR/logs/rebuttal_pog_s_corrupt.log"
[[ -f "$R" ]] || echo -e "task\ts\tseed\tsr" > "$R"
for s in 0.1 0.3 0.5 1.0 2.0; do
  for seed in 0 1 42; do
    grep -qP "^pusht\t${s}\t${seed}\t" "$R" && continue
    out="$SCRIPT_DIR/logs/pogs_${s}_s${seed}.stdout"
    timeout 2400 python "$SCRIPT_DIR/eval_dp_prior.py" --config-name=pusht \
      policy=pusht/lewm solver=mppi solver.num_samples=128 \
      +dpprior.injection_mode=pog \
      +dpprior.ckpt="$SCRIPT_DIR/dp_baseline_pusht_corrupt50.pt" \
      +dpprior.n_dp_samples=16 +dpprior.sigma_scale=$s \
      eval.num_eval=50 seed=$seed > "$out" 2>&1
    sr=$(grep -oE "success_rate[^a-z]*[0-9]+\.[0-9]+" "$out" | head -1 | grep -oE "[0-9]+\.[0-9]+")
    echo -e "pusht\t$s\t$seed\t${sr:-NA}" >> "$R"
    echo "[$(date '+%F %T')] s=$s seed=$seed sr=${sr:-NA}" | tee -a "$LOG"
  done
done
touch "$SCRIPT_DIR/logs/rebuttal_pog_s_corrupt.DONE"
