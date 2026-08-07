#!/bin/bash
# PiJEPA-style baseline: sweep the clamp on the empirical std.
# PiJEPA specifies a CLAMPED empirical std; with a degraded policy the raw
# sample std is ~3.2 vs the planner's var_scale 1.0, so the clamp is the
# baseline's safety mechanism. We sweep it rather than assume one value.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export STABLEWM_HOME="$SCRIPT_DIR/.stable-wm"
source "$SCRIPT_DIR/.venv/bin/activate"
R="$SCRIPT_DIR/logs/rebuttal_pijepa_clamp.tsv"
LOG="$SCRIPT_DIR/logs/rebuttal_pijepa_clamp.log"
[[ -f "$R" ]] || echo -e "task\tckpt\tclamp\tseed\tsr" > "$R"
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

run(){ local ck="$1" tag="$2" clamp="$3" seed="$4"
  grep -qP "^pusht\t${tag}\t${clamp}\t${seed}\t" "$R" && return
  local out="$SCRIPT_DIR/logs/pijclamp_${tag}_c${clamp}_s${seed}.stdout"
  timeout 2400 python "$SCRIPT_DIR/eval_dp_prior.py" --config-name=pusht \
    policy=pusht/lewm solver=mppi solver.num_samples=128 \
    +dpprior.injection_mode=replace +dpprior.ckpt="$SCRIPT_DIR/$ck" \
    +dpprior.n_dp_samples=16 +dpprior.sigma_scale=1.0 \
    +dpprior.sigma_clamp_max=$clamp \
    eval.num_eval=50 seed=$seed > "$out" 2>&1
  local sr=$(grep -oE "success_rate[^a-z]*[0-9]+\.[0-9]+" "$out" | head -1 | grep -oE "[0-9]+\.[0-9]+")
  echo -e "pusht\t$tag\t$clamp\t$seed\t${sr:-NA}" >> "$R"
  log "  $tag clamp=$clamp seed=$seed sr=${sr:-NA}"
}

# degraded prior: the regime where the clamp matters
for clamp in 0.5 1.0 2.0; do
  for seed in 0 1 42; do run dp_baseline_pusht_corrupt50.pt corrupt50 $clamp $seed; done
done
# clean prior sanity: clamp should barely bind (raw sigma ~0.34)
for clamp in 1.0; do
  for seed in 0 1 42; do run dp_baseline_pusht.pt clean $clamp $seed; done
done
log "PIJEPA CLAMP SWEEP DONE"
touch "$SCRIPT_DIR/logs/rebuttal_pijepa_clamp.DONE"
