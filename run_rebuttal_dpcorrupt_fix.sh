#!/bin/bash
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
export STABLEWM_HOME="$SCRIPT_DIR/.stable-wm"
source "$SCRIPT_DIR/.venv/bin/activate"
LOG="$SCRIPT_DIR/logs/rebuttal_overnight_aug5.log"
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
CK="$SCRIPT_DIR/dp_baseline_pusht_corrupt50.pt"
if [[ ! -f "$CK" ]]; then
  log "(a) retry: training DP pusht corrupt=0.5 via python -m ..."
  python -m dp_baseline.train --task pusht --corrupt-ep-frac 0.5 \
    --num-workers 8 --out "$CK" >> "$LOG" 2>&1 || { log "(a) TRAIN FAILED AGAIN"; exit 1; }
fi
log "(a) training done"
RA="$SCRIPT_DIR/logs/rebuttal_dp_corrupt.tsv"
[[ -f "$RA" ]] || echo -e "task\teps\tseed\tsr" > "$RA"
for seed in 0 1 42; do
  out="$SCRIPT_DIR/logs/dp_corrupt50_pusht_s${seed}.stdout"
  timeout 2400 python "$SCRIPT_DIR/eval_diffusion_policy.py" --config-name=pusht \
    policy=pusht/lewm solver=mppi \
    +dp.ckpt="$CK" +dp.n_action_steps=25 eval.num_eval=50 seed=$seed > "$out" 2>&1
  sr=$(grep -oE "success_rate[^a-z]*[0-9]+\.[0-9]+" "$out" | head -1 | grep -oE "[0-9]+\.[0-9]+")
  echo -e "pusht\t50\t$seed\t${sr:-NA}" >> "$RA"
  log "(a) eval seed=$seed sr=${sr:-NA}"
done
log "OVERNIGHT CHAIN COMPLETE (retry)"
touch "$SCRIPT_DIR/logs/rebuttal_overnight_aug5.DONE"
