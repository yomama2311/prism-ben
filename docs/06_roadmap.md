# 06. Roadmap — From Probe to Paper

This document lays out what we should do next on the PRISM-JEPA work, organized
into phases with clear go/no-go decision points.

---

## 6.1 Where we are now (Phase 0 complete)

✅ Infrastructure: LeWM environment running, baseline 96% (CEM) / 60% (MPPI)
✅ Method: PoG-fusion MPPI subclass implemented and verified
✅ Probe: oracle prior boosts MPPI 60% → 92% on PushT
✅ Paper hypothesis validated: PRISM mechanism transfers to JEPA world models

This single result is enough to justify continued investment. The next phases
should turn it into a paper.

---

## 6.2 Phase 1: Validate generality (1 week)

**Goal:** Confirm 60→92 isn't a PushT-specific accident.

### Tasks

| # | Description | Estimated GPU time |
|---|---|---|
| 1.1 | Download Cube, Reacher, TwoRooms checkpoints + datasets | minutes |
| 1.2 | Convert each HF checkpoint to `_object.ckpt` | minutes |
| 1.3 | Run baseline MPPI on each task (50 eps each) | ~20 min total |
| 1.4 | Run Oracle PRISM-MPPI σ=0.1 on each task (50 eps each) | ~20 min total |
| 1.5 | Run 3-seed repeat of pusht baseline + oracle | ~30 min |

### Decision criterion

If oracle PRISM gives meaningful improvement (>5 pp) on at least 3 of the 4
tasks, proceed to Phase 2. If only 1 of 4, the result is task-specific and
we'd need to either find a better mechanism or pivot back to Newt-PRISM.

### Concrete commands to run

```bash
cd /home/yuhai/project/prism-jepa
source .venv/bin/activate
export STABLEWM_HOME=/home/yuhai/project/prism-jepa/.stable-wm

# 1. Download checkpoints + data for all 4 tasks
for task in cube reacher tworooms; do
  hf download quentinll/lewm-$task --local-dir $STABLEWM_HOME/hf_$task
  hf download quentinll/lewm-$task --repo-type dataset --local-dir $STABLEWM_HOME/hf_${task}_data
  zstd -d $STABLEWM_HOME/hf_${task}_data/${task}_expert_train.h5.zst \
       -o $STABLEWM_HOME/${task}_expert_train.h5
done

# 2. Convert each (needs adapting the conversion script per task — config
#    might differ slightly)

# 3. Run the matrix
for task in pusht cube reacher tworoom; do
  python eval.py --config-name=$task.yaml policy=$task/lewm solver=mppi
  python eval_oracle.py --config-name=$task.yaml policy=$task/lewm \
                        solver=mppi +oracle.sigma=0.1
done
```

---

## 6.3 Phase 2: Compute-quality trade-off curves (1 week)

**Goal:** Show PRISM's compute efficiency story explicitly.

### Tasks

| # | Description |
|---|---|
| 2.1 | Sweep `num_samples` ∈ {32, 64, 128, 256, 512} for baseline MPPI on PushT |
| 2.2 | Same sweep for Oracle PRISM σ=0.1 |
| 2.3 | Compute `K*` for each method = sample budget required to reach 90% SR |
| 2.4 | Plot SR vs K (log-x), with PRISM curve left-shifted relative to baseline |
| 2.5 | Repeat for at least Cube and Reacher |

### Key paper figure

```
SR
1.0 ┤        ╭──── PRISM (oracle)
    │       ╱
0.8 ┤      ╱   ╭──── Baseline MPPI
    │     ╱   ╱
0.6 ┤    ╱   ╱
    │   ╱   ╱
0.4 ┤  ╱   ╱
    │ ╱   ╱
0.2 ┤╱   ╱
    │   ╱
0.0 ┤──────────────────
    32  64  128  256  512  K (sample budget)
```

PRISM achieving baseline-quality SR with ~2× lower K is a clean compute-
efficiency claim. **This figure is the paper's centerpiece.**

---

## 6.4 Phase 3: Add a real VLM prior (3 weeks)

**Goal:** Replace the oracle (which uses dataset access) with a real VLM-based
prior. This is the core paper contribution.

### Approach

Two variants to try:

**Variant A: VLM features → projection head**
- Feed RGB to a frozen VLM (Qwen2.5-VL-3B or similar)
- Pool visual tokens to a fixed-size feature vector
- Train a small projection head (MLP) on the expert dataset:
  - Input: VLM features + task language embedding
  - Output: Gaussian (μ, σ) over the next 25 raw actions
- BC loss: NLL of expert actions under predicted Gaussian
- This mirrors PRISM-Newt's `prism_head.py` architecture

**Variant B: VLM as direct policy via prompting**
- Feed RGB + task description to VLM, ask it to output an action via prompt
  engineering (or a small LoRA-tuned action head on top of VLM)
- Skip the projection head training
- Faster setup but action quality unclear

We start with **Variant A** since it directly mirrors what worked in
PRISM-Newt.

### Sub-tasks

| # | Task | Estimated time |
|---|---|---|
| 3.1 | Set up VLM (Qwen2.5-VL or smaller) inference pipeline | 2 days |
| 3.2 | Implement projection head training (`train_projection_head.py`) | 2 days |
| 3.3 | Train head on PushT expert dataset (~1M VLM features × pusht actions) | 1 day GPU |
| 3.4 | Replace oracle in `oracle_policy.py` with `vlm_policy.py` | 2 days |
| 3.5 | Run VLM-PRISM eval on PushT, compare to oracle 92% | 1 day |
| 3.6 | Tune `vlm_sigma_scale` (or sigma directly) | 1 day |
| 3.7 | Repeat on Cube/Reacher | 1 week |

### Critical risk: PushT abstract images

PushT renders **abstract 2D shapes** (green circle, gray T, red T-target).
VLMs are trained on real-world photographs. The semantic match may be poor
on PushT specifically. Mitigation:

- Try Cube/Reacher first (more 3D-realistic)
- Or: use a custom-trained projection head that doesn't rely on VLM's
  language prior, only its visual encoder
- Or: replace VLM with a vision encoder pretrained on robotics data
  (e.g., R3M, VC-1)

### Go/no-go criterion

Phase 3 succeeds if VLM-PRISM achieves at least **half the gap** between
baseline (60%) and oracle (92%) — i.e., ≥76% on PushT. Anything above 76% is
publication-worthy. Below 70% is a hard pivot.

---

## 6.5 Phase 4: Paper writing (3 weeks)

**Goal:** First draft for NeurIPS 2027 (May 2027 deadline) or RSS 2027
(Dec 2026 / Jan 2027).

### Outline

1. **Introduction**
   - Problem: deployment-time MPPI is the bottleneck for real-time JEPA
     planning
   - Insight: VLMs encode task-aware visual semantics that can guide MPPI
   - Method: Product-of-Gaussians fusion of VLM prior with MPPI sampling
2. **Related Work**
   - JEPA family (V-JEPA, I-JEPA, LeWM)
   - World-model planning (TDMPC2, Newt, Dreamer)
   - VLM in robotics (RT-2, Octo, π0)
   - Prior-injection methods (RLPD, IBRL, etc.)
3. **Method**
   - PoG fusion math
   - PRISM-MPPI algorithm with diagrams
   - VLM → action distribution head architecture
4. **Experiments**
   - Setup: 4 tasks (PushT, Cube, Reacher, TwoRooms)
   - Result 1: Oracle prior boosts SR by N pp
   - Result 2: VLM prior recovers M% of the oracle gain
   - Result 3: Compute-quality curve — PRISM matches CEM at K/2
   - Ablation: Sigma sweep, sequence vs single-action prior, etc.
5. **Discussion**
   - Why goal-conditioned (no Q) is the natural setting
   - Limits: PushT-style abstract images challenge VLM
   - Future: per-step state-conditioned prior, mixture-of-Gaussians

### Cross-citation with Newt-PRISM paper (CoRL 2026)

If the Newt-PRISM paper is accepted, we cite it as:
> "Concurrent work by [authors] explores a related PoG-fusion mechanism in
> reward-driven model-based RL [Newt et al., 2026]. Our setting (goal-
> conditioned planning) provides a cleaner test of VLM contribution because
> our world model has no learned value function that could absorb the prior's
> directional knowledge."

This gives the LeWM paper the "complementary contribution" framing that
strengthens both papers.

---

## 6.6 Decisions to make soon

### Decision 1: Cube vs PushT for VLM

PushT is what we have working, but its abstract rendering may hurt VLM
performance. Cube has more realistic graphics. Decide based on Phase 1 data —
if Cube has comparable oracle gain, prefer it for VLM work.

### Decision 2: Which VLM

Candidates:
- **Qwen2.5-VL-3B** — what we used in Newt-PRISM (proven to work)
- **SigLIP** — lighter, vision-only
- **R3M / VC-1** — robotics-pretrained, no language

Recommend starting with Qwen (least-effort transfer of PRISM-Newt's projection
head architecture). Try SigLIP or R3M if Qwen underperforms on abstract
graphics.

### Decision 3: Concurrent or sequential paper writing

Option A: Finish CoRL Newt-PRISM submission first (May 28), then start LeWM
paper. Risk: lose momentum.

Option B: Run Phase 1 + 2 in parallel with Newt-PRISM finalization. CoRL
takes 2 weeks of writing; Phase 1 is ~1 week of GPU time. Run them in
parallel.

Recommended: Option B — it parallelizes nicely.

---

## 6.7 Resource estimate

For a publishable LeWM paper:

| Resource | Estimate |
|---|---|
| GPU hours | ~100 (Phase 1) + ~60 (Phase 2) + ~40 (Phase 3 training) + ~50 (Phase 3 eval) ≈ **250 GPU-hrs** |
| Wall time | 7–8 weeks (with overlap with Newt-PRISM finalization) |
| Engineering | mostly Phase 3 (VLM integration) — ~3 weeks |
| Experiments | mostly Phase 1/2 — ~2 weeks |
| Writing | ~3 weeks |

This fits a NeurIPS 2027 / RSS 2027 timeline comfortably (deadlines are
~6+ months out).

---

## 6.8 Risks & mitigations

| Risk | Mitigation |
|---|---|
| Oracle gain doesn't transfer to other tasks | Phase 1 catches this early; pivot to most-favorable task |
| PushT abstract images break VLM | Switch to Cube/Reacher; or use vision-only encoder (SigLIP, R3M) |
| Projection head doesn't beat baseline | Compare against the oracle ceiling; iterate on architecture (single vs mixture Gaussian, etc.) |
| Reviewer says "this is just IBRL/RLPD" | Stress the goal-conditioned (no-Q) setting and the JEPA backbone — these distinguish |
| LeWM upstream changes break compatibility | Pin to current commit hash; review upstream periodically |

---

## 6.9 Immediate next action (today/tomorrow)

1. **Push these docs to the private repo** — done after this writeup
2. **Take a break** — today was 6+ hours of intense debugging; ship docs and
   let the work settle overnight
3. **Tomorrow morning**: spawn Phase 1.1 — download cube + reacher + tworooms
   datasets and run 50-ep oracle probes on each

Proposed concrete plan: download all 3 remaining tasks tonight, kick off
sequential evals, see results in the morning.
