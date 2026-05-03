# 12. PRISM-JEPA — Self-contained State Summary (2026-05-02)

**In 5 lines:**
> 1. PRISM = Product-of-Gaussians fusion of an external action prior into LeWM's MPPI sampling distribution; subclass of `MPPISolver` named `PrismMPPISolver`.
> 2. Done: PoG mechanism implemented & verified; 192-d JEPA-features → MLP head → (μ, σ) trained with β-NLL; 2 tasks evaluated end-to-end on simulator (Cube, PushT) with 3 seeds each on the headline cells.
> 3. Cube (good prior, BC-only=62%): PRISM σ=2.0 vs warm-start = **+20.0 ± 8.0 pp** across n=3 seeds (3/3 positive, paired-t p≈0.05). PRISM 77.3% vs baseline 48%.
> 4. PushT (bad prior, BC-only=32%): PRISM σ=2.0 vs warm-start = **+3.3 ± 6.1 pp** across n=3 seeds (2/3 positive, paired-t p≈0.45 — n.s.). PRISM 80.7% vs seed=0 baseline 66%.
> 5. Pending: Tworoom (the originally-identified acid test, KNN ceiling 27%, untested by us); MoG head (PushT BC-only failure mode); real-robot extension (CoRL submission lever); paper-direction decision.

---

## 1. Project one-liner and current state

PRISM is a research probe injecting a Gaussian action prior into LeWM's
goal-conditioned MPPI planner via Product-of-Gaussians fusion. The
mechanism is a constructor argument and an `init_action_distrib()`
override on the upstream `MPPISolver`; nothing else is modified. The
prior is supplied externally per planning step (μ, σ tensors of shape
`(n_envs, horizon, action_dim_flat)`). Two prior sources have been
implemented and evaluated: a KNN-against-expert-h5 oracle (PRISM-Newt
descendant) and a learned MLP head on JEPA features (this project's
contribution). End-to-end 5-row reviewer-defense ablations have been
populated for Cube and PushT; multi-seed (n=3) verification is done
for the two headline cells. Tworoom is untested.

**Current decision pending:** CoRL submission with a real-robot
extension, or alternative directions (paper-write the sim 2-task
picture; pivot to MoG; run Tworoom; etc.). This document supports the
decision; it does not make a recommendation.

---

## 2. System components

| #   | Component                      | Role                                                                          | Status / origin                                                                                                                                                                                                                                                                                                |
| --- | ------------------------------ | ----------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | **JEPA encoder**               | RGB 224×224 → 192-d embedding                                                 | Frozen, lives in `lewm_object.ckpt` per task. LeWM upstream training. ~5.5 M params (ViT-tiny). Used both as world-model encoder AND head input encoder.                                                                                                                                                       |
| 2   | **AR-Predictor / world model** | (z_t, a_t) → z_{t+1}                                                          | Frozen, in same `lewm_object.ckpt`. LeWM upstream training. Drives MPPI cost via `MSE(predicted_z_T, z_goal)`.                                                                                                                                                                                                 |
| 3   | **Prior head**                 | concat(z_t, z_g) ∈ ℝ^384 → (μ, σ) over the 25-step normalized action sequence | **Trained by us.** 3-layer GELU MLP, 512 hidden, ~1.0 M params. β-NLL (β=0.5), AdamW lr=3e-4 cosine + 1k warmup. σ via `softplus + 0.05` floor. Per-task checkpoint: `prior_head_cube.pt`, `prior_head_pusht.pt`.                                                                                              |
| 4   | **PoG fusion**                 | `pog_fusion(μ_π, σ_π, μ_p, σ_p)` → fused (μ, σ) at MPPI init                  | **Written by us** in `prism_mppi.py`. `PrismMPPISolver` subclasses `stable_worldmodel.solver.MPPISolver` and overrides `init_action_distrib`. Branches on `injection_mode ∈ {pog, warm_start, none}`. The fusion math is precision-weighted: `τ_fused = τ_π + τ_p`, `μ_fused = (τ_π·μ_π + τ_p·μ_p) / τ_fused`. |

Aux glue we wrote:
- `jepa_cache.py` — caches the 192-d JEPA pooled features for every frame in an expert h5, plus per-episode goal-frame embeddings. Used by head training only (not eval).
- `train_prior_head.py` — trains the head, fits a StandardScaler to match `eval.py`'s, saves checkpoint with scaler params + history + σ-stats.
- `prism_head_policy.py` — `WorldModelPolicy` subclass `PrismHeadPolicy`. At each `get_action()` it JEPA-encodes the current obs (and goal once per episode), runs the head, dispatches to solver per `injection_mode` (or bypasses solver entirely for `bc_only`).
- `eval_prism_head.py` — Hydra entry. Mirrors `eval_oracle.py` exactly except for policy class.
- `phase_c_selfcheck.py` — synthetic test: verifies all 4 modes produce correct shapes and that `mode=none` leaves `solver._prior_mean = None`.

---

## 3. Per-task results

### 3.1 Cube

| Method                          | seed | N   | SR  | source                                   |
| ------------------------------- | ---- | --- | --- | ---------------------------------------- |
| Baseline MPPI                   | 42   | 50  | 48% | §05 historical                           |
| Oracle PRISM σ_abs=0.1          | 42   | 50  | 80% | §05 historical                           |
| mode=none (Phase D, superseded) | 42   | 20  | 45% | single seed=42 (within ±1 SE of 48%)     |
| mode=none (re-validation)       | 42   | 50  | 48% | single seed=42 — equals baseline exactly |
| BC-only                         | 42   | 50  | 62% | single seed=42                           |
| warm-start                      | 42   | 50  | 56% | single seed=42                           |
| PRISM σ=0.5                     | 42   | 50  | 76% | single seed=42                           |
| PRISM σ=1.0                     | 42   | 50  | 80% | single seed=42                           |
| PRISM σ=2.0                     | 42   | 50  | 84% | single seed=42                           |
| PRISM σ=3.0                     | 42   | 50  | 64% | single seed=42                           |
| PRISM σ=5.0                     | 42   | 50  | 56% | single seed=42                           |
| PRISM σ=10.0                    | 42   | 50  | 48% | single seed=42 — at baseline             |
| PRISM σ=10000                   | 42   | 50  | 50% | single seed=42 — asymptote sanity        |

**Multi-seed (Cube, headline cells, all N=50):**

| Method | seed=0 | seed=1 | seed=42 | mean | sample std |
|---|---|---|---|---|---|
| PRISM σ=2.0 | 76% | 72% | 84% | **77.3%** | 6.1 pp |
| warm-start | 64% | 52% | 56% | **57.3%** | 6.1 pp |
| Δ (PRISM − warm-start) | +12 pp | +20 pp | +28 pp | **+20.0 pp** | **8.0 pp** |

- **n = 3, all seeds positive.** Paired-t = 20.0 / (8.0/√3) ≈ 4.33; df=2; **p ≈ 0.049** (two-tailed). Cohen's d ≈ 2.5.
- Per-seed swing (max–min): PRISM 12 pp, warm-start 12 pp. Both ≤ 15 pp threshold.

**Cube Δ vs baseline (multi-seed mean):** PRISM σ=2.0 = 77.3% − 48% = **+29.3 pp** above baseline. Warm-start = 57.3% − 48% = **+9.3 pp** above baseline. BC-only (single seed): 62% − 48% = +14 pp.

### 3.2 PushT

| Method | seed | N | SR | source |
|---|---|---|---|---|
| Baseline MPPI (HISTORICAL) | 42 | 50 | 60% | §05 historical reference |
| Baseline MPPI (re-run, anchor) | **0** | 50 | **66%** | re-run via `eval.py` 2026-05-02 — the anchor used for all PushT Δ vs baseline below |
| Oracle PRISM σ_abs=0.1 (HISTORICAL) | 42 | 50 | 92% | §05 historical |
| mode=none | 0 | 50 | 66% | single seed=0 — equals baseline exactly |
| BC-only | 0 | 50 | 32% | single seed=0 — well below baseline |
| warm-start | 0 | 50 | 84% | single seed=0 |
| PRISM σ=2.0 | 0 | 50 | 82% | single seed=0 |
| PRISM σ=10000 | 0 | 50 | 62% | single seed=0 — asymptote sanity (within 1 SE of 66%) |

**Multi-seed (PushT, headline cells, all N=50):**

| Method | seed=0 | seed=1 | seed=42 | mean | sample std |
|---|---|---|---|---|---|
| PRISM σ=2.0 | 82% | 86% | 74% | **80.7%** | 6.1 pp |
| warm-start | 84% | 76% | 72% | **77.3%** | 6.1 pp |
| Δ (PRISM − warm-start) | −2 pp | +10 pp | +2 pp | **+3.3 pp** | **6.1 pp** |

- **n = 3, 2/3 seeds positive (seed=0 reversed).** Paired-t = 3.3 / (6.1/√3) ≈ 0.94; df=2; **p ≈ 0.45** (two-tailed) — **not significant**. 95% CI on the mean spans zero (~[−12, +18] pp).
- Per-seed swing (max–min): PRISM 12 pp, warm-start 12 pp. Both ≤ 15 pp threshold.

**PushT Δ vs baseline (multi-seed mean, anchored at seed=0 baseline 66%):** PRISM σ=2.0 = 80.7% − 66% = **+14.7 pp** above baseline. Warm-start = 77.3% − 66% = **+11.3 pp** above baseline. BC-only (single seed): 32% − 66% = **−34 pp** below baseline.

### 3.3 Tworoom

| Method | seed | N | SR | source |
|---|---|---|---|---|
| Oracle KNN PRISM | 42 | 50 | 27% | §05 historical |
| **everything else** | — | — | **NOT YET EVALUATED** | — |

This task was originally identified as the project's "acid test": KNN
ceiling is 27% because retrieval can't synthesize for novel state-goal
pairs (multi-room navigation needs generalization). No Phase 3 head has
been trained or evaluated for Tworoom.

---

## 4. Head training characteristics

| Statistic | Cube head | PushT head |
|---|---|---|
| Architecture | 384→512→512→2·H·A_block·A_raw GELU MLP | same arch (A_raw differs: 5 vs 2) |
| Parameters | ~1.04 M | ~0.94 M |
| Training data | 9000-train / 1000-val episodes (90/10 ep-level split) | 16817-train / 1868-val episodes (~90/10 split) |
| Training windows (length-25, post-NaN-filter) | ~1.58 M | ~1.65 M |
| Loss | β-NLL, β=0.5 | β-NLL, β=0.5 |
| Optimizer | AdamW lr=3e-4, wd=1e-4, cosine + 1k warmup | same |
| Best epoch | **7** (out of 50) | **1** (early-stopped at 6 with patience=5) |
| Init val MSE | 1.04 | 0.91 |
| Best val MSE | 0.61 | 0.73 |
| % drop init → best | **41%** | **19.5%** |
| Init val NLL | 0.4784 | 0.3939 |
| Best val NLL | 0.1920 | 0.5179 (note: best NLL > init — see comment below) |
| Training behavior | smooth descent through epoch 7, gentle overfit after | val MSE peaks at epoch 1, monotonically worsens thereafter |
| σ output: mean | 0.6015 | 0.5262 |
| σ output: std | 0.4069 | 0.3295 |
| σ output: median | 0.4526 | 0.4526 (identical to Cube) |
| σ output: min | 0.0514 (at floor) | 0.0550 |
| σ output: max | 4.4939 | 3.0353 |
| σ output: frac near floor (≤ 0.0525) | (not measured — predates gate) | **0.00%** |
| Per-step σ mean | [0.58, 0.56, 0.59, 0.63, 0.65] | [0.52, 0.51, 0.52, 0.53, 0.55] |

Note on PushT NLL: best val NLL (0.518) is *higher* than init val NLL
(0.394) despite val MSE being lower (0.73 vs 0.91). The best-by-NLL
checkpoint at epoch 1 has worse-calibrated σ (relative to its MSE)
than random-init does, but predicts more accurate means. The σ-output
is non-degenerate per the gate.

---

## 5. Sanity gates — pass/fail log

| Gate | Cube | PushT |
|---|---|---|
| Training: best val MSE drop ≥ 15% from init | (gated at 30% originally; would pass 15% gate at 41%) ✅ | 19.5% ✅ |
| Training: train NLL strictly ↓ to best epoch | ✅ (Cube 50-ep run) | ✅ (1-epoch trivially) |
| σ-degeneracy: frac near floor < 50% | not measured (gate added after run) | 0.00% ✅ |
| σ-degeneracy: std ≥ 0.05 | 0.41 ✅ | 0.33 ✅ |
| Scaler match (head-train vs eval) | not explicitly verified for Cube | bit-exact, max diff = 0.00e+00 ✅ |
| Eval: `mode=none` reproduces baseline | seed=42, N=50: 48% == 48% ✅ | seed=0, N=50: 66% == 66% (re-run baseline) ✅ |
| Eval: `σ=10000` asymptote ≈ baseline | 50% vs 48%, within ~1 SE ✅ | 62% vs 66%, within ~1 SE ✅ |
| σ_scale interior peak (curve has inverted-U) | yes — peak at 2.0, decay to baseline at 10 ✅ | NOT MEASURED (only ran σ ∈ {2.0, 10000}) |
| Hook no-op for `mode=none` (synthetic check) | ✅ (`phase_c_selfcheck.py`) | ✅ (same script, task-agnostic) |
| Re-run determinism (4 σ_scale points re-run at same seed) | bit-exact ✅ | not explicitly tested for PushT |

---

## 6. Findings — three tiers

### 6a. Strong claims (multi-seed, effect ≥ 1 SE)

| Claim | Numbers | Sample |
|---|---|---|
| Cube: PRISM σ=2.0 beats warm-start | Δ = +20.0 ± 8.0 pp | n=3 seeds, paired-t p ≈ 0.05 |
| Cube: PRISM σ=2.0 beats baseline (mean) | 77.3% − 48% = +29.3 pp | n=3 seeds for PRISM; baseline single seed=42 |
| PushT: PRISM σ=2.0 beats baseline (mean) | 80.7% − 66% = +14.7 pp | n=3 seeds for PRISM; baseline seed=0 anchor |
| PushT: warm-start beats baseline (mean) | 77.3% − 66% = +11.3 pp | n=3 seeds for warm-start |

### 6b. Suggestive observations (single-seed or n.s. effect)

| Observation | Numbers | Sample |
|---|---|---|
| PushT: PRISM and warm-start are not statistically distinguishable | Δ = +3.3 ± 6.1 pp, p ≈ 0.45 | n=3 seeds; CI spans zero |
| Cube single-seed: warm-start (56%) UNDERPERFORMS BC-only (62%) | gap −6 pp | single seed=42 |
| Cube single-seed: σ_scale=2.0 outperforms σ_scale=1.0 by 4 pp | 84% vs 80% | single seed=42 |
| Cube single-seed: σ_scale curve has interior peak at 2.0, drops sharply afterward | 76 → 80 → 84 → 64 → 56 → 48 → 50 | single seed=42 across {0.5, 1, 2, 3, 5, 10, 10000} |
| PushT seed=1: PRISM (86%) beats warm-start (76%) by +10 pp | one of three seeds | single observation; the cross-seed mean of the same comparison is +3.3 pp |
| PushT seed=0: warm-start (84%) beats PRISM (82%) by 2 pp | one of three seeds | single observation; reversed at seed=1 |

### 6c. Open questions / not yet validated

- **PushT BC-only = 32% (single seed=0).** Hypothesized cause: multimodal expert action distribution (head's single-Gaussian mean collapses to inter-mode average → bad action). **Not directly verified.** Multi-seed for BC-only not run; at multiple seeds the value could swing.
- **Tworoom:** entire 5-row eval matrix uncollected. No head trained.
- **Cube σ_scale sweep is single-seed only.** Whether the peak position (σ=2.0) is robust across seeds, or whether peak SR is significantly higher than σ=1.0 across seeds, is unknown.
- **PushT σ_scale interior peak is uncharacterized.** Only σ ∈ {2.0, 10000} ran; no σ_scale = 0.5 / 1.0 / 3.0 / 5.0 / 10.0 data on PushT.
- **§9.5 ablation rows skipped on PushT:** distill-from-oracle, zero-shot VLM, BC + MPPI warm-start with oracle prior (the "synthetic-prior control" mentioned in §10.3 as a follow-up).
- **Drift hypothesis (§10.3):** the explanation for why warm-start underperforms BC-only on Cube — that MPPI's wide-σ exploration drifts toward cost-rewarded but task-suboptimal actions — is hypothesized only. Per-iteration MPPI trajectory tracking, cost-vs-success correlation, and oracle-μ-into-warm-start controls have not been run.
- **Real-robot data: zero.** All current results are simulator-only (LeWM `swm/OGBCube-v0` and `swm/PushT-v1`).
- **Reacher and any other LeWM task:** untested.

---

## 7. What we ruled out (with reasoning)

| Ruled out | Reason |
|---|---|
| 50-epoch training without early stopping | On PushT, val MSE peaks at epoch 1 and worsens monotonically through epoch 50; the recipe overfits well past best. Switched to early-stopping with patience=5, max_epochs=15. Cube run was unaffected (best at epoch 7 < max). |
| Language conditioning of the head input | Pivoted away during design. Never trained a final head with language. The Qwen-VL+text cache (7.7 GB Cube) was built but the language-free paradigm replaced it. JEPA-encoder + (z_t, z_g) is the language-free architecture currently in use. |
| Qwen2.5-VL-3B as the visual encoder | Encoder-probe linear-MSE on Cube: LeWM-JEPA 0.83 / DINOv2-L/14 0.77 / Qwen-VL-vis post-merger 0.81 (and pre-merger 0.98 — barely above trivial). JEPA→DINOv2 gap = 6.8% MSE reduction, below the 20% "meaningful" threshold. JEPA is competitive at ~1/55 the parameters and is the encoder LeWM's world model already uses. |
| `mode=none` introducing drift on either task | Verified bit-exact match between mode=none and vanilla baseline at the same seed: Cube N=50 48% = 48%, PushT N=50 66% = 66%. Hook is a clean no-op. |
| Single-Gaussian PoG-style fusion for the prior parameterization | Not ruled out — this is the *current* architecture. MoG was a candidate but is deferred (§8). |

---

## 8. What's deferred (with reasoning)

| Deferred | Reason for deferral |
|---|---|
| MoG (mixture-of-Gaussians) head | Would address PushT's BC-only=32% failure *if* the multimodal-expert hypothesis is correct. Hypothesis is unverified (§6c). Estimated cost: 3–5 days (architecture change to head and to PoG fusion math; MoG-Gaussian product is not Gaussian, so fusion would need approximation). Currently lower priority than CoRL timeline + real-robot prep. |
| Tworoom (the original acid test) | Half-day cost (cache + train + eval). Originally identified as the strongest paper-headline opportunity (KNN ceiling 27% is a regime where our method's claim "we don't need test-time data" matters most). Deferred because the same compute could go to multi-seed verification of Cube/PushT, which has now happened. The decision to do Tworoom or skip is currently open. |
| Real-robot experiments | Zero current evidence on hardware. Required for CoRL framing as a robotics paper. Estimated cost: 2–4 weeks (env setup, calibration, head re-training on real rollouts or sim2real adaptation). Largest single deferred item. |
| Multi-seed σ_scale sweeps | Cube σ_scale curve is single-seed; PushT σ_scale curve is two-points-only single-seed. Cost ~10 min compute per task per seed. Cheap, just hasn't been done. |
| §9.5 ablation rows: distill-from-oracle, zero-shot VLM, oracle-μ-into-warm-start | These are reviewer-defense rows from the original design doc. Distill-from-oracle would test whether BC-on-expert is a stronger training signal than KNN-distillation. Zero-shot VLM addresses the "you require expert data" critique. Oracle-μ-into-warm-start would isolate σ-anchoring from prior-quality. None have been run. |
| Drift hypothesis empirical validation (§10.3) | Per-iteration MPPI trajectory tracking; cost-vs-success correlation. Cheap (~1 day). Would convert §10.3 from hypothesis to mechanism evidence. |
| Reacher, additional LeWM tasks | No tasks beyond Cube/PushT/Tworoom have been considered. Reacher's expert dataset is `_random` (per CLAUDE.md lesson #4) so prior quality may be poor. |

---

## 9. Decision points open right now

These are the choices that the data above can inform; this doc takes
no position.

| # | Choice | Cheapest version | Most thorough version |
|---|---|---|---|
| 1 | Run Tworoom (original acid test, KNN ceiling 27%) | Cube/PushT pipeline, 1 head, 4-config eval at single seed=0 — half day | Multi-seed verification + σ_scale sweep + multimodal-prior MoG comparison if needed — 2–3 days |
| 2 | Skip to real-robot prep (CoRL framing) | Pick a single Cube-like manipulation rig, 50 trials, σ=2.0 only — 1 week | Full sim2real bridge with multi-task hardware results — 3–4 weeks |
| 3 | Reframe paper around current 2-task evidence | Existing data already supports a "PoG essential when prior is good, equivalent when prior is bad, robust across regimes" story — start writing now | Add multi-seed σ_scale sweeps and missing §9.5 rows before drafting (~2 days extra compute) |
| 4 | Verify the multimodal-expert hypothesis (drives MoG go/no-go) | Plot per-action-pair distance from PushT held-out expert traces, look for bimodality — ~1 day | Train MoG-2 head on PushT, re-run BC-only, see whether SR jumps from 32% — ~3–5 days |
| 5 | Run the missing §9.5 rows (distill-from-oracle, zero-shot VLM, oracle-μ-into-warm-start) | One row at a time, ~1 hour each — pick the most strategic one | All three plus retrying with new ckpts — 1 day |
| 6 | Multi-seed all of Cube's σ_scale curve | 6 σ_scale × 2 extra seeds × 75 s/run ≈ 15 min compute — trivial | Add PushT σ_scale curve at multi-seed too — ~30 min |
| 7 | Validate the drift hypothesis (§10.3) | Pick 3 episodes where warm-start fails and PRISM succeeds; log per-iteration μ — ~half day | Full cost-vs-success correlation analysis on a held-out batch — 1 day |
| 8 | Other / multiple of the above | — | — |

---

*End of state summary. No recommendations.*
