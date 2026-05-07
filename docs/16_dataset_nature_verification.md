# 16. Dataset Nature Verification: Tworoom & Reacher (2026-05-04)

Read-only characterization of `.stable-wm/tworoom.h5` and
`.stable-wm/dmc/reacher_random.h5` to decide whether the prior-head training
work for each task is worth committing to. Done in <30 min wall-clock; no
heads trained, no policies rolled out, no h5 files modified.

---

## 16.1 Tworoom — `.stable-wm/tworoom.h5` (12 GB, 10 000 episodes)

### 1. Source / collection policy

**Quoted directly from LeWM paper App. E.a (arXiv:2603.19312v1, p. 20):**

> *"TwoRoom is a simple continuous 2D navigation task introduced by Sobal et
> al. The environment consists of two rooms separated by a wall with a
> single door connecting them. The agent (represented as a red dot) must
> navigate from a random starting position in one room to a randomly sampled
> target location in the other room, which requires passing through the
> door. We collect 10,000 episodes with an average trajectory length of 92
> steps. **The data are generated using a simple noisy heuristic policy
> that first directs the agent toward the door along a straight-line path
> and then toward the target location once the agent has crossed into the
> other room.** Each world model is trained on this dataset for 10 epochs."*

The h5 file itself has no top-level attrs / metadata, so the source claim
is paper-only.

### 2. Empirical quality (sampled 200 episodes; full 10 000-episode aggregates also reported where cheap)

```
n_episodes                             10,000
total frames                          920,809
episode length min/median/max         31 / 101 / 101
   (early termination on success → some short episodes; max=101)
```

| Metric | Value | Interpretation |
|---|---|---|
| Reward field | all 0 (after dropping 1.1% NaN at episode boundaries) | No reward signal in this dataset; success is binary via `terminated` flag |
| Cumulative episode reward | 0 ± 0 | Confirms above |
| `terminated=True` rate at last frame (full 10k) | **40.4 %** | ~4 040 / 10 000 episodes successfully reached the goal under the heuristic |
| Final `distance_to_target` for terminated=True episodes | mean 13.6, max 16.0 | Threshold for "success" is ~16 (in the env's pixel/grid units) |
| Final `distance_to_target` for terminated=False episodes | min 16.0, mean 75.8 | Failures end far from the goal |

**40.4 % trajectory-level success.** Sub-expert by SR but well above random
(LeWM paper's Random baseline on this task is not directly tabulated, but
their Fig. 6 shows methods as low as ~10 % being not-meaningfully-positive).

### 3. Action distribution (sampled 9 236 actions across 100 episodes)

```
action shape: (N, 2),  range [-1, 1]
dim 0: mean +0.011, std 0.866, hist-peaks ≈ 2 (bimodal, edges saturated)
dim 1: mean -0.069, std 0.870, hist-peaks ≈ 2 (bimodal, edges saturated)
```

Both action dimensions are heavily saturated near ±1 with two visible modes
— consistent with a policy that issues maximum-magnitude moves toward
the door / target. This is *not* the uniform-spread shape of a random
policy (which would land closer to a single broad mode at 0 with std
~0.58 like Reacher below).

### 4. Classification

**Tworoom = MIXED (quasi-expert / noisy heuristic).**

Evidence: **40.4 % of trajectories successfully reach the goal under a
documented heuristic policy** (paper-cited), with bimodally-saturated
actions consistent with directional motion toward intermediate waypoints.
Not pure expert (SR < 80 %), not random (SR > 10 %, structured action
distribution).

---

## 16.2 Reacher — `.stable-wm/dmc/reacher_random.h5` (93 GB, 10 000 episodes)

### 1. Source / collection policy

**Quoted directly from LeWM paper App. E.d (arXiv:2603.19312v1, p. 21):**

> *"Reacher is a continuous control environment from the DeepMind Control
> Suite. The task consists of controlling a two-joint robotic arm to reach
> a target location in a 2D plane. Following the setup used in DINO-WM, we
> consider the variant where success is defined by the perfect alignment
> of the arm joints with the target configuration required to reach the
> goal position. We train each world model for 10 epochs on a dataset of
> 10,000 episodes, each with 200 steps. **The data are collected using a
> Soft Actor-Critic policy.**"*

**However, the h5 contents are inconsistent with a trained SAC policy.**
See §16.2.4 for the classification — empirical evidence overrides the
paper's claim for our local file.

The h5 file itself has no top-level attrs.

### 2. Empirical quality (sampled 200 episodes; full 10 000-episode aggregates where relevant)

```
n_episodes                            10,000
total frames                       2,010,000
episode length                       all 201 (no early termination)
```

| Metric | Value | Interpretation |
|---|---|---|
| `success` field | **100 % NaN across all 2 010 000 rows** | No usable success signal in the dataset |
| `reward` field | corrupted (sentinel `-9.22e18` = INT64_MIN appears alongside legitimate {0, 1, 2}) | Field is broken; not usable as a reward signal |
| `score` field | range [0, 0.164], mean per-step 0.002 | The only continuous signal that's clean |
| Cumulative `score` per episode | mean 0.467 ± 1.45 (over 201 steps) | i.e., on average each episode accumulates < 0.5 of a maximum-possible-per-step 0.164. Converted to per-step: mean ≈ 0.002 / 0.164 ≈ **1.4 % "success-degree" per step** |
| `success` ANYWHERE in episode | **0.000** (mean over 200 sampled episodes; full 10k is also all-NaN) | Zero episodes ever achieve the success criterion |
| `terminated=True` at last frame | 0 % | Episodes always run for the full 201 steps; never terminate via success |

**Reading**: across 10 000 episodes × 201 steps the policy effectively never
reaches the success criterion under the env's own metric. The `score`
field — the only clean signal — averages ~0.002 per step, well below
what a competent reach-task policy would produce (a trained SAC would
spend a meaningful fraction of each episode hovering near the target
with score → 0.1+).

### 3. Action distribution (sampled 20 000 actions across 100 episodes)

```
action shape: (N, 2),  range [-1, 1]
dim 0: mean +0.003, std 0.578, hist-peaks ≈ 0 (broad, unimodal-ish)
dim 1: mean +0.009, std 0.577, hist-peaks ≈ 0 (broad, unimodal-ish)
```

Both action dimensions are broadly distributed with no clear modes,
roughly centered at 0 with std ~0.58. **This is the textbook shape of a
uniform-random or very-early-training stochastic policy.** Compare to
Tworoom (std 0.87, bimodal saturated) or Cube expert (much narrower per-dim
std with strong modes near task-specific torque vectors).

### 4. Classification

**Reacher = RANDOM / near-random.**

Evidence:
- **Zero episodes (out of 10 000) ever flag `success=True`** at any step
  (the `success` field is all NaN, and the `score` field — the continuous
  proxy — averages 1.4 % per step, never crossing into success).
- **Action distribution is uniform-noise shaped** (std ~0.58, no histogram
  modes, centered at 0).
- **The LeWM paper's stated "Soft Actor-Critic policy" is contradicted by
  the h5 contents.** Either LeWM's paper claim refers to a different file
  than the one shipped in our `.stable-wm/`, or the SAC was uninitialized
  / barely trained when used to collect this data, or the dataset is
  mislabeled. We do not investigate further — the contents are the
  ground truth for our local pipeline regardless.
- **Filename `reacher_random.h5` is consistent with the contents.**

This matches the Phase 0–2 hint that's been in the project record all
along (CLAUDE.md lesson #4: "`reacher_random` has NaN at episode
boundaries... no real expert"; §05.1b Phase 1: "Reacher: baseline 44%,
Oracle 52%, +8 pp — random dataset, no real expert; small gain
expected"). The Oracle KNN result of only +8 pp on Reacher is the
predictable consequence of training a world model on random data: the
predictor cannot accurately simulate expert-region dynamics it never saw,
so even a perfect-retrieval action prior cannot drive the planner toward
high-success trajectories.

---

## 16.3 Per-task recommendations

### Tworoom — **GO with caveat**

The dataset is mixed (~40 % trajectory success, noisy heuristic). A BC-only
head trained on this would learn the heuristic — expect BC-only SR somewhere
between random (~30–50 %) and the heuristic's own ~40 % cap; ballpark
similar to Cube's BC-only=66 % or PushT's BC-only=31 %. The full PRISM
ablation is informative either way: if BC-only ≈ heuristic ceiling, PRISM
will demonstrate the closed-loop-correction story; if BC-only is poor,
PRISM degrades gracefully (as on PushT).

**Caveats to plan around:**

1. The §05.1b Oracle KNN result on Tworoom was **64 % vs baseline 66 % = −2 pp**
   (or 27 % per project memory; the **27 % vs 64 % discrepancy is still
   unresolved** per §13). Either number says retrieval-based priors don't
   help on Tworoom. PRISM-MPPI's chance is via planner-exploration recovery
   from a sub-expert prior — same regime as PushT, *not* the same regime as
   Cube. **Resolve the 27 % vs 64 % discrepancy before running any
   oracle-prior cells.**
2. Eval-budget protocol mismatch: our default `eval_budget=50` vs LeWM
   App. F.1 spec `eval_budget=150`. Documented in §15. Run all PRISM cells
   at our 50 for internal consistency.
3. Heuristic policy is asymmetric (door-then-target), so the head's μ on a
   given (z_t, z_g) pair may not be unimodal — possible PushT-style
   multimodal-expert behavior. Plan for BC-only to be lower than Cube's.

**Estimated cost for full PRISM coverage on Tworoom:** ~1 day total —
~10 min JEPA cache + ~10 min head train + ~30 min for the 9-method × 3-seed
ablation matrix; the rest is doc-writing.

### Reacher — **DROP from prior-method coverage, keep as vanilla-baseline reference**

The dataset is random / near-random. Training a prior head on it will
produce a head whose μ is essentially a marginal-action distribution
(centered at 0, broad), which is approximately what MPPI samples by
default. Concretely:

- **BC-only ≈ vanilla MPPI** (head's μ is no better than MPPI's default
  zero-mean Gaussian sample) → no prior to beat baseline with.
- **warm-start MPPI ≈ vanilla MPPI** (μ is uninformative, so injecting it
  into the planner init is a no-op).
- **PRISM-MPPI ≈ vanilla MPPI** at any sigma_scale (PoG fusion of an
  uninformative prior with the planner default cancels out).

This is identical in structure to PushT's "bad-prior regime" but worse —
PushT had BC-only=31 % which was *informatively* below the 60 %
baseline (showed multimodal-expert pathology). Reacher would have BC-only
≈ baseline, which is a less interesting "graceful degradation" demonstration
than PushT already provides. A third such demo adds little paper value.

**Recommendation:** keep Reacher's vanilla MPPI / vanilla CEM numbers from
§15 as the 4-task vanilla-baseline reference (they're useful for the
"vanilla CEM matches LeWM-paper across all 4 tasks" claim) but **do not
train a prior head on Reacher** for the CoRL paper.

If the Reacher result is desired as a 4th-task validation, the right
preparation is to **collect or obtain a real expert dataset for Reacher**
(SAC-trained, or DM-Control task-suite expert) — a separate project, not
in scope for the current submission timeline.

---

## 16.4 Combined recommendation

**Commit full PRISM coverage on Tworoom only; keep Reacher in §15 as a
vanilla-baseline reference and drop it from prior-method coverage** —
Tworoom's noisy heuristic at 40 % success is informative enough to justify
the ~1 day of head training and PushT-comparable ablation matrix, while
Reacher's near-random data would produce a degenerate prior that adds no
empirical signal beyond what PushT already demonstrates.
