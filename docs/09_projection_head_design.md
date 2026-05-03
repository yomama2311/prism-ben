# 09. VLM Projection Head — Design Specification

This is the v1 spec for the head that turns frozen-VLM features into a
Gaussian over MPPI action sequences. Phase 3 of the roadmap (§6.4 Variant A).

The design is built around one constraint: **the mechanism in §3 needs a
single diagonal Gaussian as input.** Anything more expressive (mixture,
flow) would change the fusion math and is deferred to ablation.

---

## 9.1 What the head must output

For each MPPI re-plan, given current obs (and optional task description),
the head produces:

- `μ ∈ R^{H × A_block × A_raw}` — predicted action sequence
- `σ ∈ R^{H × A_block × A_raw}` — per-element std (diagonal)

with `H = 5`, `A_block = 5`, `A_raw ∈ {2, …, 7}` depending on task.

These plug straight into `pog_fusion(...)` in `prism_mppi.py`. **The output
must be in StandardScaler-normalized action space**, not raw — see CLAUDE
lesson #1 and #2.

The "specification of correctness" is already in hand: oracle KNN with
`μ = expert action sequence, σ = 0.1` gives 92% / 80% SR. A successful
head reproduces that quality without dataset access at test time.

---

## 9.2 The five design choices, with the smart option in **bold**

### Choice 1 — Output structure: per-step factorized vs flat vs autoregressive

A flat `R^{H·A_block·A_raw}` MLP output ignores temporal structure;
autoregressive μ_t | μ_{<t} can't be expressed as a single diagonal
Gaussian without dropping back to marginals.

**Pick:** per-step diagonal Gaussian, with a model that *can* mix temporal
information internally even though the output is factorized. Smallest
parameter count, exactly matches what PoG consumes, lets the network
share temporal context via attention.

### Choice 2 — Backbone: MLP vs transformer decoder

PRISM-Newt used an MLP (`prism_head.py`). For sequences of length H=5,
the difference vs a transformer decoder is small in raw capacity but
big in inductive bias: the decoder forces the head to think "what
should step t be given the *plan*", not "decode 25 numbers in one shot".

**Pick: 4-layer transformer decoder, 512-d, 8 heads.** H=5 learned
positional queries cross-attend over VLM features and self-attend across
plan steps. Output per query: `y_t ∈ R^{2 · A_block · A_raw}` → split
into μ_t and pre-σ_t. Total params ~4–6 M — trivial next to the frozen
VLM.

If the transformer overfits or fails to train, the MLP fallback is
trivial (one ablation row).

### Choice 3 — Task / goal conditioning

LeWM tasks are **goal-conditioned**, so identical-looking observations
can require opposite actions depending on the goal. The head MUST see
the goal somehow.

Three regimes:

- **PushT, Cube, Reacher** — goal is a target rendered into the obs
  itself, OR describable in language ("push the T to the red target",
  "move the red cube to the green target"). Use the VLM's native
  cross-modal attention: pass `(image, text_goal)`, then read out the
  *language tokens'* last-layer hidden states. This forces the VLM to
  ground language in pixels and gives us a task-aware feature vector
  for free. No hand-crafted pooling.
- **Tworoom** — goal is a 2D position, awkward to verbalize. Render
  the goal as a separate top-down image and feed it as a second image
  to the VLM (Qwen2.5-VL supports multi-image input), or concatenate
  a small `(x, y)` MLP embedding to the pooled feature.
- **Universal fallback** — mean-pool visual tokens + trainable
  per-task embedding (`task_id ∈ {pusht, cube, …}`). Use this only if
  language conditioning underperforms.

**Pick: language-token readout when the task verbalizes naturally; goal-
image second input for Tworoom.** Both go through the same downstream
pipeline; only the pooling differs per task.

### Choice 4 — σ parameterization: fixed, learned-global, or learned-per-input

The oracle works at fixed σ = 0.1. We could just hard-code σ and only
learn μ. But the head sees the input — it can know when to be confident
("clear push") vs uncertain ("near goal, multiple valid actions"). That
information is wasted with fixed σ.

**Pick: per-input σ via softplus head, with a floor σ_min = 0.05 and a
deployment-time global multiplier `vlm_sigma_scale`.**

```
log_σ_raw = transformer_output[..., A_block·A_raw:]
σ_per_input = softplus(log_σ_raw) + σ_min
σ_at_deploy = σ_per_input × vlm_sigma_scale
```

This gives us:
- Aleatoric awareness via `σ_per_input` (model says "I'm not sure").
- A global trust dial via `vlm_sigma_scale` for the paper sweep
  (mirrors oracle's `oracle.sigma`).
- A floor that prevents the head from making over-confident bad
  predictions during MPPI init.

### Choice 5 — Training objective: NLL vs β-NLL vs distill-from-oracle vs MSE

Plain Gaussian NLL has a well-known pathology: when μ is hard to fit,
the gradient pushes σ up to lower the loss without improving the mean.
β-NLL (Seitzer et al. 2022) scales the loss by `σ^β` to counter this;
β = 0.5 is the standard robust default.

Distill-from-oracle (use the KNN-retrieved expert sequence as the
training target with fixed σ=0.1) is conceptually cleaner — it directly
imitates what works at 92% SR — but it inherits KNN's failure modes
(see Tworoom: KNN can't generalize, so a head distilled on KNN can't
either). The whole point of a generative prior is to *escape* KNN's
ceiling, so distillation is wrong in spirit.

**Pick: β-NLL (β = 0.5) on expert (obs, action_seq) pairs from the h5,
on normalized actions.** Use the same StandardScaler the eval pipeline
uses (CLAUDE lesson #1, #2). Add weight decay 1e-4, AdamW, cosine LR
with peak 3e-4 and 1k step warmup, batch ~256 sequences.

Run **distill-from-oracle as an ablation row** — if BC catches up to or
beats it, the paper has a clean comparison; if BC trails badly, that
itself is a finding worth reporting.

---

## 9.3 The full v1 architecture (one paragraph)

> Frozen VLM (Qwen2.5-VL-3B) takes `(o_t, language_goal)` and emits
> per-token last-layer hidden states. We read the language-token
> hiddens (or mean-pool visual tokens + task embedding when no clean
> language goal exists) into a feature vector `c ∈ R^{2048}`. A
> 4-layer transformer decoder with 512-d hidden, 8 heads, and `H = 5`
> learned positional queries cross-attends over the VLM hiddens and
> self-attends across plan steps. Each query emits
> `R^{2 · A_block · A_raw}`, split into `μ_t` (mean) and `log_σ_t`
> (raw). σ is `softplus(log_σ) + 0.05`. At deploy time, σ is multiplied
> by `vlm_sigma_scale` and PoG-fused with MPPI's default
> `N(0, var_scale·I)`.

---

## 9.4 What we are explicitly NOT doing in v1, and why

| Skipped                             | Reason                                                                                                                                |
| ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| End-to-end VLM fine-tune            | Expensive, fragile, obscures the contribution. Keep VLM frozen — only the head trains.                                                |
| Mixture-of-Gaussians head           | Changes the PoG fusion math. Defer to ablation §6.4 follow-up.                                                                        |
| Diffusion / flow head               | Doesn't yield a Gaussian for fusion. Not compatible with the mechanism.                                                               |
| Learn `vlm_sigma_scale`             | No training signal aligned with downstream SR. Sweep at eval time.                                                                    |
| Predict raw (un-normalized) actions | Would require denormalizing inside MPPI. CLAUDE lesson #1, #2: stay in StandardScaler space.                                          |
| Per-MPPI-iteration prior updates    | The mechanism (§3) fuses *once at init*. Re-fusing would change the math and erase MPPI's exploration.                                |
| Caching VLM features into the head  | We DO cache VLM features to disk for *training* speedup, but at inference time they must be recomputed each env step (state changes). |

---

## 9.5 Training and eval recipe

### Training

- **Data:** the same h5 files LeWM's world model trains on
  (`pusht_expert_train.h5`, `ogbench/cube_single_expert.h5`,
  `tworoom.h5`, `dmc/reacher_random.h5`). No new dataset is introduced
  vs the baseline — we just extract a different signal (action distribution
  given obs+goal) from data the predictor already consumes. Pre-extract VLM
  features once per (episode, frame) and cache to
  `.stable-wm/vlm_cache/<task>/...`. Cache size estimate: ~50 GB per task
  at fp16.
- **Reacher caveat — file is `_random`, not `_expert`.** BC on a random
  policy gives a random prior, which would *hurt* MPPI. Fix at data-load
  time: filter rows by `f['success'] == True` (or top-quantile `f['score']`)
  and only train on the surviving subset. If too little data survives,
  drop Reacher from the BC story entirely and use it only for the K-sweep
  curves where dataset quality doesn't matter.
- **Filter NaN actions** (CLAUDE lesson #4: `reacher_random` has NaNs
  at episode boundaries). Apply `valid_mask = ~np.isnan(actions).any(axis=1)`.
- **Normalize actions** with the same StandardScaler used by `eval.py`.
- **Loss:** β-NLL with β = 0.5, summed over `H · A_block · A_raw` dims,
  averaged over batch.
- **Validation:** held-out 10% of episodes, track NLL + per-step MSE
  vs expert, plus a small batch of 5 rollouts every 5k steps to track
  downstream SR — the loss is a proxy, SR is the truth.
- **GPU budget:** ~1 RTX 5090 day per task end-to-end, dominated by
  the one-time VLM feature caching (~6h) and 24h of head training.

#### A note on the "expert-data critique"

A reviewer will reasonably ask: *"LeWM's world model doesn't require
expert data; your projection head does. Isn't your method just BC in
disguise?"* Two-part defense:

1. **Same files, stronger quality assumption.** We use the exact h5
   datasets that ship with LeWM — no extra collection. The honest
   delta is: the dynamics learner can train on mixed-quality data, the
   prior head benefits from expert-quality data. State this upfront.
2. **The mechanism is what's novel, not the prior source.** PRISM-MPPI
   = imperfect prior + closed-loop replanning, fused via PoG. Both
   pieces are necessary; neither alone suffices. The §9.5 eval block
   below proves this with explicit BC-as-policy and warm-start
   ablations.

### Eval

- Drop the head's `(μ, σ)` into `prism_mppi.py` exactly where the
  oracle's tensor goes today. Do *not* re-touch the fusion code.
- Sweep `vlm_sigma_scale ∈ {0.3, 0.5, 1.0, 2.0, 5.0}`.
- Rerun Phase 2 K-sweep (§7) with VLM prior to confirm the
  compute-quality story still holds.

The full ablation table (these are not optional — every row is needed
to defuse a specific reviewer concern):

| # | Method | What it isolates | PushT target |
|---|---|---|---|
| 1 | Baseline MPPI (no prior) | "Does the world model alone work?" — sanity floor. | 60% (known) |
| 2 | CEM (LeWM upstream solver) | "Is MPPI a reasonable solver to bolt PRISM onto?" — addresses the "you picked a weak baseline" critique. | 96% (known) |
| 3 | **BC head as direct policy** (sample from `μ_BC`, no MPPI) | **Is the head doing all the work?** If this row ≈ row 5, the mechanism is contributing nothing. | ? — must measure |
| 4 | BC head + MPPI warm-start (use `μ_BC` as MPPI initial mean, but with normal `σ_π = var_scale`, no PoG fusion) | "Is precision-weighting actually doing anything, vs just centering MPPI on the prior?" Isolates PoG vs trivial warm-start. | ? — must measure |
| 5 | **VLM-PRISM (full method)** | The headline number. | ≥76% (§6.4 go/no-go) |
| 6 | Oracle PRISM (KNN prior) | "How close did we get to the prior-quality ceiling?" | 92% (known) |
| 7 | Distill-from-oracle PRISM (head trained on KNN targets, not BC on expert) | "Does training-signal choice matter?" Cheap diagnostic. | ? — must measure |
| 8 | (Stretch) Zero-shot VLM PRISM (prompt Qwen for actions, no BC training) | "Is even a *minimal-supervision* VLM useful?" Strongest hedge against the data-requirement critique. | ? — must measure |

The reading we want from this table:

- Row 3 (BC alone) **substantially below** row 5 (PRISM) — proves the
  mechanism contributes beyond the prior.
- Row 4 (warm-start) **below** row 5 — proves PoG fusion is doing
  more than dumb mean-substitution.
- Row 5 **between** row 1 and row 6 — proves the prior helps but
  doesn't have to be perfect.
- Row 7 ≤ row 5 — confirms BC is a stronger training signal than
  oracle distillation (because it can generalize beyond the dataset).
- Row 8 > row 1 — would be a remarkable bonus, says the mechanism
  bootstraps even an unsupervised prior.

Rows 3, 4, 5 are the load-bearing comparison: they determine whether
the paper is "VLM helps planning" (vulnerable) or "PoG fusion turns an
imperfect learned prior into a closed-loop planner that beats both
components" (much harder to attack).

### Tworoom — the key acid test

Tworoom oracle KNN tops out at 27% (CLAUDE lesson #5–6, §7.3). If the
VLM head clears, say, 50% on Tworoom, **that single number sells the
paper**: it's direct evidence that a generative prior beats retrieval
exactly where retrieval breaks down. Run this experiment early, even
before fully tuning the easier tasks.

---

## 9.6 Risks and what we'd do if v1 underperforms

| Symptom | Likely cause | Mitigation |
|---|---|---|
| Low SR on PushT, head NLL is low | VLM doesn't see PushT's abstract rendering well (§6.4 risk) | Switch backbone: SigLIP or R3M, or add a 3-layer adapter trained on PushT |
| σ collapses to floor everywhere | β-NLL not aggressive enough for this data | Increase β to 1.0; add a KL penalty on σ |
| Head matches expert mean but SR doesn't move | Expert is multimodal in this state — single Gaussian averages two modes into bad action | Promote MoG ablation to v1.1; meanwhile widen σ floor |
| Tworoom doesn't beat KNN | Goal conditioning didn't take | Switch from text goal to second-image goal; add explicit `(x, y)` goal embedding |
| Cube vs PushT split: VLM helps one, hurts the other | Backbone–task mismatch | Per-task backbone choice — paper-acceptable as long as we're upfront |
| **BC-as-policy ≥ PRISM-MPPI on most tasks** (§9.5 row 3 ≈ row 5) | The head is the policy; PRISM's mechanism contributes nothing on these tasks | **Reframe the paper**: pivot from "VLM prior for planning" to "closed-loop replanning under an imperfect learned prior." Showcase PRISM's edge under (a) execution noise / action perturbations, (b) OOD initial states, (c) goal generalization to unseen targets — settings where open-loop BC fails but closed-loop MPPI recovers. The σ-trust knob also becomes the headline (BC alone has no equivalent). |
| Warm-start ≈ PoG (§9.5 row 4 ≈ row 5) | The precision-weighting machinery isn't earning its keep — a trivial mean substitution does the same job | Stress that PoG's contribution is *uncertainty-aware*: report results under a **noisy** prior (corrupted BC predictions, or low-quality data subsets) where warm-start collapses but PoG's variance-weighting recovers gracefully. If even that fails, demote PoG to "principled re-derivation of warm-start" and lead with the σ-knob ablation as the main finding. |
| Reviewer says "MPPI is a weak baseline; CEM is what real LeWM users run" (CLAUDE lesson #10) | Fair critique — CEM hits 96% on PushT vanilla | Lead with the §9.5 row-2 vs row-5 comparison: PRISM-MPPI matches CEM-quality SR at MPPI's compute budget. The story is "PRISM lets you keep MPPI's per-step compute and get CEM-grade outcomes." |
| Reviewer says "you require expert data, baseline doesn't" | Real concern (see §9.5 expert-data note) | Lead with row 8 (zero-shot VLM PRISM) if it shows any uplift; if it doesn't, fall back to "same h5 files as the world model, no extra collection" + emphasize that the mechanism is invariant to prior source. |

---

## 9.7 Decision summary

| Choice | v1 commitment | Easy ablation |
|---|---|---|
| Output | per-step diagonal Gaussian | flat-MLP output |
| Backbone | 4-layer transformer decoder, 512-d | MLP head (PRISM-Newt style) |
| Conditioning | language-token readout (or goal-image for Tworoom) | mean-pool + task embedding |
| σ | per-input softplus + floor 0.05 + global scale | fixed σ = 0.1 |
| Loss | β-NLL, β=0.5, on normalized actions | distill-from-oracle, MSE |
| VLM | Qwen2.5-VL-3B, frozen | SigLIP, R3M, VC-1 |

The "really smart" choices vs a naive baseline:
1. **Transformer decoder over MLP** — captures action-sequence
   coherence at near-zero param cost.
2. **Language-token readout for task conditioning** — leverages the
   VLM's existing cross-modal attention instead of bolting a new
   pooling head on top.
3. **β-NLL with σ floor + global scale** — robust optimization, plus
   keeps the σ trust knob the paper needs for sweeps.
4. **Cache VLM features to disk** — makes per-task fine-tunes
   tractable on a single 5090.
5. **Train on expert BC, not oracle distillation** — the whole point
   of the VLM is to break KNN's retrieval ceiling.

---

## 9.8 Implementation order (Phase 3.2 sub-tasks)

Suggested ordering when we move from design to code:

1. Stand up the VLM forward + caching script (one task: cube).
2. Write the head (`vlm_head.py`) and training loop
   (`train_projection_head.py`), validate on cached features.
3. Write `vlm_policy.py` mirroring `oracle_policy.py`'s contract but
   sourcing the prior from the head.
4. Smoke test on 5 PushT episodes.
5. Full eval matrix: PushT, Cube, Reacher, Tworoom × σ sweep.
6. Write up vs oracle ceiling (§6.4 go/no-go).

Each step independently testable; if step 5 fails we still have a
training pipeline and a head we can iterate on.
