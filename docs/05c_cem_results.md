# 05c. CEM Ablation Results (2026-05-02 → 2026-05-03)

Vanilla CEM and CEM-warm-start on Cube and PushT, 3 seeds each (`{0, 1, 42}`),
N=50 episodes per cell. Companion to §10 (Cube MPPI), §11 (PushT MPPI),
and §05b (baseline gap analysis).

Originally run 2026-05-02 with seeds `{1, 2, 42}`. On 2026-05-03, seed=0
was added and seed=2 deprecated to align with MPPI's multi-seed list
(`{0, 1, 42}`) so all three seeds are paired across planner families.
The deprecated `{1, 2, 42}` numbers are retained in appendix §A for
reproducibility. See §1.1 for the rationale.

---

## 1. Setup

### Per-cell config

- **Solver:** `stable_worldmodel.solver.CEMSolver` (vanilla) and a new
  `prism_cem.PrismCEMSolver` subclass (warm-start) — see §2 for the
  warm-start implementation.
- **CEM hyperparameters** (from `config/eval/solver/cem.yaml`):
  num_samples=300, n_steps=30, topk=30, var_scale=1.0, seed=`${seed}`.
  These match our MPPI defaults (apart from MPPI's `temperature=0.5`).
  *Note:* the LeWM paper uses `n_steps=10` for OGBench-Cube but `n_steps=30`
  for PushT (App. D). We left `n_steps=30` for both — the same setting
  used in our §10 / §11 MPPI experiments — so this matches *us*, not the
  paper's per-task tuning.
- **N=50 episodes** per cell.
- **Seeds:** `{0, 1, 42}` — same as MPPI multi-seed in §10.9 / §11.9, so
  all per-seed comparisons across planner families are paired.
- **Tasks:** Cube and PushT only. Tworoom NOT run.
- **Head checkpoints:** unchanged from §10 / §11 — `prior_head_cube.pt`
  (best-by-val-NLL at epoch 7) and `prior_head_pusht.pt` (best at
  epoch 1). NOT retrained.
- **Action StandardScaler:** unchanged. Bit-exact match between
  head-train fit and eval-time fit verified for PushT in §11.5.1; same
  fit code path used here.
- **JEPA cache:** unchanged. The cached features were not consumed by
  these CEM runs (CEM at eval time does not use the cache; the head
  re-runs the JEPA encoder live on the current obs and goal obs).

### 1.1 Seed list correction (2026-05-03)

The original 12-cell run on 2026-05-02 used seeds `{1, 2, 42}` per an
explicit instruction in that night's spec. The same spec also stated
the alignment goal "和MPPI保持一样" (same as MPPI), but MPPI's multi-seed
runs (§10.9, §11.9) used `{0, 1, 42}`. I executed `{1, 2, 42}` per the
explicit list and surfaced the inconsistency in the morning review.

Decision (this morning): align with MPPI by adding seed=0 cells and
demoting seed=2. Concretely:
- 4 new cells run today: Cube/PushT × {vanilla CEM, CEM-warm-start} at
  seed=0, N=50 each. All four 15-pp swing checks pass against the
  seed-1/seed-42 means already on file.
- Headline tables in §4–§6 below now use `{0, 1, 42}`. All Δ rows in §5
  are now paired across all 3 seeds (each seed eval'd identical
  episodes across planner families because the sampling is keyed
  deterministically on `cfg.seed`; verified §A.1 of
  `docs/05c_appendix_82pct_investigation.md`).
- The deprecated `{1, 2, 42}` numbers are retained in **appendix §A**
  (half-page summary) for reproducibility but are not used for any
  paper claim.

---

## 2. Pre-flight sanity check results

### 2.1 Does CEM expose an init-mean interface?

**Yes.** `stable_worldmodel/solver/cem.py:init_action_distrib(actions=None)`
returns `(mean, var)`, where `mean` is initialized from the `actions`
argument when provided. `solve(info_dict, init_action=...)` forwards
`init_action` to `init_action_distrib`. This is the same interface
shape as MPPISolver (lines 84–97 of upstream `mppi.py`). The variable
named `var` is actually std — same upstream mislabel as MPPI. Sampling
distribution shape: `(n_envs, horizon, action_dim_flat)` — bit-identical
to MPPI's, so head outputs from `prior_head.py` plug in unmodified.

### 2.2 New code: `prism_cem.py` (PrismCEMSolver)

51-line subclass mirroring `PrismMPPISolver`'s warm-start branch only
— no PoG. `set_prior(μ, σ=None)` stores μ, **discards σ by design**
(CEM-warm-start is the no-precision-anchoring baseline). `init_action_distrib`
replaces `mean` with μ if a prior is set; std unchanged at the planner
default. CEM's elite selection, σ update rule, and core algorithm are
**not modified**.

### 2.3 Dry runs (1 episode each, Cube, seed=1)

| Run | Outcome | Verified |
|---|---|---|
| Vanilla CEM (`solver=cem`, no head) | success_rate 100% (N=1, noise) | pipeline runs end-to-end |
| CEM-warm-start (`solver=cem`, `injection_mode=warm_start`) | solver class = `PrismCEMSolver` ✓; pre-inject `mean[0,0,:5] = [0,0,0,0,0]`; **post-inject `mean[0,0,:5] = [0.21, -0.14, 1.02, 0.54, 1.61]`** ← head's μ injected; std unchanged at 1.0 ← head's σ correctly ignored | both injection conditions hold |

Both gates passed; proceeded to the 12-cell batch.

---

## 3. Headline table (single-seed seed=42, the directly-comparable column)

```
                                    Cube      PushT
─────────────────────────────────────────────────────
vanilla MPPI            seed=42      48%       60%       (§05 historical / §10)
PRISM-MPPI σ=2.0        seed=42      84%       74%       (§10 / §11.9 multi-seed table)
vanilla CEM             seed=42      70%       96%       (this doc)
CEM-warm-start          seed=42      82%       94%       (this doc)
─────────────────────────────────────────────────────
Reference: §05 historical
  Random (LeWM Fig. 6)             ~48%      ~33%
  Oracle PRISM σ_abs=0.1            80%       92%
  LeWM paper Fig. 6 / Table 5       74%      92–96%
```

PushT vanilla MPPI seed=42 = 60% is the §05 historical reference; the
seed=0 re-run gave 66% in §11.5.1. We use the seed-42 number above
because the column is anchored at seed=42.

---

## 4. Multi-seed table (3 seeds = `{0, 1, 42}`, mean ± std)

```
Task   Method            seed=0   seed=1   seed=42   mean ± std (sample)
─────────────────────────────────────────────────────────────────
Cube   vanilla CEM         62       58       70       63.3 ± 6.11
Cube   CEM-warm-start      80       82       82       81.3 ± 1.16
PushT  vanilla CEM         94       86       96       92.0 ± 5.29
PushT  CEM-warm-start      90       92       94       92.0 ± 2.00
```

Per-cell N=50. All four cells survive the 15-pp cross-seed swing check;
deviations of seed=0 from the seed-1/seed-42 mean of each cell:

| Cell | seed=0 | seed-1/seed-42 mean | Δ |
|---|---|---|---|
| Cube vanilla CEM | 62 | 64 | 2 pp |
| Cube CEM-warm-start | 80 | 82 | 2 pp |
| PushT vanilla CEM | 94 | 91 | 3 pp |
| PushT CEM-warm-start | 90 | 93 | 3 pp |

All ≤ 15 pp. No outlier seed.

**Update on the 82% pattern:** with seed=0 added, Cube CEM-warm-start
runs are `{80, 82, 82}` — seed=0 broke the exact-41/50 triple from the
deprecated `{1, 2, 42}` set. Aggregating across the 4 distinct seeds
{0, 1, 2, 42} (deprecated seed=2 included for context only), the
overall SR is `(40+41+41+41) / 200 = 81.5%` with binomial 95% CI
≈ [75.5%, 86.6%]. Outcome (b) — independent binomial coincidence —
holds; see appendix `docs/05c_appendix_82pct_investigation.md` for
the full investigation. The {0, 1, 42} headline mean is **81.3 ± 1.2**.

---

## 5. Direct comparisons

All Δ rows below report **paired per-seed differences**, mean ± sample-std,
unless noted.

### 5.1 Δ (CEM-warm-start − vanilla CEM)

```
Task   per-seed Δ (s=0, s=1, s=42)   mean Δ ± std    paired-t (df=2)
─────────────────────────────────────────────────────────────
Cube   +18, +24, +12                 +18.0 ± 6.00    t≈5.20, p≈0.035
PushT   -4,  +6,  -2                  +0.0 ± 5.29    t≈0.00, p≈1.00
```

### 5.2 Δ (CEM-warm-start − PRISM-MPPI σ=2.0) — fully paired across all 3 seeds

PRISM-MPPI σ=2.0 numbers come from §10.9 (Cube) and §11.9 (PushT)
multi-seed tables, both at seeds `{0, 1, 42}`. Identical seed →
identical 50-episode sample (sampling is deterministic in `cfg.seed`),
so the per-seed Δ is paired.

```
Task   seed   PRISM-MPPI σ=2.0   CEM-warm-start   Δ
──────────────────────────────────────────────────────
Cube    0     76%                80%              +4
Cube    1     72%                82%              +10
Cube    42    84%                82%              −2
PushT   0     82%                90%              +8
PushT   1     86%                92%              +6
PushT   42    74%                94%              +20
```

```
Task   per-seed Δ (s=0, s=1, s=42)   mean Δ ± std    paired-t (df=2)
─────────────────────────────────────────────────────────────────
Cube    +4, +10, -2                  +4.0 ± 6.00     t≈1.16, p≈0.37
PushT   +8, +6, +20                  +11.3 ± 7.57    t≈2.59, p≈0.12
```

- **Cube:** mean Δ = +4.0 pp; CI spans zero; 2/3 seeds positive, 1/3
  negative. Not significant at n=3.
- **PushT:** mean Δ = +11.3 pp; **3/3 seeds positive** (every seed
  shows CEM-warm > PRISM-MPPI). Paired-t = 2.59 with df=2 → p ≈ 0.12;
  not significant at the n=3 level but the consistent direction is a
  stronger signal than the p-value alone suggests.

### 5.3 Δ (vanilla CEM − vanilla MPPI)

Vanilla MPPI baseline numbers come from §05 / §10 / §11.5.1:

```
Task   seed   vanilla MPPI   vanilla CEM   Δ
──────────────────────────────────────────────
Cube   42     48% (§10)      70%           +22 pp
PushT  42     60% (§05)      96%           +36 pp
PushT   0     66% (§11.5.1)  94%           +28 pp
```

CEM substantially exceeds MPPI on the vanilla baselines: +22 pp on
Cube at seed=42; +28 to +36 pp on PushT depending on seed. Consistent
with §05b's planner-gap analysis.

---

## 6. Observations (factual only)

1. **Cube: CEM-warm-start improved over vanilla CEM by +18.0 ± 6.00 pp
   (paired, n=3, seeds `{0, 1, 42}`).** All three Cube seeds had
   warm-start > vanilla CEM. Paired-t ≈ 5.20, df=2, p ≈ 0.035.

2. **PushT: CEM-warm-start did not differ from vanilla CEM in mean.**
   Δ = 0.0 ± 5.29 pp (paired, n=3); 1 of 3 seeds positive, 2 of 3
   negative; CI spans zero. Mean SR identical at 92.0% for both methods.

3. **Vanilla CEM substantially exceeded vanilla MPPI on every paired
   comparison.** seed=42: Cube +22 pp (70 vs 48), PushT +36 pp
   (96 vs 60). seed=0: PushT +28 pp (94 vs 66). The Cube vanilla MPPI
   number 48% matches the LeWM paper's reported "Random" baseline
   (~48% per §05b).

4. **PRISM-MPPI σ=2.0 mean SR was below CEM-warm-start mean SR on
   both tasks**, with all seeds paired:
   - Cube: PRISM-MPPI 77.3% vs CEM-warm 81.3% (Δ = +4.0 ± 6.0 pp,
     2/3 seeds positive, p ≈ 0.37 — not significant at n=3).
   - PushT: PRISM-MPPI 80.7% vs CEM-warm 92.0% (Δ = +11.3 ± 7.6 pp,
     **3/3 seeds positive**, p ≈ 0.12 — not significant at n=3 but
     direction is consistent across all seeds).

5. **The Cube CEM-warm-start "82% triple" from the deprecated
   `{1, 2, 42}` set did not extend to seed=0.** With seed=0 added
   (today's run), Cube CEM-warm-start values are `{80, 82, 82}` —
   seed=0 broke the exact-41/50 pattern by exactly one episode
   (40/50 instead of 41/50). The aggregate over the 4 distinct seeds
   {0, 1, 2, 42} is 163/200 = 81.5% with binomial 95% CI ≈
   [75.5%, 86.6%]. Outcome (b) — independent binomial coincidence —
   is the explanation; see appendix `docs/05c_appendix_82pct_investigation.md`.

6. **Variance across seeds within the same (planner × task × method)
   cell:** at N=50, CEM cells span std 1.2–6.1 pp, MPPI cells (§10.9,
   §11.9) span 6.1–8.0 pp. Cube CEM-warm-start has notably low
   per-seed variance (1.2 pp) — consistent with a population SR
   ≈ 81% under binomial-N=50 sampling, not necessarily a planner
   property.

---

## 7. Confirmed assumptions

| Assumption | Verification |
|---|---|
| Head checkpoints unchanged (`prior_head_cube.pt`, `prior_head_pusht.pt`) | mtimes are 2026-05-01 14:54 and 2026-05-02 10:59 — predates this experiment. Not retrained. |
| Action StandardScaler unchanged | eval-time scaler is fit live in `eval_prism_head.py` from the same h5 column the head's saved scaler was fit on; bit-exact match verified for PushT in §11.5.1. Same fit code path runs here. |
| JEPA cache unchanged | `.stable-wm/jepa_cache/{cube,pusht}/` files unmodified. The CEM warm-start path does **not** consume the cache; the head re-runs the JEPA encoder on each replan via `PrismHeadPolicy._compute_z_t`. |
| Same eval episodes per seed across MPPI and CEM | Both `eval.py` and `eval_prism_head.py` route through `swm.World.evaluate_from_dataset` with the same seed-deterministic episode-sampling logic (`g = np.random.default_rng(cfg.seed)` followed by `g.choice(...)` over valid starting points). At identical seeds, the sampled episodes are identical regardless of solver. After the seed-list correction (§1.1), all three seeds `{0, 1, 42}` are paired across MPPI (§10.9 / §11.9) and CEM matrices. The deprecated `{1, 2, 42}` set is appendix-only (§A). |
| `prism_mppi.py` and head training code untouched | `git status` shows no modifications to `prism_mppi.py`, `prior_head.py`, `train_prior_head.py`, `prism_head_policy.py`. New files added: `prism_cem.py` (51 LoC). `eval_prism_head.py` modified to add solver dispatch (8 LoC change). |
| No closed-form PoG implemented for CEM | Confirmed in `prism_cem.py` — `set_prior` stores μ only; `σ` argument is accepted for API compatibility but not stored or used. |
| No modification to CEM's elite selection or σ update | Confirmed: `PrismCEMSolver` overrides only `init_action_distrib`. The body of `solve()` (lines 99+ of upstream cem.py) runs unmodified via `super().init_action_distrib(...)` returning the prior-injected mean to the parent's solve loop. |

---

## Appendix A — Deprecated `{1, 2, 42}` results (superseded by §4)

The original 12-cell run on 2026-05-02 used seeds `{1, 2, 42}` per the
spec's explicit list. Those results are retained below for
reproducibility; **they are not used for any paper claim**. Headline
tables in §4–§6 use the corrected `{0, 1, 42}` set.

### A.1 Multi-seed table (deprecated seeds)

```
Task   Method            seed=1   seed=2   seed=42   mean ± std
─────────────────────────────────────────────────────────────────
Cube   vanilla CEM         58       70       70       66.0 ± 6.93
Cube   CEM-warm-start      82       82       82       82.0 ± 0.00
PushT  vanilla CEM         86       88       96       90.0 ± 5.29
PushT  CEM-warm-start      92       84       94       90.0 ± 5.29
```

### A.2 Δ values (deprecated seeds)

```
Δ (warm − vanilla CEM)
  Cube  : +24, +12, +12  →  +16.0 ± 6.93 pp  (paired-t≈4.00, p≈0.06)
  PushT :  +6,  -4,  -2  →   +0.0 ± 5.29 pp

Δ (warm − PRISM-MPPI σ=2.0) — UNPAIRED on seed=2 vs MPPI's seed=0
  Cube  : (mean − mean) +4.7 pp;  paired-on-{1,42} mean +4.0 pp
  PushT : (mean − mean) +9.3 pp;  paired-on-{1,42} mean +13.0 pp
```

### A.3 Why deprecated

`{1, 2, 42}` left only seed=42 paired with MPPI's `{0, 1, 42}`, so the
"Δ vs PRISM-MPPI σ=2.0" comparison was unpaired across 2 of 3 seeds.
The {0, 1, 42} re-run added seed=0 and dropped seed=2 from the headline
analysis (run on 2026-05-03; raw seed=2 logs preserved at
`results/cem_logs/{vanilla,warm}_{cube,pusht}_seed2.log`). The
qualitative conclusions are unchanged across the two seed sets:

- Cube CEM-warm > vanilla CEM (with a 16 vs 18 pp mean Δ depending
  on seed set).
- PushT CEM-warm ≈ vanilla CEM (Δ = 0 pp in both seed sets).
- CEM-warm mean SR ≥ PRISM-MPPI σ=2.0 mean SR on both tasks (with
  PushT showing a consistent 3/3 positive direction in `{0, 1, 42}`).
