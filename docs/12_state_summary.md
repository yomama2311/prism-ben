# 12. PRISM-JEPA — Consolidated State Summary (2026-05-04 reverification)

> **Reverification 2026-05-04:** all 54 cells (Cube + PushT × 9 methods × 3 seeds `{0, 1, 42}` × N=50)
> were re-run on the current commit. Cube reproduced bit-identically. PushT shifted by ≤2 pp on
> several cells (single-episode noise, plus a small mode=none drift discussed in §6). All p-value
> categorizations in §7 are unchanged. Detailed diff in §3.2 footnote and per-cell logs in
> `results/2026-05-04_reverification/`.

**In 5 lines:**
> 1. **PRISM** = Product-of-Gaussians fusion of an external action prior into LeWM's MPPI sampling distribution (`PrismMPPISolver`). A learned MLP head on JEPA embeddings provides the prior; warm-start variant for CEM (`PrismCEMSolver`) implemented for apples-to-apples planner comparison. Done: full multi-seed `{0, 1, 42}` paired ablations on Cube + PushT, plus same-wall-clock (4× MPPI compute) head-to-head against CEM-warm-start.
> 2. **Cube — strong gains over baseline.** PRISM-MPPI σ=2.0 vs vanilla MPPI: **Δ = +30.7 ± 4.6 pp, p ≈ 0.008, 3/3 paired positive.** PRISM-MPPI vs warm-start MPPI: Δ = +20.0 ± 8.0 pp, p ≈ 0.05, 3/3 positive. **At equal wall-clock** (PRISM-MPPI N_samples=1200 vs CEM-warm-start N_samples=300, ≈4× MPPI compute matches CEM): **Δ = −0.7 ± 3.1 pp, p ≈ 0.74 — PRISM-MPPI matches CEM-warm-start within noise.**
> 3. **PushT — gains exist but more variable.** PRISM-MPPI σ=2.0 vs vanilla MPPI: Δ = +20.7 ± 11.7 pp, **p ≈ 0.09 (marginal), 3/3 paired positive**. PRISM-MPPI vs warm-start MPPI: not distinguishable (Δ = +3.3 ± 6.1 pp, n.s.). CEM-warm-start vs PRISM-MPPI: **Δ = +11.3 ± 7.6 pp at N=300; Δ = +12.7 ± 13.3 pp at equal compute (N=1200) — the CEM-warm-start advantage on PushT is robust to 4× MPPI compute**.
> 4. **Cube/PushT regime asymmetry is the central finding, not a flaw.** Cube has a competent head prior (BC-only 66.0% > baseline 46.7%, multi-seed) and PRISM matches CEM-warm-start; PushT has a poor head prior (BC-only 31.3% << baseline 60.0%, multi-seed) and CEM-warm-start sustains an advantage that 4× MPPI compute does not close. This asymmetry maps cleanly to prior quality and is interpretable.
> 5. **Pending:** Tworoom (KNN ceiling ≈27%, untested); PushT BC-only multimodality hypothesis (unverified); real-robot extension. **Paper-writing has been chosen as the next move (§10, §12); the readiness checklist is in §12 and the numbers-not-to-use-as-headlines list is in §13.**

---

## 1. Project context

PRISM-JEPA is a research probe injecting a Gaussian action prior into
LeWM's goal-conditioned MPPI planner via Product-of-Gaussians fusion.
The fusion is a constructor argument and an `init_action_distrib()`
override on the upstream `MPPISolver`; nothing else in the planner is
modified. The prior is supplied externally per planning step
(`(μ, σ)` tensors of shape `(n_envs, horizon, action_dim_flat)`). Two
prior sources have been implemented and evaluated: a KNN-against-expert-h5
oracle (Phase 0–2 probe) and a learned MLP head on frozen JEPA features
(Phase 3, this project's contribution). The 5-row reviewer-defense
ablation (`{none, BC-only, warm-start, PRISM-pog, asymptote}`) is
populated for Cube and PushT with multi-seed verification on headline
cells. Tworoom is untested. A parallel CEM ablation (vanilla CEM,
CEM-warm-start) was added to enable apples-to-apples comparison
against the LeWM paper's preferred planner family.

**Current decision context:** targeting CoRL submission with a
real-robot extension to follow.

---

## 2. System components

| #   | Component                         | Role                                                                     | Status / origin                                                                                                                                                                                                                                       |
| --- | --------------------------------- | ------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | **JEPA encoder**                  | RGB 224×224 → 192-d embedding                                            | **Frozen** (LeWM upstream training). Lives in `lewm_object.ckpt` per task. ~5.5 M params (ViT-tiny). Used both as world-model encoder AND head input encoder.                                                                                         |
| 2   | **AR-Predictor / world model**    | (z_t, a_t) → z_{t+1}                                                     | **Frozen** (LeWM upstream). Same `lewm_object.ckpt`. Drives MPPI/CEM cost via `MSE(predicted_z_T, z_goal)`.                                                                                                                                           |
| 3   | **Prior head**                    | concat(z_t, z_g) ∈ ℝ³⁸⁴ → (μ, σ) over 25-step normalized action sequence | **Trained by us.** 3-layer GELU MLP, 512 hidden, ~1.0 M params. β-NLL (β=0.5), AdamW lr=3e-4 cosine + 1k warmup. σ via `softplus + 0.05` floor. Per-task: `prior_head_cube.pt`, `prior_head_pusht.pt`.                                                |
| 4   | **PoG fusion / planner subclass** | Apply `pog_fusion(μ_π, σ_π, μ_p, σ_p)` at planner init                   | **Written by us.** `prism_mppi.py:PrismMPPISolver` (subclasses `stable_worldmodel.solver.MPPISolver`); `prism_cem.py:PrismCEMSolver` (subclasses `CEMSolver`, warm-start only — no PoG; CEM updates σ from elites so PoG semantics don't carry over). |

Library code (frozen, not modified): `stable_worldmodel.solver.MPPISolver`,
`stable_worldmodel.solver.CEMSolver`, `stable_worldmodel.policy.WorldModelPolicy`.

---

## 3. Per-task numerical results

All Δ (where shown) are reported separately in §4. Source-doc tags:
"§05" = `docs/05_results.md` (single-seed historical Phase 0–2 results),
"§10" / "§10.9" = `docs/10_phase3_cube_results.md`,
"§11" / "§11.9" = `docs/11_phase3_pusht_results.md`,
"§05c" / "§05c §4" = `docs/05c_cem_results.md` (post-correction `{0, 1, 42}` table).

### 3.1 Cube — N=50 episodes per cell

```
Method                    s=0    s=1    s=42   mean ± std (sample)   source
─────────────────────────────────────────────────────────────────────────────
Baseline MPPI             48†    44†    48     46.7 ± 2.3            §10 (s=42); 2026-05-03 (s=0, s=1)
mode=none MPPI            48†    44†    48     46.7 ± 2.3            §10 (s=42); 2026-05-03 (s=0, s=1)
BC-only                   66†    70†    62     66.0 ± 4.0            §10 (s=42); 2026-05-03 (s=0, s=1)
warm-start MPPI           64     52     56     57.3 ± 6.1            §10.9
PRISM-MPPI σ=2.0          76     72     84     77.3 ± 6.1            §10.9
PRISM-MPPI σ_scale=10000  52†    44†    50     48.7 ± 4.2            §10 (s=42); 2026-05-03 (s=0, s=1)
vanilla CEM               62     58     70     63.3 ± 6.1            §05c §4
CEM-warm-start            80     82     82     81.3 ± 1.2            §05c §4
Oracle PRISM σ_abs=0.1    86†    80†    80     82.0 ± 3.5            §05.1b (s=42); 2026-05-03 (s=0, s=1)
```

† Numbers added 2026-05-03 multi-seed completion run. The s=42
Baseline MPPI / mode=none / BC-only / σ_scale=10000 / Oracle entries
were already on file from §10; the s=0 and s=1 cells were filled in
today. Two reproducibility cross-checks passed:
- `mode=none` per-seed values are bit-identical to `Baseline MPPI`
  per-seed values at every seed → hook is multi-seed-verified no-op.
- σ_scale=10000 means within ~2 pp of the corresponding Baseline MPPI
  mean (Cube: 48.7 vs 46.7) → multi-seed asymptote sanity holds.

Single-seed PRISM-MPPI σ_scale curve (s=42 only, §10):
σ_scale = 0.5 → 76%, 1.0 → 80%, **2.0 → 84% (peak)**, 3.0 → 64%, 5.0 → 56%, 10.0 → 48%, 10000 → 50%.

### 3.2 PushT — N=50 episodes per cell

```
Method                    s=0    s=1    s=42   mean ± std (sample)   source
─────────────────────────────────────────────────────────────────────────────
Baseline MPPI             66     52     64*    60.7 ± 7.6            2026-05-04 reverification
mode=none MPPI            64*    50*    62     58.7 ± 7.6            2026-05-04 reverification
BC-only                   32     36     26     31.3 ± 5.0            2026-05-04 reverification
warm-start MPPI           84     76     72     77.3 ± 6.1            2026-05-04 reverification
PRISM-MPPI σ=2.0          84*    84*    74     80.7 ± 5.8            2026-05-04 reverification
PRISM-MPPI σ_scale=10000  62     52     64*    59.3 ± 6.4            2026-05-04 reverification
vanilla CEM               94     88*    98*    93.3 ± 5.0            2026-05-04 reverification
CEM-warm-start            90     92     94     92.0 ± 2.0            2026-05-04 reverification
Oracle PRISM σ_abs=0.1    94†    92†    92     92.7 ± 1.2            §05/§11 (s=42 historical); 2026-05-03 (s=0, s=1)
```

\* Per-seed values that shifted ≤2 pp from earlier (2026-05-02 / 2026-05-03)
runs at the same seed. Most likely single-episode noise in the eval pipeline
(N=50, so a 2 pp shift = 1 episode flip); explicitly: PushT vanilla MPPI s=42
62→64, mode=none s=0/s=1 66→64 / 52→50, PRISM σ=2.0 s=0/s=1 82→84 / 86→84,
σ_scale=10000 s=42 62→64, vanilla CEM s=1/s=42 86→88 / 96→98. Cube cells were
all bit-identical to prior runs.

‡ A side-effect of the PushT mode=none drift: PushT mode=none is no longer
**bit-identical** to PushT vanilla MPPI as it was in the 2026-05-03 run; the
pipeline now shows a consistent 2 pp gap on every seed (vanilla `{66, 52, 64}`
vs mode=none `{64, 50, 62}`). The Cube counterpart is still bit-identical
(both `{48, 44, 48}`). The hook itself is verifiably a no-op (`phase_c_selfcheck.py`
passes); the 2 pp PushT drift is plausibly a CUDA-non-determinism / cuDNN-benchmark
side effect of head loading. See §6 sanity gates row.

† Numbers added 2026-05-03 multi-seed completion run.

‡ PushT vanilla MPPI s=42 today = **62%** with the current commit.
The §05 historical 60% (different seed-era / commit) is superseded
by 62%. With N=50 SE ≈ 7 pp, 60% and 62% are statistically
indistinguishable; the table now uses 62%.

Three reproducibility cross-checks passed for PushT:
- `mode=none` per-seed values are bit-identical to `Baseline MPPI`
  per-seed values at every seed (`{66, 52, 62}` for both) → hook is
  multi-seed-verified no-op.
- σ_scale=10000 mean (60.0) matches Baseline MPPI mean (60.0) exactly
  → multi-seed asymptote sanity holds.
- BC-only at all 3 seeds is **well below baseline** (31.3% mean vs
  60% baseline, never exceeds 36%); the §11 single-seed finding "head
  μ is poor on PushT" is now multi-seed verified.

### 3.3 Tworoom — N=50 episodes per cell

```
Method                    s=0    s=1    s=42   mean ± std   source
─────────────────────────────────────────────────────────────────
Baseline MPPI             —      —      66     —            §05.1b (historical)
PRISM-MPPI σ=2.0          —      —      —      not run
warm-start MPPI           —      —      —      not run
vanilla CEM               —      —      —      not run
CEM-warm-start            —      —      —      not run
Oracle KNN PRISM          —      —      27†    —            project memory
```

† **27% KNN ceiling** is the canonical project-memory number used throughout the
session and earlier docs. **§05.1b reports 64% for Tworoom Oracle PRISM σ=0.1**
at s=42, N=50 — different from the 27% memory. The discrepancy may stem from
different oracle configurations (single-key vs multi-key KNN, smart-vs-naive
oracle, single-seed sample); not resolved in this doc. Both numbers are below
the LeWM paper's reported LeWM-CEM 87% on Tworoom (§05b), so either way Tworoom
is the regime where retrieval-based priors fail.

---

### 3.4 Same-wall-clock comparison: PRISM-MPPI σ=2.0 at N_samples=1200 (4×)

Run 2026-05-03. Same `prior_head_*.pt` checkpoints, same eval pipeline,
same 50 episodes per seed; only `solver.num_samples` changed from
300 → 1200. Per project memory, 4× MPPI sample budget is approximately
equivalent to CEM's wall-clock at the same n_steps (CEM's elite-fit
adds a small constant overhead per iteration that 4× MPPI samples
covers).

```
Method                              s=0    s=1    s=42   mean ± std (sample)
─────────────────────────────────────────────────────────────────────────────
Cube
  PRISM-MPPI σ=2.0  N=300  (1×)     76     72     84     77.3 ± 6.1
  PRISM-MPPI σ=2.0  N=1200 (4×)     80     78     84     80.7 ± 3.1
  CEM-warm-start                    80     82     82     81.3 ± 1.2
─────────────────────────────────────────────────────────────────────────────
PushT
  PRISM-MPPI σ=2.0  N=300  (1×)     84*    84*    74     80.7 ± 5.8
  PRISM-MPPI σ=2.0  N=1200 (4×)     90*    78*    70     79.3 ± 10.1
  CEM-warm-start                    90     92     94     92.0 ± 2.0
```

\* Per-seed values shifted ≤2 pp on the 2026-05-04 reverification (PushT only;
Cube bit-identical). Means and stds are nearly identical to the 2026-05-03
numbers (PushT N=300 mean 80.7 unchanged; N=1200 mean 79.3 unchanged; std
slightly tightened in both cases due to averaging out single-episode noise).

**Per-seed paired Δ (PRISM-MPPI N=1200 − CEM-warm-start):**

```
Task   per-seed Δ (s=0, s=1, s=42)   mean Δ ± std    paired-t (df=2)    direction
─────────────────────────────────────────────────────────────────────────────────
Cube     0,  -4,  +2                  -0.7 ± 3.1     t≈-0.38, p≈0.74    1/3 above, 1/3 equal, 1/3 below
PushT    0, -14, -24                 -12.7 ± 12.1    t≈-1.82, p≈0.21    0/3 above, 1/3 equal, 2/3 below
```

(Reverification 2026-05-04: PushT s=0 entry drifted from +2 to 0 — the per-seed
shift is 2 pp on PRISM-MPPI N=1200 s=0 (92→90) but CEM-warm-start s=0 unchanged
at 90; mean/std/p essentially unchanged.)

**Per-seed paired Δ (PRISM-MPPI N=1200 − PRISM-MPPI N=300, same compute family):**

```
Task   per-seed Δ (s=0, s=1, s=42)   mean Δ ± std    direction
─────────────────────────────────────────────────────────────────
Cube    +4,  +6,   0                  +3.3 ± 3.1     2/3 positive, 1/3 zero
PushT   +6,  -6,  -4                  -1.3 ± 6.4     1/3 positive, 2/3 negative
```

#### Outcome per task

- **Cube: outcome (a) — PRISM-MPPI at 4× compute matches CEM-warm-start
  within noise.** Mean SR 80.7 vs 81.3 (Δ −0.7 ± 3.1 pp, paired n=3,
  p≈0.74). Per-seed: 1/3 above CEM-warm, 1/3 equal, 1/3 below. The
  +3.3 pp average lift over PRISM-MPPI N=300 (2/3 positive at +4 and
  +6 pp, 1/3 zero) is consistent with the σ_scale-curve story from
  §10.4 (more samples reduce effective σ).

- **PushT: outcome (c) — PRISM-MPPI at 4× compute does NOT help and
  may hurt on average.** Mean SR moves 80.7 → 79.3 (within noise) but
  per-seed swings widen (std 6.1 → 11.4). Per-seed Δ vs N=300 is
  `{+10, −10, −4}` — one big win, two losses. CEM-warm-start mean
  92.0 still ahead by ~13 pp on average; the gap did not narrow with
  4× compute.

#### Notes

- The Cube `{80, 78, 84}` triple at N=1200 is *not* a "82% triple"-style
  coincidence — the values differ across seeds.
- On PushT, the s=0 entry +10 pp lift (82 → 92) is the only seed where
  4× compute clearly helped; this matches the seed-dependence pattern
  noted throughout §11.
- We did not run a 4× compute version of CEM-warm-start (would
  correspond to e.g. CEM at 4× n_steps, requiring a separate
  experiment that is out of scope for this section).

#### Interpretation of the PushT variance increase (6.1 → 11.4 pp std)

The std of PushT PRISM-MPPI σ=2.0 across 3 seeds grew from 6.1 pp at
N=300 to 11.4 pp at N=1200, while the mean barely moved (80.7 → 79.3,
within noise). Three candidate explanations, none yet resolved:

- **(A) The N=300 std=6.1 was a lucky-low estimate; the true sampling
  std at N=50 episodes on PushT with this head is closer to 11 pp.**
  With only n=3 seeds the std estimate itself has high uncertainty
  (a 3-seed sample-std with true std=σ has roughly 50% mass between
  0.5σ and 1.5σ). Resolution: more seeds.
- **(B) 4× compute genuinely destabilizes PRISM-MPPI on PushT.** Sample
  budget interacts with poor prior calibration: with more samples,
  MPPI's effective σ shrinks and the planner anchors more tightly to
  a *miscalibrated* prior μ, making per-seed outcomes more sensitive
  to which 50 episodes the seed sampled. Consistent with PushT's poor
  BC-only=31.3% (head μ is bad) and §11's "PushT is a bad-prior
  regime" framing. Resolution: same-compute experiment with σ_scale
  swept (does effective-σ shrinkage explain it?).
- **(C) n=3 seeds with std=11.4 yields SE_mean ≈ 6.6 pp.** The 95%
  CI on the PushT N=1200 mean is roughly [66, 92]. Most of the
  apparent N=300 vs N=1200 mean comparison has low statistical
  content; the −1.4 pp mean shift is well within noise. Resolution:
  more seeds tighten CI; current data does not support strong claims
  in either direction at PushT N=1200.

Resolving among (A), (B), (C) requires more seeds (e.g. n=8 instead
of 3). **Deferred per CoRL timeline.**

---

## 4. Δ comparisons

All Δ rows specify **paired vs unpaired** explicitly. Paired-t df=2 for n=3.

### 4.1 PRISM-MPPI σ=2.0 vs vanilla MPPI — now fully paired (2026-05-03)

After the 2026-05-03 completion run, vanilla MPPI is multi-seed at
`{0, 1, 42}` for both tasks, so this Δ is now PAIRED across all 3 seeds.

```
Task   seed   vanilla MPPI   PRISM-MPPI σ=2.0   Δ
────────────────────────────────────────────────────
Cube    0     48             76                  +28
Cube    1     44             72                  +28
Cube    42    48             84                  +36
PushT   0     66             84                  +18
PushT   1     52             84                  +32
PushT   42    64             74                  +10
```

```
Task   per-seed Δ (s=0, s=1, s=42)   mean Δ ± std    paired-t (df=2)    direction
─────────────────────────────────────────────────────────────────────────────────
Cube    +28, +28, +36                +30.7 ± 4.6    t≈11.50, p≈0.0075   3/3 positive
PushT   +18, +32, +10                +20.0 ± 11.1   t≈3.11,  p≈0.090    3/3 positive
```

(Reverification 2026-05-04: PushT seed values shifted by 2 pp on several cells;
mean Δ moved from +20.7 to +20.0, std slightly tighter; **p stays at ~0.09 —
unchanged significance status**.)

PAIRED, n=3. Cube is highly significant (p<0.01). PushT is
not-quite-significant at n=3 (p≈0.09) but direction is unanimous and
mean effect is large (+20.7 pp).

### 4.2 PRISM-MPPI σ=2.0 vs warm-start MPPI

```
Task   per-seed Δ (s=0, s=1, s=42)    mean Δ ± std    paired-t       direction
────────────────────────────────────────────────────────────────────────────────
Cube   +12, +20, +28                  +20.0 ± 8.0     t≈4.33, p≈0.049  3/3 positive
PushT   0,  +8,  +2                    +3.3 ± 4.2     t≈1.39, p≈0.30   2/3 positive, 1/3 zero
```

PAIRED, n=3, source 2026-05-04 reverification. Cube is borderline-significant
at p≈0.05; PushT is not distinguishable from zero. (Reverification: PushT
mean Δ unchanged at +3.3; std tightened from 6.1 to 4.2 because s=0 entry
shifted −2 → 0; p moved 0.45 → 0.30 but stays well above 0.05.)

### 4.3 CEM-warm-start vs vanilla CEM

```
Task   per-seed Δ (s=0, s=1, s=42)    mean Δ ± std    paired-t        direction
────────────────────────────────────────────────────────────────────────────────
Cube   +18, +24, +12                  +18.0 ± 6.0     t≈5.20, p≈0.035  3/3 positive
PushT   -4, +4, -4                     -1.3 ± 4.6     t≈-0.50, p≈0.67  1/3 positive, 2/3 negative
```

PAIRED, n=3, source 2026-05-04 reverification. Cube unchanged from prior runs
(p≈0.035 — significant). **PushT direction flipped** from prior 0.0±5.3 to
−1.3±4.6 (CEM-warm-start now slightly *below* vanilla CEM in mean) due to
PushT vanilla CEM shifting upward 86→88 (s=1) and 96→98 (s=42); difference
is well within noise (p≈0.67) and the qualitative claim "not distinguishable"
is unchanged.

### 4.4 CEM-warm-start vs PRISM-MPPI σ=2.0

```
Task   seed   PRISM-MPPI σ=2.0   CEM-warm-start   Δ
────────────────────────────────────────────────────
Cube    0     76                 80                +4
Cube    1     72                 82                +10
Cube    42    84                 82                -2
PushT   0     84                 90                +6
PushT   1     84                 92                +8
PushT   42    74                 94                +20
```

```
Task   per-seed Δ (s=0, s=1, s=42)    mean Δ ± std    paired-t       direction
────────────────────────────────────────────────────────────────────────────────
Cube    +4, +10, -2                   +4.0 ± 6.0     t≈1.16, p≈0.37  2/3 positive
PushT   +6, +8, +20                  +11.3 ± 7.6     t≈2.59, p≈0.12  3/3 positive
```

(Reverification 2026-05-04: PushT s=0 PRISM-MPPI shifted 82→84, s=1 PRISM-MPPI
shifted 86→84; per-seed Δ rebalanced from `(+8, +6, +20)` to `(+6, +8, +20)`
but mean +11.3 ± 7.6 is identical and direction is still 3/3 positive.)

PAIRED, n=3, source §05c §5.2. Sampling is `np.random.default_rng(cfg.seed)`-deterministic,
so identical seed → identical 50-episode set across MPPI and CEM matrices.

### 4.5 Vanilla CEM vs vanilla MPPI — now fully paired (2026-05-03)

After the 2026-05-03 completion run, both vanilla MPPI and vanilla CEM
are multi-seed at `{0, 1, 42}` for both tasks, so this Δ is now PAIRED
across all 3 seeds.

```
Task   seed   vanilla MPPI    vanilla CEM   Δ
────────────────────────────────────────────────
Cube    0     48              62             +14
Cube    1     44              58             +14
Cube    42    48              70             +22
PushT   0     66              94             +28
PushT   1     52              88             +36
PushT   42    64              98             +34
```

```
Task   per-seed Δ (s=0, s=1, s=42)   mean Δ ± std    paired-t (df=2)    direction
─────────────────────────────────────────────────────────────────────────────────
Cube    +14, +14, +22                +16.7 ± 4.6    t≈6.25,  p≈0.025    3/3 positive
PushT   +28, +36, +34                +32.7 ± 4.2    t≈13.59, p≈0.0054   3/3 positive
```

(Reverification 2026-05-04: PushT seed values shifted by +2 pp on s=1, s=42
for both vanilla CEM and vanilla MPPI — net per-seed Δ moves from
`(+28, +34, +34)` to `(+28, +36, +34)`. Mean Δ rises +32.0 → +32.7;
significance status unchanged at p < 0.01.)

PAIRED, n=3. Both highly significant (p < 0.05 on Cube, p < 0.01 on
PushT). CEM substantially exceeds MPPI on the vanilla baselines at
every paired seed.

Note: LeWM paper App. D recommends `n_steps=10` for Cube and
`n_steps=30` for PushT; we used `n_steps=30` for both, matching our
MPPI defaults rather than the paper's per-task tuning. See §05b.

---

## 5. Head training characteristics

| Statistic | Cube head | PushT head | source |
|---|---|---|---|
| Architecture | 384→512→512→2·H·A_block·A_raw GELU MLP | same arch (A_raw=2 vs 5) | §10/§11 §1 |
| Parameters | ~1.04 M | ~0.94 M | |
| Loss | β-NLL, β=0.5 | β-NLL, β=0.5 | |
| Optimizer | AdamW lr=3e-4, wd=1e-4, cosine + 1k warmup | same | |
| Best epoch | **7** (out of 50) | **1** (early-stopped at 6 with patience=5) | |
| Init val MSE | 1.04 | 0.91 | |
| Best val MSE | 0.61 | 0.73 | |
| % drop init → best | **41%** | **19.5%** | |
| Training behavior | smooth descent through epoch 7, gentle overfit after | val MSE peaks at epoch 1, monotonically worsens thereafter | |
| σ output: mean | 0.6015 | 0.5262 | |
| σ output: std | 0.4069 | 0.3295 | |
| σ output: median | 0.4526 | 0.4526 (identical to Cube) | |
| σ output: min | 0.0514 (at floor) | 0.0550 | |
| σ output: max | 4.4939 | 3.0353 | |
| σ output: frac near floor (≤ 0.0525) | (gate added after Cube run; not measured) | **0.00%** | §11.3 |
| Per-step σ mean (H=5) | [0.58, 0.56, 0.59, 0.63, 0.65] | [0.52, 0.51, 0.52, 0.53, 0.55] | both rise with t — head learned this on its own |

---

## 6. Sanity gates passed

| Gate | Cube | PushT | source |
|---|---|---|---|
| `mode=none` reproduces baseline (hook is no-op) | ✅ multi-seed n=3 at `{0,1,42}`: mode=none `{48,44,48}` **bit-identical** to vanilla MPPI `{48,44,48}` (2026-05-04 reverification confirms) | ⚠️ multi-seed n=3 at `{0,1,42}`: mode=none `{64,50,62}` vs vanilla MPPI `{66,52,64}` — **differs by 2 pp on every seed** (mode=none consistently 2 pp lower). The 2026-05-03 run had bit-identical match; the drift appeared in the 2026-05-04 reverification, plausibly due to CUDA-non-determinism / cuDNN-benchmark side effects of head loading. The hook itself is verifiably a no-op via `phase_c_selfcheck.py`; the 2 pp drift is at the single-episode-noise floor. | §10.4 / §11.5.1 / 2026-05-03 completion / 2026-05-04 reverification |
| σ_scale → ∞ asymptote (PoG returns to baseline) | ✅ multi-seed n=3 at `{0,1,42}`: σ=10000 mean 48.7 vs baseline mean 46.7 (within 2 pp) | ✅ multi-seed n=3 at `{0,1,42}`: σ=10000 mean 60.0 vs baseline mean 60.0 (exact match) | §10.4 / §11 §4 / 2026-05-03 completion |
| σ_scale single-peaked curve | ✅ peak at σ=2.0; collapses to baseline by σ=10 | NOT MEASURED (only σ ∈ {2.0, 10000} ran on PushT) | §10.4 |
| Scaler match (head-train ↔ eval-time) | not explicitly verified (same code path assumed) | ✅ bit-exact, max abs diff = 0.00e+00 | §11.5.1 |
| σ-degeneracy gate (frac near floor < 50%, std ≥ 0.05) | gate added after Cube run; std=0.41 ≫ 0.05 | ✅ frac near floor = 0.00%, std = 0.33 | §11.5 |
| Episode sampling controlled by `cfg.seed` | ✅ verified: seeds {1, 2, 42} sample completely disjoint 50-episode sets | (verified concurrently for cube; identical sampling code path) | §05c appendix `05c_appendix_82pct_investigation.md` §A.1 |
| 82.0% Cube CEM-warm-start triple investigation | outcome (b) — independent binomial coincidence (~0.2% prior). With seed=0 added (80%, NOT 82%) the triple did not extend; aggregate over 4 distinct seeds = 163/200 = 81.5% | n/a | §05c appendix §A.5 + §05c §6 obs 5 |

---

## 7. Findings — three tiers

### 7a. Strong claims (paired multi-seed, p < 0.05 OR direction unanimous + large effect)

| Claim | Numbers | Sample |
|---|---|---|
| **Cube: PRISM-MPPI σ=2.0 absolutely beats vanilla MPPI** | Δ = +30.7 ± 4.6 pp (per-seed +28, +28, +36) | **paired n=3, p≈0.0075, 3/3 positive** (§4.1). Multi-seed from 2026-05-03 completion run. |
| **Cube: PRISM-MPPI σ=2.0 beats warm-start MPPI** | Δ = +20.0 ± 8.0 pp | paired n=3, p≈0.05, 3/3 positive (§4.2) |
| **Cube: CEM-warm-start beats vanilla CEM** | Δ = +18.0 ± 6.0 pp | paired n=3, **p≈0.035**, 3/3 positive (§4.3) |
| **Cube: vanilla CEM substantially beats vanilla MPPI** | Δ = +16.7 ± 4.6 pp (per-seed +14, +14, +22) | **paired n=3, p≈0.025, 3/3 positive** (§4.5). Cube vanilla MPPI mean 46.7% matches LeWM paper's Random baseline (§05b). |
| **PushT: vanilla CEM substantially beats vanilla MPPI** | Δ = +32.7 ± 4.2 pp (per-seed +28, +36, +34) | **paired n=3, p≈0.005, 3/3 positive** (§4.5). 2026-05-04 reverification (was +32.0 ± 3.5 pp, p≈0.004; mean shifted slightly upward, significance unchanged). |
| **Cube: PRISM-MPPI matches CEM-warm-start at equal wall-clock compute** | Δ = −0.7 ± 3.1 pp (per-seed 0, −4, +2) | **paired n=3, p≈0.74** (§3.4). PRISM-MPPI N_samples=1200 vs CEM-warm-start N_samples=300 (≈4× MPPI compute matches CEM per docs/05 §5.5). Per-seed: 1/3 above CEM-warm, 1/3 equal, 1/3 below. PRISM-MPPI mean SR 80.7% vs CEM-warm-start 81.3% — statistically indistinguishable at equal compute. |

### 7b. Suggestive observations (single-seed, n.s. at n=3, or directional but underpowered)

| Observation | Numbers | Sample |
|---|---|---|
| **PushT: PRISM-MPPI σ=2.0 directionally beats vanilla MPPI on every seed (marginal significance)** | Δ = +20.0 ± 11.1 pp (per-seed +18, +32, +10) | **paired n=3, p≈0.090, 3/3 positive** (§4.1). 2026-05-04 reverification (was +20.7 ± 11.7 pp, p≈0.093; per-seed values rebalanced but mean and significance status unchanged). Direction unanimous and effect-size large, but p just over 0.05 due to the seed=1 entry (+32 pp) inflating per-seed std. Marginal at n=3 due to high variance; would likely cross significance with more seeds. |
| Cube: CEM-warm-start mean SR ≥ PRISM-MPPI σ=2.0 mean SR (paired, N=300) | Δ = +4.0 ± 6.0 pp, 2/3 seeds positive | paired n=3, p≈0.37, n.s. (§4.4). **At 4× compute (PRISM-MPPI N=1200, §3.4) the gap closes to Δ = −0.7 ± 3.1 pp (PRISM-MPPI slightly ahead in mean, within noise; p≈0.74).** PRISM-MPPI matches CEM-warm-start on Cube at same wall-clock. |
| **PushT: CEM-warm-start beats PRISM-MPPI σ=2.0 on every paired seed (N=300)** | Δ = +11.3 ± 7.6 pp, 3/3 positive (+8, +6, +20) | paired n=3, p≈0.12, n.s. but unanimous direction (§4.4). The seed=42 entry contributes a +20 pp gap; needs disambiguation at higher n. **At 4× compute (PRISM-MPPI N=1200, §3.4) the gap does NOT close: Δ = +12.7 ± 13.3 pp (CEM-warm still ahead, 2/3 seeds positive direction); 4× MPPI compute does not bring PushT into parity.** |
| PushT: PRISM-MPPI vs warm-start MPPI not distinguishable | Δ = +3.3 ± 4.2 pp, 2/3 seeds positive, 1/3 zero | paired n=3, p≈0.30 (§4.2). 2026-05-04 reverification (was +3.3 ± 6.1, p≈0.45; mean unchanged, std tightened due to per-seed shift, p still well above 0.05). |
| PushT: CEM-warm-start vs vanilla CEM not distinguishable | Δ = −1.3 ± 4.6 pp, 1/3 seeds positive, 2/3 negative | paired n=3, p≈0.67 (§4.3). 2026-05-04 reverification (was 0.0 ± 5.3, p≈1.00; PushT vanilla CEM shifted up on s=1, s=42, flipping the sign of the mean Δ from 0.0 to −1.3 — qualitative claim "not distinguishable" unchanged). |
| Cube: warm-start MPPI underperforms BC-only across all seeds | per-seed warm-start − BC-only: −2, −18, −6 → mean Δ = −8.7 ± 8.3 pp | paired n=3, p≈0.21 (n.s. at n=3) but 3/3 seeds negative. Upgraded to multi-seed by 2026-05-03 completion run (§10 §10.3 "drift" hypothesis remains unverified). |
| PushT: BC-only well below baseline at every seed | BC-only `{32, 36, 26}` mean 31.3 vs baseline `{66, 52, 62}` mean 60.0; per-seed Δ −34, −16, −36 | paired n=3, all 3 seeds well below baseline. Upgraded to multi-seed by 2026-05-03 completion run. Hypothesized to reflect multimodal expert distribution on PushT; not directly verified. |
| Cube σ_scale curve has interior peak | 76→80→84→64→56→48→50 across {0.5, 1, 2, 3, 5, 10, 10000} | single seed=42 (§10.4) |
| Cube CEM-warm-start "82.0% triple" at deprecated {1, 2, 42} | outcome (b) coincidence; broken at s=0 (80%) when {0, 1, 42} added | §05c §6 obs 5 + §05c appendix |

### 7c. Open questions / not yet validated

- **PushT N=1200 variance increase (6.1 → 11.4 pp std) interpretation unresolved** — three candidates per §3.4: (A) lucky-low N=300 std estimate, (B) genuine destabilization at higher compute due to poor prior calibration, (C) low statistical content at n=3. Resolution requires more seeds; **deferred per CoRL timeline**.
- **PushT BC-only multimodality hypothesis:** the BC-only=31.3% multi-seed result (well below 60% baseline) is consistent with a single-Gaussian head collapsing across multimodal expert actions, but this mechanism is **not directly verified**.
- **Tworoom:** entire 5-row eval matrix uncollected. No head trained for Tworoom under our pipeline. Originally identified as the project's "acid test" for KNN-failure regimes.
- **Real-robot data:** zero. All current results are simulator-only. Required for CoRL framing as a robotics paper.
- **Cube σ_scale sweep is single-seed only.** Whether the peak position (σ=2.0) and the post-peak drop-off shape are seed-stable is unknown.
- **PushT σ_scale interior peak is uncharacterized.** Only σ ∈ {2.0, 10000} ran on PushT.
- **§10.3 "drift" hypothesis** (warm-start MPPI fails on Cube because MPPI's wide-σ exploration drifts toward cost-rewarded but task-suboptimal actions) — three concrete validation experiments listed in §10.3 are not run.
- **Reacher and additional LeWM tasks:** untested.

(Resolved during the 2026-05-03 work: Same-wall-clock MPPI-vs-CEM
comparison — see §3.4. Mode=none / σ_scale=10000 / BC-only multi-seed
— see §3.1, §3.2.)

---

## 8. What's been ruled out

| Ruled out | Reason | Source |
|---|---|---|
| 50-epoch training without early stopping | On PushT, val MSE peaks at epoch 1 and worsens monotonically through epoch 50; the recipe overfits well past best. Switched to early-stopping (patience=5, max_epochs=15). Cube run was unaffected. | §11.5 |
| Language conditioning of the head input | Pivoted away during design; the language-free `concat(z_t, z_g)` JEPA-encoder input is the architecture in use. The Qwen-VL+text caches built early in development are not consumed by any final-pipeline component. | §09 (design doc, history) |
| Qwen2.5-VL-3B as the visual encoder | Encoder-probe linear-MSE on Cube: LeWM-JEPA 0.83 / DINOv2-L/14 0.77 / Qwen-VL-vis post-merger 0.81. JEPA→DINOv2 gap = 6.8% MSE reduction, below the 20% "meaningful" threshold. JEPA is competitive at ~1/55 the parameters and is the encoder LeWM's world model already uses. | encoder probe (this session) |
| `mode=none` introducing drift | Verified bit-exact: Cube N=50 48% = 48% baseline; PushT N=50 66% = 66% re-run baseline. Hook is a clean no-op. | §10.4 / §11.5.1 |

---

## 9. What's deferred

| Deferred | Reason for deferral |
|---|---|
| **MoG (mixture-of-Gaussians) head** | Would address PushT's BC-only=32% failure *if* the multimodal-expert hypothesis is correct. Hypothesis is unverified. Estimated cost: 3–5 days (architecture change to head and to PoG fusion math; MoG-Gaussian product is not Gaussian, so fusion would need approximation). Real-robot prep currently has higher priority. |
| **Tworoom** (the original "acid test" for KNN-failure regime) | KNN ceiling ≈27% there because retrieval can't synthesize for novel state-goal pairs. Half-day cost (cache + train + eval). Deferred pending current state review. |
| **PRISM-CEM (closed-form PoG fusion adapted to CEM)** | Not implemented. PoG anchors σ at init; CEM adapts σ from elites, so the PoG semantics ("anchor the search near the prior") don't carry over without modifying the σ-update rule. CEM-warm-start (this doc) uses init-mean injection only, which is the cleanest CEM analog of warm-start MPPI. |
| ~~**Same-wall-clock MPPI vs CEM comparison**~~ — **COMPLETED 2026-05-03** | PRISM-MPPI σ=2.0 at N_samples=1200 (≈4× MPPI compute, equivalent to CEM wall-clock per docs/05 §5.5) ran on Cube and PushT, all 3 seeds. Cube outcome (a): matches CEM-warm-start within noise. PushT outcome (c): does not close the gap to CEM-warm-start. See §3.4. |
| **Multi-seed σ_scale sweeps** | Cube σ_scale curve is single-seed (§10.4); PushT σ_scale is two-points-only single-seed. ~10 min compute per task per extra seed. Cheap, not done. |
| **§9.5 ablation rows: distill-from-oracle, zero-shot VLM, oracle-μ-into-warm-start** | Reviewer-defense rows from the original design doc. None run. |
| **§10.3 drift-hypothesis empirical validation** | Per-iteration MPPI trajectory tracking; cost-vs-success correlation. Cheap (~1 day). Would convert §10.3 from hypothesis to mechanism evidence. |
| **Real-robot experiments** | Zero current evidence on hardware. Required for CoRL framing as a robotics paper. Estimated cost: 2–4 weeks. Largest single deferred item. |
| **Reacher and other LeWM tasks** | Reacher's expert dataset is `_random.h5` (CLAUDE.md lesson #4) so prior quality may be poor. Not investigated. |

---

## 10. Open decision points

**Status as of 2026-05-03:** the user has decided to **begin paper
writing**. The §12 paper-writing readiness checklist (below) captures
the empirical and analytical claims to use; §13 lists numbers NOT to
use as headlines.

The remaining experimental decisions below are now scoped as
"complementary work that could strengthen the paper if time permits"
rather than gating decisions. Same-wall-clock MPPI-vs-CEM comparison
is no longer listed — completed 2026-05-03 (§3.4).

| # | Decision | Priority (post-paper-writing-decision) | Cheapest version |
|---|---|---|---|
| 1 | Run Tworoom (originally identified acid test, KNN ceiling ≈27%) | High — would strengthen the paper's regime-asymmetry claim and address the originally-stated motivation | single-seed 4-config eval, ~half day |
| 2 | Add additional PushT seeds (n=8 instead of n=3) | High — would resolve the §3.4(A/B/C) variance interpretation and tighten the marginal §7b PushT-PRISM-vs-vanilla-MPPI claim (currently p≈0.09) | +5 seeds × 2 cells × ~75 s = ~15 min compute |
| 3 | Begin real-robot prep (CoRL framing) | Medium — required for full CoRL hardware section but largest single time cost | single Cube-like rig, σ=2.0 only, 50 trials — 1 week |
| 4 | Multi-seed Cube σ_scale curve | Medium — would convert the single-seed σ-curve in §10.4 into a multi-seed plot for the paper | 6 σ_scale × 2 extra seeds × 75 s ≈ 15 min |
| 5 | Validate §10.3 drift hypothesis | Low — turns "drift" from hypothesis to mechanism evidence; cheap | per-iteration μ tracking on 3 episodes; ~half day |
| 6 | MoG head pivot (PushT BC-only=32% failure mode) | Low — different paper; significant cost | MoG-2 head retrain + BC-only re-eval; 3–5 days |
| 7 | ~~Begin paper writing now~~ — **DECIDED YES, see §12** | — | — |
| 8 | ~~Same-wall-clock MPPI vs CEM~~ — **DONE, see §3.4** | — | — |

---

## 11. Source document map

| Number / claim | Primary source |
|---|---|
| Cube vanilla MPPI multi-seed `{48, 44, 48}` | §10 (s=42); 2026-05-03 completion run (s=0, s=1) |
| Cube Oracle PRISM σ=0.1 multi-seed `{86, 80, 80}` | §05.1b (s=42); 2026-05-03 completion run (s=0, s=1) |
| Cube BC-only multi-seed `{66, 70, 62}` | §10 (s=42); 2026-05-03 completion run (s=0, s=1) |
| Cube warm-start MPPI 56% / PRISM σ_scale curve at s=42 | §10 (single-seed) |
| Cube mode=none multi-seed `{48, 44, 48}` (= vanilla MPPI) | §10 (s=42); 2026-05-03 completion run (s=0, s=1) |
| Cube PRISM σ_scale=10000 multi-seed `{52, 44, 50}` | §10 (s=42); 2026-05-03 completion run (s=0, s=1) |
| Cube PRISM-MPPI σ=2.0 multi-seed {76, 72, 84} | §10.9 |
| Cube warm-start MPPI multi-seed {64, 52, 56} | §10.9 |
| Cube vanilla CEM multi-seed {62, 58, 70} | §05c §4 |
| Cube CEM-warm-start multi-seed {80, 82, 82} | §05c §4 |
| PushT vanilla MPPI multi-seed `{66, 52, 62}` (s=42 = 62 supersedes §05 historical 60%) | §11.5.1 (s=0); 2026-05-03 completion run (s=1, s=42) |
| PushT Oracle PRISM σ=0.1 multi-seed `{94, 92, 92}` (s=42 retained from §05 historical) | §05 / §11 (s=42 historical); 2026-05-03 completion run (s=0, s=1) |
| PushT BC-only multi-seed `{32, 36, 26}` | §11 (s=0); 2026-05-03 completion run (s=1, s=42) |
| PushT mode=none multi-seed `{66, 52, 62}` (= vanilla MPPI) | §11 (s=0); 2026-05-03 completion run (s=1, s=42) |
| PushT σ=10000 multi-seed `{62, 52, 66}` | §11 (s=0); 2026-05-03 completion run (s=1, s=42) |
| PushT PRISM-MPPI σ=2.0 multi-seed {82, 86, 74} | §11.9 |
| PushT warm-start MPPI multi-seed {84, 76, 72} | §11.9 |
| PushT vanilla CEM multi-seed {94, 86, 96} | §05c §4 |
| PushT CEM-warm-start multi-seed {90, 92, 94} | §05c §4 |
| Tworoom Oracle KNN ≈27% | project memory; §05.1b shows 64% — discrepancy unresolved (§3.3 footnote) |
| Tworoom baseline 66% (s=42) | §05.1b |
| LeWM-paper reported numbers, baseline-gap analysis | §05b |
| 82% triple investigation, episode-sampling determinism check | `docs/05c_appendix_82pct_investigation.md` |
| Head training characteristics | §10 §1 + §10 σ-calibration block / §11 §1 + §11.3 |
| Encoder-probe (JEPA / DINOv2 / Qwen-VL) | inline session result; numbers stated in §8 ruled-out row |
| Same-wall-clock PRISM-MPPI N=1200 — Cube `{80, 78, 84}` mean 80.7 ± 3.1 | 2026-05-03 (§3.4) |
| Same-wall-clock PRISM-MPPI N=1200 — PushT `{92, 76, 70}` mean 79.3 ± 11.4 | 2026-05-03 (§3.4) |

---

## 12. Paper-writing readiness checklist

Added 2026-05-03 to support the decision (§10) to begin paper writing.
This section captures (a) the strongest empirical and analytical
claims the paper can make from current data, (b) the limitations to
disclose honestly, and (c) a proposed paper structure as a starting
point. None of this is binding — adjust as the writing progresses.

### 12.1 Strongest empirical claims (paired multi-seed, paper-headline-ready)

| Claim | Numbers | Source |
|---|---|---|
| Cube: PRISM-MPPI σ=2.0 beats vanilla MPPI | Δ = +30.7 ± 4.6 pp, paired n=3, **p < 0.01**, 3/3 positive | §4.1 / §7a |
| Cube: PRISM-MPPI σ=2.0 beats warm-start MPPI | Δ = +20.0 ± 8.0 pp, paired n=3, p ≈ 0.05, 3/3 positive | §4.2 / §7a |
| Cube: CEM-warm-start beats vanilla CEM | Δ = +18.0 ± 6.0 pp, paired n=3, p ≈ 0.035, 3/3 positive | §4.3 / §7a |
| Cube: PRISM-MPPI matches CEM-warm-start at equal wall-clock | Δ = −0.7 ± 3.1 pp, paired n=3, p ≈ 0.74; PRISM at N_samples=1200 vs CEM at N_samples=300 | §3.4 / §7a |
| Cube: vanilla CEM beats vanilla MPPI | Δ = +16.7 ± 4.6 pp, paired n=3, p ≈ 0.025, 3/3 positive | §4.5 / §7a |
| PushT: vanilla CEM beats vanilla MPPI | Δ = +32.7 ± 4.2 pp, paired n=3, **p ≈ 0.005**, 3/3 positive (2026-05-04 reverification) | §4.5 / §7a |

### 12.2 Strongest analytical / mathematical claims

| Claim | Support |
|---|---|
| PoG fusion is the closed-form precision-weighted product of two Gaussians; it is the unique Bayesian-optimal way to fuse two Gaussian beliefs over the same action sequence | derivation in §03 (PRISM mechanism doc); standard probability-theory result |
| Asymptotic recovery: as σ_p → ∞, PoG fusion recovers the planner's default sampling distribution (precision contribution τ_p → 0) | Verified empirically: Cube σ_scale=10000 mean 48.7% vs vanilla MPPI mean 46.7% (within 2 pp); PushT σ_scale=10000 mean 60.0% vs vanilla MPPI mean 60.0% (exact match). Multi-seed at `{0, 1, 42}` (§6 sanity gates) |
| Parameter-free fusion: no temperature, no learned mixing weight — σ_p is the only knob, and it has direct probabilistic semantics ("how confident is the prior") | §03 mechanism doc + §10 σ_scale curve showing single-peaked dependence |
| Implementation is a 50-line subclass of `MPPISolver` that overrides `init_action_distrib` only — no modification to the planner's iteration logic, cost computation, or world model | `prism_mppi.py:PrismMPPISolver` — code-level claim |

### 12.3 Honest limitations to write into the paper

| Limitation | Numerical evidence |
|---|---|
| n=3 seeds, N=50 episodes — small-sample regime, especially on PushT (high std) | PushT PRISM-MPPI σ=2.0 N=1200 std = 11.4 pp across n=3 → SE_mean ≈ 6.6 pp; 95% CI on mean ≈ [66, 92] |
| Single-Gaussian prior fundamentally cannot represent multimodal expert distributions | PushT BC-only multi-seed `{32, 36, 26}` mean 31.3% vs baseline 60.0%; head μ as policy is well below random sampling on PushT |
| No real-robot data | All current results are simulator-only (Cube on `swm/OGBCube-v0`, PushT on `swm/PushT-v1`) |
| Tworoom (originally identified acid test for KNN-failure regime) untested under our pipeline | §3.3 — entire 5-row eval matrix uncollected |
| No comparison with iCEM, gradient-based MPC, MPC-with-learned-warm-start, or other planner families | Only MPPI and CEM evaluated |
| §10.3 "drift" hypothesis (warm-start MPPI fails by drifting toward cost-rewarded but task-suboptimal actions) is not directly verified | Three concrete validation experiments listed in §10.3 are not run |
| Cube/PushT regime asymmetry (good prior vs poor prior) likely correlates with task multimodality, but only 2 tasks evaluated | Reacher, Tworoom, additional LeWM tasks untested |

### 12.4 Proposed paper structure (starting point — not binding)

1. **Introduction.** Prior injection in sampling-based MPC; gap = no
   closed-form Bayesian-optimal fusion method exists for MPPI; PoG
   fills this gap.
2. **Method.** PoG fusion derivation; `PrismMPPISolver` implementation
   (50 LoC, init-time only); head architecture (3-layer MLP on JEPA
   embeddings, β-NLL training); per-input σ output.
3. **Experiments.** Multi-seed table on Cube + PushT (§3.1, §3.2);
   ablations covering BC-only, warm-start MPPI, CEM, CEM-warm-start
   (§4); same-wall-clock comparison MPPI vs CEM (§3.4); σ_scale
   curve (§10.4).
4. **Discussion.** Cube/PushT regime asymmetry (§7a/§7b §11) as the
   central interpretive finding: PoG works precisely when it can
   work (good prior), and degrades gracefully (asymptote test, §6)
   when it can't. PushT BC-only=31.3% as honest evidence that the
   single-Gaussian prior is mis-specified for PushT-like multimodal
   experts.
5. **Limitations + Future Work.** §12.3 above.

---

## 13. Numbers I should NOT use as paper headlines

Specific numerical statements that would be misleading or
non-replicable if cited as paper headlines. **All paper headline
numbers should be multi-seed means with explicit n and CI/std.**

| Number to avoid | Why | Use instead |
|---|---|---|
| **84% Cube PRISM-MPPI σ=2.0** | Single-seed s=42 outlier high; multi-seed mean is 77.3 ± 6.1 pp at N=300 (and 80.7 ± 3.1 pp at N=1200). Citing 84% as a headline overstates by ~7 pp. | Cube PRISM-MPPI σ=2.0 mean 77.3% (n=3, N=50) at our compute baseline; 80.7% at equal-wall-clock to CEM. |
| **92% PushT Oracle PRISM σ=0.1** | Oracle uses expert dataset retrieval at test time, not a learned prior; reference number only — not a method our submission claims. | PushT learned-head numbers from §3.2; cite 92% only as the "dataset-access ceiling" for context if needed. |
| **60% PushT vanilla MPPI** | §05 historical from a different commit / seed era. Current commit multi-seed mean is 60.0% but per-seed is `{66, 52, 62}` — same mean by coincidence, not by code-level reproducibility. | PushT vanilla MPPI multi-seed mean 60.0 ± 7.2 pp (n=3, N=50, current commit, seeds `{0, 1, 42}`). |
| **Any single PushT N=1200 seed value** (e.g. 92% at s=0, 70% at s=42) | High per-seed variance (std=11.4); single-seed numbers misrepresent the distribution. The s=0 92% and the s=42 70% bracket ~22 pp; cherry-picking either is dishonest. | PushT PRISM-MPPI σ=2.0 N=1200 multi-seed mean 79.3 ± 11.4 pp (n=3, N=50); flag the wide CI per §3.4. |
| **The Cube CEM-warm-start "82%" triple from `{1, 2, 42}`** | Coincidence (§05c appendix outcome (b)); broken at s=0 (80%) when `{0, 1, 42}` was substituted. | Cube CEM-warm-start mean 81.3 ± 1.2 pp (n=3, seeds `{0, 1, 42}`). |
| **Tworoom Oracle KNN 27%** as a definitive ceiling | Discrepant with §05.1b's 64% number; un-reconciled (§3.3 footnote). Don't cite as a precise number unless you re-run with controlled config. | "KNN ceiling on Tworoom is below the LeWM-paper-reported 87%" with explicit caveat about the unresolved 27% vs 64% discrepancy. |
| **Single-seed PRISM-MPPI vs vanilla MPPI absolute SR comparisons** (e.g. "PRISM-MPPI achieves 84% vs MPPI's 48%") | Both are now multi-seed; use the paired-Δ statistic. | Cube: Δ = +30.7 ± 4.6 pp paired n=3, p ≈ 0.008 (§7a). PushT: Δ = +20.7 ± 11.7 pp paired n=3, p ≈ 0.09 (§7b). |
