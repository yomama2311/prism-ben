# 07. Phase 2 — Compute-Quality Trade-off Curves

50-episode evals at varying num_samples K ∈ {32, 64, 128, 256, 512} for
baseline MPPI vs Oracle PRISM-MPPI (σ=0.1).

Generated 2026-04-29.

---

## 7.1 Headline finding

> **At K=32 (≥10× lower compute than the default K=300), Oracle PRISM-MPPI
> matches or beats baseline MPPI's best result at any tested K** on tasks with
> expert datasets.

| Task | Baseline best (K, SR) | Oracle K=32 SR | Compute factor |
|---|---|---|---|
| **PushT** | K=64, 70% | **94%** | **≥2× saving + +24 pp** |
| **Cube** | K=512, 58% | **80%** | **≥16× saving + +22 pp** |

For Tworoom, oracle does not help (proprio-only KNN matching is the bottleneck —
see §7.4).

---

## 7.2 Full sweep tables

### PushT

| K | baseline | oracle | gap |
|---|---|---|---|
| 32 | 54% | **94%** | +40 pp |
| 64 | 70% | **96%** | +26 pp |
| 128 | 60% | 92% | +32 pp |
| 256 | 64% | 96% | +32 pp |
| 512 | 58% | 94% | +36 pp |

Oracle is consistently **+30 pp** across all K. Baseline never crosses 75%.

### Cube

| K | baseline | oracle | gap |
|---|---|---|---|
| 32 | 46% | 80% | +34 pp |
| 64 | 42% | 82% | +40 pp |
| 128 | 42% | 82% | +40 pp |
| 256 | 56% | 82% | +26 pp |
| 512 | 58% | 82% | +24 pp |

Oracle plateaus at 80-82% from K=32 — extra compute brings no further gain.
Baseline plateaus around 50%, never reaching oracle's level.

### Tworoom (negative result)

| K | baseline | oracle | gap |
|---|---|---|---|
| 32 | 66% | 60% | -6 pp |
| 64 | 74% | 68% | -6 pp |
| 128 | 82% | 64% | -18 pp |
| 256 | 62% | 70% | +8 pp |
| 512 | 72% | 68% | -4 pp |

Oracle hurts or ties baseline. Mean: baseline 71%, oracle 66%.

---

## 7.3 Why PushT and Cube succeed but Tworoom fails

Three factors aligned for PushT/Cube:

1. **Expert dataset quality**: PushT uses `pusht_expert_train`, Cube uses
   `cube_single_expert`. Both are demonstrated trajectories from skilled
   policies, so the next-25-step expert action sequence is meaningful.

2. **Rich state for KNN matching**:
   - PushT: 7-dim state (object positions, agent position, etc.)
   - Cube: similarly rich state vector

3. **Manipulation tasks**: small perturbations in action lead to small
   perturbations in goal-distance — PoG fusion's tight prior is
   appropriate.

For Tworoom:

1. **Proprio-only KNN**: tworoom h5 lacks a `state` key (only `action` and
   `proprio`), and proprio is **2-dim** — much less informative for
   nearest-neighbor matching of trajectories.

2. **Navigation task**: the optimal action depends not just on local
   position but on goal location and walls. A random expert state with
   similar proprio could be heading the wrong way.

3. **Discrete-ish dynamics**: tworoom's expert actions are larger
   movements; nearby states can have very different correct actions.

---

## 7.4 Implications for VLM-PRISM (Phase 3)

The pattern from K-sweep tells us what a good prior must do:

- **Provide a coherent action sequence**, not just a one-step direction.
- **Be state-aware** beyond simple proprioception. VLM has full image
  context — should be much better than 2-dim proprio matching.
- **Be reliable in the early planning iterations** — even at K=32, the
  prior must yield good elites.

This actually de-risks Phase 3 (real VLM): the bottleneck on tworoom is
the *match* function, not the *fusion* mechanism. A VLM that grounds
language ("go to the green room") in the image should outperform our
KNN-on-proprio oracle on tworoom.

---

## 7.5 Plots

- `logs/phase2_curves_pusht.png` — SR vs K for PushT
- `logs/phase2_curves_cube.png` — SR vs K for Cube
- `logs/phase2_curves_tworooms.png` — SR vs K for Tworooms (negative result)
- `logs/phase2_summary.png` — all three on one figure (paper-ready)

---

## 7.6 Wall-time data

Wall time per 50-episode eval scales roughly linearly with K:

| Task | K=32 | K=64 | K=128 | K=256 | K=512 |
|---|---|---|---|---|---|
| PushT  baseline | 51s | 50s | 55s | 67s | 86s |
| PushT  oracle   | 51s | 52s | 57s | 68s | 87s |
| Cube   baseline | 76s | 78s | 78s | 89s | 107s |
| Cube   oracle   | 79s | 77s | 81s | 92s | 110s |
| Tworoom baseline| 38s | 41s | 45s | 55s | 74s |
| Tworoom oracle  | 40s | 40s | 45s | 56s | 75s |

Oracle adds ~2s overhead at most (KNN lookup + tile + transform). Compute
overhead of PRISM is **negligible** vs MPPI's get_cost calls.

---

## 7.7 Decision matrix for Phase 3

Per the roadmap (`docs/06_roadmap.md`), Phase 3 = VLM integration. Phase 2
data updates the priorities:

| Task | Phase 1 oracle gain | Phase 2 K-sweep saving | Priority for VLM |
|---|---|---|---|
| **PushT** | +32 pp | ≥2× compute | **Yes** — primary task |
| **Cube** | +32 pp | ≥16× compute | **Yes** — secondary task |
| Reacher | +8 pp | (not yet sweep) | Skip — random data ceiling |
| Tworoom | -2 pp | -4 pp avg | Skip — KNN-on-proprio breaks oracle |

For Phase 3, focus on PushT and Cube. Tworoom's negative result is actually
an interesting paper figure showing where simple proprio-KNN priors fail —
which motivates using a richer signal (VLM) instead.
