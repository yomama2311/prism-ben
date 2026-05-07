# 19. Tworoom Full PRISM Method Coverage (2026-05-04 / 2026-05-05 overnight batch)

> **Status:** writing incrementally as cells complete; if you see this header alone, batch is still running or crashed early.

Commit: `e8a1fbeb52a13988a477e03f20bc168abbfc356e`
Run start: `2026-05-04T21:40:36`

---

## Phase 1 — head training: FAILED (gate-triggered, not crash-failed)

**Status: FAILED at stage `criteria`** (no `prior_head_tworoom.pt` written)
**Reason:** the gate inside `train_prior_head.py` itself ("head must reduce
val MSE by ≥15 % from init by epoch 10") triggered. The training ran
cleanly, no NaN/divergence, no crash — the head just barely learned anything.

### Training trajectory (full early-stopped run)

```
EPOCH 0 (init):  val NLL=0.4554  val MSE=1.0017
EPOCH 1/15:      val NLL=0.4442  val MSE=0.9281  *best
EPOCH 2/15:      val NLL=0.4446  val MSE=0.9281  (no-improve 1)
EPOCH 3/15:      val NLL=0.4456  val MSE=0.9285  (no-improve 2)
EPOCH 4/15:      val NLL=0.4454  val MSE=0.9294  (no-improve 3)
EPOCH 5/15:      val NLL=0.4456  val MSE=0.9303  (no-improve 4)
EARLY STOP after 5 no-improve epochs.

Drop from init: 1.0017 → 0.9281 = 7.3 %  (script gate threshold = 15 %)
```

### Reconciliation with the user's spec criteria

The user's stated Phase 1 success criteria are a **looser superset** of the
script's internal gate:

| Criterion                                        | User's threshold | Tworoom result                 | User-criterion passes? |
| ------------------------------------------------ | ---------------- | ------------------------------ | ---------------------- |
| Best val NLL finite                              | yes              | 0.4442 finite                  | ✅ pass                 |
| Best val MSE within 10× PushT (10 × 0.73 = 7.30) | yes              | 0.9281 ≪ 7.30                  | ✅ pass                 |
| **Checkpoint saved to `prior_head_tworoom.pt`**  | yes              | not saved (script gate vetoed) | ❌ **fail**             |

So under the user's first two criteria the head would qualify, but the
**checkpoint-saved criterion fails because the script's own quality gate
triggered** — and that gate is intentionally stricter (it was added during
PushT debugging to catch "head not learning" cases). Tworoom's 7.3 % drop
is closer to "not learning" than to either Cube's 41 % or PushT's 19.5 %.

Phase 2 SKIPPED per the user's hard rule "do not run eval on a broken head."

### Comparison to Cube and PushT

| Statistic | Cube | PushT | Tworoom (this run) |
|---|---|---|---|
| Init val MSE | 1.04 | 0.91 | 1.00 |
| Best val MSE | 0.61 | 0.73 | 0.93 |
| % drop init → best | **41 %** | **19.5 %** | **7.3 %** |
| Best epoch (vs max=15) | 7 | 1 | 1 |
| Action dim | 5 | 2 | 2 |
| Train windows | ~1.58 M | ~1.65 M | 0.60 M |

### Interpretation — consistent with `docs/16`'s prediction

`docs/16` §16.3 ("Tworoom — GO with caveat") flagged exactly this risk:

> *"Heuristic policy is asymmetric (door-then-target), so the head's μ on
> a given (z_t, z_g) pair may not be unimodal — possible PushT-style
> multimodal-expert behavior. Plan for BC-only to be lower than Cube's."*

Today's data confirms the caveat is real and **stronger than predicted**:

1. **60 % of training trajectories never reached the goal** (`docs/16`
   §16.1.2: 40.4 % terminated rate). The "last frame of episode" used as
   `z_g` is the agent-gave-up state for those 60 %, not the true target.
   Training supervision is mismatched on the majority of pairs.
2. **Heuristic is two-stage** (toward door, then toward target). The
   correct action for a given (z_t, z_g) depends on which side of the
   wall the agent is on — a bimodal-action structure a single-Gaussian
   head cannot represent.
3. **Action distribution saturated ±1, std 0.87** (`docs/16` §16.1.3),
   indicating maximum-magnitude moves with two visible histogram peaks
   per dim — the marginal-only structure the head learns at the 7.3 %
   floor reflects this.

This is **not a code bug**. The cache-build, training loop, and
data-loading paths all worked. The script's quality gate worked exactly
as designed (it correctly refused to save a head that barely learned).

### Possible next moves (NOT taken; user decides)

Per the spec ("match Cube/PushT protocol exactly; do not retune"), I did
not modify anything to chase a saveable head. Options for the morning:

1. **Apply the `terminated=True` filter** (proposed in `docs/16` §16.3
   caveat). Restricts training to the 40 % successful trajectories
   where last-frame ≈ true target. ~5 lines of code in
   `train_prior_head.py`. Plausible val-MSE drop > 15 % gate; head
   BC-only might land in the 30–50 % range.
2. **Soften the script gate** for Tworoom only (e.g. 5 %). This is
   retuning per the spec; needs explicit approval.
3. **Switch to MoG head** to handle bimodal-action structure. 3–5 day cost.
4. **Drop Tworoom from PRISM coverage** parallel to Reacher; keep `docs/15`
   vanilla-baselines as 4-task reference, no prior-method cells.

### One-paragraph summary

**Tworoom's mixed-quality data (40 % heuristic-success per `docs/16`),
at the same protocol as Cube/PushT, does not yield a head clean enough
for `train_prior_head.py` to save a checkpoint** — the val-MSE drop is
7.3 % vs the script's 15 % "head learning real structure" threshold. The
training is *not* broken (no NaN, no divergence; init→best is monotone
in val MSE direction); the head just isn't learning enough above the
trivial mean predictor to qualify. This is consistent with `docs/16`'s
"GO with caveat" verdict, where the caveat (data quality / multimodal
expert structure) turned out to be a hard blocker rather than a soft
one. **Phase 2 was correctly skipped per the user's "do not run eval on
a broken head" rule.** Recommend the morning decision be either (1) apply
the `terminated=True` filter and re-run, or (4) drop Tworoom and ship
the 2-task paper, in roughly that order of effort/cost.

---

*Phase 2 SKIPPED because Phase 1 failed by the script's quality gate.*
