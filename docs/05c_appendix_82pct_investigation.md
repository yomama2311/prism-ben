# 05c.A — Investigation of Cube CEM-warm-start 82% triple coincidence

Per docs/05c §6 observation 5, Cube CEM-warm-start hit exactly 41/50
on all three seeds {1, 2, 42}. This appendix investigates whether the
coincidence is (a) a sampling-seed bug, (b) independent binomial
coincidence, or (c) a deterministic fixed point.

---

## A.1 Step-1 gate: do seeds control episode sampling?

Replicated `eval_prism_head.py`'s sampling logic for Cube at each seed
without running the planner (cheap, ~5 s):

```
seed=1 :  first 10 episode_ids = [198, 275, 348, 620, 857,
                                  1169, 1239, 1244, 1340, 1441]
seed=2 :  first 10 episode_ids = [397, 551, 919, 1093, 1500,
                                  1872, 1878, 2013, 2197, 2616]
seed=42:  first 10 episode_ids = [638, 679, 859, 892, 921,
                                  941, 1281, 1652, 1825, 2014]
```

Pairwise comparison of the full 50-element samples (using the
`(episode_id, start_step)` pair as the unique evaluation key):

| Pair | Position match | Set overlap | Triple-shared |
|---|---|---|---|
| seed 1 vs seed 2  | 0 / 50 | 0 / 50 | — |
| seed 1 vs seed 42 | 0 / 50 | 0 / 50 | — |
| seed 2 vs seed 42 | 0 / 50 | 0 / 50 | — |
| All three        | — | — | **0 / 50** |

**Outcome (a) ruled out.** Seeds are correctly controlling episode
sampling. The three 50-episode samples are completely disjoint.

---

## A.2 What this means for outcome (c)

Outcome (c) — "deterministic fixed point regardless of MPPI/CEM
sampling RNG" — would require some mechanism forcing the success rate
on any 50-episode sample to be exactly 41/50. No such mechanism exists
in the planner: the per-episode outcome depends on (initial state,
goal, head's μ given that obs, CEM stochastic sampling). Three
*disjoint* 50-episode samples have entirely different per-episode
difficulty profiles; the only way each of them lands at exactly 41 is
that the underlying population SR happens to be ≈82% AND each sample
binomially collapses to that point.

Outcome (c) is therefore structurally implausible given the disjoint
samples. It would require either:
- The per-episode success probability to be uniform across all
  Cube states (not plausible — manipulation difficulty varies per
  initial cube/target configuration), AND
- The variance of the binomial sample to be zero (mathematically
  impossible at finite N).

---

## A.3 Steps 1.2 / 1.3 are moot under disjoint samples

Spec step 1.2 asked "How many episodes succeed in all 3 seeds? How
many fail in all 3 seeds? How many have inconsistent success?" These
counts are structurally undefined when no episodes are shared across
seeds — the answers are 0, 0, and "all 50 are in different samples"
respectively, and they don't speak to the (b) / (c) question.

Spec step 1.3 asked for action-trajectory comparison on episodes that
succeed in all 3 seeds. With zero shared episodes, no such comparison
is possible.

I did not extract the per-episode success masks for the three warm
CEM cube cells (would require modifying `eval_prism_head.py` to print
`metrics["episode_successes"]` and re-running the 3 cells, ~10 min).
The conclusion below does not depend on those masks; the disjoint-
sample evidence in §A.1 is sufficient.

---

## A.4 Probability check on outcome (b)

Under the independent-binomial null with N=50 and per-episode SR p:

```
P(SR = 41/50 | p = 0.82) = C(50, 41) · 0.82⁴¹ · 0.18⁹  ≈  0.131
P(all three samples = 41/50 | p = 0.82) ≈ 0.131³        ≈  0.0022  (≈0.2%)
```

A 1-in-500 event happened. Surprising, not implausible. The three
seed samples are statistically independent (set overlap = 0), so the
joint probability is the product of marginals.

For comparison, the same calculation for plus-or-minus-1 of the mode
gives P(SR ∈ {40, 41, 42}) ≈ 0.39 per seed → 0.06 = 6% for all three
landing in {40, 41, 42}. So if you relax "exactly 41" to "within 1
of the mode", the surprise drops by a factor of ~30.

---

## A.5 Conclusion

**Outcome (b) — independent binomial coincidence — is the explanation
most consistent with the data.**

- (a) sampling-seed bug: ruled out (§A.1, disjoint samples).
- (b) coincidence under binomial: ~0.2% prior probability, not crazy
  small; consistent with the disjoint-sample evidence.
- (c) deterministic fixed point: structurally implausible (§A.2), would
  require zero-variance binomial sampling.

The 82% number itself is real (it is the underlying population SR
estimate from 3 independent N=50 samples; aggregating gives 123/150
total = 82.0% with binomial 95% CI ≈ [75.0%, 87.6%]). Only the
exactly-41-three-times pattern is a coincidence.

Per the spec: outcome is (b), proceed to STEP 2.
