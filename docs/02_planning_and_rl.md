# 02. Planning, RL, and Where LeWM Fits

This document clarifies where LeWM sits in the broader landscape of decision-
making methods, how it differs from reward-driven RL (e.g., Newt/TDMPC2), and
why this distinction matters for the PRISM contribution.

---

## 2.1 The decision-making spectrum

Consider four ways to produce a robot action policy:

```
┌─ Behavior Cloning  ─┬─ Goal-Conditioned BC  ─┬─ Goal-Conditioned Planning  ─┬─ Reward-Driven RL  ─┐
│                     │                         │  (LeWM, our setting)         │  (TDMPC2, Newt)     │
│  s, a ~ expert      │  s, a, g ~ expert       │  s_t, g, world model         │  s_t, r, world model│
│  → π(a | s)         │  → π(a | s, g)          │  → plan via WM rollout       │  → plan + Q value   │
│                     │                         │  cost = ‖predicted_z, g_z‖   │  cost = -Q(s,a)     │
└─────────────────────┴─────────────────────────┴──────────────────────────────┴─────────────────────┘
                Feed-forward                             Optimization-based
                (no rollout at deploy time)              (rollout at every plan step)
```

The two right columns share an architecture (world model + planner), but they
differ in **what the planner optimizes for**:

- **Goal-conditioned planning** (LeWM): minimize the embedding distance between
  predicted final state and a target image's embedding.
- **Reward-driven RL** (Newt): maximize cumulative reward as estimated by Q-network.

This distinction has consequences for how an external action prior (PRISM/VLM)
helps.

---

## 2.2 LeWM's planning paradigm

### What problem is being solved?

Given a target image `g` and current observation `s_t`, find an action sequence
`a_{t:t+H}` such that, after executing these actions, the agent's state matches
`g`.

Mathematically:

$$
\arg\min_{a_{t:t+H}} \| \phi(\text{rollout}(s_t, a_{t:t+H})) - \phi(g) \|^2
$$

where `φ` is the encoder, `rollout` is the JEPA dynamics model.

### Where does this come from?

- **Imitation learning context:** the dataset provides expert trajectories that
  reach goals. At deploy time, you sample a `(s_t, g)` pair and ask the agent
  to "fill in the missing actions."
- **No reward signal:** there is no scalar reward. Success is binary (did you
  reach the goal). The world model is purely predictive.

### Why it works (when it does)

JEPA's representation, having been trained to predict future embeddings under
known actions, captures task-relevant structure. Embedding distance becomes a
reasonable proxy for task progress — if your latent goes "nearby" the goal
embedding, you've made progress.

### Why it can fail

JEPA's embedding is **task-agnostic**: it learns whatever predicts next-frame
embeddings well. Some tasks (like pushing T-shapes) have good alignment between
embedding distance and task progress. Others (e.g., compositional manipulation)
might not — the embedding might cluster on visual features that don't reflect
task structure.

This is exactly the kind of bottleneck a VLM prior could address: VLMs encode
*language-grounded* visual semantics, which can complement JEPA's
self-supervised representations.

---

## 2.3 Reward-driven RL (TDMPC2 / Newt) — for contrast

### What's optimized?

$$
\arg\min_{a_{t:t+H}} -\sum_{k=0}^{H-1} \gamma^k \hat r(s_{t+k}, a_{t+k})
                    -\gamma^H \hat Q(s_{t+H}, \pi(s_{t+H}))
$$

where `r̂`, `Q̂`, and `π` are all learned components of the agent.

### Architectural consequences

To support this, TDMPC2 has:
- Encoder
- Latent dynamics
- **Reward head**
- **Q-network ensemble** (5 heads)
- **Policy network** (max-Q + entropy)

The Q-network is the secret sauce: it accumulates value information across many
training trajectories, so even a single rollout step is informative.

### How VLM prior interacts

In Newt, the Q-network can absorb whatever directional knowledge a VLM might
provide:
- VLM hint: "push the spoon left"
- Q-network: also learns that pushing-left states have high value (after enough
  training steps)
- → VLM becomes redundant once Q is informative

We documented this empirically in the Newt-PRISM ablation
(see `/home/yuhai/project/prism/docs/cp_ablation_stage1_results.md`):
PRISM's gain shrinks toward zero as training progresses, because the agent's
Q-network learns the same task knowledge.

---

## 2.4 The crucial difference for PRISM

| Property | LeWM (goal-conditioned) | Newt (reward-driven) |
|---|---|---|
| Learned signal of "good direction" | None (only embedding distance) | Q-network |
| Can the agent bypass the prior? | No — only embedding-distance scoring | Yes — Q can subsume VLM knowledge |
| Is VLM contribution measurable? | **Yes — directly** | Diminishes with training |
| What controls SR? | Quality of action prior + JEPA's representation | Q-network maturity + planner exploration |

**Implication:** PRISM's value is much **easier to demonstrate** in LeWM. With
no Q-shortcut, any non-trivial improvement from a prior is unambiguous evidence
that the prior is informative. The story becomes:

> "We replace the absent task-progress signal with VLM's task-aware visual
> semantics, fused via PoG into the MPPI sampling distribution."

vs. the Newt story:

> "We provide a small directional bias that helps early exploration, before the
> Q-network catches up."

The LeWM story is **causally cleaner**.

---

## 2.5 The "compute efficiency" framing

Because LeWM's evaluation is at **deployment time** (you've already trained the
JEPA, now you plan), the natural metric is:

> **How much compute (sample budget K, iteration count N) does the planner
> need to reach a target SR?**

A good prior reduces required compute:
- Without prior: K=300, N=30 → 60% SR (baseline MPPI on PushT)
- With strong prior: K=300, N=30 → 92% SR (our oracle result)
- With strong prior: K=64, N=10 → maybe still 90% (compute-efficient operating
  point, paper's target claim)

Plotting SR vs K (with and without PRISM) gives a compute-quality trade-off
curve — exactly the kind of figure that makes a clean paper claim.

---

## 2.6 Why goal-conditioned planning is "hot" in 2026

Three trends make this an attractive research direction:

1. **JEPA renaissance** — LeCun's group has been publishing JEPA papers
   (V-JEPA, I-JEPA, LeWM) for 2 years. Editors at NeurIPS/RSS/ICRA know
   the framework.

2. **Foundation models for control** — companies want to deploy pretrained
   world models without per-task RL fine-tuning. Goal-conditioned with VLM
   priors fits this template.

3. **Real-time robotics** — MPPI's compute is the bottleneck for high-rate
   control. "PRISM gives X× compute speedup at same SR" is a directly
   shippable robotics claim.

---

## 2.7 The two stories the same method tells

We now have a method (PoG-fusion of action prior with MPPI sampling distribution)
that can be told two ways:

| Story | Backbone | Venue | Strength |
|---|---|---|---|
| **A: PRISM as RL acceleration** | Newt (reward-driven) | CoRL 2026 | Existing data + infrastructure |
| **B: PRISM as planning compute efficiency** | LeWM (goal-conditioned) | NeurIPS 2027 / RSS 2027 | Cleaner mechanism, no Q-shortcut |

Story A has the implementation cost already paid. Story B has the higher
upside in causal clarity. The same PRISM math (PoG fusion) underlies both,
so the methods section is reusable across two papers.
