# 03. PRISM Mechanism: PoG Fusion + MPPI Injection

This document specifies the mathematical and code-level details of how PRISM
fuses an external action prior with MPPI's sampling distribution in the LeWM
codebase.

---

## 3.1 Product of Gaussians (PoG) fusion

Given two Gaussian beliefs over each action dimension:

- MPPI prior: $\mathcal{N}(\mu_\pi, \sigma_\pi^2)$ — initial sampling distribution
- External prior: $\mathcal{N}(\mu_p, \sigma_p^2)$ — VLM, oracle, or otherwise

Their product is also Gaussian. Define **precision** as inverse variance:

$$\tau_\pi = \frac{1}{\sigma_\pi^2}, \quad \tau_p = \frac{1}{\sigma_p^2}$$

The fused (posterior) Gaussian has:

$$
\mu_{\text{fused}} = \frac{\tau_\pi \mu_\pi + \tau_p \mu_p}{\tau_\pi + \tau_p},
\qquad
\sigma_{\text{fused}}^2 = \frac{1}{\tau_\pi + \tau_p}
$$

This is a **precision-weighted convex combination** of the two means. Define
the **prior influence** as:

$$
w_p = \frac{\tau_p}{\tau_\pi + \tau_p}
$$

so that $\mu_{\text{fused}} = (1 - w_p)\mu_\pi + w_p \mu_p$.

Three regimes:
- $\sigma_p \to \infty$: $w_p \to 0$, fusion is a no-op (we recover $\mu_\pi$)
- $\sigma_p = \sigma_\pi$: $w_p = 0.5$, equal weighting
- $\sigma_p \to 0$: $w_p \to 1$, fused mean is fully the prior

The hyperparameter `oracle_sigma` (or `vlm_sigma_scale` in earlier PRISM-Newt
work) directly controls $w_p$ — that's our knob.

---

## 3.2 PoG implementation in `prism_mppi.py`

```python
def pog_fusion(mean, std, mu_prior, sigma_prior, sigma_floor=0.05):
    """
    Args:
      mean, std:       (..., A) — MPPI initial mean and std (NOT variance,
                                  despite stable-worldmodel naming)
      mu_prior, sigma_prior: (..., A) — external prior, same shape
      sigma_floor:     minimum output std after fusion
    Returns:
      fused_mean, fused_std: (..., A)
    """
    eps = 1e-8
    tau_base  = 1.0 / (std**2 + eps)
    tau_prior = 1.0 / (sigma_prior**2 + eps)
    tau_combined = tau_base + tau_prior
    fused_mean = (tau_base * mean + tau_prior * mu_prior) / tau_combined
    fused_std  = (1.0 / tau_combined).sqrt().clamp(min=sigma_floor)
    return fused_mean, fused_std
```

A `vlm_influence(std, sigma_prior)` helper returns $w_p$ for diagnostics.

---

## 3.3 Where to inject — choice and rationale

### The MPPI loop

Recall stable-worldmodel's MPPISolver:

```python
mean, var = self.init_action_distrib(init_action)   # (n_envs, H, A) zeros / ones
for step in range(n_steps):                         # 30 iterations
    candidates = mean + noise * var                  # sample
    cost = model.get_cost(info, candidates)          # JEPA rollout + MSE
    elites = topk(cost)                              # top-K
    mean = weighted_avg(elites, softmin_weights)     # update mean (var unchanged)
```

### Where PRISM fuses

We chose **once before the optimization loop**, by overriding
`init_action_distrib` to apply PoG:

```python
class PrismMPPISolver(MPPISolver):
    def init_action_distrib(self, actions=None):
        mean, std = super().init_action_distrib(actions)
        if self._prior_mean is not None:
            mean, std = pog_fusion(mean, std, self._prior_mean, self._prior_sigma,
                                   self.sigma_floor)
        return mean, std
```

### Why "once at the start" and not "every iteration"

- **Once at start**: simple, low overhead, matches Newt-PRISM behavior. The
  prior biases the initial sampling cloud; subsequent iterations refine using
  reward-based reweighting.
- **Every iteration**: more aggressive — keeps the prior pulling the sampling
  cloud toward the prior even as MPPI iterates. Could be useful if the prior
  is high-quality but MPPI tends to drift.

Empirical result on the oracle probe: once-at-start works dramatically well
(60% → 92%). No need to complicate.

### Alternative we did NOT take

We could also modify the `noise * std` step inside the inner loop to mix in
a Gaussian centered on the prior. This would let prior influence compound
across iterations. Reserved as future ablation if simple PoG-at-init isn't
enough on harder tasks.

---

## 3.4 Action-space normalization (the bug that ate 4 hours)

### Problem

LeWM's `WorldModelPolicy._prepare_info` applies a `process` dict to incoming
observations:

```python
for k, v in info_dict.items():
    if k in self.process:
        v = self.process[k].transform(v)   # apply scaler
    if k in self.transform:
        v = self.transform[k](v)           # apply image transform
```

Looking at LeWM's `eval.py`:

```python
process = {}
for col in cfg.dataset.keys_to_cache:        # 'action', 'proprio', 'state'
    if col in ['pixels']: continue
    processor = preprocessing.StandardScaler()
    processor.fit(dataset.get_col_data(col))
    process[col] = processor
    if col != 'action':
        process[f'goal_{col}'] = process[col]   # goal-state shares scaler
```

So **all non-pixel observations are z-score normalized** before going to the
JEPA model. Action is also normalized — it gets `inverse_transform` at output.

### What I had wrong

In my `eval_minimal_oracle.py` and `eval_oracle.py` first drafts:

```python
process = {"action": preprocessing.MinMaxScaler(feature_range=(-1, 1))}
```

This:
- Uses the wrong scaler (MinMax vs Standard)
- Only normalizes `action`, leaves `state` and `proprio` raw

Result: the JEPA encoder receives **raw state values (300, 600, etc.)** and
**raw action values** when the model expects standardized inputs. Cost
computation becomes meaningless — MPPI elite selection is essentially random.

### Manifestation

Even a *no-op subclass* `class OracleWMP(WorldModelPolicy): pass` gave 0% SR
because we copied this wrong `process` setup from our debug script.

### Fix

Match `eval.py` exactly:

```python
process = {}
for col in cfg.dataset.keys_to_cache:
    if col in ['pixels']: continue
    processor = preprocessing.StandardScaler()
    col_data = dataset.get_col_data(col)
    col_data = col_data[~np.isnan(col_data).any(axis=1)]
    processor.fit(col_data)
    process[col] = processor
    if col != 'action':
        process[f'goal_{col}'] = process[col]
```

After fix: stock WMP gives 80%, oracle PRISM gives 60% (5-ep) → 100% (5-ep)
with proper sigma. Bug solved.

### Lesson

When porting code that uses a learned model, **always inherit the full
preprocessing pipeline**. Even one missing scaler can silently destroy SR.

---

## 3.5 Oracle-action normalization

A second normalization issue: the oracle produces actions in **raw space**
(read from h5), but the solver works in **normalized space** (StandardScaler-
transformed).

If we pass raw oracle actions to `set_prior(mu, sigma)`, the prior is in the
wrong scale → fusion produces nonsense.

### Fix in `oracle_policy.py`

```python
oracle_actions_raw = self._expert_actions[indices]   # raw, from h5

# Normalize to match solver's space
if self.process is not None and "action" in self.process:
    flat = oracle_actions_raw.reshape(-1, raw_action_dim)
    flat = self.process["action"].transform(flat)     # StandardScaler.transform
    oracle_actions = flat.reshape(...)
```

Now `mu` passed to `solver.set_prior` is in the correct space.

---

## 3.6 The horizon × action-block tiling problem

### Problem

LeWM's MPPI uses `horizon=5` and `action_block=5`. The solver's internal
`action_dim = raw_action_dim * action_block`, so each horizon step covers
`action_block` consecutive raw actions, flattened.

For a planning call we need a `(n_envs, horizon, action_block × raw_action_dim)`
prior — i.e., `25 raw actions worth of plan` per env.

### What I tried first (WRONG)

```python
oracle_block = np.tile(oracle_actions, (1, block))      # tile single action
oracle_seq = np.tile(oracle_block[:, None, :], (1, H, 1))  # repeat across horizon
```

Result: a single 2D action repeated 25 times. The expert never repeats the same
action 25 times — this is a degenerate plan that pulls MPPI toward bad
trajectories.

Empirically: SR dropped to 40% with σ=0.1 (worse than baseline 80%).

### What worked (smart oracle)

Look up the **expert's next 25 raw actions** starting from the KNN-matched
state's index in the full dataset:

```python
plan_actions = np.zeros((n_envs, plan_steps=25, raw_action_dim))
for e in range(n_envs):
    start = orig_indices[e]
    seg = self._all_actions[start:start + plan_steps]
    plan_actions[e, :seg.shape[0]] = seg
    if seg.shape[0] < plan_steps:
        plan_actions[e, seg.shape[0]:] = seg[-1]   # pad if dataset ran out

# Reshape into (n_envs, H, block × raw_dim)
oracle_seq = plan_actions.reshape(n_envs, H, block * raw_action_dim)
```

This gives MPPI a coherent expert action **trajectory**, not a constant prior.

### Empirical impact

| Tile strategy | σ=0.1 SR (5 ep) |
|---|---|
| Single action × 25 (WRONG) | 40% |
| Full expert sequence (CORRECT) | **100%** |

The lesson: priors over **sequences** matter when the planning horizon is
multi-step. A VLM that produces a single per-step direction would suffer the
same issue — you'd want the VLM to predict an action *plan*, or at least
condition each horizon step's prior on a different state estimate.

---

## 3.7 Putting it all together

```python
# In oracle_policy.py:
class OracleWorldModelPolicy(WorldModelPolicy):
    def __init__(self, solver, config, process=None, transform=None,
                 dataset_h5_path=..., oracle_sigma=0.1, ...):
        super().__init__(solver, config, process, transform)
        # Pre-load h5: subsample for KNN, but keep full action sequence
        ...
        self._kdtree = cKDTree(subsampled_states)
        self._all_actions = full_actions     # for sequence lookup
        self._sub_idx = sub_idx              # KNN-result idx → original idx

    def get_action(self, info_dict, **kwargs):
        # 1. Find KNN-matched expert state
        current_states = self._extract_current_state(info_dict)  # raw
        _, sub_indices = self._kdtree.query(current_states, k=1)
        orig_indices = self._sub_idx[sub_indices]

        # 2. Extract expert's next plan_steps actions (smart oracle)
        plan_actions = lookup_action_sequence(self._all_actions, orig_indices)

        # 3. Normalize to solver's action space
        plan_actions = self.process['action'].transform(plan_actions.reshape(-1, A)).reshape(...)

        # 4. Reshape to (n_envs, horizon, action_dim_flattened)
        oracle_seq = plan_actions.reshape(n_envs, H, block * raw_action_dim)

        # 5. Set as prior on PrismMPPISolver
        mu = torch.from_numpy(oracle_seq).float()
        sigma = torch.ones_like(mu) * self.oracle_sigma
        self.solver.set_prior(mu, sigma)

        # 6. Delegate to parent (which calls solver.solve()
        #    → solver.init_action_distrib()
        #    → PoG fusion happens here)
        return super().get_action(info_dict, **kwargs)
```

The `PrismMPPISolver.init_action_distrib` does the actual PoG fusion:

```python
def init_action_distrib(self, actions=None):
    mean, std = super().init_action_distrib(actions)   # zeros, ones
    if self._prior_mean is not None:
        mean, std = pog_fusion(mean, std, self._prior_mean, self._prior_sigma,
                               sigma_floor=self.sigma_floor)
    return mean, std
```

Then MPPI iterates as usual on the fused (mean, std).

---

## 3.8 Diagnostic: vlm_influence

The `vlm_influence()` helper computes per-element influence:

```python
def vlm_influence(std, sigma_prior):
    eps = 1e-8
    tau_base  = 1.0 / (std**2 + eps)
    tau_prior = 1.0 / (sigma_prior**2 + eps)
    return tau_prior / (tau_base + tau_prior + eps)
```

For PushT with var_scale=1.0 (so std=1.0) and various sigma_prior values:

| sigma_prior | predicted influence |
|---|---|
| 10000 | ~0.00% |
| 1.0 | 50.0% |
| 0.5 | 80.0% |
| 0.1 | 99.0% |
| 0.05 | 99.75% |

The PoG influence saturates near 100% quickly as sigma_prior shrinks.

---

## 3.9 Open variants for future ablation

1. **Per-step state-conditioned prior** instead of constant-from-current.
   Look up the expert state at each horizon step, find its action.
2. **Mixture-of-Gaussians prior** to capture multimodal expert action
   distributions.
3. **Iteration-by-iteration PoG** to keep the prior pulling during MPPI's
   30 inner iterations.
4. **Cosine-gated influence** (similar to Newt's `prism_cos_gate_*`) — only
   apply prior when current state is close to a reliable expert state.
