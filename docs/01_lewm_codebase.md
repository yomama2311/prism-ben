# 01. LeWM Codebase & Theory

## 1.1 What is LeWM?

LeWM (LeWorldModel) is a Joint-Embedding Predictive Architecture (JEPA)-based
world model from Maes, Le Lidec, Scieur, LeCun, and Balestriero (arXiv 2603.19312,
March 2026). The core claim:

> "The first JEPA that trains stably end-to-end from raw pixels using only two
> loss terms: a next-embedding prediction loss and a regularizer enforcing
> Gaussian-distributed latent embeddings."

Reductions vs prior JEPA approaches:
- 2 loss terms (vs 6 in alternatives)
- ~15M parameters
- Trainable on a single GPU in hours
- Plans up to 48× faster than foundation-model-based world models

## 1.2 Repository ecosystem

LeWM itself is intentionally minimal — it ships only the model architecture and
training objective. The full pipeline depends on two libraries:

| Repo                               | Role                                                                                                   |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------ |
| `lucas-maes/le-wm`                 | Model definition (`jepa.py`, `module.py`), training loop, eval loop                                    |
| `galilai-group/stable-worldmodel`  | Environments (PushT, Cube, TwoRooms, Reacher), planners (CEM, MPPI, Random, GD), policies, dataset I/O |
| `galilai-group/stable-pretraining` | Backbone utilities (ViT-Hugging Face wrapping), training callbacks                                     |

Our `prism-jepa` is forked from `lucas-maes/le-wm` with stable-worldmodel
and stable-pretraining installed via pip in the local venv.

## 1.3 Files in the LeWM repo

```
jepa.py        – class JEPA: encoder + predictor + action_encoder + projector
                  Defines forward path: encode → rollout → criterion (cost)
module.py       – ARPredictor (autoregressive transformer with AdaLN-zero blocks)
                – Embedder, MLP utilities
train.py        – Hydra-driven training entry point (data loader, optimizer,
                  trainer wired through stable-pretraining)
eval.py         – Hydra-driven eval entry point. Loads cost-model checkpoint,
                  builds policy, runs `world.evaluate_from_dataset`
config/train/   – Per-env training configs (cube.yaml, pusht.yaml, reacher.yaml,
                  tworoom.yaml) + lewm.yaml model config + data/ directory
config/eval/    – Per-env eval configs + solver/ subfolder (cem.yaml, adam.yaml,
                  + our new mppi.yaml)
```

## 1.4 The JEPA model architecture

### Components (defined in `jepa.py`)

```python
class JEPA(nn.Module):
    def __init__(self, encoder, predictor, action_encoder, projector, pred_proj):
        ...
```

| Component        | Type                                                       | Role                                              |
| ---------------- | ---------------------------------------------------------- | ------------------------------------------------- |
| `encoder`        | ViT-tiny (patch=14, image=224)                             | RGB image → 192-dim embedding                     |
| `action_encoder` | `Embedder` (MLP)                                           | action sequence → action representation           |
| `predictor`      | `ARPredictor` (autoregressive Transformer with AdaLN-zero) | (embedding seq, action seq) → next-embedding seq  |
| `projector`      | MLP (BatchNorm-normed)                                     | embedding → projected space (for prediction loss) |
| `pred_proj`      | MLP (BatchNorm-normed)                                     | predicted embedding → projected space             |

### Forward pass for cost computation (`get_cost`)

```python
# Pseudo-code from jepa.py
def get_cost(self, info_dict, candidates):
    # candidates: (batch, num_samples, horizon, action_dim)
    goal_emb = self.encode(info_dict['goal'])           # encode goal image
    obs_emb  = self.encode(info_dict['pixels'])         # encode current image
    action_emb = self.action_encoder(candidates)        # encode action seqs
    pred_emb = self.predict(obs_emb, action_emb)        # rollout in latent space
    cost = F.mse_loss(pred_emb[..., -1:, :], goal_emb[..., -1:, :].detach(),
                      reduction='none').sum(-1)
    return cost  # (batch, num_samples)
```

**Key property:** The model has **no Q-network, no reward head** — only an
embedding-prediction-based cost. This makes the planner's behavior critically
dependent on whether the embedding distance correlates with task progress.

### Predictor details (`module.py:ARPredictor`)

```python
class ARPredictor(nn.Module):
    """
    Autoregressive transformer with adaptive layer normalization (AdaLN-zero)
    blocks. Each block conditions on action embeddings to produce next-step
    embedding predictions.

    Inputs:  x = (B, T, embedding_dim)  — sequence of state embeddings
             c = (B, T, action_emb_dim) — action conditioning
    Output:  (B, T, output_dim)         — next-step embedding predictions
    """
    def forward(self, x, c):
        x = x + self.pos_embedding[:, :T]
        x = self.dropout(x)
        x = self.transformer(x, c)  # AdaLN-modulated attention + FFN
        return x
```

The transformer uses `ConditionalBlock` with AdaLN-zero — action embeddings
modulate attention/FFN scale and shift.

### Loss formulation (only 2 terms)

```python
total_loss = next_embedding_prediction_loss(pred_emb, target_emb_detached)
           + gaussian_regularizer(pred_emb)
```

The Gaussian regularizer is the key innovation that prevents representation
collapse without needing pre-trained encoders, EMA targets, or auxiliary
supervision.

## 1.5 Tasks LeWM was evaluated on

| Task                     | Description                | Action dim (raw) | Goal type  |
| ------------------------ | -------------------------- | ---------------- | ---------- |
| `PushT` (`swm/PushT-v1`) | Push T-shape onto target   | 2D (Δx, Δy)      | Goal image |
| `Cube`                   | Manipulation with cube     | (TBD)            | Goal image |
| `TwoRooms`               | Navigation through 2 rooms | (TBD)            | Goal image |
| `Reacher`                | DM-control reacher         | (TBD)            | Goal image |

All tasks are visually-driven — the cost is `MSE(predicted_final_embedding,
goal_image_embedding)`. There's no reward signal during planning.

## 1.6 The stable-worldmodel solver interface

All planners implement the `Solver` protocol with a common interface:

```python
class Solver(Protocol):
    def configure(self, *, action_space, n_envs, config) -> None: ...
    def solve(self, info_dict: dict, init_action=None) -> dict: ...
```

### MPPISolver (the one we extend)

Located in `.venv/.../stable_worldmodel/solver/mppi.py`. Key state:

```python
self.num_samples  # 300 default
self.var_scale    # 1.0 default; initial std for action distribution
self.n_steps      # 30 MPPI iterations
self.topk         # 30 elites per iteration
self.temperature  # 0.5; softmin temperature for elite reweighting
```

The `solve()` method:

1. `mean, var = self.init_action_distrib(init_action)`
   - mean: (n_envs, horizon, action_dim) zeros
   - var: (n_envs, horizon, action_dim) all `var_scale` (despite the name `var`,
     it's used as **std** in sampling: `candidates = mean + noise * var`)

2. For each MPPI iteration:
   - Sample `num_samples` action sequences: `mean + noise * var`
   - First sample is forced to `mean` (zero-noise candidate)
   - Score via `model.get_cost(info_dict, candidates)`
   - Take top-K elites by lowest cost
   - Reweight with `softmax(-cost / temperature)`
   - Update mean as weighted average of elites
   - Variance is **not** updated — std stays fixed at var_scale across
     iterations (this differs from CEM)

3. Return `outputs['actions']` of shape (n_envs, horizon, action_dim)

### CEMSolver (the default LeWM uses)

Same loop structure but **also updates variance** based on elite spread.
This is why CEM achieves 96% SR while MPPI gets 60-80% on PushT — CEM's
adaptive variance helps escape narrow regions of action space.

## 1.7 Policy & World

### `WorldModelPolicy` (`stable_worldmodel/policy.py`)

```python
class WorldModelPolicy(BasePolicy):
    def __init__(self, solver, config, process=None, transform=None, **kwargs):
        ...
        self.action_buffer = deque(maxlen=receding_horizon * action_block)
        self._next_init = None  # for warm-starting MPPI
```

The receding-horizon controller:

```python
def get_action(self, info_dict, **kwargs):
    info_dict = self._prepare_info(info_dict)  # ← apply process+transform

    if len(self._action_buffer) == 0:
        outputs = self.solver(info_dict, init_action=self._next_init)
        actions = outputs['actions']                     # (E, H, A)
        plan = actions[:, :receding_horizon]              # commit first segment
        self._next_init = actions[:, receding_horizon:]   # save for warm start
        plan = plan.reshape(E, flatten_receding_horizon, -1)
        self._action_buffer.extend(plan.transpose(0, 1))

    action = self._action_buffer.popleft()
    action = action.reshape(*self.env.action_space.shape).numpy()
    if 'action' in self.process:
        action = self.process['action'].inverse_transform(action)  # ← un-normalize
    return action
```

**Crucial:** the solver works in **normalized action space** (StandardScaler-
transformed). The policy un-normalizes via `inverse_transform` before sending
to the env.

### `World` and `evaluate_from_dataset`

The world wraps a vectorized gym env and a policy. `evaluate_from_dataset`:

1. Sets each env's state to `dataset[start_step]` for each episode
2. Computes goal as `dataset[start_step + goal_offset_steps]`
3. Runs eval_budget env steps with the policy
4. Returns `{'success_rate': %, 'episode_successes': [bool,...]}`

## 1.8 Hyperparameters that matter

From `config/eval/pusht.yaml`:

```yaml
plan_config:
  horizon: 5
  receding_horizon: 5
  action_block: 5         # frame skip — each plan-step covers 5 env-steps

eval:
  num_eval: 50
  goal_offset_steps: 25   # how far ahead the goal is set
  eval_budget: 50         # max env steps per episode
  img_size: 224
  dataset_name: pusht_expert_train

solver (MPPI):
  num_samples: 300
  n_steps: 30
  topk: 30
  var_scale: 1.0
  temperature: 0.5
```

So one MPPI plan call = 300 samples × 30 iterations × 5-step horizon = 45000
forward passes through the JEPA model. With 50 episodes × ~2 plan calls each
(due to receding horizon), full eval is ~100 plan calls.

## 1.9 Pretrained checkpoint

The PushT pretrained model from HuggingFace (`quentinll/lewm-pusht`) gives:
- ViT-tiny encoder (192-dim embedding)
- 18M total parameters
- Trained on `pusht_expert_train.h5` (2.3M frames)
- Reaches **96% SR with CEM** and **60% with MPPI** at default hyperparameters
  on a 50-episode held-out eval

## 1.10 Why this codebase is a good fit for PRISM

1. **Action-conditional prediction** — `action_encoder` already projects actions,
   so an external action prior is a natural injection point.
2. **Standard MPPI** — the fusion injection point (right after `init_action_distrib`,
   before sampling) matches Newt's PRISM injection exactly.
3. **No Q-network shortcut** — unlike Newt, the agent cannot learn to bypass
   the prior; planning relies entirely on world-model rollouts. This makes
   PRISM's contribution unambiguous (no "agent learned the same thing on its
   own" alternative explanation).
4. **Compute-efficient** — full eval in ~10 minutes lets us iterate fast on
   probes and ablations.
5. **Goal-image cost** — VLM (a vision-language model) is naturally aligned with
   visual-goal tasks; the semantic match is cleaner than reward-driven RL.
