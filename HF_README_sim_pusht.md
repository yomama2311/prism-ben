---
license: mit
tags:
- robotics
- pusht
- jepa
- world-model
- prism-mppi
- model-predictive-control
- manipulation
library_name: pytorch
pipeline_tag: robotics
---

# PRISM-JEPA — PushT (sim)

JEPA world model + PRISM action prior for the classic **PushT** task. These
are the exact weights used to produce the headline **PRISM-MPPI** number in
the paper.

**Sister repo for OGBench Cube:** [`YuhaiW/prism-jepa-cube`](https://huggingface.co/YuhaiW/prism-jepa-cube)
**Code:** [`YuhaiW/prism-jepa`](https://github.com/YuhaiW/prism-jepa)

---

## Headline result (mean ± std over 3 seeds {0, 1, 42}, K = 128)

|              | Vanilla MPPI | BC-only | **PRISM-MPPI (s = 1)** |
|:------------:|:------------:|:-------:|:----------------------:|
| **PushT SR (%)** | 57.0 | 31.0 | **88.7 ± 4.2** |

`s = 1` is the only PRISM-specific hyperparameter; see paper §4.4 for the
sigma-scale sweep.

---

## Bundle

| File | Size | Role |
|------|------|------|
| `lewm_object.ckpt`      | ~72 MB | Pickled LeWM (frozen JEPA encoder + AR predictor) |
| `prior_head_pusht.pt`   | ~2 MB  | PRISM prior head (3-layer MLP, β-NLL β=0.5, σ-floor 0.05) |
| `jepa.py`, `module.py`  | ~10 KB | Model classes (needed to unpickle the LeWM ckpt) |
| `prior_head.py`         | ~3 KB  | `PriorHead` class |
| `requirements.txt`      | <1 KB  | Pinned runtime dependencies |
| `README.md`             | —      | This file |

---

## Reproduce the paper result

```bash
# 1. Clone the eval/training code
git clone https://github.com/YuhaiW/prism-jepa.git
cd prism-jepa
uv venv --python=3.10 && source .venv/bin/activate
uv pip install stable-worldmodel[train]
uv pip install opencv-python pygame mujoco pymunk scikit-image hdf5plugin
export STABLEWM_HOME=$PWD/.stable-wm

# 2. Pull the weights from this repo
pip install huggingface_hub
hf download YuhaiW/prism-jepa-pusht --local-dir ./hf_pusht
mkdir -p $STABLEWM_HOME/pusht
mv hf_pusht/lewm_object.ckpt $STABLEWM_HOME/pusht/
mv hf_pusht/prior_head_pusht.pt .

# 3. Run PRISM-MPPI (paper main result)
python eval_prism_head.py --config-name=pusht policy=pusht/lewm solver=mppi \
    +head.injection_mode=pog +head.sigma_scale=1.0 \
    +head.ckpt=prior_head_pusht.pt \
    solver.num_samples=128 eval.num_eval=50 seed=0
# repeat with seed=1, seed=42 to reproduce the mean (~89%)
```

The eval also needs the PushT expert dataset (used for normalization stats at
eval time). See the upstream LeWM collection
[`quentinll/lewm`](https://huggingface.co/collections/quentinll/lewm)
and drop `pusht_expert_train.h5` under `$STABLEWM_HOME/`.

### Vanilla MPPI baseline (no prior)

```bash
python eval_prism_head.py --config-name=pusht policy=pusht/lewm solver=mppi \
    +head.injection_mode=none solver.num_samples=128 eval.num_eval=50 seed=0
```

---

## Training recipe

The world model was trained from scratch on PushT-expert following the
upstream LeWM recipe (`python train.py data=pusht`). The prior head was then
trained with the world model frozen:

```bash
python train_prior_head.py task=pusht epochs=50 batch_size=512
```

β-NLL loss (β = 0.5), σ floored at 0.05, AdamW, cosine LR. ~30 min on a single
RTX 5090.

---

## How PRISM-MPPI works (one paragraph)

A standard MPPI planner samples action sequences from `N(0, σ_π²)` and scores
them by `‖ẑ_{t+H} − z_g‖²` in JEPA latent space. PRISM trains a lightweight
prior head `g_φ(z_t, z_g) → (μ_p, σ_p)` from offline demonstrations, then
fuses it with the planner's default sampling distribution at the initial step
via the closed-form Product-of-Gaussians:

```
σ_init² = ((s·σ_p)⁻² + σ_π⁻²)⁻¹
μ_init  = σ_init² · μ_p / (s·σ_p)²
```

The MPPI cost stays purely visual (embedding MSE to goal) — no reward, no
Q-shortcut. PRISM only re-shapes where samples are drawn from, not how they
are scored, which is why the eval-time goal mismatch that hurts pure
BC-style policies does **not** hurt PRISM-MPPI.

---

## Citation

_BibTeX TBA — paper under review._

## License

MIT. World-model code vendored from [LeWM](https://github.com/lucas-maes/le-wm)
retains its upstream MIT copyright.
