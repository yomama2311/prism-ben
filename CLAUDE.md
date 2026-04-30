# CLAUDE.md — PRISM-JEPA Project Context

请使用第一性原理思考。你不能总是假设我非常清楚自己想要什么和该怎么得到。请保持审慎，从原始需求和问题出发，如果动机和目标不清晰，停下来和我讨论。如果目标清晰但是路径不是最短，告诉我，并且建议更好的办法。

## Project

**PRISM-JEPA** — A research probe injecting a Gaussian action prior into LeWM's
goal-conditioned MPPI planner via Product-of-Gaussians fusion. Sister project
of `/home/yuhai/project/prism` (Newt-based PRISM, CoRL 2026 submission).
Author: Yuhai, 1st-year CS PhD @ Northeastern University.

## Status (as of 2026-04-30)

**Phase 0-2 complete; Phase 3 (VLM integration) is the next target.**

Headline result: oracle PRISM-MPPI achieves **92% SR on PushT** (vs baseline
60%) and **80% on Cube** (vs 48%) with σ=0.1, demonstrating that the
PoG-fusion mechanism transfers cleanly to JEPA-based world models. K-sweep
shows >10× compute savings on these tasks.

## Hardware & Environment

- RTX 5090 32GB, CUDA 12.8
- `uv` venv at `.venv/` (Python 3.10), NOT a conda env
- Activation: `source .venv/bin/activate`
- Always set `STABLEWM_HOME=/home/yuhai/project/prism-jepa/.stable-wm` before
  running anything (eval scripts, conversion, etc.)

## Architecture Overview

```
RGB → JEPA encoder → embedding → ARPredictor (action-conditional)
                                  → next-embedding rollout
                                  → cost = MSE(predicted_final_emb, goal_emb)

MPPI planner samples action sequences, scores via cost, reweights elites.

PRISM injects:
  Oracle KNN → expert action sequence → PoG fusion at MPPI init
                                       → biased sampling distribution
                                       → faster convergence to good actions
```

**Crucial:** LeWM has no Q-network or reward head. Cost is purely visual
(embedding-MSE-to-goal). This is what makes PRISM's contribution unambiguous
in this setting (no Q-shortcut to absorb prior knowledge).

## Repository Layout

```
prism-jepa/
├── jepa.py, module.py           # LeWM upstream model (don't modify)
├── train.py, eval.py            # LeWM upstream entry points (don't modify)
├── prism_mppi.py                # ★ OUR contribution: PoG-fusion MPPI subclass
├── oracle_policy.py             # ★ OUR contribution: KNN oracle policy
├── eval_oracle.py               # ★ OUR contribution: PRISM eval entry
├── eval_minimal_oracle.py       # debug helper (kept for future use)
├── debug_oracle.py              # 3-way diagnostic script
├── convert_hf_checkpoint.py     # HF weights → swm _object.ckpt
├── plot_phase2.py               # Phase 2 SR-vs-K curve plotter
├── run_phase1.sh                # 4-task generality validation
├── run_phase2.sh                # K-sweep launcher
├── config/                      # Hydra configs (LeWM + our mppi.yaml)
│   └── eval/solver/mppi.yaml    # ★ NEW: MPPI Hydra config
├── .stable-wm/                  # gitignored: ~250GB datasets + ckpts
└── docs/                        # ★ READ FIRST
    ├── 00_session_handover.md   # ENTRY POINT for any new session
    ├── 01_lewm_codebase.md
    ├── 02_planning_and_rl.md
    ├── 03_prism_mechanism.md
    ├── 04_experiments.md
    ├── 05_results.md
    ├── 06_roadmap.md
    └── 07_phase2_results.md
```

## Build & Run

```bash
# Activate environment (always)
cd /home/yuhai/project/prism-jepa
source .venv/bin/activate
export STABLEWM_HOME=/home/yuhai/project/prism-jepa/.stable-wm

# Sanity check (2 min, 5 episodes)
python eval.py policy=pusht/lewm solver=mppi eval.num_eval=5

# Full baseline MPPI on a task
python eval.py --config-name=<task> policy=<task>/lewm solver=mppi

# Oracle PRISM-MPPI
python eval_oracle.py --config-name=<task> policy=<task>/lewm \
    solver=mppi +oracle.sigma=0.1

# Multi-key oracle (e.g. tworoom needs goal-aware matching)
python eval_oracle.py --config-name=tworoom policy=tworooms/lewm solver=mppi \
    +oracle.sigma=0.1 \
    '+oracle.match_keys=[state,goal_state]' \
    '+oracle.h5_match_keys=[proprio,pos_target]'

# Phase 1 (4-task generality)
bash run_phase1.sh tworooms cube reacher

# Phase 2 (K-sweep)
bash run_phase2.sh pusht cube tworooms
python plot_phase2.py
```

## Critical Lessons (do not repeat these mistakes)

1. **StandardScaler for ALL non-pixel observations, not just MinMaxScaler for
   action.** Eval.py applies `process` dict with StandardScaler on each of
   `action`, `state`, `proprio`, `goal_*`. If you create a new eval script
   and only normalize action with MinMaxScaler, the world model receives
   raw inputs (e.g. coords up to 500) and cost computation is meaningless
   → SR drops to 0 even with no PRISM. **Copy `eval.py`'s preprocessing
   block verbatim** when writing new eval entry points.

2. **Oracle actions must be normalized too.** The MPPI solver works in
   StandardScaler-normalized action space. Raw oracle actions from h5 must
   be passed through `process['action'].transform()` before set_prior.

3. **Smart oracle = expert action SEQUENCE over horizon, not single tile.**
   Tiling a single action across the planning horizon (25 env steps) yields
   a degenerate "do the same action 25× in a row" plan that hurts MPPI.
   Look up `dataset_actions[start_idx : start_idx + plan_steps]` and reshape
   to `(n_envs, horizon, action_block × raw_dim)`.

4. **NaN actions in some datasets** (e.g. `reacher_random` has NaN at
   episode boundaries from truncated rollouts). `oracle_policy.py` filters
   these at h5 load time. If you load actions yourself, do `valid_mask =
   ~np.isnan(actions).any(axis=1)`.

5. **Multi-key KNN matching for navigation tasks.** Tworoom only has 2D
   `proprio`, which doesn't disambiguate trajectories with different goals.
   Use `match_keys=['state', 'goal_state']` + `h5_match_keys=['proprio',
   'pos_target']`. **But this STILL doesn't solve tworoom** — KNN retrieval
   fundamentally limits navigation oracles. See `docs/07_phase2_results.md`
   §7.3 for the full diagnosis.

6. **Tworoom is a paper-positive negative result, not a regression.** Don't
   try to fix it by tuning sigma — the ceiling is structural (KNN retrieval
   can't synthesize for novel state-goal pairs). It motivates why VLM-PRISM
   is needed (generative prior > retrieval prior).

7. **Hardcoded venv paths break after folder rename.** If you ever rename
   `prism-jepa/`, run `for f in .venv/bin/*; do sed -i 's|<old_path>|<new_path>|g' "$f"; done`
   to fix all entry-point shebangs.

8. **`gym==0.21.0` doesn't build with modern setuptools.** Skip the `[env]`
   pip extra (which pulls in stable-baselines3 → gym 0.21). Install only
   what's needed manually: `cv2 pygame mujoco pymunk scikit-image`.

9. **Newt-PRISM (`/home/yuhai/project/prism`) is the SISTER project**, not a
   parent. Don't modify it from here. Their lessons (sigma_scale, generic
   prompt > task-specific, etc.) inform our architecture but PRISM-JEPA is a
   separate codebase.

10. **CEM solver gives higher SR than MPPI baseline** (96% vs 60% on PushT)
    because it adapts variance per iteration. Our PRISM works on top of
    MPPI specifically; we don't claim to beat CEM. The story is:
    "PRISM-MPPI matches CEM at MPPI's compute speed."

## Key Parameters

| Parameter | Value | Notes |
|---|---|---|
| `oracle.sigma` | 0.1 | tight prior; σ ≥ 1.0 = essentially no influence |
| `solver.num_samples` | 300 (default), 32-512 (sweep) | |
| `solver.n_steps` | 30 | MPPI iterations |
| `solver.var_scale` | 1.0 | initial std for sampling distribution |
| `cfg.plan_config.horizon` | 5 | planning horizon (in plan-steps) |
| `cfg.plan_config.action_block` | 5 | env steps per plan-step (frame skip) |

## Conventions

- `STABLEWM_HOME` env var must always be set; defaults to
  `~/.stable-wm/` if unset which is **not** what we want.
- Hydra config name = task name (`pusht`, `cube`, `reacher`, `tworoom`).
  Note `tworoom` (singular) is the config name even though `tworooms` is
  the dataset/checkpoint folder name.
- Dataset h5 paths: `STABLEWM_HOME/<dataset_name>.h5` where
  `dataset_name` matches the Hydra eval config:
  - pusht: `pusht_expert_train.h5`
  - cube: `ogbench/cube_single_expert.h5`
  - reacher: `dmc/reacher_random.h5`
  - tworooms: `tworoom.h5`
- `_object.ckpt` (full pickled JEPA model) goes at `STABLEWM_HOME/<task>/lewm_object.ckpt`.
- Commit messages start with action-oriented prefix (e.g. "Phase 2 K-sweep
  complete:", "Tworoom diagnosis:"). Always co-authored with Claude.

## Open Questions for Phase 3

See `docs/06_roadmap.md` §6.6. Top decisions:

1. Which VLM? Recommend **Qwen2.5-VL-3B** (transferable from Newt-PRISM
   architecture). Alternative: SigLIP, R3M, VC-1.
2. Which task first? Recommend **Cube** (best signal-to-effort), not PushT
   (abstract images may underperform with VLM).
3. Architecture: single Gaussian projection head (MoG ablation later).

## Where to find help

- For LeWM internals: `docs/01_lewm_codebase.md`
- For PoG math: `docs/03_prism_mechanism.md`
- For "why does X behave this way?" → check `docs/04_experiments.md` (bug list)
- For Phase 3 plan: `docs/06_roadmap.md`
- For Tworoom failure analysis: `docs/07_phase2_results.md` §7.3
- For Newt-PRISM cross-references: `/home/yuhai/project/prism/docs/`
