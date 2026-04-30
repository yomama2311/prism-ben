# PRISM-JEPA Documentation

Comprehensive documentation of the PRISM-on-LeWM exploration, written 2026-04-29
during initial probe development.

**Headline result:** Oracle PRISM-MPPI achieves **92% SR** on PushT (50 episodes)
vs **60% SR** for baseline MPPI — a **+32 pp / +53% relative** improvement,
demonstrating that PoG fusion of an action prior with MPPI sampling can
dramatically boost goal-conditioned planning sample efficiency in JEPA-based
world models like LeWM.

---

## Document index

| File | Topic |
|---|---|
| `00_session_handover.md` | **READ FIRST** — entry point for fresh-terminal sessions |
| `01_lewm_codebase.md` | LeWM repository structure, JEPA architecture, key code paths |
| `02_planning_and_rl.md` | How LeWM relates to RL & planning; JEPA vs reward-driven RL |
| `03_prism_mechanism.md` | PoG fusion math, MPPI injection point, normalization |
| `04_experiments.md` | Chronological log of 2026-04-29 debugging journey |
| `05_results.md` | Headline numbers, sigma sweep, comparison with Newt-PRISM |
| `06_roadmap.md` | Phase 1-4 plans toward a publishable paper |
| `07_phase2_results.md` | Phase 2 K-sweep results + Tworoom diagnosis |

---

## Quick start

```bash
cd /home/yuhai/project/prism-jepa
source .venv/bin/activate
export STABLEWM_HOME=/home/yuhai/project/prism-jepa/.stable-wm

# Baseline LeWM eval (CEM solver, 50 episodes)
python eval.py policy=pusht/lewm

# Baseline LeWM eval with MPPI solver
python eval.py policy=pusht/lewm solver=mppi

# Oracle PRISM-MPPI (the new contribution)
python eval_oracle.py policy=pusht/lewm solver=mppi +oracle.sigma=0.1
```

Pretrained checkpoints and dataset are stored in `.stable-wm/` (gitignored, ~46GB).

---

## Repository layout

```
prism-jepa/
├── jepa.py, module.py           # LeWM model (forked from upstream)
├── train.py, eval.py            # LeWM training/eval (upstream)
├── prism_mppi.py                # NEW: PoG-fusion MPPI subclass
├── oracle_policy.py             # NEW: KNN-based oracle prior provider
├── eval_oracle.py               # NEW: PRISM probe entry point
├── eval_minimal_oracle.py       # NEW: minimal repro for debug
├── debug_oracle.py              # NEW: 3-way diagnostic harness
├── config/                      # Hydra configs (LeWM + our mppi.yaml)
└── docs/                        # ← this directory
```

`origin` → `git@github.com:YuhaiW/prism-jepa.git` (private)
`upstream` → `https://github.com/lucas-maes/le-wm.git` (LeWM original)
