# 00. Session Handover — Read This First

**Last updated:** 2026-04-30 (after Phase 1 + Phase 2 + Tworoom diagnosis)

This document is the entry point for resuming PRISM-JEPA work in a fresh
terminal. Read sections 1-3 before doing anything; sections 4-6 are
reference for continuing development.

---

## 1. Current state in one paragraph

**PRISM-JEPA is a probe of injecting a Gaussian action prior into LeWM's
goal-conditioned MPPI planner via Product-of-Gaussians fusion.** Phase 0-2
are complete: we've validated the mechanism on 4 tasks (PushT/Cube show
+32 pp gains with oracle prior, Reacher +8 pp, Tworoom no gain), and a
K-sweep shows >10× compute savings on the two manipulation tasks. All
infrastructure is built and pushed to `git@github.com:YuhaiW/prism-jepa.git`.
Next step is **Phase 3 — replace the oracle (KNN over expert dataset) with
a real VLM-based prior** that can generalize beyond the dataset's coverage.

---

## 2. Quick start in a fresh terminal

```bash
# Activate environment
cd /home/yuhai/project/prism-jepa
source .venv/bin/activate
export STABLEWM_HOME=/home/yuhai/project/prism-jepa/.stable-wm

# Sanity check that everything still works (≤2 min)
python eval.py policy=pusht/lewm solver=mppi eval.num_eval=5
# Expected: success_rate around 70-80% (matches Phase 1 results)
```

If the sanity check passes, you're good to go. If it fails, see §6 for
common issues.

---

## 3. Headline numbers to anchor on

### Phase 1 (50 episodes, default K=300, σ=0.1)

| Task | Baseline MPPI | Oracle PRISM | Δ | Verdict |
|---|---|---|---|---|
| **PushT** | 60% | **92%** | **+32 pp** | ★ strong |
| **Cube** | 48% | **80%** | **+32 pp** | ★ strong |
| Reacher | 44% | 52% | +8 pp | dataset is `reacher_random` (not expert) |
| Tworoom | 66% | 64% | -2 pp | KNN-oracle ceiling for navigation |

### Phase 2 K-sweep (PushT)

| K | Baseline | Oracle | Comment |
|---|---|---|---|
| 32 | 54% | **94%** | oracle K=32 > baseline best K |
| 64 | 70% | 96% | |
| 128 | 60% | 92% | |
| 256 | 64% | 96% | |
| 512 | 58% | 94% | |

**Key claim**: oracle PRISM at K=32 already exceeds best baseline at any K
— ≥10× compute saving on PushT/Cube.

---

## 4. Repository tour

```
prism-jepa/
├── jepa.py, module.py, train.py, eval.py    # LeWM upstream files
├── prism_mppi.py                             # ★ PoG-fusion MPPI subclass
├── oracle_policy.py                          # ★ KNN oracle policy (multi-key support)
├── eval_oracle.py                            # ★ PRISM eval entry point
├── eval_minimal_oracle.py                    # debug helper
├── debug_oracle.py                           # 3-way diagnostic script
├── convert_hf_checkpoint.py                  # HF weights → swm _object.ckpt
├── plot_phase2.py                            # Phase 2 curve plotter
├── run_phase1.sh                             # 4-task generality validation
├── run_phase2.sh                             # K-sweep launcher
├── config/                                   # Hydra configs (LeWM + our mppi.yaml)
├── .stable-wm/                               # gitignored: ~250GB datasets + ckpts
└── docs/                                     # ★ all knowledge documented here
    ├── 00_session_handover.md   ← THIS FILE
    ├── 01_lewm_codebase.md
    ├── 02_planning_and_rl.md
    ├── 03_prism_mechanism.md
    ├── 04_experiments.md
    ├── 05_results.md
    ├── 06_roadmap.md            ← Phase 3 plan in §6.4
    └── 07_phase2_results.md     ← K-sweep + Tworoom diagnosis
```

### Read order for new contributors

1. This file (00_session_handover.md) — orientation
2. `01_lewm_codebase.md` — what LeWM is, how its MPPI works
3. `03_prism_mechanism.md` — PoG fusion math, normalization, smart oracle
4. `05_results.md` + `07_phase2_results.md` — what we know empirically
5. `06_roadmap.md` — what's next (Phase 3: VLM)

---

## 5. Reproducible commands

### Single-task probe

```bash
cd /home/yuhai/project/prism-jepa && source .venv/bin/activate
export STABLEWM_HOME=/home/yuhai/project/prism-jepa/.stable-wm

# Baseline MPPI (no PRISM)
python eval.py --config-name=pusht policy=pusht/lewm solver=mppi

# Oracle PRISM (KNN-based prior)
python eval_oracle.py --config-name=pusht policy=pusht/lewm \
    solver=mppi +oracle.sigma=0.1
```

### Full Phase 1 sweep

```bash
bash run_phase1.sh tworooms cube reacher    # ~10 min
# pusht is already cached; results in logs/phase1.log
```

### Full Phase 2 K-sweep

```bash
bash run_phase2.sh pusht cube tworooms       # ~45 min
python plot_phase2.py                        # generates logs/phase2_*.png
```

### Multi-key oracle (tworoom-style)

```bash
python eval_oracle.py --config-name=tworoom policy=tworooms/lewm \
    solver=mppi +oracle.sigma=0.1 \
    '+oracle.match_keys=[state,goal_state]' \
    '+oracle.h5_match_keys=[proprio,pos_target]'
```

---

## 6. Open questions for Phase 3 + decisions to make

### What's clearly next (low ambiguity)

1. **Train a VLM-action projection head** — see `06_roadmap.md` §6.4.
   - Input: VLM features of the current image + task language
   - Output: Gaussian (μ, σ) over the next 25 raw actions
   - BC loss against expert actions in the dataset
2. **Replace `OracleWorldModelPolicy.get_action`'s KNN call with VLM
   inference**. Everything else (PoG fusion, action-block tiling,
   normalization) stays.

### Decisions you need to make

| Decision | Options | Recommendation |
|---|---|---|
| Which VLM? | Qwen2.5-VL-3B (PRISM-Newt's choice), SigLIP, R3M, VC-1 | Qwen first (transfer existing PRISM-Newt projection head architecture) |
| Which task first? | PushT (works, abstract images may fail VLM), Cube (works, more 3D), Tworoom (KNN failed, VLM should help most) | **Cube** — best signal-to-effort ratio. Tworoom is the *interesting* case but riskier. |
| Seed budget? | Phase 1/2 used single seed; need 3 seeds for paper | After VLM-PRISM works on 1 task, then 3-seed |
| BC loss formulation? | Single Gaussian (mirror PRISM-Newt) vs MoG (handles multimodality) | Single first; MoG ablation later |

### Known limitations (don't be surprised)

- **Tworoom**: KNN-oracle hits a ceiling at ~64% (baseline 66%). Needs
  generative prior. VLM is the natural fix; document this as a positive
  motivation for VLM-PRISM, not a regression.
- **Reacher**: dataset is `reacher_random`, not expert. Don't expect
  oracle to dramatically improve. Skip for VLM work.
- **PushT abstract graphics**: 2D rendering may not align with VLM's
  natural-image training distribution. If VLM underperforms here, try
  Cube first (more 3D-realistic).

---

## 7. Common issues / troubleshooting

### `bad interpreter: No such file or directory` after rename

If you renamed the directory (e.g., from `lewm-prism`), run:

```bash
cd /home/yuhai/project/prism-jepa/.venv/bin
for f in *; do
    if [[ -f "$f" ]] && grep -q "lewm-prism" "$f" 2>/dev/null; then
        sed -i 's|/home/yuhai/project/lewm-prism|/home/yuhai/project/prism-jepa|g' "$f"
    fi
done
```

### `CUDA driver too old`

Reinstall torch with cu128 wheel:

```bash
uv pip install torch==2.8.0 torchvision==0.23.0 \
    --index-url https://download.pytorch.org/whl/cu128
```

### `TiffWriter.write() got 'fps'` (mp4 export fails)

```bash
uv pip install 'imageio[ffmpeg]' imageio-ffmpeg
```

### `KeyError: state` (oracle fails on tworoom)

Use multi-key matching:

```
+oracle.match_keys=[state,goal_state]
+oracle.h5_match_keys=[proprio,pos_target]
```

### `dm_control PhysicsError: Physics state is invalid`

Reacher dataset (`reacher_random`) has NaN action rows. Fix is in
`oracle_policy.py` (NaN filter at h5 load time). Should be auto-applied;
if not, check that branch is active.

---

## 8. Git state

- **`origin`** = `git@github.com:YuhaiW/prism-jepa.git` (private)
- **`upstream`** = `https://github.com/lucas-maes/le-wm.git` (LeWM original)
- **6 commits as of 2026-04-30**:
  - `7600f5a` Tworoom diagnosis
  - `65ca0e0` Phase 2 K-sweep
  - `a18570b` Phase 1 results
  - `237285c` Documentation
  - `449db47` 60→92 success
  - `bd8140a` Initial scaffolding

To pull LeWM upstream updates: `git fetch upstream && git merge upstream/main`.
To push changes: `git push origin main`.

---

## 9. Mental model for resuming work

**The mechanism is proven.** PoG fusion of an action prior with MPPI
sampling, when given a good prior, dramatically improves success rate
(60→92 on PushT, 48→80 on Cube) and slashes compute (10× on K).

**The bottleneck for the paper is the prior quality.** Oracle is the
ceiling (it has access to the expert dataset directly). VLM-PRISM aims to
**recover most of that ceiling without dataset access** — i.e., demonstrate
that VLMs encode enough task-relevant action priors that we don't need
expert demonstrations at deploy time.

**Tworoom's failure is a feature, not a bug.** It's the negative result
that motivates why VLM (a generative prior) is needed beyond KNN
retrieval.

---

## 10. End-of-session checklist (for whoever continues)

When finishing a session, before closing the terminal:

- [ ] Run `git status` — anything uncommitted should be staged + committed
- [ ] Push to origin: `git push origin main`
- [ ] If any new findings, append to relevant `docs/` file
- [ ] If any new bugs/quirks, add to §7 of this file
- [ ] Mention current state in this file's "Last updated" line at top
