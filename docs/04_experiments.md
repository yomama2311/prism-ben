# 04. Experiment Log — 2026-04-29

Chronological log of today's PRISM-JEPA development. Useful as a record of
what worked, what didn't, and the subtle bugs we hit. Future probes should
benefit from this trail.

---

## 4.1 Pre-experiment: Newt-PRISM cp=False results that motivated the pivot

Before today, we'd been working on PRISM with the Newt (TDMPC2-based) backbone.
Key findings from `prism/docs/cp_ablation_stage1_results.md`:

- **cp=True regime**: PRISM gives massive gains (92.8% vs 37%), but baseline
  is artificially weak due to Newt's `constrained_planning` flag.
- **cp=False regime**: baseline is strong (98% on ms-pick-spoon), and PRISM
  provides little or no sample-efficiency improvement.
- **Sweet spot at scale=16**: PRISM scale=16 reaches 99.5% but cos(VLM, action)
  stays at 0.21 (low) — the agent isn't really using VLM, just gets a tiny
  early bias.

This led to the realization (in `prism/docs/related_work_lewm.md` would-be):
PRISM's value is hard to demonstrate when a Q-network can absorb VLM knowledge.
LeWM, having no Q-network, would be a cleaner testbed.

---

## 4.2 Today's roadmap

```
1. Pause all running Newt experiments
2. Clone LeWM, set up environment
3. Get baseline LeWM eval working (sanity check)
4. Implement PRISM-MPPI (PoG fusion subclass)
5. Implement oracle prior provider (KNN-based)
6. Run probe: does the mechanism move SR?
7. Push to private GitHub
8. Document findings
```

We completed all 8.

---

## 4.3 Step-by-step log

### Step 1: pause Newt jobs (15:30 ish)

- Killed TIER1 chain (sync baseline + PRISM s16 seed extension)
- Sync baseline seed=1 was 950K/1M when interrupted (97% SR, basically done)
- All data preserved on disk for potential later use

### Step 2: clone & set up LeWM

```bash
cd /home/yuhai/project
git clone https://github.com/lucas-maes/le-wm.git lewm-prism
cd lewm-prism
uv venv --python=3.10
source .venv/bin/activate
uv pip install 'stable-worldmodel[train]'
# [env] extra failed due to gym 0.21.0 build issue → install needed extras manually
uv pip install opencv-python pygame pymunk mujoco shapely scikit-image \
               diffusers dm_control 'datasets'
```

**Snags hit:**
- `gym==0.21.0` build error from `stable-baselines3` dep → side-step by skipping
  `[env]` extras and installing only what we need
- `datasets==1.1.1` too old, breaks `stable_pretraining` → upgrade to `4.8.5`

### Step 3: download pusht checkpoint + dataset

```bash
export STABLEWM_HOME=/home/yuhai/project/lewm-prism/.stable-wm
hf download quentinll/lewm-pusht --local-dir $STABLEWM_HOME/hf_pusht
# Convert HF weights.pt → swm-compatible _object.ckpt (per README)
hf download quentinll/lewm-pusht --repo-type dataset --local-dir $STABLEWM_HOME/hf_pusht_data
zstd -d hf_pusht_data/pusht_expert_train.h5.zst -o pusht_expert_train.h5
# Result: 46GB decompressed h5
```

**Snag hit:**
- Conversion script in README's docstring used `**cfg["..."]` which passes
  Hydra `_target_` keys as kwargs to PyTorch modules → had to strip `_target_`
  before unpacking.

### Step 4: first baseline eval (CEM, default)

```bash
python eval.py policy=pusht/lewm
```

**Snags hit:**
- `torch 2.11.0+cu130` installed by uv requires CUDA 13 driver; system has
  CUDA 12.8 → reinstall with `torch==2.8.0+cu128` from PyTorch wheel index
- `imageio` falls back to `tifffile` for `.mp4` writes when ffmpeg backend
  missing → install `imageio[ffmpeg]` and `imageio-ffmpeg`

After fixes: **CEM baseline → 96% SR (50 episodes)**. Infrastructure works.

### Step 5: write `prism_mppi.py`

Subclass `MPPISolver`, override `init_action_distrib` to apply PoG fusion when
a prior is set. Math sanity-checked with synthetic inputs:

```
Test 1: huge sigma_prior → fused ≈ original mean ✓
Test 2: tight sigma_prior → fused ≈ prior mean ✓
Test 3: vlm_influence behaves correctly across sigma values ✓
```

### Step 6: first oracle probe — disaster (then debug)

Wrote `oracle_policy.py` (KNN-based) and `eval_oracle.py`. First run:

```
sigma=0.1: SR = 20%
```

vs baseline MPPI 80%. PRISM made things WORSE. Sigma sweep:

```
sigma=10000 (no influence): 0%   ← SHOULD MATCH BASELINE!
sigma=0.5:  0%
sigma=0.1:  20%
```

Even with infinite sigma (effectively no fusion), SR was 0%. Something deeply
wrong.

### Step 7: bug isolation marathon

Successively narrowed:

a) **Replace OracleWMP with stock WorldModelPolicy in eval_oracle.py**: still 0%.
   → bug not in our subclass.

b) **Use `swm.policy.WorldModelPolicy` directly (no subclass at all)**: still 0%.
   → bug not in subclass mechanism.

c) **Diff `eval.py` (80%) vs `eval_minimal_oracle.py` (0%)**: found massive
   diff in `process` setup.

### Step 8: ROOT CAUSE FOUND — preprocessing mismatch

`eval.py` uses **StandardScaler for ALL non-pixel cols** (`action`, `state`,
`proprio` + `goal_state`, `goal_proprio`). My script used **only MinMaxScaler
on `action`**. Result: world model receives unnormalized state inputs (raw
values like 305, 600) → cost computation produces noise → MPPI selects elites
randomly → agent fails.

**Fix:** copy `eval.py`'s preprocessing setup verbatim.

After fix: stock WMP gives 80%, matching `eval.py` ✓

### Step 9: oracle action normalization

With preprocessing fixed, oracle PRISM at sigma=0.1 gave **20% SR (worse than
baseline 80%)**. Hypothesis: oracle actions also need normalization.

Inspected: solver works in StandardScaler-normalized action space, but my
oracle reads raw actions from h5. Fix: apply `process['action'].transform()`
to oracle actions before `set_prior`.

After fix: sigma sweep starts looking sensible:

```
sigma=10000: 80% (≡ baseline) ✓
sigma=1.0:   60% (slightly worse)
sigma=0.5:   40%
sigma=0.1:   40%
```

Better — sigma=10000 now matches baseline. But oracle still HURTS as
influence increases. This suggested another issue.

### Step 10: smart oracle (action sequence)

Realized: I was tiling a SINGLE oracle action across the full 25-step planning
horizon (`horizon=5 × action_block=5`). Real expert behavior varies across
those 25 steps. So my "oracle" was telling MPPI: "do the same action 25 times
in a row" — that's a degenerate plan.

**Fix:** at KNN match, get the expert's **next 25 actions** from the dataset,
not just the matched-state's action. Stored full action h5 in memory (~18MB,
trivial).

After fix:

```
sigma=10000: 80% (= baseline) ✓
sigma=1.0:   80%
sigma=0.5:   80%
sigma=0.1:   100% ★ (perfect on 5 episodes)
sigma=0.05:  100% ★
```

**Bingo!** Smart oracle + tight sigma → perfect SR. The mechanism works.

### Step 11: full 50-episode comparison

```
Baseline MPPI (50 ep):              60% SR
Oracle PRISM σ=0.1 (50 ep):         92% SR ★ (+32 pp, +53% relative)
(Reference) CEM Baseline (50 ep):   96% SR
```

Headline result: **PRISM gives MPPI a 53% relative boost in success rate**
when given a perfect (oracle) action prior. This sets the **upper bound** on
what a VLM-based prior could achieve.

### Step 12: rename folder, set up private repo

```bash
mv lewm-prism prism-jepa
# Fix venv hardcoded paths
sed -i 's|/home/yuhai/project/lewm-prism|/home/yuhai/project/prism-jepa|g' .venv/bin/activate
# Set up remotes
cd prism-jepa
git remote rename origin upstream     # save lewm upstream
git remote add origin git@github.com:YuhaiW/prism-jepa.git
git push -u origin main
```

All work safely backed up on the user's GitHub.

---

## 4.4 Bugs found & fixes

| # | Symptom | Root cause | Fix |
|---|---|---|---|
| 1 | `gym==0.21.0` build error | Old gym uses deprecated setuptools API | Skip `[env]` extra; install only needed env packages individually |
| 2 | `cannot import name 'config' from 'datasets'` | `datasets==1.1.1` too old (HF datasets pre-`config` module) | `uv pip install -U datasets` (→ 4.8.5) |
| 3 | `_target_` kwarg breaks model construction | HF config json has Hydra metadata | Strip `_target_` before `**` unpacking |
| 4 | `CUDA driver too old` | Default torch from uv had cu130 wheel | Reinstall `torch==2.8.0+cu128` from pytorch index |
| 5 | `TiffWriter.write() got 'fps'` | imageio fell back to tiff backend for .mp4 | `uv pip install 'imageio[ffmpeg]' imageio-ffmpeg` |
| 6 | venv broken after folder rename | activate script has hardcoded path | `sed -i 's|old|new|g' .venv/bin/activate*` |
| 7 | Oracle PRISM gave 0% SR even with infinite sigma | `process` dict missed StandardScaler for state/proprio | Copy eval.py's preprocessing verbatim |
| 8 | Oracle PRISM hurt SR as sigma decreased | Oracle actions in raw space, solver in normalized | Apply `process['action'].transform` before `set_prior` |
| 9 | Oracle still hurt at strong sigma | Single tiled action across 25-step horizon | Look up expert's next 25 actions from full dataset |

Bug 7 was the deepest — even a no-op subclass triggered it because we'd copied
the wrong preprocessing in our test scripts. **Lesson: when debugging a
"subclass breaks behavior" puzzle, first check that the test harness itself
is identical to the working baseline.**

---

## 4.5 Wall-clock summary

| Phase | Wall time |
|---|---|
| Repo setup, env install, fixing dep issues | ~1.5 h |
| Baseline LeWM eval working | ~30 min |
| Writing prism_mppi.py + oracle_policy.py + eval_oracle.py | ~30 min |
| Debugging bug 7 (wrong preprocessing) | ~1.5 h |
| Debugging bug 8 (action normalization) | ~30 min |
| Debugging bug 9 (sequence vs tile) | ~30 min |
| Final 50-episode runs + cleanup + docs | ~1 h |

Total: ~6 hours, ending with a published-quality probe result.

---

## 4.6 Files produced today

```
prism-jepa/
├── prism_mppi.py            ← PoG-fusion MPPI subclass
├── oracle_policy.py         ← KNN-based oracle policy with action-sequence lookup
├── eval_oracle.py           ← Hydra entry point for PRISM eval
├── eval_minimal_oracle.py   ← debug script (kept for future debugging)
├── debug_oracle.py          ← 3-way diagnostic harness
├── config/eval/solver/mppi.yaml   ← Hydra config for MPPISolver
└── docs/                    ← this directory
```

Two commits pushed to `git@github.com:YuhaiW/prism-jepa.git`:

```
bd8140a  PRISM-JEPA: initial PRISM probe scaffolding for LeWM
449db47  PRISM-JEPA probe SUCCESS: 60% → 92% on pusht with smart oracle
```
