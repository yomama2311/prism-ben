# 38 — cube-double attempt: what worked, what broke (2026-08-09)

Attempted during the CoRL rebuttal week as a third task answering the
reviewers' "higher-dimensional, longer-horizon, multi-stage" request.
**Outcome: abandoned — the evaluation protocol does not transfer to this task.**
No cube-double number appears in the rebuttal.

## What worked (reusable assets)

| Asset | Location | Notes |
|---|---|---|
| Dataset, 3000 eps / 606k frames / 30.8 GB | `.stable-wm/ogbench/cube_double_expert.h5` | `ob_type=states`, `terminate_at_goal=False`, full 202-step episodes |
| Collector | `collect_cube_double.py` | uses `swm.World.record_dataset` + OGBench oracle; needs `mode='data_collection'` for `privileged/target_task` |
| World model, 20 epochs | `.stable-wm/cube_double/lewm_object.ckpt` | pred_loss 0.079 → 0.0056, no overfitting |
| WM quality gate | `check_wm_quality.py` | **ratio 0.067 @H=5, 0.106 @H=25 — better than the paper's cube-single WM (0.152/0.157)** |

The reduced recipe (30% of cube-single's data, 20 epochs vs 100) produced a
**perfectly usable world model**. That worry was unfounded.

## What broke: the protocol, not the method

`eval_replay_oracle.py` on cube-double: **5.0 %**.

Replaying the *logged expert actions* from the eval start reaches the goal in
only 1 of 20 episodes, versus 99.3 % (PushT) and 100 % (cube-single). The
protocol's own ceiling is 5 %, so every method's score on it is meaningless —
vanilla MPPI's 0–2 % says nothing about planning.

Likely cause: `evaluate_from_dataset` sets the goal 25 env-steps ahead, but
cube-double's success check requires **both** blocks within tolerance. At
t+25 the expert is typically mid-transport of the first block while the second
has not moved, and block permutation/randomisation at reset compounds it.

## Second, independent failure: the prior head

`train_prior_head.py --task cube_double` → **HARD GATE FAIL**: best val MSE
dropped only 10.9 % from init (0.9931 → 0.8846; gate needs ≥15 %), and val NLL
rose from epoch 1 while train NLL fell — immediate overfitting.

This is separate from the protocol problem and would need investigating on its
own (the two-block action distribution may be genuinely multi-modal in a way a
single Gaussian cannot capture — consistent with the paper's stated limitation).

## For camera-ready

1. Fix the protocol first: either place the goal far enough ahead that the
   expert has finished both blocks, or score per-block. Until the replay
   ceiling is ~100 %, no comparison on this task is interpretable.
2. Only then revisit the head. If it still fails the gate, that is a real
   (and reportable) limitation of a unimodal Gaussian prior on multi-stage
   tasks, and motivates the mixture head.
3. **Always run the replay-ceiling check before evaluating a new task.** It
   cost 2 minutes here and would have saved ~6 hours had it been run first.
