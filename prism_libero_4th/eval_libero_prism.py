"""Closed-loop LIBERO evaluation for flat/chunked PRISM prior and causal MPPI.

All raw controller actions are logged. Per-block logs measure LeWM's 5-step
predictions against the actual JEPA latent; success uses env.check_success(),
never the generic ``done`` bit. Outputs JSON/CSV/JSONL, optional images/videos.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from collections import deque
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from cache_features_libero import encode_frames
from chunk_prior_head import ChunkedPriorHead, load_head
from latent_mppi import LatentPoGMPPI

MAX_STEPS = {"libero_spatial": 220, "libero_object": 300, "libero_goal": 300,
             "libero_10": 520, "libero_90": 400}


def task_slug(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower().replace("_demo", ""))


def obs_features(obs, object_key=None):
    """Physical state diagnostics, with nullable fields for task-dependent keys."""
    def vec(key):
        x = obs.get(key)
        return np.asarray(x, np.float64).reshape(-1) if x is not None else None

    eef = vec("robot0_eef_pos")
    objkey = object_key
    if objkey is None:
        candidates = [k for k in obs.keys() if k.endswith("_pos") and
                      not k.startswith("robot") and "basket" not in k.lower() and
                      "table" not in k.lower() and "cabinet" not in k.lower()]
        if len(candidates) == 1:
            objkey = candidates[0]
    obj = vec(objkey) if objkey else None
    gr = vec("robot0_gripper_qpos")
    out = {"object_key": objkey,
           "eef_pos": eef.tolist() if eef is not None else None,
           "object_pos": obj.tolist() if obj is not None else None,
           "object_z": float(obj[2]) if obj is not None and len(obj) >= 3 else None,
           "gripper_qpos": gr.tolist() if gr is not None else None,
           "eef_object_distance": (float(np.linalg.norm(eef[:3] - obj[:3]))
                                   if eef is not None and obj is not None and len(eef)>=3 and len(obj)>=3 else None)}
    return out


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)


def to_number(x):
    if isinstance(x, (float, np.floating)):
        return float(x) if np.isfinite(x) else None
    return x


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suite", required=True, choices=list(MAX_STEPS))
    ap.add_argument("--h5", required=True)
    ap.add_argument("--cache-name", required=True)
    ap.add_argument("--jepa-ckpt", required=True)
    ap.add_argument("--head", required=True, help="best.pt or legacy --out checkpoint")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--mode", choices=["pog", "warm_start", "none", "bc_only"], default="pog")
    ap.add_argument("--goal-mode", choices=["subgoal", "final"], default="subgoal")
    ap.add_argument("--sigma-scale", type=float, default=1.)
    ap.add_argument("--tasks", default="0", help="task IDs separated by commas, or all")
    ap.add_argument("--episodes", type=int, default=3, help="init states per task")
    ap.add_argument("--receding", type=int, default=1, help="blocks of five actions/replan")
    ap.add_argument("--num-samples", type=int, default=32)
    ap.add_argument("--n-steps", type=int, default=3, help="MPPI optimization iterations")
    ap.add_argument("--topk", type=int, default=0, help="0 uses all candidates (MPPI)")
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--base-std", type=float, default=1.)
    ap.add_argument("--sigma-floor", type=float, default=0.05)
    ap.add_argument("--gripper-mode", choices=["continuous", "binary"], default="continuous")
    ap.add_argument("--object-key", default=None, help="e.g. alphabet_soup_1_pos; auto if unique")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--settle-steps", type=int, default=5)
    ap.add_argument("--video", action="store_true", help="one agentview video per episode")
    ap.add_argument("--video-fps", type=int, default=20)
    ap.add_argument("--log-every-action", action="store_true", help="also echo action lines to stdout")
    ap.add_argument("--out", required=True, help="summary.json; adjacent diagnostics folder")
    args = ap.parse_args()
    if args.episodes < 1 or args.num_samples < 2 or args.receding < 1:
        ap.error("episodes, num-samples, receding must be positive")
    return args


def main():
    args = parse_args()
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    import h5py
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    home = Path(os.environ.get("STABLEWM_HOME", str(Path.home() / ".stable-wm")))
    cache_dir = home / "jepa_cache" / args.cache_name
    meta = json.loads(Path(str(args.h5) + ".meta.json").read_text())
    cache_meta = json.loads((cache_dir / "meta.json").read_text())
    if Path(cache_meta["h5"]).resolve() != Path(args.h5).resolve():
        raise ValueError("Feature cache belongs to another HDF5 file; rebuild cache")
    if args.img_size != int(cache_meta["img_size"]):
        raise ValueError("--img-size must match cached encoder input size")
    if Path(cache_meta["jepa_ckpt"]).resolve() != Path(args.jepa_ckpt).resolve():
        raise ValueError("Selected LeWM checkpoint differs from the one used to build latent cache")

    with h5py.File(args.h5) as hf:
        if hf["action"].shape[1] != 7:
            raise ValueError("LIBERO needs 7D raw controller actions")
    jepa = torch.load(args.jepa_ckpt, map_location="cpu", weights_only=False).to(device).eval()
    jepa.requires_grad_(False)
    head, ck = load_head(args.head, device)
    head.requires_grad_(False)
    if ck.get("quality_gate", {}).get("pass") is False:
        print("[WARN] Prior checkpoint was saved for diagnostics despite failing its quality gate.", flush=True)

    C, Ab, Ar = int(ck["chunk_steps"]), int(ck["a_block"]), 7
    if Ab != 5:
        raise ValueError("LeWM frameskip=5 requires 5 raw actions per block")
    if args.receding > C:
        raise ValueError("--receding cannot exceed head chunk length")
    prior_mean = torch.tensor(ck["scaler_mean"], dtype=torch.float32, device=device)
    prior_std = torch.tensor(ck["scaler_scale"], dtype=torch.float32, device=device)
    stats = json.loads((cache_dir / "action_stats.json").read_text())
    wm_mean = torch.tensor(stats["mean"], device=device, dtype=torch.float32)
    wm_std = torch.tensor(stats["std"], device=device, dtype=torch.float32)
    if (wm_std <= 0).any():
        raise ValueError("Nonpositive world model action std")
    task_goals = torch.from_numpy(np.load(cache_dir / "task_goals.npy").astype(np.float32)).to(device)

    planner = LatentPoGMPPI(
        jepa, horizon=C, action_dim=Ab * Ar, num_samples=args.num_samples,
        n_steps=args.n_steps, topk=args.topk, temperature=args.temperature,
        var_scale=args.base_std, sigma_floor=args.sigma_floor,
        device=device, seed=args.seed, gripper_mode=args.gripper_mode)
    bench = benchmark.get_benchmark_dict()[args.suite]()
    cached_names = {task_slug(x): i for i, x in enumerate(meta["task_names"])}
    tids = range(bench.n_tasks) if args.tasks == "all" else [int(x) for x in args.tasks.split(",")]
    out = Path(args.out).resolve()
    diagdir = out.parent / (out.stem + "_diagnostics")
    diagdir.mkdir(parents=True, exist_ok=True)
    max_steps = args.max_steps or MAX_STEPS[args.suite]

    def encode_obs(obs):
        cam = meta["camera"]
        key = "agentview_image" if cam == "agentview_rgb" else "robot0_eye_in_hand_image"
        frame = np.asarray(obs[key], dtype=np.uint8)
        if meta["flip"]:
            frame = frame[::-1, ::-1]
        frame = np.ascontiguousarray(frame)[None]
        return encode_frames(jepa, frame, device, args.img_size)[0]

    def video_frame(obs):
        return np.ascontiguousarray(np.asarray(obs["agentview_image"])[::-1, ::-1])

    records = []
    try:
        for tid in tids:
            task = bench.get_task(tid)
            slug = task_slug(task.name)
            if slug not in cached_names:
                print(f"[SKIP] task={tid} name={task.name!r} absent from cached demo goal bank", flush=True)
                continue
            z_goal = task_goals[cached_names[slug]]
            bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
            if not bddl.is_file():
                raise FileNotFoundError(bddl)
            env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=128, camera_widths=128)
            try:
                env.seed(args.seed)
                low0, high0 = env.env.action_spec
                low = torch.as_tensor(low0, dtype=torch.float32, device=device).reshape(-1)
                high = torch.as_tensor(high0, dtype=torch.float32, device=device).reshape(-1)
                if low.shape != (7,) or high.shape != (7,):
                    raise ValueError("Expected 7D robosuite action_spec")
                init_states = bench.get_task_init_states(tid)
                for epi in range(args.episodes):
                    init_idx = epi % len(init_states)
                    env.reset()
                    obs = env.set_init_state(init_states[init_idx])
                    for _ in range(args.settle_steps):
                        obs, _, done, _ = env.step(np.zeros(7, np.float32))
                    z0 = encode_obs(obs).detach()
                    z_history = deque([z0.clone() for _ in range(3)], maxlen=3)
                    zero_block = ((torch.zeros(5, 7, device=device) - wm_mean) / wm_std).reshape(-1)
                    history_actions = deque([zero_block.clone() for _ in range(2)], maxlen=2)
                    nstep, nreplan, success, done, episode_actions = 0, 0, False, False, []
                    block_errors, prediction_gaps, elapsed_plans = [], [], []
                    video = [video_frame(obs)] if args.video else None
                    trace = diagdir / f"task{tid:02d}_trial{epi:03d}.jsonl"
                    print(f"[EPISODE] task={tid} trial={epi} init={init_idx} goal={args.goal_mode} mode={args.mode} log={trace}", flush=True)
                    with trace.open("w", encoding="utf-8") as jf:
                        def emit(d):
                            jf.write(json.dumps(d, ensure_ascii=False, allow_nan=False) + "\n")
                            jf.flush()

                        emit({"type": "meta", "suite": args.suite, "task_id": tid,
                              "task_name": task.name, "trial": epi, "init_index": init_idx,
                              "goal_mode": args.goal_mode, "mode": args.mode,
                              "object_candidates": [k for k in obs if k.endswith("_pos") and not k.startswith("robot")],
                              "initial_state": obs_features(obs, args.object_key)})
                        while nstep < max_steps and not done and not success:
                            with torch.inference_mode():
                                start = time.perf_counter()
                                zt = z_history[-1]
                                if isinstance(head, ChunkedPriorHead):
                                    anchors = head.predict_anchors(zt[None], z_goal[None], 1)
                                    mu, sigma = head.decode_chunk(anchors[:, 0], anchors[:, 1], z_goal[None], 0)
                                    subgoal = anchors[0, 1]
                                else:
                                    mu_all, sigma_all = head(zt[None], z_goal[None])
                                    mu, sigma = mu_all[:, :C], sigma_all[:, :C]
                                    subgoal = z_goal
                                mu, sigma = mu[0].reshape(C, 35), sigma[0].reshape(C, 35)
                                target = subgoal if args.goal_mode == "subgoal" else z_goal
                                if args.mode == "bc_only":
                                    raw = (mu.reshape(C, 5, 7) * prior_std + prior_mean)
                                    raw = torch.maximum(torch.minimum(raw, high), low)
                                    if args.gripper_mode == "binary":
                                        raw = raw.clone()
                                        raw[..., 6] = torch.where(raw[..., 6] >= 0, 1., -1.)
                                    selected = {"selected_horizon_cost": None, "selection": "prior_only"}
                                else:
                                    raw, selected = planner.plan(
                                        torch.stack(list(z_history)), torch.stack(list(history_actions)),
                                        target, mu, sigma, prior_mean, prior_std, wm_mean, wm_std,
                                        low, high, mode=args.mode, sigma_scale=args.sigma_scale)
                                plan_ms = 1000 * (time.perf_counter() - start)
                                elapsed_plans.append(plan_ms)
                            nreplan += 1
                            emit({"type": "plan", "step": nstep, "replan": nreplan,
                                  "elapsed_ms": plan_ms, "target_type": args.goal_mode,
                                  "current_goal_mse": float(((zt-z_goal)**2).mean()),
                                  "current_target_mse": float(((zt-target)**2).mean()),
                                  **selected})
                            if nreplan <= 3 or nreplan % 10 == 0:
                                print(f"[PLAN] task={tid} trial={epi} step={nstep} ms={plan_ms:.1f} metrics={selected}", flush=True)

                            for block_idx in range(min(args.receding, C)):
                                if nstep >= max_steps or done or success:
                                    break
                                chosen = raw[block_idx].detach()
                                old_z = z_history[-1].detach()
                                with torch.inference_mode():
                                    predicted = planner.predict_one(
                                        torch.stack(list(z_history)),
                                        torch.stack(list(history_actions)),
                                        chosen, wm_mean, wm_std)
                                    pred_goal = float(((predicted - z_goal)**2).mean())
                                    current_goal = float(((old_z - z_goal)**2).mean())
                                    pred_target = float(((predicted - target)**2).mean())
                                    current_target = float(((old_z - target)**2).mean())
                                count = 0
                                for idx, a in enumerate(chosen.cpu().numpy()):
                                    if nstep >= max_steps:
                                        break
                                    if not np.isfinite(a).all():
                                        raise RuntimeError("nonfinite simulator action")
                                    obs, reward, done, info = env.step(a.astype(np.float32))
                                    nstep += 1
                                    count += 1
                                    episode_actions.append(a.tolist())
                                    success = bool(env.check_success())
                                    phys = obs_features(obs, args.object_key)
                                    record = {"type": "action", "task_id": tid, "trial": epi,
                                              "step": nstep, "replan": nreplan, "block": block_idx,
                                              "action_idx": idx, "action": a.tolist(),
                                              "gripper_action": float(a[-1]),
                                              "reward": to_number(reward), "done": bool(done),
                                              "success": success, **phys}
                                    emit(record)
                                    if args.log_every_action:
                                        print("[ACTION]", json.dumps(record, ensure_ascii=False), flush=True)
                                    if video is not None:
                                        video.append(video_frame(obs))
                                    if success or done:
                                        break
                                with torch.inference_mode():
                                    actual = encode_obs(obs)
                                    actual_goal = float(((actual - z_goal)**2).mean())
                                    actual_target = float(((actual - target)**2).mean())
                                # A partial block is NOT a valid 5-step prediction test.
                                if count == 5:
                                    mse = float(((predicted - actual)**2).mean())
                                    persist = float(((old_z - actual)**2).mean())
                                    prediction_gap = actual_goal - pred_goal
                                    target_gap = actual_target - pred_target
                                    prediction_gaps.append(prediction_gap)
                                    block_errors.append(mse)
                                else:
                                    mse, persist, prediction_gap, target_gap = None, None, None, None
                                physical = obs_features(obs, args.object_key)
                                block_record = {
                                    "type": "block", "task_id": tid, "trial": epi,
                                    "step": nstep, "actions_executed": count,
                                    "complete_5_step_block": count == 5, "replan": nreplan,
                                    "current_goal_mse": current_goal,
                                    "predicted_goal_mse": pred_goal,
                                    "actual_goal_mse": actual_goal,
                                    "predicted_delta": pred_goal - current_goal,
                                    "actual_delta": actual_goal - current_goal,
                                    "prediction_gap": prediction_gap,
                                    "current_target_mse": current_target,
                                    "predicted_target_mse": pred_target,
                                    "actual_target_mse": actual_target,
                                    "predicted_target_delta": pred_target - current_target,
                                    "actual_target_delta": actual_target - current_target,
                                    "target_prediction_gap": target_gap,
                                    "prediction_mse": mse, "persistence_mse": persist,
                                    "ratio": mse/max(persist, 1e-12) if mse is not None else None,
                                    "success": success, **physical,
                                }
                                emit(block_record)
                                print(f"[BLOCK] task={tid} trial={epi} step={nstep} "
                                      f"complete={count==5} eef_obj_dist={physical['eef_object_distance']} "
                                      f"goal={actual_goal:.4f} predicted={pred_goal:.4f} "
                                      f"target={actual_target:.4f} target_pred={pred_target:.4f} "
                                      f"gap={prediction_gap} pred_mse={mse} "
                                      f"persistence_mse={persist} success={success}", flush=True)
                                if count == 5:
                                    history_actions.append(((chosen - wm_mean) / wm_std).reshape(-1).detach())
                                    z_history.append(actual.detach())
                                else:
                                    # Terminal partial blocks should not be reused as 5-step history.
                                    break
                        row = {"suite": args.suite, "task_id": tid, "task_name": task.name,
                               "trial": epi, "init_index": init_idx, "mode": args.mode,
                               "goal_mode": args.goal_mode, "steps": nstep, "replans": nreplan,
                               "success": bool(success), "terminated": bool(done),
                               "mean_prediction_mse": to_number(np.mean(block_errors)) if block_errors else None,
                               "mean_abs_prediction_gap": to_number(np.mean(np.abs(prediction_gaps))) if prediction_gaps else None,
                               "mean_planning_ms": to_number(np.mean(elapsed_plans)) if elapsed_plans else None,
                               "trace": str(trace), "video": None}
                        if video is not None and len(video) > 1:
                            import imageio.v2 as imageio
                            vidpath = diagdir / f"task{tid:02d}_trial{epi:03d}.mp4"
                            imageio.mimsave(str(vidpath), video, fps=args.video_fps)
                            row["video"] = str(vidpath)
                        emit({"type": "summary", **row})
                    records.append(row)
                    print(f"[RESULT] task={tid} trial={epi} success={success} steps={nstep} trace={trace}", flush=True)
            finally:
                env.close()
    finally:
        counts = {}
        for r in records:
            key = str(r["task_id"])
            d = counts.setdefault(key, {"task_name": r["task_name"], "success": 0, "trials": 0})
            d["trials"] += 1
            d["success"] += int(r["success"])
        for d in counts.values():
            d["success_rate"] = d["success"]/d["trials"]
        summary = {"args": vars(args), "run_count": len(records), "per_task": counts,
                   "mean_task_success_rate": float(np.mean([x["success_rate"] for x in counts.values()])) if counts else None,
                   "episodes": records}
        write_json(out, summary)
        if records:
            with out.with_suffix(".csv").open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(records[0])); w.writeheader(); w.writerows(records)
        print(f"[SAVED] summary={out} diagnostics={diagdir} episodes={len(records)}", flush=True)


if __name__ == "__main__":
    main()
