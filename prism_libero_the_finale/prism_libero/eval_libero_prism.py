"""Step 5 — closed-loop evaluation of PRISM-JEPA on LIBERO.

Requires the LIBERO benchmark package (pip install -e LIBERO; robosuite + mujoco) and a GPU.
Run from the prism-jepa repo root so that jepa.py / module.py are importable.

    python prism_libero/eval_libero_prism.py --suite libero_spatial \
        --h5 $STABLEWM_HOME/libero/libero_spatial.h5 --cache-name libero_spatial \
        --jepa-ckpt $STABLEWM_HOME/<run>/lewm_epoch_100_object.ckpt \
        --head prior_head_libero_spatial_chunked.pt --mode pog --goal-mode subgoal \
        --episodes 10 --out results_pog_subgoal.json

Modes:  pog | warm_start | none (vanilla MPPI) | bc_only (execute the head's mean, no planner)
Goal :  subgoal = plan towards anchor a_1 predicted by the chunked head (default),
        final   = plan towards the task goal latent (original PRISM behaviour; ablation).

NOTE on the goal: LIBERO is language-conditioned and has no goal image. The goal latent of a
task is the medoid of the final-frame latents of its demonstrations (task_goals.npy).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))  # repo root (jepa.py, module.py, prior_head.py)

from cache_features_libero import encode_frames  # noqa: E402
from chunk_prior_head import ChunkedPriorHead, load_head  # noqa: E402
from latent_mppi import LatentPoGMPPI  # noqa: E402

MAX_STEPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300,
             "libero_10": 520, "libero_90": 400}
DUMMY_ACTION = [0, 0, 0, 0, 0, 0, -1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_spatial", choices=list(MAX_STEPS))
    ap.add_argument("--h5", required=True, help="converted dataset (for <h5>.meta.json)")
    ap.add_argument("--cache-name", required=True)
    ap.add_argument("--jepa-ckpt", required=True)
    ap.add_argument("--head", required=True)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--mode", default="pog", choices=["pog", "warm_start", "none", "bc_only"])
    ap.add_argument("--goal-mode", default="subgoal", choices=["subgoal", "final"])
    ap.add_argument("--sigma-scale", type=float, default=1.0)
    ap.add_argument("--tasks", default="all", help="comma separated task ids or 'all'")
    ap.add_argument("--episodes", type=int, default=10, help="episodes per task")
    ap.add_argument("--receding", type=int, default=2, help="plan-steps executed per replan")
    ap.add_argument("--num-samples", type=int, default=300)
    ap.add_argument("--n-steps", type=int, default=30)
    ap.add_argument("--topk", type=int, default=30)
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument("--no-binarize-gripper", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--out", default="libero_results.json")
    args = ap.parse_args()

    try:
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
    except ImportError as e:  # pragma: no cover
        raise SystemExit("LIBERO is not installed: pip install -e <LIBERO repo> (needs robosuite, mujoco)") from e

    device = "cuda" if torch.cuda.is_available() else "cpu"
    home = Path(os.environ.get("STABLEWM_HOME", Path.home() / ".stable-wm"))
    cache = home / "jepa_cache" / args.cache_name
    meta = json.load(open(str(args.h5) + ".meta.json"))
    task_goals = torch.from_numpy(np.load(cache / "task_goals.npy")).float().to(device)

    jepa = torch.load(args.jepa_ckpt, map_location=device, weights_only=False).eval()
    for p in jepa.parameters():
        p.requires_grad_(False)
    head, ck = load_head(args.head, device)
    for p in head.parameters():
        p.requires_grad_(False)
    chunked = isinstance(head, ChunkedPriorHead)
    C, Ab, Ar = ck["chunk_steps"], ck["a_block"], head.A_raw
    a_mean = torch.tensor(ck["scaler_mean"], device=device)
    a_std = torch.tensor(ck["scaler_scale"], device=device)
    planner = LatentPoGMPPI(jepa, horizon=C, action_dim=Ab * Ar, num_samples=args.num_samples,
                            n_steps=args.n_steps, topk=args.topk, temperature=args.temperature,
                            device=device, seed=args.seed)

    bench = benchmark.get_benchmark_dict()[args.suite]()
    name_to_cached = {n: i for i, n in enumerate(meta["task_names"])}
    task_ids = range(bench.n_tasks) if args.tasks == "all" else [int(x) for x in args.tasks.split(",")]
    max_steps = args.max_steps or MAX_STEPS[args.suite]

    def prep(img):
        img = np.asarray(img)
        if meta["flip"]:
            img = img[::-1, ::-1]
        return np.ascontiguousarray(img)[None]

    @torch.no_grad()
    def act_chunk(img, z_g):
        z_t = encode_frames(jepa, prep(img), device, args.img_size)             # (1, D)
        if chunked:
            anchors = head.predict_anchors(z_t, z_g[None], 1)
            mu, sg = head.decode_chunk(anchors[:, 0], anchors[:, 1], z_g[None], 0)
            sub = anchors[0, 1]
        else:
            mu, sg = head(z_t, z_g[None]); mu, sg = mu[:, :C], sg[:, :C]; sub = z_g
        mu, sg = mu[0].reshape(C, Ab * Ar), sg[0].reshape(C, Ab * Ar)
        if args.mode == "bc_only":
            return mu
        target = sub if args.goal_mode == "subgoal" else z_g
        return planner.plan(z_t[0], target, mu, sg, mode=args.mode, sigma_scale=args.sigma_scale)

    def to_env_actions(plan):                                                     # (n, Ab*Ar) -> (n*Ab, 7)
        a = plan.reshape(-1, Ar) * a_std + a_mean
        a = a.clamp(-1, 1)
        if not args.no_binarize_gripper:
            a[:, -1] = torch.sign(a[:, -1])
        return a.cpu().numpy()

    results, t0 = {}, time.time()
    for tid in task_ids:
        task = bench.get_task(tid)
        if task.name not in name_to_cached:
            print(f"[eval] task {tid} '{task.name}' not in converted dataset, skipping")
            continue
        z_g = task_goals[name_to_cached[task.name]]
        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=128, camera_widths=128)
        env.seed(args.seed)
        init_states = bench.get_task_init_states(tid)
        succ = 0
        for ep in range(args.episodes):
            env.reset()
            obs = env.set_init_state(init_states[ep % len(init_states)])
            for _ in range(10):                                                    # let objects settle
                obs, _, _, _ = env.step(DUMMY_ACTION)
            t, done = 0, False
            while t < max_steps and not done:
                plan = act_chunk(obs["agentview_image"], z_g)
                for a in to_env_actions(plan[: args.receding]):
                    obs, _, done, _ = env.step(a.tolist())
                    t += 1
                    if done or t >= max_steps:
                        break
            succ += int(done)
            print(f"[eval] task {tid} ep {ep}: {'SUCCESS' if done else 'fail'} (t={t})")
        env.close()
        results[tid] = dict(name=task.name, success=succ, episodes=args.episodes,
                            rate=succ / args.episodes)
        print(f"[eval] task {tid} '{task.name}': {succ}/{args.episodes}   elapsed {(time.time() - t0) / 60:.1f} min")

    mean_rate = float(np.mean([r["rate"] for r in results.values()])) if results else 0.0
    print(f"\n[eval] mean success rate over {len(results)} tasks: {100 * mean_rate:.1f}%")
    json.dump(dict(args=vars(args), per_task=results, mean_success_rate=mean_rate),
              open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
