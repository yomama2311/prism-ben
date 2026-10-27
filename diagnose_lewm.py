#!/usr/bin/env python3
"""Diagnose a trained LeWM checkpoint on original LIBERO HDF5 demonstrations.

Tests multiple rollout start points, teacher-forced vs free-running prediction,
recorded vs zero vs shuffled actions, and a no-motion (persistence) baseline.

This is OFFLINE latent prediction, not robot task success. It reuses the existing
`eval_lewm_libero_offline.py` for STRICT checkpoint load, image preprocessing,
and encoding, so the results can be compared directly to the previous eval.

Keep this file beside `eval_lewm_libero_offline.py` and `inspect_checkpoint.py`.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch
from omegaconf import OmegaConf

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
try:
    import eval_lewm_libero_offline as base
except ImportError as exc:
    raise SystemExit(
        "Please put diagnose_lewm.py NEXT TO eval_lewm_libero_offline.py "
        "and inspect_checkpoint.py. Original import error: " + str(exc)
    ) from exc

VARIANTS = ("actual", "zero", "shuffled")
KEY_METRICS = ("free_mse", "teacher_mse", "persistence_mse",
               "zero_free_mse", "zero_teacher_mse",
               "shuffled_free_mse", "shuffled_teacher_mse",
               "teacher_delta_zero", "teacher_delta_shuffled",
               "free_delta_zero", "free_delta_shuffled")


def choose_balanced_demos(root: Path, max_demos: int, per_task: int, seed: int):
    """Pick across task files rather than only random episodes from a few tasks."""
    by_task = {}
    for filename in sorted(set(root.rglob("*.hdf5")) | set(root.rglob("*.h5"))):
        groups = []
        try:
            with h5py.File(filename, "r") as f:
                if "data" not in f:
                    continue
                for episode in f["data"]:
                    g = f["data"][episode]
                    if not isinstance(g, h5py.Group):
                        continue
                    if "actions" not in g or "obs/agentview_rgb" not in g:
                        continue
                    if len(g["actions"].shape) != 2 or g["actions"].shape[-1] != 7:
                        continue
                    n = min(len(g["actions"]), len(g["obs/agentview_rgb"]))
                    if n >= 5:
                        groups.append((str(filename), "data/" + episode, n, filename.stem))
        except (OSError, KeyError) as e:
            print("[WARN] Skipping file", filename, e, flush=True)
        if groups:
            by_task[filename.stem] = groups
    if not by_task:
        raise RuntimeError("No LIBERO HDF5 demo found with data/demo_N/actions [T,7] "
                           "and data/demo_N/obs/agentview_rgb [T,H,W,3].")

    rng = random.Random(seed)
    for episodes in by_task.values():
        rng.shuffle(episodes)
    tasks = sorted(by_task)
    chosen, counters = [], {t: 0 for t in tasks}
    for round_idx in range(per_task):
        for task in tasks:
            if len(chosen) >= max_demos:
                break
            if round_idx < len(by_task[task]):
                chosen.append(by_task[task][round_idx])
                counters[task] += 1
        if len(chosen) >= max_demos:
            break
    return chosen, len(tasks), counters


def get_starts(n: int, history: int, windows: int, longest: int,
               extra_long_windows: int):
    latest_start = n - history - 1
    if latest_start < 0:
        return []
    starts = set(int(x) for x in np.rint(np.linspace(0, latest_start, windows)))
    max_long_start = n - history - longest
    if max_long_start >= 0 and extra_long_windows:
        starts.update(int(x) for x in np.rint(
            np.linspace(0, max_long_start, extra_long_windows)))
    return sorted(starts)


def metrics_for_predictions(pred_free, pred_teacher, ground_truth, baseline):
    # Each prediction has shape [3, hidden_dim]; 0: actual, 1: zero, 2: shuffled.
    score = {}
    for idx, mode in enumerate(VARIANTS):
        name = "" if mode == "actual" else mode + "_"
        score[name + "free_mse"] = (pred_free[idx] - ground_truth).square().mean().item()
        score[name + "teacher_mse"] = (pred_teacher[idx] - ground_truth).square().mean().item()
    score["persistence_mse"] = (baseline - ground_truth).square().mean().item()
    score["teacher_delta_zero"] = (pred_teacher[0] - pred_teacher[1]).square().mean().item()
    score["teacher_delta_shuffled"] = (pred_teacher[0] - pred_teacher[2]).square().mean().item()
    score["free_delta_zero"] = (pred_free[0] - pred_free[1]).square().mean().item()
    score["free_delta_shuffled"] = (pred_free[0] - pred_free[2]).square().mean().item()
    return score


@torch.inference_mode()
def diagnose_window(model, z: torch.Tensor, raw_a: torch.Tensor,
                    history: int, horizons, mean, std, perm_seed: int):
    """Runs three action branches with matched observations and targets.

    raw_a and z are LOCAL (window-relative) slices, z[t] before action a[t].
    For zero action, zero is inserted in RAW action space prior to normalization.
    """
    max_steps = min(max(horizons), len(z) - history)
    if max_steps <= 0:
        return {}
    device = z.device
    T = history + max_steps
    raw = raw_a[:T].float().to(device)
    if raw.shape != (T, 7):
        raise ValueError(f"Unexpected action shape: {tuple(raw.shape)}")
    mean = torch.as_tensor(mean, device=device, dtype=raw.dtype)
    std = torch.as_tensor(std, device=device, dtype=raw.dtype)
    rng = np.random.default_rng(perm_seed)
    perm = torch.tensor(rng.permutation(T), device=device, dtype=torch.long)
    action_batch_raw = torch.stack((raw, torch.zeros_like(raw), raw[perm]), dim=0)
    # Apply identical normalization to all action conditions, including zero actions.
    action_batch = (action_batch_raw - mean[None, None]) / std[None, None]
    action_batch = torch.nan_to_num(action_batch)
    action_embeddings = model.action_encoder(action_batch).float()
    if action_embeddings.ndim != 3 or action_embeddings.shape[:2] != (3, T):
        raise RuntimeError(f"Unexpected action embeddings shape {tuple(action_embeddings.shape)}")
    latent = z.float()
    if latent.ndim != 2:
        raise RuntimeError(f"Expected [T,latent_dim], got {tuple(latent.shape)}")
    free_context = latent[:history][None].repeat(3, 1, 1)
    first_observation = latent[history - 1]
    results = {}
    for step in range(1, max_steps + 1):
        target_t = history - 1 + step
        act_ctx = action_embeddings[:, target_t - history:target_t]
        real_context = latent[target_t - history:target_t][None].expand(3, -1, -1)
        # One batched forward for free-running and teacher-forced predictions.
        predicted = model.predict(
            torch.cat((free_context, real_context), dim=0),
            torch.cat((act_ctx, act_ctx), dim=0),
        )
        if predicted.ndim != 3 or predicted.shape[0] != 6:
            raise RuntimeError(f"Unexpected predict() output shape {tuple(predicted.shape)}")
        predicted = predicted[:, -1, :].float()
        if not torch.isfinite(predicted).all():
            raise FloatingPointError(f"Nonfinite latent prediction at step={step}")
        predicted_free, predicted_teacher = predicted[:3], predicted[3:]
        free_context = torch.cat((free_context[:, 1:, :], predicted_free[:, None, :]), dim=1)
        if step in horizons:
            results[step] = metrics_for_predictions(
                predicted_free, predicted_teacher, latent[target_t], first_observation)
    return results


def write_preview(path, images):
    """Write original / flipped RGB previews; neither is assumed training-correct."""
    try:
        from PIL import Image, ImageDraw
        samples = [images[0], images[len(images)//2], images[-1]]
        tile_width = 224
        pad, label_h = 8, 20
        canvas = Image.new("RGB", (3 * tile_width + 4 * pad,
                                   2 * (tile_width + label_h) + 3 * pad), "white")
        draw = ImageDraw.Draw(canvas)
        for col, arr in enumerate(samples):
            for row, frame in enumerate((arr, np.flipud(arr))):
                x, y = pad + col*(tile_width + pad), pad + row*(tile_width + label_h + pad)
                draw.text((x, y), f"{'raw' if row == 0 else 'flipud'} frame {col+1}", fill="black")
                thumb = Image.fromarray(frame.astype(np.uint8)).resize((tile_width, tile_width))
                canvas.paste(thumb, (x, y + label_h))
        path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(path)
    except ImportError:
        print("[WARN] Pillow not installed; preview omitted (pip install pillow)")


def group_summary(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[str(r["horizon"])].append(r)
    summaries = {}
    for horizon, group in sorted(groups.items(), key=lambda kv: int(kv[0])):
        means = {field: float(np.mean([r[field] for r in group]))
                 for field in KEY_METRICS}
        meds = {field: float(np.median([r[field] for r in group]))
                for field in ("free_mse", "teacher_mse", "persistence_mse")}
        means.update({
            "windows": len(group),
            "episodes": len({(r["file"], r["episode"]) for r in group}),
            "tasks": len({r["task"] for r in group}),
            "median_free_mse": meds["free_mse"],
            "median_teacher_mse": meds["teacher_mse"],
            "median_persistence_mse": meds["persistence_mse"],
            "free_over_persistence": means["free_mse"] / max(means["persistence_mse"], 1e-12),
            "teacher_over_persistence": means["teacher_mse"] / max(means["persistence_mse"], 1e-12),
            "fraction_actual_free_beats_persistence": float(np.mean([
                r["free_mse"] < r["persistence_mse"] for r in group])),
            "fraction_actual_teacher_beats_zero": float(np.mean([
                r["teacher_mse"] < r["zero_teacher_mse"] for r in group])),
            "fraction_actual_teacher_beats_shuffled": float(np.mean([
                r["teacher_mse"] < r["shuffled_teacher_mse"] for r in group])),
        })
        summaries[horizon] = means
    return summaries


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--libero-root", type=Path, required=True)
    p.add_argument("--code-root", type=Path, default=Path("/workspace/prism-ben"))
    p.add_argument("--output", type=Path, default=Path("results/diagnose_lewm.json"))
    p.add_argument("--builder", default="")
    p.add_argument("--max-demos", type=int, default=20)
    p.add_argument("--per-task", type=int, default=2)
    p.add_argument("--windows-per-demo", type=int, default=6)
    p.add_argument("--long-window-count", type=int, default=3)
    p.add_argument("--horizons", type=int, nargs="+", default=[1, 5, 10, 25, 50, 100])
    p.add_argument("--encode-batch", type=int, default=16)
    p.add_argument("--action-norm", choices=("raw", "estimate"), default="raw")
    p.add_argument("--action-stats", type=Path, help="NPZ containing training mean[7] and std[7]; overrides --action-norm")
    p.add_argument("--seed", type=int, default=3072)
    p.add_argument("--preview-dir", type=Path, help="Save contact sheets of raw vs vertically flipped RGB frames")
    p.add_argument("--preview-count", type=int, default=2)
    p.add_argument("--unsafe-pickle", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    if any(v <= 0 for v in (args.max_demos, args.per_task, args.windows_per_demo,
                            args.encode_batch)) or not args.horizons or min(args.horizons) < 1:
        p.error("Positive --max-demos, --per-task, --windows-per-demo, --encode-batch, --horizons required")
    if args.long_window_count < 0 or args.preview_count < 0:
        p.error("--long-window-count and --preview-count must be non-negative")
    if not args.checkpoint.is_file() or not args.config.is_file():
        p.error("Checkpoint or config file not found")
    if not args.libero_root.is_dir():
        p.error("LIBERO root is not a directory")
    args.horizons = sorted(set(args.horizons))
    return args


def main():
    args = parse_args()
    sys.path.insert(0, str(args.code_root.resolve()))
    cfg = OmegaConf.load(args.config)
    history = int(cfg.wm.history_size)
    frameskip = int(cfg.data.dataset.frameskip)
    action_dim = int(cfg.wm.action_dim)
    if history < 1 or frameskip != 1 or action_dim != 7:
        raise ValueError(f"Script expects LIBERO action_dim=7 frameskip=1; "
                         f"history={history} action_dim={action_dim} frameskip={frameskip}")
    device = torch.device(args.device)
    demos, task_count, task_counts = choose_balanced_demos(
        args.libero_root, args.max_demos, args.per_task, args.seed)
    # base.get_stats expects (file, group, frames), exactly as previous evaluator.
    stats_demos = [(file, group, n) for file, group, n, _ in demos]
    mean, std, norm_source = base.get_stats(args, stats_demos)
    model, checkpoint = base.build_and_load(args, cfg, device)
    print("Strict checkpoint load: PASS", flush=True)
    print(f"Found {len(demos)} demos across {task_count} tasks. Task coverage: {task_counts}", flush=True)
    print("Evaluation: actual / RAW-zero / within-window shuffled actions."
          " Includes teacher-forced, free-running, persistence.", flush=True)
    rows = []
    failures = []
    previews_saved = 0
    for idx, (file, group, n, task) in enumerate(demos, start=1):
        actions_np, images = base.read_demo(file, group, n)
        if args.preview_dir and previews_saved < args.preview_count:
            write_preview(args.preview_dir / f"{idx:02d}_{task[:70]}.png", images)
            previews_saved += 1
        latents = base.encode_sequence(model, images, int(cfg.img_size), device, args.encode_batch)
        raw_actions = torch.from_numpy(np.nan_to_num(actions_np, nan=0.0,
                                                      posinf=0.0, neginf=0.0)).float().to(device)
        starts = get_starts(len(latents), history, args.windows_per_demo,
                            max(args.horizons), args.long_window_count)
        done_windows = 0
        for start in starts:
            avail = len(latents) - start - history
            hs = [h for h in args.horizons if h <= avail]
            if not hs:
                continue
            # Start seed stable across raw/estimate runs, avoiding different permutations.
            window_seed = args.seed + (idx * 1_000_003) + start
            end = start + history + max(hs)
            try:
                metrics = diagnose_window(model, latents[start:end], raw_actions[start:end],
                                          history, hs, mean, std, window_seed)
            except (RuntimeError, ValueError, FloatingPointError) as exc:
                failures.append({"file": file, "episode": group, "start": start, "error": str(exc)})
                print(f"[WARN] failed {task}/{group} start={start}: {exc}", flush=True)
                continue
            frac = start / max(1, len(latents) - history - 1)
            phase = "early" if frac < 1/3 else "middle" if frac < 2/3 else "late"
            for h, score in metrics.items():
                rows.append({"file": file, "task": task, "episode": group, "frames": n,
                             "start": start, "phase": phase, "horizon": h, **score})
            done_windows += 1
        print(f"[{idx}/{len(demos)}] {task}: {group}, frames={n}, windows={done_windows}, "
              f"rows={len(rows)}", flush=True)

    if not rows:
        raise RuntimeError("No valid window evaluated. Check horizon list and dataset length.")
    summary = group_summary(rows)
    per_task = {task: group_summary([r for r in rows if r["task"] == task])
                for task in sorted({r["task"] for r in rows})}
    per_phase = {phase: group_summary([r for r in rows if r["phase"] == phase])
                 for phase in ("early", "middle", "late")}
    # 'High motion' here means a large change in encoder latent from the last observed frame;
    # this is only a proxy, NOT a semantic movement / grasp segmentation.
    motion = {}
    for h in args.horizons:
        subset = [r for r in rows if r["horizon"] == h]
        if len(subset) < 2:
            continue
        threshold = float(np.median([r["persistence_mse"] for r in subset]))
        motion[str(h)] = {"threshold_latent_persistence_mse": threshold,
                          "low": group_summary([r for r in subset if r["persistence_mse"] <= threshold]).get(str(h)),
                          "high": group_summary([r for r in subset if r["persistence_mse"] > threshold]).get(str(h))}

    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "test_type": "offline action-conditioned latent predictions; NOT LIBERO success rate",
        "checkpoint": str(args.checkpoint), "config": str(args.config),
        "dataset_root": str(args.libero_root), "code_root": str(args.code_root),
        "history_size": history, "horizons": args.horizons, "task_coverage": task_counts,
        "count_episodes": len(demos), "count_windows_unique": len({(r['file'],r['episode'],r['start']) for r in rows}),
        "action_norm": norm_source, "action_mean": mean.tolist(), "action_std": std.tolist(),
        "image_preprocessing": "Inherited from previous evaluator: resize224 + Imagenet RGB; check against training",
        "action_variants": {
            "actual": "Recorded actions from LIBERO demonstrations",
            "zero": "RAW zero action, then normalized using same mean/std",
            "shuffled": "Window-level temporal permutation of RAW actions, same values but changed order",
        },
        "warning": "Checkpoint loading strict does not validate data semantics / train-test disjointness. "
                   "Latent MSE is NOT task success. Verify original image pipeline and action normalization.",
        "summary": summary, "by_task": per_task, "by_phase": per_phase,
        "by_latent_motion": motion, "failed_windows": failures,
        "csv": str(csv_path),
    }
    output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print("\nROLLING-WINDOW DIAGNOSIS (per-dimension latent MSE; lower is better)")
    print(" K | windows | actual_free | teacher | persistence | zero_free | shuffle_free | free/base | teacher_delta_zero")
    for h in args.horizons:
        s = summary.get(str(h))
        if not s:
            print(f"{h:3d} | 0 (no valid windows)")
            continue
        print(f"{h:3d} | {s['windows']:7d} | {s['free_mse']:.6f} | {s['teacher_mse']:.6f} | "
              f"{s['persistence_mse']:.6f} | {s['zero_free_mse']:.6f} | "
              f"{s['shuffled_free_mse']:.6f} | {s['free_over_persistence']:.2f} | "
              f"{s['teacher_delta_zero']:.8f}")
    print("\nTeacher true-actions win rates vs zero/shuffled (larger than 0.5 is better):")
    for h, s in summary.items():
        print(f" K={h}: vs_zero={s['fraction_actual_teacher_beats_zero']:.1%}; "
              f"vs_shuffled={s['fraction_actual_teacher_beats_shuffled']:.1%}; "
              f"free_beats_persistence={s['fraction_actual_free_beats_persistence']:.1%}")
    print(f"\nSaved JSON: {output}\nSaved CSV: {csv_path}")
    if failures:
        print(f"WARNING: {len(failures)} windows failed; inspect failed_windows in JSON")
    print("CAUTION: Open-loop divergences may reflect incorrect preprocessing, action alignment, "
          "or generalization; use trained normalization and verify dataset split.")


if __name__ == "__main__":
    main()
