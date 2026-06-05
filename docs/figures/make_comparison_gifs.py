"""Build side-by-side comparison GIFs: Vanilla MPPI vs PRISM-MPPI.

For each task, reads success arrays + rollout MP4s from two staged eval dirs
(vanilla / prism) — both runs use the same `cfg.seed` so env index i refers to
the same (episode, start_step) pair across the two — then renders comparison
GIFs for env indices where vanilla=fail but PRISM=success.

Source MP4 layout (from swm.World.evaluate_from_dataset, world.py:989-995):
    top-left   = rollout (live env render)
    top-right  = static goal image
    bot-left   = dataset demo trajectory at time t (drop)
    bot-right  = goal duplicate (drop)

Output layout per frame (with a 32-px header strip):
    [Vanilla MPPI rollout | PRISM-MPPI rollout | Goal]

Run:  python docs/figures/make_comparison_gifs.py --task pusht --max 5
      python docs/figures/make_comparison_gifs.py --task cube  --max 5
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np


HEADER_H = 32
FRAME_H = 224          # one rollout-tile is 224x224
FPS = 8

LABELS = {"vanilla": "Vanilla MPPI",
          "prism":   "PRISM-MPPI",
          "goal":    "Goal"}
COLORS = {"vanilla": (200, 60, 60),    # red (failure column)
          "prism":   (60, 160, 90),    # green (success column)
          "goal":    (60, 60, 60)}     # dark grey (reference)


def read_mp4_frames(path: Path) -> np.ndarray:
    """Read all frames as (T, H, W, 3) uint8 RGB."""
    reader = imageio.get_reader(str(path), "ffmpeg")
    frames = [f for f in reader]
    reader.close()
    return np.stack(frames)


def crop_topleft_topright(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Source 448x448 2x2 grid → (topleft, topright) each 224x224."""
    half_h, half_w = frame.shape[0] // 2, frame.shape[1] // 2
    return frame[:half_h, :half_w], frame[:half_h, half_w:]


def make_header(total_width: int, n_cols: int = 3) -> np.ndarray:
    """3-column header strip with colored labels."""
    img = np.full((HEADER_H, total_width, 3), 255, dtype=np.uint8)
    col_w = total_width // n_cols
    for i, key in enumerate(("vanilla", "prism", "goal")):
        x0 = i * col_w
        if i > 0:
            cv2.line(img, (x0, 0), (x0, HEADER_H), (215, 215, 215), 1)
        cv2.putText(img, LABELS[key],
                    (x0 + 10, HEADER_H - 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLORS[key], 1, cv2.LINE_AA)
    return img


def build_comparison_gif(van_mp4: Path, prism_mp4: Path, out_path: Path) -> int:
    van = read_mp4_frames(van_mp4)        # (T, 448, 448, 3)
    pri = read_mp4_frames(prism_mp4)
    T = min(van.shape[0], pri.shape[0])
    van, pri = van[:T], pri[:T]

    # Static goal — take top-right of frame 0 from vanilla (same in both runs).
    _, goal = crop_topleft_topright(van[0])

    header = make_header(3 * FRAME_H)
    frames = []
    for t in range(T):
        van_tile, _ = crop_topleft_topright(van[t])
        pri_tile, _ = crop_topleft_topright(pri[t])
        row = np.hstack([van_tile, pri_tile, goal])
        frames.append(np.vstack([header, row]))

    imageio.mimsave(out_path, frames, fps=FPS, loop=0)
    return T


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=("pusht", "cube"))
    ap.add_argument("--max", type=int, default=5,
                    help="Maximum contrast pairs to render")
    ap.add_argument("--vanilla-dir", default=None)
    ap.add_argument("--prism-dir", default=None)
    ap.add_argument("--out-dir", default="docs/figures")
    args = ap.parse_args()

    van_dir = Path(args.vanilla_dir or f"/tmp/cmp_{args.task}/vanilla")
    pri_dir = Path(args.prism_dir   or f"/tmp/cmp_{args.task}/prism")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    v_succ = np.load(van_dir / "episode_successes.npy")
    p_succ = np.load(pri_dir / "episode_successes.npy")
    contrast = np.nonzero((~v_succ) & p_succ)[0]

    print(f"task={args.task}: "
          f"vanilla={int(v_succ.sum())}/{len(v_succ)}, "
          f"prism={int(p_succ.sum())}/{len(p_succ)}, "
          f"contrast={len(contrast)}")
    print(f"  contrast indices: {contrast.tolist()}")

    picks = contrast[: args.max]
    print(f"  rendering first {len(picks)}: {picks.tolist()}\n")

    for i in picks:
        v = van_dir / f"rollout_{i}.mp4"
        p = pri_dir / f"rollout_{i}.mp4"
        if not (v.exists() and p.exists()):
            print(f"  SKIP env {i}: missing one of {v.name} / {p.name}")
            continue
        out = out_dir / f"cmp_{args.task}_env{int(i)}.gif"
        T = build_comparison_gif(v, p, out)
        print(f"  wrote {out}  ({T} frames)")


if __name__ == "__main__":
    main()
