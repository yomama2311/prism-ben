"""LeWM baseline scoping run (2026-05-04).

12 cells = 2 tasks × 2 methods × 3 seeds:
  Tworoom × {vanilla MPPI, vanilla CEM} × {0, 1, 42}
  Reacher × {vanilla MPPI, vanilla CEM} × {0, 1, 42}

Vanilla baselines only — no head, no PRISM. Uses eval.py with the existing
hydra configs (config/eval/{tworoom,reacher}.yaml). Hyperparameters identical
to those used for Cube/PushT (no per-task retuning, per spec).

Per-task policy paths follow the .stable-wm/<dir>/lewm_object.ckpt convention:
  Tworoom → policy=tworooms/lewm  (note: plural per directory name)
  Reacher → policy=reacher/lewm
"""
import datetime
import os
import re
import subprocess
import time
from pathlib import Path

ROOT = Path("/home/yuhai/project/prism-jepa")
LOG_DIR = ROOT / "results" / "2026-05-04_lewm_baselines"
LOG_DIR.mkdir(parents=True, exist_ok=True)

COMMIT_SHA = subprocess.check_output(
    ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
).strip()

CELLS = [
    # Tworoom
    ("tworoom", "vanilla_mppi", 0),
    ("tworoom", "vanilla_mppi", 1),
    ("tworoom", "vanilla_mppi", 42),
    ("tworoom", "vanilla_cem",  0),
    ("tworoom", "vanilla_cem",  1),
    ("tworoom", "vanilla_cem",  42),
    # Reacher
    ("reacher", "vanilla_mppi", 0),
    ("reacher", "vanilla_mppi", 1),
    ("reacher", "vanilla_mppi", 42),
    ("reacher", "vanilla_cem",  0),
    ("reacher", "vanilla_cem",  1),
    ("reacher", "vanilla_cem",  42),
]

# Map task → policy path (plural for tworoom per the directory name)
POLICY = {"tworoom": "tworooms/lewm", "reacher": "reacher/lewm"}


def cmd_for(task: str, method: str, seed: int) -> list[str]:
    solver = "mppi" if method == "vanilla_mppi" else "cem"
    return [
        "python", "eval.py",
        f"--config-name={task}",
        f"policy={POLICY[task]}",
        f"solver={solver}",
        "eval.num_eval=50",
        f"seed={seed}",
    ]


def run_cell(task: str, method: str, seed: int):
    log = LOG_DIR / f"{task}_{method}_seed{seed}.log"
    cmd = cmd_for(task, method, seed)
    header = (
        f"# task={task} method={method} seed={seed} N=50\n"
        f"# commit={COMMIT_SHA}\n"
        f"# date={datetime.datetime.now().isoformat()}\n"
        f"# command: {' '.join(cmd)}\n"
        f"---\n"
    )
    log.write_text(header)
    t0 = time.time()
    with open(log, "a") as f:
        proc = subprocess.run(
            cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT,
            env={**os.environ, "STABLEWM_HOME": str(ROOT / ".stable-wm")},
        )
    elapsed = time.time() - t0

    text = log.read_text()
    m = re.search(r"success_rate[^0-9]+([0-9.]+)", text)
    sr = float(m.group(1)) if m else None
    status = "OK" if proc.returncode == 0 and sr is not None else f"FAIL(rc={proc.returncode})"
    return status, sr, elapsed


def main():
    print(f"=== LeWM baselines scoping run — commit {COMMIT_SHA[:8]} ===")
    print(f"Log dir: {LOG_DIR}")
    print(f"Total cells: {len(CELLS)}")
    print()

    results = []
    t_start = time.time()
    for i, (task, method, seed) in enumerate(CELLS, 1):
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        status, sr, elapsed = run_cell(task, method, seed)
        sr_str = f"{sr:.1f}" if sr is not None else "----"
        print(f"[{ts}] {i:2d}/{len(CELLS)}  {task:<8s} {method:<14s} seed={seed:<3d}  "
              f"SR={sr_str:>6s}  ({elapsed:5.1f}s)  {status}", flush=True)
        results.append((task, method, seed, status, sr, elapsed))

    total = time.time() - t_start
    print(f"\nTotal wall-clock: {total/60:.1f} min ({total:.0f}s)")
    print()
    print("=" * 60)
    print("SUMMARY (per-cell SR%)")
    print("=" * 60)
    for task, method, seed, status, sr, elapsed in results:
        sr_str = f"{sr:.1f}" if sr is not None else status
        print(f"  {task:<8s} {method:<14s} seed={seed:<3d}  {sr_str:>6s}")


if __name__ == "__main__":
    main()
