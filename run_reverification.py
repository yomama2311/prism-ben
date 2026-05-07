"""54-cell PRISM reverification orchestrator (2026-05-04).

Replaces a fragile nested bash+heredoc batch script. Iterates the
9 methods × 2 tasks × 3 seeds matrix from today's spec, writes each
cell's command + provenance + raw output to its own log file, then
parses success_rates into a summary.
"""
import datetime
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/home/yuhai/project/prism-jepa")
LOG_DIR = ROOT / "results" / "2026-05-04_reverification"
LOG_DIR.mkdir(parents=True, exist_ok=True)

CUBE_CKPT = str(ROOT / "prior_head_cube.pt")
PUSHT_CKPT = str(ROOT / "prior_head_pusht.pt")

COMMIT_SHA = subprocess.check_output(
    ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
).strip()


def cmd_for(task: str, method: str, seed: int) -> list[str]:
    ckpt = CUBE_CKPT if task == "cube" else PUSHT_CKPT
    common = [
        "python", "", "--config-name=" + task,
        f"policy={task}/lewm",
        "eval.num_eval=50",
        f"seed={seed}",
    ]
    if method == "vanilla_mppi":
        return ["python", "eval.py", f"--config-name={task}",
                f"policy={task}/lewm", "solver=mppi",
                "eval.num_eval=50", f"seed={seed}"]
    if method == "vanilla_cem":
        return ["python", "eval.py", f"--config-name={task}",
                f"policy={task}/lewm", "solver=cem",
                "eval.num_eval=50", f"seed={seed}"]

    head_args = [f"+head.ckpt={ckpt}"]
    if method == "mode_none":
        return ["python", "eval_prism_head.py", f"--config-name={task}",
                f"policy={task}/lewm", "solver=mppi", *head_args,
                "+head.injection_mode=none", "eval.num_eval=50", f"seed={seed}"]
    if method == "bc_only":
        return ["python", "eval_prism_head.py", f"--config-name={task}",
                f"policy={task}/lewm", "solver=mppi", *head_args,
                "+head.injection_mode=bc_only", "eval.num_eval=50", f"seed={seed}"]
    if method == "warm_start_mppi":
        return ["python", "eval_prism_head.py", f"--config-name={task}",
                f"policy={task}/lewm", "solver=mppi", *head_args,
                "+head.injection_mode=warm_start", "eval.num_eval=50", f"seed={seed}"]
    if method == "prism_sigma2":
        return ["python", "eval_prism_head.py", f"--config-name={task}",
                f"policy={task}/lewm", "solver=mppi", *head_args,
                "+head.injection_mode=pog", "+head.sigma_scale=2.0",
                "eval.num_eval=50", f"seed={seed}"]
    if method == "prism_sigma2_n1200":
        return ["python", "eval_prism_head.py", f"--config-name={task}",
                f"policy={task}/lewm", "solver=mppi", *head_args,
                "+head.injection_mode=pog", "+head.sigma_scale=2.0",
                "solver.num_samples=1200", "eval.num_eval=50", f"seed={seed}"]
    if method == "prism_sigma10000":
        return ["python", "eval_prism_head.py", f"--config-name={task}",
                f"policy={task}/lewm", "solver=mppi", *head_args,
                "+head.injection_mode=pog", "+head.sigma_scale=10000",
                "eval.num_eval=50", f"seed={seed}"]
    if method == "cem_warm_start":
        return ["python", "eval_prism_head.py", f"--config-name={task}",
                f"policy={task}/lewm", "solver=cem", *head_args,
                "+head.injection_mode=warm_start", "eval.num_eval=50", f"seed={seed}"]
    raise ValueError(f"unknown method: {method}")


METHODS_TIER_1 = ["vanilla_mppi", "mode_none", "bc_only"]
METHODS_TIER_2 = ["warm_start_mppi", "prism_sigma2",
                  "prism_sigma2_n1200", "prism_sigma10000"]
METHODS_TIER_3 = ["vanilla_cem", "cem_warm_start"]
TASKS = ["cube", "pusht"]
SEEDS = [0, 1, 42]


def run_cell(task: str, method: str, seed: int) -> tuple[str, float | None]:
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
    print(f"=== PRISM full reverification — commit {COMMIT_SHA[:8]} ===")
    print(f"Log dir: {LOG_DIR}")
    print()

    cells = []
    for tier_methods in (METHODS_TIER_1, METHODS_TIER_2, METHODS_TIER_3):
        for task in TASKS:
            for method in tier_methods:
                for seed in SEEDS:
                    cells.append((task, method, seed))

    print(f"Total cells: {len(cells)}")
    print()

    results = []
    t_start = time.time()
    for i, (task, method, seed) in enumerate(cells, 1):
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        status, sr, elapsed = run_cell(task, method, seed)
        sr_str = f"{sr:.1f}" if sr is not None else "----"
        print(f"[{ts}] {i:2d}/{len(cells)}  {task:<6s} {method:<22s} seed={seed:<3d}  "
              f"SR={sr_str:>6s}  ({elapsed:5.1f}s)  {status}", flush=True)
        results.append((task, method, seed, status, sr, elapsed))

    total_elapsed = time.time() - t_start
    print()
    print(f"Total wall-clock: {total_elapsed/60:.1f} min ({total_elapsed:.0f}s)")

    # Summary
    print()
    print("=" * 60)
    print("SUMMARY (per-cell SR%)")
    print("=" * 60)
    print(f"{'task':<6s} {'method':<22s} {'seed':<5s} {'SR':>6s}")
    print("-" * 50)
    for task, method, seed, status, sr, elapsed in results:
        sr_str = f"{sr:.1f}" if sr is not None else status
        print(f"{task:<6s} {method:<22s} {seed:<5d} {sr_str:>6s}")


if __name__ == "__main__":
    main()
