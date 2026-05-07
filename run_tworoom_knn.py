"""Tworoom KNN-Oracle resolution run (2026-05-04).

6 cells = Tworoom × {Oracle PRISM-MPPI σ=0.1, vanilla MPPI} × seeds {0, 1, 42}.

Resolves the 27% vs 64% discrepancy in our records (which turned out to be a
doc-only issue — 27% was hallucinated by me earlier in this conversation;
64% is the only real historical value from §05.1b / §07.3 single-key proprio).

KNN-Oracle config matches §05.1b's single-key proprio setup (auto-fallback
from default match_key='state' since tworoom h5 lacks 'state').
"""
import datetime
import os
import re
import subprocess
import time
from pathlib import Path

ROOT = Path("/home/yuhai/project/prism-jepa")
LOG_DIR = ROOT / "results" / "2026-05-04_tworoom_knn_resolution"
LOG_DIR.mkdir(parents=True, exist_ok=True)

COMMIT_SHA = subprocess.check_output(
    ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
).strip()

# Priority order: seed=42 first (matches §05.1b's single-seed 64%) so we can
# confirm reproducibility of that exact historical number first.
CELLS = [
    ("oracle",       42),
    ("oracle",       0),
    ("oracle",       1),
    ("vanilla_mppi", 42),
    ("vanilla_mppi", 0),
    ("vanilla_mppi", 1),
]


def cmd_for(method: str, seed: int) -> list[str]:
    if method == "oracle":
        return [
            "python", "eval_oracle.py",
            "--config-name=tworoom",
            "policy=tworooms/lewm",
            "solver=mppi",
            "+oracle.sigma=0.1",
            "eval.num_eval=50",
            f"seed={seed}",
        ]
    elif method == "vanilla_mppi":
        return [
            "python", "eval.py",
            "--config-name=tworoom",
            "policy=tworooms/lewm",
            "solver=mppi",
            "eval.num_eval=50",
            f"seed={seed}",
        ]
    raise ValueError(method)


def run_cell(method: str, seed: int):
    log = LOG_DIR / f"{method}_seed{seed}.log"
    cmd = cmd_for(method, seed)
    header = (
        f"# task=tworoom method={method} seed={seed} N=50\n"
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
    print(f"=== Tworoom KNN resolution — commit {COMMIT_SHA[:8]} ===")
    print(f"Log dir: {LOG_DIR}")
    print()

    results = []
    t_start = time.time()
    for i, (method, seed) in enumerate(CELLS, 1):
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        status, sr, elapsed = run_cell(method, seed)
        sr_str = f"{sr:.1f}" if sr is not None else "----"
        print(f"[{ts}] {i}/{len(CELLS)}  {method:<14s} seed={seed:<3d}  "
              f"SR={sr_str:>6s}  ({elapsed:5.1f}s)  {status}", flush=True)
        results.append((method, seed, status, sr, elapsed))

    total = time.time() - t_start
    print(f"\nTotal wall-clock: {total/60:.1f} min ({total:.0f}s)")

    # Paired-Δ recompute
    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    by_method = {"oracle": {}, "vanilla_mppi": {}}
    for method, seed, status, sr, _ in results:
        by_method[method][seed] = sr

    for method in ("vanilla_mppi", "oracle"):
        srs = [by_method[method].get(s) for s in [0, 1, 42]]
        srs_str = ", ".join(f"{s:.1f}" if s is not None else "—" for s in srs)
        valid = [s for s in srs if s is not None]
        if valid:
            mean = sum(valid) / len(valid)
            std = (sum((x - mean) ** 2 for x in valid) / max(1, len(valid) - 1)) ** 0.5
            print(f"  {method:<14s}: per-seed (s=0,1,42) = [{srs_str}]  mean={mean:.1f} ± {std:.2f}")

    # Paired Δ
    if all(s is not None for method in by_method.values() for s in method.values()):
        diffs = [by_method["oracle"][s] - by_method["vanilla_mppi"][s] for s in [0, 1, 42]]
        mean_d = sum(diffs) / 3
        std_d = (sum((d - mean_d) ** 2 for d in diffs) / 2) ** 0.5
        signs_pos = sum(1 for d in diffs if d > 0)
        signs_neg = sum(1 for d in diffs if d < 0)
        signs_zero = sum(1 for d in diffs if d == 0)
        print(f"\n  Paired Δ (Oracle − vanilla MPPI):")
        print(f"    per-seed (s=0,1,42): {tuple(round(d, 1) for d in diffs)}")
        print(f"    mean Δ = {mean_d:+.2f} ± {std_d:.2f}")
        print(f"    signs: {signs_pos}/3 positive, {signs_neg}/3 negative, {signs_zero}/3 zero")
        if std_d > 1e-6:
            from math import sqrt
            t = mean_d / (std_d / sqrt(3))
            try:
                from scipy import stats as scstats
                p = float(2 * (1 - scstats.t.cdf(abs(t), df=2)))
            except ImportError:
                p = float("nan")
            print(f"    paired-t = {t:+.2f}, df=2, p ≈ {p:.4f}")


if __name__ == "__main__":
    main()
