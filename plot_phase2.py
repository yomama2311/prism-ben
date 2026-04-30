"""
Plot Phase 2 compute-quality trade-off curves.

Reads logs/phase2_results.tsv (produced by run_phase2.sh) and produces:
  logs/phase2_curves_{task}.png  — one figure per task
  logs/phase2_summary.png        — all tasks on one grid

Each figure: SR vs num_samples K, log-x, baseline + oracle curves.
"""

from pathlib import Path
import csv
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).parent
TSV = ROOT / "logs" / "phase2_results.tsv"


def load() -> dict:
    """Returns {task: {method: {K: sr}}}."""
    data: dict = {}
    if not TSV.exists():
        print(f"  Missing {TSV}", file=sys.stderr)
        return data
    with open(TSV) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            task = r["task"]
            method = r["method"]
            K = int(r["num_samples"])
            sr = r["sr"]
            if sr == "NA":
                continue
            data.setdefault(task, {}).setdefault(method, {})[K] = float(sr)
    return data


def plot_one(task: str, methods: dict, ax):
    colors = {"baseline": "C0", "oracle": "C1"}
    markers = {"baseline": "o", "oracle": "s"}
    for method, kv in methods.items():
        Ks = sorted(kv.keys())
        srs = [kv[K] for K in Ks]
        ax.plot(Ks, srs, label=method, color=colors.get(method, "gray"),
                marker=markers.get(method, "."), lw=2, ms=6)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("num_samples K")
    ax.set_ylabel("Success rate (%)")
    ax.set_title(task)
    ax.set_ylim(-5, 105)
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="lower right")


def main():
    data = load()
    if not data:
        print("No data to plot.")
        return

    # Per-task figures
    for task, methods in data.items():
        fig, ax = plt.subplots(figsize=(6, 4))
        plot_one(task, methods, ax)
        out = ROOT / "logs" / f"phase2_curves_{task}.png"
        fig.tight_layout()
        fig.savefig(out, dpi=120)
        plt.close(fig)
        print(f"  Saved {out}")

    # Summary grid
    n = len(data)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)
    for i, (task, methods) in enumerate(sorted(data.items())):
        ax = axes[i // cols, i % cols]
        plot_one(task, methods, ax)
    # hide unused
    for i in range(n, rows * cols):
        axes[i // cols, i % cols].axis("off")
    fig.suptitle("Phase 2: Compute-quality trade-off (SR vs num_samples K)",
                 fontweight="bold")
    fig.tight_layout()
    out = ROOT / "logs" / "phase2_summary.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  Saved {out}")

    # Print quantitative summary
    print("\n=== K* (smallest K with SR ≥ 90%) ===")
    print(f"{'Task':<12} {'Baseline':>10} {'Oracle':>10} {'Compute Saving':>16}")
    print("-" * 50)
    for task, methods in sorted(data.items()):
        def k_for_90(d):
            for K in sorted(d.keys()):
                if d[K] >= 90.0:
                    return K
            return None
        b = k_for_90(methods.get("baseline", {}))
        o = k_for_90(methods.get("oracle", {}))
        bs = f"{b}" if b else "—"
        os_ = f"{o}" if o else "—"
        sav = f"{b/o:.1f}×" if (b and o) else "—"
        print(f"{task:<12} {bs:>10} {os_:>10} {sav:>16}")


if __name__ == "__main__":
    main()
