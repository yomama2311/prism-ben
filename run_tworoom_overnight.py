"""Tworoom overnight batch (2026-05-04 / 2026-05-05).

Phase 1: Build JEPA cache → train prior head → check success criteria.
Phase 2: 7 methods × 3 seeds = 21 cells, in priority order, each capped at 30 min.

Incremental writes to docs/19_tworoom_full_coverage.md so partial completion
is recoverable. All cell raw outputs to results/2026-05-04_tworoom_overnight/.
"""
import datetime
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/home/yuhai/project/prism-jepa")
LOG_DIR = ROOT / "results" / "2026-05-04_tworoom_overnight"
LOG_DIR.mkdir(parents=True, exist_ok=True)
DOC_PATH = ROOT / "docs" / "19_tworoom_full_coverage.md"
FAIL_LOG = LOG_DIR / "failures.log"
TRAINING_LOG = LOG_DIR / "phase1_training_log.txt"
CACHE_LOG = LOG_DIR / "phase1_cache_log.txt"
CKPT_PATH = ROOT / "prior_head_tworoom.pt"

CUBE_CKPT = ROOT / "prior_head_cube.pt"
PUSHT_CKPT = ROOT / "prior_head_pusht.pt"
TWO_CKPT = CKPT_PATH

ENV = {**os.environ, "STABLEWM_HOME": str(ROOT / ".stable-wm")}
COMMIT = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
SEEDS = [0, 1, 42]
PHASE2_CELL_TIMEOUT_SEC = 30 * 60
PHASE2_TOTAL_TIMEOUT_SEC = 8 * 3600
PHASE1_TIMEOUT_SEC = 4 * 3600


# ---------------- Logging helpers ----------------
def log_msg(msg):
    ts = datetime.datetime.now().isoformat(timespec="seconds")
    print(f"[{ts}] {msg}", flush=True)


def append_failure(method, seed, err_msg):
    with open(FAIL_LOG, "a") as f:
        f.write(f"[{datetime.datetime.now().isoformat()}] {method} seed={seed}: {err_msg}\n")


# ---------------- Doc update (incremental) ----------------
DOC_HEADER = """# 19. Tworoom Full PRISM Method Coverage (2026-05-04 / 2026-05-05 overnight batch)

> **Status:** writing incrementally as cells complete; if you see this header alone, batch is still running or crashed early.

Commit: `{commit}`
Run start: `{start}`
"""


def init_doc():
    if not DOC_PATH.exists():
        DOC_PATH.write_text(DOC_HEADER.format(
            commit=COMMIT, start=datetime.datetime.now().isoformat(timespec="seconds")
        ))


def append_doc(text):
    with open(DOC_PATH, "a") as f:
        f.write(text)


# ---------------- Phase 1 helpers ----------------
def run_subprocess(cmd, log_path, timeout_sec, cwd=ROOT):
    """Run a command, capture full output to log_path. Returns (returncode, elapsed_s, timed_out)."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        f"# command: {' '.join(map(str, cmd))}\n"
        f"# commit: {COMMIT}\n"
        f"# date: {datetime.datetime.now().isoformat()}\n"
        f"---\n"
    )
    t0 = time.time()
    try:
        with open(log_path, "a") as f:
            proc = subprocess.run(
                cmd, cwd=cwd, stdout=f, stderr=subprocess.STDOUT,
                env=ENV, timeout=timeout_sec,
            )
        elapsed = time.time() - t0
        return proc.returncode, elapsed, False
    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        with open(log_path, "a") as f:
            f.write(f"\n--- TIMED OUT after {elapsed:.0f}s (cap={timeout_sec}s) ---\n")
        return -1, elapsed, True


def parse_training_log_metrics(log_path):
    """Extract best epoch, best val NLL, best val MSE from train_prior_head.py log."""
    text = Path(log_path).read_text()
    # Best line pattern: "saved best (val NLL=X.XXXX, val MSE=X.XXXX, epoch N) → ..."
    m = re.search(
        r"saved best.*?val NLL=([\d.\-eE]+).*?val MSE=([\d.\-eE]+).*?epoch (\d+)",
        text,
    )
    if m:
        return {
            "best_val_nll": float(m.group(1)),
            "best_val_mse": float(m.group(2)),
            "best_epoch": int(m.group(3)),
            "saved": True,
        }
    # Try alternate older pattern
    m_alt = re.search(r"best.*?val NLL.*?(-?\d+\.\d+)", text)
    if m_alt:
        return {"best_val_nll": float(m_alt.group(1)),
                "best_val_mse": None, "best_epoch": None, "saved": True}
    # Init values
    init_mse = None
    m_init = re.search(r"EPOCH 0 \(init\).*?val MSE=([\d.]+)", text)
    if m_init:
        init_mse = float(m_init.group(1))
    return {"best_val_nll": None, "best_val_mse": None, "best_epoch": None,
            "saved": False, "init_val_mse": init_mse}


# ---------------- Phase 1: cache + train ----------------
def phase1():
    log_msg("=" * 60)
    log_msg("PHASE 1: JEPA cache + prior head training")
    log_msg("=" * 60)
    phase1_t0 = time.time()

    # 1a. JEPA cache for tworoom
    log_msg("Phase 1a: building JEPA cache for tworoom...")
    rc, elapsed, timed_out = run_subprocess(
        ["python", "jepa_cache.py", "--task", "tworoom", "--batch", "128"],
        CACHE_LOG, timeout_sec=30 * 60,
    )
    if rc != 0:
        log_msg(f"Phase 1a FAILED (rc={rc}, timed_out={timed_out}, elapsed={elapsed:.0f}s)")
        return {"phase1_ok": False, "stage": "cache", "elapsed": time.time() - phase1_t0,
                "rc": rc, "timed_out": timed_out}
    cache_dir = ROOT / ".stable-wm" / "jepa_cache" / "tworoom"
    if not (cache_dir / "features.npy").exists():
        log_msg("Phase 1a FAILED: features.npy not produced")
        return {"phase1_ok": False, "stage": "cache",
                "elapsed": time.time() - phase1_t0,
                "rc": rc, "missing_features": True}
    log_msg(f"  cache OK in {elapsed:.0f}s")

    # 1b. Train prior head
    log_msg("Phase 1b: training prior head...")
    train_remaining = PHASE1_TIMEOUT_SEC - (time.time() - phase1_t0)
    train_remaining = max(60, int(train_remaining))
    rc, elapsed, timed_out = run_subprocess(
        ["python", "train_prior_head.py",
         "--task", "tworoom", "--max-epochs", "15", "--patience", "5"],
        TRAINING_LOG, timeout_sec=train_remaining,
    )
    if rc != 0 or timed_out:
        log_msg(f"Phase 1b FAILED (rc={rc}, timed_out={timed_out}, elapsed={elapsed:.0f}s)")
        return {"phase1_ok": False, "stage": "train",
                "elapsed": time.time() - phase1_t0,
                "rc": rc, "timed_out": timed_out}
    log_msg(f"  training run completed in {elapsed:.0f}s")

    # 1c. Success criteria
    metrics = parse_training_log_metrics(TRAINING_LOG)
    log_msg(f"  training metrics: {metrics}")

    if not CKPT_PATH.exists():
        log_msg("Phase 1c FAILED: prior_head_tworoom.pt not saved")
        return {"phase1_ok": False, "stage": "criteria",
                "elapsed": time.time() - phase1_t0,
                "reason": "checkpoint not saved",
                "metrics": metrics}

    if metrics["best_val_nll"] is None or not (metrics["best_val_nll"] == metrics["best_val_nll"]):
        log_msg("Phase 1c FAILED: best val NLL is NaN or unparseable")
        return {"phase1_ok": False, "stage": "criteria",
                "elapsed": time.time() - phase1_t0,
                "reason": "best val NLL NaN/unparseable",
                "metrics": metrics}

    # PushT val MSE was 0.73; threshold = 10x → 7.3
    PUSHT_VAL_MSE = 0.73
    THRESHOLD = 10.0 * PUSHT_VAL_MSE
    if metrics["best_val_mse"] is not None and metrics["best_val_mse"] > THRESHOLD:
        log_msg(f"Phase 1c FAILED: best val MSE {metrics['best_val_mse']:.4f} > 10x PushT ({THRESHOLD:.2f})")
        return {"phase1_ok": False, "stage": "criteria",
                "elapsed": time.time() - phase1_t0,
                "reason": f"val MSE {metrics['best_val_mse']:.4f} exceeds 10x PushT threshold {THRESHOLD:.2f}",
                "metrics": metrics}

    log_msg("Phase 1 SUCCESS — proceeding to Phase 2")
    return {"phase1_ok": True, "elapsed": time.time() - phase1_t0,
            "metrics": metrics}


# ---------------- Phase 2: 7 methods × 3 seeds ----------------
def cmd_for(method, seed):
    """Return command list for a Phase 2 cell."""
    common_head = ["+head.ckpt=" + str(TWO_CKPT)]
    base = ["python", "eval_prism_head.py",
            "--config-name=tworoom", "policy=tworooms/lewm",
            "eval.num_eval=50", f"seed={seed}"]
    if method == "mode_none":
        return base + common_head + ["solver=mppi", "+head.injection_mode=none"]
    if method == "bc_only":
        return base + common_head + ["solver=mppi", "+head.injection_mode=bc_only"]
    if method == "warm_start_mppi":
        return base + common_head + ["solver=mppi", "+head.injection_mode=warm_start"]
    if method == "prism_sigma2":
        return base + common_head + ["solver=mppi", "+head.injection_mode=pog",
                                     "+head.sigma_scale=2.0"]
    if method == "prism_sigma2_n1200":
        return base + common_head + ["solver=mppi", "+head.injection_mode=pog",
                                     "+head.sigma_scale=2.0", "solver.num_samples=1200"]
    if method == "prism_sigma10000":
        return base + common_head + ["solver=mppi", "+head.injection_mode=pog",
                                     "+head.sigma_scale=10000"]
    if method == "cem_warm_start":
        return base + common_head + ["solver=cem", "+head.injection_mode=warm_start"]
    raise ValueError(method)


PHASE2_METHODS = [
    ("mode_none",          "1. mode=none MPPI (sanity)"),
    ("bc_only",            "2. BC-only"),
    ("prism_sigma10000",   "3. PRISM-MPPI σ_scale=10000 (asymptote)"),
    ("warm_start_mppi",    "4. warm-start MPPI"),
    ("prism_sigma2",       "5. PRISM-MPPI σ=2.0 (HEADLINE)"),
    ("cem_warm_start",     "6. CEM-warm-start"),
    ("prism_sigma2_n1200", "7. PRISM-MPPI σ=2.0 N=1200"),
]


def parse_sr_from_log(log_path):
    text = Path(log_path).read_text()
    m = re.search(r"success_rate[^0-9]+([0-9.]+)", text)
    return float(m.group(1)) if m else None


def phase2():
    log_msg("=" * 60)
    log_msg("PHASE 2: full PRISM method coverage on Tworoom")
    log_msg("=" * 60)
    phase2_t0 = time.time()
    consecutive_failures = 0
    method_results = {}  # method -> {seed: {sr, status, elapsed}}

    for method, label in PHASE2_METHODS:
        if time.time() - phase2_t0 > PHASE2_TOTAL_TIMEOUT_SEC:
            log_msg(f"Phase 2 TOTAL TIMEOUT reached ({PHASE2_TOTAL_TIMEOUT_SEC}s); stopping launches")
            break

        log_msg(f"\n--- {label} ---")
        method_results[method] = {}

        for seed in SEEDS:
            assert seed in (0, 1, 42), f"BAD SEED {seed}"
            log_path = LOG_DIR / f"{method}_seed{seed}.log"
            cmd = cmd_for(method, seed)
            rc, elapsed, timed_out = run_subprocess(
                cmd, log_path, timeout_sec=PHASE2_CELL_TIMEOUT_SEC,
            )
            sr = parse_sr_from_log(log_path) if rc == 0 else None
            if timed_out:
                status = "TIMED_OUT"
                consecutive_failures += 1
            elif rc != 0:
                status = f"FAILED(rc={rc})"
                append_failure(method, seed,
                               f"rc={rc}, log={log_path.name}")
                consecutive_failures += 1
            elif sr is None:
                status = "PARSE_FAIL"
                append_failure(method, seed,
                               f"could not parse success_rate from {log_path.name}")
                consecutive_failures += 1
            else:
                status = "COMPLETED"
                consecutive_failures = 0

            method_results[method][seed] = {"sr": sr, "status": status, "elapsed": elapsed}
            log_msg(f"  seed={seed}: SR={sr if sr is not None else '----'}  "
                    f"({elapsed:.0f}s)  {status}")

            if consecutive_failures >= 3:
                log_msg(f"3 consecutive failures — aborting Phase 2 (likely systemic issue)")
                append_failure("BATCH", "ABORT", "3 consecutive failures")
                return method_results, time.time() - phase2_t0, True

        # Append to doc after each method
        write_method_section(method, label, method_results[method])

    return method_results, time.time() - phase2_t0, False


# ---------------- Doc writing ----------------
def write_method_section(method, label, seed_results):
    """Append a method-row to docs/19."""
    line = f"\n### {label}\n\n"
    line += "| seed | SR | status | elapsed (s) |\n"
    line += "|---|---|---|---|\n"
    valid_srs = []
    for seed in SEEDS:
        r = seed_results[seed]
        sr_disp = f"{r['sr']:.1f}" if r["sr"] is not None else "—"
        line += f"| {seed} | {sr_disp} | {r['status']} | {r['elapsed']:.0f} |\n"
        if r["sr"] is not None:
            valid_srs.append(r["sr"])
    if len(valid_srs) >= 2:
        mean = sum(valid_srs) / len(valid_srs)
        std = (sum((x - mean) ** 2 for x in valid_srs) / max(1, len(valid_srs) - 1)) ** 0.5
        line += f"\n**mean ± std (n={len(valid_srs)}):** {mean:.1f} ± {std:.2f}\n"
    elif len(valid_srs) == 1:
        line += f"\n**single seed only:** {valid_srs[0]:.1f}\n"
    else:
        line += "\n**all seeds failed** — see failures.log\n"
    append_doc(line)


def write_phase1_section(p1_result):
    """Phase 1 outcome section."""
    text = "\n---\n\n## Phase 1 — head training\n\n"
    if p1_result["phase1_ok"]:
        m = p1_result["metrics"]
        text += (
            f"**Status: COMPLETED**\n\n"
            f"- best epoch: {m['best_epoch']}\n"
            f"- best val NLL: {m['best_val_nll']:.4f}\n"
            f"- best val MSE: {m['best_val_mse']:.4f}\n"
            f"- elapsed: {p1_result['elapsed']:.0f}s\n"
            f"- reference (PushT): val MSE = 0.73 → 10× threshold = 7.30 (passed)\n"
            f"- reference (Cube):  val MSE = 0.61\n\n"
            f"Tworoom val MSE relative to PushT/Cube: "
            f"{m['best_val_mse'] / 0.73:.2f}× PushT, "
            f"{m['best_val_mse'] / 0.61:.2f}× Cube.\n"
        )
    else:
        text += (
            f"**Status: FAILED at stage `{p1_result.get('stage')}`**\n\n"
            f"- reason: {p1_result.get('reason') or 'rc != 0 or timed out'}\n"
            f"- elapsed: {p1_result['elapsed']:.0f}s\n"
            f"- metrics: {p1_result.get('metrics')}\n\n"
            f"Phase 2 SKIPPED.\n"
        )
    append_doc(text)


# ---------------- Final summary section ----------------
def write_summary(p1_result, p2_results, p2_elapsed, p2_aborted):
    text = "\n---\n\n## Summary\n\n"

    # Sanity checks
    sanity_text = ""
    flagged = []
    docs15_vanilla_mppi = {0: 70.0, 1: 76.0, 42: 66.0}
    docs15_vanilla_cem = {0: 88.0, 1: 96.0, 42: 86.0}

    if "mode_none" in p2_results:
        modes = p2_results["mode_none"]
        sanity_text += "\n### Sanity: mode=none MPPI vs vanilla MPPI (docs/15)\n\n"
        sanity_text += "| seed | mode=none | docs/15 vanilla | Δ |\n|---|---|---|---|\n"
        for seed in SEEDS:
            mr = modes.get(seed, {}).get("sr")
            vr = docs15_vanilla_mppi[seed]
            diff = (mr - vr) if mr is not None else None
            sanity_text += f"| {seed} | {mr if mr is not None else '—'} | {vr} | {diff if diff is not None else '—'} |\n"

    if "prism_sigma10000" in p2_results:
        sanity_text += "\n### Sanity: PRISM σ=10000 vs vanilla MPPI (docs/15)\n\n"
        sanity_text += "| seed | σ=10000 | docs/15 vanilla | Δ |\n|---|---|---|---|\n"
        for seed in SEEDS:
            mr = p2_results["prism_sigma10000"].get(seed, {}).get("sr")
            vr = docs15_vanilla_mppi[seed]
            diff = (mr - vr) if mr is not None else None
            sanity_text += f"| {seed} | {mr if mr is not None else '—'} | {vr} | {diff if diff is not None else '—'} |\n"

    # Headline paired Δs
    headline_text = ""

    def _paired_delta(a_method, b_lookup, label):
        """Compute paired Δ for a_method (in p2_results) vs b_lookup (dict of seed→SR)."""
        if a_method not in p2_results:
            return ""
        diffs = []
        per_seed = []
        for seed in SEEDS:
            ar = p2_results[a_method].get(seed, {}).get("sr")
            br = b_lookup.get(seed)
            if ar is None or br is None:
                per_seed.append((seed, None))
            else:
                diffs.append(ar - br)
                per_seed.append((seed, ar - br))
        if len(diffs) < 2:
            return f"\n### {label}\n\nInsufficient data ({len(diffs)} valid seeds).\n"
        mean_d = sum(diffs) / len(diffs)
        std_d = (sum((d - mean_d) ** 2 for d in diffs) / max(1, len(diffs) - 1)) ** 0.5
        signs_pos = sum(1 for d in diffs if d > 0)
        out = f"\n### {label}\n\nper-seed Δ:\n\n| seed | Δ |\n|---|---|\n"
        for s, d in per_seed:
            out += f"| {s} | {d:+.1f} |\n" if d is not None else f"| {s} | — |\n"
        out += f"\n**mean Δ = {mean_d:+.2f} ± {std_d:.2f} pp** (n={len(diffs)}, "
        out += f"{signs_pos}/{len(diffs)} positive"
        if std_d > 1e-6:
            from math import sqrt
            t = mean_d / (std_d / sqrt(len(diffs)))
            try:
                from scipy import stats as scstats
                p = float(2 * (1 - scstats.t.cdf(abs(t), df=len(diffs) - 1)))
                out += f", paired-t = {t:+.2f}, p ≈ {p:.4f})\n"
            except ImportError:
                out += f", paired-t = {t:+.2f})\n"
        else:
            out += ")\n"
        return out

    headline_text += _paired_delta("prism_sigma2", docs15_vanilla_mppi,
                                   "Headline 1: PRISM-MPPI σ=2.0 vs vanilla MPPI (docs/15)")
    if "warm_start_mppi" in p2_results:
        warm_lookup = {s: p2_results["warm_start_mppi"].get(s, {}).get("sr") for s in SEEDS}
        headline_text += _paired_delta("prism_sigma2", warm_lookup,
                                       "Headline 2: PRISM-MPPI σ=2.0 vs warm-start MPPI (this run)")
    headline_text += _paired_delta("cem_warm_start", docs15_vanilla_cem,
                                   "Headline 3: CEM-warm-start vs vanilla CEM (docs/15)")

    text += sanity_text + headline_text
    text += f"\n---\n\n## Wall-clock\n\n"
    text += f"- Phase 1: {p1_result['elapsed']:.0f}s ({p1_result['elapsed']/60:.1f} min)\n"
    text += f"- Phase 2: {p2_elapsed:.0f}s ({p2_elapsed/60:.1f} min)"
    if p2_aborted:
        text += " — **ABORTED early due to 3 consecutive failures**"
    text += f"\n- Total: {(p1_result['elapsed'] + p2_elapsed)/60:.1f} min\n"

    # Failures
    if FAIL_LOG.exists():
        text += f"\n---\n\n## Failures\n\nSee `{FAIL_LOG.name}`. Excerpt:\n\n```\n"
        text += FAIL_LOG.read_text()[:3000]
        text += "\n```\n"

    append_doc(text)


# ---------------- Anomaly flagging ----------------
def write_anomalies(p2_results):
    """Insert ⚠️ ATTENTION section near top if any flag triggered."""
    flags = []
    docs15_vanilla_mppi_seed42 = 66.0  # for reference

    if "bc_only" in p2_results:
        bc_srs = [p2_results["bc_only"][s]["sr"] for s in SEEDS
                  if p2_results["bc_only"].get(s, {}).get("sr") is not None]
        if bc_srs and all(s < 20 for s in bc_srs):
            flags.append(f"BC-only SR < 20% on all valid seeds: {bc_srs} → head essentially random")

    if "prism_sigma2" in p2_results:
        prism_srs = [p2_results["prism_sigma2"][s]["sr"] for s in SEEDS
                     if p2_results["prism_sigma2"].get(s, {}).get("sr") is not None]
        baseline_mean = 70.7  # docs/15 mean
        if prism_srs and all(s < baseline_mean for s in prism_srs):
            flags.append(f"PRISM-MPPI σ=2.0 < vanilla MPPI baseline {baseline_mean} on all seeds: {prism_srs}")

    if "prism_sigma10000" in p2_results:
        srs10k = [p2_results["prism_sigma10000"][s]["sr"] for s in SEEDS
                  if p2_results["prism_sigma10000"].get(s, {}).get("sr") is not None]
        baseline_mean = 70.7
        if srs10k:
            mean_10k = sum(srs10k) / len(srs10k)
            if abs(mean_10k - baseline_mean) > 30:
                flags.append(f"PRISM σ=10000 mean {mean_10k:.1f} differs from baseline {baseline_mean} by >30pp — pipeline issue?")

    if flags:
        text = "\n## ⚠️ ATTENTION — anomalies detected\n\n"
        for f in flags:
            text += f"- {f}\n"
        text += "\nReview these carefully before drawing conclusions.\n\n"
        append_doc(text)


# ---------------- Main ----------------
def main():
    log_msg(f"Tworoom overnight batch start | commit={COMMIT[:8]}")
    init_doc()

    p1 = phase1()
    write_phase1_section(p1)

    if not p1["phase1_ok"]:
        log_msg("Phase 1 failed — skipping Phase 2")
        append_doc("\n*Phase 2 SKIPPED because Phase 1 failed.*\n")
        return

    p2_results, p2_elapsed, p2_aborted = phase2()
    write_anomalies(p2_results)
    write_summary(p1, p2_results, p2_elapsed, p2_aborted)
    log_msg("Tworoom overnight batch complete")


if __name__ == "__main__":
    main()
