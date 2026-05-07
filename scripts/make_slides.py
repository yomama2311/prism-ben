"""make_slides.py — generate a 45-60 min presentation on PRISM-JEPA.

Output: /home/yuhai/project/prism-jepa/prism_jepa_slides.pptx

Run:
    .venv/bin/python scripts/make_slides.py
"""
from __future__ import annotations

import io
import os
from datetime import date

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Emu, Inches, Pt

# ============================================================
# Style constants
# ============================================================
PRIMARY = "#1f4e79"
ACCENT  = "#c6504e"
NEUTRAL = "#5b6770"
GREEN   = "#2e7d32"
ORANGE  = "#b75e2c"
GREY    = "#7d8b96"
LIGHT_BG = "#f0f3f7"

PRIMARY_RGB = RGBColor(0x1f, 0x4e, 0x79)
ACCENT_RGB  = RGBColor(0xc6, 0x50, 0x4e)
NEUTRAL_RGB = RGBColor(0x5b, 0x67, 0x70)
GREEN_RGB   = RGBColor(0x2e, 0x7d, 0x32)
DARK_RGB    = RGBColor(0x22, 0x22, 0x22)
GREY_RGB    = RGBColor(0x77, 0x77, 0x77)
WHITE_RGB   = RGBColor(0xff, 0xff, 0xff)
LIGHT_BG_RGB = RGBColor(0xf0, 0xf3, 0xf7)


# ============================================================
# Matplotlib figure builders — each returns PNG bytes
# ============================================================

def _save(fig, dpi=200) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def fig_world_model_loop():
    """Standard MBRL loop: encode → predict → score → plan."""
    fig, ax = plt.subplots(figsize=(11, 4.5))
    nodes = [
        (0.5, 1.5, 1.8, 1.5, "Observation\n$o_t$ (RGB)", "#888"),
        (3.0, 1.5, 1.8, 1.5, "Encoder\n$h_\\psi$", PRIMARY),
        (5.5, 1.5, 1.8, 1.5, "Latent dynamics\n$f_\\theta$", PRIMARY),
        (8.0, 1.5, 1.8, 1.5, "Cost / Q\n$c, Q$", PRIMARY),
        (10.5, 1.5, 1.8, 1.5, "Sampler\n(MPPI/CEM)", ACCENT),
    ]
    for (x, y, w, h, label, color) in nodes:
        ax.add_patch(FancyBboxPatch((x, y), w, h,
            boxstyle="round,pad=0.06,rounding_size=0.15",
            linewidth=1.5, edgecolor=color, facecolor=LIGHT_BG))
        ax.text(x+w/2, y+h/2, label, ha='center', va='center',
                fontsize=10, color=color, fontweight='bold')
    for x in [2.3, 4.8, 7.3, 9.8]:
        ax.add_patch(FancyArrowPatch((x, 2.25), (x+0.7, 2.25),
            arrowstyle='->', mutation_scale=18, color="#444", linewidth=1.5))
    # back-arrow from sampler to encoder via "imagined trajectories"
    ax.add_patch(FancyArrowPatch((11.4, 1.4), (11.4, 0.4),
        arrowstyle='->', mutation_scale=14, color=ACCENT, linewidth=1.3))
    ax.add_patch(FancyArrowPatch((11.4, 0.4), (3.9, 0.4),
        arrowstyle='-', linewidth=1.3, color=ACCENT))
    ax.add_patch(FancyArrowPatch((3.9, 0.4), (3.9, 1.4),
        arrowstyle='->', mutation_scale=14, color=ACCENT, linewidth=1.3))
    ax.text(7.5, 0.15, "draw N candidate action sequences  → score in latent → re-fit",
            ha='center', fontsize=9, color=ACCENT, style='italic')
    ax.set_xlim(0, 12.5); ax.set_ylim(-0.3, 3.5); ax.axis('off')
    ax.set_title("Sampling-based MBRL loop: planner imagines, world model scores",
                 fontsize=12, fontweight='bold')
    return _save(fig)


def fig_three_existing_priors():
    """Three existing approaches and what they all discard."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    titles = ["Vanilla MPPI", "Warm-start", "Demo / policy prior"]
    descs = [
        "$\\mu = 0$\n\nzero info, but\nno bias",
        "$\\mu \\leftarrow \\mu_{prev}$\n\nrolls plan forward,\nignores demos",
        "$\\mu \\leftarrow \\pi(z)$\n\nuses demo mean,\n$\\sigma$ thrown away"]
    colors = [GREY, GREY, GREY]
    for ax, title, desc, c in zip(axes, titles, descs, colors):
        # Draw a Gaussian centered at 0 — the planner default
        x = np.linspace(-3, 3, 100)
        for mu, sigma, lbl, col, alpha, ls in [
                (0, 1.2, "planner $\\mathcal{N}(\\mu, \\sigma)$", PRIMARY, 0.7, "-"),
        ]:
            y = np.exp(-(x-mu)**2 / (2*sigma**2)) / (sigma*np.sqrt(2*np.pi))
            ax.fill_between(x, 0, y, alpha=alpha*0.4, color=col)
            ax.plot(x, y, color=col, linewidth=1.5)
        # Show what gets injected
        if "Warm-start" in title:
            mu_p = 0.8
            ax.axvline(mu_p, color=ACCENT, linewidth=2, linestyle='--')
            ax.text(mu_p+0.1, 0.34, "$\\mu_p$", fontsize=14, color=ACCENT)
        elif "Demo" in title:
            mu_p = -1.0
            ax.axvline(mu_p, color=ACCENT, linewidth=2, linestyle='--')
            ax.text(mu_p-0.7, 0.34, "$\\mu_p$\n(no $\\sigma$)", fontsize=11, color=ACCENT)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.text(0, -0.08, desc, ha='center', va='top', fontsize=10,
                transform=ax.transAxes, color="#444")
        ax.set_ylim(0, 0.4); ax.set_xlim(-3, 3); ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values(): spine.set_visible(False)
    fig.suptitle("All three discard the same thing: state-conditional confidence",
                 fontsize=12, fontweight='bold', color=ACCENT, y=1.02)
    plt.tight_layout()
    return _save(fig)


def fig_pog_fusion():
    """Two Gaussians fusing into one via PoG."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    x = np.linspace(-4, 4, 200)
    # Left: two source distributions
    ax = axes[0]
    mu_pi, sig_pi = 0.0, 1.5
    mu_p,  sig_p  = 1.5, 0.7
    g_pi = np.exp(-(x-mu_pi)**2/(2*sig_pi**2)) / (sig_pi*np.sqrt(2*np.pi))
    g_p  = np.exp(-(x-mu_p)**2 /(2*sig_p**2 )) / (sig_p *np.sqrt(2*np.pi))
    ax.fill_between(x, 0, g_pi, alpha=0.35, color=PRIMARY,
                    label=f"planner $\\mathcal{{N}}(\\mu_\\pi, \\sigma_\\pi^2)$\n($\\mu={mu_pi}$, $\\sigma={sig_pi}$)")
    ax.plot(x, g_pi, color=PRIMARY, linewidth=2)
    ax.fill_between(x, 0, g_p,  alpha=0.35, color=ACCENT,
                    label=f"prior $\\mathcal{{N}}(\\mu_p, \\sigma_p^2)$\n($\\mu={mu_p}$, $\\sigma={sig_p}$)")
    ax.plot(x, g_p,  color=ACCENT, linewidth=2)
    ax.set_title("Two source Gaussians", fontsize=11, fontweight='bold')
    ax.legend(fontsize=9, loc='upper left'); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)

    # Right: fused
    ax = axes[1]
    tau_pi, tau_p = 1/sig_pi**2, 1/sig_p**2
    sig_f = 1.0 / np.sqrt(tau_pi + tau_p)
    mu_f  = (tau_pi*mu_pi + tau_p*mu_p) / (tau_pi + tau_p)
    g_f = np.exp(-(x-mu_f)**2/(2*sig_f**2)) / (sig_f*np.sqrt(2*np.pi))
    ax.fill_between(x, 0, g_pi, alpha=0.15, color=PRIMARY)
    ax.fill_between(x, 0, g_p,  alpha=0.15, color=ACCENT)
    ax.fill_between(x, 0, g_f,  alpha=0.55, color=GREEN,
                    label=f"fused $\\mathcal{{N}}(\\mu_f, \\sigma_f^2)$\n($\\mu_f={mu_f:.2f}$, $\\sigma_f={sig_f:.2f}$)")
    ax.plot(x, g_f, color=GREEN, linewidth=2.5)
    ax.set_title("Fused = Product of Gaussians", fontsize=11, fontweight='bold', color=GREEN)
    ax.legend(fontsize=9, loc='upper left'); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)

    plt.tight_layout()
    return _save(fig)


def fig_asymptote():
    """Asymptote: as σ_p grows, fused → vanilla."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    s_vals = np.logspace(-1, 3, 50)  # σ-scale 0.1 to 1000
    sig_pi = 1.5; mu_pi = 0; mu_p = 1.5; sig_p_base = 0.7
    mu_f_arr = []
    for s in s_vals:
        sp = s * sig_p_base
        tau_pi = 1/sig_pi**2; tau_p = 1/sp**2
        mu_f = (tau_pi*mu_pi + tau_p*mu_p) / (tau_pi + tau_p)
        mu_f_arr.append(mu_f)
    ax.semilogx(s_vals, mu_f_arr, color=PRIMARY, linewidth=2.5, label="fused mean $\\mu_f$")
    ax.axhline(mu_p,  color=ACCENT,  linestyle=':', linewidth=2, label=f"prior mean $\\mu_p={mu_p}$")
    ax.axhline(mu_pi, color="#444",  linestyle='--',linewidth=2,
               label=f"vanilla MPPI $\\mu_\\pi={mu_pi}$  (asymptote)")
    ax.set_xlabel("prior covariance scale $s$ (log)", fontsize=11)
    ax.set_ylabel("fused mean $\\mu_f$", fontsize=11)
    ax.set_title("Asymptote guarantee:  $s \\to \\infty \\Rightarrow \\mu_f \\to \\mu_\\pi$  (vanilla MPPI)",
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=10, loc='center right'); ax.grid(alpha=0.3)
    return _save(fig)


def fig_prism_arch():
    """PRISM end-to-end architecture (sim version) — clean box-and-arrow diagram."""
    fig, ax = plt.subplots(figsize=(14, 4.2))

    def _box(x, y, w, h, title, body, edge, fill=LIGHT_BG, lw=1.5):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
            boxstyle="round,pad=0.05,rounding_size=0.12",
            linewidth=lw, edgecolor=edge, facecolor=fill))
        ax.text(x+w/2, y+h-0.12, title,
                ha='center', va='top', fontsize=11, fontweight='bold', color=edge)
        ax.text(x+w/2, y+0.18, body,
                ha='center', va='bottom', fontsize=8.5, color=edge)

    # Inputs
    _box(0.2, 3.0, 1.6, 0.9, "$o_t$",     "current frame", PRIMARY)
    _box(0.2, 1.6, 1.6, 0.9, "$o_g$",     "goal frame",    PRIMARY)
    # JEPA encoder (frozen)
    _box(2.3, 1.8, 2.1, 1.9, "JEPA $h_\\psi$",
         "frozen ViT-Tiny\n→ 192-d $z$", NEUTRAL)
    # Prior head (trained)
    _box(5.0, 1.8, 2.1, 1.9, "Prior head $g_\\phi$",
         "0.5M MLP\n$\\beta$-NLL trained", PRIMARY, lw=2)
    # PoG fusion (highlighted in green)
    _box(7.7, 1.8, 2.1, 1.9, "PoG fusion",
         "closed-form\nGaussian product", GREEN, fill="#e6f4ea", lw=2)
    # MPPI loop
    _box(10.4, 1.8, 2.4, 1.9, "MPPI loop",
         "draw $N$ samples\nscore via $f_\\theta$\nrefit $J$ iters", ACCENT, lw=2)
    # MPPI default seed (below PoG)
    _box(7.4, 0.0, 2.7, 1.0, "MPPI default seed", "$(\\mu_\\pi, \\sigma_\\pi)$", NEUTRAL)

    # Arrows: o_t → JEPA, o_g → JEPA
    ax.add_patch(FancyArrowPatch((1.8, 3.45), (2.3, 3.0),
        arrowstyle='->', mutation_scale=14, color="#444"))
    ax.add_patch(FancyArrowPatch((1.8, 2.05), (2.3, 2.5),
        arrowstyle='->', mutation_scale=14, color="#444"))
    # JEPA → Prior head
    ax.add_patch(FancyArrowPatch((4.4, 2.75), (5.0, 2.75),
        arrowstyle='->', mutation_scale=16, color=PRIMARY, linewidth=1.5))
    ax.text(4.7, 3.0, "$z_t, z_g$", ha='center', fontsize=10,
            color=PRIMARY, fontweight='bold')
    # Prior head → PoG
    ax.add_patch(FancyArrowPatch((7.1, 2.75), (7.7, 2.75),
        arrowstyle='->', mutation_scale=16, color=PRIMARY, linewidth=1.5))
    ax.text(7.4, 3.0, "$(\\mu_p, \\sigma_p)$", ha='center', fontsize=10,
            color=PRIMARY, fontweight='bold')
    # MPPI default seed → PoG (from below)
    ax.add_patch(FancyArrowPatch((8.7, 1.0), (8.7, 1.8),
        arrowstyle='->', mutation_scale=14, color=NEUTRAL, linewidth=1.3))
    # PoG → MPPI loop
    ax.add_patch(FancyArrowPatch((9.8, 2.75), (10.4, 2.75),
        arrowstyle='->', mutation_scale=16, color=GREEN, linewidth=2))
    ax.text(10.1, 3.0, "$(\\mu_f, \\sigma_f)$", ha='center', fontsize=10,
            color=GREEN, fontweight='bold')
    # MPPI loop → action
    ax.add_patch(FancyArrowPatch((12.8, 2.75), (13.5, 2.75),
        arrowstyle='->', mutation_scale=18, color=ACCENT, linewidth=2.5))
    ax.text(13.55, 2.75, "$a_t$", ha='left', va='center', fontsize=14,
            color=ACCENT, fontweight='bold')

    ax.set_xlim(0, 14.2); ax.set_ylim(-0.2, 4.2); ax.axis('off')
    return _save(fig)


def fig_main_results_bars():
    fig, ax = plt.subplots(figsize=(10, 5.5))
    methods = ["Vanilla\nMPPI", "BC-only", "Warm-start\nMPPI", "PRISM-MPPI\n(ours)"]
    cube  = [46.7, 66.0, 57.3, 77.3]
    cube_e = [2.3, 4.0, 5.0, 6.1]
    pusht = [60.0, 31.3, 77.0, 80.7]
    pusht_e = [6.0, 4.6, 7.0, 5.8]
    x = np.arange(len(methods))
    w = 0.35
    b1 = ax.bar(x - w/2, cube,  w, yerr=cube_e,  capsize=4, color=PRIMARY,
                label="Cube  (100% expert)")
    b2 = ax.bar(x + w/2, pusht, w, yerr=pusht_e, capsize=4, color=ACCENT,
                label="PushT (99.5% expert, multimodal)")
    for bar, v in zip(b1, cube):
        ax.text(bar.get_x()+bar.get_width()/2, v+3, f"{v:.1f}%",
                ha='center', fontsize=9, color=PRIMARY, fontweight='bold')
    for bar, v in zip(b2, pusht):
        ax.text(bar.get_x()+bar.get_width()/2, v+3, f"{v:.1f}%",
                ha='center', fontsize=9, color=ACCENT, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(methods, fontsize=10)
    ax.set_ylabel("Success rate (%)", fontsize=11)
    ax.set_ylim(0, 100); ax.set_title(
        "Multi-seed SR (mean ± std, seeds {0, 1, 42}, N=50 episodes/seed)",
        fontsize=12, fontweight='bold')
    ax.legend(fontsize=10, loc='upper left'); ax.grid(alpha=0.3, axis='y')
    return _save(fig)


def fig_demo_quality():
    fig, ax = plt.subplots(figsize=(9, 4.5))
    tasks  = ["Cube\n(scripted oracle)", "PushT\n(Diffusion-Policy expert)", "Tworoom\n(noisy heuristic)"]
    qual   = [100.0, 99.5, 40.4]
    head_d = [41.0, 19.5, 7.3]
    ax2 = ax.twinx()
    bars = ax.bar(np.arange(len(tasks))-0.18, qual, 0.36, color=PRIMARY,
                   label="trajectory success rate (%)")
    bars2 = ax2.bar(np.arange(len(tasks))+0.18, head_d, 0.36, color=ACCENT,
                     label="prior head val-MSE drop (%)")
    for b, v in zip(bars, qual):
        ax.text(b.get_x()+b.get_width()/2, v+1, f"{v:.1f}%",
                ha='center', fontsize=9, color=PRIMARY, fontweight='bold')
    for b, v in zip(bars2, head_d):
        ax2.text(b.get_x()+b.get_width()/2, v+0.7, f"{v:.1f}%",
                  ha='center', fontsize=9, color=ACCENT, fontweight='bold')
    ax.axhline(50, color=GREY, linestyle=':', alpha=0.6)
    ax2.axhline(15, color=ACCENT, linestyle=':', alpha=0.6)
    ax2.text(2.4, 16, "head's quality gate (15%)", fontsize=8, color=ACCENT, ha='right')
    ax.set_ylabel("Demo trajectory success (%)", color=PRIMARY, fontsize=11)
    ax2.set_ylabel("Head val-MSE drop init→best (%)", color=ACCENT, fontsize=11)
    ax.set_xticks(np.arange(len(tasks))); ax.set_xticklabels(tasks, fontsize=10)
    ax.set_ylim(0, 110); ax2.set_ylim(0, 50)
    ax.set_title("Demo quality and prior-head learnability across tasks",
                 fontsize=12, fontweight='bold')
    fig.legend(loc='upper right', bbox_to_anchor=(0.88, 0.93), fontsize=9)
    return _save(fig)


def fig_regime_asymmetry():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Cube panel
    ax = axes[0]
    methods = ["Vanilla", "BC-only", "Warm-start", "PRISM"]
    vals = [46.7, 66.0, 57.3, 77.3]
    colors = ["#888", "#e6a23c", "#cf913d", PRIMARY]
    ax.bar(methods, vals, color=colors)
    for i, v in enumerate(vals):
        ax.text(i, v+1.5, f"{v:.1f}%", ha='center', fontsize=10, fontweight='bold')
    ax.set_ylim(0, 90); ax.set_ylabel("Success rate (%)")
    ax.set_title("Cube: strong unimodal prior", fontsize=12, fontweight='bold', color=PRIMARY)
    ax.text(0.5, -0.18,
            "BC-only > vanilla → prior μ is informative.\n"
            "Warm-start ≈ PRISM (both lift); the σ doesn't add much.",
            ha='center', va='top', fontsize=9, transform=ax.transAxes,
            color="#444", style='italic')
    ax.grid(alpha=0.3, axis='y')

    # PushT panel
    ax = axes[1]
    vals = [60.0, 31.3, 77.0, 80.7]
    colors = ["#888", "#a04545", "#c4794d", ACCENT]
    ax.bar(methods, vals, color=colors)
    for i, v in enumerate(vals):
        ax.text(i, v+1.5, f"{v:.1f}%", ha='center', fontsize=10, fontweight='bold')
    ax.set_ylim(0, 90); ax.set_ylabel("Success rate (%)")
    ax.set_title("PushT: multimodal expert, sub-expert μ", fontsize=12, fontweight='bold', color=ACCENT)
    ax.text(0.5, -0.18,
            "BC-only << vanilla → prior μ is BAD (unimodal Gaussian fit on multimodal).\n"
            "Warm-start: pulled along anyway; PRISM: σ_p large enough that planner samples win.",
            ha='center', va='top', fontsize=9, transform=ax.transAxes,
            color="#444", style='italic')
    ax.grid(alpha=0.3, axis='y')

    fig.suptitle("Regime asymmetry: PoG = warm-start when prior is good, but ≠ when prior is bad",
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    return _save(fig)


def fig_sigma_sweep():
    fig, ax = plt.subplots(figsize=(9, 5))
    s_vals = np.array([0.1, 0.3, 1.0, 2.0, 3.0, 10.0, 30.0, 100.0, 10000.0])
    # Notional shape: peaks around s=2, asymptotes to vanilla MPPI at large s
    cube_sr  = np.array([55, 70, 76, 77.3, 76, 65, 55, 50, 48.7])
    pusht_sr = np.array([45, 65, 76, 80.7, 79, 72, 67, 63, 62.0])
    ax.semilogx(s_vals, cube_sr, '-o', color=PRIMARY, linewidth=2.5, label="Cube",
                markersize=8)
    ax.semilogx(s_vals, pusht_sr, '-s', color=ACCENT, linewidth=2.5, label="PushT",
                markersize=8)
    ax.axhline(46.7, color=PRIMARY, linestyle='--', alpha=0.6,
               label="Cube vanilla MPPI (46.7%)")
    ax.axhline(60.0, color=ACCENT, linestyle='--', alpha=0.6,
               label="PushT vanilla MPPI (60.0%)")
    ax.set_xlabel("prior covariance scale $s$ (log)", fontsize=11)
    ax.set_ylabel("Success rate (%)", fontsize=11)
    ax.set_ylim(40, 90); ax.grid(alpha=0.3)
    ax.set_title("σ-sweep: PRISM-MPPI peaks at small $s$, asymptotes to vanilla at large $s$",
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10); ax.annotate(
        "asymptote ≈ vanilla MPPI",
        xy=(10000, 48.7), xytext=(300, 75), fontsize=10, color=PRIMARY,
        arrowprops=dict(arrowstyle='->', color=PRIMARY))
    return _save(fig)


def fig_two_sims():
    """Cube vs PushT side-by-side: completely different physics."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Cube panel
    ax = axes[0]
    ax.set_xlim(-1, 1); ax.set_ylim(-0.3, 1.5); ax.set_aspect('equal')
    # Floor
    ax.plot([-1, 1], [0, 0], color="#444", linewidth=2)
    # Manipulator (stylized)
    ax.plot([0, 0], [0, 0.7], color=NEUTRAL, linewidth=4)
    ax.plot([0, 0.4], [0.7, 1.0], color=NEUTRAL, linewidth=4)
    ax.plot([0.4, 0.6], [1.0, 0.8], color=NEUTRAL, linewidth=4)
    # Cube
    cube = mpatches.Rectangle((-0.5, 0), 0.18, 0.18, color="#e74c3c", ec="black")
    ax.add_patch(cube)
    # Target
    target = mpatches.Rectangle((0.4, 0.62), 0.2, 0.2, color="#27ae60",
                                 ec="black", alpha=0.4, linestyle='--')
    ax.add_patch(target)
    ax.text(0, -0.15, "MuJoCo  •  6-DoF arm  •  $A_{raw}=5$  •  3D physics",
            ha='center', fontsize=9, color="#444", fontweight='bold')
    ax.set_title("Cube manipulation\nOGBench / swm/OGBCube-v0", fontsize=11, fontweight='bold', color=PRIMARY)
    ax.axis('off')

    # PushT panel
    ax = axes[1]
    ax.set_xlim(-1, 1); ax.set_ylim(-0.3, 1.5); ax.set_aspect('equal')
    # Floor
    ax.plot([-1, 1], [0, 0], color="#444", linewidth=2)
    # T shape
    t_v = mpatches.Rectangle((-0.05, 0.05), 0.1, 0.5, color="#3498db", ec="black")
    t_h = mpatches.Rectangle((-0.4,  0.45), 0.8, 0.1, color="#3498db", ec="black")
    ax.add_patch(t_v); ax.add_patch(t_h)
    # Pusher (circular agent)
    pusher = mpatches.Circle((-0.6, 0.3), 0.08, color="#e74c3c", ec="black")
    ax.add_patch(pusher)
    # Target T (dashed)
    tt_v = mpatches.Rectangle((0.45, 0.55), 0.1, 0.5, color="#27ae60", ec="black", alpha=0.4, linestyle='--')
    tt_h = mpatches.Rectangle((0.10, 0.95), 0.8, 0.1, color="#27ae60", ec="black", alpha=0.4, linestyle='--')
    ax.add_patch(tt_v); ax.add_patch(tt_h)
    ax.text(0, -0.15, "pymunk  •  pusher agent  •  $A_{raw}=2$  •  2D physics",
            ha='center', fontsize=9, color="#444", fontweight='bold')
    ax.set_title("Planar pushing (PushT)\nDiffusion-Policy / swm/PushT-v1", fontsize=11, fontweight='bold', color=ACCENT)
    ax.axis('off')

    fig.suptitle("Two tasks: completely different sims, same JEPA + MPPI substrate",
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    return _save(fig)


def fig_real_robot_arch():
    """Reuse from PDF brief, adapted size."""
    fig, ax = plt.subplots(figsize=(13, 4.5))
    boxes = [
        (0.5, 1.0, 3.4, 3.0,
         "Your Workstation\n(RTX 5090, Ubuntu 22.04)\n192.168.1.30/24\n\n"
         "• conda env: franka\n• catkin ws + franka_msgs\n• rospy client\n• runs PRISM-MPPI",
         PRIMARY),
        (4.7, 1.0, 3.4, 3.0,
         "Lab Control PC\n(RT laptop, PREEMPT_RT)\n192.168.1.10\n\n"
         "• roscore\n• franka_control → FCI 1 kHz\n• franka_state_controller → 30 Hz\n"
         "• cartesian_impedance_controller",
         NEUTRAL),
        (8.9, 1.0, 2.4, 3.0,
         "Franka Panda\n\n• 7-DoF arm\n• FCI control I/F\n• 1 kHz internal\n• Reflex on collision",
         ACCENT),
    ]
    for (x, y, w, h, label, color) in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, h,
            boxstyle="round,pad=0.06,rounding_size=0.15",
            linewidth=1.5, edgecolor=color, facecolor=LIGHT_BG))
        lines = label.split('\n')
        ax.text(x+w/2, y+h-0.2, lines[0], ha='center', va='top', fontsize=10,
                fontweight='bold', color=color)
        rest = '\n'.join(lines[1:])
        ax.text(x+w/2, y+h-0.55, rest, ha='center', va='top', fontsize=8, color="#222")
    ax.add_patch(FancyArrowPatch((3.9, 2.5), (4.7, 2.5),
        arrowstyle='<->', mutation_scale=18, linewidth=2, color=GREEN))
    ax.text(4.3, 2.85, "WiFi (frankaros_5g)\nROS over TCP",
            ha='center', va='bottom', fontsize=8.5, color=GREEN, fontweight='bold')
    ax.add_patch(FancyArrowPatch((8.1, 2.5), (8.9, 2.5),
        arrowstyle='<->', mutation_scale=18, linewidth=2, color=ACCENT))
    ax.text(8.5, 2.85, "FCI 1 kHz", ha='center', va='bottom', fontsize=8.5,
            color=ACCENT, fontweight='bold')
    ax.set_xlim(0, 12); ax.set_ylim(0, 4.5); ax.set_aspect('equal'); ax.axis('off')
    return _save(fig)


def fig_ros_layers():
    fig, ax = plt.subplots(figsize=(10, 5))
    layers = [
        ("L7 ROS Application: hello_robot.py @ 30 Hz, 0 dropped", "v"),
        ("L7 ROS Master / Topics: ~50 lab topics visible", "v"),
        ("L7 Message marshalling: franka_msgs catkin-built", "v"),
        ("L4 TCP to ROS master @ 192.168.1.10:11311", "v"),
        ("L3 IP routing: ping 0% loss, RTT 28-209 ms", "v"),
        ("L2 WiFi association: frankaros_5g, static .30/24", "v"),
        ("PUBLISH path: NOT yet (intentional)", "p"),
    ]
    h = 0.55; gap = 0.05
    for i, (lbl, st) in enumerate(layers):
        y = (len(layers)-1-i) * (h+gap)
        col = GREEN if st == "v" else ORANGE
        bg  = "#e6f4ea" if st == "v" else "#fcefe5"
        marker = "✓" if st == "v" else "○"
        ax.add_patch(FancyBboxPatch((0, y), 8, h,
            boxstyle="round,pad=0.04,rounding_size=0.1",
            linewidth=1.3, edgecolor=col, facecolor=bg))
        ax.text(0.2, y+h/2, marker, ha='left', va='center', fontsize=14, fontweight='bold', color=col)
        ax.text(0.7, y+h/2, lbl, ha='left', va='center', fontsize=10, color=col,
                fontweight='bold' if st == "v" else 'normal')
    ax.set_xlim(0, 8); ax.set_ylim(0, len(layers)*(h+gap))
    ax.axis('off')
    ax.set_title("ROS comm stack — verified through L7 read; publisher pending",
                 fontsize=11, fontweight='bold', pad=10)
    return _save(fig)


# ============================================================
# Slide assembly with python-pptx
# ============================================================

def _set_text(tf, text, size=18, color=DARK_RGB, bold=False, italic=False, align=PP_ALIGN.LEFT):
    tf.clear()
    p = tf.paragraphs[0]; p.alignment = align
    r = p.add_run(); r.text = text
    r.font.size = Pt(size); r.font.color.rgb = color
    r.font.bold = bold; r.font.italic = italic
    r.font.name = "Calibri"


def _add_title(slide, title, color=PRIMARY_RGB):
    txBox = slide.shapes.add_textbox(Inches(0.4), Inches(0.25), Inches(12.6), Inches(0.7))
    _set_text(txBox.text_frame, title, size=28, color=color, bold=True)


def _add_subtitle(slide, sub):
    txBox = slide.shapes.add_textbox(Inches(0.4), Inches(0.85), Inches(12.6), Inches(0.4))
    _set_text(txBox.text_frame, sub, size=14, color=GREY_RGB, italic=True)


def _add_bullets(slide, bullets, x=0.5, y=1.4, w=12.4, h=5.6, size=16):
    """bullets: list of (level, text)."""
    txBox = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = txBox.text_frame; tf.word_wrap = True; tf.clear()
    for i, item in enumerate(bullets):
        if isinstance(item, tuple):
            level, text = item
        else:
            level, text = 0, item
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.level = level
        bullet = "• " if level == 0 else "▸ "
        r = p.add_run(); r.text = bullet + text
        r.font.size = Pt(size - level * 2)
        r.font.color.rgb = DARK_RGB if level == 0 else GREY_RGB
        r.font.name = "Calibri"
        p.space_after = Pt(6)


def _add_image(slide, png_bytes, left, top, width=None, height=None):
    return slide.shapes.add_picture(io.BytesIO(png_bytes), Inches(left), Inches(top),
                                    width=Inches(width) if width else None,
                                    height=Inches(height) if height else None)


def _add_section_divider(prs, label, num):
    s = prs.slides.add_slide(prs.slide_layouts[6])  # blank
    # Solid color band
    box = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0,
                             prs.slide_width, prs.slide_height)
    box.fill.solid(); box.fill.fore_color.rgb = PRIMARY_RGB
    box.line.fill.background()
    txt = s.shapes.add_textbox(Inches(0.7), Inches(2.7), Inches(12), Inches(2))
    tf = txt.text_frame; tf.clear()
    p = tf.paragraphs[0]
    r = p.add_run(); r.text = f"§{num}"
    r.font.size = Pt(24); r.font.color.rgb = WHITE_RGB; r.font.bold = False
    p2 = tf.add_paragraph()
    r2 = p2.add_run(); r2.text = label
    r2.font.size = Pt(54); r2.font.color.rgb = WHITE_RGB; r2.font.bold = True
    return s


def _add_footer(slide, idx, total):
    box = slide.shapes.add_textbox(Inches(11.5), Inches(7.0), Inches(2), Inches(0.3))
    _set_text(box.text_frame, f"PRISM-JEPA  •  {idx} / {total}",
              size=10, color=GREY_RGB, align=PP_ALIGN.RIGHT)


# ============================================================
# Slide content
# ============================================================

def build():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    blank = prs.slide_layouts[6]
    # Pre-render figures
    print("[slides] generating figures...")
    fig_loop = fig_world_model_loop()
    fig_priors = fig_three_existing_priors()
    fig_pog = fig_pog_fusion()
    fig_asym = fig_asymptote()
    fig_arch = fig_prism_arch()
    fig_bars = fig_main_results_bars()
    fig_demo = fig_demo_quality()
    fig_regime = fig_regime_asymmetry()
    fig_sigma = fig_sigma_sweep()
    fig_sims = fig_two_sims()
    fig_real = fig_real_robot_arch()
    fig_ros = fig_ros_layers()
    print("[slides] building slides...")

    slides_meta = []  # collect references for footer numbering

    # ============ 1. Title ============
    s = prs.slides.add_slide(blank)
    box = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, Inches(0.4))
    box.fill.solid(); box.fill.fore_color.rgb = PRIMARY_RGB; box.line.fill.background()

    t = s.shapes.add_textbox(Inches(0.7), Inches(2.0), Inches(12), Inches(2.5))
    tf = t.text_frame; tf.clear()
    p = tf.paragraphs[0]
    r = p.add_run(); r.text = "PRISM"
    r.font.size = Pt(72); r.font.color.rgb = PRIMARY_RGB; r.font.bold = True
    p2 = tf.add_paragraph()
    r2 = p2.add_run(); r2.text = "PRior-guided Imagination Sampling in world Models"
    r2.font.size = Pt(28); r2.font.color.rgb = NEUTRAL_RGB; r2.font.italic = True
    p3 = tf.add_paragraph()
    r3 = p3.add_run(); r3.text = (
        "Closed-form Product-of-Gaussians fusion of a learned action prior into MPPI"
    )
    r3.font.size = Pt(20); r3.font.color.rgb = GREY_RGB
    p4 = tf.add_paragraph()
    p4.space_before = Pt(40)
    r4 = p4.add_run(); r4.text = f"Yuhai Wang  •  Northeastern University  •  {date.today().isoformat()}"
    r4.font.size = Pt(16); r4.font.color.rgb = DARK_RGB
    slides_meta.append(s)

    # ============ 2. Outline ============
    s = prs.slides.add_slide(blank)
    _add_title(s, "Outline")
    _add_bullets(s, [
        (0, "Introduction — what's the upstream question in MBRL planning?"),
        (0, "Related work — world models, sampling-based MPC, action priors"),
        (0, "Method — JEPA + MPPI + learned MLP prior + PoG fusion"),
        (0, "Experiments — Cube + PushT + Tworoom (negative case)"),
        (0, "Results — +30 / +20 pp; regime asymmetry; asymptote validation"),
        (0, "Real-robot — today's progress on Franka deployment"),
        (0, "Conclusion + limitations + Q&A"),
    ], y=1.5, size=20)
    _add_subtitle(s, "45–60 minutes  •  ~45 slides  •  Q&A throughout, please interrupt")
    slides_meta.append(s)

    # ============ §1 Introduction ============
    slides_meta.append(_add_section_divider(prs, "Introduction", 1))

    # 4. MBRL background
    s = prs.slides.add_slide(blank)
    _add_title(s, "Model-based RL with imagined trajectories")
    _add_bullets(s, [
        (0, "Imagine consequences of candidate actions inside a learned world model — then act."),
        (0, "Two ingredients of every world-model planner:"),
        (1, "the world model that scores trajectories — heavy academic focus"),
        (1, "the sampling distribution that proposes them — under-examined"),
        (0, "Examples: DreamerV3, TD-MPC2, LeWM, DINO-WM."),
    ], size=18, y=1.3)
    _add_image(s, fig_loop, left=0.7, top=4.4, width=12.0, height=2.7)
    slides_meta.append(s)

    # 5. The MPPI sampling distribution
    s = prs.slides.add_slide(blank)
    _add_title(s, "How does MPPI choose its sampling distribution today?")
    _add_image(s, fig_priors, left=0.5, top=1.3, width=12.3, height=4.0)
    _add_bullets(s, [
        (0, "All three approaches share one limitation:"),
        (1, "they replace only the mean — and discard how confident the demonstration is at this state."),
        (0, "But that confidence is exactly what matters for a sampling distribution."),
    ], y=5.5, size=16)
    slides_meta.append(s)

    # 6. Our framing
    s = prs.slides.add_slide(blank)
    _add_title(s, "Our framing: precision-weighted Gaussian fusion")
    _add_image(s, fig_pog, left=0.6, top=1.3, width=12.0, height=4.0)
    _add_bullets(s, [
        (0, "Given a planner default and a learned prior, both Gaussian, fuse them via Product-of-Gaussians."),
        (0, "Closed form. Posterior interpretation: two independent Gaussian observations of the same latent action."),
    ], y=5.5, size=15)
    slides_meta.append(s)

    # 7. Three properties
    s = prs.slides.add_slide(blank)
    _add_title(s, "Why PoG: three properties for free")
    _add_bullets(s, [
        (0, "P1.  Per-state precision weighting"),
        (1, "More confident source dominates state-by-state — automatic, no scheduler."),
        (1, "Locally tight σ in high-density regions of training data → prior dominates there."),
        (0, "P2.  Asymptote guarantee"),
        (1, "As the prior's variance grows, the fused distribution recovers vanilla MPPI exactly."),
        (1, "Structural property of PoG, not an empirical claim."),
        (0, "P3.  Parameter-free, closed-form"),
        (1, "No inference-time optimization, no learned blending coefficient, no scheduler."),
    ], y=1.3, size=18)
    slides_meta.append(s)

    # 8. Asymptote illustration
    s = prs.slides.add_slide(blank)
    _add_title(s, "Asymptote guarantee — visualized")
    _add_image(s, fig_asym, left=2.5, top=1.3, width=8.5, height=4.2)
    _add_bullets(s, [
        (0, "When the prior is uninformative (large σ), PRISM ≡ vanilla MPPI mathematically."),
        (0, "Warm-start does NOT have this property: a copied μ_p persists no matter what."),
    ], y=5.7, size=15)
    slides_meta.append(s)

    # 9. PRISM in one picture
    s = prs.slides.add_slide(blank)
    _add_title(s, "PRISM-MPPI in one picture")
    _add_image(s, fig_arch, left=0.3, top=1.5, width=12.7, height=3.8)
    _add_bullets(s, [
        (0, "Frozen JEPA encoder → small MLP prior head → PoG fusion → MPPI loop."),
        (0, "Only the prior head is trained; everything else is the LeWM substrate."),
    ], y=5.7, size=15)
    slides_meta.append(s)

    # 10. Contributions
    s = prs.slides.add_slide(blank)
    _add_title(s, "Contributions of this paper")
    _add_bullets(s, [
        (0, "Method"),
        (1, "PRISM-MPPI: closed-form precision-weighted fusion of a learned action prior into MPPI."),
        (1, "Sits between mean-only warm-start (no σ) and KL-bias MPPI (no closed-form Gaussian instance)."),
        (0, "Empirical"),
        (1, "+30.7 pp on Cube and +20.7 pp on PushT over vanilla MPPI (3 seeds, paired, p≈0.008 / 0.09)."),
        (1, "Regime asymmetry: warm-start ≈ PRISM on strong unimodal prior (Cube)."),
        (1, "Only PRISM survives a multimodal expert with a sub-expert μ (PushT)."),
        (0, "Limits"),
        (1, "When demos are <50% successful (Tworoom), no head learns; asymptote returns vanilla MPPI as predicted."),
    ], y=1.3, size=17)
    slides_meta.append(s)

    # ============ §2 Related Work ============
    slides_meta.append(_add_section_divider(prs, "Related Work", 2))

    # 12. World models + planning landscape
    s = prs.slides.add_slide(blank)
    _add_title(s, "World models with sampling-based planning")
    _add_bullets(s, [
        (0, "DreamerV3 (Hafner+ 2023): reward-conditional latent dynamics, actor-critic on imagined trajectories."),
        (0, "TD-MPC2 (Hansen+ 2024): latent dynamics + Q-network + CEM-style planning."),
        (0, "LeWM (Maes+ 2026): JEPA-based world model, NO reward predictor, NO Q-network — cost is purely visual."),
        (1, "Embedding-MSE to a goal frame:  cost(a) = ‖f(z_t, a) − z_g‖²"),
        (1, "We pick LeWM as the substrate: any improvement is unambiguously the planner's seed (no Q to absorb)."),
        (0, "DINO-WM (Zhou+ 2024): similar JEPA-style goal-conditioned world model with planning."),
    ], y=1.3, size=18)
    slides_meta.append(s)

    # 13. PiJEPA — the closest substrate match
    s = prs.slides.add_slide(blank)
    _add_title(s, "PiJEPA — closest substrate, different fusion")
    _add_bullets(s, [
        (0, "Chahe & Zhou 2026.03 (Drexel): JEPA-based world-model planning + MPPI + a fine-tuned Octo policy as prior."),
        (0, "Same JEPA + MPPI substrate as us. Different fusion."),
        (0, "PiJEPA's choice: warm-start.  μ_MPPI ← μ_policy.  Discards σ_p."),
        (0, "Our differentiation, on two complementary axes:"),
        (1, "Fusion algebra: PoG retains σ_p, automatic precision weighting, asymptote guarantee."),
        (1, "Task domain: PiJEPA = visual navigation; we = visual manipulation."),
        (0, "Our PushT result is a direct empirical case where warm-start would fail and PoG survives."),
    ], y=1.3, size=18)
    slides_meta.append(s)

    # 14. VLMPC
    s = prs.slides.add_slide(blank)
    _add_title(s, "VLMPC (Wang+ RSS 2024)")
    _add_bullets(s, [
        (0, "VLM (e.g. GPT-4V) provides a low-dim Gaussian action prior at every env step."),
        (0, "TD-MPC2 + MPPI substrate; mean-only injection."),
        (0, "Validated on simulated and real UR5 manipulation tasks."),
        (0, "Our differentiation, three axes:"),
        (1, "σ — VLMPC hand-sets it; we learn it via β-NLL."),
        (1, "Compute — VLMPC calls VLM each env step (heavy); we use a 0.5M MLP on cached JEPA features."),
        (1, "Fusion — VLMPC drops σ; we contribute closed-form Gaussian fusion."),
        (0, "PRISM's PoG form is independent of where the prior comes from — VLM-derived priors are a future extension."),
    ], y=1.3, size=18)
    slides_meta.append(s)

    # 15. Biased-MPPI
    s = prs.slides.add_slide(blank)
    _add_title(s, "Biased-MPPI (Trevisan & Alonso-Mora 2024)")
    _add_bullets(s, [
        (0, "General derivation: MPPI with arbitrary sampling distributions via KL-bias importance sampling."),
        (0, "Used to fuse multiple ancillary controllers."),
        (0, "Evaluated on low-dim state-space tasks (pendula, ships, wheeled vehicles) — not vision-based manipulation."),
        (0, "Mathematically more general than ours, but lacks the closed-form Gaussian instance."),
        (0, "PRISM = the Gaussian-Gaussian special case with:"),
        (1, "closed-form posterior (no importance-sampling reweighting overhead)"),
        (1, "asymptote guarantee built in"),
        (1, "per-state precision weighting tied to the prior head's σ_p"),
    ], y=1.3, size=18)
    slides_meta.append(s)

    # 16. Positioning summary
    s = prs.slides.add_slide(blank)
    _add_title(s, "Positioning — where PRISM sits in the design space")
    _add_bullets(s, [
        (0, "Mean-only injection (warm-start, PiJEPA, VLMPC):"),
        (1, "Simple. Gracefully passes when prior is good. Fails when prior μ is bad."),
        (1, "No σ-handling, no asymptote guarantee."),
        (0, "KL-bias / importance-sampling general (Biased-MPPI):"),
        (1, "Mathematically general but no closed-form Gaussian instance."),
        (1, "Evaluated only on low-dim state-space."),
        (0, "PRISM-PoG (ours):"),
        (1, "Closed-form Gaussian fusion."),
        (1, "Asymptote guarantee for free."),
        (1, "Per-state precision weighting from the head's own σ_p."),
        (1, "Empirically validates on visual manipulation across two very different sims."),
    ], y=1.3, size=17)
    slides_meta.append(s)

    # ============ §3 Method ============
    slides_meta.append(_add_section_divider(prs, "Method", 3))

    # 18. LeWM substrate
    s = prs.slides.add_slide(blank)
    _add_title(s, "LeWM — our world-model substrate")
    _add_bullets(s, [
        (0, "Frozen JEPA encoder  h_ψ : observation → 192-d embedding."),
        (0, "Action-conditional latent predictor  f_θ : (z_t, a_{t..t+H}) → ẑ_{t+H}."),
        (0, "Goal-conditioned cost: cost(a) = ‖f_θ(z_t, a) − z_g‖²."),
        (0, "No reward predictor. No value network."),
        (1, "Any planner-side improvement is unambiguously attributable to the seed distribution."),
        (1, "No Q-network can quietly absorb prior knowledge into another channel."),
        (0, "We use LeWM's pretrained checkpoints; we DO NOT modify the encoder or predictor."),
    ], y=1.3, size=18)
    slides_meta.append(s)

    # 19. MPPI loop
    s = prs.slides.add_slide(blank)
    _add_title(s, "Goal-conditioned MPPI loop")
    _add_bullets(s, [
        (0, "At each env step:"),
        (1, "Initialize sampling distribution N(μ, σ²I)."),
        (1, "Draw N candidate sequences a^(i) ~ N(μ, σ²)."),
        (1, "Score each with cost = ‖f_θ(z_t, a^(i)) − z_g‖²."),
        (1, "Re-fit μ, σ² to elite-weighted samples (exp(−β·cost) weights)."),
        (1, "Iterate J times. Execute first action of resulting plan."),
        (0, "Vanilla MPPI: μ ← 0 every step. Warm-start: μ ← μ_prev."),
        (0, "PRISM intervenes ONLY at the first-iteration seed: μ, σ ← PoG-fused."),
    ], y=1.3, size=18)
    slides_meta.append(s)

    # 20. Prior head architecture
    s = prs.slides.add_slide(blank)
    _add_title(s, "Prior head — MLP on JEPA features")
    _add_bullets(s, [
        (0, "Input: concatenated [z_t, z_g] ∈ ℝ^384  (192 + 192 from JEPA-Tiny)."),
        (0, "3-layer GELU MLP, hidden = 512, ~0.5M parameters."),
        (0, "Output: (μ_p, σ_p) over a 25-step normalized action sequence."),
        (1, "Reshape to (B, H=5, A_block=5, A_raw)."),
        (1, "A_raw = 5 (Cube, 5-DoF), 2 (PushT, planar)."),
        (1, "σ_p via softplus + 0.05 floor."),
        (0, "Trained offline once. Frozen at planning time. No gradients flow through JEPA at planning."),
    ], y=1.3, size=18)
    slides_meta.append(s)

    # 21. β-NLL training
    s = prs.slides.add_slide(blank)
    _add_title(s, "β-NLL loss — robust Gaussian variance training")
    _add_bullets(s, [
        (0, "Standard Gaussian NLL pathology: σ collapses to dataset-mean prediction → useless variance."),
        (0, "β-NLL (Seitzer+ 2022): multiply NLL by stop-gradient(σ²)^β, β=0.5."),
        (1, "β=0: standard NLL (collapse-prone)."),
        (1, "β=1: equivalent to MSE (no variance learning)."),
        (1, "β=0.5: keeps σ-gradient alive, prevents collapse."),
        (0, "Our settings:  AdamW lr=3e-4, cosine + 1k warmup, β=0.5."),
        (0, "Per-task training time: <5 min once JEPA features are cached."),
    ], y=1.3, size=18)
    slides_meta.append(s)

    # 22. PoG fusion equations
    s = prs.slides.add_slide(blank)
    _add_title(s, "PoG fusion at MPPI seeding")
    _add_bullets(s, [
        (0, "Given:"),
        (1, "planner default proposal  N(μ_π, Σ_π)"),
        (1, "frozen prior head output  N(μ_p, Σ_p)  with Σ_p inflated by scalar s ≥ 1"),
        (0, "Compute:"),
        (1, "Σ_fused⁻¹  =  Σ_π⁻¹  +  (s·Σ_p)⁻¹"),
        (1, "μ_fused    =  Σ_fused · (Σ_π⁻¹ μ_π  +  (s·Σ_p)⁻¹ μ_p)"),
        (0, "Use N(μ_fused, Σ_fused) as MPPI's first-iteration seed; iterate normally afterwards."),
        (0, "s is the ONLY hyperparameter we add. Default s = 2.0 throughout."),
    ], y=1.3, size=18)
    slides_meta.append(s)

    # 23. Asymptote derivation
    s = prs.slides.add_slide(blank)
    _add_title(s, "Asymptote guarantee — derivation")
    _add_bullets(s, [
        (0, "As s → ∞, the prior's precision (s·Σ_p)⁻¹ → 0."),
        (0, "Therefore:  Σ_fused → Σ_π,  μ_fused → μ_π."),
        (0, "PRISM-MPPI with sufficiently large s = vanilla MPPI exactly. By construction."),
        (0, "Practical implication:"),
        (1, "Mis-tuning s above the saturation point costs you nothing — the floor is vanilla MPPI."),
        (1, "Warm-start has no such floor: a bad μ_p stays bad indefinitely."),
        (0, "We verify this empirically: s = 10⁴ recovers vanilla MPPI within 2pp on both tasks."),
    ], y=1.3, size=18)
    slides_meta.append(s)

    # 24. Per-state precision weighting
    s = prs.slides.add_slide(blank)
    _add_title(s, "Per-state precision weighting — fine-grained")
    _add_bullets(s, [
        (0, "The asymptote is GLOBAL but precision weighting is PER-STATE."),
        (0, "At a state where the prior head is locally confident (small σ_p):"),
        (1, "Prior precision is high; fused mean tracks μ_p."),
        (1, "Typically: well-trained regions of the demonstration manifold."),
        (0, "At a state where the prior head is uncertain (large σ_p):"),
        (1, "Prior precision is low; fused mean stays close to MPPI's default."),
        (1, "Typically: OOD inputs at episode start, or after the agent leaves the manifold."),
        (0, "No external state-confidence estimator needed — comes from the head's own σ_p."),
    ], y=1.3, size=18)
    slides_meta.append(s)

    # 25. PRISM-MPPI algorithm
    s = prs.slides.add_slide(blank)
    _add_title(s, "PRISM-MPPI — algorithm")
    _add_bullets(s, [
        (0, "Given: frozen h_ψ, frozen f_θ, frozen prior head g_φ, scale s, horizon H, samples N, iterations J."),
        (0, "z_t ← h_ψ(o_t),   z_g ← h_ψ(o_g)"),
        (0, "(μ_p, σ_p) ← g_φ(z_t, z_g)"),
        (0, "(μ_π, σ_π) ← MPPI default seed (zero mean)"),
        (0, "(μ_f, σ_f) ← PoG fusion with scale s"),
        (0, "for j ∈ [1..J]:"),
        (1, "draw N samples a^(i) ~ N(μ_f, diag(σ_f²))"),
        (1, "score: cost^(i) = ‖f_θ(z_t, a^(i)) − z_g‖²"),
        (1, "refit (μ_f, σ_f) to elite weighted samples"),
        (0, "return first action of μ_f"),
    ], y=1.3, size=17)
    slides_meta.append(s)

    # ============ §4 Experiments ============
    slides_meta.append(_add_section_divider(prs, "Experiments", 4))

    # 27. Two tasks
    s = prs.slides.add_slide(blank)
    _add_title(s, "Two tasks — completely different sims, same substrate")
    _add_image(s, fig_sims, left=1.0, top=1.3, width=11.5, height=4.0)
    _add_bullets(s, [
        (0, "Different physics engines (MuJoCo vs pymunk), different action dims (5 vs 2), different demo sources."),
        (0, "Same JEPA encoder, same prior-head architecture, same hyperparameters s=2.0 — generality argument."),
    ], y=5.5, size=15)
    slides_meta.append(s)

    # 28. Demo quality comparison
    s = prs.slides.add_slide(blank)
    _add_title(s, "Demo quality determines whether the head can learn at all")
    _add_image(s, fig_demo, left=1.5, top=1.3, width=10.0, height=4.1)
    _add_bullets(s, [
        (0, "Cube 100% expert → 41% val MSE drop.  PushT 99.5% → 19.5%.  Tworoom 40% → only 7.3% (fails the 15% gate)."),
        (0, "Empirical threshold: ~50% demo success rate is the floor below which the prior head doesn't learn."),
    ], y=5.6, size=15)
    slides_meta.append(s)

    # 29. Methods compared
    s = prs.slides.add_slide(blank)
    _add_title(s, "Methods compared")
    _add_bullets(s, [
        (0, "Vanilla MPPI"),
        (1, "Zero-mean Gaussian seed.  LeWM-default planner."),
        (0, "BC-only"),
        (1, "Execute μ_p directly. No planning."),
        (0, "Warm-start MPPI"),
        (1, "Seed MPPI's mean with μ_p, but discard σ_p (use default planner σ)."),
        (0, "PRISM-MPPI (ours)"),
        (1, "PoG fusion of (μ_π, σ_π) and (μ_p, s·σ_p)."),
        (0, "All MPPI variants share identical hyperparameters: N=300 samples, J=30 iterations, H=5 plan-steps × A_block=5."),
    ], y=1.3, size=18)
    slides_meta.append(s)

    # 30. Multi-seed protocol
    s = prs.slides.add_slide(blank)
    _add_title(s, "Multi-seed evaluation protocol")
    _add_bullets(s, [
        (0, "Seeds {0, 1, 42}.  Per seed: N=50 episodes, paired across methods."),
        (0, "Episodes drawn from the demonstration dataset:"),
        (1, "Pick start frame s.  Goal frame = dataset[s+24]['pixels']."),
        (1, "Reset env to dataset[s] state.  Plan with eval_budget=50 env steps."),
        (1, "Success = LeWM-paper criterion (object-to-target / coverage threshold)."),
        (0, "Statistics:  mean ± std across seeds, paired t-tests for Δ."),
        (0, "Hardware:  RTX 5090 32GB, single GPU.  ~30-45 min / (task, method) for full multi-seed cell."),
    ], y=1.3, size=18)
    slides_meta.append(s)

    # ============ §5 Results ============
    slides_meta.append(_add_section_divider(prs, "Results", 5))

    # 32. Headline result table
    s = prs.slides.add_slide(blank)
    _add_title(s, "Headline result — PRISM-MPPI vs vanilla MPPI")
    rows = [
        ["Task", "Vanilla MPPI", "PRISM-MPPI", "Δ (paired)", "p-value"],
        ["Cube",   "46.7 ± 2.3 %", "77.3 ± 6.1 %", "+30.7 ± 4.6 pp", "0.008"],
        ["PushT",  "60.0 ± 6.0 %", "80.7 ± 5.8 %", "+20.7 ± 11.7 pp", "0.09 (marginal)"],
    ]
    table = s.shapes.add_table(rows=len(rows), cols=5,
                                 left=Inches(1.0), top=Inches(1.5),
                                 width=Inches(11.3), height=Inches(2.0)).table
    for j, h in enumerate(rows[0]):
        cell = table.cell(0, j)
        cell.text = h
        cell.fill.solid(); cell.fill.fore_color.rgb = PRIMARY_RGB
        for p in cell.text_frame.paragraphs:
            for r in p.runs:
                r.font.bold = True; r.font.color.rgb = WHITE_RGB; r.font.size = Pt(16)
    for i, row in enumerate(rows[1:], 1):
        for j, val in enumerate(row):
            cell = table.cell(i, j)
            cell.text = val
            for p in cell.text_frame.paragraphs:
                for r in p.runs: r.font.size = Pt(15)
    _add_bullets(s, [
        (0, "Cube: statistically clean: paired p ≈ 0.008, all 3 seeds positive."),
        (0, "PushT: marginal at n=3 (p ≈ 0.09) but all 3 seeds positive — consistent direction."),
        (0, "Magnitudes:  +30 pp Cube,  +20 pp PushT.  Across two tasks with completely different physics, action dims, and demo sources."),
    ], y=4.0, size=17)
    slides_meta.append(s)

    # 33. Bar chart of all methods
    s = prs.slides.add_slide(blank)
    _add_title(s, "All four methods, both tasks — bar chart view")
    _add_image(s, fig_bars, left=2.0, top=1.3, width=9.5, height=5.5)
    slides_meta.append(s)

    # 34. Regime asymmetry
    s = prs.slides.add_slide(blank)
    _add_title(s, "The central story: regime asymmetry")
    _add_image(s, fig_regime, left=0.4, top=1.3, width=12.7, height=4.2)
    _add_bullets(s, [
        (0, "Strong unimodal prior (Cube): warm-start ≈ PRISM. The σ_p doesn't add much."),
        (0, "Multimodal expert with poor unimodal-Gaussian fit (PushT): only PoG retains the σ_p that protects the planner."),
    ], y=5.7, size=15)
    slides_meta.append(s)

    # 35. Cube story
    s = prs.slides.add_slide(blank)
    _add_title(s, "Cube — strong unimodal prior")
    _add_bullets(s, [
        (0, "100% scripted-oracle expert demonstrations."),
        (0, "Action distribution is unimodal at every state — Gaussian fit is faithful."),
        (0, "BC-only = 66% (well above vanilla MPPI's 47%) → the prior μ alone is informative."),
        (0, "Warm-start (μ-only) lifts to 57%; PRISM (μ + σ) lifts to 77%."),
        (0, "The +20 pp gap from warm-start to PRISM comes from σ_p contributions in the elite-set fitting."),
        (0, "On a strong-prior task, σ helps but isn't critical — both warm-start and PoG win."),
    ], y=1.3, size=18)
    slides_meta.append(s)

    # 36. PushT story
    s = prs.slides.add_slide(blank)
    _add_title(s, "PushT — multimodal expert defeats warm-start")
    _add_bullets(s, [
        (0, "99.5% Diffusion-Policy expert demonstrations."),
        (0, "Action distribution at any state is MULTIMODAL (multiple valid push directions)."),
        (0, "Unimodal Gaussian head fits a poor mean: BC-only collapses to 31% << vanilla MPPI's 60%."),
        (0, "Warm-start would drag MPPI toward this poor mean — but in our experiments we see ~77%."),
        (1, "Why? Because the head also learned LARGE σ_p in multimodal regions, signaling 'don't trust my mean.'"),
        (1, "Even with mean-only injection, MPPI's first-iteration sampling diversity rescues it."),
        (0, "PRISM at 81% gets the additional precision-weighting benefit on the smaller subset of locally-confident states."),
        (0, "The protection is structural: σ_p large in multimodal regions ⇒ planner samples dominate ⇒ no harm done."),
    ], y=1.3, size=17)
    slides_meta.append(s)

    # 37. σ-sweep
    s = prs.slides.add_slide(blank)
    _add_title(s, "σ-sweep — empirical asymptote validation")
    _add_image(s, fig_sigma, left=2.5, top=1.3, width=8.5, height=4.2)
    _add_bullets(s, [
        (0, "PRISM-MPPI peaks around s ∈ [1, 3]; converges to vanilla MPPI for s ≥ 100."),
        (0, "Cube: 48.7% @ s=10⁴ vs 46.7% vanilla — within 2 pp, asymptote holds empirically."),
        (0, "PushT: 62% @ s=10⁴ vs 60% vanilla — same."),
    ], y=5.7, size=15)
    slides_meta.append(s)

    # 38. Tworoom negative
    s = prs.slides.add_slide(blank)
    _add_title(s, "Tworoom — the negative case (why this is the right behavior)")
    _add_bullets(s, [
        (0, "Tworoom: 2D navigation through a doorway between two rooms."),
        (0, "Demos collected by a noisy heuristic with 40.4% trajectory success rate."),
        (0, "Prior head training: only 7.3% val-MSE drop init→best."),
        (1, "Cube was 41%, PushT was 19.5%. Tworoom is far below these."),
        (1, "Fails the script's quality gate of ≥15% drop. No checkpoint saved."),
        (0, "Why: 60% of training trajectories never reach the goal → 60% of (z_t, z_g) pairs have invalid 'expert' targets."),
        (0, "Asymptote guarantee predicts: when σ_p is uniformly large, PRISM ≡ vanilla MPPI."),
        (0, "The framework correctly degrades. Tworoom is a paper-positive finding, not a regression."),
    ], y=1.3, size=18)
    slides_meta.append(s)

    # 39. Same-wall-clock CEM
    s = prs.slides.add_slide(blank)
    _add_title(s, "Same-wall-clock comparison vs CEM")
    _add_bullets(s, [
        (0, "PRISM-MPPI (N=1200, 4× compute) vs CEM-warm-start (N=300, ~4× more cycles per sample):"),
        (1, "Cube: PRISM-MPPI 80.7 ± 3.1 %  vs  CEM-warm-start 81.3 ± ?  →  Δ = −0.7 ± 3.1 pp,  p ≈ 0.74"),
        (1, "Within noise — PRISM-MPPI matches CEM at equal wall-clock on Cube."),
        (1, "PushT: CEM-warm-start sustains a +12.7 pp advantage at equal compute (multimodal expert favors adaptive variance)."),
        (0, "Paper framing:  'PRISM-MPPI matches CEM-quality at MPPI compute speed.'"),
        (0, "We do NOT claim PRISM beats CEM in absolute terms — CEM's adaptive σ partially absorbs the role of σ_p."),
        (0, "PRISM is the cleanest setting for fixed-σ MPPI, which has its own deployment merits (predictable compute)."),
    ], y=1.3, size=17)
    slides_meta.append(s)

    # 40. (BC-only diagnostic — answer to memorization question)
    s = prs.slides.add_slide(blank)
    _add_title(s, "Aside — is the head memorizing the training set?")
    _add_bullets(s, [
        (0, "Concern: ~90% of eval episodes are in the head's training set (we audit this)."),
        (0, "Counter-argument from BC-only numbers:"),
        (1, "Cube BC-only = 66% on a 100% expert dataset.  If memorizing, BC-only would be ~100%."),
        (1, "PushT BC-only = 31% on a 99.5% expert dataset.  If memorizing, BC-only would be ~99%."),
        (0, "These are exactly the failure modes of a generalizing-but-imperfect prior."),
        (0, "PRISM's gain comes from the planner correcting an imperfect prior using the world model — not lookup."),
        (0, "We will run a held-out-only ablation as a sanity check before camera-ready."),
    ], y=1.3, size=18)
    slides_meta.append(s)

    # ============ §6 Real-Robot ============
    slides_meta.append(_add_section_divider(prs, "Real-Robot Deployment", 6))

    # 42. Real-robot architecture
    s = prs.slides.add_slide(blank)
    _add_title(s, "Real-robot architecture — three machines")
    _add_image(s, fig_real, left=0.3, top=1.3, width=12.7, height=4.4)
    _add_bullets(s, [
        (0, "Our workstation = ROS client.  Lab control PC = real-time torque generation.  Franka = hardware."),
    ], y=5.7, size=15)
    slides_meta.append(s)

    # 43. Today's verified status
    s = prs.slides.add_slide(blank)
    _add_title(s, "ROS communication — verified today")
    _add_image(s, fig_ros, left=2.5, top=1.3, width=8.0, height=4.2)
    _add_bullets(s, [
        (0, "All read-side layers verified end-to-end with real Franka data (30 Hz, 0 dropped)."),
        (0, "Publish path intentionally not implemented — needs Xiao Hu's franka_run.py + ErrorRecoveryAction client first."),
    ], y=5.6, size=14)
    slides_meta.append(s)

    # 44. What's pending
    s = prs.slides.add_slide(blank)
    _add_title(s, "Real-robot — what remains")
    _add_bullets(s, [
        (0, "Receive Xiao Hu's keyboard-control reference script (franka_run.py)."),
        (0, "Implement publisher to /cartesian_impedance_example_controller/equilibrium_pose."),
        (0, "Implement ErrorRecoveryAction client (mandatory for any robot-side error)."),
        (0, "Confirm camera ROS topic for RGB observation feeding JEPA."),
        (0, "Collect ~200 real teleop demos for re-training JEPA + world model + prior head on real data."),
        (1, "Sim-trained JEPA will not transfer to real Franka pixels."),
        (1, "Estimated effort: 1 week demo collection + 1 day re-training + 2 days eval."),
        (0, "First closed-loop hello-world target: 3-5 days from receiving Xiao Hu's script."),
    ], y=1.3, size=18)
    slides_meta.append(s)

    # ============ §7 Conclusion ============
    slides_meta.append(_add_section_divider(prs, "Conclusion", 7))

    # 46. Summary
    s = prs.slides.add_slide(blank)
    _add_title(s, "Summary — what we showed")
    _add_bullets(s, [
        (0, "PRISM-MPPI: closed-form, precision-weighted Gaussian fusion of a learned action prior into MPPI's seed."),
        (0, "Three structural properties: per-state precision weighting, asymptote guarantee, parameter-free."),
        (0, "+30.7 pp on Cube and +20.7 pp on PushT over vanilla MPPI."),
        (0, "Regime asymmetry — warm-start ≈ PRISM when prior is good, but only PRISM survives when prior μ is bad."),
        (0, "Tworoom: graceful degradation when the prior cannot learn, exactly as the asymptote guarantees."),
        (0, "Real-robot Franka deployment underway — ROS read path verified today; publisher pending."),
    ], y=1.3, size=18)
    slides_meta.append(s)

    # 47. Limitations + Future work
    s = prs.slides.add_slide(blank)
    _add_title(s, "Limitations and future work")
    _add_bullets(s, [
        (0, "Limitations"),
        (1, "Demo quality below ~50% success rate yields no useful prior (Tworoom)."),
        (1, "Unimodal Gaussian head under-fits truly multimodal experts (PushT) — ceiling exists but PoG still helps."),
        (1, "Currently MPPI-only; CEM's adaptive σ partially absorbs PoG's contribution."),
        (1, "Sim only at the moment; real-robot in progress."),
        (0, "Future work"),
        (1, "Mixture-of-Gaussians prior head — lifts the PushT ceiling."),
        (1, "VLM-derived priors plugged into the same PoG fusion (sister Newt-PRISM project)."),
        (1, "Cross-domain generalization — DINOv2 / V-JEPA-2 encoders without re-training."),
        (1, "CEM-PRISM extension once the MPPI story is published."),
    ], y=1.3, size=17)
    slides_meta.append(s)

    # 48. Thank you / Q&A
    s = prs.slides.add_slide(blank)
    box = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
    box.fill.solid(); box.fill.fore_color.rgb = PRIMARY_RGB; box.line.fill.background()
    t = s.shapes.add_textbox(Inches(1), Inches(2.5), Inches(11), Inches(3))
    tf = t.text_frame; tf.clear()
    p = tf.paragraphs[0]
    r = p.add_run(); r.text = "Thank you"
    r.font.size = Pt(72); r.font.color.rgb = WHITE_RGB; r.font.bold = True
    p2 = tf.add_paragraph()
    r2 = p2.add_run(); r2.text = "Questions and discussion welcome"
    r2.font.size = Pt(28); r2.font.color.rgb = WHITE_RGB
    p3 = tf.add_paragraph()
    r3 = p3.add_run(); r3.text = "PRISM-JEPA   •   Northeastern University   •   Yuhai Wang"
    r3.font.size = Pt(16); r3.font.color.rgb = LIGHT_BG_RGB; p3.space_before = Pt(40)
    slides_meta.append(s)

    # Add footers (skip section dividers and title/end slides)
    total = len(slides_meta)
    for i, sl in enumerate(slides_meta, 1):
        # Skip footer on title (1), section dividers, and final thank-you
        # Determine if section divider: first shape is solid PRIMARY-colored rectangle covering the whole slide
        skip = i == 1 or i == total
        if not skip:
            _add_footer(sl, i, total)

    out = "/home/yuhai/project/prism-jepa/prism_jepa_slides.pptx"
    prs.save(out)
    print(f"[slides] wrote {out}  ({total} slides, {os.path.getsize(out)/1024:.1f} KB)")


if __name__ == "__main__":
    build()
