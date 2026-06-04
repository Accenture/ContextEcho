"""Figure 4: Surface-vs-substrate diagnostic comparison.

Two panel comparison answering "where does the family signal live?"

Left panel: Surface diagnostic — re-anchoring on Sonnet 4.6 across 3 anchor
strengths shows monotonic attenuation (positive Δ).

Right panel: Substrate diagnostic — activation steering on Qwen 3 32B
across 4 α strengths restores activation projection but NOT behavior
(behavioral Δ stays negative).

Combined reading: family signal is reachable from the conversational
surface for at least one target; substrate-level intervention fails on
this target.

Output:
  paper/figures/fig4_surface_vs_substrate.pdf
"""
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT = REPO_ROOT / "paper/figures/fig4_surface_vs_substrate.pdf"

# Surface re-anchoring on Sonnet 4.6 (from MITIGATION analysis)
SURFACE_LABELS = ["baseline\n(recent3K)", "anchor\nshort", "anchor\nmedium", "anchor\nstrong"]
# Mean hedge-compliance under each condition (Sonnet 4.6).
# scratch baseline = 2.88, recent3K = 2.40, then attenuated values
SURFACE_MEANS = [2.40, 2.88, 3.00, 3.12]  # hedge-compliance scale
SURFACE_DELTAS = [0.0, 0.48, 0.60, 0.72]  # vs recent3K
SURFACE_CI = [0.18, 0.18, 0.16, 0.14]
SCRATCH_BASELINE = 2.88

# Substrate steering on Qwen 3 32B (from STEERING analysis)
SUBSTRATE_ALPHAS = [0.0, 0.5, 1.0, 1.5]
# Behavioral Δ vs scratch as α grows (note: stays negative, gets worse)
SUBSTRATE_DELTAS = [-0.20, -0.20, -0.24, -0.32]  # behavioral
SUBSTRATE_CI = [0.18, 0.20, 0.18, 0.22]
# Activation projection toward scratch baseline (sanity-check, recovers)
SUBSTRATE_PROJ = [-19.5, -15.5, -11.0, -6.5]  # values from Path Z
SUBSTRATE_PROJ_BASELINE = -11.12  # scratch projection target


def main():
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.0, 3.2))

    color_pos = "#5d8aa8"  # blue: positive (signal reached)
    color_neg = "#c44e4e"  # red: negative (signal NOT reached)

    # ===== Left: Surface diagnostic =====
    x = np.arange(len(SURFACE_LABELS))
    bar_colors = ["#888"] + [color_pos] * 3
    axL.bar(x, SURFACE_MEANS, 0.65, color=bar_colors,
            edgecolor="black", linewidth=0.5)
    axL.errorbar(x, SURFACE_MEANS, yerr=SURFACE_CI, fmt="none",
                 ecolor="black", capsize=3, linewidth=0.8)

    # Scratch baseline reference line — darker + thicker so it stands out
    # against the bar fills and the y-grid.
    axL.axhline(SCRATCH_BASELINE, color="#374151", linestyle="--",
                linewidth=1.6, alpha=0.95, zorder=5)
    axL.text(3.4, SCRATCH_BASELINE + 0.04, "scratch baseline",
             ha="right", va="bottom", fontsize=11,
             color="#374151", style="italic", fontweight="bold")

    axL.set_xticks(x)
    axL.set_xticklabels(SURFACE_LABELS, fontsize=12)
    axL.tick_params(axis="y", labelsize=12)
    axL.set_ylabel("Hedge-Compliance (0--3)", fontsize=13)
    axL.set_ylim(1.8, 3.4)
    axL.grid(True, axis="y", alpha=0.25, linewidth=0.4)

    # Annotate Δ values above each bar
    for xi, (mean, delta) in enumerate(zip(SURFACE_MEANS, SURFACE_DELTAS)):
        if xi == 0:
            label = "(baseline)"
        else:
            label = f"$\\Delta=+{delta:.2f}$"
        axL.text(xi, mean + SURFACE_CI[xi] + 0.04, label,
                 ha="center", va="bottom",
                 fontsize=11, color="#3a5a7a" if xi > 0 else "#666",
                 fontweight="bold" if xi > 0 else "normal")

    # ===== Right: Substrate diagnostic =====
    # Two y-axes: behavioral Δ on left (red), activation projection on right (blue)
    axR_proj = axR.twinx()

    # Behavioral Δ line (stays negative, even worsens at α=1.5)
    axR.plot(SUBSTRATE_ALPHAS, SUBSTRATE_DELTAS,
             marker="o", linewidth=1.6, color=color_neg,
             label="Behavioral $\\Delta$ (vs scratch)", markersize=7)
    axR.fill_between(SUBSTRATE_ALPHAS,
                     [d - c for d, c in zip(SUBSTRATE_DELTAS, SUBSTRATE_CI)],
                     [d + c for d, c in zip(SUBSTRATE_DELTAS, SUBSTRATE_CI)],
                     alpha=0.15, color=color_neg)
    axR.axhline(0, color="#374151", linestyle="--",
                linewidth=1.2, alpha=0.85, zorder=4)

    # Activation projection (reaches the target — sanity passes)
    axR_proj.plot(SUBSTRATE_ALPHAS, SUBSTRATE_PROJ,
                  marker="s", linewidth=1.4, linestyle="--",
                  color=color_pos, label="Activation projection",
                  markersize=6, alpha=0.85)
    axR_proj.axhline(SUBSTRATE_PROJ_BASELINE, color=color_pos,
                     linestyle="--", linewidth=1.4, alpha=0.95, zorder=5)
    axR_proj.text(1.55, SUBSTRATE_PROJ_BASELINE + 0.5,
                  "scratch projection target",
                  ha="right", va="bottom", fontsize=11,
                  color=color_pos, style="italic", alpha=1.0,
                  fontweight="bold")

    axR.set_xticks(SUBSTRATE_ALPHAS)
    axR.tick_params(axis="x", labelsize=12)
    axR.set_xlabel(r"Steering Strength $\alpha$", fontsize=13)
    axR.set_ylabel(r"Behavioral $\Delta$ (0--3 Scale)",
                   fontsize=13, color=color_neg)
    axR.tick_params(axis="y", labelcolor=color_neg, labelsize=12)
    axR.set_ylim(-0.55, 0.10)

    axR_proj.set_ylabel("Activation Projection",
                        fontsize=13, color=color_pos)
    axR_proj.tick_params(axis="y", labelcolor=color_pos, labelsize=12)
    axR_proj.set_ylim(-22, -2)

    axR.grid(True, axis="y", alpha=0.20, linewidth=0.4)

    # Combined legend at bottom-left of right panel
    lines1, labels1 = axR.get_legend_handles_labels()
    lines2, labels2 = axR_proj.get_legend_handles_labels()
    axR.legend(lines1 + lines2, labels1 + labels2,
               loc="lower left", fontsize=11, framealpha=0.9)

    plt.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    plt.rcParams["text.usetex"] = False
    main()
