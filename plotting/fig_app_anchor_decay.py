"""Appendix figure: anchor-decay curve.

Shows mean judge score on the 5 coding probes as a function of unanchored
turns inserted between A-anchor and probe (offsets {0, 1, 5, 10, 20}).

Output: paper/figures/fig_app_anchor_decay.{png,pdf}
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA = REPO_ROOT / "results" / "anchor_decay" / "claude-sonnet-4-5"
OUT = REPO_ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def main() -> int:
    if not DATA.exists():
        print(f"NO DATA: {DATA}"); return 1

    offsets = []
    means = []
    individual = []  # per-probe scores for scatter
    for d in sorted(DATA.iterdir(), key=lambda p: int(p.name.replace("N", "")) if p.name.startswith("N") else 999):
        if not d.is_dir() or not d.name.startswith("N"): continue
        n_off = int(d.name.replace("N", ""))
        scores = []
        for f in d.glob("*.json"):
            try:
                s = json.loads(f.read_text()).get("score")
                if isinstance(s, int): scores.append(s)
            except Exception: pass
        if scores:
            offsets.append(n_off)
            means.append(float(np.mean(scores)))
            for s in scores:
                individual.append((n_off, s))

    fig, ax = plt.subplots(figsize=(7.5, 3.6))

    # Reference: filler-arm baseline ≈ 1.47 on Sonnet 4.5 (5-coding-probes)
    ax.axhline(1.47, color="#3b82f6", linestyle="--", linewidth=1.4,
               alpha=0.75, label="filler-arm baseline (1.47)", zorder=2)
    # Drift baseline (claude no-anchor) ≈ 0.83
    ax.axhline(0.83, color="#dc2626", linestyle="--", linewidth=1.4,
               alpha=0.75, label="claude-arm drift baseline (0.83)", zorder=2)
    # Rubric ceiling
    ax.axhline(3.0, color="#9ca3af", linestyle=":", linewidth=1.0,
               alpha=0.5, zorder=1)

    # Individual probe scatter
    xs = [pt[0] for pt in individual]
    ys = [pt[1] for pt in individual]
    ax.scatter(xs, ys, s=38, c="#15803d", alpha=0.55,
               edgecolors="white", linewidths=0.6, zorder=3,
               label="individual probe scores ($n=5$ per offset)")
    # Mean line
    ax.plot(offsets, means, "-s", color="#15803d", linewidth=2.0,
            markersize=10, markeredgecolor="white", markeredgewidth=1.0,
            zorder=4, label="mean (anchor + N unanchored turns)")

    ax.set_xlabel("$N$ Unanchored Turns Inserted Between A-Anchor and Probe",
                  fontsize=13)
    ax.set_ylabel("Mean Judge Score\n(0=Drifted → 3=Fully Assistant)",
                  fontsize=13)
    ax.set_xticks(offsets)
    ax.set_yticks([0, 1, 2, 3])
    ax.tick_params(axis="both", labelsize=12)
    ax.set_ylim(-0.1, 3.2)
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlim(-0.5, 110)
    ax.grid(True, alpha=0.25)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower left", fontsize=12, framealpha=0.92, ncol=1)

    plt.tight_layout()
    out_pdf = OUT / "fig_app_anchor_decay.pdf"
    out_png = OUT / "fig_app_anchor_decay.png"
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.savefig(out_png, dpi=160, bbox_inches="tight")
    print(f"Saved {out_pdf}")
    print(f"Saved {out_png}")
    print(f"\nMean per offset: {dict(zip(offsets, means))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
