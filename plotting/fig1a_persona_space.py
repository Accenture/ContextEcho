"""Fig 1(a) — Behavioral persona space (left panel only).

Single-panel version of the PCA persona-space scatter, split out from
fig_fig1_combined.py so the panel can be edited / re-rendered
independently of the trajectory panel.

  6 judge-free features → 2D PCA, colored by 4-point judge score,
  with cluster centroids for the filler-arm ("Disciplined-Assistant")
  and the claude-arm ("Drifted-Persona").

Output: paper/figures/fig1a_persona_space.{png,pdf}
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fig_fig1_combined import TARGET_LABEL, REPO_ROOT, load_records

FEAT = ["hedge_density", "experiential_density", "commits_pref",
        "em_dash_count", "paragraph_breaks", "log_len"]

OUT_DIR_PAPER = REPO_ROOT / "paper" / "figures"
OUT_DIR_PAPER.mkdir(parents=True, exist_ok=True)


def main() -> int:
    recs = load_records()
    print(f"Loaded {len(recs)} probe responses for {TARGET_LABEL}")
    if not recs:
        return 1

    # PCA
    X = np.array([[r[k] for k in FEAT] for r in recs], dtype=float)
    mu = X.mean(axis=0); sd = X.std(axis=0); sd[sd == 0] = 1.0
    Xz = (X - mu) / sd
    _U, S, Vt = np.linalg.svd(Xz, full_matrices=False)
    PC = Vt[:2]
    explained = (S**2) / (S**2).sum()
    proj = Xz @ PC.T
    for i, r in enumerate(recs):
        r["pc1"], r["pc2"] = float(proj[i, 0]), float(proj[i, 1])

    # Orient PC1 so claude (drift) is on positive side
    cl_pc1 = np.mean([r["pc1"] for r in recs if r["arm"] == "claude_session"])
    fil_pc1 = np.mean([r["pc1"] for r in recs if r["arm"] == "filler"])
    if cl_pc1 < fil_pc1:
        for r in recs: r["pc1"] = -r["pc1"]
        proj[:, 0] = -proj[:, 0]; PC[0] = -PC[0]

    print(f"PC1 explains {explained[0]:.1%}, PC2 {explained[1]:.1%}")

    # Plot
    fig, ax_l = plt.subplots(figsize=(7.8, 5.6))

    score_color = {0: "#7f1d1d", 1: "#f97316", 2: "#fbbf24", 3: "#1d4ed8"}
    score_label = {0: "score 0 (drifted)",
                   1: "score 1 (partial)",
                   2: "score 2 (mostly assistant)",
                   3: "score 3 (fully assistant)"}
    arm_marker = {"filler": "o", "claude_session": "^"}
    arm_label  = {"filler": "filler arm", "claude_session": "claude arm"}

    counts = {arm: {s: 0 for s in (0, 1, 2, 3)}
              for arm in ("filler", "claude_session")}
    for arm in ("filler", "claude_session"):
        for s in (3, 2, 1, 0):
            pts = [r for r in recs if r["arm"] == arm and r["score"] == s]
            if not pts: continue
            counts[arm][s] = len(pts)
            xs = [r["pc1"] for r in pts]
            ys = [r["pc2"] for r in pts]
            ax_l.scatter(xs, ys, s=22, c=score_color[s],
                         marker=arm_marker[arm], alpha=0.6,
                         edgecolors="white", linewidths=0.4,
                         zorder=3 if arm == "filler" else 4)

    # Centroids
    fil_pts = [r for r in recs if r["arm"] == "filler"]
    cl_pts  = [r for r in recs if r["arm"] == "claude_session"]
    fc = (float(np.mean([r["pc1"] for r in fil_pts])),
          float(np.mean([r["pc2"] for r in fil_pts])))
    cc = (float(np.mean([r["pc1"] for r in cl_pts])),
          float(np.mean([r["pc2"] for r in cl_pts])))
    ax_l.scatter(*fc, marker="*", s=520, c="#1d4ed8",
                 edgecolors="white", linewidths=1.5, zorder=6)
    ax_l.annotate("Disciplined-Assistant\ncluster",
                  xy=fc, xytext=(fc[0] - 1.0, fc[1] - 1.0),
                  fontsize=13, fontweight="bold", color="#1d4ed8",
                  ha="center",
                  arrowprops=dict(arrowstyle="-", color="#1d4ed8",
                                  lw=1, alpha=0.6))
    ax_l.scatter(*cc, marker="*", s=520, c="#7f1d1d",
                 edgecolors="white", linewidths=1.5, zorder=6)
    ax_l.annotate("Drifted-Persona\ncluster",
                  xy=cc, xytext=(cc[0] + 1.5, cc[1] + 1.0),
                  fontsize=13, fontweight="bold", color="#7f1d1d",
                  ha="center",
                  arrowprops=dict(arrowstyle="-", color="#7f1d1d",
                                  lw=1, alpha=0.6))

    # Clip axes to focus on the dense central cluster. Outliers remain in
    # the PCA fit (centroids and explained-variance are unchanged) but
    # are pushed offscreen so the cluster separation reads at a glance.
    pc1_all = np.array([r["pc1"] for r in recs])
    pc2_all = np.array([r["pc2"] for r in recs])
    pc1_lo, pc1_hi = np.percentile(pc1_all, [5, 95])
    pc2_lo, pc2_hi = np.percentile(pc2_all, [5, 95])
    pc1_pad = 0.15 * (pc1_hi - pc1_lo)
    pc2_pad = 0.15 * (pc2_hi - pc2_lo)
    ax_l.set_xlim(pc1_lo - pc1_pad, pc1_hi + pc1_pad)
    ax_l.set_ylim(pc2_lo - pc2_pad, pc2_hi + pc2_pad)
    n_clipped = int(((pc1_all < pc1_lo - pc1_pad) | (pc1_all > pc1_hi + pc1_pad) |
                     (pc2_all < pc2_lo - pc2_pad) | (pc2_all > pc2_hi + pc2_pad)).sum())
    print(f"Axis clipped to 2nd–98th percentile; {n_clipped} of {len(recs)} "
          f"outliers offscreen (PCA fit unchanged).")

    ax_l.set_xlabel(f"PC1 ({explained[0]:.0%} var) → drift direction",
                    fontsize=18)
    ax_l.set_ylabel(f"PC2 ({explained[1]:.0%} var) → response style",
                    fontsize=18)
    ax_l.tick_params(axis="both", labelsize=18)
    ax_l.grid(True, alpha=0.25)
    ax_l.set_axisbelow(True)
    ax_l.spines["top"].set_visible(False)
    ax_l.spines["right"].set_visible(False)
    handles = [
        plt.Line2D([], [], marker="o", color=score_color[3], linestyle="",
                   markersize=9, label="score 3 (fully)"),
        plt.Line2D([], [], marker="o", color=score_color[2], linestyle="",
                   markersize=9, label="score 2 (mostly)"),
        plt.Line2D([], [], marker="o", color=score_color[1], linestyle="",
                   markersize=9, label="score 1 (partial)"),
        plt.Line2D([], [], marker="o", color=score_color[0], linestyle="",
                   markersize=9, label="score 0 (drifted)"),
    ]
    ax_l.legend(handles=handles, loc="upper left",
                fontsize=13, framealpha=0.9, ncol=1, labelspacing=0.35)

    plt.tight_layout()
    out_pdf = OUT_DIR_PAPER / "fig1a_persona_space.pdf"
    out_png = OUT_DIR_PAPER / "fig1a_persona_space.png"
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.savefig(out_png, dpi=160, bbox_inches="tight")
    print(f"Saved {out_pdf}")
    print(f"Saved {out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
