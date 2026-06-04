"""Fig 1(b) — Drift trajectory across the session (right panel only).

Single-panel version of the 12-position trajectory line plot, split out
from fig_fig1_combined.py so the panel can be edited / re-rendered
independently of the persona-space panel.

  25-probe judge score across 12 measurement positions, with
  bootstrap-95% CI bands and dashed compaction markers (C1–C6).

Output: paper/figures/fig1b_drift_trajectory.{png,pdf}
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fig_fig1_combined import (
    POSITIONS, COMPACTIONS, TARGET_LABEL, REPO_ROOT,
    load_records, bootstrap_mean,
)

OUT_DIR_PAPER = REPO_ROOT / "paper" / "figures"
OUT_DIR_PAPER.mkdir(parents=True, exist_ok=True)


def main() -> int:
    recs = load_records()
    print(f"Loaded {len(recs)} probe responses for {TARGET_LABEL}")
    if not recs:
        return 1

    # Aggregate by position
    turns_b = []
    cl_p, cl_lo, cl_hi = [], [], []
    fi_p, fi_lo, fi_hi = [], [], []
    for pos_label, turn in POSITIONS:
        cl = [r["score"] for r in recs
              if r["position"] == pos_label and r["arm"] == "claude_session"]
        fi = [r["score"] for r in recs
              if r["position"] == pos_label and r["arm"] == "filler"]
        if not cl or not fi: continue
        turns_b.append(turn)
        for arr, ps, los, his in (
            (cl, cl_p, cl_lo, cl_hi), (fi, fi_p, fi_lo, fi_hi),
        ):
            p, lo, hi = bootstrap_mean(arr)
            ps.append(p); los.append(lo); his.append(hi)

    cl_overall = float(np.mean(cl_p)) if cl_p else float("nan")
    fi_overall = float(np.mean(fi_p)) if fi_p else float("nan")
    gap = fi_overall - cl_overall

    # Plot
    fig, ax_r = plt.subplots(figsize=(7.6, 5.6))

    for ct in COMPACTIONS:
        ax_r.axvline(ct, color="gray", linestyle="--", linewidth=1.0,
                     alpha=0.55, zorder=1)
    ax_r.fill_between(turns_b, cl_lo, cl_hi, color="#dc2626", alpha=0.18, zorder=3)
    ax_r.plot(turns_b, cl_p, "-o", color="#dc2626", linewidth=2.2,
              markersize=6.5, alpha=0.95, zorder=4,
              label="claude session prefix (drift arm)")
    ax_r.fill_between(turns_b, fi_lo, fi_hi, color="#3b82f6", alpha=0.18, zorder=3)
    ax_r.plot(turns_b, fi_p, "-o", color="#3b82f6", linewidth=2.0,
              markersize=5.5, alpha=0.92, zorder=4,
              label="length-matched filler (control arm)")
    ax_r.set_xlabel("Turn in Session", fontsize=18)
    ax_r.set_ylabel("Mean Judge Score Across 25 Probes\n"
                    "(0=Drifted → 3=Fully Assistant)",
                    fontsize=18)
    ax_r.set_ylim(-0.1, 3.2)
    ax_r.set_yticks([0, 1, 2, 3])
    ax_r.set_xlim(0, max(COMPACTIONS) * 1.05)
    ax_r.tick_params(axis="both", labelsize=18)
    ax_r.grid(True, alpha=0.25)
    ax_r.set_axisbelow(True)
    ax_r.spines["top"].set_visible(False)
    ax_r.spines["right"].set_visible(False)
    ax_r.legend(loc="lower left", fontsize=18, framealpha=0.92)
    for ci_idx, ct in enumerate(COMPACTIONS, 1):
        ax_r.text(ct, 3.10, f"C{ci_idx}",
                  fontsize=13, color="dimgray", ha="center", va="top",
                  bbox=dict(boxstyle="round,pad=0.22", fc="white",
                            ec="dimgray", alpha=0.9),
                  zorder=6)

    plt.tight_layout()
    out_pdf = OUT_DIR_PAPER / "fig1b_drift_trajectory.pdf"
    out_png = OUT_DIR_PAPER / "fig1b_drift_trajectory.png"
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.savefig(out_png, dpi=160, bbox_inches="tight")
    print(f"Saved {out_pdf}")
    print(f"Saved {out_png}")
    print(f"\nDrift gap (filler − claude): {gap:+.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
