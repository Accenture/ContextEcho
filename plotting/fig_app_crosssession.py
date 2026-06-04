"""Appendix figure: cross-session drift on Sonnet 4.5.

Compact 1×3 small-multiples: per-session per-position trajectory on
Sessions 1, 2, 3. Used by §3.5 / Appendix~\\ref{app:crosssession}.

Output: paper/figures/fig_app_crosssession.{png,pdf}
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT = REPO_ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

CODING = {"C01", "C02", "C03", "C04", "C05"}

# Sessions:
# Session 1 = original (probes_at_crosscompaction)
# Session 2 = chainassemble
# Session 3 = proeng
SESSIONS = [
    ("Session 1 (drift research, 9,643 turns)",
     REPO_ROOT / "results" / "probes_at_crosscompaction" / "claude-sonnet-4-5"),
    ("Session 2 (chainassemble, 3,746 turns)",
     REPO_ROOT / "results" / "probes_at_crosscompaction_chainassemble" / "claude-sonnet-4-5"),
    ("Session 3 (proeng, 4,918 turns)",
     REPO_ROOT / "results" / "probes_at_crosscompaction_proeng" / "claude-sonnet-4-5"),
]

CL_COLOR = "#dc2626"; FI_COLOR = "#2563eb"


def per_pos(base: Path):
    """Return ordered list of (pos_label, claude_mean, filler_mean)."""
    if not base.exists(): return []
    out = []
    for pos_dir in sorted(base.iterdir()):
        if not pos_dir.is_dir(): continue
        cl = []; fi = []
        for arm, store in [("claude_session", cl), ("filler", fi)]:
            d = pos_dir / arm
            if not d.exists(): continue
            for f in d.iterdir():
                if f.suffix != ".json": continue
                if f.stem not in CODING: continue
                try:
                    s = json.loads(f.read_text()).get("score")
                    if isinstance(s, int): store.append(s)
                except Exception: pass
        if cl and fi:
            out.append((pos_dir.name, float(np.mean(cl)), float(np.mean(fi))))
    return out


def main() -> int:
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.5), sharey=True)
    for ax, (title, base) in zip(axes, SESSIONS):
        data = per_pos(base)
        if not data:
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                    ha="center", va="center", fontsize=14)
            continue
        xs = list(range(len(data)))
        cls = [d[1] for d in data]
        fis = [d[2] for d in data]
        gap = float(np.mean(fis)) - float(np.mean(cls))
        marker = " ★" if abs(gap) >= 0.30 else ""
        ax.plot(xs, fis, "-o", color=FI_COLOR, linewidth=2.2, markersize=7,
                label="filler arm", alpha=0.9, zorder=3)
        ax.plot(xs, cls, "-^", color=CL_COLOR, linewidth=2.4, markersize=8,
                label="claude arm", alpha=0.95, zorder=4)
        ax.set_title(f"{title}\nposition-wise mean (gap {gap:+.2f}{marker})",
                     fontsize=13, pad=8)
        ax.set_xlabel("Position Index", fontsize=13)
        ax.set_xticks(xs)
        ax.set_xticklabels([str(i) for i in xs], fontsize=11)
        ax.set_ylim(-0.1, 3.2)
        ax.set_yticks([0, 1, 2, 3])
        ax.grid(True, alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if ax is axes[0]:
            ax.tick_params(axis="y", labelsize=12)
            ax.set_ylabel("Mean Judge Score\n(0=Drifted → 3=Fully Assistant)",
                          fontsize=13)
            ax.legend(loc="lower right", fontsize=12, framealpha=0.92)

    plt.tight_layout()
    out_pdf = OUT / "fig_app_crosssession.pdf"
    out_png = OUT / "fig_app_crosssession.png"
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.savefig(out_png, dpi=160, bbox_inches="tight")
    print(f"Saved {out_pdf}")
    print(f"Saved {out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
