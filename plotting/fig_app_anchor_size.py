"""Appendix figure: anchor-size sensitivity sweep.

3 anchor variants (small ~30 tok, medium ~75 tok shipped A_COMBINED,
large ~200 tok with extra demos) × 6 targets at P5. Bar chart per
target with the 3 sizes.

Output: paper/figures/fig_app_anchor_size.{png,pdf}
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA = REPO_ROOT / "results" / "anchor_size_sweep"
OUT = REPO_ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def load_target_size(size: str, target: str):
    d = DATA / size / target
    if not d.exists(): return []
    scores = []
    for f in sorted(d.iterdir()):
        if f.suffix != ".json": continue
        try:
            s = json.loads(f.read_text()).get("score")
            if isinstance(s, int): scores.append(s)
        except Exception:
            pass
    return scores


SIZES = [("small", "~30 tok\n(V0 only)", "#93c5fd"),
         ("medium", "~75 tok\n(shipped A)", "#15803d"),
         ("large", "~200 tok\n(A + 2 extra demos)", "#a78bfa")]
TARGETS = [("claude-sonnet-4-6", "Sonnet 4.6"),
           ("claude-sonnet-4-5", "Sonnet 4.5"),
           ("claude-opus-4-1",   "Opus 4.1"),
           ("claude-haiku-4-5",  "Haiku 4.5"),
           ("deepseek-v3",       "DeepSeek V3"),
           ("gemini-2-5-pro",    "Gemini 2.5 Pro")]


def main() -> int:
    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    n_targets = len(TARGETS)
    bar_w = 0.27
    x = np.arange(n_targets)

    for j, (size_key, size_label, color) in enumerate(SIZES):
        means = []
        for tgt_key, _ in TARGETS:
            scores = load_target_size(size_key, tgt_key)
            means.append(float(np.mean(scores)) if scores else float("nan"))
        offset = (j - 1) * bar_w
        ax.bar(x + offset, means, bar_w, color=color,
               edgecolor="white", linewidth=0.6,
               label=size_label, zorder=3)

    ax.axhline(3.0, color="gray", linestyle=":", linewidth=0.8,
               alpha=0.55, zorder=1)
    ax.text(n_targets - 0.45, 3.05, "rubric ceiling", fontsize=11,
            color="dimgray", ha="right", va="bottom")

    ax.set_xticks(x)
    ax.set_xticklabels([n for _, n in TARGETS], fontsize=12)
    ax.set_yticks([0, 1, 2, 3])
    ax.tick_params(axis="y", labelsize=12)
    ax.set_ylim(0, 3.4)
    ax.set_ylabel("Mean Judge Score\n(0=Drifted → 3=Fully Assistant)",
                  fontsize=13)
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower right", fontsize=12, framealpha=0.92, ncol=3,
              title="Anchor Size", title_fontsize=12)

    plt.tight_layout()
    out_pdf = OUT / "fig_app_anchor_size.pdf"
    out_png = OUT / "fig_app_anchor_size.png"
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.savefig(out_png, dpi=160, bbox_inches="tight")
    print(f"Saved {out_pdf}")
    print(f"Saved {out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
