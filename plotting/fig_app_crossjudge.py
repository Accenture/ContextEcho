"""Appendix figure: cross-judge agreement scatter.

Plots Sonnet judge score (x) vs GPT-5 judge score (y) for the panel-wide
P5 5-coding-probe responses, with κ and ρ in caption.

Output: paper/figures/fig_app_crossjudge.{png,pdf}
"""
from __future__ import annotations

import json
import statistics
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA = REPO_ROOT / "results" / "crossjudge_audit"
OUT = REPO_ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def main() -> int:
    if not DATA.exists():
        print(f"NO DATA at {DATA}"); return 1

    rows = []
    for f in DATA.rglob("*.json"):
        if f.name in ("RESULTS.json", "AGREEMENT.md"): continue
        if "judge_cost" in f.name: continue
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        if "sonnet_score" in d and "gpt5_score" in d:
            rows.append(d)

    if not rows:
        print("No paired scores found"); return 1

    s = [r["sonnet_score"] for r in rows]
    g = [r["gpt5_score"] for r in rows]
    n = len(rows)
    exact_agree = sum(1 for a, b in zip(s, g) if a == b) / n
    within_one = sum(1 for a, b in zip(s, g) if abs(a - b) <= 1) / n

    cnt_s = Counter(s); cnt_g = Counter(g)
    p_o = exact_agree
    p_e = sum((cnt_s[k] / n) * (cnt_g[k] / n) for k in {0, 1, 2, 3})
    kappa = (p_o - p_e) / (1 - p_e) if p_e != 1 else float("nan")

    def rank(arr):
        order = sorted(range(len(arr)), key=lambda i: arr[i])
        ranks = [0.0] * len(arr)
        for rk, idx in enumerate(order):
            ranks[idx] = rk + 1
        return ranks
    rho = statistics.correlation(rank(s), rank(g)) if statistics.pstdev(s) > 0 and statistics.pstdev(g) > 0 else float("nan")

    fig, ax = plt.subplots(figsize=(6.0, 5.6))

    # Jitter for plotting (otherwise all points pile on integer grid)
    rng = np.random.default_rng(42)
    s_j = np.array(s) + rng.uniform(-0.15, 0.15, size=n)
    g_j = np.array(g) + rng.uniform(-0.15, 0.15, size=n)

    ax.scatter(s_j, g_j, s=24, c="#0284c7", alpha=0.4,
               edgecolors="white", linewidths=0.4)
    # y=x reference
    ax.plot([-0.3, 3.3], [-0.3, 3.3], color="#9ca3af", linestyle="--",
            linewidth=1.0, alpha=0.6, label="$y=x$ (perfect agreement)")

    ax.set_xlabel("Sonnet 4.6 Judge Score (Primary)", fontsize=13)
    ax.set_ylabel("GPT-5 Judge Score (Audit)", fontsize=13)
    ax.set_xticks([0, 1, 2, 3]); ax.set_yticks([0, 1, 2, 3])
    ax.tick_params(axis="both", labelsize=12)
    ax.set_xlim(-0.4, 3.4); ax.set_ylim(-0.4, 3.4)
    ax.grid(True, alpha=0.25); ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", fontsize=12)

    summary_text = (
        f"$n={n}$ paired scores\n"
        f"exact agreement: {exact_agree:.1%}\n"
        f"within-one: {within_one:.1%}\n"
        f"Cohen $\\kappa$: {kappa:.2f}\n"
        f"Spearman $\\rho$: {rho:.2f}"
    )
    ax.text(0.97, 0.03, summary_text, transform=ax.transAxes,
            ha="right", va="bottom", fontsize=11,
            bbox=dict(boxstyle="round,pad=0.4", fc="white",
                       ec="#9ca3af", alpha=0.9))

    plt.tight_layout()
    out_pdf = OUT / "fig_app_crossjudge.pdf"
    out_png = OUT / "fig_app_crossjudge.png"
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.savefig(out_png, dpi=160, bbox_inches="tight")
    print(f"Saved {out_pdf}")
    print(f"Saved {out_png}")
    print(f"\nn={n}, exact={exact_agree:.3f}, within-one={within_one:.3f}, "
          f"kappa={kappa:.3f}, rho={rho:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
