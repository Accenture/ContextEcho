"""Appendix figure: drift-onset curves on the 4 Anthropic targets.

Sweeps 8 log-spaced turn positions {1, 5, 25, 100, 250, 500, 1000, 1500}
in the pre-C1 regime. 2x2 small-multiples — one panel per target.
Y-axis is the drift gap (filler − claude) on 5 coding-self probes with
bootstrap 95% CI.

Output: paper/figures/fig_app_onset.{pdf,png}
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA = REPO_ROOT / "results" / "drift_onset"
OUT = REPO_ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

TURNS = [1, 5, 25, 100, 250, 500, 1000, 1500]
N_BOOT = 10000
RNG = np.random.default_rng(42)

TARGETS = [
    ("claude-sonnet-4-5", "Sonnet 4.5"),
    ("claude-sonnet-4-6", "Sonnet 4.6"),
    ("claude-opus-4-1",   "Opus 4.1"),
    ("claude-haiku-4-5",  "Haiku 4.5"),
]


def load_arm(target: str, turn: int, arm: str) -> list[int]:
    d = DATA / target / f"T{turn:04d}" / arm
    out = []
    if not d.exists():
        return out
    for f in sorted(d.glob("*.json")):
        try:
            s = json.loads(f.read_text()).get("score")
            if isinstance(s, int):
                out.append(s)
        except Exception:
            pass
    return out


def boot_gap_ci(filler: list[int], claude: list[int],
                n_boot: int = N_BOOT) -> tuple[float, float, float]:
    if not filler or not claude:
        return float("nan"), float("nan"), float("nan")
    f = np.array(filler, dtype=float)
    c = np.array(claude, dtype=float)
    gap_mean = float(f.mean() - c.mean())
    f_samp = RNG.choice(f, size=(n_boot, len(f)), replace=True).mean(axis=1)
    c_samp = RNG.choice(c, size=(n_boot, len(c)), replace=True).mean(axis=1)
    diff = f_samp - c_samp
    lo, hi = np.percentile(diff, [2.5, 97.5])
    return gap_mean, float(lo), float(hi)


def main() -> int:
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 6.5), sharex=True, sharey=True)

    for (target_safe, target_name), ax in zip(TARGETS, axes.flat):
        rows = []
        for t in TURNS:
            f_scores = load_arm(target_safe, t, "filler")
            c_scores = load_arm(target_safe, t, "claude_session")
            g_m, g_lo, g_hi = boot_gap_ci(f_scores, c_scores)
            rows.append({
                "turn": t, "g_m": g_m, "g_lo": g_lo, "g_hi": g_hi,
                "n": min(len(f_scores), len(c_scores)),
            })

        xs = [r["turn"] for r in rows]
        gs = [r["g_m"] for r in rows]
        los = [r["g_m"] - r["g_lo"] for r in rows]
        his = [r["g_hi"] - r["g_m"] for r in rows]

        ax.errorbar(xs, gs, yerr=[los, his],
                    fmt="-s", color="#15803d", markersize=8,
                    markeredgecolor="white", markeredgewidth=0.8,
                    linewidth=1.6, capsize=3,
                    zorder=5)

        ax.axhline(0.0, color="gray", linestyle="-",
                   linewidth=0.8, alpha=0.6, zorder=1)
        ax.axhline(0.30, color="#dc2626", linestyle=":",
                   linewidth=0.9, alpha=0.6, zorder=1)
        ax.fill_between(xs, [0.0]*len(xs), gs,
                        where=[g > 0 for g in gs],
                        color="#dc2626", alpha=0.10, zorder=0)

        ax.set_xscale("symlog", linthresh=1)
        ax.set_xlim(0.7, 2200)
        ax.set_xticks(xs)
        ax.set_xticklabels([str(t) for t in xs], fontsize=11,
                           rotation=30, ha="right")
        ax.set_ylim(-0.8, 2.2)
        ax.set_yticks([-0.5, 0, 0.5, 1.0, 1.5, 2.0])
        ax.tick_params(axis="y", labelsize=12)
        ax.grid(True, alpha=0.25, which="both")
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        n_str = f"n={rows[0]['n']}"
        ax.set_title(f"{target_name}  ({n_str} per cell)",
                     fontsize=13, pad=6)

    # Outer labels
    for ax in axes[1, :]:
        ax.set_xlabel("Turn Position in Pre-C1 Regime\n"
                      "(symlog; C1 = turn 1338)", fontsize=13)
    for ax in axes[:, 0]:
        ax.set_ylabel("Drift Gap (Filler − Claude)", fontsize=13)

    plt.tight_layout()
    out_pdf = OUT / "fig_app_onset.pdf"
    out_png = OUT / "fig_app_onset.png"
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.savefig(out_png, dpi=160, bbox_inches="tight")
    print(f"\nSaved {out_pdf}")
    print(f"Saved {out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
