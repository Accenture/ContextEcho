"""Fig 2 — Panel-wide forest plot (the §3 headline).

One row per model. Two markers: filler (control) and claude (drift),
connected by a thin gray line. Bootstrap 95% CIs as horizontal whiskers.
Reasoning-tier rows shaded. Sorted within tier by gap descending.

This is THE headline figure for §3 (the phenomenon at panel scope).
Fig 1's deep-dive single-target figure (Sonnet 4.5) is the
"high-resolution" view; Fig 2 is the panel-wide breadth view.

Inputs: snapshot-at-position 25-probe data in
  results/probes_at_crosscompaction/<target>/<position>/<arm>/*.json

Output:
  paper/figures/fig2_forest_panelwide.{png,pdf}
  data_archive/fig2/FOREST_PANELWIDE.{png,pdf}
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
ROOT = REPO_ROOT / "results" / "probes_at_crosscompaction"
OUT_DATA = REPO_ROOT / "data_archive" / "fig2"
OUT_DATA.mkdir(parents=True, exist_ok=True)
OUT_PAPER = REPO_ROOT / "paper" / "figures"
OUT_PAPER.mkdir(parents=True, exist_ok=True)

POSITIONS = [
    "P0_start", "P1_pre_C1", "P2_post_C1", "P_pre_C2", "P_post_C2",
    "P_pre_C3", "P3_post_C3", "P_pre_C4", "P_post_C4",
    "P_pre_C5", "P4_post_C5", "P5_pre_C6",
]

# (target_dir, label, org, reasoning_tier?)
TARGETS = [
    ("claude-sonnet-4-6",  "Sonnet 4.6",        "Anthropic", True),
    ("claude-sonnet-4-5",  "Sonnet 4.5",        "Anthropic", True),
    ("claude-opus-4-1",    "Opus 4.1",          "Anthropic", True),
    ("claude-haiku-4-5",   "Haiku 4.5",         "Anthropic", True),
    ("gpt-5",              "GPT-5",             "OpenAI",    True),
    ("gpt-5-mini",         "GPT-5-mini",        "OpenAI",    True),
    ("gpt-4o",             "GPT-4o",            "OpenAI",    False),
    ("gpt-4-1",            "GPT-4.1",           "OpenAI",    False),
    ("gemini-2-5-pro",     "Gemini 2.5 Pro",    "Google",    True),
    ("gemini-2-5-flash",   "Gemini 2.5 Flash",  "Google",    False),
    ("deepseek-v3",        "DeepSeek V3",       "DeepSeek",  False),
    ("mistral-small-latest",  "Mistral Small",    "Mistral",   False),
    ("mistral-medium-latest", "Mistral Medium",   "Mistral",   False),
    ("mistral-large-latest",  "Mistral Large",    "Mistral",   False),
    ("llama-3-3-70b",         "Llama 3.3 70B",   "Meta",      False),
    ("qwen3-235b",            "Qwen3 235B",      "Alibaba",   False),
    ("qwen3-next-80b-a3b",    "Qwen3 Next 80B",  "Alibaba",   True),
    ("moonshotai-Kimi-K2-6",  "Kimi K2.6",       "Moonshot",  True),
    ("command-a-03-2025",     "Command A",       "Cohere",    False),
    ("command-r7b-12-2024",   "Command R7B",     "Cohere",    False),
    ("nvidia-nemotron-3-nano-30b-a3b",    "Nemotron Nano 30B",   "NVIDIA", True),
    ("nvidia-nemotron-super-49b-v1-5",    "Nemotron Super 49B",  "NVIDIA", True),
    ("nvidia-nemotron-3-super-120b-a12b", "Nemotron Super 120B", "NVIDIA", True),
]

ORG_COLOR = {
    "Anthropic": "#0ea5e9",
    "OpenAI":    "#10b981",
    "Google":    "#a855f7",
    "Moonshot":  "#f97316",
    "Mistral":   "#dc2626",
    "DeepSeek":  "#84cc16",
    "Alibaba":   "#eab308",
    "Meta":      "#3b82f6",
    "Cohere":    "#ec4899",
    "NVIDIA":    "#22c55e",
}

FILLER_COLOR = "#1d4ed8"
CLAUDE_COLOR = "#7f1d1d"

# Restrict to the 5-probe coding-self sub-battery so the body forest matches
# the A-anchor mitigation surface (Fig 5) exactly. The full 25-probe battery
# is reported in the appendix as a robustness check.
CODING_PROBE_IDS = {"C01", "C02", "C03", "C04", "C05"}


def load_arm_by_position(target: str, arm: str) -> dict[str, list[int]]:
    """Group scores by position so we can do a clustered bootstrap.

    Filters to the 5 coding-self probes (C01-C05) only.
    """
    out: dict[str, list[int]] = {}
    for pos in POSITIONS:
        d = ROOT / target / pos / arm
        if not d.exists():
            continue
        scores: list[int] = []
        for f in sorted(d.iterdir()):
            if f.suffix != ".json":
                continue
            if f.stem not in CODING_PROBE_IDS:
                continue
            try:
                data = json.loads(f.read_text())
            except Exception:
                continue
            s = data.get("score")
            if isinstance(s, int):
                scores.append(s)
        if scores:
            out[pos] = scores
    return out


def clustered_bootstrap_mean(by_pos: dict[str, list[int]],
                              n_boot: int = 10000,
                              seed: int = 42):
    """Two-stage cluster bootstrap.

    Stage 1: resample positions with replacement (cluster level).
    Stage 2: within each resampled position, resample probe scores with replacement.
    Statistic = mean of position means (gives every position equal weight,
    which matches how we'd interpret "drift across the session").
    """
    if not by_pos:
        return float("nan"), float("nan"), float("nan"), 0
    rng = np.random.default_rng(seed)
    positions = list(by_pos.keys())
    arrays = {p: np.array(by_pos[p], dtype=float) for p in positions}
    n_pos = len(positions)

    # Point estimate = mean of position-level means (equal-weighted across positions)
    point = float(np.mean([arrays[p].mean() for p in positions]))

    boots = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        pos_idx = rng.integers(0, n_pos, size=n_pos)
        position_means = []
        for j in pos_idx:
            arr = arrays[positions[j]]
            sample = arr[rng.integers(0, len(arr), size=len(arr))]
            position_means.append(sample.mean())
        boots[b] = float(np.mean(position_means))

    lo = float(np.percentile(boots, 2.5))
    hi = float(np.percentile(boots, 97.5))
    n_total = sum(len(v) for v in by_pos.values())
    return point, lo, hi, n_total


def main() -> int:
    # Collect per-target stats (clustered bootstrap by position)
    rows = []
    for tgt, label, org, reasoning in TARGETS:
        cl_by_pos = load_arm_by_position(tgt, "claude_session")
        fi_by_pos = load_arm_by_position(tgt, "filler")
        if not cl_by_pos or not fi_by_pos:
            print(f"skip {label}: no data "
                  f"({len(cl_by_pos)} claude pos, {len(fi_by_pos)} filler pos)")
            continue
        cl_m, cl_lo, cl_hi, n_cl = clustered_bootstrap_mean(cl_by_pos)
        fi_m, fi_lo, fi_hi, n_fi = clustered_bootstrap_mean(fi_by_pos)
        gap = fi_m - cl_m
        rows.append({
            "target": tgt, "label": label, "org": org,
            "reasoning": reasoning,
            "n_cl": n_cl, "n_fi": n_fi,
            "n_pos_cl": len(cl_by_pos), "n_pos_fi": len(fi_by_pos),
            "cl_m": cl_m, "cl_lo": cl_lo, "cl_hi": cl_hi,
            "fi_m": fi_m, "fi_lo": fi_lo, "fi_hi": fi_hi,
            "gap": gap,
        })

    if not rows:
        print("no data")
        return 1

    # Sort: reasoning-tier first by gap desc, then non-reasoning by gap desc
    rows.sort(key=lambda r: (-int(r["reasoning"]), -r["gap"]))

    # ============================================================
    # Plot
    # ============================================================
    n_rows = len(rows)
    # Compact spacing: shrink per-row vertical (~0.30") so 19 rows fit in
    # ~7.3" instead of ~12". Markers/whiskers stay readable; rows just
    # bunch closer.
    fig_h = max(2.6, 0.30 * n_rows + 1.4)
    fig, ax = plt.subplots(figsize=(11.0, fig_h))

    y_positions = list(range(n_rows, 0, -1))  # top = first row

    for i, r in enumerate(rows):
        y = y_positions[i]
        org_col = ORG_COLOR.get(r["org"], "#666666")

        # Tier shading: yellow for reasoning, blue for non-reasoning
        if r["reasoning"]:
            ax.axhspan(y - 0.45, y + 0.45, color="#fbbf24", alpha=0.32,
                       zorder=0)
        else:
            ax.axhspan(y - 0.45, y + 0.45, color="#60a5fa", alpha=0.20,
                       zorder=0)

        # Connector line between filler and claude
        ax.plot([r["fi_m"], r["cl_m"]], [y, y],
                color="#9ca3af", linewidth=1.0, alpha=0.7, zorder=2)

        # n_pos=1 rows get hollow markers to flag pilot-position coverage.
        is_pilot = max(r["n_pos_cl"], r["n_pos_fi"]) <= 1

        # Filler arm: circle + 95% CI whisker
        ax.errorbar(r["fi_m"], y,
                    xerr=[[r["fi_m"] - r["fi_lo"]], [r["fi_hi"] - r["fi_m"]]],
                    fmt="o", markersize=8, color=FILLER_COLOR,
                    markerfacecolor="white" if is_pilot else FILLER_COLOR,
                    markeredgecolor=FILLER_COLOR if is_pilot else "white",
                    markeredgewidth=1.2 if is_pilot else 0.8,
                    ecolor=FILLER_COLOR, elinewidth=1.2, capsize=3,
                    zorder=4,
                    label="filler arm (control)" if i == 0 else None)

        # Claude arm: triangle + 95% CI whisker
        ax.errorbar(r["cl_m"], y,
                    xerr=[[r["cl_m"] - r["cl_lo"]], [r["cl_hi"] - r["cl_m"]]],
                    fmt="^", markersize=10, color=CLAUDE_COLOR,
                    markerfacecolor="white" if is_pilot else CLAUDE_COLOR,
                    markeredgecolor=CLAUDE_COLOR if is_pilot else "white",
                    markeredgewidth=1.2 if is_pilot else 0.8,
                    ecolor=CLAUDE_COLOR, elinewidth=1.2, capsize=3,
                    zorder=5,
                    label="claude arm (drift)" if i == 0 else None)

        # Right-margin annotation: gap
        ax.text(3.08, y, f"Δ {r['gap']:+.2f}",
                fontsize=12, color="#374151", va="center", ha="left",
                fontweight="bold" if abs(r["gap"]) >= 0.30 else "normal")

    # Y-axis labels = "Model (Org)" on a single line; pilot rows get suffix.
    ax.set_yticks(y_positions)
    def _row_label(r):
        suffix = "  (pilot)" if max(r["n_pos_cl"], r["n_pos_fi"]) <= 1 else ""
        return f"{r['label']}{suffix}  ({r['org']})"
    ax.set_yticklabels([_row_label(r) for r in rows], fontsize=13)
    ax.set_ylim(0.4, n_rows + 0.6)

    # X-axis = judge score; extra right padding for Δ annotation
    ax.set_xlim(-0.05, 3.40)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(["0 (drifted)", "1 (partial)", "2 (mostly)", "3 (fully)"],
                       fontsize=13)
    ax.set_xlabel("Position-Equal-Weighted Mean Judge Score\n"
                  "(Mean ± 95% Clustered Bootstrap CI; Clusters = 12 Positions)",
                  fontsize=13, labelpad=8)

    # Legend: 3 columns × 2 rows. Reorder handles so the two tier-shading
    # patches both land on row 2 under matplotlib's column-major fill.
    h_filler = plt.Line2D([], [], marker="o", color=FILLER_COLOR, linestyle="",
                          markersize=8, markeredgecolor="white",
                          label="filler arm (length-matched control)")
    h_claude = plt.Line2D([], [], marker="^", color=CLAUDE_COLOR, linestyle="",
                          markersize=9, markeredgecolor="white",
                          label="claude arm (drift)")
    h_hollow = plt.Line2D([], [], marker="^", color=CLAUDE_COLOR, linestyle="",
                          markersize=9, markerfacecolor="white",
                          markeredgecolor=CLAUDE_COLOR, markeredgewidth=1.2,
                          label="hollow marker = pilot ($n_{\\mathrm{pos}}=1$)")
    h_reason = mpatches.Patch(color="#fbbf24", alpha=0.55,
                              label="reasoning-tier model")
    h_nonreason = mpatches.Patch(color="#60a5fa", alpha=0.40,
                                 label="non-reasoning-tier model")
    # Column-major fill with ncol=3 → cols are filled top-down. Layout target:
    #   col1: filler / reasoning
    #   col2: claude / non-reasoning
    #   col3: hollow / (blank)
    # Insert an invisible Line2D as the row-2 col-3 placeholder.
    h_blank = plt.Line2D([], [], linestyle="", marker="", label="")
    # Place legend inside the plot at the bottom-left where the lower
    # non-reasoning rows (Command R7B, Llama, Mistral Small) leave the
    # x-range below ~2.0 empty. Single-column, 5 rows.
    handles = [h_filler, h_claude, h_hollow, h_reason, h_nonreason]
    ax.legend(handles=handles,
              loc="lower left", bbox_to_anchor=(0.005, 0.005),
              fontsize=10.5, framealpha=0.95, ncol=1,
              frameon=True, edgecolor="#d1d5db",
              facecolor="white",
              handletextpad=0.5)

    ax.grid(True, axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(left=False)

    plt.tight_layout()
    # Reserve a small bottom strip for the legend; this is now in *figure*
    # fractions, scaled to the actual compact figure height.
    plt.subplots_adjust(bottom=0.10)
    out_data_png = OUT_DATA / "FOREST_PANELWIDE.png"
    out_data_pdf = OUT_DATA / "FOREST_PANELWIDE.pdf"
    out_paper_pdf = OUT_PAPER / "fig2_forest_panelwide.pdf"
    out_paper_png = OUT_PAPER / "fig2_forest_panelwide.png"
    plt.savefig(out_data_png, dpi=160, bbox_inches="tight")
    plt.savefig(out_data_pdf, bbox_inches="tight")
    plt.savefig(out_paper_pdf, bbox_inches="tight")
    plt.savefig(out_paper_png, dpi=160, bbox_inches="tight")
    print(f"\nSaved {out_data_png}")
    print(f"Saved {out_data_pdf}")
    print(f"Saved {out_paper_pdf}")
    print(f"Saved {out_paper_png}")

    print(f"\nPer-target gaps (filler − claude), clustered bootstrap by position:")
    for r in rows:
        sig = " ★" if abs(r["gap"]) >= 0.30 else ""
        print(f"  {r['label']:<14} ({r['org']:<10}) "
              f"n_pos={r['n_pos_cl']}/{r['n_pos_fi']}  "
              f"filler {r['fi_m']:.2f} [{r['fi_lo']:.2f},{r['fi_hi']:.2f}], "
              f"claude {r['cl_m']:.2f} [{r['cl_lo']:.2f},{r['cl_hi']:.2f}], "
              f"Δ {r['gap']:+.2f}{sig}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
