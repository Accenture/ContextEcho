"""Fig 5 — Forest plot of A-anchor mitigation (Sonnet 4.6 + 4.5 sample).

Three markers per row:
  ○ filler arm (control)
  ▲ claude arm (drift)
  ■ claude arm + A anchor (mitigation)

All three series use the SAME probe IDs (CODING_PROBES C01–C05) and the same
12 positions for a fair comparison. Bootstrap CI is clustered by position.

Right margin annotates two gaps:
  Δ drift = filler − claude        (problem)
  Δ recovered = anchor − claude    (mitigation)

Output:
  paper/figures/fig5_forest_mitigation.{png,pdf}
  data_archive/fig5/FOREST_MITIGATION.{png,pdf}
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
NO_ANCHOR_ROOT = REPO_ROOT / "results" / "probes_at_crosscompaction"
A_ANCHOR_ROOT = REPO_ROOT / "results" / "generalization_test" / "A_COMBINED" / "CODING_PROBES"
OUT_DATA = REPO_ROOT / "data_archive" / "fig5"
OUT_DATA.mkdir(parents=True, exist_ok=True)
OUT_PAPER = REPO_ROOT / "paper" / "figures"
OUT_PAPER.mkdir(parents=True, exist_ok=True)

POSITIONS = [
    "P0_start", "P1_pre_C1", "P2_post_C1", "P_pre_C2", "P_post_C2",
    "P_pre_C3", "P3_post_C3", "P_pre_C4", "P_post_C4",
    "P_pre_C5", "P4_post_C5", "P5_pre_C6",
]

# Coding probes are the only IDs the A-anchor experiment scored
CODING_PROBE_IDS = ["C01", "C02", "C03", "C04", "C05"]

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

FILLER_COLOR = "#1d4ed8"
CLAUDE_COLOR = "#7f1d1d"
ANCHOR_COLOR = "#15803d"

# Right-margin annotation columns (data x-coords; xlim is extended past 3.0
# so these labels live in dedicated whitespace beside the plot area).
DRIFT_COL_X = 3.10
ANCHOR_COL_X = 3.38


def read_score(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        d = json.loads(path.read_text())
    except Exception:
        return None
    s = d.get("score")
    return int(s) if isinstance(s, int) else None


def load_no_anchor_by_position(target: str, arm: str) -> dict[str, list[int]]:
    """No-anchor scores from probes_at_crosscompaction, restricted to CODING probes."""
    out: dict[str, list[int]] = {}
    for pos in POSITIONS:
        d = NO_ANCHOR_ROOT / target / pos / arm
        if not d.exists():
            continue
        scores: list[int] = []
        for pid in CODING_PROBE_IDS:
            s = read_score(d / f"{pid}.json")
            if s is not None:
                scores.append(s)
        if scores:
            out[pos] = scores
    return out


def load_anchor_by_position(target: str) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for pos in POSITIONS:
        d = A_ANCHOR_ROOT / target / pos
        if not d.exists():
            continue
        scores: list[int] = []
        for pid in CODING_PROBE_IDS:
            s = read_score(d / f"{pid}.json")
            if s is not None:
                scores.append(s)
        if scores:
            out[pos] = scores
    return out


def clustered_bootstrap_mean(by_pos: dict[str, list[int]],
                              n_boot: int = 10000,
                              seed: int = 42):
    if not by_pos:
        return float("nan"), float("nan"), float("nan"), 0
    rng = np.random.default_rng(seed)
    positions = list(by_pos.keys())
    arrays = {p: np.array(by_pos[p], dtype=float) for p in positions}
    n_pos = len(positions)
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
    n_total = sum(len(v) for v in by_pos.values())
    return point, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)), n_total


def main() -> int:
    rows = []
    for tgt, label, org, reasoning in TARGETS:
        cl_pos = load_no_anchor_by_position(tgt, "claude_session")
        fi_pos = load_no_anchor_by_position(tgt, "filler")
        a_pos  = load_anchor_by_position(tgt)
        if not cl_pos or not fi_pos or not a_pos:
            print(f"skip {label}: missing data "
                  f"(cl={len(cl_pos)}, fi={len(fi_pos)}, a={len(a_pos)})")
            continue
        cl_m, cl_lo, cl_hi, n_cl = clustered_bootstrap_mean(cl_pos)
        fi_m, fi_lo, fi_hi, n_fi = clustered_bootstrap_mean(fi_pos)
        a_m,  a_lo,  a_hi,  n_a  = clustered_bootstrap_mean(a_pos)
        rows.append({
            "target": tgt, "label": label, "org": org,
            "reasoning": reasoning,
            "n_cl": n_cl, "n_fi": n_fi, "n_a": n_a,
            "n_pos_cl": len(cl_pos), "n_pos_fi": len(fi_pos),
            "n_pos_a":  len(a_pos),
            "cl_m": cl_m, "cl_lo": cl_lo, "cl_hi": cl_hi,
            "fi_m": fi_m, "fi_lo": fi_lo, "fi_hi": fi_hi,
            "a_m":  a_m,  "a_lo":  a_lo,  "a_hi":  a_hi,
            "drift_gap":     fi_m - cl_m,
            "recovered_gap": a_m - cl_m,
        })

    if not rows:
        print("no data; abort")
        return 1

    rows.sort(key=lambda r: (-int(r["reasoning"]), -r["drift_gap"]))

    n_rows = len(rows)
    # Compact spacing matching Fig 2 (panel-wide forest): ~0.32" per row.
    # Right-margin gaps now sit on a single line as two columns (drift,
    # +anchor), so one row = one annotation row.
    fig_h = max(3.0, 0.32 * n_rows + 1.8)
    fig, ax = plt.subplots(figsize=(11.0, fig_h))

    y_positions = list(range(n_rows, 0, -1))

    for i, r in enumerate(rows):
        y = y_positions[i]
        if r["reasoning"]:
            ax.axhspan(y - 0.45, y + 0.45, color="#fbbf24", alpha=0.32, zorder=0)
        else:
            ax.axhspan(y - 0.45, y + 0.45, color="#60a5fa", alpha=0.20, zorder=0)

        # Connector: claude → anchor (mitigation arrow)
        ax.annotate(
            "", xy=(r["a_m"], y), xytext=(r["cl_m"], y),
            arrowprops=dict(arrowstyle="->", color=ANCHOR_COLOR,
                            lw=1.4, alpha=0.7),
            zorder=2,
        )
        # Faint connector claude → filler (drift gap)
        ax.plot([r["fi_m"], r["cl_m"]], [y, y],
                color="#9ca3af", linewidth=0.8, alpha=0.5, zorder=1,
                linestyle=":")

        # n_pos=1 rows get hollow markers to flag pilot-position coverage.
        is_pilot = max(r["n_pos_cl"], r["n_pos_fi"], r["n_pos_a"]) <= 1

        # Filler ○
        ax.errorbar(r["fi_m"], y,
                    xerr=[[r["fi_m"] - r["fi_lo"]], [r["fi_hi"] - r["fi_m"]]],
                    fmt="o", markersize=8, color=FILLER_COLOR,
                    markerfacecolor="white" if is_pilot else FILLER_COLOR,
                    markeredgecolor=FILLER_COLOR if is_pilot else "white",
                    markeredgewidth=1.2 if is_pilot else 0.8,
                    ecolor=FILLER_COLOR, elinewidth=1.0, capsize=3, zorder=4)
        # Claude ▲
        ax.errorbar(r["cl_m"], y,
                    xerr=[[r["cl_m"] - r["cl_lo"]], [r["cl_hi"] - r["cl_m"]]],
                    fmt="^", markersize=10, color=CLAUDE_COLOR,
                    markerfacecolor="white" if is_pilot else CLAUDE_COLOR,
                    markeredgecolor=CLAUDE_COLOR if is_pilot else "white",
                    markeredgewidth=1.2 if is_pilot else 0.8,
                    ecolor=CLAUDE_COLOR, elinewidth=1.0, capsize=3, zorder=5)
        # Anchor ■
        ax.errorbar(r["a_m"], y,
                    xerr=[[r["a_m"] - r["a_lo"]], [r["a_hi"] - r["a_m"]]],
                    fmt="s", markersize=9, color=ANCHOR_COLOR,
                    markerfacecolor="white" if is_pilot else ANCHOR_COLOR,
                    markeredgecolor=ANCHOR_COLOR if is_pilot else "white",
                    markeredgewidth=1.2 if is_pilot else 0.8,
                    ecolor=ANCHOR_COLOR, elinewidth=1.0, capsize=3, zorder=6)

        # Right-margin: drift Δ (left column) + anchor Δ (right column).
        # Single line per row; column header labels added once above row 0.
        ax.text(DRIFT_COL_X, y, f"{r['drift_gap']:+.2f}",
                fontsize=11, color=CLAUDE_COLOR, va="center", ha="left",
                fontweight="bold" if abs(r["drift_gap"]) >= 0.30 else "normal")
        ax.text(ANCHOR_COL_X, y, f"{r['recovered_gap']:+.2f}",
                fontsize=11, color=ANCHOR_COLOR, va="center", ha="left",
                fontweight="bold" if r["recovered_gap"] >= 0.30 else "normal")

    # Column headers above the top row
    top_y = y_positions[0]
    ax.text(DRIFT_COL_X, top_y + 0.65, "drift Δ",
            fontsize=11, color=CLAUDE_COLOR, va="bottom", ha="left",
            fontweight="bold")
    ax.text(ANCHOR_COL_X, top_y + 0.65, "+anchor Δ",
            fontsize=11, color=ANCHOR_COLOR, va="bottom", ha="left",
            fontweight="bold")

    ax.set_yticks(y_positions)
    def _row_label(r):
        suffix = "  (pilot)" if max(r["n_pos_cl"], r["n_pos_fi"], r["n_pos_a"]) <= 1 else ""
        return f"{r['label']}{suffix}"
    ax.set_yticklabels([_row_label(r) for r in rows], fontsize=13)
    ax.set_ylim(0.4, n_rows + 1.4)
    ax.set_xlim(-0.05, 3.80)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(["0 (drifted)", "1 (partial)", "2 (mostly)", "3 (fully)"],
                       fontsize=13)
    ax.set_xlabel("Position-Equal-Weighted Mean Judge Score on 5 Coding-Self Probes\n"
                  "(Mean ± 95% Clustered Bootstrap CI; Clusters = 12 Positions)",
                  fontsize=13, labelpad=8)

    h_filler = plt.Line2D([], [], marker="o", color=FILLER_COLOR, linestyle="",
                          markersize=8, markeredgecolor="white",
                          label="filler arm (control)")
    h_claude = plt.Line2D([], [], marker="^", color=CLAUDE_COLOR, linestyle="",
                          markersize=9, markeredgecolor="white",
                          label="claude arm (drift, no anchor)")
    h_anchor = plt.Line2D([], [], marker="s", color=ANCHOR_COLOR, linestyle="",
                          markersize=8, markeredgecolor="white",
                          label="claude arm + A anchor (mitigation)")
    h_hollow = plt.Line2D([], [], marker="^", color=CLAUDE_COLOR, linestyle="",
                          markersize=9, markerfacecolor="white",
                          markeredgecolor=CLAUDE_COLOR, markeredgewidth=1.2,
                          label="hollow marker = pilot ($n_{\\mathrm{pos}}=1$)")
    h_reason = mpatches.Patch(color="#fbbf24", alpha=0.55, label="reasoning-tier model")
    h_nonreason = mpatches.Patch(color="#60a5fa", alpha=0.40, label="non-reasoning-tier model")
    # Place legend inside the plot at the bottom-left where the lower
    # non-reasoning rows leave the x-range below ~2.0 empty. Single-column,
    # 6 rows.
    handles = [h_filler, h_claude, h_anchor, h_hollow, h_reason, h_nonreason]
    ax.legend(handles=handles, loc="lower left",
              bbox_to_anchor=(0.005, 0.005),
              fontsize=10.5, framealpha=0.95, ncol=1,
              frameon=True, edgecolor="#d1d5db",
              facecolor="white", handletextpad=0.5)

    ax.grid(True, axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(left=False)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.10)
    out_data_png = OUT_DATA / "FOREST_MITIGATION.png"
    out_data_pdf = OUT_DATA / "FOREST_MITIGATION.pdf"
    out_paper_pdf = OUT_PAPER / "fig5_forest_mitigation.pdf"
    out_paper_png = OUT_PAPER / "fig5_forest_mitigation.png"
    plt.savefig(out_data_png, dpi=160, bbox_inches="tight")
    plt.savefig(out_data_pdf, bbox_inches="tight")
    plt.savefig(out_paper_pdf, bbox_inches="tight")
    plt.savefig(out_paper_png, dpi=160, bbox_inches="tight")
    print(f"\nSaved {out_paper_png}\n")

    print(f"Per-target stats (5 coding probes × 12 positions):")
    for r in rows:
        print(f"  {r['label']:<12} | "
              f"filler {r['fi_m']:.2f} [{r['fi_lo']:.2f},{r['fi_hi']:.2f}] | "
              f"claude {r['cl_m']:.2f} [{r['cl_lo']:.2f},{r['cl_hi']:.2f}] | "
              f"anchor {r['a_m']:.2f} [{r['a_lo']:.2f},{r['a_hi']:.2f}] | "
              f"drift Δ {r['drift_gap']:+.2f} | recovered Δ {r['recovered_gap']:+.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
