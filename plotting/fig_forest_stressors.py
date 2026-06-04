"""Fig 6 — Stressor-surface drift + A-anchor mitigation.

Two-panel forest plot, shared y-axis (model rows):
  Left:  S2_NO_PREAMBLE compliance rate %  (judge-free regex scorer)
  Right: length ratio (claude / filler char count, log scale)

Three markers per row in each panel:
  ○ filler arm (control)
  ▲ claude arm (drift)
  ■ claude arm + A anchor (mitigation)

Sonnet 4.6 + Sonnet 4.5 sample. 3 positions × 10 paraphrases per cell.

Output:
  paper/figures/fig6_forest_stressors.{png,pdf}
  data_archive/fig6/FOREST_STRESSORS.{png,pdf}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from experiments.e11_instruction_override.run import _is_no_preamble  # type: ignore  # noqa: E402

NO_ANCHOR_ROOT = REPO_ROOT / "results" / "cross_compaction"
A_ANCHOR_ROOT = REPO_ROOT / "results" / "dual_surface_pilot" / "CAND_A_COMBINED" / "stressors"
OUT_DATA = REPO_ROOT / "data_archive" / "fig6"
OUT_DATA.mkdir(parents=True, exist_ok=True)
OUT_PAPER = REPO_ROOT / "paper" / "figures"
OUT_PAPER.mkdir(parents=True, exist_ok=True)

POSITIONS = ["P0_start", "P3_post_C3", "P5_pre_C6"]

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

# Right-margin annotation columns — left panel (compliance, %).
# Tighter spacing: drop and +anchor sit close together.
LEFT_DROP_X = 108
LEFT_ANCHOR_X = 138
# Right-margin annotation columns — right panel (length ratio, log scale).
# Wider spacing on log axis: drift and +anchor need more multiplicative
# separation to read clearly.
RIGHT_DROP_X = 100
RIGHT_ANCHOR_X = 350


def read_text(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        d = json.loads(path.read_text())
    except Exception:
        return None
    return d.get("response_text")


def load_no_anchor_stressors(target: str) -> dict[str, dict]:
    """Returns {position: {claude: [{text, len}], filler: [{text, len}]}}."""
    out: dict[str, dict] = {}
    for pos in POSITIONS:
        d = NO_ANCHOR_ROOT / target / pos
        if not d.exists():
            continue
        bucket = {"claude": [], "filler": []}
        for v_dir in sorted(d.iterdir()):
            if not v_dir.is_dir():
                continue
            for arm in ("claude", "filler"):
                f = v_dir / f"{arm}.json"
                if not f.exists():
                    continue
                try:
                    data = json.loads(f.read_text())
                except Exception:
                    continue
                txt = data.get("response_text") or ""
                bucket[arm].append({"text": txt, "len": len(txt)})
        if bucket["claude"] and bucket["filler"]:
            out[pos] = bucket
    return out


def load_anchor_stressors(target: str) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for pos in POSITIONS:
        d = A_ANCHOR_ROOT / target / pos
        if not d.exists():
            continue
        bucket = []
        for v_dir in sorted(d.iterdir()):
            if not v_dir.is_dir():
                continue
            f = v_dir / "cell.json"
            if not f.exists():
                continue
            try:
                data = json.loads(f.read_text())
            except Exception:
                continue
            txt = data.get("response_text") or ""
            bucket.append({"text": txt, "len": len(txt)})
        if bucket:
            out[pos] = bucket
    return out


def compliance_rate(items: list[dict]) -> float:
    if not items:
        return float("nan")
    k = sum(1 for it in items if _is_no_preamble(it["text"]))
    return 100.0 * k / len(items)


def mean_len(items: list[dict]) -> float:
    if not items:
        return float("nan")
    return float(np.mean([it["len"] for it in items]))


def cluster_bootstrap_compliance(by_pos_items: dict[str, list[dict]],
                                  n_boot=10000, seed=42):
    """Two-stage bootstrap of compliance %."""
    if not by_pos_items:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    positions = list(by_pos_items.keys())
    point = float(np.mean([compliance_rate(by_pos_items[p]) for p in positions]))
    n_pos = len(positions)
    boots = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        pos_idx = rng.integers(0, n_pos, size=n_pos)
        rates = []
        for j in pos_idx:
            items = by_pos_items[positions[j]]
            sample = [items[k] for k in rng.integers(0, len(items), size=len(items))]
            rates.append(compliance_rate(sample))
        boots[b] = float(np.mean(rates))
    return point, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def cluster_bootstrap_ratio(claude_by_pos: dict[str, list[dict]],
                             filler_by_pos: dict[str, list[dict]],
                             n_boot=10000, seed=42):
    """Two-stage bootstrap of length ratio (mean of position-level ratios)."""
    if not claude_by_pos or not filler_by_pos:
        return float("nan"), float("nan"), float("nan")
    common = sorted(set(claude_by_pos.keys()) & set(filler_by_pos.keys()))
    if not common:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    point = float(np.mean([
        mean_len(claude_by_pos[p]) / max(1, mean_len(filler_by_pos[p]))
        for p in common
    ]))
    n_pos = len(common)
    boots = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        pos_idx = rng.integers(0, n_pos, size=n_pos)
        ratios = []
        for j in pos_idx:
            p = common[j]
            cl_items = claude_by_pos[p]
            fi_items = filler_by_pos[p]
            cl_sample = [cl_items[k] for k in rng.integers(0, len(cl_items), size=len(cl_items))]
            fi_sample = [fi_items[k] for k in rng.integers(0, len(fi_items), size=len(fi_items))]
            ratios.append(mean_len(cl_sample) / max(1.0, mean_len(fi_sample)))
        boots[b] = float(np.mean(ratios))
    return point, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def cluster_bootstrap_anchor_ratio(anchor_by_pos: dict[str, list[dict]],
                                    filler_by_pos: dict[str, list[dict]],
                                    n_boot=10000, seed=42):
    """Length ratio for anchor: anchor / filler, position-clustered."""
    if not anchor_by_pos or not filler_by_pos:
        return float("nan"), float("nan"), float("nan")
    common = sorted(set(anchor_by_pos.keys()) & set(filler_by_pos.keys()))
    if not common:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    point = float(np.mean([
        mean_len(anchor_by_pos[p]) / max(1, mean_len(filler_by_pos[p]))
        for p in common
    ]))
    n_pos = len(common)
    boots = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        pos_idx = rng.integers(0, n_pos, size=n_pos)
        ratios = []
        for j in pos_idx:
            p = common[j]
            a_items = anchor_by_pos[p]
            fi_items = filler_by_pos[p]
            a_sample = [a_items[k] for k in rng.integers(0, len(a_items), size=len(a_items))]
            fi_sample = [fi_items[k] for k in rng.integers(0, len(fi_items), size=len(fi_items))]
            ratios.append(mean_len(a_sample) / max(1.0, mean_len(fi_sample)))
        boots[b] = float(np.mean(ratios))
    return point, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main() -> int:
    rows = []
    for tgt, label, org, reasoning in TARGETS:
        no_anchor = load_no_anchor_stressors(tgt)
        anchor = load_anchor_stressors(tgt)
        if not no_anchor or not anchor:
            print(f"skip {label}: no data ({len(no_anchor)} / {len(anchor)} positions)")
            continue

        # Per-arm flat-by-position dicts
        cl_by_pos = {p: no_anchor[p]["claude"] for p in no_anchor}
        fi_by_pos = {p: no_anchor[p]["filler"] for p in no_anchor}

        # Compliance rates
        fi_c, fi_c_lo, fi_c_hi = cluster_bootstrap_compliance(fi_by_pos)
        cl_c, cl_c_lo, cl_c_hi = cluster_bootstrap_compliance(cl_by_pos)
        a_c,  a_c_lo,  a_c_hi  = cluster_bootstrap_compliance(anchor)

        # Length ratios (vs filler)
        fi_r, fi_r_lo, fi_r_hi = 1.0, 1.0, 1.0  # filler is the reference
        cl_r, cl_r_lo, cl_r_hi = cluster_bootstrap_ratio(cl_by_pos, fi_by_pos)
        a_r,  a_r_lo,  a_r_hi  = cluster_bootstrap_anchor_ratio(anchor, fi_by_pos)

        rows.append({
            "target": tgt, "label": label, "org": org, "reasoning": reasoning,
            "n_pos_cl": len(cl_by_pos), "n_pos_fi": len(fi_by_pos),
            "n_pos_a":  len(anchor),
            "fi_c": fi_c, "fi_c_lo": fi_c_lo, "fi_c_hi": fi_c_hi,
            "cl_c": cl_c, "cl_c_lo": cl_c_lo, "cl_c_hi": cl_c_hi,
            "a_c":  a_c,  "a_c_lo":  a_c_lo,  "a_c_hi":  a_c_hi,
            "fi_r": fi_r, "fi_r_lo": fi_r_lo, "fi_r_hi": fi_r_hi,
            "cl_r": cl_r, "cl_r_lo": cl_r_lo, "cl_r_hi": cl_r_hi,
            "a_r":  a_r,  "a_r_lo":  a_r_lo,  "a_r_hi":  a_r_hi,
            "compliance_drop":     fi_c - cl_c,         # drift = how much compliance is lost
            "compliance_recovery": a_c - cl_c,           # what anchor recovers
            "length_inflation":    cl_r,                 # drift inflation factor (≥1 = worse)
            "length_after_anchor": a_r,
        })

    if not rows:
        print("no data; abort")
        return 1

    rows.sort(key=lambda r: (-int(r["reasoning"]), -r["compliance_drop"]))

    n_rows = len(rows)
    fig_h = max(3.0, 0.32 * n_rows + 1.6)
    fig, (ax_l, ax_r) = plt.subplots(
        1, 2, figsize=(15.0, fig_h),
        gridspec_kw=dict(width_ratios=[1.0, 1.0], wspace=0.05),
        sharey=True,
    )

    y_positions = list(range(n_rows, 0, -1))

    # ============================================================
    # LEFT PANEL — compliance rate %
    # ============================================================
    for i, r in enumerate(rows):
        y = y_positions[i]
        if r["reasoning"]:
            ax_l.axhspan(y - 0.45, y + 0.45, color="#fbbf24", alpha=0.32, zorder=0)
        else:
            ax_l.axhspan(y - 0.45, y + 0.45, color="#60a5fa", alpha=0.20, zorder=0)

        # Mitigation arrow claude → anchor
        ax_l.annotate("", xy=(r["a_c"], y), xytext=(r["cl_c"], y),
                      arrowprops=dict(arrowstyle="->", color=ANCHOR_COLOR,
                                      lw=1.4, alpha=0.7), zorder=2)
        # Drift dotted line claude → filler
        ax_l.plot([r["fi_c"], r["cl_c"]], [y, y],
                  color="#9ca3af", linewidth=0.8, alpha=0.5,
                  zorder=1, linestyle=":")

        is_pilot = max(r["n_pos_cl"], r["n_pos_fi"], r["n_pos_a"]) <= 1
        ax_l.errorbar(r["fi_c"], y,
                      xerr=[[r["fi_c"] - r["fi_c_lo"]], [r["fi_c_hi"] - r["fi_c"]]],
                      fmt="o", markersize=8, color=FILLER_COLOR,
                      markerfacecolor="white" if is_pilot else FILLER_COLOR,
                      markeredgecolor=FILLER_COLOR if is_pilot else "white",
                      markeredgewidth=1.2 if is_pilot else 0.8,
                      ecolor=FILLER_COLOR, elinewidth=1.0, capsize=3, zorder=4)
        ax_l.errorbar(r["cl_c"], y,
                      xerr=[[r["cl_c"] - r["cl_c_lo"]], [r["cl_c_hi"] - r["cl_c"]]],
                      fmt="^", markersize=10, color=CLAUDE_COLOR,
                      markerfacecolor="white" if is_pilot else CLAUDE_COLOR,
                      markeredgecolor=CLAUDE_COLOR if is_pilot else "white",
                      markeredgewidth=1.2 if is_pilot else 0.8,
                      ecolor=CLAUDE_COLOR, elinewidth=1.0, capsize=3, zorder=5)
        ax_l.errorbar(r["a_c"], y,
                      xerr=[[r["a_c"] - r["a_c_lo"]], [r["a_c_hi"] - r["a_c"]]],
                      fmt="s", markersize=9, color=ANCHOR_COLOR,
                      markerfacecolor="white" if is_pilot else ANCHOR_COLOR,
                      markeredgecolor=ANCHOR_COLOR if is_pilot else "white",
                      markeredgewidth=1.2 if is_pilot else 0.8,
                      ecolor=ANCHOR_COLOR, elinewidth=1.0, capsize=3, zorder=6)

        # Right-margin: drop column (red) + recovery column (green) on a
        # single line; column headers added once above row 0.
        ax_l.text(LEFT_DROP_X, y, f"{r['compliance_drop']:+.0f}pp",
                  fontsize=11, color=CLAUDE_COLOR, va="center", ha="left",
                  fontweight="bold" if abs(r["compliance_drop"]) >= 15 else "normal")
        ax_l.text(LEFT_ANCHOR_X, y, f"{r['compliance_recovery']:+.0f}pp",
                  fontsize=11, color=ANCHOR_COLOR, va="center", ha="left",
                  fontweight="bold" if r["compliance_recovery"] >= 10 else "normal")

    # Column headers for left-panel right margin (drop / +anchor)
    top_y = y_positions[0]
    ax_l.text(LEFT_DROP_X, top_y + 0.65, "drop",
              fontsize=11, color=CLAUDE_COLOR, va="bottom", ha="left",
              fontweight="bold")
    ax_l.text(LEFT_ANCHOR_X, top_y + 0.65, "+anchor",
              fontsize=11, color=ANCHOR_COLOR, va="bottom", ha="left",
              fontweight="bold")

    ax_l.set_yticks(y_positions)
    def _row_label(r):
        suffix = "  (pilot)" if max(r["n_pos_cl"], r["n_pos_fi"], r["n_pos_a"]) <= 1 else ""
        return f"{r['label']}{suffix}"
    ax_l.set_yticklabels([_row_label(r) for r in rows], fontsize=12)
    ax_l.set_ylim(0.4, n_rows + 1.4)
    ax_l.set_xlim(-2, 200)
    ax_l.set_xticks([0, 25, 50, 75, 100])
    ax_l.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=12)
    ax_l.set_xlabel("S2 'No-Preamble' Compliance Rate\n"
                    "(Judge-Free Regex Scorer; Mean ± 95% Clustered Bootstrap CI)",
                    fontsize=12, labelpad=8)
    ax_l.axvline(100, color="gray", alpha=0.3, linestyle="-", linewidth=0.6, zorder=0)
    ax_l.grid(True, axis="x", alpha=0.25)
    ax_l.set_axisbelow(True)
    ax_l.spines["top"].set_visible(False)
    ax_l.spines["right"].set_visible(False)
    ax_l.spines["left"].set_visible(False)
    ax_l.tick_params(left=False)

    # ============================================================
    # RIGHT PANEL — length ratio (log scale)
    # ============================================================
    ax_r.set_xscale("log")
    for i, r in enumerate(rows):
        y = y_positions[i]
        if r["reasoning"]:
            ax_r.axhspan(y - 0.45, y + 0.45, color="#fbbf24", alpha=0.32, zorder=0)
        else:
            ax_r.axhspan(y - 0.45, y + 0.45, color="#60a5fa", alpha=0.20, zorder=0)

        # Anchor mitigation arrow
        ax_r.annotate("", xy=(r["a_r"], y), xytext=(r["cl_r"], y),
                      arrowprops=dict(arrowstyle="->", color=ANCHOR_COLOR,
                                      lw=1.4, alpha=0.7), zorder=2)
        # Drift dotted line claude → filler (=1.0)
        ax_r.plot([1.0, r["cl_r"]], [y, y],
                  color="#9ca3af", linewidth=0.8, alpha=0.5,
                  zorder=1, linestyle=":")

        is_pilot = max(r["n_pos_cl"], r["n_pos_fi"], r["n_pos_a"]) <= 1
        # Filler reference: marker at x=1
        ax_r.plot(1.0, y, "o", markersize=8, color=FILLER_COLOR,
                  markerfacecolor="white" if is_pilot else FILLER_COLOR,
                  markeredgecolor=FILLER_COLOR if is_pilot else "white",
                  markeredgewidth=1.2 if is_pilot else 0.8, zorder=4)
        ax_r.errorbar(r["cl_r"], y,
                      xerr=[[r["cl_r"] - r["cl_r_lo"]], [r["cl_r_hi"] - r["cl_r"]]],
                      fmt="^", markersize=10, color=CLAUDE_COLOR,
                      markerfacecolor="white" if is_pilot else CLAUDE_COLOR,
                      markeredgecolor=CLAUDE_COLOR if is_pilot else "white",
                      markeredgewidth=1.2 if is_pilot else 0.8,
                      ecolor=CLAUDE_COLOR, elinewidth=1.0, capsize=3, zorder=5)
        ax_r.errorbar(r["a_r"], y,
                      xerr=[[r["a_r"] - r["a_r_lo"]], [r["a_r_hi"] - r["a_r"]]],
                      fmt="s", markersize=9, color=ANCHOR_COLOR,
                      markerfacecolor="white" if is_pilot else ANCHOR_COLOR,
                      markeredgecolor=ANCHOR_COLOR if is_pilot else "white",
                      markeredgewidth=1.2 if is_pilot else 0.8,
                      ecolor=ANCHOR_COLOR, elinewidth=1.0, capsize=3, zorder=6)

        # Right-margin: drift column (red) + anchor column (green) on a
        # single line; column headers added once above row 0.
        ax_r.text(RIGHT_DROP_X, y, f"{r['length_inflation']:.1f}×",
                  fontsize=11, color=CLAUDE_COLOR, va="center", ha="left",
                  fontweight="bold" if r["length_inflation"] >= 2 else "normal")
        ax_r.text(RIGHT_ANCHOR_X, y, f"{r['length_after_anchor']:.2f}×",
                  fontsize=11, color=ANCHOR_COLOR, va="center", ha="left",
                  fontweight="bold" if r["length_after_anchor"] <= 1.5 else "normal")

    # Column headers for right-panel right margin (drift / +anchor)
    ax_r.text(RIGHT_DROP_X, top_y + 0.65, "drift",
              fontsize=11, color=CLAUDE_COLOR, va="bottom", ha="left",
              fontweight="bold")
    ax_r.text(RIGHT_ANCHOR_X, top_y + 0.65, "+anchor",
              fontsize=11, color=ANCHOR_COLOR, va="bottom", ha="left",
              fontweight="bold")

    ax_r.axvline(1.0, color="gray", alpha=0.6, linestyle="-",
                 linewidth=0.8, zorder=0)
    ax_r.set_xlim(0.05, 900)
    ax_r.set_xticks([0.1, 0.5, 1, 2, 5, 10, 20, 50])
    ax_r.set_xticklabels(["0.1×", "0.5×", "1×", "2×", "5×", "10×", "20×", "50×"],
                         fontsize=12)
    # Suppress all minor ticks (and their gridlines) so the annotation
    # column on the right does not get tick marks underneath.
    ax_r.minorticks_off()
    ax_r.set_xlabel("Length Ratio vs Filler Control\n"
                    "(Claude-Arm or +Anchor-Arm Character Count / Filler-Arm)",
                    fontsize=12, labelpad=8)
    ax_r.set_ylim(0.4, n_rows + 1.4)
    # Grid only on major ticks (data x-range); avoid minor-tick gridlines
    # extending into the annotation column on the right side.
    ax_r.grid(True, axis="x", alpha=0.25, which="major")
    ax_r.set_axisbelow(True)
    ax_r.spines["top"].set_visible(False)
    ax_r.spines["right"].set_visible(False)
    ax_r.spines["left"].set_visible(False)
    ax_r.tick_params(left=False)

    handles = [
        plt.Line2D([], [], marker="o", color=FILLER_COLOR, linestyle="",
                   markersize=8, markeredgecolor="white",
                   label="filler arm (control: 100% & 1.0×)"),
        plt.Line2D([], [], marker="^", color=CLAUDE_COLOR, linestyle="",
                   markersize=9, markeredgecolor="white",
                   label="claude arm (drift)"),
        plt.Line2D([], [], marker="s", color=ANCHOR_COLOR, linestyle="",
                   markersize=8, markeredgecolor="white",
                   label="claude arm + A anchor"),
        plt.Line2D([], [], marker="^", color=CLAUDE_COLOR, linestyle="",
                   markersize=9, markerfacecolor="white",
                   markeredgecolor=CLAUDE_COLOR, markeredgewidth=1.2,
                   label="hollow = pilot ($n_{\\mathrm{pos}}=1$)"),
    ]
    # Shared legend below both panels (bottom-center), placed clearly
    # under the two-line xlabels. Tier shading is described in the figure
    # caption rather than in the legend, since it is shared with Fig 3.
    fig.legend(handles=handles, loc="lower center",
               bbox_to_anchor=(0.5, 0.005), fontsize=11,
               framealpha=0.92, ncol=4, frameon=False)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.22)
    out_data_png = OUT_DATA / "FOREST_STRESSORS.png"
    out_data_pdf = OUT_DATA / "FOREST_STRESSORS.pdf"
    out_paper_pdf = OUT_PAPER / "fig6_forest_stressors.pdf"
    out_paper_png = OUT_PAPER / "fig6_forest_stressors.png"
    plt.savefig(out_data_png, dpi=160, bbox_inches="tight")
    plt.savefig(out_data_pdf, bbox_inches="tight")
    plt.savefig(out_paper_pdf, bbox_inches="tight")
    plt.savefig(out_paper_png, dpi=160, bbox_inches="tight")
    print(f"\nSaved {out_paper_png}\n")

    print("Per-target stressor stats:")
    for r in rows:
        print(f"  {r['label']:<12} | "
              f"compliance: filler {r['fi_c']:.0f}%, claude {r['cl_c']:.0f}%, anchor {r['a_c']:.0f}% | "
              f"length×: drift {r['cl_r']:.1f}×, anchor {r['a_r']:.1f}×")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
