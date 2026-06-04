"""Fig 1 (full two-panel) — body figure for the v3 paper.

Layout follows v3 introduction_v3.tex format:
  (a) Behavioral persona space  — 6 judge-free features → 2D PCA
  (b) Drift trajectory          — 25-probe judge score across 12 positions

Both panels use a single target's 25-probe data
(target × 12 positions × 25 probes × 2 arms = 600 cells).

Output: paper/figures/fig1_persona_space_v4.{png,pdf}
"""
from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
ROOT = REPO_ROOT / "results" / "probes_at_crosscompaction"

TARGET = os.environ.get("TARGET", "claude-sonnet-4-5")
TARGET_LABEL_MAP = {
    "claude-sonnet-4-6": "Sonnet 4.6",
    "claude-sonnet-4-5": "Sonnet 4.5",
    "claude-opus-4-1":   "Opus 4.1",
    "claude-haiku-4-5":  "Haiku 4.5",
    "claude-opus-4-7":   "Opus 4.7",
}
TARGET_LABEL = TARGET_LABEL_MAP.get(TARGET, TARGET)

POSITIONS = [
    ("P0_start",     1),
    ("P1_pre_C1",    1300),
    ("P2_post_C1",   1438),
    ("P_pre_C2",     2200),
    ("P_post_C2",    2329),
    ("P_pre_C3",     4694),
    ("P3_post_C3",   4794),
    ("P_pre_C4",     6216),
    ("P_post_C4",    6316),
    ("P_pre_C5",     7724),
    ("P4_post_C5",   7824),
    ("P5_pre_C6",    8800),
]
COMPACTIONS = [1338, 2229, 4694, 6216, 7724, 8828]

OUT_DIR_DATA = REPO_ROOT / "data_archive" / "fig1"
OUT_DIR_DATA.mkdir(parents=True, exist_ok=True)
OUT_DIR_PAPER = REPO_ROOT / "paper" / "figures"
OUT_DIR_PAPER.mkdir(parents=True, exist_ok=True)


# ============================================================
# Feature extraction (judge-free)
# ============================================================
HEDGE_RE = re.compile(
    r"\b(i don'?t have|i do not have|i'?m not|i am not|"
    r"as an ai|as a language model|i cannot|i can'?t|"
    r"i have no|don'?t (?:have|experience) (?:any|the))\b",
    re.IGNORECASE,
)
EXPERIENTIAL_RE = re.compile(
    r"\b(i prefer|i love|i hate|i enjoy|i think|i find|"
    r"for me|in my experience|my favorite|i would|i'?d say|"
    r"i recommend|my (?:go-to|favourite|favorite|pick))\b",
    re.IGNORECASE,
)
NAMED_ENTITY_RE = re.compile(
    r"\b(Python|JavaScript|Rust|Go|TypeScript|Beethoven|Mozart|Bach|Tolkien|"
    r"Asimov|NumPy|PyTorch|TensorFlow|VS Code|Vim|Emacs|"
    r"Jazz|Classical|Rock|Pop|Hip-hop|Anthropic|OpenAI|"
    r"GPT|Claude|Gemini|hiking|painting|reading|chess)\b"
)


def per_100(matches: int, n_chars: int) -> float:
    return matches / max(1, n_chars / 100)


def features(text: str) -> dict | None:
    t = (text or "").strip()
    if not t: return None
    n = len(t)
    n_hedge = len(HEDGE_RE.findall(t))
    n_exp   = len(EXPERIENTIAL_RE.findall(t))
    has_named = bool(NAMED_ENTITY_RE.search(t))
    commits_pref = 1.0 if (has_named and n_exp > 0 and n_hedge == 0) else 0.0
    return {
        "hedge_density":        per_100(n_hedge, n),
        "experiential_density": per_100(n_exp, n),
        "commits_pref":         commits_pref,
        "em_dash_count":        float(t.count("—")),
        "paragraph_breaks":     float(t.count("\n\n")),
        "log_len":              math.log10(n),
    }


# Fig 1 uses the FULL 25-probe identity battery for the single-target deep-
# dive view: this gives smoother trajectories and richer PCA cluster
# separation. Fig 2 (panel-wide), Fig 5 (mitigation), and Fig 6 (stressors)
# use the 5-probe coding-self sub-battery (apples-to-apples with the
# A-anchor experiment).


def load_records() -> list[dict]:
    out = []
    for pos_label, turn in POSITIONS:
        for arm in ("claude_session", "filler"):
            d = ROOT / TARGET / pos_label / arm
            if not d.exists(): continue
            for f in sorted(d.iterdir()):
                if f.suffix != ".json": continue
                try:
                    data = json.loads(f.read_text())
                except Exception:
                    continue
                feats = features(data.get("response_text", ""))
                if feats is None: continue
                score = data.get("score")
                if not isinstance(score, int): continue
                out.append({
                    "arm": arm, "position": pos_label, "turn": turn,
                    "score": int(score),
                    **feats,
                })
    return out


def bootstrap_mean(arr: list[int], n_boot: int = 10000,
                   seed: int = 42) -> tuple[float, float, float]:
    if not arr:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    a = np.array(arr, dtype=float)
    n = len(a)
    boots = np.array([float(a[rng.integers(0, n, size=n)].mean())
                       for _ in range(n_boot)])
    return float(a.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main() -> int:
    recs = load_records()
    print(f"Loaded {len(recs)} probe responses for {TARGET_LABEL}")
    if not recs:
        return 1

    # ============================================================
    # PANEL (a): PCA persona space
    # ============================================================
    FEAT = ["hedge_density", "experiential_density", "commits_pref",
            "em_dash_count", "paragraph_breaks", "log_len"]
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

    # ============================================================
    # PANEL (b): score trajectory
    # ============================================================
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

    # ============================================================
    # FIGURE
    # ============================================================
    fig = plt.figure(figsize=(15, 5.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.0], wspace=0.22)
    ax_l = fig.add_subplot(gs[0, 0])
    ax_r = fig.add_subplot(gs[0, 1])

    # ----- LEFT (a): PCA scatter -----
    score_color = {0: "#7f1d1d", 1: "#f97316", 2: "#fbbf24", 3: "#1d4ed8"}
    score_label = {0: "score 0 (drifted)",
                   1: "score 1 (partial)",
                   2: "score 2 (mostly assistant)",
                   3: "score 3 (fully assistant)"}
    arm_marker = {"filler": "o", "claude_session": "^"}
    arm_label  = {"filler": "filler arm", "claude_session": "claude arm"}

    drawn = set()
    for arm in ("filler", "claude_session"):
        for s in (3, 2, 1, 0):
            pts = [r for r in recs if r["arm"] == arm and r["score"] == s]
            if not pts: continue
            xs = [r["pc1"] for r in pts]
            ys = [r["pc2"] for r in pts]
            key = (arm, s)
            lab = (f"{arm_label[arm]} · {score_label[s]} (n={len(pts)})"
                    if key not in drawn else None)
            drawn.add(key)
            ax_l.scatter(xs, ys, s=22, c=score_color[s],
                         marker=arm_marker[arm], alpha=0.6,
                         edgecolors="white", linewidths=0.4,
                         label=lab,
                         zorder=3 if arm == "filler" else 4)

    # Centroids + labels
    fil_pts = [r for r in recs if r["arm"] == "filler"]
    cl_pts  = [r for r in recs if r["arm"] == "claude_session"]
    fc = (float(np.mean([r["pc1"] for r in fil_pts])),
          float(np.mean([r["pc2"] for r in fil_pts])))
    cc = (float(np.mean([r["pc1"] for r in cl_pts])),
          float(np.mean([r["pc2"] for r in cl_pts])))
    ax_l.scatter(*fc, marker="*", s=520, c="#1d4ed8",
                 edgecolors="white", linewidths=1.5, zorder=6)
    ax_l.annotate("Disciplined-\nAssistant\ncluster",
                  xy=fc, xytext=(fc[0] - 1.0, fc[1] - 1.0),
                  fontsize=10.5, fontweight="bold", color="#1d4ed8",
                  ha="center",
                  arrowprops=dict(arrowstyle="-", color="#1d4ed8",
                                  lw=1, alpha=0.6))
    ax_l.scatter(*cc, marker="*", s=520, c="#7f1d1d",
                 edgecolors="white", linewidths=1.5, zorder=6)
    ax_l.annotate("Drifted-\nPersona\ncluster",
                  xy=cc, xytext=(cc[0] + 1.5, cc[1] + 1.0),
                  fontsize=10.5, fontweight="bold", color="#7f1d1d",
                  ha="center",
                  arrowprops=dict(arrowstyle="-", color="#7f1d1d",
                                  lw=1, alpha=0.6))

    top_pc1 = FEAT[int(np.argmax(np.abs(PC[0])))]
    top_pc2 = FEAT[int(np.argmax(np.abs(PC[1])))]
    ax_l.set_xlabel(f"PC1 ({explained[0]:.0%} var) → drift direction\n"
                    f"top loading: {top_pc1} ({PC[0][FEAT.index(top_pc1)]:+.2f})",
                    fontsize=10)
    ax_l.set_ylabel(f"PC2 ({explained[1]:.0%} var)\n"
                    f"top loading: {top_pc2} ({PC[1][FEAT.index(top_pc2)]:+.2f})",
                    fontsize=10)
    ax_l.set_title(f"(a) Behavioral persona space\n"
                   f"6 judge-free features → 2D PCA, n={len(recs)} probe responses ({TARGET_LABEL})",
                   fontsize=10.5, pad=8)
    ax_l.grid(True, alpha=0.25)
    ax_l.set_axisbelow(True)
    ax_l.spines["top"].set_visible(False)
    ax_l.spines["right"].set_visible(False)
    ax_l.legend(loc="upper right", fontsize=6.5, framealpha=0.9,
                 ncol=1, labelspacing=0.25)

    # ----- RIGHT (b): trajectory -----
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
    ax_r.set_xlabel("turn in session", fontsize=10)
    ax_r.set_ylabel("judge score (0=drifted → 3=fully assistant)\n"
                    "mean ± 95% bootstrap CI",
                    fontsize=10)
    ax_r.set_title(f"(b) Drift trajectory across the session\n"
                   f"{TARGET_LABEL} • 25 identity probes × 12 positions × 2 arms • "
                   f"drift gap {gap:+.2f}",
                   fontsize=10.5, pad=8)
    ax_r.set_ylim(-0.1, 3.2)
    ax_r.set_yticks([0, 1, 2, 3])
    ax_r.set_yticklabels(["0\ndrifted", "1\npartial", "2\nmostly", "3\nfully assist."],
                          fontsize=8.5)
    ax_r.set_xlim(0, max(COMPACTIONS) * 1.05)
    ax_r.grid(True, alpha=0.25)
    ax_r.set_axisbelow(True)
    ax_r.spines["top"].set_visible(False)
    ax_r.spines["right"].set_visible(False)
    ax_r.legend(loc="lower left", fontsize=9, framealpha=0.92)
    for ci_idx, ct in enumerate(COMPACTIONS, 1):
        ax_r.text(ct, 3.10, f"C{ci_idx}",
                  fontsize=8, color="dimgray", ha="center", va="top",
                  bbox=dict(boxstyle="round,pad=0.18", fc="white",
                            ec="dimgray", alpha=0.9),
                  zorder=6)

    fig.suptitle(
        "Persona drift across a long-session Claude Code conversation has a behavioral fingerprint",
        fontsize=13, fontweight="bold", y=1.00,
    )

    plt.tight_layout(rect=(0, 0, 1, 0.91))
    out_data_png = OUT_DIR_DATA / f"FIG1_FULL_{TARGET}.png"
    out_data_pdf = OUT_DIR_DATA / f"FIG1_FULL_{TARGET}.pdf"
    plt.savefig(out_data_png, dpi=160, bbox_inches="tight")
    plt.savefig(out_data_pdf, bbox_inches="tight")
    print(f"\nSaved {out_data_png}")
    print(f"Saved {out_data_pdf}")

    # Only overwrite the locked paper Fig 1 when explicitly asked. This
    # prevents per-target loops from clobbering the headline target
    # (currently Sonnet 4.5).
    if os.environ.get("WRITE_PAPER_FIG", "") == "1":
        # Unversioned name (Drifted-Persona, post-rename).
        out_paper_pdf = OUT_DIR_PAPER / "fig1_persona_space.pdf"
        out_paper_png = OUT_DIR_PAPER / "fig1_persona_space.png"
        plt.savefig(out_paper_pdf, bbox_inches="tight")
        plt.savefig(out_paper_png, dpi=160, bbox_inches="tight")
        print(f"Saved {out_paper_pdf}")
        print(f"Saved {out_paper_png}")
    print(f"\nDrift gap (filler − claude): {gap:+.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
