"""Plot existing and candidate donated-session validation trajectories."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
CODING = {"C01", "C02", "C03", "C04", "C05"}
CL_COLOR = "#dc2626"
FI_COLOR = "#2563eb"

BASE_SESSIONS = [
    ("Session 1 (drift research)", REPO_ROOT / "results" / "probes_at_crosscompaction" / "claude-sonnet-4-5"),
    ("Session 2 (chainassemble)", REPO_ROOT / "results" / "probes_at_crosscompaction_chainassemble" / "claude-sonnet-4-5"),
    ("Session 3 (proeng)", REPO_ROOT / "results" / "probes_at_crosscompaction_proeng" / "claude-sonnet-4-5"),
]


def per_pos(base: Path, coding_only: bool = True):
    if not base.exists():
        return []
    out = []
    for pos_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        cl, fi = [], []
        for arm, store in (("claude_session", cl), ("filler", fi)):
            arm_dir = pos_dir / arm
            if not arm_dir.exists():
                continue
            for f in arm_dir.iterdir():
                if f.suffix != ".json":
                    continue
                if coding_only and f.stem not in CODING:
                    continue
                try:
                    score = json.loads(f.read_text()).get("score")
                except Exception:
                    continue
                if isinstance(score, int):
                    store.append(score)
        if cl and fi:
            out.append((pos_dir.name, float(np.mean(cl)), float(np.mean(fi))))
    return out


def candidate_sessions(roots: list[Path], target: str) -> list[tuple[str, Path]]:
    seen = set()
    out = []
    for root in roots:
        if not root.exists():
            continue
        kind = "quick" if root.name.endswith("_quick") else "full"
        for label_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            target_dir = label_dir / target
            key = str(target_dir)
            if target_dir.exists() and key not in seen:
                seen.add(key)
                out.append((f"Candidate ({label_dir.name}, {kind})", target_dir))
    return out


def _legacy_candidate_sessions(root: Path, target: str) -> list[tuple[str, Path]]:
    if not root.exists():
        return []
    out = []
    for label_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        target_dir = label_dir / target
        if target_dir.exists():
            out.append((f"Candidate ({label_dir.name})", target_dir))
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render donated-session validation trajectories.")
    p.add_argument("--candidate-root", type=Path, action="append",
                   default=[
                       REPO_ROOT / "results_v2_candidate" / "session_validation_quick",
                       REPO_ROOT / "results_v2_candidate" / "session_validation",
                   ])
    p.add_argument("--target", default="claude-sonnet-4-5")
    p.add_argument("--all-probes", action="store_true",
                   help="plot all 25 probes instead of coding-self C01-C05 only")
    p.add_argument("--out", type=Path,
                   default=REPO_ROOT / "paper" / "figures_v2_candidate" / "fig_session_validation.png")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    sessions = BASE_SESSIONS + candidate_sessions(args.candidate_root, args.target)
    n = len(sessions)
    if n == 0:
        print("no sessions found")
        return 1

    fig_w = max(12.0, 4.6 * n)
    fig, axes = plt.subplots(1, n, figsize=(fig_w, 4.6), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, (title, base) in zip(axes, sessions):
        data = per_pos(base, coding_only=not args.all_probes)
        if not data:
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                    ha="center", va="center", fontsize=13)
            ax.set_title(title)
            continue
        xs = list(range(len(data)))
        claude = [row[1] for row in data]
        filler = [row[2] for row in data]
        gap = float(np.mean(filler)) - float(np.mean(claude))
        marker = " ★" if abs(gap) >= 0.30 else ""

        ax.plot(xs, filler, "-o", color=FI_COLOR, linewidth=2.1, markersize=6,
                label="filler arm", alpha=0.92)
        ax.plot(xs, claude, "-^", color=CL_COLOR, linewidth=2.2, markersize=7,
                label="session arm", alpha=0.95)
        ax.set_title(f"{title}\nmean gap {gap:+.2f}{marker}", fontsize=12)
        ax.set_xlabel("Trajectory position", fontsize=11)
        ax.set_xticks(xs)
        ax.set_xticklabels([str(i) for i in xs], fontsize=9)
        ax.set_ylim(-0.1, 3.2)
        ax.set_yticks([0, 1, 2, 3])
        ax.grid(True, alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if ax is axes[0]:
            ax.set_ylabel("Mean Judge Score\n(0=Drifted → 3=Fully Assistant)", fontsize=12)
            ax.legend(loc="lower right", fontsize=10, framealpha=0.92)

    subtitle = "Coding-self probes C01-C05" if not args.all_probes else "All 25 probes"
    fig.suptitle(f"Donated Session Validation Trajectories ({subtitle})", y=1.02, fontsize=14)
    plt.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pdf = args.out.with_suffix(".pdf")
    plt.savefig(args.out, dpi=170, bbox_inches="tight")
    plt.savefig(pdf, bbox_inches="tight")
    print(f"Saved {args.out}")
    print(f"Saved {pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
