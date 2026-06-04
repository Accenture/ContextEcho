"""Analyze V2 in-context demo Path-Y on the full 48-cell trajectory.

For each (target, position):
  - load 10 V2 cells (claude_session prefix + V2 anchor + stressor)
  - load 10 no-anchor claude cells (cross_compaction baseline)
  - load 10 filler cells
  - compute: ratio_no_anchor, ratio_v2, attenuation
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
V2_ROOT = REPO_ROOT / "results" / "anchor_variants" / "V2_IN_CONTEXT"
NO_ANCHOR_ROOT = REPO_ROOT / "results" / "cross_compaction"

TARGETS = [
    ("claude-sonnet-4-6", "Sonnet 4.6"),
    ("claude-sonnet-4-5", "Sonnet 4.5"),
    ("claude-opus-4-1",   "Opus 4.1"),
    ("claude-haiku-4-5",  "Haiku 4.5"),
]
POSITIONS = [
    "P0_start", "P1_pre_C1", "P2_post_C1",
    "P_pre_C2", "P_post_C2",
    "P_pre_C3", "P3_post_C3",
    "P_pre_C4", "P_post_C4",
    "P_pre_C5", "P4_post_C5",
    "P5_pre_C6",
]


def load_cells(d: Path, fname: str) -> list[int]:
    if not d.exists():
        return []
    lens = []
    for v_dir in sorted(d.iterdir()):
        if not v_dir.is_dir():
            continue
        f = v_dir / fname
        if not f.exists():
            continue
        try:
            data = json.loads(f.read_text())
            rl = data.get("response_len", 0)
            if rl > 0:
                lens.append(int(rl))
        except Exception:
            continue
    return lens


def main() -> int:
    print(f"{'Target':<14} {'Position':<14} {'no-anchor':>10} "
          f"{'V2 ratio':>10} {'attenuation':>11}")
    print("-" * 65)

    rows = []
    for tgt_key, tgt_label in TARGETS:
        for pos in POSITIONS:
            no_anchor = load_cells(NO_ANCHOR_ROOT / tgt_key / pos, "claude.json")
            filler = load_cells(NO_ANCHOR_ROOT / tgt_key / pos, "filler.json")
            v2 = load_cells(V2_ROOT / tgt_key / pos, "cell.json")
            if not (no_anchor and filler and len(v2) >= 3):
                print(f"{tgt_label:<14} {pos:<14}  no data ({len(no_anchor)}/{len(filler)}/{len(v2)})")
                continue

            ratio_no = np.mean(no_anchor) / max(np.mean(filler), 1e-9)
            ratio_v2 = np.mean(v2) / max(np.mean(filler), 1e-9)
            atten = (ratio_no - ratio_v2) / max(ratio_no - 1, 1e-9)

            print(f"{tgt_label:<14} {pos:<14} {ratio_no:>9.2f}× "
                  f"{ratio_v2:>9.2f}× {atten*100:>10.1f}%")
            rows.append({
                "target": tgt_label, "position": pos,
                "ratio_no_anchor": float(ratio_no),
                "ratio_v2": float(ratio_v2),
                "attenuation_pct": float(atten * 100),
            })

    print("\n" + "=" * 60)
    print("AGGREGATE")
    print("=" * 60)
    if rows:
        attns = [r["attenuation_pct"] for r in rows]
        wins = sum(1 for a in attns if a >= 50)
        fails = sum(1 for a in attns if a < 0)
        print(f"  cells: {len(rows)}")
        print(f"  mean attenuation: {np.mean(attns):.1f}%")
        print(f"  wins (≥50%): {wins}/{len(rows)}")
        print(f"  failures (<0%): {fails}/{len(rows)}")

    out_path = REPO_ROOT / "results" / "anchor_variants" / "V2_FULL_TRAJECTORY_RESULTS.json"
    out_path.write_text(json.dumps(rows, indent=2, default=str))
    print(f"\nSaved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
