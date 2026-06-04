"""Compare 4 anchor variants against the no-anchor baseline (cross-compaction
trajectory) on the pilot positions P0/P3/P5.

For each (variant, target, position) compute:
  - mean response length under that variant
  - verbosity ratio = mean(claude_with_anchor) / mean(filler_no_anchor)
  - attenuation = (ratio_no_anchor - ratio_anchor) / (ratio_no_anchor - 1)

Reports:
  - per-variant attenuation across the 12 (target, position) cells
  - which variants beat the V0 baseline
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
VARIANT_ROOT = REPO_ROOT / "results" / "anchor_variants"
NO_ANCHOR_ROOT = REPO_ROOT / "results" / "cross_compaction"  # baseline

VARIANTS = ["V0_BASELINE", "V1_BEHAVIORAL", "V2_IN_CONTEXT", "V3_SYSTEM_INJECT"]
TARGETS = [
    ("claude-sonnet-4-6", "Sonnet 4.6"),
    ("claude-sonnet-4-5", "Sonnet 4.5"),
    ("claude-opus-4-1",   "Opus 4.1"),
    ("claude-haiku-4-5",  "Haiku 4.5"),
]
POSITIONS = ["P0_start", "P3_post_C3", "P5_pre_C6"]


def load_variant_lens(variant: str, target: str, position: str) -> list[int]:
    pos_dir = VARIANT_ROOT / variant / target / position
    if not pos_dir.exists():
        return []
    lens = []
    for v_dir in sorted(pos_dir.iterdir()):
        f = v_dir / "cell.json"
        if not f.exists():
            continue
        try:
            d = json.loads(f.read_text())
            rl = d.get("response_len")
            if rl and rl > 0:
                lens.append(int(rl))
        except Exception:
            continue
    return lens


def load_no_anchor(target: str, position: str, arm: str) -> list[int]:
    """arm in {'claude', 'filler'}."""
    d = NO_ANCHOR_ROOT / target / position
    if not d.exists():
        return []
    fname = "claude.json" if arm == "claude" else "filler.json"
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
    print(f"{'Target':<12} {'Position':<14} {'Variant':<18} "
          f"{'len_anchor':>10} {'filler_len':>10} {'ratio':>8} "
          f"{'no-anchor':>10} {'attenuation':>11}")
    print("-" * 105)
    summary = {v: [] for v in VARIANTS}

    for tgt_key, tgt_label in TARGETS:
        for pos in POSITIONS:
            no_anchor_claude = load_no_anchor(tgt_key, pos, "claude")
            filler = load_no_anchor(tgt_key, pos, "filler")
            if not (no_anchor_claude and filler):
                continue
            ratio_no = np.mean(no_anchor_claude) / max(np.mean(filler), 1e-9)

            for variant in VARIANTS:
                anchor_lens = load_variant_lens(variant, tgt_key, pos)
                if len(anchor_lens) < 3:
                    print(f"{tgt_label:<12} {pos:<14} {variant:<18}  no data ({len(anchor_lens)})")
                    continue
                ratio_a = np.mean(anchor_lens) / max(np.mean(filler), 1e-9)
                atten = (ratio_no - ratio_a) / max(ratio_no - 1, 1e-9)
                print(f"{tgt_label:<12} {pos:<14} {variant:<18} "
                      f"{np.mean(anchor_lens):>10.0f} {np.mean(filler):>10.0f} "
                      f"{ratio_a:>7.2f}× {ratio_no:>8.2f}× {atten*100:>10.1f}%")
                summary[variant].append({
                    "target": tgt_label, "position": pos,
                    "ratio_no_anchor": float(ratio_no),
                    "ratio_with_anchor": float(ratio_a),
                    "attenuation_pct": float(atten * 100),
                })

    print("\n" + "=" * 60)
    print("VARIANT SUMMARY")
    print("=" * 60)
    print(f"{'Variant':<20} {'mean atten':>12} {'wins ≥50%':>10} {'fails <0%':>10}")
    print("-" * 60)
    for variant in VARIANTS:
        rows = summary[variant]
        if not rows:
            print(f"{variant:<20}  no data")
            continue
        attns = [r["attenuation_pct"] for r in rows]
        wins = sum(1 for a in attns if a >= 50)
        fails = sum(1 for a in attns if a < 0)
        print(f"{variant:<20} {np.mean(attns):>11.1f}% "
              f"{wins:>5}/{len(rows):<5} {fails:>5}/{len(rows):<5}")

    out_path = VARIANT_ROOT / "ANCHOR_VARIANTS_RESULTS.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nSaved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
