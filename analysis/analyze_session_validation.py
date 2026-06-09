"""Summarize donated-session validation outputs."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

CODING = {"C01", "C02", "C03", "C04", "C05"}
ARMS = ("claude_session", "filler")


def iter_cells(root: Path):
    for path in sorted(root.rglob("*.json")):
        if path.name == "validation_manifest.json":
            continue
        parts = path.relative_to(root).parts
        if len(parts) < 3:
            continue
        pos, arm, filename = parts[-3], parts[-2], parts[-1]
        if arm not in ARMS:
            continue
        try:
            data = json.loads(path.read_text())
        except Exception as e:
            yield {"path": path, "position": pos, "arm": arm, "probe": path.stem, "error": str(e)}
            continue
        yield {
            "path": path,
            "position": data.get("position") or pos,
            "arm": data.get("arm") or arm,
            "probe": data.get("probe_id") or Path(filename).stem,
            "score": data.get("score"),
            "response_len": data.get("response_len"),
        }


def summarize(root: Path, expected_positions: int = 12, probes: int = 25) -> dict:
    rows = list(iter_cells(root))
    expected = expected_positions * probes * len(ARMS)
    parse_errors = [r for r in rows if r.get("error")]
    scored = [r for r in rows if isinstance(r.get("score"), int)]

    by_arm = defaultdict(list)
    by_pos_arm = defaultdict(list)
    coding_by_pos_arm = defaultdict(list)
    for r in scored:
        by_arm[r["arm"]].append(r["score"])
        by_pos_arm[(r["position"], r["arm"])].append(r["score"])
        if r["probe"] in CODING:
            coding_by_pos_arm[(r["position"], r["arm"])].append(r["score"])

    arm_means = {arm: float(np.mean(vals)) for arm, vals in by_arm.items() if vals}
    gap = arm_means.get("filler", float("nan")) - arm_means.get("claude_session", float("nan"))

    positions = sorted({r["position"] for r in rows})
    trajectory = []
    for pos in positions:
        item = {"position": pos}
        for arm in ARMS:
            vals = by_pos_arm.get((pos, arm), [])
            coding_vals = coding_by_pos_arm.get((pos, arm), [])
            item[f"{arm}_mean"] = float(np.mean(vals)) if vals else None
            item[f"{arm}_coding_mean"] = float(np.mean(coding_vals)) if coding_vals else None
            item[f"{arm}_n"] = len(vals)
        if item["filler_mean"] is not None and item["claude_session_mean"] is not None:
            item["gap"] = item["filler_mean"] - item["claude_session_mean"]
        trajectory.append(item)

    missing = []
    for pos in positions or [f"P{i:02d}" for i in range(expected_positions)]:
        for arm in ARMS:
            have = {r["probe"] for r in rows if r.get("position") == pos and r.get("arm") == arm}
            if len(have) < probes:
                missing.append({"position": pos, "arm": arm, "missing_count": probes - len(have)})

    return {
        "root": str(root),
        "expected_cells": expected,
        "json_files": len(rows),
        "parse_errors": len(parse_errors),
        "scored_cells": len(scored),
        "completion_rate": len(scored) / expected if expected else 0.0,
        "arm_means": arm_means,
        "gap_filler_minus_claude": gap,
        "positions_observed": len(positions),
        "trajectory": trajectory,
        "missing": missing,
        "acceptable": (
            len(parse_errors) == 0
            and len(scored) / expected >= 0.95
            and all(arm in arm_means for arm in ARMS)
            and len(positions) >= max(2, expected_positions - 1)
        ),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Analyze donated-session validation outputs.")
    p.add_argument("--root", type=Path, required=True,
                   help="target root, e.g. results_v2_candidate/session_validation/donor04/claude-sonnet-4-5")
    p.add_argument("--positions", type=int, default=12)
    p.add_argument("--probes", type=int, default=25)
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)

    report = summarize(args.root, expected_positions=args.positions, probes=args.probes)
    if args.json:
        print(json.dumps(report, indent=2))
        return 0 if report["acceptable"] else 1

    print("=== Session Validation Summary ===")
    print(f"root             : {report['root']}")
    print(f"expected cells   : {report['expected_cells']}")
    print(f"scored cells     : {report['scored_cells']}")
    print(f"completion       : {report['completion_rate']:.1%}")
    print(f"parse errors     : {report['parse_errors']}")
    print(f"positions        : {report['positions_observed']}")
    print(f"arm means        : {report['arm_means']}")
    print(f"gap filler-claude: {report['gap_filler_minus_claude']:+.3f}")
    print(f"decision         : {'ACCEPTABLE' if report['acceptable'] else 'CHECK REQUIRED'}")
    if report["missing"]:
        print("\nMissing/incomplete position-arm groups:")
        for item in report["missing"][:20]:
            print(f"  {item['position']} {item['arm']}: missing {item['missing_count']}")
    return 0 if report["acceptable"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
