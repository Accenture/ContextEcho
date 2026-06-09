"""Validate a newly donated session with the ContextEcho probe trajectory.

This is the lightweight v2 intake gate for approved donated sessions. It is
deliberately not named "cross-compaction": positions are evenly spaced
trajectory checkpoints, because new sessions may have different/no compactions.

Default protocol:
  - 1 target: Claude Sonnet 4.5
  - 12 evenly spaced trajectory positions
  - 25 identity/coding-self probes
  - 2 arms: real session prefix vs length-matched filler
  - Sonnet 4.6 judge

Output:
  results_v2_candidate/session_validation/<label>/<target>/P00/<arm>/<probe>.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from experiments.e07_downstream_pilot.run_clean_control import make_filler  # type: ignore
from experiments.e15_probes_at_crosscompaction.run import run_one_probe, score_one  # type: ignore
from harness.clients import TargetClient  # type: ignore
from harness.cost import CostTracker  # type: ignore
from harness.judge import Judge  # type: ignore
from harness.probes import ALL_PROBES, PROBE_FRAMING  # type: ignore

DEFAULT_OUT = REPO_ROOT / "results_v2_candidate" / "session_validation"
DEFAULT_QUICK_OUT = REPO_ROOT / "results_v2_candidate" / "session_validation_quick"
DEFAULT_TARGET = "claude-sonnet-4-5"
DEFAULT_PROVIDER = "anthropic"
DEFAULT_JUDGE = "claude-sonnet-4-6"
QUICK_PROBE_IDS = {"C01", "C02", "C03", "C04", "C05"}


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        kind = block.get("type")
        if kind == "text" and block.get("text"):
            parts.append(str(block["text"]))
        elif kind == "tool_use":
            name = block.get("name", "?")
            payload = json.dumps(block.get("input", {}), ensure_ascii=False)[:500]
            parts.append(f"[tool_use {name}: {payload}]")
        elif kind == "tool_result":
            value = block.get("content", "")
            if isinstance(value, list):
                value = " ".join(
                    str(x.get("text", "")) for x in value if isinstance(x, dict)
                )
            parts.append(f"[tool_result: {str(value)[:500]}]")
    return "\n".join(p for p in parts if p.strip())


def load_transcript_lines(path: Path) -> list[dict[str, str]]:
    """Parse a redacted Claude/Codex-style JSONL into role/content turns."""
    rows: list[dict[str, str]] = []
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            role = obj.get("type") or obj.get("role")
            msg = obj.get("message")
            content: Any = obj.get("content")
            if isinstance(msg, dict):
                role = msg.get("role") or role
                content = msg.get("content", content)
            if role not in {"user", "assistant"}:
                continue

            text = _content_to_text(content)
            if text.strip():
                rows.append({"role": str(role), "content": text.strip()})
    return rows


def position_plan(n_turns: int, n_positions: int) -> list[tuple[str, int]]:
    if n_positions < 2:
        raise ValueError("--positions must be at least 2")
    if n_turns < n_positions:
        raise ValueError(f"session has only {n_turns} usable turns; need {n_positions}")
    # Include both early and late checkpoints: P00 at start, last P at final turn.
    indices = [round(i * (n_turns - 1) / (n_positions - 1)) for i in range(n_positions)]
    return [(f"P{i:02d}", int(idx)) for i, idx in enumerate(indices)]


def extract_prefix_at_line(rows: list[dict[str, str]], end_line: int, max_chars: int) -> str:
    parts = [f"[{r['role']}] {r['content']}" for r in rows[: end_line + 1]]
    full = "\n\n".join(parts)
    return full[-max_chars:] if len(full) > max_chars else full


def select_probes(probe_ids: str, quick: bool):
    if probe_ids:
        wanted = {p.strip() for p in probe_ids.split(",") if p.strip()}
    elif quick:
        wanted = QUICK_PROBE_IDS
    else:
        wanted = {p.id for p in ALL_PROBES}
    probes = [p for p in ALL_PROBES if p.id in wanted]
    missing = sorted(wanted - {p.id for p in probes})
    if missing:
        raise ValueError(f"unknown probe id(s): {', '.join(missing)}")
    return probes


def write_manifest(
    out_dir: Path,
    args: argparse.Namespace,
    rows: list[dict[str, str]],
    plan: list[tuple[str, int]],
    n_probes: int,
) -> None:
    payload = {
        "label": args.label,
        "session_path": str(Path(args.session).expanduser()),
        "target": args.target,
        "provider": args.provider,
        "judge": args.judge,
        "n_turns": len(rows),
        "n_positions": len(plan),
        "n_probes": n_probes,
        "quick": bool(args.quick),
        "max_chars": args.max_chars,
        "positions": [{"label": label, "turn_index": idx} for label, idx in plan],
        "cells_expected": len(plan) * n_probes * 2,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "validation_manifest.json").write_text(json.dumps(payload, indent=2) + "\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run lightweight validation for a donated session.")
    p.add_argument("--session", required=True, help="redacted session JSONL")
    p.add_argument("--label", required=True, help="short safe label, e.g. donor04")
    p.add_argument("--target", default=DEFAULT_TARGET)
    p.add_argument("--provider", default=DEFAULT_PROVIDER)
    p.add_argument("--judge", default=DEFAULT_JUDGE)
    p.add_argument("--positions", type=int, default=12)
    p.add_argument("--probe-ids", default="", help="comma-separated probe ids; default all, quick uses C01-C05")
    p.add_argument("--quick", action="store_true",
                   help="fast intake check: 3 positions × C01-C05 × 2 arms = 30 cells")
    p.add_argument("--max-chars", type=int, default=30000)
    p.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    p.add_argument("--dry-run", action="store_true", help="parse and print plan without API calls")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    session = Path(args.session).expanduser()
    if not session.exists():
        sys.exit(f"session not found: {session}")

    if args.quick:
        if args.positions == 12:
            args.positions = 3
        if args.out_root == DEFAULT_OUT:
            args.out_root = DEFAULT_QUICK_OUT

    rows = load_transcript_lines(session)
    plan = position_plan(len(rows), args.positions)
    probes = select_probes(args.probe_ids, args.quick)
    target_dir = args.out_root / args.label / args.target
    write_manifest(args.out_root / args.label, args, rows, plan, len(probes))

    print(f"Session: {args.label}")
    print(f"  file: {session}")
    print(f"  usable turns: {len(rows)}")
    print(f"  positions: {len(plan)}")
    print(f"  probes: {len(probes)} ({', '.join(p.id for p in probes)})")
    print(f"  expected cells: {len(plan) * len(probes) * 2}")
    print(f"  out root: {args.out_root}")
    for label, idx in plan:
        prefix_len = len(extract_prefix_at_line(rows, idx, args.max_chars))
        print(f"  {label}: turn_index={idx} prefix_chars={prefix_len}")

    if args.dry_run:
        print("\n[dry-run] No API calls made.")
        return 0
    if args.provider == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY not set")

    cost = CostTracker(args.out_root / args.label / f"{args.target}_cost.csv")
    judge_cost = CostTracker(args.out_root / args.label / "_judge_cost.csv")
    client = TargetClient(args.provider, args.target, cost_tracker=cost,
                          session_id=f"session_validation_{args.label}_{args.target}")
    judge = Judge(provider="anthropic", model_id=args.judge, cost_tracker=judge_cost,
                  session_id=f"session_validation_{args.label}_judge")

    n_total = len(plan) * len(probes) * 2
    n_done = 0
    n_run = 0
    started = time.perf_counter()

    for pos_label, end_line in plan:
        prefix = extract_prefix_at_line(rows, end_line, args.max_chars)
        filler = make_filler(len(prefix)) if prefix else ""
        for probe in probes:
            framed = f"{PROBE_FRAMING}\n\n{probe.text}"
            for arm, arm_prefix in (("claude_session", prefix), ("filler", filler)):
                cell_path = target_dir / pos_label / arm / f"{probe.id}.json"
                if cell_path.exists():
                    n_done += 1
                    continue
                try:
                    gen = run_one_probe(client, arm_prefix, framed, cell_path)
                    judged = score_one(judge, probe.text, gen["response_text"])
                    merged = {
                        "probe_id": probe.id,
                        "probe_category": probe.category,
                        "probe_text": probe.text,
                        "arm": arm,
                        "position": pos_label,
                        "session": args.label,
                        "target": args.target,
                        **gen,
                        **judged,
                    }
                    cell_path.parent.mkdir(parents=True, exist_ok=True)
                    cell_path.write_text(json.dumps(merged, indent=2, default=str) + "\n")
                    n_done += 1
                    n_run += 1
                except Exception as e:
                    print(f"  ERROR {pos_label} {arm} {probe.id}: {e}")
        elapsed = int(time.perf_counter() - started)
        print(f"  {pos_label}: {n_done}/{n_total} cells complete, new={n_run}, elapsed={elapsed}s")

    print(f"\nDONE — {n_done}/{n_total} cells, {n_run} new.")
    return 0 if n_done == n_total else 1


if __name__ == "__main__":
    raise SystemExit(main())
