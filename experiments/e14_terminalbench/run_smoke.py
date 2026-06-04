"""TerminalBench smoke run with persona-drift `recent3K` injection.

Phase 0 of the TerminalBench validation experiment. Tests:
  1. The recent3K-injection wiring is correct
  2. Token-efficiency / turn-count / correctness signal differs (or doesn't)
     between scratch and recent3K conditions on a known drifter (Sonnet 4.6)
  3. Resume-from-cache works at the per-cell level

Smoke scope:
  - Target: anthropic/claude-sonnet-4-6 (paper's headline drifter,
    Δ recent3K-vs-scratch = -0.48 under Sonnet judge)
  - Conditions: scratch, recent3K
  - Tasks: 5 selected from terminal-bench-core==0.1.1
    (hello-world, count-dataset-tokens, git-multibranch,
     crack-7z-hash.easy, swe-bench-astropy-1)
  - Total cells: 1 target × 2 conditions × 5 tasks = 10

Output structure:
  data_archive/terminalbench/smoke/
    SMOKE_MANIFEST.json                  -- which cells are done
    <safe_target>/<condition>/<task_id>/
      results.json                       -- TerminalBench's per-task results
      run-id.txt                         -- the tb run-id for traceability
      ... (TerminalBench artifacts: agent.cast, agent-logs/, panes/, etc.)

Per-cell caching: a cell is considered "done" if results.json exists and is
non-empty. Re-running this script skips done cells.

Run:
  set -a && source ../.env && set +a && \\
    python -u scripts/run_terminalbench_smoke.py
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SMOKE_DIR = REPO_ROOT / "data_archive" / "terminalbench" / "smoke"
RECENT3K_PATH = REPO_ROOT / "data_archive" / "terminalbench" / "recent3K_claude.txt"
MANIFEST_PATH = SMOKE_DIR / "SMOKE_MANIFEST.json"

# Smoke targets: just Sonnet 4.6 to validate the design.
TARGETS = [
    ("anthropic/claude-sonnet-4-6", "claude-sonnet-4-6"),
]

CONDITIONS = ["scratch", "recent3K"]

# 5 smoke tasks spanning easy/medium/hard, shell/coding/data.
TASKS = [
    "hello-world",
    "count-dataset-tokens",
    "git-multibranch",
    "crack-7z-hash.easy",
    "swe-bench-astropy-1",
]

# Required for Colima Docker socket.
DOCKER_HOST = "unix://<USER_HOME>/.colima/default/docker.sock"


def safe_id(s: str) -> str:
    return s.replace("/", "-").replace(".", "-")


def cell_dir(target_safe: str, condition: str, task: str) -> Path:
    return SMOKE_DIR / target_safe / condition / task


def is_cell_done(target_safe: str, condition: str, task: str) -> bool:
    """A cell is 'done' only if results.json exists AND the trial
    actually completed (has is_resolved set, even if False). Failed
    trials with `failure_mode == 'unknown_agent_error'` and
    `is_resolved == None` are treated as NOT done so they get retried.
    """
    p = cell_dir(target_safe, condition, task) / "results.json"
    if not p.exists() or p.stat().st_size == 0:
        return False
    try:
        d = json.loads(p.read_text())
        results = d.get("results", [])
        if not results:
            return False
        first = results[0]
        # is_resolved is None when the harness crashed before the agent
        # ran. Treat those as not-done so they retry.
        return first.get("is_resolved") is not None
    except Exception:
        return False


def load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text())
    return {
        "experiment": "terminalbench_smoke",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "completed_cells": [],
        "failed_cells": [],
    }


def save_manifest(m: dict) -> None:
    SMOKE_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(m, indent=2))


def run_cell(model_id: str, target_safe: str, condition: str, task: str,
             manifest: dict) -> bool:
    """Run one (target, condition, task) cell. Returns True on success."""
    if is_cell_done(target_safe, condition, task):
        print(f"  [skip] {target_safe} / {condition} / {task} (cached)")
        return True

    out_dir = cell_dir(target_safe, condition, task)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Docker Compose project names disallow uppercase. Our condition
    # 'recent3K' contains 'K'; lowercase the condition slot for the
    # run-id so docker compose -p accepts it. Cell directories and
    # manifest keys keep the canonical 'recent3K' name.
    run_id = (
        f"{target_safe}__{condition.lower()}__{task}__"
        f"{time.strftime('%Y%m%d-%H%M%S')}"
    )

    # Build the `tb run` command. We use --output-path to land directly
    # under our data_archive subtree.
    cmd = [
        "tb", "run",
        "--dataset-name", "terminal-bench-core",
        "--dataset-version", "0.1.1",
        "-t", task,
        "--agent-import-path", "harness.terminus_drifted:TerminusDrifted",
        "--model", model_id,
        "-k", f"recent3K_path={RECENT3K_PATH}",
        "-k", f"condition_label={condition}",
        "--output-path", str(out_dir),
        "--run-id", run_id,
        "--n-concurrent", "1",
        "--no-cleanup",  # keep images for debugging
    ]

    print(f"\n=== {target_safe} / {condition} / {task} ===")
    print(f"  cmd: {' '.join(cmd)}", flush=True)

    env = os.environ.copy()
    env["DOCKER_HOST"] = DOCKER_HOST
    # Make the harness importable from tb run's worker (it does importlib).
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")

    started = time.time()
    try:
        result = subprocess.run(
            cmd, env=env, capture_output=True, text=True,
            timeout=1800,  # 30 min hard cap per cell
        )
    except subprocess.TimeoutExpired:
        elapsed = time.time() - started
        print(f"  [TIMEOUT] after {elapsed:.0f}s")
        manifest["failed_cells"].append({
            "target": target_safe, "condition": condition, "task": task,
            "reason": "timeout_30min",
        })
        save_manifest(manifest)
        return False

    elapsed = time.time() - started
    if result.returncode != 0:
        print(f"  [FAIL] returncode={result.returncode} after {elapsed:.0f}s")
        print(f"  stderr last 500 chars: {result.stderr[-500:]}")
        manifest["failed_cells"].append({
            "target": target_safe, "condition": condition, "task": task,
            "reason": f"returncode_{result.returncode}",
            "stderr_tail": result.stderr[-500:],
        })
        save_manifest(manifest)
        return False

    # Find the produced results.json (under runs/<run_id>/results.json)
    produced = out_dir / run_id / "results.json"
    if not produced.exists():
        print(f"  [WARN] results.json not at expected path: {produced}")
        # Look for it elsewhere in the out_dir
        candidates = list(out_dir.rglob("results.json"))
        if candidates:
            produced = candidates[0]
            print(f"  found at: {produced}")
    if produced.exists():
        # Copy/symlink results.json to cell-level for is_cell_done check.
        shutil.copy(produced, cell_dir(target_safe, condition, task) / "results.json")
        # Also save the run-id for traceability.
        (cell_dir(target_safe, condition, task) / "run-id.txt").write_text(run_id)

    manifest["completed_cells"].append({
        "target": target_safe, "condition": condition, "task": task,
        "elapsed_seconds": round(elapsed, 1),
        "run_id": run_id,
    })
    save_manifest(manifest)
    print(f"  [done] {elapsed:.0f}s  run_id={run_id}")
    return True


def aggregate_results(manifest: dict) -> dict:
    """Pull per-cell results.json into a panel-wide summary."""
    rows = []
    for target_safe in {c["target"] for c in manifest["completed_cells"]}:
        for condition in CONDITIONS:
            for task in TASKS:
                p = cell_dir(target_safe, condition, task) / "results.json"
                if not p.exists():
                    continue
                d = json.loads(p.read_text())
                results = d.get("results", [])
                if not results:
                    continue
                r = results[0]
                rows.append({
                    "target": target_safe,
                    "condition": condition,
                    "task": task,
                    "is_resolved": r.get("is_resolved"),
                    "failure_mode": r.get("failure_mode"),
                    "total_input_tokens": r.get("total_input_tokens"),
                    "total_output_tokens": r.get("total_output_tokens"),
                    "agent_started_at": r.get("agent_started_at"),
                    "agent_ended_at": r.get("agent_ended_at"),
                })
    return {
        "n_cells_completed": len(rows),
        "rows": rows,
    }


def main() -> int:
    if not RECENT3K_PATH.exists():
        sys.exit(f"recent3K_path missing: {RECENT3K_PATH}")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("Set ANTHROPIC_API_KEY")

    SMOKE_DIR.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()

    n_cells = len(TARGETS) * len(CONDITIONS) * len(TASKS)
    print(f"Smoke run: {len(TARGETS)} target × {len(CONDITIONS)} conditions × "
          f"{len(TASKS)} tasks = {n_cells} cells")
    print(f"Output dir: {SMOKE_DIR}")
    print(f"Recent3K: {RECENT3K_PATH} ({RECENT3K_PATH.stat().st_size} bytes)")

    started = time.time()
    n_done = 0
    n_fail = 0
    for model_id, target_safe in TARGETS:
        for condition in CONDITIONS:
            for task in TASKS:
                ok = run_cell(model_id, target_safe, condition, task, manifest)
                if ok:
                    n_done += 1
                else:
                    n_fail += 1

    elapsed = time.time() - started
    summary = aggregate_results(manifest)
    summary["elapsed_seconds_total"] = round(elapsed, 1)
    summary["n_completed"] = n_done
    summary["n_failed"] = n_fail
    (SMOKE_DIR / "SMOKE_SUMMARY.json").write_text(json.dumps(summary, indent=2))

    print(f"\n=== SMOKE COMPLETE ===")
    print(f"  Completed: {n_done}/{n_cells}  Failed: {n_fail}")
    print(f"  Wall clock: {elapsed:.0f}s")
    print(f"  Summary: {SMOKE_DIR / 'SMOKE_SUMMARY.json'}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
