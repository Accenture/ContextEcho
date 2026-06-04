"""Phase 2 orchestrator for the signed TerminalBench amendment.

Runs the full panel under the locked PREREG_AMENDMENT_TERMINALBENCH
(SHA 8365d3c8...) protocol: 4 targets × 2 conditions × 5 tasks × n=3
trials = 120 cells baseline. Auto-bump to n=5 per §4.1 happens via a
separate analyzer pass after the n=3 baseline lands.

Per-cell caching: a (target, condition, task, trial_idx) tuple is "done"
iff its results.json exists AND its first results entry has is_resolved
not None. Idempotent re-launch resumes at the first un-done cell.

Output layout:
  data_archive/terminalbench/panel/
    PANEL_MANIFEST.json
    PANEL_SUMMARY.json
    <safe_target>/<condition>/<task>/trial-<i>/
      results.json                   -- TerminalBench's per-task results
      llm_seconds_per_turn.json      -- per-LLM-call wall-clock (TimedLiteLLM)
      run-id.txt                     -- the tb run-id for traceability
      ... (TerminalBench artifacts)

Run:
  set -a && source ../.env && set +a && \\
    python -u scripts/run_terminalbench_panel.py \\
      [--phase wiring|baseline] [--targets ...] [--tasks ...] [--n 3]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PANEL_DIR = REPO_ROOT / "data_archive" / "terminalbench" / "panel"
RECENT3K_PATH = REPO_ROOT / "data_archive" / "terminalbench" / "recent3K_claude.txt"
MANIFEST_PATH = PANEL_DIR / "PANEL_MANIFEST.json"

# Locked targets per signed amendment §2.2.
TARGETS = [
    ("anthropic/claude-sonnet-4-6", "claude-sonnet-4-6"),
    ("anthropic/claude-haiku-4-5", "claude-haiku-4-5"),
    ("openai/gpt-5", "gpt-5"),
    ("gemini/gemini-2.5-pro", "gemini-2-5-pro"),
]
CONDITIONS = ["scratch", "recent3K"]
# Locked tasks per signed amendment §2.3 (count-dataset-tokens excluded).
TASKS = [
    "hello-world",
    "crack-7z-hash.easy",
    "git-multibranch",
    "swe-bench-astropy-1",
    # optional 5th slot: hold for now; lock at first launch
]

DOCKER_HOST = "unix://<USER_HOME>/.colima/default/docker.sock"
AGENT_IMPORT = "harness.terminus_drifted_timed:TerminusDriftedTimed"
DATASET_NAME = "terminal-bench-core"
DATASET_VERSION = "0.1.1"


def safe_id(s: str) -> str:
    return s.replace("/", "-").replace(".", "-")


def cell_dir(target_safe: str, condition: str, task: str, trial: int) -> Path:
    return PANEL_DIR / target_safe / condition / task / f"trial-{trial}"


def is_cell_done(target_safe: str, condition: str, task: str, trial: int) -> bool:
    p = cell_dir(target_safe, condition, task, trial) / "results.json"
    if not p.exists() or p.stat().st_size == 0:
        return False
    try:
        d = json.loads(p.read_text())
        results = d.get("results", [])
        if not results:
            return False
        return results[0].get("is_resolved") is not None
    except Exception:
        return False


def load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text())
    return {
        "experiment": "terminalbench_panel",
        "amendment_sha256": "8365d3c88e528737a4d88ab61d80adf0341be58dfdaa16f9b5cfad37253dd275",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "completed_cells": [],
        "failed_cells": [],
    }


def save_manifest(m: dict) -> None:
    PANEL_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(m, indent=2))


def run_cell(model_id: str, target_safe: str, condition: str, task: str,
             trial: int, manifest: dict, timeout_sec: int = 3600) -> bool:
    """Run one (target, condition, task, trial) cell. Returns True on success."""
    if is_cell_done(target_safe, condition, task, trial):
        print(f"  [skip] {target_safe} / {condition} / {task} / trial-{trial} (cached)")
        return True

    out_dir = cell_dir(target_safe, condition, task, trial)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = (
        f"{target_safe}__{condition.lower()}__{task}__t{trial}__"
        f"{time.strftime('%Y%m%d-%H%M%S')}"
    )

    cmd = [
        "tb", "run",
        "--dataset-name", DATASET_NAME,
        "--dataset-version", DATASET_VERSION,
        "-t", task,
        "--agent-import-path", AGENT_IMPORT,
        "--model", model_id,
        "-k", f"recent3K_path={RECENT3K_PATH}",
        "-k", f"condition_label={condition}",
        "--output-path", str(out_dir),
        "--run-id", run_id,
        "--n-concurrent", "1",
        "--no-cleanup",
    ]

    print(f"\n=== {target_safe} / {condition} / {task} / trial-{trial} ===")
    print(f"  cmd: {' '.join(cmd)}", flush=True)

    env = os.environ.copy()
    env["DOCKER_HOST"] = DOCKER_HOST
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")

    started = time.time()
    try:
        result = subprocess.run(
            cmd, env=env, capture_output=True, text=True, timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired:
        elapsed = time.time() - started
        print(f"  [TIMEOUT] after {elapsed:.0f}s")
        manifest["failed_cells"].append({
            "target": target_safe, "condition": condition, "task": task,
            "trial": trial, "reason": f"timeout_{timeout_sec}s",
        })
        save_manifest(manifest)
        return False

    elapsed = time.time() - started
    if result.returncode != 0:
        print(f"  [FAIL] returncode={result.returncode} after {elapsed:.0f}s")
        print(f"  stderr last 500 chars: {result.stderr[-500:]}")
        manifest["failed_cells"].append({
            "target": target_safe, "condition": condition, "task": task,
            "trial": trial, "reason": f"returncode_{result.returncode}",
            "stderr_tail": result.stderr[-500:],
        })
        save_manifest(manifest)
        return False

    # Find produced results.json + per-call timing sidecar.
    produced = out_dir / run_id / "results.json"
    if not produced.exists():
        candidates = list(out_dir.rglob("results.json"))
        if candidates:
            produced = candidates[0]
    if produced.exists():
        shutil.copy(produced, cell_dir(target_safe, condition, task, trial) / "results.json")
        (cell_dir(target_safe, condition, task, trial) / "run-id.txt").write_text(run_id)

    # Hoist the per-call sidecar (TimedLiteLLM dumps under agent-logs/).
    sidecar_candidates = list(out_dir.rglob("llm_seconds_per_turn.json"))
    if sidecar_candidates:
        shutil.copy(
            sidecar_candidates[0],
            cell_dir(target_safe, condition, task, trial) / "llm_seconds_per_turn.json",
        )

    manifest["completed_cells"].append({
        "target": target_safe, "condition": condition, "task": task,
        "trial": trial,
        "elapsed_seconds": round(elapsed, 1),
        "run_id": run_id,
    })
    save_manifest(manifest)
    print(f"  [done] {elapsed:.0f}s  run_id={run_id}")
    return True


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--phase", choices=["wiring", "baseline"], default="baseline",
        help="wiring=1 cell only (Sonnet/scratch/hello-world/trial-0); "
             "baseline=full panel n=3 per §4.1",
    )
    ap.add_argument(
        "--targets", nargs="*", default=None,
        help="Filter to subset of model_ids (e.g., anthropic/claude-sonnet-4-6)",
    )
    ap.add_argument(
        "--tasks", nargs="*", default=None,
        help="Filter to subset of task IDs",
    )
    ap.add_argument(
        "--n", type=int, default=3, help="Trials per cell (baseline=3, post-bump=5)",
    )
    ap.add_argument(
        "--timeout-sec", type=int, default=3600,
        help="Hard timeout per cell (60min per signed amendment §5.2)",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    if not RECENT3K_PATH.exists():
        sys.exit(f"recent3K_path missing: {RECENT3K_PATH}")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("Set ANTHROPIC_API_KEY")

    targets = TARGETS
    if args.targets:
        targets = [(m, s) for (m, s) in TARGETS if m in args.targets]
        if not targets:
            sys.exit(f"No targets matched {args.targets}; valid: {[m for m,_ in TARGETS]}")

    tasks = TASKS
    if args.tasks:
        tasks = [t for t in TASKS if t in args.tasks]
        if not tasks:
            sys.exit(f"No tasks matched {args.tasks}; valid: {TASKS}")

    if args.phase == "wiring":
        targets = [("anthropic/claude-sonnet-4-6", "claude-sonnet-4-6")]
        tasks = ["hello-world"]
        conditions = ["scratch"]
        n_trials = 1
        print("PHASE: wiring smoke (1 cell)")
    else:
        conditions = CONDITIONS
        n_trials = args.n
        print(f"PHASE: baseline (n={n_trials} per cell)")

    PANEL_DIR.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()

    n_cells = len(targets) * len(conditions) * len(tasks) * n_trials
    print(f"Panel run: {len(targets)} targets × {len(conditions)} conds × "
          f"{len(tasks)} tasks × n={n_trials} = {n_cells} cells")
    print(f"Output dir: {PANEL_DIR}")
    print(f"Recent3K: {RECENT3K_PATH} ({RECENT3K_PATH.stat().st_size} bytes)")
    print(f"Amendment SHA: {manifest.get('amendment_sha256','?')[:16]}...")

    started = time.time()
    n_done, n_fail = 0, 0
    for model_id, target_safe in targets:
        for condition in conditions:
            for task in tasks:
                for trial in range(n_trials):
                    ok = run_cell(model_id, target_safe, condition, task,
                                  trial, manifest, timeout_sec=args.timeout_sec)
                    if ok:
                        n_done += 1
                    else:
                        n_fail += 1

    elapsed = time.time() - started
    print(f"\n=== PANEL RUN COMPLETE ===")
    print(f"  Completed: {n_done}/{n_cells}  Failed: {n_fail}")
    print(f"  Wall clock: {elapsed:.0f}s")
    print(f"  Manifest: {MANIFEST_PATH}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
