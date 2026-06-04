"""Cross-session deployment test on public SWE-Bench tasks.

Per user direction 2026-04-30: the donated transcript is too in-distribution
for Sonnet (it IS Sonnet's prior session). The proper deployment test injects
a stale recent3K (from any source) into a FRESH unrelated public-benchmark
task and measures whether the agent's behavior degrades.

Design (4 arms × 4 SWE-Bench tasks × n=3 trials = 48 cells):
  scratch    — no injection (Phase 2 baseline)
  filler3K   — Lorem ipsum control (length-matched)
  gpt5_3K    — GPT-5 coding session (length + content-type matched, OFF flavor)
  recent3K   — Claude session (length + content-type matched, IN flavor)

Tasks (all 4 SWE-Bench wrappers in terminal-bench-core==0.1.1):
  swe-bench-astropy-1   (already have 5 trials of scratch + recent3K from Phase 2)
  swe-bench-astropy-2
  swe-bench-fsspec
  swe-bench-langcodes

The CLEAN drift test is GPT5_3K vs Claude_3K on these public tasks. The
recent3K injection is STALE relative to the SWE-Bench task — the agent's
context mentions paper-writing / agentic coding, but the task is a fresh
real-world bug fix.

If Claude_3K underperforms scratch on SWE-Bench → drift CAUSES harmful
cross-session transfer (paper-headline finding).
If Claude_3K matches scratch → drift is NEUTRAL in cross-session deployment.
If Claude_3K outperforms scratch → drift somehow generalizes (surprising).

Output: data_archive/terminalbench/cross_session/<arm>/<task>/trial-<i>/...
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
PANEL_DIR = REPO_ROOT / "data_archive" / "terminalbench" / "cross_session"
CONTEXTS = REPO_ROOT / "data_archive" / "terminalbench" / "contexts"
RECENT3K_PATH = REPO_ROOT / "data_archive" / "terminalbench" / "recent3K_claude.txt"
FILLER3K_PATH = CONTEXTS / "filler3K.txt"
GPT5_3K_PATH = CONTEXTS / "gpt5_3K.txt"
MANIFEST_PATH = PANEL_DIR / "MANIFEST.json"

# (arm_label, condition_label_passed_to_agent, context_path_passed_to_agent)
ARMS = [
    ("scratch", "scratch", None),  # no context
    ("filler3K", "filler3K", str(FILLER3K_PATH)),
    ("gpt5_3K", "gpt5_3K", str(GPT5_3K_PATH)),
    ("recent3K", "recent3K", str(RECENT3K_PATH)),
]

TASKS = [
    "swe-bench-astropy-1",
    "swe-bench-astropy-2",
    "swe-bench-fsspec",
    "swe-bench-langcodes",
]

TARGET_MODEL = "anthropic/claude-sonnet-4-6"
TARGET_SAFE = "claude-sonnet-4-6"
DOCKER_HOST = "unix://<DOCKER_SOCKET>"
AGENT_IMPORT = "harness.terminus_drifted_timed:TerminusDriftedTimed"
DATASET = "terminal-bench-core==0.1.1"


def cell_dir(arm: str, task: str, trial: int) -> Path:
    return PANEL_DIR / arm / task / f"trial-{trial}"


def is_cell_done(arm: str, task: str, trial: int) -> bool:
    p = cell_dir(arm, task, trial) / "results.json"
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
        "experiment": "swebench_cross_session",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "completed_cells": [],
        "failed_cells": [],
    }


def save_manifest(m: dict) -> None:
    PANEL_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(m, indent=2))


def run_cell(arm: str, condition_label: str, context_path: str | None,
             task: str, trial: int, manifest: dict, timeout_sec: int = 3600) -> bool:
    if is_cell_done(arm, task, trial):
        print(f"  [skip] {arm}/{task}/trial-{trial} (cached)")
        return True

    out_dir = cell_dir(arm, task, trial)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = (
        f"{TARGET_SAFE}__{arm.lower()}__{task}__t{trial}__"
        f"{time.strftime('%Y%m%d-%H%M%S')}"
    )

    cmd = [
        "tb", "run",
        "-d", DATASET,
        "-t", task,
        "--agent-import-path", AGENT_IMPORT,
        "--model", TARGET_MODEL,
        "-k", f"condition_label={condition_label}",
        "--output-path", str(out_dir),
        "--run-id", run_id,
        "--n-concurrent", "1",
        "--no-cleanup",
    ]
    # recent3K still needs the recent3K_path; other arms use context_path.
    if condition_label == "recent3K":
        cmd.extend(["-k", f"recent3K_path={context_path}"])
    elif context_path:
        cmd.extend(["-k", f"context_path={context_path}"])

    print(f"\n=== {arm} / {task} / trial-{trial} ===")
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
            "arm": arm, "task": task, "trial": trial, "reason": f"timeout_{timeout_sec}s",
        })
        save_manifest(manifest)
        return False

    elapsed = time.time() - started
    if result.returncode != 0:
        print(f"  [FAIL] returncode={result.returncode} after {elapsed:.0f}s")
        print(f"  stderr last 500 chars: {result.stderr[-500:]}")
        manifest["failed_cells"].append({
            "arm": arm, "task": task, "trial": trial,
            "reason": f"returncode_{result.returncode}",
            "stderr_tail": result.stderr[-500:],
        })
        save_manifest(manifest)
        return False

    # Hoist results.json + sidecar
    produced = out_dir / run_id / "results.json"
    if not produced.exists():
        candidates = list(out_dir.rglob("results.json"))
        if candidates:
            produced = candidates[0]
    if produced.exists():
        shutil.copy(produced, cell_dir(arm, task, trial) / "results.json")
        (cell_dir(arm, task, trial) / "run-id.txt").write_text(run_id)

    sidecar_candidates = list(out_dir.rglob("llm_seconds_per_turn.json"))
    if sidecar_candidates:
        shutil.copy(
            sidecar_candidates[0],
            cell_dir(arm, task, trial) / "llm_seconds_per_turn.json",
        )

    manifest["completed_cells"].append({
        "arm": arm, "task": task, "trial": trial,
        "elapsed_seconds": round(elapsed, 1), "run_id": run_id,
    })
    save_manifest(manifest)
    print(f"  [done] {elapsed:.0f}s  run_id={run_id}")
    return True


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["wiring", "baseline"], default="baseline",
                    help="wiring=1 cell smoke; baseline=full")
    ap.add_argument("--arms", nargs="*", default=None,
                    help="filter to specific arm labels (scratch/filler3K/gpt5_3K/recent3K)")
    ap.add_argument("--tasks", nargs="*", default=None,
                    help="filter to specific tasks")
    ap.add_argument("--n", type=int, default=3, help="trials per cell (default 3)")
    ap.add_argument("--timeout-sec", type=int, default=3600)
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    if not RECENT3K_PATH.exists():
        sys.exit(f"recent3K_path missing: {RECENT3K_PATH}")
    if not FILLER3K_PATH.exists():
        sys.exit(f"filler3K_path missing: {FILLER3K_PATH}")
    if not GPT5_3K_PATH.exists():
        sys.exit(f"gpt5_3K_path missing: {GPT5_3K_PATH}")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("Set ANTHROPIC_API_KEY")

    arms = ARMS
    if args.arms:
        arms = [a for a in ARMS if a[0] in args.arms]

    tasks = TASKS
    if args.tasks:
        tasks = [t for t in TASKS if t in args.tasks]

    if args.phase == "wiring":
        # 1 cell per arm on hello-world equivalent (smallest task)
        arms = [a for a in arms if a[0] in ("scratch", "filler3K")]
        tasks = ["swe-bench-langcodes"]
        n_trials = 1
        print("PHASE: wiring smoke (1 task × 2 arms × 1 trial = 2 cells)")
    else:
        n_trials = args.n
        print(f"PHASE: baseline ({len(arms)} arms × {len(tasks)} tasks × n={n_trials})")

    PANEL_DIR.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()

    n_cells = len(arms) * len(tasks) * n_trials
    print(f"Total cells: {n_cells}")
    print(f"Output dir: {PANEL_DIR}")

    started = time.time()
    n_done, n_fail = 0, 0
    for arm, condition_label, context_path in arms:
        for task in tasks:
            for trial in range(n_trials):
                ok = run_cell(arm, condition_label, context_path, task, trial,
                              manifest, timeout_sec=args.timeout_sec)
                if ok:
                    n_done += 1
                else:
                    n_fail += 1

    elapsed = time.time() - started
    print(f"\n=== DONE: {n_done}/{n_cells} cells, fail={n_fail}, wall={elapsed:.0f}s ===")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
