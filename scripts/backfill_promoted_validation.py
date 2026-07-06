"""Backfill quick validation for promoted donations that are missing it."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TARGET = "claude-sonnet-4-5"


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            rows.append(item)
    return rows


def validation_root(label: str, target: str) -> Path:
    return REPO_ROOT / "results_v2_candidate" / "session_validation_quick" / label / target


def validation_acceptable(root: Path, python: str) -> bool:
    if not root.exists():
        return False
    cmd = [
        python,
        "analysis/analyze_session_validation.py",
        "--root",
        str(root.relative_to(REPO_ROOT)),
        "--positions",
        "3",
        "--probes",
        "5",
        "--json",
    ]
    proc = subprocess.run(cmd, cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return proc.returncode == 0


def invalid_validation_cells(root: Path) -> list[Path]:
    invalid: list[Path] = []
    if not root.exists():
        return invalid
    for path in sorted(root.rglob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            invalid.append(path)
            continue
        score = data.get("score")
        if not isinstance(score, int) or score not in {0, 1, 2, 3}:
            invalid.append(path)
    return invalid


def promoted_missing_validation(dataset_root: Path, target: str, python: str) -> list[dict[str, Any]]:
    ledger = dataset_root / "data" / "donations" / "ledger.jsonl"
    registry = {
        str(row.get("submission_id") or ""): row
        for row in iter_jsonl(dataset_root / "data" / "donations" / "reviewed_submissions.jsonl")
    }
    missing: list[dict[str, Any]] = []
    for row in iter_jsonl(ledger):
        if row.get("decision") != "ACCEPTABLE":
            continue
        label = str(row.get("label") or "").strip()
        submission_id = str(row.get("submission_id") or "").strip()
        if not label or not submission_id:
            continue
        previous = registry.get(submission_id, {})
        acceptable = validation_acceptable(validation_root(label, target), python)
        needs_registry_sync = (
            previous.get("decision") == "ACCEPTABLE"
            and previous.get("quick_validation") is not True
        )
        if not acceptable or needs_registry_sync:
            missing.append(row)
    return missing


def run(cmd: list[str]) -> int:
    print("$ " + " ".join(cmd))
    return subprocess.run(cmd, cwd=REPO_ROOT).returncode


def run_review_json(cmd: list[str]) -> tuple[int, dict[str, Any]]:
    print("$ " + " ".join(cmd))
    proc = subprocess.run(cmd, cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        report = json.loads(proc.stdout)
    except json.JSONDecodeError:
        print(proc.stdout)
        report = {}
    return proc.returncode, report


def write_backfill_report(dataset_root: Path, row: dict[str, Any], report: dict[str, Any]) -> None:
    label = str(row.get("label") or "").strip()
    if not label or not report:
        return
    report_path = dataset_root / "data" / "donations" / label / "review_report.json"
    if report_path.parent.exists():
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def update_review_registry(dataset_root: Path, row: dict[str, Any], report: dict[str, Any]) -> None:
    registry = dataset_root / "data" / "donations" / "reviewed_submissions.jsonl"
    submission_id = str(row.get("submission_id") or "").strip()
    if not registry.exists() or not submission_id:
        return
    records = iter_jsonl(registry)
    updated = False
    for record in records:
        if record.get("submission_id") == submission_id:
            record["decision"] = report.get("decision", record.get("decision", "ACCEPTABLE"))
            record["quick_validation"] = bool(report.get("checks", {}).get("quick_validation", {}).get("acceptable"))
            record["promoted"] = True
            record["reviewed_utc"] = datetime.now(timezone.utc).isoformat()
            updated = True
            break
    if not updated:
        records.append({
            "submission_id": submission_id,
            "decision": report.get("decision", "ACCEPTABLE"),
            "quick_validation": bool(report.get("checks", {}).get("quick_validation", {}).get("acceptable")),
            "promoted": True,
            "reviewed_utc": datetime.now(timezone.utc).isoformat(),
        })
    registry.write_text("\n".join(json.dumps(r, sort_keys=True) for r in records) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run quick validation for promoted donations missing validation output.")
    p.add_argument("--dataset-root", type=Path, default=Path("data_archive_release_v2"))
    p.add_argument("--staging-dir", type=Path, default=Path("hf_staging_download"))
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--target", default=DEFAULT_TARGET)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    missing = promoted_missing_validation(args.dataset_root, args.target, args.python)
    print(f"[validation-backfill] missing quick validation: {len(missing)}")
    failures = 0
    for row in missing:
        submission_id = str(row["submission_id"])
        label = str(row["label"])
        sub = args.staging_dir / "pending" / submission_id
        if not sub.exists():
            failures += 1
            print(f"[validation-backfill] missing staged submission for {submission_id}: {sub}")
            continue
        print(f"[validation-backfill] run quick validation: {submission_id} ({label})")
        if args.dry_run:
            continue
        root = validation_root(label, args.target)
        for path in invalid_validation_cells(root):
            print(f"[validation-backfill] remove invalid validation cell: {path.relative_to(REPO_ROOT)}")
            path.unlink()
        rc, report = run_review_json([
            args.python,
            "scripts/review_donation.py",
            str(sub),
            "--label",
            label,
            "--run-quick",
            "--json",
        ])
        if rc != 0:
            failures += 1
            continue
        write_backfill_report(args.dataset_root, row, report)
        update_review_registry(args.dataset_root, row, report)
    print(f"[validation-backfill] needs attention: {failures}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
