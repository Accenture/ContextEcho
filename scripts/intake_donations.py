"""Maintainer intake loop: download staging donations, review, optionally promote."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def run(cmd: list[str]) -> int:
    print("$ " + " ".join(cmd))
    proc = subprocess.run(cmd, cwd=REPO_ROOT)
    return proc.returncode


def promoted_submission_ids(dataset_root: Path) -> set[str]:
    ledger = dataset_root / "data" / "donations" / "ledger.jsonl"
    ids: set[str] = set()
    if not ledger.exists():
        return ids
    for line in ledger.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        submission_id = record.get("submission_id")
        decision = record.get("decision")
        if isinstance(submission_id, str) and decision == "ACCEPTABLE":
            ids.add(submission_id)
    return ids


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download and review all staged donations.")
    p.add_argument("--staging-dir", type=Path, default=Path("hf_staging_download"))
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--skip-download", action="store_true")
    p.add_argument("--run-quick", action="store_true")
    p.add_argument("--promote", action="store_true", help="promote submissions that pass review")
    p.add_argument("--dataset-root", type=Path, default=Path("data_archive_release_v2"))
    p.add_argument("--include-promoted", action="store_true",
                   help="re-review submissions already recorded as promoted in the local ledger")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.skip_download:
        rc = run([args.python, "scripts/download_donations.py", "--local-dir", str(args.staging_dir)])
        if rc != 0:
            return rc

    pending = sorted((args.staging_dir / "pending").glob("submission-*"))
    if not pending:
        print(f"[intake] no submissions found under {args.staging_dir / 'pending'}")
        return 0

    already_promoted = promoted_submission_ids(args.dataset_root)
    failures = 0
    accepted: list[Path] = []
    skipped = 0
    for sub in pending:
        if sub.name in already_promoted and not args.include_promoted:
            skipped += 1
            print(f"[intake] skip already promoted: {sub.name}")
            continue
        cmd = [args.python, "scripts/review_donation.py", str(sub)]
        if args.run_quick:
            cmd.append("--run-quick")
        rc = run(cmd)
        if rc == 0:
            accepted.append(sub)
        else:
            failures += 1

    if args.promote:
        for sub in accepted:
            cmd = [
                args.python,
                "scripts/promote_donation.py",
                str(sub),
                "--dataset-root",
                str(args.dataset_root),
            ]
            if args.run_quick:
                cmd.append("--run-quick")
            rc = run(cmd)
            if rc != 0:
                failures += 1

    print(f"[intake] accepted: {len(accepted)}")
    print(f"[intake] skipped promoted: {skipped}")
    print(f"[intake] needs attention: {failures}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
