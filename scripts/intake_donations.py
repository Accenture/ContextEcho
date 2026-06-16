"""Maintainer intake loop: download staging donations, review, optionally promote."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SUBMISSION_FILES = ("session.redacted.jsonl", "manifest.json", "CONSENT.md")
SESSION_NAME = "session.redacted.jsonl"
MIN_SESSION_GROWTH_RATIO = 0.20
MIN_SESSION_GROWTH_TURNS = 50


def run(cmd: list[str]) -> int:
    print("$ " + " ".join(cmd))
    proc = subprocess.run(cmd, cwd=REPO_ROOT)
    return proc.returncode


def iter_jsonl_records(path: Path):
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue


def promoted_submission_ids(dataset_root: Path) -> set[str]:
    ledger = dataset_root / "data" / "donations" / "ledger.jsonl"
    ids: set[str] = set()
    for record in iter_jsonl_records(ledger):
        submission_id = record.get("submission_id")
        decision = record.get("decision")
        if isinstance(submission_id, str) and decision == "ACCEPTABLE":
            ids.add(submission_id)
    return ids


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def submission_fingerprint(submission: Path) -> str:
    h = hashlib.sha256()
    for name in SUBMISSION_FILES:
        path = submission / name
        h.update(name.encode())
        h.update(b"\0")
        if path.exists():
            h.update(sha256_file(path).encode())
        else:
            h.update(b"MISSING")
        h.update(b"\0")
    return h.hexdigest()


def submission_session_hash(submission: Path) -> str:
    path = submission / SESSION_NAME
    return sha256_file(path) if path.exists() else ""


def submission_manifest(submission: Path) -> dict:
    path = submission / "manifest.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def submission_lineage(submission: Path) -> dict[str, str]:
    manifest = submission_manifest(submission)
    return {
        "source_session_id": str(manifest.get("source_session_id") or "").strip(),
        "conversation_fingerprint": str(manifest.get("conversation_fingerprint") or "").strip(),
    }


def count_value(value: object) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def submission_scale(submission: Path) -> dict[str, int]:
    manifest = submission_manifest(submission)
    return {
        "turns": count_value(manifest.get("turns")),
        "records": count_value(manifest.get("records")),
    }


def review_registry_path(dataset_root: Path) -> Path:
    return dataset_root / "data" / "donations" / "reviewed_submissions.jsonl"


def load_review_registry(dataset_root: Path) -> dict[str, dict]:
    path = review_registry_path(dataset_root)
    records: dict[str, dict] = {}
    for record in iter_jsonl_records(path):
        submission_id = record.get("submission_id")
        if isinstance(submission_id, str):
            records[submission_id] = record
    return records


def known_session_hashes(dataset_root: Path, reviewed: dict[str, dict] | None = None) -> dict[str, str]:
    hashes: dict[str, str] = {}
    ledger = dataset_root / "data" / "donations" / "ledger.jsonl"
    for record in iter_jsonl_records(ledger):
        session_hash = record.get("session_sha256") or record.get("artifact_sha256")
        submission_id = record.get("submission_id")
        session_path = record.get("session_path")
        if not session_hash and isinstance(session_path, str):
            promoted_session = dataset_root / session_path
            if promoted_session.exists():
                session_hash = sha256_file(promoted_session)
        if isinstance(session_hash, str) and isinstance(submission_id, str):
            hashes[session_hash] = submission_id
    for submission_id, record in (reviewed or load_review_registry(dataset_root)).items():
        session_hash = record.get("session_sha256")
        if isinstance(session_hash, str) and session_hash:
            hashes.setdefault(session_hash, submission_id)
    return hashes


def known_session_lineage(dataset_root: Path, reviewed: dict[str, dict] | None = None) -> dict[str, dict]:
    lineage: dict[str, dict] = {}
    ledger = dataset_root / "data" / "donations" / "ledger.jsonl"
    for record in iter_jsonl_records(ledger):
        submission_id = record.get("submission_id")
        if not isinstance(submission_id, str) or record.get("decision") not in {None, "", "ACCEPTABLE"}:
            continue
        for key in ("source_session_id", "conversation_fingerprint"):
            value = record.get(key)
            if isinstance(value, str) and value:
                lineage[f"{key}:{value}"] = record
    for submission_id, record in (reviewed or load_review_registry(dataset_root)).items():
        if record.get("decision") != "ACCEPTABLE":
            continue
        for key in ("source_session_id", "conversation_fingerprint"):
            value = record.get(key)
            if isinstance(value, str) and value:
                enriched = dict(record)
                enriched.setdefault("submission_id", submission_id)
                lineage.setdefault(f"{key}:{value}", enriched)
    return lineage


def enough_lineage_growth(new_scale: dict[str, int], old_record: dict) -> bool:
    old_turns = count_value(old_record.get("turns"))
    old_records = count_value(old_record.get("records"))
    new_turns = count_value(new_scale.get("turns"))
    new_records = count_value(new_scale.get("records"))
    turn_delta = max(0, new_turns - old_turns)
    record_delta = max(0, new_records - old_records)
    turn_growth = (turn_delta / old_turns) if old_turns else (1.0 if turn_delta else 0.0)
    record_growth = (record_delta / old_records) if old_records else (1.0 if record_delta else 0.0)
    return (
        turn_growth >= MIN_SESSION_GROWTH_RATIO
        or record_growth >= MIN_SESSION_GROWTH_RATIO
        or turn_delta >= MIN_SESSION_GROWTH_TURNS
    )


def append_review_record(dataset_root: Path, record: dict) -> None:
    path = review_registry_path(dataset_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = load_review_registry(dataset_root)
    existing[record["submission_id"]] = record
    path.write_text(
        "\n".join(json.dumps(r, sort_keys=True) for r in existing.values()) + "\n",
        encoding="utf-8",
    )


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
    p.add_argument("--include-reviewed", action="store_true",
                   help="re-review unchanged submissions already recorded in the review registry")
    p.add_argument("--include-duplicates", action="store_true",
                   help="review submissions whose redacted session hash matches an already processed submission")
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
    reviewed = load_review_registry(args.dataset_root)
    session_hashes = known_session_hashes(args.dataset_root, reviewed)
    session_lineage = known_session_lineage(args.dataset_root, reviewed)
    failures = 0
    accepted: list[tuple[Path, str, str]] = []
    skipped_promoted = 0
    skipped_reviewed = 0
    skipped_duplicates = 0
    for sub in pending:
        fingerprint = submission_fingerprint(sub)
        session_hash = submission_session_hash(sub)
        lineage = submission_lineage(sub)
        scale = submission_scale(sub)
        duplicate_of = session_hashes.get(session_hash) if session_hash else ""
        lineage_duplicate_record: dict = {}
        for key, value in lineage.items():
            if value:
                lineage_duplicate_record = session_lineage.get(f"{key}:{value}", {})
                if lineage_duplicate_record:
                    break
        if sub.name in already_promoted and not (args.include_promoted or args.include_reviewed):
            skipped_promoted += 1
            print(f"[intake] skip already promoted: {sub.name}")
            continue
        duplicate_reason = ""
        supersedes_submission = ""
        if duplicate_of and duplicate_of != sub.name:
            duplicate_reason = "session_sha256"
        elif lineage_duplicate_record and lineage_duplicate_record.get("submission_id") != sub.name:
            duplicate_of = str(lineage_duplicate_record.get("submission_id") or "")
            if enough_lineage_growth(scale, lineage_duplicate_record):
                supersedes_submission = duplicate_of
            else:
                duplicate_reason = "session_lineage_low_growth"
        if duplicate_reason and duplicate_of != sub.name and not args.include_duplicates:
            skipped_duplicates += 1
            print(f"[intake] skip duplicate session: {sub.name} matches {duplicate_of} by {duplicate_reason}")
            append_review_record(args.dataset_root, {
                "submission_id": sub.name,
                "fingerprint": fingerprint,
                "session_sha256": session_hash,
                **lineage,
                **scale,
                "decision": "DUPLICATE",
                "duplicate_of": duplicate_of,
                "duplicate_reason": duplicate_reason,
                "reviewed_utc": datetime.now(timezone.utc).isoformat(),
                "quick_validation": False,
                "promoted": False,
            })
            continue
        previous = reviewed.get(sub.name)
        if (
            previous
            and previous.get("fingerprint") == fingerprint
            and not args.include_reviewed
        ):
            skipped_reviewed += 1
            print(f"[intake] skip already reviewed: {sub.name} ({previous.get('decision', 'unknown')})")
            continue
        cmd = [args.python, "scripts/review_donation.py", str(sub)]
        if args.run_quick:
            cmd.append("--run-quick")
        rc = run(cmd)
        if rc == 0:
            accepted.append((sub, fingerprint, session_hash))
            if not args.promote:
                append_review_record(args.dataset_root, {
                    "submission_id": sub.name,
                    "fingerprint": fingerprint,
                    "session_sha256": session_hash,
                    **lineage,
                    **scale,
                    "supersedes_submission": supersedes_submission,
                    "decision": "ACCEPTABLE",
                    "reviewed_utc": datetime.now(timezone.utc).isoformat(),
                    "quick_validation": bool(args.run_quick),
                    "promoted": False,
                })
                if session_hash:
                    session_hashes.setdefault(session_hash, sub.name)
                for key, value in lineage.items():
                    if value:
                        session_lineage[f"{key}:{value}"] = {
                            "submission_id": sub.name,
                            **lineage,
                            **scale,
                        }
        else:
            failures += 1
            append_review_record(args.dataset_root, {
                "submission_id": sub.name,
                "fingerprint": fingerprint,
                "session_sha256": session_hash,
                **lineage,
                **scale,
                "decision": "CHECK_REQUIRED",
                "reviewed_utc": datetime.now(timezone.utc).isoformat(),
                "quick_validation": bool(args.run_quick),
                "promoted": False,
            })
            if session_hash:
                session_hashes.setdefault(session_hash, sub.name)
            for key, value in lineage.items():
                if value:
                    session_lineage[f"{key}:{value}"] = {
                        "submission_id": sub.name,
                        **lineage,
                        **scale,
                    }

    if args.promote:
        for sub, fingerprint, session_hash in accepted:
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
                continue
            append_review_record(args.dataset_root, {
                "submission_id": sub.name,
                "fingerprint": fingerprint,
                "session_sha256": session_hash,
                **submission_lineage(sub),
                **submission_scale(sub),
                "decision": "ACCEPTABLE",
                "reviewed_utc": datetime.now(timezone.utc).isoformat(),
                "quick_validation": bool(args.run_quick),
                "promoted": True,
            })
            if session_hash:
                session_hashes.setdefault(session_hash, sub.name)

    print(f"[intake] accepted: {len(accepted)}")
    print(f"[intake] skipped promoted: {skipped_promoted}")
    print(f"[intake] skipped reviewed: {skipped_reviewed}")
    print(f"[intake] skipped duplicates: {skipped_duplicates}")
    print(f"[intake] needs attention: {failures}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
