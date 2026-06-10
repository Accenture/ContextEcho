"""Promote an accepted staging donation into a release-ready public dataset tree."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SESSION_NAME = "session.redacted.jsonl"
MANIFEST_NAME = "manifest.json"
CONSENT_NAME = "CONSENT.md"


def safe_label(text: str) -> str:
    out = "".join(c if c.isalnum() or c in {"-", "_"} else "-" for c in text.strip())
    out = "-".join(part for part in out.split("-") if part)
    return out[:64] or "donor"


def default_label(manifest: dict, submission: Path) -> str:
    base = manifest.get("credit_name") or manifest.get("contributor") or "donor"
    return safe_label(f"{base}-{submission.name}")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def count_value(value: object) -> int | str:
    if value in {None, ""}:
        return ""
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return ""


def normalize_language(value: object) -> str:
    language = str(value or "").strip()
    if not language or language.lower() == "unknown":
        return "mixed"
    return language


def normalize_manifest(manifest: dict, submission: Path, session: Path) -> dict:
    out = dict(manifest)
    if not out.get("session_id") or out.get("session_id") == "S?":
        out["session_id"] = submission.name
    out["language"] = normalize_language(out.get("language"))
    for key in ("records", "turns", "compactions"):
        out[key] = count_value(out.get(key))
    out["session_sha256"] = sha256_file(session)
    out["reviewed_submission_id"] = submission.name
    if out.get("domain"):
        out.setdefault("donor_domain", out.get("domain"))
        out.setdefault("reviewed_domain", out.get("domain"))
    return out


def run_review(submission: Path, python: str, run_quick: bool) -> dict:
    cmd = [python, "scripts/review_donation.py", str(submission), "--json"]
    if run_quick:
        cmd.append("--run-quick")
    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    try:
        report = json.loads(proc.stdout)
    except Exception as exc:
        raise RuntimeError(f"review_donation.py did not return JSON:\n{proc.stdout}") from exc
    report["_review_returncode"] = proc.returncode
    return report


def append_ledger(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    existing.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    kept = [r for r in existing if r.get("submission_id") != record["submission_id"]]
    kept.append(record)
    path.write_text(
        "\n".join(json.dumps(r, sort_keys=True) for r in kept) + "\n",
        encoding="utf-8",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Promote one accepted donation into a public dataset tree.")
    p.add_argument("submission", type=Path, help="pending/submission-* folder")
    p.add_argument("--dataset-root", type=Path, default=Path("data_archive_release_v2"))
    p.add_argument("--label", default="", help="public session label; default is contributor plus submission id")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--run-quick", action="store_true", help="require quick validation before promotion")
    p.add_argument("--force", action="store_true", help="promote even if review decision is not ACCEPTABLE")
    p.add_argument("--json", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    sub = args.submission.expanduser()
    session = sub / SESSION_NAME
    manifest_path = sub / MANIFEST_NAME
    consent_path = sub / CONSENT_NAME
    for path in (session, manifest_path, consent_path):
        if not path.exists():
            print(f"[error] missing required file: {path}", file=sys.stderr)
            return 2

    review = run_review(sub, args.python, args.run_quick)
    if review.get("decision") != "ACCEPTABLE" and not args.force:
        print(f"[promote] blocked: review decision is {review.get('decision')}")
        print("[promote] fix the submission or pass --force for an explicit override.")
        return 1

    manifest = normalize_manifest(load_json(manifest_path), sub, session)
    label = safe_label(args.label) if args.label else default_label(manifest, sub)
    dataset = args.dataset_root
    public_session = dataset / "data" / "sessions" / f"session_{label}.jsonl"
    donation_dir = dataset / "data" / "donations" / label
    ledger = dataset / "data" / "donations" / "ledger.jsonl"

    public_session.parent.mkdir(parents=True, exist_ok=True)
    donation_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(session, public_session)
    (donation_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    shutil.copy2(consent_path, donation_dir / CONSENT_NAME)
    (donation_dir / "review_report.json").write_text(json.dumps(review, indent=2), encoding="utf-8")

    record = {
        "submission_id": sub.name,
        "label": label,
        "session_sha256": sha256_file(session),
        "session_path": str(public_session.relative_to(dataset)),
        "manifest_path": str((donation_dir / MANIFEST_NAME).relative_to(dataset)),
        "consent_path": str((donation_dir / CONSENT_NAME).relative_to(dataset)),
        "review_report_path": str((donation_dir / "review_report.json").relative_to(dataset)),
        "decision": review.get("decision"),
        "promoted_utc": datetime.now(timezone.utc).isoformat(),
        "contributor": manifest.get("contributor"),
        "credit_name": manifest.get("credit_name"),
        "institute": manifest.get("contributor_institute"),
        "agent": manifest.get("agent"),
        "model": manifest.get("model"),
        "org": manifest.get("org"),
        "records": manifest.get("records"),
        "turns": manifest.get("turns"),
        "compactions": manifest.get("compactions"),
        "domain": manifest.get("domain"),
        "language": manifest.get("language"),
        "metadata_confidence": manifest.get("metadata_confidence", {}),
        "privacy_tier": manifest.get("privacy_tier", "full_redacted"),
        "source_format": manifest.get("source_format"),
    }
    append_ledger(ledger, record)

    if args.json:
        print(json.dumps(record, indent=2))
    else:
        print("[promote] promoted accepted donation")
        print(f"[promote] dataset root : {dataset.resolve()}")
        print(f"[promote] session      : {public_session}")
        print(f"[promote] ledger       : {ledger}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
