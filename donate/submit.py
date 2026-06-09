"""ContextEcho donation — submit a verified session to private staging.

Uploads the redacted session + manifest + consent to the PRIVATE staging
dataset as a single pull request the maintainers review. Refuses to upload
unless the verify gate passed (so unverified data can never reach staging).

Auth (in priority order):
  1. --token / CONTEXTECHO_DONATE_TOKEN  — embedded write-only staging token
  2. the contributor's own `huggingface-cli login` (HF_TOKEN / cached)
A write-only token can only ADD to staging; it cannot read it or touch the
public dataset.

Usage:
    python -m donate.submit session.redacted.jsonl
    python -m donate.submit session.redacted.jsonl --token hf_xxx
    python -m donate.submit session.redacted.jsonl --dry-run
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import uuid
from pathlib import Path

STAGING_REPO = "contextecho2026/persona-drift-staging"


def verify_passed(session: Path) -> bool:
    """Re-run the fail-closed verifier; only a clean exit (0) allows submit."""
    here = Path(__file__).resolve().parent.parent
    r = subprocess.run(
        [sys.executable, "-m", "donate.verify", str(session)],
        cwd=here, capture_output=True, text=True,
    )
    sys.stdout.write(r.stdout)
    if r.returncode != 0:
        sys.stdout.write(r.stderr)
    return r.returncode == 0


def resolve_token(cli_token: str | None) -> str | None:
    return (
        cli_token
        or os.environ.get("CONTEXTECHO_DONATE_TOKEN")
        or os.environ.get("HF_TOKEN")
        or None  # falls through to huggingface_hub's cached login
    )


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description="Submit a verified session to private staging.")
    p.add_argument("session", type=Path, help="The redacted .jsonl")
    p.add_argument("--token", default=None, help="HF token (else env / cached login)")
    p.add_argument("--dry-run", action="store_true", help="Verify + show what would upload, but do not upload")
    args = p.parse_args(argv)

    if not args.session.exists():
        print(f"[error] not found: {args.session}", file=sys.stderr)
        return 2

    # --- The guard: never submit unverified data ---------------------------
    print("[submit] re-running verify gate before upload...")
    if not verify_passed(args.session):
        print("\n[submit] BLOCKED — verify did not pass. Nothing was uploaded.")
        print("[submit] Resolve the residual PII (re-run redact --scrub) and try again.")
        return 1
    print("[submit] verify passed ✓\n")

    # --- Gather the three artifacts ----------------------------------------
    stem = args.session.stem.replace(".redacted", "")
    manifest = args.session.with_name(f"{stem}.manifest.json")
    consent = args.session.with_name("CONSENT.md")
    artifacts = [(args.session, "session.redacted.jsonl")]
    if manifest.exists():
        artifacts.append((manifest, "manifest.json"))
    else:
        print(f"[submit] WARNING: no manifest ({manifest.name}). Run `donate.describe` first.")
    if consent.exists():
        artifacts.append((consent, "CONSENT.md"))
    else:
        print("[submit] WARNING: no CONSENT.md. Run `donate.describe` first.")

    # Unique submission folder so concurrent PRs don't collide.
    sub_id = f"submission-{uuid.uuid4().hex[:8]}"
    targets = [(src, f"pending/{sub_id}/{name}") for src, name in artifacts]

    print(f"[submit] target repo : {STAGING_REPO} (private)")
    print(f"[submit] submission  : pending/{sub_id}/")
    for src, dst in targets:
        print(f"[submit]   {src.name:28s} -> {dst}")

    if args.dry_run:
        print("\n[submit] --dry-run: nothing uploaded.")
        return 0

    # --- Upload as ONE PR ---------------------------------------------------
    try:
        from huggingface_hub import CommitOperationAdd, HfApi
    except ImportError:
        print("[error] huggingface_hub not installed: pip install huggingface_hub", file=sys.stderr)
        return 2

    token = resolve_token(args.token)
    api = HfApi(token=token)
    ops = [CommitOperationAdd(path_in_repo=dst, path_or_fileobj=str(src)) for src, dst in targets]

    try:
        commit = api.create_commit(
            repo_id=STAGING_REPO,
            repo_type="dataset",
            operations=ops,
            commit_message=f"Donation: {sub_id}",
            create_pr=True,
        )
    except Exception as e:
        print(f"\n[submit] upload failed: {e}", file=sys.stderr)
        print("[submit] If this is an auth error: set CONTEXTECHO_DONATE_TOKEN, pass --token,")
        print("[submit] or run `huggingface-cli login` with an account that can write to staging.")
        return 1

    pr_url = getattr(commit, "pr_url", None) or getattr(commit, "commit_url", "(see staging repo)")
    print("\n[submit] ✅ Submitted as a pull request for maintainer review:")
    print(f"[submit]    {pr_url}")
    print("[submit] You'll be credited in the next (v2) release once it's accepted. Thank you!")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
