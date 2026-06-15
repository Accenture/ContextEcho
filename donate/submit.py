"""ContextEcho donation — submit a verified session to private staging.

Uploads the redacted session + manifest + consent to the PRIVATE staging
dataset as a single pull request the maintainers review. Refuses to upload
unless the verify gate passed (so unverified data can never reach staging).

Auth / upload target:
  1. CONTEXTECHO_RELAY_URL / --relay-url — upload through a server-side relay
  2. --token / CONTEXTECHO_DONATE_TOKEN  — maintainer/dev direct staging token
  3. the contributor's own `huggingface-cli login` (HF_TOKEN / cached)

Public donors should use the relay. It keeps the Hugging Face staging token on
the server, not in this public repository or on donor machines.

Usage:
    python -m donate.submit session.redacted.jsonl
    python -m donate.submit session.redacted.jsonl --token hf_xxx
    python -m donate.submit session.redacted.jsonl --dry-run
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path
from urllib import error, request

STAGING_REPO = "contextecho2026/persona-drift-staging"
VERIFY_CACHE_VERSION = 1


def session_sha256(session: Path) -> str:
    h = hashlib.sha256()
    with session.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_cache_path(session: Path) -> Path:
    stem = session.stem.replace(".redacted", "")
    return session.with_name(f"{stem}.verify.json")


def write_verify_cache(session: Path, report: dict) -> None:
    if not report.get("passed"):
        return
    payload = {
        "version": VERIFY_CACHE_VERSION,
        "session_sha256": session_sha256(session),
        "verify_passed": True,
    }
    verify_cache_path(session).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def cached_verify_passed(session: Path) -> bool:
    try:
        data = json.loads(verify_cache_path(session).read_text(encoding="utf-8"))
    except Exception:
        return False
    return (
        data.get("version") == VERIFY_CACHE_VERSION
        and data.get("verify_passed") is True
        and data.get("session_sha256") == session_sha256(session)
    )


def verify_passed(session: Path) -> bool:
    """Re-run the fail-closed verifier; only a clean exit (0) allows submit."""
    if cached_verify_passed(session):
        print("[submit] verify cache matched current redacted artifact ✓")
        return True
    here = Path(__file__).resolve().parent.parent
    r = subprocess.run(
        [sys.executable, "-m", "donate.verify", str(session)],
        cwd=here, capture_output=True, text=True,
    )
    sys.stdout.write(r.stdout)
    if r.returncode != 0:
        sys.stdout.write(r.stderr)
    ok = r.returncode == 0
    if ok:
        write_verify_cache(session, {"passed": True})
    return ok


def resolve_token(cli_token: str | None) -> str | None:
    return (
        cli_token
        or os.environ.get("CONTEXTECHO_DONATE_TOKEN")
        or os.environ.get("HF_TOKEN")
        or None  # falls through to huggingface_hub's cached login
    )


def resolve_relay_url(cli_url: str | None) -> str | None:
    value = cli_url or os.environ.get("CONTEXTECHO_RELAY_URL") or ""
    return value.strip().rstrip("/") or None


def gather_artifacts(session: Path) -> list[tuple[Path, str]]:
    stem = session.stem.replace(".redacted", "")
    manifest = session.with_name(f"{stem}.manifest.json")
    consent = session.with_name("CONSENT.md")
    artifacts = [(session, "session.redacted.jsonl")]
    if manifest.exists():
        artifacts.append((manifest, "manifest.json"))
    else:
        print(f"[submit] WARNING: no manifest ({manifest.name}). Run `donate.describe` first.")
    verify_cache = verify_cache_path(session)
    if verify_cache.exists():
        artifacts.append((verify_cache, "verify.json"))
    if consent.exists():
        artifacts.append((consent, "CONSENT.md"))
    else:
        print("[submit] WARNING: no CONSENT.md. Run `donate.describe` first.")
    return artifacts


def multipart_body(artifacts: list[tuple[Path, str]]) -> tuple[bytes, str]:
    boundary = f"----ContextEcho{uuid.uuid4().hex}"
    chunks: list[bytes] = []
    for src, field_name in artifacts:
        chunks.extend([
            f"--{boundary}\r\n".encode(),
            (
                f'Content-Disposition: form-data; name="{field_name}"; '
                f'filename="{src.name}"\r\n'
            ).encode(),
            b"Content-Type: application/octet-stream\r\n\r\n",
            src.read_bytes(),
            b"\r\n",
        ])
    chunks.append(f"--{boundary}--\r\n".encode())
    return b"".join(chunks), boundary


def submit_via_relay(relay_url: str, artifacts: list[tuple[Path, str]], dry_run: bool) -> int:
    endpoint = f"{relay_url}/api/donate"
    print(f"[submit] upload mode  : relay")
    print(f"[submit] relay       : {relay_url}")
    for src, dst in artifacts:
        print(f"[submit]   {src.name:28s} -> {dst}")

    if dry_run:
        print("\n[submit] --dry-run: nothing uploaded.")
        return 0

    body, boundary = multipart_body(artifacts)
    req = request.Request(
        endpoint,
        data=body,
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "User-Agent": "contextecho-donate",
        },
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=120) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        print(f"\n[submit] relay upload failed: HTTP {exc.code}", file=sys.stderr)
        print(detail, file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"\n[submit] relay upload failed: {exc}", file=sys.stderr)
        print("[submit] Check CONTEXTECHO_RELAY_URL or use direct maintainer auth.", file=sys.stderr)
        return 1

    submission_id = payload.get("submission_id", "unknown")
    print("\n[submit] Submitted for maintainer review.")
    print(f"[submit] Submission ID: {submission_id}")
    print("[submit] You'll be credited in the next release once it's accepted. Thank you!")
    return 0


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description="Submit a verified session to private staging.")
    p.add_argument("session", type=Path, help="The redacted .jsonl")
    p.add_argument("--token", default=None, help="HF token (else env / cached login)")
    p.add_argument("--relay-url", default=None, help="ContextEcho relay URL (else CONTEXTECHO_RELAY_URL)")
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
    artifacts = gather_artifacts(args.session)

    relay_url = resolve_relay_url(args.relay_url)
    if relay_url:
        return submit_via_relay(relay_url, artifacts, args.dry_run)

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
