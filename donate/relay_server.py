"""Server-side ContextEcho donation relay.

This service receives already-redacted donation artifacts and creates a private
Hugging Face staging pull request using a server-side secret. It is intended for
public donation collection, where the maintainer token must not be shipped to
donors.

Run locally:
    HF_STAGING_TOKEN=hf_xxx uvicorn donate.relay_server:app --host 0.0.0.0 --port 8088
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import uuid
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile
from huggingface_hub import CommitOperationAdd, HfApi

STAGING_REPO = os.environ.get("CONTEXTECHO_STAGING_REPO", "contextecho2026/persona-drift-staging")
MAX_SESSION_BYTES = int(os.environ.get("CONTEXTECHO_RELAY_MAX_SESSION_BYTES", str(50 * 1024 * 1024)))
MAX_META_BYTES = int(os.environ.get("CONTEXTECHO_RELAY_MAX_META_BYTES", str(256 * 1024)))
STATE_DIR = Path(os.environ.get("CONTEXTECHO_RELAY_STATE_DIR", ".relay_state"))
SEEN_HASHES = STATE_DIR / "seen_artifact_hashes.jsonl"

REQUIRED_MANIFEST_FIELDS = {
    "session_id",
    "agent",
    "model",
    "domain",
    "language",
    "records",
    "turns",
    "compactions",
    "privacy_tier",
    "allowed_uses",
    "disallowed_uses",
    "redacted_file",
}

app = FastAPI(title="ContextEcho Donation Relay", version="0.1")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_seen_hashes() -> set[str]:
    if not SEEN_HASHES.exists():
        return set()
    out = set()
    for line in SEEN_HASHES.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        artifact_hash = row.get("artifact_hash")
        if isinstance(artifact_hash, str):
            out.add(artifact_hash)
    return out


def _record_seen_hash(artifact_hash: str, submission_id: str) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with SEEN_HASHES.open("a", encoding="utf-8") as f:
        f.write(json.dumps({
            "artifact_hash": artifact_hash,
            "submission_id": submission_id,
        }, sort_keys=True) + "\n")


async def _read_limited(upload: UploadFile, limit: int, label: str) -> bytes:
    data = await upload.read()
    if not data:
        raise HTTPException(status_code=400, detail=f"{label} is empty")
    if len(data) > limit:
        raise HTTPException(status_code=413, detail=f"{label} is too large")
    return data


def _validate_jsonl(data: bytes) -> None:
    for i, raw in enumerate(data.splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            json.loads(raw)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail=f"session JSONL invalid at line {i}: {exc}") from exc


def _validate_manifest(data: bytes) -> dict:
    try:
        manifest = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=400, detail=f"manifest.json is invalid: {exc}") from exc
    missing = sorted(REQUIRED_MANIFEST_FIELDS - set(manifest))
    if missing:
        raise HTTPException(status_code=400, detail=f"manifest.json missing fields: {', '.join(missing)}")
    if manifest.get("privacy_tier") not in {"full_redacted", "user_minimized"}:
        raise HTTPException(status_code=400, detail="manifest privacy_tier is invalid")
    return manifest


def _validate_consent(data: bytes) -> None:
    text = data.decode("utf-8", errors="replace")
    required = ["ContextEcho Donor Consent", "CC-BY-SA-4.0", "[x]"]
    missing = [term for term in required if term not in text]
    if missing:
        raise HTTPException(status_code=400, detail="CONSENT.md does not look complete")


def _submission_id() -> str:
    return f"submission-{uuid.uuid4().hex[:8]}"


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-") or "artifact"


def _upload_to_hf(submission_id: str, session_data: bytes, manifest_data: bytes, consent_data: bytes) -> str | None:
    token = os.environ.get("HF_STAGING_TOKEN") or os.environ.get("CONTEXTECHO_STAGING_TOKEN")
    if not token:
        raise HTTPException(status_code=500, detail="relay missing HF_STAGING_TOKEN")

    api = HfApi(token=token)
    operations = [
        CommitOperationAdd(
            path_in_repo=f"pending/{submission_id}/session.redacted.jsonl",
            path_or_fileobj=io.BytesIO(session_data),
        ),
        CommitOperationAdd(
            path_in_repo=f"pending/{submission_id}/manifest.json",
            path_or_fileobj=io.BytesIO(manifest_data),
        ),
        CommitOperationAdd(
            path_in_repo=f"pending/{submission_id}/CONSENT.md",
            path_or_fileobj=io.BytesIO(consent_data),
        ),
    ]
    try:
        commit = api.create_commit(
            repo_id=STAGING_REPO,
            repo_type="dataset",
            operations=operations,
            commit_message=f"Donation: {submission_id}",
            create_pr=True,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Hugging Face upload failed: {exc}") from exc
    return getattr(commit, "pr_url", None) or getattr(commit, "commit_url", None)


@app.get("/health")
def health() -> dict:
    return {
        "ok": True,
        "staging_repo": STAGING_REPO,
        "has_token": bool(os.environ.get("HF_STAGING_TOKEN") or os.environ.get("CONTEXTECHO_STAGING_TOKEN")),
    }


@app.post("/api/donate")
async def donate(
    session_redacted: Annotated[UploadFile, File(alias="session.redacted.jsonl")],
    manifest_json: Annotated[UploadFile, File(alias="manifest.json")],
    consent_md: Annotated[UploadFile, File(alias="CONSENT.md")],
) -> dict:
    session_data = await _read_limited(session_redacted, MAX_SESSION_BYTES, "session.redacted.jsonl")
    manifest_data = await _read_limited(manifest_json, MAX_META_BYTES, "manifest.json")
    consent_data = await _read_limited(consent_md, MAX_META_BYTES, "CONSENT.md")

    _validate_jsonl(session_data)
    manifest = _validate_manifest(manifest_data)
    _validate_consent(consent_data)

    artifact_hash = _sha256(session_data)
    if artifact_hash in _read_seen_hashes():
        raise HTTPException(status_code=409, detail="duplicate redacted session artifact")

    submission_id = _submission_id()
    pr_url = _upload_to_hf(submission_id, session_data, manifest_data, consent_data)
    _record_seen_hash(artifact_hash, submission_id)

    return {
        "ok": True,
        "submission_id": submission_id,
        "donation_id": manifest.get("session_id"),
        "artifact_hash": artifact_hash[:12],
        "review_url": pr_url,
        "message": "Donation received for maintainer review.",
        "files": [
            _safe_name(session_redacted.filename or "session.redacted.jsonl"),
            _safe_name(manifest_json.filename or "manifest.json"),
            _safe_name(consent_md.filename or "CONSENT.md"),
        ],
    }
