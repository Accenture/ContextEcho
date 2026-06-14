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

from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from huggingface_hub import CommitOperationAdd, HfApi

STAGING_REPO = os.environ.get("CONTEXTECHO_STAGING_REPO", "contextecho2026/persona-drift-staging")
MAX_SESSION_BYTES = int(os.environ.get("CONTEXTECHO_RELAY_MAX_SESSION_BYTES", str(50 * 1024 * 1024)))
MAX_META_BYTES = int(os.environ.get("CONTEXTECHO_RELAY_MAX_META_BYTES", str(256 * 1024)))
STATE_DIR = Path(os.environ.get("CONTEXTECHO_RELAY_STATE_DIR", ".relay_state"))
SEEN_HASHES = STATE_DIR / "seen_artifact_hashes.jsonl"
ADMIN_TOKEN = os.environ.get("CONTEXTECHO_RELAY_ADMIN_TOKEN")
MIN_SESSION_GROWTH_RATIO = float(os.environ.get("CONTEXTECHO_RELAY_MIN_SESSION_GROWTH_RATIO", "0.20"))
MIN_SESSION_GROWTH_TURNS = int(os.environ.get("CONTEXTECHO_RELAY_MIN_SESSION_GROWTH_TURNS", "50"))

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


def _read_seen_records() -> list[dict]:
    if not SEEN_HASHES.exists():
        return []
    out = []
    for line in SEEN_HASHES.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            out.append(row)
    return out


def _record_seen_hash(artifact_hash: str, submission_id: str, manifest: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with SEEN_HASHES.open("a", encoding="utf-8") as f:
        f.write(json.dumps({
            "artifact_hash": artifact_hash,
            "conversation_fingerprint": manifest.get("conversation_fingerprint", ""),
            "fingerprint_version": manifest.get("fingerprint_version", ""),
            "submission_id": submission_id,
            "source_session_id": manifest.get("source_session_id", ""),
            "records": manifest.get("records", 0),
            "turns": manifest.get("turns", 0),
        }, sort_keys=True) + "\n")


def _count_value(value: object) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _near_duplicate_detail(manifest: dict, seen_records: list[dict]) -> str | None:
    fingerprint = str(manifest.get("conversation_fingerprint") or "").strip()
    source_id = str(manifest.get("source_session_id") or "").strip()
    if not (fingerprint or source_id):
        return None
    new_turns = _count_value(manifest.get("turns"))
    new_records = _count_value(manifest.get("records"))
    for row in seen_records:
        same_conversation = fingerprint and str(row.get("conversation_fingerprint") or "") == fingerprint
        same_source = source_id and str(row.get("source_session_id") or "") == source_id
        if not (same_conversation or same_source):
            continue
        old_turns = _count_value(row.get("turns"))
        old_records = _count_value(row.get("records"))
        turn_delta = max(0, new_turns - old_turns)
        record_delta = max(0, new_records - old_records)
        turn_growth = (turn_delta / old_turns) if old_turns else (1.0 if turn_delta else 0.0)
        record_growth = (record_delta / old_records) if old_records else (1.0 if record_delta else 0.0)
        if (
            turn_growth < MIN_SESSION_GROWTH_RATIO
            and record_growth < MIN_SESSION_GROWTH_RATIO
            and turn_delta < MIN_SESSION_GROWTH_TURNS
        ):
            pct = int(MIN_SESSION_GROWTH_RATIO * 100)
            return (
                "same source session changed too little since prior submission "
                f"(turns +{turn_delta}, records +{record_delta}; require >= {pct}% growth "
                f"or >= {MIN_SESSION_GROWTH_TURNS} new turns)"
            )
    return None


def _reset_seen_hashes() -> int:
    if not SEEN_HASHES.exists():
        return 0
    count = sum(1 for line in SEEN_HASHES.read_text(encoding="utf-8").splitlines() if line.strip())
    SEEN_HASHES.unlink()
    return count


def _require_admin_token(token: str | None) -> None:
    if not ADMIN_TOKEN:
        raise HTTPException(status_code=404, detail="admin endpoint is not enabled")
    if token != ADMIN_TOKEN:
        raise HTTPException(status_code=403, detail="invalid admin token")


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


@app.delete("/api/admin/seen-hashes")
def reset_seen_hashes(
    x_admin_token: Annotated[str | None, Header(alias="X-Admin-Token")] = None,
) -> dict:
    _require_admin_token(x_admin_token)
    removed = _reset_seen_hashes()
    return {"ok": True, "removed": removed}


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
    seen_records = _read_seen_records()
    if artifact_hash in {row.get("artifact_hash") for row in seen_records}:
        raise HTTPException(status_code=409, detail="duplicate redacted session artifact")
    near_duplicate = _near_duplicate_detail(manifest, seen_records)
    if near_duplicate:
        raise HTTPException(status_code=409, detail=near_duplicate)

    submission_id = _submission_id()
    pr_url = _upload_to_hf(submission_id, session_data, manifest_data, consent_data)
    _record_seen_hash(artifact_hash, submission_id, manifest)

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
