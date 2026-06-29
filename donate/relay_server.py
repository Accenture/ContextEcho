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
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

from fastapi import Body, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

from donate.adapters.base import conversation_fingerprint
from donate.redact import apply_scrub_terms_to_file, _literal_case_insensitive_counts

STAGING_REPO = os.environ.get("CONTEXTECHO_STAGING_REPO", "contextecho2026/persona-drift-staging")
MAX_SESSION_BYTES = int(os.environ.get("CONTEXTECHO_RELAY_MAX_SESSION_BYTES", str(1024 * 1024 * 1024)))
MAX_META_BYTES = int(os.environ.get("CONTEXTECHO_RELAY_MAX_META_BYTES", str(256 * 1024)))
LFS_JSONL_RULE = "*.jsonl filter=lfs diff=lfs merge=lfs -text"
STATE_DIR = Path(os.environ.get("CONTEXTECHO_RELAY_STATE_DIR", ".relay_state"))
SEEN_HASHES = STATE_DIR / "seen_artifact_hashes.jsonl"
SUBMISSION_EVENTS = STATE_DIR / "submission_events.jsonl"
METADATA_UPDATES = STATE_DIR / "metadata_updates.jsonl"
SUPPORT_REQUESTS = STATE_DIR / "support_requests.jsonl"
REDACTION_UPDATES = STATE_DIR / "redaction_updates.jsonl"
ADMIN_TOKEN = os.environ.get("CONTEXTECHO_RELAY_ADMIN_TOKEN")
MIN_SESSION_GROWTH_RATIO = float(os.environ.get("CONTEXTECHO_RELAY_MIN_SESSION_GROWTH_RATIO", "0.20"))
MIN_SESSION_GROWTH_TURNS = int(os.environ.get("CONTEXTECHO_RELAY_MIN_SESSION_GROWTH_TURNS", "50"))
BACKFILL_REPOS = [
    repo.strip()
    for repo in os.environ.get(
        "CONTEXTECHO_RELAY_BACKFILL_REPOS",
        f"{STAGING_REPO},contextecho2026/persona-drift-contextecho",
    ).split(",")
    if repo.strip()
]
REVIEW_STATUS_PATH = os.environ.get(
    "CONTEXTECHO_RELAY_REVIEW_STATUS_PATH",
    "maintainer/reviewed_submissions.jsonl",
)

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
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        origin.strip()
        for origin in os.environ.get(
            "CONTEXTECHO_RELAY_CORS_ORIGINS",
            "https://accenture.github.io,http://127.0.0.1:8000,http://localhost:8000",
        ).split(",")
        if origin.strip()
    ],
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "X-Admin-Token"],
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _append_submission_event(event: str, **fields: object) -> None:
    record = {"ts": _utc_now(), "event": event}
    record.update({k: v for k, v in fields.items() if v not in (None, "")})
    try:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        with SUBMISSION_EVENTS.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")
    except OSError:
        # Audit logging should never make donation submission fail.
        pass


def _read_submission_events(limit: int = 200) -> list[dict]:
    if not SUBMISSION_EVENTS.exists():
        return []
    rows = []
    for line in SUBMISSION_EVENTS.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows[-max(1, min(limit, 1000)) :]


def _read_jsonl(path: Path, limit: int = 200) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows[-max(1, min(limit, 1000)) :]


def _status_backfill_marker() -> Path:
    return STATE_DIR / "status_autobackfill_complete.json"


def _status_backfill_completed() -> bool:
    if _status_backfill_marker().exists():
        return True
    return any(row.get("event") == "status_autobackfill_finished" for row in _read_submission_events(limit=1000))


def _mark_status_backfill_completed(result: dict) -> None:
    try:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        _status_backfill_marker().write_text(
            json.dumps(
                {
                    "ts": _utc_now(),
                    "scanned": result.get("scanned", 0),
                    "added": result.get("added", 0),
                    "refreshed": result.get("refreshed", 0),
                    "errors": result.get("errors", [])[:20],
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    except OSError:
        pass


def _metadata_update_request(payload: dict) -> dict:
    submission_id = str(payload.get("submission_id") or "").strip()
    if not re.fullmatch(r"submission-[A-Za-z0-9_-]+", submission_id):
        raise HTTPException(status_code=400, detail="submission_id must look like submission-xxxxxxxx")
    credit_name = str(payload.get("credit_name") or payload.get("contributor") or "").strip()
    email = str(payload.get("contributor_email") or payload.get("email") or "").strip()
    institute = str(payload.get("contributor_institute") or payload.get("institute") or "").strip()
    if not any([credit_name, email, institute, "public_anonymous" in payload]):
        raise HTTPException(status_code=400, detail="provide at least one metadata field to update")
    record = {
        "request_id": f"metadata-{uuid.uuid4().hex[:8]}",
        "submitted_utc": _utc_now(),
        "status": "pending",
        "submission_id": submission_id,
        "credit_name": credit_name,
        "contributor_email": email,
        "contributor_institute": institute,
        "public_anonymous": bool(payload.get("public_anonymous")),
        "source_session_id": str(payload.get("source_session_id") or "").strip(),
        "conversation_fingerprint": str(payload.get("conversation_fingerprint") or "").strip(),
        "reason": str(payload.get("reason") or "donor requested contributor metadata update").strip(),
    }
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with METADATA_UPDATES.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")
    _append_submission_event(
        "metadata_update_requested",
        request_id=record["request_id"],
        submission_id=submission_id,
        contributor_email=email,
        contributor_institute=institute,
    )
    return record


SUPPORT_REASONS = {
    "remove_submission",
    "reset_for_resubmit",
    "wrong_session",
    "duplicate",
    "wizard_error",
    "other",
}


def _request_log_rows(path: Path, id_key: str, limit: int = 200) -> list[dict]:
    merged: dict[str, dict] = {}
    order: list[str] = []
    for row in _read_jsonl(path, limit=1000):
        request_id = str(row.get(id_key) or "").strip()
        if not request_id:
            continue
        if request_id not in merged:
            merged[request_id] = {}
            order.append(request_id)
        merged[request_id].update(row)
    rows = [merged[request_id] for request_id in order]
    rows.sort(key=lambda row: str(row.get("resolved_utc") or row.get("submitted_utc") or ""), reverse=True)
    return rows[: max(1, min(limit, 1000))]


def _support_request(payload: dict) -> dict:
    submission_id = str(payload.get("submission_id") or "").strip()
    reason = str(payload.get("reason") or "other").strip()
    if reason not in SUPPORT_REASONS:
        raise HTTPException(status_code=400, detail="support reason is invalid")
    if reason == "wizard_error" and not submission_id:
        submission_id = "wizard-error"
    if not re.fullmatch(r"(submission-[A-Za-z0-9_-]+|wizard-error)", submission_id):
        raise HTTPException(status_code=400, detail="submission_id must look like submission-xxxxxxxx")
    message = str(payload.get("message") or "").strip()[:2000]
    record = {
        "support_id": f"{'support-wizard' if reason == 'wizard_error' else 'support'}-{uuid.uuid4().hex[:8]}",
        "submitted_utc": _utc_now(),
        "status": "pending",
        "submission_id": submission_id,
        "reason": reason,
        "message": message,
        "source_session_id": str(payload.get("source_session_id") or "").strip(),
        "conversation_fingerprint": str(payload.get("conversation_fingerprint") or "").strip(),
    }
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with SUPPORT_REQUESTS.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")
    _append_submission_event(
        "support_requested",
        support_id=record["support_id"],
        submission_id=submission_id,
        reason=reason,
    )
    return record


def _support_request_rows(limit: int = 200) -> list[dict]:
    return _request_log_rows(SUPPORT_REQUESTS, "support_id", limit)


def _submission_session_paths(submission_id: str) -> tuple[str, str]:
    prefix = f"pending/{submission_id}"
    return f"{prefix}/session.redacted.jsonl", f"{prefix}/manifest.json"


def _normalize_scrub_terms(raw_terms: object) -> set[str]:
    if isinstance(raw_terms, str):
        items = raw_terms.split(",")
    elif isinstance(raw_terms, list):
        items = [str(item) for item in raw_terms]
    else:
        items = []
    return {str(term).strip() for term in items if str(term).strip()}


def _redaction_update_request(payload: dict) -> dict:
    submission_id = str(payload.get("submission_id") or "").strip()
    if not re.fullmatch(r"submission-[A-Za-z0-9_-]+", submission_id):
        raise HTTPException(status_code=400, detail="submission_id must look like submission-xxxxxxxx")
    scrub_terms = _normalize_scrub_terms(payload.get("scrub_terms") or payload.get("terms"))
    if not scrub_terms:
        raise HTTPException(status_code=400, detail="scrub_terms is required")
    note = str(payload.get("note") or "").strip()[:1000]

    token = os.environ.get("HF_STAGING_TOKEN") or os.environ.get("CONTEXTECHO_STAGING_TOKEN")
    if not token:
        raise HTTPException(status_code=500, detail="relay missing HF_STAGING_TOKEN")

    api = HfApi(token=token)
    session_path, manifest_path = _submission_session_paths(submission_id)
    try:
        session_local = Path(
            hf_hub_download(
                repo_id=STAGING_REPO,
                repo_type="dataset",
                filename=session_path,
                token=token,
            )
        )
    except Exception as exc:
        raise HTTPException(status_code=404, detail="submission redacted session not found") from exc
    manifest = _read_hf_json(STAGING_REPO, manifest_path, token)
    if not manifest:
        raise HTTPException(status_code=502, detail=f"{manifest_path}: manifest unavailable")

    with tempfile.TemporaryDirectory(prefix="contextecho-redaction-update-") as td:
        updated_session = Path(td) / "session.redacted.jsonl"
        stats = apply_scrub_terms_to_file(session_local, updated_session, scrub_terms)
        if not stats:
            record = {
                "redaction_id": f"redaction-{uuid.uuid4().hex[:8]}",
                "submission_id": submission_id,
                "submitted_utc": _utc_now(),
                "status": "no_changes",
                "terms": sorted(scrub_terms),
                "note": note,
            }
            STATE_DIR.mkdir(parents=True, exist_ok=True)
            with REDACTION_UPDATES.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, sort_keys=True) + "\n")
            _append_submission_event(
                "redaction_update_noop",
                submission_id=submission_id,
                terms=sorted(scrub_terms),
            )
            return {
                "changed": False,
                "submission_id": submission_id,
                "terms": sorted(scrub_terms),
                "stats": stats,
                "message": "No additional redaction matches were found.",
            }

        updated_manifest = dict(manifest)
        updated_manifest["maintenance_redaction_updated_utc"] = _utc_now()
        updated_manifest["maintenance_redaction_terms"] = sorted(scrub_terms)
        updated_manifest["maintenance_redaction_stats"] = stats
        if note:
            updated_manifest["maintenance_redaction_note"] = note
        manifest_data = (json.dumps(updated_manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
        try:
            commit = api.create_commit(
                repo_id=STAGING_REPO,
                repo_type="dataset",
                operations=[
                    CommitOperationAdd(path_in_repo=session_path, path_or_fileobj=io.BytesIO(updated_session.read_bytes())),
                    CommitOperationAdd(path_in_repo=manifest_path, path_or_fileobj=io.BytesIO(manifest_data)),
                ],
                commit_message=f"Redaction update: {submission_id}",
                create_pr=True,
            )
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Hugging Face redaction update failed: {exc}") from exc

        artifact_hash = _sha256_file(updated_session)
        _record_seen_hash(artifact_hash, submission_id, updated_manifest)
        pr_url = getattr(commit, "pr_url", None) or getattr(commit, "commit_url", None)
        record = {
            "redaction_id": f"redaction-{uuid.uuid4().hex[:8]}",
            "submission_id": submission_id,
            "submitted_utc": _utc_now(),
            "status": "updated",
            "terms": sorted(scrub_terms),
            "note": note,
            "artifact_hash": artifact_hash,
            "review_url": pr_url or "",
            "stats": stats,
        }
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        with REDACTION_UPDATES.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")
        _append_submission_event(
            "redaction_updated",
            submission_id=submission_id,
            terms=sorted(scrub_terms),
            artifact_hash=artifact_hash,
            review_url=pr_url,
            stats=stats,
        )
        return {
            "changed": True,
            "submission_id": submission_id,
            "terms": sorted(scrub_terms),
            "stats": stats,
            "artifact_hash": artifact_hash[:12],
            "review_url": pr_url,
            "message": "Redaction update submitted for maintainer review.",
        }


def _search_submission_redacted(payload: dict) -> dict:
    submission_id = str(payload.get("submission_id") or "").strip()
    if not re.fullmatch(r"submission-[A-Za-z0-9_-]+", submission_id):
        raise HTTPException(status_code=400, detail="submission_id must look like submission-xxxxxxxx")
    scrub_terms = _normalize_scrub_terms(payload.get("terms") or payload.get("scrub_terms"))
    if not scrub_terms:
        raise HTTPException(status_code=400, detail="terms is required")

    token = os.environ.get("HF_STAGING_TOKEN") or os.environ.get("CONTEXTECHO_STAGING_TOKEN")
    if not token:
        raise HTTPException(status_code=500, detail="relay missing HF_STAGING_TOKEN")

    api = HfApi(token=token)
    session_path, _manifest_path = _submission_session_paths(submission_id)
    try:
        session_local = Path(
            hf_hub_download(
                repo_id=STAGING_REPO,
                repo_type="dataset",
                filename=session_path,
                token=token,
            )
        )
    except Exception as exc:
        raise HTTPException(status_code=404, detail="submission redacted session not found") from exc
    text = session_local.read_text(encoding="utf-8", errors="replace")
    results = []
    for term in sorted(scrub_terms):
        variant_counts = _literal_case_insensitive_counts(text, term)
        results.append(
            {
                "term": term,
                "count": sum(variant_counts.values()),
                "variants": [
                    {"value": variant, "count": count}
                    for variant, count in sorted(variant_counts.items(), key=lambda item: (item[0].casefold(), item[0]))
                ],
            }
        )
    return {
        "ok": True,
        "submission_id": submission_id,
        "results": results,
        "any_hit": any(row["count"] > 0 for row in results),
    }


def _resolve_support_request(support_id: str, note: str = "") -> dict:
    support_id = str(support_id or "").strip()
    rows = _support_request_rows(limit=1000)
    found = next((row for row in rows if row.get("support_id") == support_id), None)
    if not found:
        raise HTTPException(status_code=404, detail="support request not found")
    if found.get("status") == "resolved":
        return {"request": found, "already_resolved": True}
    record = {
        "support_id": support_id,
        "status": "resolved",
        "resolved_utc": _utc_now(),
        "resolution_note": str(note or "").strip()[:1000],
        "submission_id": found.get("submission_id", ""),
    }
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with SUPPORT_REQUESTS.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")
    _append_submission_event(
        "support_resolved",
        support_id=support_id,
        submission_id=found.get("submission_id", ""),
    )
    merged = dict(found)
    merged.update(record)
    return {"request": merged, "already_resolved": False}


def _metadata_update_patch(record: dict) -> dict:
    patch: dict[str, object] = {}
    if str(record.get("credit_name") or "").strip():
        patch["credit_name"] = str(record.get("credit_name") or "").strip()
    if str(record.get("contributor_email") or "").strip():
        patch["contributor_email"] = str(record.get("contributor_email") or "").strip()
    if str(record.get("contributor_institute") or "").strip():
        patch["contributor_institute"] = str(record.get("contributor_institute") or "").strip()
    if "public_anonymous" in record:
        patch["public_anonymous"] = bool(record.get("public_anonymous"))
    return patch


def _metadata_update_requests(limit: int = 200) -> list[dict]:
    rows = _request_log_rows(METADATA_UPDATES, "request_id", limit)
    rows.sort(key=lambda row: str(row.get("approved_utc") or row.get("submitted_utc") or ""), reverse=True)
    return rows


def _find_metadata_update_request(request_id: str) -> dict:
    for row in _metadata_update_requests(limit=1000):
        if row.get("request_id") == request_id:
            return row
    raise HTTPException(status_code=404, detail="metadata update request not found")


def _append_metadata_update_status(request_id: str, status: str, **fields: object) -> dict:
    record = {
        "request_id": request_id,
        "status": status,
        f"{status}_utc": _utc_now(),
    }
    record.update({k: v for k, v in fields.items() if v not in (None, "")})
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with METADATA_UPDATES.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")
    return record


def _apply_metadata_update_to_seen(request: dict) -> int:
    submission_id = str(request.get("submission_id") or "").strip()
    patch = _metadata_update_patch(request)
    if not patch:
        raise HTTPException(status_code=400, detail="metadata update request has no fields to apply")
    records = _read_seen_records()
    changed = 0
    for row in records:
        if str(row.get("submission_id") or "") != submission_id:
            continue
        for key, value in patch.items():
            row[key] = value
        changed += 1
    if changed:
        _write_seen_records(records)
    return changed


def _apply_metadata_update_to_staging_manifest(request: dict) -> bool:
    token = os.environ.get("HF_STAGING_TOKEN") or os.environ.get("CONTEXTECHO_STAGING_TOKEN")
    if not token:
        return False
    submission_id = str(request.get("submission_id") or "").strip()
    manifest_path = f"pending/{submission_id}/manifest.json"
    api = HfApi(token=token)
    try:
        files = api.list_repo_files(repo_id=STAGING_REPO, repo_type="dataset")
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Hugging Face list failed: {exc}") from exc
    if manifest_path not in files:
        return False
    manifest = _read_hf_json(STAGING_REPO, manifest_path, token, files)
    if not manifest:
        return False
    manifest.update(_metadata_update_patch(request))
    manifest["metadata_updated_utc"] = _utc_now()
    manifest["metadata_update_request_id"] = str(request.get("request_id") or "")
    data = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    try:
        api.create_commit(
            repo_id=STAGING_REPO,
            repo_type="dataset",
            operations=[CommitOperationAdd(path_in_repo=manifest_path, path_or_fileobj=data)],
            commit_message=f"Approve metadata update: {submission_id}",
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Hugging Face manifest update failed: {exc}") from exc
    return True


def _approve_metadata_update(request_id: str) -> dict:
    request = _find_metadata_update_request(request_id)
    if request.get("status") == "approved":
        return {"request": request, "seen_updated": 0, "staging_manifest_updated": False, "already_approved": True}
    seen_updated = _apply_metadata_update_to_seen(request)
    staging_updated = _apply_metadata_update_to_staging_manifest(request)
    status_record = _append_metadata_update_status(
        request_id,
        "approved",
        submission_id=request.get("submission_id", ""),
        seen_updated=seen_updated,
        staging_manifest_updated=staging_updated,
    )
    _append_submission_event(
        "metadata_update_approved",
        request_id=request_id,
        submission_id=request.get("submission_id", ""),
        seen_updated=seen_updated,
        staging_manifest_updated=staging_updated,
    )
    merged = dict(request)
    merged.update(status_record)
    return {
        "request": merged,
        "seen_updated": seen_updated,
        "staging_manifest_updated": staging_updated,
        "already_approved": False,
    }


def _record_seen_hash(artifact_hash: str, submission_id: str, manifest: dict) -> None:
    _append_seen_record(_seen_record(artifact_hash, submission_id, manifest))


def _append_seen_record(record: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with SEEN_HASHES.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def _write_seen_records(records: list[dict]) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with SEEN_HASHES.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, sort_keys=True) + "\n")


def _seen_record(artifact_hash: str, submission_id: str, manifest: dict) -> dict:
    return {
        "artifact_hash": artifact_hash,
        "conversation_fingerprint": manifest.get("conversation_fingerprint", ""),
        "contributor_email": manifest.get("contributor_email", ""),
        "contributor_institute": manifest.get("contributor_institute", ""),
        "credit_name": manifest.get("credit_name") or manifest.get("contributor") or "",
        "fingerprint_version": manifest.get("fingerprint_version", ""),
        "public_anonymous": bool(manifest.get("public_anonymous")),
        "submission_id": submission_id,
        "source_session_id": manifest.get("source_session_id", ""),
        "records": manifest.get("records", 0),
        "turns": manifest.get("turns", 0),
    }


def _safe_int(value: object) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _seen_record_summary(records: list[dict]) -> dict:
    submissions = {str(row.get("submission_id") or "") for row in records if row.get("submission_id")}
    sources = {str(row.get("source_session_id") or "") for row in records if row.get("source_session_id")}
    conversations = {
        str(row.get("conversation_fingerprint") or "")
        for row in records
        if row.get("conversation_fingerprint")
    }
    return {
        "records": len(records),
        "submissions": len(submissions),
        "source_sessions": len(sources),
        "conversations": len(conversations),
        "turns": sum(_safe_int(row.get("turns")) for row in records),
        "jsonl_records": sum(_safe_int(row.get("records")) for row in records),
    }


def _fingerprint_jsonl_bytes(data: bytes) -> str:
    with tempfile.NamedTemporaryFile(suffix=".jsonl") as tmp:
        tmp.write(data)
        tmp.flush()
        return conversation_fingerprint(Path(tmp.name))


def _count_jsonl_records(data: bytes) -> int:
    return sum(1 for line in data.splitlines() if line.strip())


def _submission_id_from_prefix(prefix: str) -> str:
    parts = [p for p in prefix.strip("/").split("/") if p]
    for part in reversed(parts):
        if part.startswith("submission-"):
            return part
    return re.sub(r"[^A-Za-z0-9_-]+", "-", prefix.strip("/")) or "backfilled"


def _backfill_submission_id(prefix: str, manifest: dict) -> str:
    prefix_id = _submission_id_from_prefix(prefix)
    if prefix_id.startswith("submission-"):
        return prefix_id
    return str(manifest.get("reviewed_submission_id") or manifest.get("session_id") or prefix_id)


def _session_prefixes(files: list[str]) -> list[str]:
    prefixes = set()
    for filename in files:
        if filename.endswith("session.redacted.jsonl"):
            parent = str(Path(filename).parent)
            prefixes.add("" if parent == "." else parent)
    return sorted(prefixes)


def _release_ledger_rows(repo_id: str, files: list[str], token: str | None) -> list[dict]:
    if "data/donations/ledger.jsonl" not in files:
        return []
    try:
        local_ledger = hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename="data/donations/ledger.jsonl",
            token=token,
        )
    except Exception:
        return []
    rows = []
    for line in Path(local_ledger).read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and row.get("session_path"):
            rows.append(row)
    return rows


def _staging_review_status_index(files: list[str], token: str | None, errors: list[str]) -> dict[str, dict]:
    if REVIEW_STATUS_PATH not in files:
        return {}
    try:
        local_path = hf_hub_download(
            repo_id=STAGING_REPO,
            repo_type="dataset",
            filename=REVIEW_STATUS_PATH,
            token=token,
        )
    except Exception as exc:
        errors.append(f"{REVIEW_STATUS_PATH}: review status unavailable: {exc}")
        return {}
    statuses: dict[str, dict] = {}
    for line in Path(local_path).read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict):
            continue
        submission_id = str(row.get("submission_id") or "").strip()
        if submission_id:
            statuses[submission_id] = row
    return statuses


def _release_session_paths(files: list[str]) -> list[str]:
    return sorted(
        filename
        for filename in files
        if filename.startswith("data/sessions/session_") and filename.endswith(".jsonl")
    )


def _pending_submission_prefixes(files: list[str]) -> list[str]:
    prefixes = set()
    for filename in files:
        parts = filename.split("/")
        if len(parts) >= 3 and parts[0] == "pending" and parts[1].startswith("submission-"):
            prefixes.add("/".join(parts[:2]))
    return sorted(prefixes)


def _pending_submissions_from_hf() -> dict:
    token = os.environ.get("HF_STAGING_TOKEN") or os.environ.get("CONTEXTECHO_STAGING_TOKEN")
    api = HfApi(token=token)
    errors: list[str] = []
    submissions: list[dict] = []
    try:
        files = api.list_repo_files(repo_id=STAGING_REPO, repo_type="dataset")
    except Exception as exc:
        return {"ok": False, "submissions": [], "errors": [f"{STAGING_REPO}: list failed: {exc}"]}
    review_statuses = _staging_review_status_index(files, token, errors)
    for prefix in _pending_submission_prefixes(files):
        submission_id = prefix.split("/")[-1]
        manifest_path = f"{prefix}/manifest.json"
        manifest = _read_hf_json(STAGING_REPO, manifest_path, token, files)
        if not manifest:
            errors.append(f"{prefix}: manifest unavailable")
        review_status = review_statuses.get(submission_id, {})
        is_promoted = bool(review_status.get("promoted")) or review_status.get("decision") == "ACCEPTABLE"
        submissions.append({
            "submission_id": submission_id,
            "prefix": prefix,
            "review_status": "promoted" if is_promoted else "needs_validation",
            "promoted": is_promoted,
            "review_decision": review_status.get("decision", ""),
            "reviewed_utc": review_status.get("reviewed_utc", ""),
            "quick_validation": bool(review_status.get("quick_validation")),
            "has_session": f"{prefix}/session.redacted.jsonl" in files,
            "has_manifest": manifest_path in files,
            "has_consent": f"{prefix}/CONSENT.md" in files,
            "agent": manifest.get("agent", ""),
            "model": manifest.get("model", ""),
            "turns": manifest.get("turns", 0),
            "records": manifest.get("records", 0),
            "compactions": manifest.get("compactions", 0),
            "source_session_id": manifest.get("source_session_id", ""),
            "conversation_fingerprint": manifest.get("conversation_fingerprint", ""),
            "contributor": manifest.get("credit_name") or manifest.get("contributor") or "",
            "email": manifest.get("contributor_email", ""),
            "institute": manifest.get("contributor_institute", ""),
            "submitted_utc": manifest.get("submitted_utc", ""),
            "privacy_tier": manifest.get("privacy_tier", ""),
        })
    return {"ok": True, "submissions": submissions, "errors": errors[:20]}


def _read_hf_file(repo_id: str, filename: str, token: str | None) -> bytes:
    local_path = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=filename, token=token)
    return Path(local_path).read_bytes()


def _read_hf_json(repo_id: str, filename: str, token: str | None, files: list[str] | None = None) -> dict:
    if files is not None and filename not in files:
        return {}
    try:
        data = _read_hf_file(repo_id, filename, token)
        value = json.loads(data.decode("utf-8"))
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def _backfill_record_from_hf(
    repo_id: str,
    session_path: str,
    manifest_path: str,
    submission_id: str,
    files: list[str],
    token: str | None,
) -> dict:
    session_data = _read_hf_file(repo_id, session_path, token)
    artifact_hash = _sha256(session_data)
    manifest = _read_hf_json(repo_id, manifest_path, token, files) if manifest_path else {}
    manifest.setdefault("conversation_fingerprint", _fingerprint_jsonl_bytes(session_data))
    manifest.setdefault("fingerprint_version", "structure-v1")
    manifest.setdefault("records", _count_jsonl_records(session_data))
    manifest.setdefault("turns", 0)
    return _seen_record(artifact_hash, submission_id, manifest)


def _backfill_seen_hashes_from_hf() -> dict:
    token = os.environ.get("HF_STAGING_TOKEN") or os.environ.get("CONTEXTECHO_STAGING_TOKEN")
    api = HfApi(token=token)
    seen = _read_seen_records()
    seen_by_hash = {row.get("artifact_hash"): row for row in seen if row.get("artifact_hash")}
    added = 0
    refreshed = 0
    scanned = 0
    errors: list[str] = []

    def add_or_refresh(record: dict, *, repo_id: str, submission_id: str, source: str) -> None:
        nonlocal added, refreshed
        artifact_hash = record.get("artifact_hash")
        existing = seen_by_hash.get(artifact_hash)
        if existing is not None:
            changed = False
            for key, value in record.items():
                if value in ("", None, False):
                    continue
                if not existing.get(key):
                    existing[key] = value
                    changed = True
            if changed:
                refreshed += 1
                _append_submission_event("backfill_refreshed", repo_id=repo_id, submission_id=submission_id, source=source)
            return
        seen.append(record)
        seen_by_hash[artifact_hash] = record
        _append_submission_event("backfill_added", repo_id=repo_id, submission_id=submission_id, source=source)
        added += 1

    for repo_id in BACKFILL_REPOS:
        try:
            files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
        except Exception as exc:
            errors.append(f"{repo_id}: list failed: {exc}")
            continue
        for prefix in _session_prefixes(files):
            scanned += 1
            session_path = f"{prefix}/session.redacted.jsonl" if prefix else "session.redacted.jsonl"
            manifest_path = f"{prefix}/manifest.json" if prefix else "manifest.json"
            try:
                manifest = _read_hf_json(repo_id, manifest_path, token, files)
                submission_id = _backfill_submission_id(prefix, manifest)
                record = _backfill_record_from_hf(repo_id, session_path, manifest_path, submission_id, files, token)
                add_or_refresh(record, repo_id=repo_id, submission_id=submission_id, source="session_prefix")
            except Exception as exc:
                errors.append(f"{repo_id}/{prefix}: {exc}")
        ledger_rows = _release_ledger_rows(repo_id, files, token)
        ledger_session_paths = {str(row.get("session_path") or "") for row in ledger_rows}
        for row in ledger_rows:
            scanned += 1
            session_path = str(row.get("session_path") or "")
            manifest_path = str(row.get("manifest_path") or "")
            submission_id = str(row.get("submission_id") or _submission_id_from_prefix(session_path))
            try:
                record = _backfill_record_from_hf(repo_id, session_path, manifest_path, submission_id, files, token)
                add_or_refresh(record, repo_id=repo_id, submission_id=submission_id, source="release_ledger")
            except Exception as exc:
                errors.append(f"{repo_id}/{session_path}: {exc}")
        for session_path in _release_session_paths(files):
            if session_path in ledger_session_paths:
                continue
            scanned += 1
            submission_id = Path(session_path).stem.replace("session_", "public-session-", 1)
            try:
                record = _backfill_record_from_hf(repo_id, session_path, "", submission_id, files, token)
                add_or_refresh(record, repo_id=repo_id, submission_id=submission_id, source="release_session")
            except Exception as exc:
                errors.append(f"{repo_id}/{session_path}: {exc}")
    if added or refreshed:
        _write_seen_records(seen)
    return {"scanned": scanned, "added": added, "refreshed": refreshed, "errors": errors[:20]}


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


def _lineage_status(item: dict, seen_records: list[dict]) -> dict:
    fingerprint = str(item.get("conversation_fingerprint") or "").strip()
    source_id = str(item.get("source_session_id") or "").strip()
    current_turns = _count_value(item.get("turns"))
    current_records = _count_value(item.get("records"))
    best: dict = {}
    best_match_type = ""
    for row in seen_records:
        same_conversation = fingerprint and str(row.get("conversation_fingerprint") or "") == fingerprint
        same_source = source_id and str(row.get("source_session_id") or "") == source_id
        if not (same_conversation or same_source):
            continue
        old_turns = _count_value(row.get("turns"))
        old_records = _count_value(row.get("records"))
        if best and old_turns <= _count_value(best.get("turns")):
            continue
        best = row
        best_match_type = "conversation_fingerprint" if same_conversation else "source_session_id"
    if not best:
        return {"received": False, "update_ready": False, "new_turns": 0, "new_records": 0}
    old_turns = _count_value(best.get("turns"))
    old_records = _count_value(best.get("records"))
    submission_id = str(best.get("submission_id") or "")
    if submission_id.startswith("public-session-") and not old_turns:
        return {
            "received": True,
            "update_ready": False,
            "new_turns": 0,
            "new_records": 0,
            "turns": old_turns,
            "records": old_records,
            "submission_id": submission_id,
            "match_type": best_match_type,
            "credit_name": best.get("credit_name", ""),
            "contributor_email": best.get("contributor_email", ""),
            "contributor_institute": best.get("contributor_institute", ""),
            "public_anonymous": bool(best.get("public_anonymous")),
        }
    turn_delta = max(0, current_turns - old_turns)
    record_delta = max(0, current_records - old_records)
    turn_growth = (turn_delta / old_turns) if old_turns else (1.0 if turn_delta else 0.0)
    record_growth = (record_delta / old_records) if old_records else (1.0 if record_delta else 0.0)
    update_ready = (
        turn_delta >= MIN_SESSION_GROWTH_TURNS
        or turn_growth >= MIN_SESSION_GROWTH_RATIO
        or record_growth >= MIN_SESSION_GROWTH_RATIO
    )
    return {
        "received": True,
        "update_ready": bool(update_ready),
        "new_turns": turn_delta,
        "new_records": record_delta,
        "turns": old_turns,
        "records": old_records,
        "submission_id": submission_id,
        "match_type": best_match_type,
        "credit_name": best.get("credit_name", ""),
        "contributor_email": best.get("contributor_email", ""),
        "contributor_institute": best.get("contributor_institute", ""),
        "public_anonymous": bool(best.get("public_anonymous")),
    }


def _status_seen_records() -> list[dict]:
    seen_records = _read_seen_records()
    auto_backfill = os.environ.get("CONTEXTECHO_RELAY_STATUS_AUTOBACKFILL", "1").strip().lower()
    if auto_backfill in {"0", "false", "no", "off"}:
        return seen_records
    if seen_records and _status_backfill_completed():
        return seen_records
    try:
        _append_submission_event("status_autobackfill_started", existing_records=len(seen_records))
        result = _backfill_seen_hashes_from_hf()
        _append_submission_event(
            "status_autobackfill_finished",
            scanned=result.get("scanned", 0),
            added=result.get("added", 0),
            refreshed=result.get("refreshed", 0),
        )
        _mark_status_backfill_completed(result)
    except Exception as exc:
        _append_submission_event("status_autobackfill_failed", error=str(exc)[:500])
        return seen_records
    return _read_seen_records()


def _reset_seen_hashes() -> int:
    if not SEEN_HASHES.exists():
        _append_submission_event("reset_all", removed=0)
        return 0
    records = _read_seen_records()
    count = len(records)
    SEEN_HASHES.unlink()
    _append_submission_event(
        "reset_all",
        removed=count,
        removed_submission_ids=sorted({str(row.get("submission_id") or "") for row in records if row.get("submission_id")}),
    )
    return count


def _remove_seen_records(match: dict) -> dict:
    allowed = {"submission_id", "artifact_hash", "source_session_id", "conversation_fingerprint"}
    criteria = {k: str(v).strip() for k, v in match.items() if k in allowed and str(v).strip()}
    if not criteria:
        raise HTTPException(
            status_code=400,
            detail="provide one of: submission_id, artifact_hash, source_session_id, conversation_fingerprint",
        )
    records = _read_seen_records()
    removed = []
    kept = []
    for row in records:
        if any(str(row.get(k) or "") == v for k, v in criteria.items()):
            removed.append(row)
        else:
            kept.append(row)
    if removed:
        _write_seen_records(kept)
    result = {
        "removed": len(removed),
        "remaining": len(kept),
        "matched_by": sorted(criteria),
        "removed_submission_ids": sorted({str(row.get("submission_id") or "") for row in removed if row.get("submission_id")}),
    }
    _append_submission_event("reset_one", criteria=criteria, **result)
    return result


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


async def _copy_upload_limited(upload: UploadFile, limit: int, label: str) -> tuple[Path, int]:
    tmp = tempfile.NamedTemporaryFile(prefix="contextecho-relay-", suffix=".upload", delete=False)
    path = Path(tmp.name)
    total = 0
    try:
        with tmp:
            while True:
                chunk = await upload.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > limit:
                    raise HTTPException(status_code=413, detail=f"{label} is too large")
                tmp.write(chunk)
    except Exception:
        path.unlink(missing_ok=True)
        raise
    if total <= 0:
        path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"{label} is empty")
    return path, total


def _validate_jsonl(data: bytes) -> None:
    for i, raw in enumerate(data.splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            json.loads(raw)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail=f"session JSONL invalid at line {i}: {exc}") from exc


def _validate_jsonl_file(path: Path) -> None:
    with path.open("rb") as f:
        for i, raw in enumerate(f, start=1):
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


def _read_staging_gitattributes(token: str) -> str:
    try:
        local_path = hf_hub_download(
            repo_id=STAGING_REPO,
            repo_type="dataset",
            filename=".gitattributes",
            token=token,
        )
    except Exception:
        return ""
    return Path(local_path).read_text(encoding="utf-8", errors="replace")


def _ensure_lfs_jsonl_rule(api: HfApi, token: str) -> None:
    existing = _read_staging_gitattributes(token)
    if LFS_JSONL_RULE in {line.strip() for line in existing.splitlines()}:
        return
    lines = [line.rstrip() for line in existing.splitlines() if line.strip()]
    lines.append(LFS_JSONL_RULE)
    content = ("\n".join(lines) + "\n").encode("utf-8")
    try:
        api.create_commit(
            repo_id=STAGING_REPO,
            repo_type="dataset",
            operations=[
                CommitOperationAdd(path_in_repo=".gitattributes", path_or_fileobj=content),
            ],
            commit_message="Configure JSONL donations for LFS",
        )
    except Exception:
        # Another relay worker may have added the rule concurrently. Re-check
        # before failing the donor upload for a shared repository housekeeping
        # race.
        latest = _read_staging_gitattributes(token)
        if LFS_JSONL_RULE not in {line.strip() for line in latest.splitlines()}:
            raise


def _submission_id() -> str:
    return f"submission-{uuid.uuid4().hex[:8]}"


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-") or "artifact"


def _upload_to_hf(submission_id: str, session_path: Path, manifest_data: bytes, consent_data: bytes) -> str | None:
    token = os.environ.get("HF_STAGING_TOKEN") or os.environ.get("CONTEXTECHO_STAGING_TOKEN")
    if not token:
        raise HTTPException(status_code=500, detail="relay missing HF_STAGING_TOKEN")

    api = HfApi(token=token)
    _ensure_lfs_jsonl_rule(api, token)
    operations = [
        CommitOperationAdd(
            path_in_repo=f"pending/{submission_id}/session.redacted.jsonl",
            path_or_fileobj=str(session_path),
        ),
        CommitOperationAdd(
            path_in_repo=f"pending/{submission_id}/manifest.json",
            path_or_fileobj=manifest_data,
        ),
        CommitOperationAdd(
            path_in_repo=f"pending/{submission_id}/CONSENT.md",
            path_or_fileobj=consent_data,
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
        "max_session_bytes": MAX_SESSION_BYTES,
    }


@app.get("/api/admin/summary")
def admin_summary(
    x_admin_token: Annotated[str | None, Header(alias="X-Admin-Token")] = None,
) -> dict:
    _require_admin_token(x_admin_token)
    records = _read_seen_records()
    return {
        "ok": True,
        "staging_repo": STAGING_REPO,
        "backfill_repos": BACKFILL_REPOS,
        "has_token": bool(os.environ.get("HF_STAGING_TOKEN") or os.environ.get("CONTEXTECHO_STAGING_TOKEN")),
        "max_session_bytes": MAX_SESSION_BYTES,
        "growth_policy": {
            "min_turns": MIN_SESSION_GROWTH_TURNS,
            "min_ratio": MIN_SESSION_GROWTH_RATIO,
        },
        "seen": _seen_record_summary(records),
    }


@app.get("/api/admin/seen-records")
def admin_seen_records(
    x_admin_token: Annotated[str | None, Header(alias="X-Admin-Token")] = None,
) -> dict:
    _require_admin_token(x_admin_token)
    records = sorted(
        _read_seen_records(),
        key=lambda row: str(row.get("submission_id") or ""),
    )
    return {
        "ok": True,
        "summary": _seen_record_summary(records),
        "records": records,
    }


@app.get("/api/admin/submission-events")
def admin_submission_events(
    x_admin_token: Annotated[str | None, Header(alias="X-Admin-Token")] = None,
    limit: int = 200,
) -> dict:
    _require_admin_token(x_admin_token)
    return {
        "ok": True,
        "events": list(reversed(_read_submission_events(limit))),
    }


@app.get("/api/admin/metadata-updates")
def admin_metadata_updates(
    x_admin_token: Annotated[str | None, Header(alias="X-Admin-Token")] = None,
    limit: int = 200,
) -> dict:
    _require_admin_token(x_admin_token)
    requests = _metadata_update_requests(limit)
    return {
        "ok": True,
        "requests": requests,
    }


@app.post("/api/admin/metadata-updates/approve")
def admin_approve_metadata_update(
    payload: Annotated[dict, Body()],
    x_admin_token: Annotated[str | None, Header(alias="X-Admin-Token")] = None,
) -> dict:
    _require_admin_token(x_admin_token)
    request_id = str((payload if isinstance(payload, dict) else {}).get("request_id") or "").strip()
    if not request_id:
        raise HTTPException(status_code=400, detail="request_id is required")
    return {"ok": True, **_approve_metadata_update(request_id)}


@app.get("/api/admin/support-requests")
def admin_support_requests(
    x_admin_token: Annotated[str | None, Header(alias="X-Admin-Token")] = None,
    limit: int = 200,
) -> dict:
    _require_admin_token(x_admin_token)
    return {"ok": True, "requests": _support_request_rows(limit)}


@app.post("/api/admin/support-requests/resolve")
def admin_resolve_support_request(
    payload: Annotated[dict, Body()],
    x_admin_token: Annotated[str | None, Header(alias="X-Admin-Token")] = None,
) -> dict:
    _require_admin_token(x_admin_token)
    data = payload if isinstance(payload, dict) else {}
    support_id = str(data.get("support_id") or "").strip()
    if not support_id:
        raise HTTPException(status_code=400, detail="support_id is required")
    return {"ok": True, **_resolve_support_request(support_id, str(data.get("note") or ""))}


@app.post("/api/admin/redaction-updates")
def admin_redaction_update(
    payload: Annotated[dict, Body()],
    x_admin_token: Annotated[str | None, Header(alias="X-Admin-Token")] = None,
) -> dict:
    _require_admin_token(x_admin_token)
    return {"ok": True, **_redaction_update_request(payload if isinstance(payload, dict) else {})}


@app.post("/api/admin/redaction-search")
def admin_redaction_search(
    payload: Annotated[dict, Body()],
    x_admin_token: Annotated[str | None, Header(alias="X-Admin-Token")] = None,
) -> dict:
    _require_admin_token(x_admin_token)
    return _search_submission_redacted(payload if isinstance(payload, dict) else {})


@app.get("/api/admin/submissions")
def admin_submissions(
    x_admin_token: Annotated[str | None, Header(alias="X-Admin-Token")] = None,
) -> dict:
    _require_admin_token(x_admin_token)
    result = _pending_submissions_from_hf()
    return {"ok": True, "staging_repo": STAGING_REPO, **result}


@app.post("/api/admin/lineage-status")
def admin_lineage_status(
    payload: Annotated[dict, Body()],
    x_admin_token: Annotated[str | None, Header(alias="X-Admin-Token")] = None,
) -> dict:
    _require_admin_token(x_admin_token)
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="payload must be an object")
    return {"ok": True, "status": _lineage_status(payload, _read_seen_records())}


@app.delete("/api/admin/seen-hashes")
def reset_seen_hashes(
    x_admin_token: Annotated[str | None, Header(alias="X-Admin-Token")] = None,
) -> dict:
    _require_admin_token(x_admin_token)
    removed = _reset_seen_hashes()
    return {"ok": True, "removed": removed}


@app.delete("/api/admin/seen-hashes/record")
def remove_seen_hash_record(
    payload: Annotated[dict, Body()],
    x_admin_token: Annotated[str | None, Header(alias="X-Admin-Token")] = None,
) -> dict:
    _require_admin_token(x_admin_token)
    result = _remove_seen_records(payload if isinstance(payload, dict) else {})
    return {"ok": True, **result}


@app.post("/api/admin/backfill-seen-hashes")
def backfill_seen_hashes(
    x_admin_token: Annotated[str | None, Header(alias="X-Admin-Token")] = None,
) -> dict:
    _require_admin_token(x_admin_token)
    result = _backfill_seen_hashes_from_hf()
    _mark_status_backfill_completed(result)
    return {"ok": True, "repos": BACKFILL_REPOS, **result}


@app.post("/api/status")
def donation_status(payload: Annotated[dict, Body()]) -> dict:
    sessions = payload.get("sessions") if isinstance(payload, dict) else []
    if not isinstance(sessions, list):
        raise HTTPException(status_code=400, detail="sessions must be a list")
    seen_records = _status_seen_records()
    statuses = []
    for item in sessions[:200]:
        if not isinstance(item, dict):
            statuses.append({"received": False, "update_ready": False, "new_turns": 0, "new_records": 0})
            continue
        statuses.append(_lineage_status(item, seen_records))
    return {"ok": True, "statuses": statuses}


@app.post("/api/metadata-update")
def metadata_update(payload: Annotated[dict, Body()]) -> dict:
    record = _metadata_update_request(payload if isinstance(payload, dict) else {})
    return {
        "ok": True,
        "request_id": record["request_id"],
        "submission_id": record["submission_id"],
        "status": record["status"],
        "message": "Contributor metadata update request received for maintainer review.",
    }


@app.post("/api/support-request")
def support_request(payload: Annotated[dict, Body()]) -> dict:
    record = _support_request(payload if isinstance(payload, dict) else {})
    return {
        "ok": True,
        "support_id": record["support_id"],
        "status": record["status"],
        "message": "Support request received for maintainer review.",
    }


@app.post("/api/donate")
async def donate(
    session_redacted: Annotated[UploadFile, File(alias="session.redacted.jsonl")],
    manifest_json: Annotated[UploadFile, File(alias="manifest.json")],
    consent_md: Annotated[UploadFile, File(alias="CONSENT.md")],
) -> dict:
    session_path, _session_bytes = await _copy_upload_limited(session_redacted, MAX_SESSION_BYTES, "session.redacted.jsonl")
    manifest_data = await _read_limited(manifest_json, MAX_META_BYTES, "manifest.json")
    consent_data = await _read_limited(consent_md, MAX_META_BYTES, "CONSENT.md")

    try:
        _validate_jsonl_file(session_path)
        manifest = _validate_manifest(manifest_data)
        _validate_consent(consent_data)

        artifact_hash = _sha256_file(session_path)
        seen_records = _read_seen_records()
        duplicate_record = next((row for row in seen_records if row.get("artifact_hash") == artifact_hash), None)
        if duplicate_record:
            _append_submission_event(
                "duplicate_rejected",
                reason="duplicate redacted session artifact",
                matched_submission_id=duplicate_record.get("submission_id", ""),
                artifact_hash=artifact_hash,
            )
            raise HTTPException(status_code=409, detail="duplicate redacted session artifact")
        near_duplicate = _near_duplicate_detail(manifest, seen_records)
        if near_duplicate:
            status = _lineage_status(manifest, seen_records)
            _append_submission_event(
                "duplicate_rejected",
                reason=near_duplicate,
                matched_submission_id=status.get("submission_id", ""),
                source_session_id=manifest.get("source_session_id", ""),
                conversation_fingerprint=manifest.get("conversation_fingerprint", ""),
                turns=manifest.get("turns", 0),
                records=manifest.get("records", 0),
            )
            raise HTTPException(status_code=409, detail=near_duplicate)

        submission_id = _submission_id()
        try:
            pr_url = _upload_to_hf(submission_id, session_path, manifest_data, consent_data)
        except HTTPException as exc:
            _append_submission_event(
                "upload_failed",
                submission_id=submission_id,
                donation_id=manifest.get("session_id", ""),
                source_session_id=manifest.get("source_session_id", ""),
                conversation_fingerprint=manifest.get("conversation_fingerprint", ""),
                artifact_hash=artifact_hash,
                status_code=exc.status_code,
                reason=exc.detail,
            )
            raise
        _record_seen_hash(artifact_hash, submission_id, manifest)
        _append_submission_event(
            "submitted",
            submission_id=submission_id,
            donation_id=manifest.get("session_id", ""),
            source_session_id=manifest.get("source_session_id", ""),
            conversation_fingerprint=manifest.get("conversation_fingerprint", ""),
            artifact_hash=artifact_hash,
            turns=manifest.get("turns", 0),
            records=manifest.get("records", 0),
            review_url=pr_url,
        )
    finally:
        session_path.unlink(missing_ok=True)

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
