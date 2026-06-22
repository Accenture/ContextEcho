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
import json
import os
import re
import tempfile
import uuid
from pathlib import Path
from typing import Annotated

from fastapi import Body, FastAPI, File, Header, HTTPException, UploadFile
from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

from donate.adapters.base import conversation_fingerprint

STAGING_REPO = os.environ.get("CONTEXTECHO_STAGING_REPO", "contextecho2026/persona-drift-staging")
MAX_SESSION_BYTES = int(os.environ.get("CONTEXTECHO_RELAY_MAX_SESSION_BYTES", str(1024 * 1024 * 1024)))
MAX_META_BYTES = int(os.environ.get("CONTEXTECHO_RELAY_MAX_META_BYTES", str(256 * 1024)))
LFS_JSONL_RULE = "*.jsonl filter=lfs diff=lfs merge=lfs -text"
STATE_DIR = Path(os.environ.get("CONTEXTECHO_RELAY_STATE_DIR", ".relay_state"))
SEEN_HASHES = STATE_DIR / "seen_artifact_hashes.jsonl"
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


def _record_seen_hash(artifact_hash: str, submission_id: str, manifest: dict) -> None:
    _append_seen_record(_seen_record(artifact_hash, submission_id, manifest))


def _append_seen_record(record: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with SEEN_HASHES.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def _seen_record(artifact_hash: str, submission_id: str, manifest: dict) -> dict:
    return {
        "artifact_hash": artifact_hash,
        "conversation_fingerprint": manifest.get("conversation_fingerprint", ""),
        "fingerprint_version": manifest.get("fingerprint_version", ""),
        "submission_id": submission_id,
        "source_session_id": manifest.get("source_session_id", ""),
        "records": manifest.get("records", 0),
        "turns": manifest.get("turns", 0),
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


def _release_session_paths(files: list[str]) -> list[str]:
    return sorted(
        filename
        for filename in files
        if filename.startswith("data/sessions/session_") and filename.endswith(".jsonl")
    )


def _read_hf_file(repo_id: str, filename: str, token: str | None) -> bytes:
    local_path = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=filename, token=token)
    return Path(local_path).read_bytes()


def _read_hf_json(repo_id: str, filename: str, token: str | None, files: list[str]) -> dict:
    if filename not in files:
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
    seen_hashes = {row.get("artifact_hash") for row in seen}
    added = 0
    scanned = 0
    errors: list[str] = []
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
                submission_id = str(manifest.get("reviewed_submission_id") or manifest.get("session_id") or _submission_id_from_prefix(prefix))
                record = _backfill_record_from_hf(repo_id, session_path, manifest_path, submission_id, files, token)
                if record["artifact_hash"] in seen_hashes:
                    continue
                _append_seen_record(record)
                seen_hashes.add(record["artifact_hash"])
                added += 1
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
                if record["artifact_hash"] in seen_hashes:
                    continue
                _append_seen_record(record)
                seen_hashes.add(record["artifact_hash"])
                added += 1
            except Exception as exc:
                errors.append(f"{repo_id}/{session_path}: {exc}")
        for session_path in _release_session_paths(files):
            if session_path in ledger_session_paths:
                continue
            scanned += 1
            submission_id = Path(session_path).stem.replace("session_", "public-session-", 1)
            try:
                record = _backfill_record_from_hf(repo_id, session_path, "", submission_id, files, token)
                if record["artifact_hash"] in seen_hashes:
                    continue
                _append_seen_record(record)
                seen_hashes.add(record["artifact_hash"])
                added += 1
            except Exception as exc:
                errors.append(f"{repo_id}/{session_path}: {exc}")
    return {"scanned": scanned, "added": added, "errors": errors[:20]}


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
        "submission_id": best.get("submission_id", ""),
        "match_type": best_match_type,
    }


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


@app.delete("/api/admin/seen-hashes")
def reset_seen_hashes(
    x_admin_token: Annotated[str | None, Header(alias="X-Admin-Token")] = None,
) -> dict:
    _require_admin_token(x_admin_token)
    removed = _reset_seen_hashes()
    return {"ok": True, "removed": removed}


@app.post("/api/admin/backfill-seen-hashes")
def backfill_seen_hashes(
    x_admin_token: Annotated[str | None, Header(alias="X-Admin-Token")] = None,
) -> dict:
    _require_admin_token(x_admin_token)
    result = _backfill_seen_hashes_from_hf()
    return {"ok": True, "repos": BACKFILL_REPOS, **result}


@app.post("/api/status")
def donation_status(payload: Annotated[dict, Body()]) -> dict:
    sessions = payload.get("sessions") if isinstance(payload, dict) else []
    if not isinstance(sessions, list):
        raise HTTPException(status_code=400, detail="sessions must be a list")
    seen_records = _read_seen_records()
    statuses = []
    for item in sessions[:200]:
        if not isinstance(item, dict):
            statuses.append({"received": False, "update_ready": False, "new_turns": 0, "new_records": 0})
            continue
        statuses.append(_lineage_status(item, seen_records))
    return {"ok": True, "statuses": statuses}


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
        if artifact_hash in {row.get("artifact_hash") for row in seen_records}:
            raise HTTPException(status_code=409, detail="duplicate redacted session artifact")
        near_duplicate = _near_duplicate_detail(manifest, seen_records)
        if near_duplicate:
            raise HTTPException(status_code=409, detail=near_duplicate)

        submission_id = _submission_id()
        pr_url = _upload_to_hf(submission_id, session_path, manifest_data, consent_data)
        _record_seen_hash(artifact_hash, submission_id, manifest)
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
