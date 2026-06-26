"""Local browser wizard for ContextEcho donations.

Run:
    python -m donate.web

The server binds to 127.0.0.1 only. Raw sessions are read locally; only the
existing submit step can upload verified redacted artifacts.
"""
from __future__ import annotations

import argparse
import datetime as dt
import errno
import hashlib
import json
import os
import platform
import queue
import re
import subprocess
import sys
import threading
import time
import traceback
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from donate import describe as describe_mod
from donate import discover as discover_mod
from donate import minimize as minimize_mod
from donate import redact as redact_mod
from donate import submit as submit_mod
from donate import verify as verify_mod
from donate.adapters.base import is_redacted_artifact


DONATION_ROOT = Path.home() / "Downloads" / "ContextEcho_donations"
DONATION_REGISTRY = DONATION_ROOT / ".donated_sessions.json"
DEFAULT_RELAY_URL = "https://contextecho2026-context-echo-donation-relay.hf.space"
MAX_AUTO_REPAIR_PASSES = 3
MIN_SESSION_GROWTH_RATIO = 0.20
MIN_SESSION_GROWTH_TURNS = 50
GOOD_SESSION_TURNS = 50
BEST_SESSION_TURNS = 100
BEST_SESSION_COMPACTIONS = 2
CLIENT_DISCONNECT_ERRNOS = {errno.EPIPE, errno.ECONNRESET, errno.ECONNABORTED}
SUBMIT_JOBS: dict[str, dict] = {}
SUBMIT_JOBS_LOCK = threading.Lock()


class ClientDisconnected(Exception):
    """Browser closed or navigated away while a local stream was active."""


def stream_error_message(exc: BaseException, action: str) -> str:
    """Return a donor-visible error that is useful even for blank exceptions."""
    name = exc.__class__.__name__
    msg = str(exc).strip()
    if isinstance(exc, ModuleNotFoundError):
        missing = getattr(exc, "name", "") or msg or "a Python package"
        return (
            f"{action} failed because the local Python environment is missing {missing}. "
            "Stop the wizard and rerun the install command so ContextEcho can refresh its private environment."
        )
    if isinstance(exc, ImportError):
        return (
            f"{action} failed while loading the local redaction engine: {msg or name}. "
            "Stop the wizard and rerun the install command so ContextEcho can refresh its private environment."
        )
    return f"{action} failed: {msg or name}"


def sanitize_diagnostic_text(text: object) -> str:
    """Remove obvious local identifiers from automatic maintainer diagnostics."""
    value = str(text or "")
    home = str(Path.home())
    if home:
        value = value.replace(home, "~")
    value = re.sub(r"/Users/[^/\s\"']+", "/Users/<USER>", value)
    value = re.sub(r"/home/[^/\s\"']+", "/home/<USER>", value)
    value = re.sub(r"[A-Za-z]:\\Users\\[^\\\s\"']+", r"C:\\Users\\<USER>", value)
    value = re.sub(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", "<EMAIL>", value)
    secret_patterns = [
        r"sk-[A-Za-z0-9_\-]{16,}",
        r"sk-ant-[A-Za-z0-9_\-]{16,}",
        r"gh[pousr]_[A-Za-z0-9]{20,}",
        r"hf_[A-Za-z0-9]{20,}",
        r"AKIA[0-9A-Z]{16}",
        r"AIza[0-9A-Za-z_\-]{35}",
        r"xox[baprs]-[0-9A-Za-z\-]{10,}",
        r"Bearer\s+[A-Za-z0-9._\-]{20,}",
    ]
    for pattern in secret_patterns:
        value = re.sub(pattern, "<SECRET>", value)
    return value[:4000]


def wizard_error_report_payload(data: dict, exc: BaseException, action: str, traceback_text: str) -> dict:
    auto = data.get("auto") if isinstance(data.get("auto"), dict) else {}
    source_session_id = ""
    try:
        source_session_id = describe_mod.source_session_id(auto) if auto else ""
    except Exception:
        source_session_id = ""
    message = "\n".join([
        f"Automatic wizard error report: {action}",
        f"Error: {stream_error_message(exc, action)}",
        f"Platform: {platform.platform()}",
        f"Python: {sys.version.split()[0]}",
        f"Agent: {auto.get('agent', '')}",
        f"Model: {auto.get('model', '')}",
        f"Turns: {auto.get('turns', '')}",
        f"Records: {auto.get('records', '')}",
        f"Compactions: {auto.get('compactions', '')}",
        "",
        sanitize_diagnostic_text(traceback_text),
    ])
    return {
        "submission_id": "wizard-error",
        "reason": "wizard_error",
        "message": sanitize_diagnostic_text(message),
        "source_session_id": source_session_id,
        "conversation_fingerprint": str(auto.get("conversation_fingerprint") or "").strip(),
    }


def report_wizard_error(data: dict, exc: BaseException, action: str, traceback_text: str) -> str:
    try:
        result = relay_support_request(wizard_error_report_payload(data, exc, action, traceback_text))
    except Exception as report_exc:
        print(f"[web] automatic wizard error report failed: {report_exc}", file=sys.stderr)
        return ""
    support_id = str(result.get("support_id") or "")
    if support_id:
        print(f"[web] automatic wizard error report sent: {support_id}", file=sys.stderr)
    return support_id


def create_server(host: str, port: int, attempts: int = 20) -> tuple[ThreadingHTTPServer, int]:
    """Bind the local wizard, trying nearby ports if the default is busy."""
    for offset in range(max(1, attempts)):
        candidate = port + offset if port else 0
        try:
            server = ThreadingHTTPServer((host, candidate), Handler)
            actual_port = int(server.server_address[1])
            return server, actual_port
        except OSError as exc:
            if exc.errno != errno.EADDRINUSE or not port or offset == attempts - 1:
                raise
    raise OSError(errno.EADDRINUSE, f"no free port found near {port}")


def safe_slug(text: str, default: str = "session") -> str:
    cleaned = "".join(c if c.isalnum() or c in ("-", "_") else "-" for c in text)
    cleaned = "-".join(part for part in cleaned.split("-") if part)
    return (cleaned[:64] or default).strip("-_") or default


def redacted_output_name(src: Path) -> str:
    stem = src.stem
    if stem.endswith(".redacted"):
        stem = stem[: -len(".redacted")]
    return f"{safe_slug(stem)}.redacted.jsonl"


def donation_output_dir(info: dict) -> Path:
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    agent = safe_slug(str(info.get("agent", "agent")).lower())
    project = safe_slug(str(info.get("project", "session")))
    return DONATION_ROOT / f"{stamp}-{agent}-{project}"


def _auto_from_existing_manifest(session: Path) -> dict:
    stem = session.stem.replace(".redacted", "")
    manifest_path = session.with_name(f"{stem}.manifest.json")
    if not manifest_path.exists():
        return {}
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    auto = dict(manifest)
    if "metadata_confidence" in auto and "confidence" not in auto:
        auto["confidence"] = auto.get("metadata_confidence") or {}
    return auto


def submit_auto_metadata(data: dict, session: Path) -> dict:
    auto = data.get("auto")
    if isinstance(auto, dict) and auto:
        return dict(auto)
    source_path = Path(data.get("source_path", "")).expanduser()
    if source_path.exists() and not is_redacted_artifact(source_path):
        try:
            return discover_mod.inspect_session(source_path)
        except Exception:
            pass
    return _auto_from_existing_manifest(session)


def count_jsonl_records(path: Path) -> int:
    records = 0
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.strip():
                records += 1
    return records


def metadata_for_redacted_artifact(data: dict, session: Path) -> dict:
    auto = submit_auto_metadata(data, session)
    try:
        auto["records"] = count_jsonl_records(session)
    except OSError:
        pass
    return auto


def required_contributor_fields(data: dict) -> dict[str, str]:
    fields = {
        "contributor": str(data.get("contributor", "") or "").strip(),
        "email": str(data.get("email", "") or "").strip(),
        "institute": str(data.get("institute", "") or "").strip(),
    }
    labels = {
        "contributor": "name or handle",
        "email": "email",
        "institute": "institute",
    }
    missing = [labels[key] for key, value in fields.items() if not value]
    if missing:
        raise ValueError(f"Name, email, and institute are required before submission. Missing: {', '.join(missing)}.")
    return fields


def session_key(path: str | Path) -> str:
    p = Path(path).expanduser()
    parts = [str(p)]
    try:
        st = p.stat()
        parts.extend([str(st.st_size), str(st.st_mtime_ns)])
    except OSError:
        pass
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]


def artifact_key(path: str | Path) -> str:
    p = Path(path).expanduser()
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def load_donation_registry() -> dict:
    try:
        data = json.loads(DONATION_REGISTRY.read_text())
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def load_donated_keys() -> set[str]:
    data = load_donation_registry()
    return {str(x) for x in data.get("source_keys", [])}


def load_donated_artifact_keys() -> set[str]:
    data = load_donation_registry()
    return {str(x) for x in data.get("artifact_keys", [])}


def source_path_key(path: str | Path) -> str:
    return hashlib.sha256(str(Path(path).expanduser()).encode("utf-8", errors="replace")).hexdigest()


def load_donated_source_records() -> dict[str, dict]:
    records: dict[str, dict] = {}
    for item in load_donation_registry().get("submissions", []):
        keys = [str(item.get("source_path_key") or ""), str(item.get("source_key") or "")]
        keys = [key for key in keys if key]
        if not keys:
            continue
        turns = int(item.get("turns") or 0)
        for key in keys:
            previous = records.get(key)
            if previous is None or turns > int(previous.get("turns") or 0):
                records[key] = item
    return records


def normalize_submission_id(value: object) -> str:
    return str(value or "").replace("pending/", "").strip("/")


def is_support_submission_id(value: object) -> bool:
    return normalize_submission_id(value).startswith("submission-")


def session_update_ready(current_turns: int, previous_turns: int) -> bool:
    delta = max(0, int(current_turns or 0) - int(previous_turns or 0))
    growth = (delta / previous_turns) if previous_turns else (1.0 if delta else 0.0)
    return bool(delta and (delta >= MIN_SESSION_GROWTH_TURNS or growth >= MIN_SESSION_GROWTH_RATIO))


def relay_url() -> str:
    return os.environ.get("CONTEXTECHO_RELAY_URL", DEFAULT_RELAY_URL).strip().rstrip("/")


def relay_donation_status(sessions: list[dict]) -> list[dict]:
    url = relay_url()
    if not url or not sessions:
        return []
    payload = {
        "sessions": [
            {
                "source_session_id": describe_mod.source_session_id(row),
                "conversation_fingerprint": row.get("conversation_fingerprint", ""),
                "turns": row.get("turns", 0),
                "records": row.get("records", 0),
            }
            for row in sessions
        ]
    }
    data = json.dumps(payload).encode("utf-8")
    req = Request(
        f"{url}/api/status",
        data=data,
        headers={"content-type": "application/json", "user-agent": "contextecho-donate"},
        method="POST",
    )
    try:
        with urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return []
    statuses = result.get("statuses", []) if isinstance(result, dict) else []
    return [x if isinstance(x, dict) else {} for x in statuses]


def relay_metadata_update(payload: dict) -> dict:
    url = relay_url()
    if not url:
        raise ValueError("Relay URL is not configured; cannot send metadata update request.")
    data = json.dumps(payload).encode("utf-8")
    req = Request(
        f"{url}/api/metadata-update",
        data=data,
        headers={"content-type": "application/json", "user-agent": "contextecho-donate"},
        method="POST",
    )
    try:
        with urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise ValueError(f"metadata update failed: HTTP {exc.code} {detail}") from exc
    except Exception as exc:
        raise ValueError(f"metadata update failed: {exc}") from exc
    return result if isinstance(result, dict) else {"ok": False}


def relay_support_request(payload: dict) -> dict:
    url = relay_url()
    if not url:
        raise ValueError("Relay URL is not configured; cannot send support request.")
    data = json.dumps(payload).encode("utf-8")
    req = Request(
        f"{url}/api/support-request",
        data=data,
        headers={"content-type": "application/json", "user-agent": "contextecho-donate"},
        method="POST",
    )
    try:
        with urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise ValueError(f"support request failed: HTTP {exc.code} {detail}") from exc
    except Exception as exc:
        raise ValueError(f"support request failed: {exc}") from exc
    return result if isinstance(result, dict) else {"ok": False}


def donation_points_range(turns: str | int = 0, compactions: str | int = 0) -> tuple[int, int]:
    turns_n = int(turns or 0)
    compactions_n = int(compactions or 0)
    return (3, 5) if turns_n >= GOOD_SESSION_TURNS or compactions_n >= 1 else (2, 4)


def donation_fit(turns: str | int = 0, compactions: str | int = 0) -> str:
    turns_n = int(turns or 0)
    compactions_n = int(compactions or 0)
    if turns_n >= BEST_SESSION_TURNS and compactions_n >= BEST_SESSION_COMPACTIONS:
        return "best"
    if turns_n >= GOOD_SESSION_TURNS:
        return "good"
    return "improve"


def donation_ready(turns: str | int = 0, compactions: str | int = 0) -> bool:
    return donation_fit(turns, compactions) != "improve"


def contributor_identity(receipt: dict) -> str:
    name = str(receipt.get("credit_name") or receipt.get("contributor") or "").strip().lower()
    email = str(receipt.get("contributor_email") or "").strip().lower()
    institute = str(receipt.get("institute") or "").strip().lower()
    if not (name and email and institute):
        return ""
    return "\n".join([name, email, institute])


def local_pending_summary(receipt: dict) -> dict:
    target = contributor_identity(receipt)
    if not target:
        low, high = donation_points_range(receipt.get("turns", 0), receipt.get("compactions", 0))
        return {"sessions": 1, "points_low": low, "points_high": high}
    sessions = 0
    points_low = 0
    points_high = 0
    turns = 0
    for item in load_donation_registry().get("submissions", []):
        if item.get("contributor_identity") != target:
            continue
        sessions += 1
        points_low += int(item.get("points_low") or 0)
        points_high += int(item.get("points_high") or 0)
        turns += int(item.get("turns") or 0)
    return {"sessions": sessions or 1, "points_low": points_low or 0, "points_high": points_high or 0, "turns": turns}


def save_donation_record(source_path: str | Path = "", artifact_path: str | Path = "", output: str = "", receipt: dict | None = None) -> None:
    DONATION_ROOT.mkdir(parents=True, exist_ok=True)
    data = load_donation_registry()
    source_keys = {str(x) for x in data.get("source_keys", [])}
    artifact_keys = {str(x) for x in data.get("artifact_keys", [])}
    submissions = list(data.get("submissions", []))
    source = str(source_path or "")
    artifact = str(artifact_path or "")
    skey = session_key(source) if source else ""
    spkey = source_path_key(source) if source else ""
    akey = artifact_key(artifact) if artifact and Path(artifact).expanduser().exists() else ""
    if skey:
        source_keys.add(skey)
    if akey:
        artifact_keys.add(akey)
    m = re.search(r"\[submit\] submission\s*:\s*(pending/submission-[^/\s]+/)", output)
    if not m:
        m = re.search(r"\[submit\]\s*Submission ID:\s*(submission-[A-Za-z0-9_-]+)", output)
    receipt = receipt or {}
    points_low, points_high = donation_points_range(receipt.get("turns", 0), receipt.get("compactions", 0))
    submission = m.group(1) if m else ""
    submission_id = normalize_submission_id(submission)
    submissions.append({
        "submitted_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_key": skey,
        "source_path_key": spkey,
        "source_path": source,
        "artifact_key": akey,
        "submission": submission,
        "submission_id": submission_id,
        "contributor_identity": contributor_identity(receipt),
        "credit_name": receipt.get("credit_name", ""),
        "public_anonymous": bool(receipt.get("public_anonymous")),
        "contributor_email": receipt.get("contributor_email", ""),
        "institute": receipt.get("institute", ""),
        "turns": int(receipt.get("turns") or 0),
        "compactions": int(receipt.get("compactions") or 0),
        "points_low": points_low,
        "points_high": points_high,
    })
    payload = {
        "updated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_keys": sorted(source_keys),
        "artifact_keys": sorted(artifact_keys),
        "submissions": submissions,
    }
    DONATION_REGISTRY.write_text(json.dumps(payload, indent=2) + "\n")


def save_donated_key(path: str | Path) -> None:
    save_donation_record(source_path=path)


def clear_donation_registry() -> bool:
    existed = DONATION_REGISTRY.exists()
    if existed:
        DONATION_REGISTRY.unlink()
    return existed


def clear_donation_record(source_path: str | Path = "", artifact_path: str | Path = "") -> bool:
    data = load_donation_registry()
    if not data:
        return False

    source = str(source_path or "")
    artifact = str(artifact_path or "")
    skey = session_key(source) if source else ""
    spkey = source_path_key(source) if source else ""
    akey = artifact_key(artifact) if artifact and Path(artifact).expanduser().exists() else ""
    if not skey and not akey:
        return False

    source_keys = {str(x) for x in data.get("source_keys", [])}
    artifact_keys = {str(x) for x in data.get("artifact_keys", [])}
    submissions = [x for x in data.get("submissions", []) if isinstance(x, dict)]

    changed = False
    if skey and skey in source_keys:
        source_keys.remove(skey)
        changed = True
    if akey and akey in artifact_keys:
        artifact_keys.remove(akey)
        changed = True

    kept_submissions = []
    for item in submissions:
        if (
            (skey and item.get("source_key") == skey)
            or (spkey and item.get("source_path_key") == spkey)
            or (akey and item.get("artifact_key") == akey)
        ):
            changed = True
            continue
        kept_submissions.append(item)

    if not changed:
        return False

    payload = {
        "updated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_keys": sorted(source_keys),
        "artifact_keys": sorted(artifact_keys),
        "submissions": kept_submissions,
    }
    DONATION_ROOT.mkdir(parents=True, exist_ok=True)
    DONATION_REGISTRY.write_text(json.dumps(payload, indent=2) + "\n")
    return True


def already_submitted(source_path: str | Path = "", artifact_path: str | Path = "") -> bool:
    if source_path and session_key(source_path) in load_donated_keys():
        return True
    if source_path:
        info = load_donated_source_records().get(source_path_key(source_path))
        if info:
            try:
                current_turns = discover_mod.inspect_session(Path(source_path).expanduser()).get("turns", 0)
            except Exception:
                current_turns = int(info.get("turns") or 0)
            if not session_update_ready(current_turns, int(info.get("turns") or 0)):
                return True
    if artifact_path:
        p = Path(artifact_path).expanduser()
        if p.exists() and artifact_key(p) in load_donated_artifact_keys():
            return True
    return False


def parse_submit_output(output: str) -> dict:
    url = (re.search(r"https?://\S+", output) or [None])[0] or ""
    repo = (re.search(r"\[submit\] target repo\s*:\s*(.+)", output) or [None, ""])[1].strip()
    submission = (re.search(r"\[submit\] submission\s*:\s*(pending/submission-[^/\s]+/)", output) or [None, ""])[1].strip()
    if not submission:
        submission = (re.search(r"\[submit\]\s*Submission ID:\s*(submission-[A-Za-z0-9_-]+)", output) or [None, ""])[1].strip()
    uploads = [
        {"source": m.group(1).strip(), "target": m.group(2).strip()}
        for m in re.finditer(r"\[submit\]\s+(.+?)\s+->\s+(.+)", output)
    ]
    return {"url": url, "repo": repo, "submission": submission, "uploads": uploads}


def is_duplicate_submit_output(output: str) -> bool:
    text = output.lower()
    return (
        "duplicate redacted session artifact" in text
        or "same source session changed too little" in text
        or "http 409" in text and "duplicate" in text
    )


def duplicate_submit_detail(output: str) -> str:
    for match in re.finditer(r"\{[^\n]*\"detail\"[^\n]*\}", output):
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        detail = str(data.get("detail") or "").strip()
        if detail:
            return detail
    if "same source session changed too little" in output.lower():
        return "same source session changed too little since prior submission"
    if "duplicate redacted session artifact" in output.lower():
        return "duplicate redacted session artifact"
    return ""


def friendly_submit_error(output: str) -> str:
    text = output.lower()
    if (
        "repository not found" in text
        or "invalid username or password" in text
        or "401 client error" in text
        or "private or gated repo" in text
    ):
        return (
            "Upload is not configured for public donors yet. The redacted file is verified "
            "and saved locally, but submitting to the private staging repo requires a "
            "ContextEcho relay URL or a maintainer Hugging Face token. Ask the maintainer "
            "for CONTEXTECHO_RELAY_URL, or set CONTEXTECHO_DONATE_TOKEN if you are a maintainer."
        )
    if "check contextecho_relay_url" in text:
        return (
            "Relay upload failed. Check CONTEXTECHO_RELAY_URL, then submit again. "
            "Your verified redacted files are still saved locally."
        )
    return output or "submit failed"


def run_submit_with_heartbeats(session: Path, emit=None) -> tuple[int, str]:
    """Run the blocking submit command while keeping the browser stream alive."""
    results: queue.Queue[tuple[str, int, str, str]] = queue.Queue(maxsize=1)

    def worker() -> None:
        import contextlib
        import io

        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                rc = submit_mod.main([str(session)])
            results.put(("done", rc, buf.getvalue(), ""))
        except Exception as exc:
            results.put(("error", 1, buf.getvalue(), str(exc)))

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    last_emit = time.monotonic()
    stream_open = True
    while True:
        try:
            kind, rc, output, error_text = results.get(timeout=5)
            if kind == "error":
                raise RuntimeError(error_text or "submit failed")
            return rc, output
        except queue.Empty:
            if not emit or not stream_open:
                continue
            now = time.monotonic()
            if now - last_emit < 15:
                continue
            last_emit = now
            try:
                emit({
                    "event": "progress",
                    "percent": 72,
                    "message": "Uploading donation; large sessions can take several minutes...",
                })
            except ClientDisconnected:
                stream_open = False


def update_submit_job(job_id: str, **updates) -> None:
    with SUBMIT_JOBS_LOCK:
        job = SUBMIT_JOBS.setdefault(job_id, {})
        job.update(updates)
        job["updated_at"] = time.time()


def get_submit_job(job_id: str) -> dict:
    with SUBMIT_JOBS_LOCK:
        return dict(SUBMIT_JOBS.get(job_id) or {})


def write_receipt(session: Path, source_path: str | Path, output: str) -> tuple[Path, dict]:
    stem = session.stem.replace(".redacted", "")
    manifest_path = session.with_name(f"{stem}.manifest.json")
    manifest = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            manifest = {}
    parsed = parse_submit_output(output)
    receipt = {
        "submitted_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "submission": parsed["submission"],
        "review_url": parsed["url"],
        "target_repo": parsed["repo"],
        "source_path": str(source_path or ""),
        "redacted_file": str(session),
        "contributor": manifest.get("contributor", "anonymous"),
        "credit_name": manifest.get("credit_name", manifest.get("contributor", "anonymous")),
        "public_anonymous": bool(manifest.get("public_anonymous")),
        "contributor_email": manifest.get("contributor_email", ""),
        "institute": manifest.get("contributor_institute", ""),
        "agent": manifest.get("agent", ""),
        "model": manifest.get("model", ""),
        "org": manifest.get("org", ""),
        "privacy_tier": manifest.get("privacy_tier", "full_redacted"),
        "records": manifest.get("records", ""),
        "turns": manifest.get("turns", ""),
        "compactions": manifest.get("compactions", ""),
        "uploads": parsed["uploads"],
    }
    lines = [
        "# ContextEcho Donation Receipt",
        "",
        "This receipt confirms that the local donation tool submitted verified redacted artifacts for maintainer review.",
        "",
        f"- Submitted UTC: {receipt['submitted_utc']}",
        f"- Submission: {receipt['submission'] or 'unknown'}",
        f"- Credit name: {receipt['credit_name']}",
        f"- Public leaderboard: {'anonymous' if receipt['public_anonymous'] else receipt['credit_name']}",
        f"- Email: {receipt['contributor_email'] or 'not provided'}",
        f"- Institute: {receipt['institute'] or 'not provided'}",
        f"- Agent/model: {receipt['agent']} / {receipt['model']}",
        f"- Privacy tier: {receipt['privacy_tier']}",
        f"- User turns: {receipt['turns']}",
        f"- Records: {receipt['records']}",
        f"- Context compactions: {receipt['compactions']}",
        "",
        "Uploaded artifacts:",
    ]
    for item in receipt["uploads"]:
        lines.append(f"- {item['source']}")
    lines.extend([
        "",
        "Status: pending maintainer review. Credit is awarded after acceptance.",
        "",
    ])
    receipt_path = session.with_name("DONATION_RECEIPT.md")
    receipt_path.write_text("\n".join(lines), encoding="utf-8")
    return receipt_path, receipt


def annotate_donated(sessions: list[dict]) -> list[dict]:
    out = [dict(session) for session in sessions]
    for row in out:
        row["donated"] = False
        row["donated_before"] = False
        row["donated_turns"] = 0
        row["new_turns"] = 0
        row["update_ready"] = False
        row["relay_submission_id"] = ""
        row["relay_public_session_id"] = ""
        row["local_credit_name"] = ""
        row["local_contributor_email"] = ""
        row["local_institute"] = ""
        row["local_public_anonymous"] = False
    relay_statuses = relay_donation_status(out)
    relay_checked = bool(relay_statuses) and len(relay_statuses) == len(out)
    for row, status in zip(out, relay_statuses):
        if relay_checked:
            row["relay_checked"] = True
        if not status.get("received"):
            continue
        previous_turns = int(status.get("turns") or 0)
        new_turns = int(status.get("new_turns") or 0)
        update_ready = bool(status.get("update_ready"))
        row["donated_before"] = True
        row["donated_turns"] = previous_turns
        row["new_turns"] = new_turns
        row["update_ready"] = update_ready
        row["donated"] = not update_ready
        row["relay_received"] = True
        relay_submission_id = normalize_submission_id(status.get("submission_id", ""))
        row["relay_submission_id"] = relay_submission_id if is_support_submission_id(relay_submission_id) else ""
        row["relay_public_session_id"] = "" if is_support_submission_id(relay_submission_id) else relay_submission_id
        row["local_credit_name"] = status.get("credit_name", "")
        row["local_contributor_email"] = status.get("contributor_email", "")
        row["local_institute"] = status.get("contributor_institute", "")
        row["local_public_anonymous"] = bool(status.get("public_anonymous"))
    if relay_checked:
        for row in out:
            row.setdefault("relay_checked", True)
    return out


def infer_domain(info: dict) -> str:
    project = str(info.get("project", "")).lower()
    if any(x in project for x in ("dashboard", "frontend", "web")):
        return "web-frontend"
    if any(x in project for x in ("paper", "research", "analysis")):
        return "research"
    if any(x in project for x in ("doc", "slide", "deck")):
        return "docs-writing"
    return "agentic-coding"


def infer_language(info: dict) -> str:
    project = str(info.get("project", "")).lower()
    if any(x in project for x in ("paper", "doc", "slide", "deck")):
        return "Markdown/docs"
    return "mixed"


def _fetch_json(url: str) -> dict:
    req = Request(url, headers={"User-Agent": "ContextEcho-donate-web/0.1"})
    with urlopen(req, timeout=5) as r:
        return json.loads(r.read().decode("utf-8"))


def _fetch_text(url: str) -> str:
    req = Request(url, headers={"User-Agent": "ContextEcho-donate-web/0.1"})
    with urlopen(req, timeout=5) as r:
        return r.read().decode("utf-8", errors="replace")


def _parse_donated_sessions(readme: str) -> int | None:
    patterns = [
        r"(\d[\d,]*)\s+redacted\s+donor\s+sessions?",
        r"(\d[\d,]*)\s+donated\s+sessions?",
        r"donated\s+sessions?\D{0,20}(\d[\d,]*)",
    ]
    for pattern in patterns:
        m = re.search(pattern, readme, re.IGNORECASE)
        if m:
            return int(m.group(1).replace(",", ""))
    return None


def _parse_contributor_leaderboard(markdown: str) -> list[dict]:
    rows: list[dict] = []
    in_table = False
    for line in markdown.splitlines():
        if line.startswith("| Rank | Contributor | Sessions | Turns |"):
            in_table = True
            continue
        if not in_table:
            continue
        if rows and not line.startswith("|"):
            break
        if line.startswith("|:") or not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 7:
            break
        rank, contributor, sessions, turns, agents, models, points = cells[:7]
        rows.append({
            "rank": rank,
            "contributor": re.sub(r"\*\*", "", contributor),
            "sessions": sessions,
            "sessions_num": int(sessions) if sessions.isdigit() else 0,
            "turns": turns,
            "turns_num": int(turns.replace(",", "")) if turns.replace(",", "").isdigit() else 0,
            "agents": agents,
            "models": models,
            "points": points,
            "points_num": int(points) if points.isdigit() else None,
        })
    return rows


def _parse_dataset_card_coverage(markdown: str) -> dict:
    fields: dict[str, str] = {}
    composition: dict[str, str] = {}
    section = ""
    for line in markdown.splitlines():
        if line.startswith("## "):
            section = line.removeprefix("## ").strip()
            continue
        if not line.startswith("|") or line.startswith("|-"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 2 or cells[0] in {"Field", "Axis"}:
            continue
        if section == "Dataset Summary":
            fields[cells[0]] = cells[1]
        elif section == "Composition":
            composition[cells[0]] = cells[1]

    def as_int(label: str) -> int:
        value = fields.get(label, "")
        m = re.search(r"\d[\d,]*", value)
        return int(m.group(0).replace(",", "")) if m else 0

    def unique_count(axis: str) -> int:
        value = composition.get(axis, "")
        if not value or value.lower() == "none yet":
            return 0
        m = re.match(r"\s*(\d[\d,]*)\b", value)
        if m:
            return int(m.group(1).replace(",", ""))
        return len([part for part in value.split(",") if part.strip() and not part.strip().startswith("+")])

    return {
        "sessions": as_int("Active public/candidate sessions tracked locally") or as_int("Public v1 founding sessions"),
        "contributors": as_int("Public contributors in leaderboard"),
        "institutions": unique_count("Institution coverage") or unique_count("Public contributor institutions"),
        "agents": unique_count("Agent / harness"),
        "models": unique_count("Model family"),
        "organizations": unique_count("Model organization"),
        "domains": unique_count("Task domain"),
        "languages": unique_count("Primary language"),
        "compactions": as_int("Active public/candidate context compactions tracked locally"),
        "turns": as_int("Active public/candidate user turns tracked locally"),
    }


def _load_contributors_markdown(local_path: Path | None = None) -> str:
    if local_path is None:
        local_path = Path(__file__).resolve().parents[1] / "CONTRIBUTORS.md"
        try:
            return _fetch_text("https://raw.githubusercontent.com/Accenture/ContextEcho/main/CONTRIBUTORS.md")
        except Exception:
            return local_path.read_text(encoding="utf-8")
    try:
        return local_path.read_text(encoding="utf-8")
    except Exception:
        return _fetch_text("https://raw.githubusercontent.com/Accenture/ContextEcho/main/CONTRIBUTORS.md")


def _load_dataset_card_markdown(local_path: Path | None = None) -> str:
    if local_path is None:
        local_path = Path(__file__).resolve().parents[1] / "DATASET_CARD.md"
        try:
            return _fetch_text("https://raw.githubusercontent.com/Accenture/ContextEcho/main/DATASET_CARD.md")
        except Exception:
            return local_path.read_text(encoding="utf-8")
    try:
        return local_path.read_text(encoding="utf-8")
    except Exception:
        return _fetch_text("https://raw.githubusercontent.com/Accenture/ContextEcho/main/DATASET_CARD.md")


def _load_tracked_project_stats(local_path: Path | None = None) -> dict:
    if local_path is None:
        local_path = Path(__file__).resolve().parents[1] / "docs" / "project_stats.json"
        try:
            return _fetch_json("https://raw.githubusercontent.com/Accenture/ContextEcho/main/docs/project_stats.json")
        except Exception:
            return json.loads(local_path.read_text(encoding="utf-8"))
    try:
        return json.loads(local_path.read_text(encoding="utf-8"))
    except Exception:
        return _fetch_json("https://raw.githubusercontent.com/Accenture/ContextEcho/main/docs/project_stats.json")


def project_stats() -> dict:
    """Best-effort public project stats. Never block the donation flow."""
    stats = {
        "github_stars": None,
        "donated_sessions": None,
        "dataset_total_downloads": None,
        "dataset_downloads": None,
        "dataset_likes": None,
        "leaderboard": [],
        "coverage": {},
    }
    try:
        gh = _fetch_json("https://api.github.com/repos/Accenture/ContextEcho")
        stats["github_stars"] = gh.get("stargazers_count")
    except Exception:
        pass
    try:
        readme = _fetch_text("https://raw.githubusercontent.com/Accenture/ContextEcho/main/README.md")
        stats["donated_sessions"] = _parse_donated_sessions(readme)
    except Exception:
        pass
    try:
        tracked = _load_tracked_project_stats()
        stats["dataset_total_downloads"] = tracked.get("dataset_total_downloads")
    except Exception:
        pass
    try:
        hf = _fetch_json("https://huggingface.co/api/datasets/contextecho2026/persona-drift-contextecho")
        stats["dataset_downloads"] = hf.get("downloads") or hf.get("downloadsAllTime")
        stats["dataset_likes"] = hf.get("likes")
    except Exception:
        pass
    try:
        text = _load_contributors_markdown()
        stats["leaderboard"] = _parse_contributor_leaderboard(text)
    except Exception:
        pass
    try:
        stats["coverage"] = _parse_dataset_card_coverage(_load_dataset_card_markdown())
    except Exception:
        pass
    return stats


def _safe_repair_terms_from_report(path: Path, verify_report: dict) -> dict[str, str]:
    """Internal repair terms for residual verify findings.

    Values returned here are used locally to rewrite the redacted file. They are
    intentionally not sent to the browser because detect-secrets findings can be
    real credentials.
    """
    blocking = verify_report.get("blocking") or {}
    terms: dict[str, str] = {}
    for category in ("email", "home_path", "api_key"):
        for value in blocking.get(category) or []:
            if isinstance(value, str) and value and not value.startswith("<"):
                terms[value] = category
    if blocking.get("detect_secrets"):
        for item in verify_mod.detect_secret_findings(path):
            value = str(item.get("secret_value") or "")
            if value and not value.startswith("<"):
                terms[value] = "credential_pattern"
    return terms


def _repair_malformed_jsonl_lines(path: Path) -> int:
    """Wrap malformed redacted lines so the donation remains valid JSONL."""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    repaired = 0
    out: list[str] = []
    for line_no, line in enumerate(lines, 1):
        if not line.strip():
            out.append(line)
            continue
        try:
            json.loads(line)
            out.append(line)
        except json.JSONDecodeError:
            out.append(json.dumps(
                {"type": "redacted_raw_line", "line_number": line_no, "text": line},
                ensure_ascii=False,
                separators=(",", ":"),
            ))
            repaired += 1
    if repaired:
        path.write_text("\n".join(out) + "\n", encoding="utf-8")
    return repaired


def _auto_repair_until_verified(
    path: Path,
    verify_report: dict,
    stats: dict,
    emit=None,
) -> tuple[dict, dict, int]:
    """Bounded automatic repair loop after verify finds exact residual values."""
    repair_passes = 0
    current_report = verify_report
    for pass_no in range(1, MAX_AUTO_REPAIR_PASSES + 1):
        if current_report.get("passed"):
            break
        blocking = current_report.get("blocking") or {}
        if blocking.get("malformed_jsonl"):
            repair_passes += 1
            if emit:
                emit({
                    "event": "repair",
                    "percent": min(98, 90 + pass_no),
                    "message": f"Auto-repair {pass_no}/{MAX_AUTO_REPAIR_PASSES}: normalizing malformed redacted JSONL lines...",
                })
            repaired = _repair_malformed_jsonl_lines(path)
            if repaired:
                stats["malformed_jsonl_wrapped"] = int(stats.get("malformed_jsonl_wrapped", 0) or 0) + repaired
            if emit:
                emit({
                    "event": "verify",
                    "percent": min(99, 93 + pass_no),
                    "message": f"Verifying after auto-repair {pass_no}/{MAX_AUTO_REPAIR_PASSES}...",
                })
            current_report = verify_mod.verify_session(path)
            continue
        repair_terms = _safe_repair_terms_from_report(path, current_report)
        if not repair_terms:
            break
        repair_passes += 1
        if emit:
            emit({
                "event": "repair",
                "percent": min(98, 90 + pass_no),
                "message": f"Auto-repair {pass_no}/{MAX_AUTO_REPAIR_PASSES}: redacting residual private patterns found by verify...",
            })
        repair_stats = redact_mod.apply_scrub_terms_to_file(path, path, set(repair_terms))
        for key, value in repair_stats.items():
            if key.startswith("private_word:"):
                term = key[len("private_word:"):]
                category = repair_terms.get(term, "private_pattern")
                aggregate_key = "credential_pattern" if category in {"api_key", "credential_pattern"} else "path_or_private_pattern"
                stats[aggregate_key] = int(stats.get(aggregate_key, 0) or 0) + int(value or 0)
                continue
            stats[key] = int(stats.get(key, 0) or 0) + int(value or 0)
        if emit:
            emit({
                "event": "verify",
                "percent": min(99, 93 + pass_no),
                "message": f"Verifying after auto-repair {pass_no}/{MAX_AUTO_REPAIR_PASSES}...",
            })
        current_report = verify_mod.verify_session(path)
    return current_report, stats, repair_passes


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>ContextEcho Donation Wizard</title>
  <style>
    :root { --ink:#111b18; --muted:#5f6662; --line:#dfe2da; --paper:#f6f7ed; --card:#fffef8; --accent:#17713f; --accent2:#e8a823; --soft:#eef6e8; --bad:#a63124; }
    body { margin:0; font:14px/1.35 ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color:var(--ink); background:radial-gradient(circle at 8% 0%,#fff7d5 0 18%,transparent 34%), radial-gradient(circle at 90% 8%,#dceedd 0 17%,transparent 34%), linear-gradient(135deg,#f7f3dd,#eef6ea 52%,#e5f1ee); }
    main { max-width:1480px; margin:0 auto; padding:14px 24px 28px; }
    h1 { margin:0 0 5px; font-size:clamp(28px,3.2vw,38px); letter-spacing:-.055em; line-height:1.02; }
    h2 { margin:0 0 6px; font-size:24px; letter-spacing:-.04em; line-height:1.08; }
    .hero, .card, .bottom-nav { background:rgba(255,255,250,.9); border:1px solid rgba(127,138,119,.28); border-radius:20px; box-shadow:0 14px 42px rgba(43,59,37,.11); backdrop-filter:blur(10px); }
    .hero { padding:24px 34px 20px; position:relative; overflow:hidden; }
    .hero-top { display:flex; justify-content:space-between; gap:20px; align-items:flex-start; }
    .card { padding:26px 34px; }
    .card.step { margin-top:16px; }
    .step { display:none; }
    .step.active { display:block; }
    .hero-flow { display:grid; grid-template-columns:minmax(560px,1fr) minmax(420px,.8fr); gap:26px; align-items:center; margin-top:20px; }
    .steps { display:grid; grid-template-columns:1fr 1fr 1fr; gap:14px; align-items:center; min-width:0; }
    .step-pill { position:relative; display:flex; align-items:center; gap:11px; color:#6a6f6b; font-size:15px; font-weight:850; }
    .step-pill:after { content:""; height:3px; flex:1; border-radius:999px; background:#e2e4df; margin-left:6px; }
    .step-pill:last-child:after { display:none; }
    .step-num { display:grid; place-items:center; width:36px; height:36px; border-radius:999px; background:#e8e9e6; color:#555b58; font-weight:950; }
    .step-pill.active { color:var(--accent); }
    .step-pill.active .step-num, .step-pill.done .step-num { background:var(--accent); color:white; box-shadow:0 8px 20px rgba(23,113,63,.24); }
    .step-pill.active:after, .step-pill.done:after { background:var(--accent); }
    .hero-side { display:flex; align-items:flex-start; gap:16px; justify-content:flex-end; max-width:780px; }
    .privacy-note { color:var(--muted); font-size:13px; line-height:1.35; text-align:right; max-width:560px; padding-top:4px; white-space:nowrap; }
    .privacy-links { display:flex; justify-content:flex-end; gap:12px; margin-top:5px; }
    .privacy-links a { color:#13552f; font-size:12px; font-weight:900; text-decoration:none; }
    .privacy-links a:hover, .privacy-links a:focus { text-decoration:underline; }
    .privacy-note strong { color:#13552f; }
    .hero-progress { display:flex; align-items:center; gap:14px; padding-top:4px; min-width:190px; justify-content:flex-end; }
    .progress-label { text-align:right; color:var(--muted); font-size:14px; }
    .progress-label strong { display:block; color:var(--accent); font-size:18px; }
    .ring { --pct:25; width:78px; height:78px; border-radius:50%; display:grid; place-items:center; background:conic-gradient(var(--accent) calc(var(--pct) * 1%), #e9ebe5 0); position:relative; font-weight:950; color:var(--accent); font-size:18px; }
    .ring:before { content:""; position:absolute; inset:7px; border-radius:50%; background:#fffef8; box-shadow:inset 0 0 0 1px rgba(0,0,0,.03); }
    .ring span { position:relative; }
    .muted { color:var(--muted); }
    .row { display:flex; gap:12px; flex-wrap:wrap; align-items:center; }
    button { border:0; border-radius:10px; padding:12px 18px; background:var(--accent); color:white; font-weight:900; cursor:pointer; box-shadow:0 10px 20px rgba(23,113,63,.2); font-size:14px; }
    button.secondary { background:#e8eddc; color:var(--ink); box-shadow:none; }
    button:hover:not(:disabled) { transform:translateY(-1px); }
    button:disabled { opacity:.5; cursor:not-allowed; }
    body.is-processing button { opacity:.5; cursor:not-allowed; pointer-events:none; }
    input, textarea { width:100%; box-sizing:border-box; border:1px solid var(--line); border-radius:14px; padding:11px 13px; background:white; color:var(--ink); font:inherit; }
    input:focus, textarea:focus { outline:3px solid rgba(31,111,67,.16); border-color:#7cb67d; }
    label { display:block; font-weight:700; margin:12px 0 6px; }
    .pick-grid { display:grid; grid-template-columns:minmax(300px,.62fr) minmax(620px,1.38fr); gap:22px; margin-top:16px; }
    .pick-intro { min-height:342px; }
    .intro-head { display:flex; gap:22px; align-items:flex-start; padding-bottom:20px; border-bottom:1px solid var(--line); }
    .folder-icon { width:76px; height:76px; border-radius:18px; display:grid; place-items:center; background:linear-gradient(135deg,#eef6d4,#f7faeb); }
    .folder-icon:before { content:""; width:42px; height:29px; border:3px solid var(--accent); border-radius:6px; box-sizing:border-box; box-shadow:0 -10px 0 -6px var(--accent); }
    .stats { display:grid; grid-template-columns:repeat(4,minmax(74px,1fr)); gap:10px; align-items:stretch; min-width:420px; }
    .stat-card { min-height:78px; box-sizing:border-box; display:flex; flex-direction:column; justify-content:center; align-items:center; text-align:center; background:#fff; border:1px solid #e3e7df; border-radius:10px; padding:8px 9px; box-shadow:0 5px 14px rgba(43,59,37,.06); color:inherit; text-decoration:none; transition:.15s ease; }
    a.stat-card:hover, a.stat-card:focus { transform:translateY(-1px); border-color:#c9dcc5; box-shadow:0 8px 18px rgba(43,59,37,.1); outline:none; }
    a.stat-card:focus-visible { outline:3px solid rgba(31,111,67,.18); }
    .stat-icon { width:28px; height:28px; margin:0 auto 5px; display:grid; place-items:center; border-radius:50%; background:#f6edd6; color:#d28b00; }
    .stat-icon svg { width:15px; height:15px; display:block; stroke:currentColor; fill:none; stroke-width:2.5; stroke-linecap:round; stroke-linejoin:round; }
    .stat-icon .icon-fill { fill:currentColor; stroke:none; }
    .stat-icon[data-icon="star"] { background:#f6edd6; color:#d28b00; }
    .stat-icon[data-icon="download"] { background:#e9f2e5; color:var(--accent); }
    .stat-icon[data-icon="heart"] { background:#f6eadb; color:#dc4b30; }
    .stat-icon[data-icon="gift"] { background:#efedf5; color:#7657a8; }
    .stat-value { font-size:17px; line-height:1; font-weight:950; letter-spacing:-.025em; }
    .stat-label { margin-top:4px; color:#3d4440; font-size:9px; font-weight:800; line-height:1.1; }
    .composition-panel { margin:0 0 14px; border:1px solid #dfe7dc; border-radius:14px; background:#fffefb; padding:14px; }
    .composition-head { margin-bottom:9px; }
    .composition-title { font-size:16px; font-weight:950; color:#14241d; }
    .composition-subtitle { color:#657069; font-size:11px; margin-top:2px; }
    .composition-list { display:grid; gap:0; }
    .composition-row { display:grid; grid-template-columns:34px minmax(82px,1fr) minmax(72px,.9fr) 72px; gap:8px; align-items:center; padding:7px 0; border-bottom:1px dashed #e5ebe1; }
    .composition-row:last-child { border-bottom:0; }
    .composition-icon { width:28px; height:28px; border-radius:50%; display:grid; place-items:center; background:#e9f4e8; color:#17713f; }
    .composition-icon svg { width:15px; height:15px; display:block; stroke:currentColor; fill:none; stroke-width:2.5; stroke-linecap:round; stroke-linejoin:round; }
    .composition-label { min-width:0; color:#14241d; font-weight:900; font-size:12px; line-height:1.05; }
    .composition-label small { display:block; color:#657069; font-size:10px; font-weight:700; margin-top:2px; }
    .composition-track { height:8px; border-radius:999px; background:#f0f1ef; overflow:hidden; }
    .composition-fill { height:100%; min-width:0; border-radius:999px; background:linear-gradient(90deg,#1b8a4b,#17713f); box-shadow:0 2px 7px rgba(23,113,63,.18); }
    .composition-value { color:#14241d; text-align:right; font-weight:950; font-size:15px; }
    .support-card { display:flex; gap:14px; align-items:center; border:1px solid #dce7d2; border-radius:16px; padding:10px 12px; background:linear-gradient(135deg,#fff8df,#eef8e8); overflow:visible; position:relative; z-index:5; }
    .bow-mascot { position:relative; flex:0 0 46px; width:46px; height:48px; }
    .bow-head { position:absolute; left:12px; top:2px; width:23px; height:23px; border-radius:50%; background:#f1bf86; box-shadow:inset 0 -3px 0 rgba(0,0,0,.08); transform-origin:50% 100%; animation:bowHead 2.4s ease-in-out infinite; }
    .bow-head:before, .bow-head:after { content:""; position:absolute; top:9px; width:3px; height:3px; border-radius:50%; background:#17201c; }
    .bow-head:before { left:7px; }
    .bow-head:after { right:7px; }
    .bow-body { position:absolute; left:10px; top:24px; width:27px; height:20px; border-radius:10px 10px 6px 6px; background:#17713f; transform-origin:50% 0; animation:bowBody 2.4s ease-in-out infinite; }
    .bow-hands { position:absolute; left:4px; top:34px; width:39px; height:9px; border-radius:999px; background:#f1bf86; transform-origin:50% 50%; animation:bowHands 2.4s ease-in-out infinite; }
    .bow-star { position:absolute; right:0; top:0; color:#d28b00; font-size:14px; line-height:1; animation:twinkle 1.4s ease-in-out infinite; }
    @keyframes bowHead { 0%,62%,100% { transform:rotate(0deg) translateY(0); } 28%,42% { transform:rotate(18deg) translateY(7px); } }
    @keyframes bowBody { 0%,62%,100% { transform:rotate(0deg); } 28%,42% { transform:rotate(10deg); } }
    @keyframes bowHands { 0%,62%,100% { transform:translateY(0) scaleX(1); } 28%,42% { transform:translateY(5px) scaleX(1.08); } }
    @keyframes twinkle { 0%,100% { transform:scale(.9) rotate(0deg); opacity:.72; } 50% { transform:scale(1.18) rotate(14deg); opacity:1; } }
    @media (prefers-reduced-motion: reduce) { .bow-head,.bow-body,.bow-hands,.bow-star { animation:none; } }
    .support-main { min-width:0; flex:1; }
    .support-title { font-weight:950; color:#13552f; }
    .support-copy { color:var(--muted); font-size:12px; margin-top:2px; }
    .discover-main { width:100%; border-radius:10px; padding:14px 20px; font-size:18px; box-shadow:0 12px 24px rgba(23,113,63,.2); }
    .reset-donated { margin-top:12px; justify-content:center; }
    .reset-donated button { padding:8px 12px; font-size:12px; }
    .sessions-card { min-height:342px; }
    .session-head { display:flex; justify-content:space-between; align-items:flex-start; gap:18px; margin-bottom:24px; }
    .session-head > div:first-child { min-width:0; }
    .session-head h2 { font-size:24px; line-height:1.05; }
    .session-subtitle { margin-top:6px; color:#59615d; font-size:14px; white-space:nowrap; }
    .session-summary { display:flex; flex-wrap:nowrap; gap:10px; justify-content:flex-end; align-items:stretch; flex:0 0 auto; }
    .count-badge { min-width:66px; border:1px solid #d9ded6; border-radius:12px; padding:8px 12px; color:#13231c; background:#fff; font-weight:900; text-align:center; box-shadow:0 8px 24px rgba(38,54,44,.05); }
    .count-badge strong { display:block; color:var(--accent); font-size:24px; line-height:1; }
    .count-badge span { display:block; margin-top:2px; font-size:12px; color:#38423d; }
    .fit-summary { display:flex; flex-wrap:nowrap; gap:10px; justify-content:flex-end; }
    .fit-chip { border-radius:10px; padding:12px 14px; font-size:14px; font-weight:950; background:#edf1e4; color:#44504a; display:inline-flex; align-items:center; justify-content:center; min-width:118px; white-space:nowrap; }
    .count-badge[data-tooltip], .fit-chip[data-tooltip] { position:relative; cursor:help; }
    .count-badge[data-tooltip]:after, .fit-chip[data-tooltip]:after { content:attr(data-tooltip); position:absolute; left:50%; top:calc(100% + 8px); transform:translateX(-50%); opacity:0; pointer-events:none; z-index:30; min-width:150px; padding:8px 10px; border-radius:8px; background:#14241d; color:#fff; box-shadow:0 8px 20px rgba(15,25,20,.18); font-size:12px; font-weight:800; line-height:1.35; text-align:left; white-space:pre-line; }
    .count-badge[data-tooltip]:before, .fit-chip[data-tooltip]:before { content:""; position:absolute; left:50%; top:100%; transform:translateX(-50%); opacity:0; pointer-events:none; z-index:31; border:5px solid transparent; border-bottom-color:#14241d; }
    .count-badge[data-tooltip]:hover:after, .count-badge[data-tooltip]:hover:before, .fit-chip[data-tooltip]:hover:after, .fit-chip[data-tooltip]:hover:before { opacity:1; }
    .fit-chip.ready { background:#dff1d9; color:#13552f; }
    .fit-chip.best { background:#dff1d9; color:#13552f; }
    .fit-chip.good { background:#dff1d9; color:#13552f; }
    .fit-chip.improve { background:#f3e5d2; color:#7a420a; }
    .fit-chip.donated { background:#dceafa; color:#1e4f87; }
    .session-tools { display:flex; gap:16px; align-items:center; justify-content:space-between; margin:-8px 0 12px; }
    .session-search { flex:0 1 360px; max-width:360px; min-width:220px; }
    .session-list { border:1px solid var(--line); border-radius:14px; overflow:hidden; background:white; }
    .session-table-head, .session-row { display:grid; grid-template-columns:40px minmax(280px,1fr) 146px 96px 92px 88px 16px; gap:18px; align-items:center; }
    .session-table-head { padding:10px 16px; background:#f2f5ef; color:#5a625d; font-size:11px; font-weight:950; text-transform:uppercase; letter-spacing:.04em; border-bottom:1px solid var(--line); }
    .session-table-head > div { display:flex; align-items:center; justify-content:center; gap:5px; text-align:center; }
    .sort-header { border:0; background:transparent; box-shadow:none; color:inherit; padding:0; min-width:0; border-radius:6px; display:inline-flex; align-items:center; gap:5px; font:inherit; font-weight:950; letter-spacing:inherit; text-transform:inherit; cursor:pointer; }
    .sort-header:hover:not(:disabled), .sort-header:focus-visible { background:#e5ecdf; color:#123d29; transform:none; outline:none; }
    .sort-arrow { min-width:9px; color:#17713f; font-size:10px; line-height:1; }
    .header-footnote { color:#1f6f43; font-size:10px; font-weight:950; vertical-align:super; letter-spacing:0; margin-left:2px; }
    .table-note { color:var(--muted); font-size:12px; text-align:right; white-space:nowrap; }
    .session-row { min-height:52px; padding:7px 16px; border-bottom:1px solid var(--line); cursor:pointer; transition:.15s ease; }
    .session-row > * { justify-self:center; text-align:center; }
    .session-row:last-child { border-bottom:0; }
    .session-row:hover, .session-row.selected { background:#f4f8ef; }
    .session-row.selected { box-shadow:inset 4px 0 0 var(--accent); }
    .session-row.donated-history-row { background:#e7eee2; box-shadow:inset 5px 0 0 #7f9a7a; }
    .session-row.donated-history-row:hover { background:#e1eadb; }
    .session-row.donated-row { cursor:not-allowed; background:#e7eee2; box-shadow:inset 5px 0 0 #7f9a7a; }
    .session-row.donated-row:hover { background:#e1eadb; }
    .session-row.improve-row { cursor:not-allowed; background:#fff9f0; opacity:.82; }
    .session-row.improve-row:hover { background:#fff9f0; }
    .session-main { min-width:0; justify-self:stretch; text-align:left; }
    .session-title-line { font-weight:950; font-size:13px; color:#0e1714; line-height:1.15; overflow-wrap:anywhere; }
    .session-chip-row { display:flex; align-items:center; gap:6px; flex-wrap:wrap; margin-top:4px; }
    .session-chip-row:empty { display:none; }
    .session-title-row { display:flex; align-items:center; gap:8px; flex-wrap:wrap; }
    .all-donated-note { margin:12px; border:1px solid #b9d6b0; background:#f2fbef; border-radius:14px; padding:14px 16px; color:#145832; font-weight:900; }
    .all-donated-note span { display:block; margin-top:4px; color:#52605a; font-weight:650; }
    .empty-sessions.thanks { color:#145832; font-weight:900; background:#f8fcf4; }
    .empty-sessions.thanks span { display:block; margin-top:6px; color:#52605a; font-weight:650; }
    .session-icon { width:30px; height:30px; display:grid; place-items:center; border-radius:8px; background:#e8f1e4; color:var(--accent); font-weight:950; font-size:15px; }
    .donated-history-row .session-icon, .donated-row .session-icon { background:#cfe0c8; color:#0f6a3a; }
    .session-title { font-weight:900; font-size:14px; }
    .session-date { color:#5f6662; font-size:13px; }
    .session-num { font-weight:950; font-size:15px; color:#07110d; }
    .session-fit { display:flex; justify-content:center; }
    .session-fit .pill { text-transform:capitalize; }
    .session-chevron { color:#5f6662; font-size:20px; line-height:1; justify-self:end; border:0; background:transparent; box-shadow:none; padding:2px 4px; min-width:0; border-radius:8px; }
    .session-chevron:hover:not(:disabled), .session-row:hover .session-chevron { color:#13552f; background:#eaf4e5; transform:none; }
    .session-menu { position:fixed; z-index:1000; min-width:168px; padding:6px; border:1px solid var(--line); border-radius:12px; background:#fffef8; box-shadow:0 16px 42px rgba(25,38,31,.18); display:none; }
    .session-menu.show { display:block; }
    .session-menu button { display:block; width:100%; border:0; border-radius:8px; background:transparent; box-shadow:none; color:#14241d; text-align:left; padding:9px 10px; font-size:13px; font-weight:900; }
    .session-menu button:hover { background:#eaf4e5; transform:none; }
    .session-menu button.danger { color:#7a2c1f; }
    .empty-sessions { padding:26px; text-align:center; color:var(--muted); }
    .bottom-nav { margin-top:16px; padding:12px 34px; display:flex; justify-content:space-between; align-items:center; gap:16px; }
    .tip { display:flex; gap:12px; align-items:center; color:#3f4843; }
    .tip:before { content:"?"; display:grid; place-items:center; width:22px; height:22px; border-radius:50%; border:2px solid var(--accent); color:var(--accent); font-weight:950; }
    .next-button { min-width:170px; font-size:16px; }
    .pill { display:inline-flex; align-items:center; gap:5px; border-radius:999px; padding:4px 9px; font-size:12px; font-weight:850; background:#edf1e4; line-height:1; box-shadow:inset 0 0 0 1px rgba(24,38,30,.05); }
    .pill.best { background:#dff1d9; color:#13552f; }
    .pill.good { background:#dff1d9; color:#13552f; }
    .pill.improve { background:#f3e5d2; color:#7a420a; }
    .pill.donated { background:#cfe1f5; color:#163f70; }
    .pill.support-id { background:#eef3e9; color:#45524b; cursor:pointer; }
    .pill.update-info { background:#eaf4e5; color:#13552f; cursor:pointer; }
    .session-list .pill { padding:3px 7px; font-size:11px; }
    .session-list .fit-star, .session-list .fit-arrow { font-size:11px; }
    .fit-legend { display:flex; flex-wrap:wrap; justify-content:space-between; gap:12px; margin-top:14px; padding:10px 14px; border:1px solid var(--line); border-radius:12px; background:#fbfcf8; color:var(--muted); font-size:12px; line-height:1.35; }
    .legend-items { display:flex; flex-wrap:wrap; gap:16px; align-items:center; }
    .legend-item { display:inline-flex; align-items:center; gap:7px; }
    .fit-star { color:#1f7a47; font-size:12px; line-height:1; }
    .fit-arrow { color:#a75009; font-size:13px; line-height:1; }
    .session-pager { display:none; margin-top:22px; align-items:center; justify-content:center; gap:24px; }
    .session-pager button { min-width:118px; }
    .session-pager #nextPage { background:var(--accent); color:white; box-shadow:0 10px 20px rgba(23,113,63,.22); }
    .session-pager #pageInfo { font-size:14px; color:#555f59; }
    .inline-status { margin-top:10px; color:var(--muted); font-size:14px; }
    .inline-status.ok { color:#12683a; font-weight:850; }
    .inline-status.error { color:#9b201c; font-weight:850; }
    .result { display:none; border:1px solid var(--line); border-radius:18px; padding:16px; background:#fbfff4; margin-top:12px; }
    .result.show { display:block; }
    .success-panel { border:1px solid rgba(127,138,119,.24); background:rgba(255,255,250,.96); box-shadow:0 18px 60px rgba(43,59,37,.12); padding:26px; }
    .success-layout { display:grid; grid-template-columns:minmax(0,1fr) 340px; gap:28px; align-items:start; }
    .success-hero { display:flex; gap:14px; align-items:flex-start; margin-bottom:18px; }
    .success-check { flex:0 0 44px; width:44px; height:44px; border-radius:50%; display:grid; place-items:center; background:#e5f9df; color:#14703d; border:2px solid #9ddd9e; box-shadow:0 8px 18px rgba(31,111,67,.1); font-size:28px; line-height:1; }
    .success-title { font-size:clamp(24px,2.4vw,32px); font-weight:950; letter-spacing:-.045em; color:#13552f; line-height:1.04; }
    .success-subtitle { font-size:14px; color:#4b5650; margin-top:6px; }
    .credit-scoreboard { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:12px; margin:0 0 16px; }
    .credit-card { display:flex; gap:12px; align-items:center; border:1px solid #e2e7dd; border-radius:16px; padding:14px; background:#fffefb; box-shadow:0 8px 20px rgba(43,59,37,.07); min-height:62px; }
    .credit-icon { flex:0 0 44px; width:44px; height:44px; display:grid; place-items:center; border-radius:50%; background:#eaf7e8; color:#17713f; font-size:22px; }
    .credit-card strong { display:block; color:#087339; font-size:24px; line-height:1; letter-spacing:-.04em; }
    .credit-card span { display:block; margin-top:5px; color:#59625d; font-size:12px; font-weight:750; }
    .leader-note { display:flex; gap:12px; align-items:center; margin:0 0 16px; padding:12px 14px; border-radius:14px; background:linear-gradient(90deg,#e8f8e5,#f6fbf1); color:#4a554f; font-size:13px; font-weight:650; }
    .leader-note strong { color:#13552f; }
    .leader-note:before { content:"i"; flex:0 0 26px; width:26px; height:26px; border-radius:50%; display:grid; place-items:center; border:2px solid #238551; color:#238551; font-weight:950; font-family:ui-serif, Georgia, serif; }
    .leaderboard-preview { border:1px solid #e0e6dc; border-radius:18px; overflow:hidden; background:#fffefb; box-shadow:0 12px 34px rgba(43,59,37,.07); }
    .leaderboard-title { padding:12px 16px; display:flex; justify-content:space-between; align-items:center; gap:10px; color:#12332a; font-size:16px; font-weight:950; border-bottom:1px solid #e6eadf; }
    .leaderboard-title-main { display:flex; gap:8px; align-items:center; }
    .leaderboard-rank-badge { border-radius:10px; padding:6px 10px; background:#eaf7e8; color:#13552f; font-size:12px; font-weight:950; white-space:nowrap; }
    .leaderboard-head, .leaderboard-row { display:grid; grid-template-columns:48px minmax(180px,1fr) 120px 100px; gap:10px; align-items:center; }
    .leaderboard-head { padding:9px 16px; color:#5f6662; font-size:12px; font-weight:900; border-bottom:1px solid #eef1e8; }
    .leaderboard-row { padding:10px 16px; border-top:1px solid #eef1e8; font-size:13px; }
    .leaderboard-row.pending { margin:0 8px 8px; border:1px solid #d8ecce; border-radius:12px; background:#f1fbeb; color:#13552f; font-weight:900; }
    .leaderboard-row span { min-width:0; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    .leaderboard-pager { display:flex; align-items:center; justify-content:center; gap:12px; padding:10px 16px 12px; border-top:1px solid #eef1e8; color:#5f6662; font-size:12px; font-weight:800; }
    .leaderboard-pager button { border:0; border-radius:10px; background:#e9efe2; color:#14241d; font-size:12px; font-weight:900; padding:8px 12px; min-height:34px; cursor:pointer; }
    .leaderboard-pager button:disabled { opacity:.45; cursor:not-allowed; }
    .leader-person { font-weight:850; }
    .success-detail-card { border:1px solid #e0e6dc; border-radius:20px; padding:22px; background:#fffefb; box-shadow:0 12px 34px rgba(43,59,37,.07); position:sticky; top:16px; }
    .detail-section { padding:0 0 20px; margin-bottom:20px; border-bottom:1px dashed #dfe5da; }
    .detail-section:last-child { margin-bottom:0; padding-bottom:0; border-bottom:0; }
    .detail-heading { display:flex; gap:12px; align-items:center; color:#28332e; font-size:18px; font-weight:950; margin-bottom:12px; }
    .detail-icon { flex:0 0 26px; color:#178047; font-size:22px; text-align:center; }
    .detail-value { color:#4d5852; font-size:15px; }
    .detail-chip { display:inline-block; border-radius:999px; padding:9px 14px; background:#e6f5e4; color:#13552f; font-weight:900; }
    .copybox { display:flex; align-items:center; gap:8px; border:1px solid var(--line); border-radius:12px; background:#fbfbf8; padding:10px 12px; font-family:ui-monospace, SFMono-Regular, Menlo, monospace; font-size:13px; overflow:hidden; }
    .copybox span { flex:1; min-width:0; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    .copy-mini { flex:0 0 auto; border:1px solid #d9dfd4; background:white; color:#5f6662; box-shadow:none; padding:4px 7px; border-radius:8px; font-size:12px; }
    .receipt-card { margin-top:28px; border:1px solid #e0e6dc; border-radius:18px; padding:22px; background:#fffefb; box-shadow:0 12px 34px rgba(43,59,37,.07); }
    .receipt-head { display:flex; align-items:center; gap:14px; font-size:20px; font-weight:950; color:#24312b; margin-bottom:16px; }
    .receipt-head:before { content:"▧"; display:grid; place-items:center; width:42px; height:42px; border-radius:50%; background:#eaf7e8; color:#17713f; }
    .file-list { display:grid; gap:8px; }
    .file-pill { display:flex; gap:10px; align-items:center; border:1px solid var(--line); border-radius:12px; background:#fbfbf8; padding:10px 12px; color:#3d4440; font-size:13px; }
    .file-pill:before { content:"▱"; color:#5c6660; }
    .result-head { display:flex; align-items:center; justify-content:space-between; gap:12px; flex-wrap:wrap; }
    .badge { display:inline-block; border-radius:999px; padding:5px 10px; font-weight:900; font-size:13px; }
    .badge.pass { background:#dff1d9; color:#13552f; }
    .badge.fail { background:#f6d8d3; color:#8a2118; }
    .field { margin-top:10px; }
    .field-label { font-size:12px; color:var(--muted); font-weight:800; text-transform:uppercase; letter-spacing:.04em; }
    .pathbox { margin-top:4px; padding:9px 10px; border-radius:10px; background:white; border:1px solid var(--line); font-family:ui-monospace, SFMono-Regular, Menlo, monospace; font-size:12px; overflow:auto; }
    .verify-fail-box { margin-top:12px; border:1px solid #efb7ad; background:#fff1ed; color:#7f241b; border-radius:14px; padding:12px 14px; }
    .verify-fail-box ul { margin:8px 0 0 18px; padding:0; }
    .verify-fail-box li { margin:4px 0; }
    .verify-fail-box code { background:#fff9f6; border:1px solid #f2c9c0; border-radius:6px; padding:1px 4px; color:#5f1610; }
    .scrub-suggestion { margin-top:10px; display:flex; gap:8px; flex-wrap:wrap; align-items:center; }
    .scrub-suggestion code { flex:1; min-width:220px; overflow:auto; white-space:nowrap; }
    .scrub-suggestion button { padding:7px 10px; font-size:12px; box-shadow:none; }
    .scrub-suggestion button.redact-primary { background:#be2e35; color:white; box-shadow:0 10px 18px rgba(190,46,53,.18); }
    .metrics { display:flex; flex-wrap:wrap; gap:8px; margin-top:12px; }
    .metric { background:#edf3e8; border:1px solid #d5e4ce; border-radius:999px; padding:6px 10px; font-size:13px; }
    .selected-card { display:none; border:2px solid #7cb67d; background:#eef8e8; border-radius:18px; padding:14px; margin-top:14px; }
    .selected-card.show { display:block; }
    .selected-card-layout { display:flex; justify-content:space-between; gap:16px; align-items:flex-start; }
    .selected-card-main { min-width:0; flex:1; }
    .selected-card-action { flex:0 0 auto; }
    .search-panel { display:none; border:1px dashed #b8c9ad; border-radius:16px; padding:14px; margin-top:12px; background:#fffef7; }
    .search-panel.show { display:block; }
    .step-headline { display:flex; gap:14px; align-items:flex-start; margin-bottom:16px; }
    .step-bubble { flex:0 0 42px; width:42px; height:42px; border-radius:50%; display:grid; place-items:center; background:var(--accent); color:white; font-weight:950; font-size:18px; box-shadow:0 10px 22px rgba(23,113,63,.18); }
    .step-bubble.warn { background:#be2e35; box-shadow:0 10px 22px rgba(190,46,53,.18); }
    .step-headline h2 { margin:0; }
    .step-headline p { margin:6px 0 0; }
    .redact-section-title { margin:12px 0 4px; display:flex; gap:8px; align-items:baseline; flex-wrap:wrap; font-weight:900; color:#111b18; }
    .redact-section-title .inline-note { color:var(--muted); font-weight:500; font-size:14px; }
    .privacy-card { position:relative; display:grid; grid-template-columns:auto 46px minmax(0,1fr); gap:14px; align-items:center; min-height:68px; }
    .privacy-card strong { display:flex; gap:8px; align-items:center; flex-wrap:wrap; }
    .privacy-icon { width:44px; height:44px; border-radius:50%; display:grid; place-items:center; background:#eef3ed; color:#1f6f43; font-size:22px; }
    .privacy-card:has(input:checked) .privacy-icon { background:#e8f8e5; }
    .redact-info-strip { display:flex; gap:10px; align-items:center; margin-top:12px; padding:10px 12px; border-radius:10px; background:#f3f4f2; color:#3f4843; font-size:13px; }
    .redact-info-strip:before { content:"i"; display:grid; place-items:center; width:18px; height:18px; border-radius:50%; border:2px solid #3c6da8; color:#275c99; font-weight:950; font-family:ui-serif, Georgia, serif; }
    .scrub-helper { margin:6px 0 0; color:#59635e; font-size:13px; }
    .scrub-helper strong { color:#28332e; }
    .redact-action-row { margin-top:12px; }
    .redact-action-row button { background:#be2e35; box-shadow:0 10px 18px rgba(190,46,53,.18); }
    .redact-review-grid { display:grid; grid-template-columns:minmax(0,1fr) 360px; gap:18px; align-items:start; margin-top:14px; }
    .redact-review-grid .result, .redact-review-grid .search-panel { margin-top:0; }
    .result.redact-card { border-radius:14px; padding:16px; background:#fffef9; }
    .result.redact-card.fail-card { border-color:#efb7ad; background:#fff8f5; }
    .result.redact-card.pass-card { border-color:#b9d6b0; background:#fbfff7; }
    .result-title { display:flex; justify-content:space-between; align-items:center; gap:12px; margin-bottom:14px; }
    .result-title-main { display:flex; gap:10px; align-items:center; font-weight:950; }
    .status-dot { width:24px; height:24px; border-radius:50%; display:grid; place-items:center; font-weight:950; }
    .status-dot.pass { color:#12683a; border:2px solid #75bd7f; }
    .status-dot.fail { color:#9b201c; background:#f7d6d2; }
    .redacted-path-row { display:flex; gap:8px; align-items:center; }
    .redacted-path-row .pathbox { flex:1; min-width:0; margin-top:4px; }
    .copy-file-btn { padding:8px 10px; box-shadow:none; background:#f4f5f1; color:#24312b; }
    .removed-count { color:var(--muted); font-weight:900; }
    .removed-note { margin-top:6px; color:#667068; font-size:13px; }
    .search-panel.compact-search { border-style:solid; background:#fffef9; }
    .search-panel.compact-search label { margin-top:0; }
    .search-panel.compact-search .row { flex-wrap:nowrap; }
    .search-panel.compact-search input { min-width:0 !important; }
    .search-panel.compact-search button { box-shadow:none; }
    .search-panel.compact-search .result { padding:12px; border-radius:12px; }
    .progress { width:100%; height:12px; border-radius:999px; overflow:hidden; background:#e5eadc; margin-top:12px; display:none; }
    .progress > div { height:100%; width:0%; background:linear-gradient(90deg,#1f6f43,#89b65b); transition:width .2s ease; }
    .progress-time { display:none; margin-top:6px; color:var(--muted); font-size:12px; font-weight:650; }
    .danger { color:#7f241b; font-weight:800; background:#fff1ed; border:1px solid #f2c9c0; padding:10px 12px; border-radius:14px; }
    .ok { color:var(--accent); font-weight:800; }
    .hint { font-size:13px; color:var(--muted); margin-top:6px; }
    .grid { display:grid; grid-template-columns:1fr 1fr; gap:16px; }
    .submit-grid { display:grid; grid-template-columns:1.25fr 1fr 1fr; gap:16px; align-items:start; }
    .submit-leaderboard { margin-top:14px; }
    .public-credit-option { padding:12px 16px; border-top:1px solid #e6eadf; background:#fbfff7; color:#26332d; }
    .public-credit-option label { margin:0; display:flex; gap:10px; align-items:flex-start; font-weight:900; }
    .public-credit-option input { width:auto; margin-top:3px; }
    .public-credit-option .hint { margin-left:26px; }
    .topline { color:var(--muted); max-width:760px; font-size:18px; }
    .actions { justify-content:space-between; margin-top:18px; padding-top:16px; border-top:1px solid var(--line); }
    .compact-input-row { flex-wrap:nowrap; align-items:center; }
    .compact-input-row label { margin:0; white-space:nowrap; }
    .compact-input-row input { flex:1 1 360px; min-width:260px; }
    .privacy-options { display:grid; grid-template-columns:1fr 1fr; gap:12px; margin-top:12px; }
    .privacy-card { border:1px solid var(--line); border-radius:12px; padding:12px; background:#fffef7; cursor:pointer; }
    .privacy-card:has(input:checked) { border-color:#1f6f43; background:#eef8e8; box-shadow:0 8px 22px rgba(31,111,67,.12); }
    .privacy-card input { width:auto; margin-right:7px; }
    @media (max-width:1000px) { .hero-top, .hero-side, .bottom-nav { align-items:flex-start; flex-direction:column; } .hero-flow { grid-template-columns:1fr; } .support-card { align-items:flex-start; flex-wrap:wrap; } .stats { width:100%; min-width:0; } .privacy-note { text-align:left; max-width:none; white-space:normal; } .privacy-links { justify-content:flex-start; } .hero-progress { justify-content:flex-start; } .pick-grid { grid-template-columns:1fr; } .session-summary,.fit-summary { flex-wrap:wrap; } .session-table-head,.session-row { grid-template-columns:46px minmax(180px,1fr) 122px 92px 92px 22px; } .session-table-head > div:nth-child(6), .session-fit { display:none; } .success-layout { grid-template-columns:1fr; } .success-detail-card { position:static; } .redact-review-grid { grid-template-columns:1fr; } }
    @media (max-width:700px) { main { padding:14px 10px 34px; } .hero,.card,.bottom-nav { border-radius:20px; padding:22px; } .grid,.submit-grid { grid-template-columns:1fr; } .stats { grid-template-columns:repeat(3,minmax(0,1fr)); } .composition-row { grid-template-columns:38px minmax(84px,1fr) minmax(64px,auto); gap:10px; } .composition-track { grid-column:2 / 4; } .steps { grid-template-columns:1fr; gap:10px; } .step-pill:after { display:none; } .session-head { flex-direction:column; } .session-summary { justify-content:flex-start; } .session-tools { align-items:stretch; } .session-search { max-width:none; width:100%; } .session-table-head,.session-row { grid-template-columns:44px minmax(0,1fr) 22px; } .session-table-head > div:nth-child(n+3):nth-child(-n+6), .session-date,.session-turns,.session-cmp,.session-fit { display:none; } .session-icon { width:36px; height:36px; font-size:16px; } .privacy-options { grid-template-columns:1fr; } .privacy-card { grid-template-columns:auto minmax(0,1fr); } .privacy-icon { display:none; } .selected-card-layout { flex-direction:column; } .compact-input-row { flex-wrap:wrap; } .compact-input-row input { flex-basis:100%; } .credit-scoreboard { grid-template-columns:1fr; } .success-hero { flex-direction:column; gap:16px; } .leaderboard-head,.leaderboard-row { grid-template-columns:42px minmax(0,1fr) 72px; } .leaderboard-head span:nth-child(4), .leaderboard-row > span:nth-child(4) { display:none; } .search-panel.compact-search .row { flex-wrap:wrap; } .actions { justify-content:flex-start; } }
  </style>
</head>
<body>
<main>
  <section class="hero">
    <div class="hero-top">
      <div>
        <h1>ContextEcho Donation Wizard</h1>
        <div class="topline">Donate a coding-agent session in a few local-first steps.</div>
      </div>
      <div class="hero-side">
        <div class="privacy-note"><strong>Donor privacy:</strong> ContextEcho analyzes assistant behavior, not donor personality.<br>Default: <strong>full redacted</strong>. Stronger privacy: <strong>user-minimized</strong>.
          <div class="privacy-links">
            <a class="ranking" href="https://github.com/Accenture/ContextEcho/blob/main/CONTRIBUTORS.md" target="_blank" rel="noopener noreferrer">Ranking</a>
            <a class="guide" href="https://accenture.github.io/ContextEcho/donate/#guideTitle" target="_blank" rel="noopener noreferrer">Guide</a>
          </div>
        </div>
        <div class="hero-progress">
          <div class="progress-label"><strong id="stepLabel">Step 1 of 3</strong><span id="stepPercentText">33% complete</span></div>
          <div id="progressRing" class="ring" style="--pct:33"><span id="progressRingText">33%</span></div>
        </div>
      </div>
    </div>
    <div class="hero-flow">
      <div class="support-card">
        <div class="bow-mascot" aria-hidden="true"><div class="bow-star">★</div><div class="bow-head"></div><div class="bow-body"></div><div class="bow-hands"></div></div>
        <div class="support-main">
          <div class="support-title">Help more donors find ContextEcho</div>
          <div class="support-copy">Star the GitHub repo or like the dataset by clicking the cards.</div>
        </div>
        <div id="projectStats" class="stats" aria-live="polite">
          <a class="stat-card" href="https://huggingface.co/datasets/contextecho2026/persona-drift-contextecho" target="_blank" rel="noopener noreferrer"><div class="stat-icon" data-icon="download"></div><div class="stat-value">...</div><div class="stat-label">Total Downloads</div></a>
          <a class="stat-card" href="https://github.com/Accenture/ContextEcho" target="_blank" rel="noopener noreferrer"><div class="stat-icon" data-icon="star"></div><div class="stat-value">...</div><div class="stat-label">GitHub Stars</div></a>
          <a class="stat-card" href="https://huggingface.co/datasets/contextecho2026/persona-drift-contextecho" target="_blank" rel="noopener noreferrer"><div class="stat-icon" data-icon="heart"></div><div class="stat-value">...</div><div class="stat-label">Dataset Likes</div></a>
        </div>
      </div>
      <div class="steps">
        <span id="pill1" class="step-pill active"><span class="step-num">1</span><span>Pick a Session</span></span>
        <span id="pill2" class="step-pill"><span class="step-num">2</span><span>Redact</span></span>
        <span id="pill3" class="step-pill"><span class="step-num">3</span><span>Submit</span></span>
      </div>
    </div>
  </section>

  <section id="step1" class="step active">
    <div class="pick-grid">
      <div class="card pick-intro">
        <div class="intro-head">
          <div class="folder-icon"></div>
          <div>
            <h2>1. Pick a Session</h2>
            <p class="muted">Ready sessions can be donated now; keep chatting sessions need more turns or a context compaction.</p>
          </div>
        </div>
        <div id="datasetComposition" class="composition-panel" aria-label="Public dataset composition"></div>
        <button id="discoverBtn" class="discover-main">Discover Sessions</button>
        <div id="discoverStatus" class="muted" style="margin-top:16px; text-align:center">Click discover to scan Claude/Codex sessions on this machine.</div>
        <div id="discoverProgress" class="progress"><div></div></div>
      </div>
      <div class="card sessions-card">
        <div class="session-head">
          <div>
            <h2>Recently discovered sessions</h2>
            <div class="session-subtitle">Detected from your local conversations.</div>
          </div>
          <div class="session-summary">
            <span id="sessionCount" class="count-badge"><strong>0</strong><span>found</span></span>
            <div id="fitSummary" class="fit-summary" aria-live="polite"></div>
          </div>
        </div>
        <div class="session-tools">
          <input id="sessionSearch" class="session-search" type="search" placeholder="Search sessions, agent, model, project">
          <div class="table-note"><sup>1</sup> Ctx cmp = context compactions detected in local logs.</div>
        </div>
        <div id="sessionList" class="session-list">
          <div class="session-table-head"><div>#</div><div><button type="button" class="sort-header" data-sort-key="session">Session<span class="sort-arrow"></span></button></div><div><button type="button" class="sort-header" data-sort-key="last_active">Last active<span class="sort-arrow"></span></button></div><div><button type="button" class="sort-header" data-sort-key="turns">User turns<span class="sort-arrow"></span></button></div><div><button type="button" class="sort-header" data-sort-key="compactions">Ctx cmp<span class="header-footnote">1</span><span class="sort-arrow"></span></button></div><div><button type="button" class="sort-header" data-sort-key="fit">Fit<span class="sort-arrow"></span></button></div><div></div></div>
          <div class="empty-sessions">Click Discover Sessions to find local Claude/Codex sessions.</div>
        </div>
        <div class="fit-legend">
          <div class="legend-items">
            <span class="legend-item"><span class="pill best"><span class="fit-star">&#9733;</span>Best</span> 100+ turns and 2+ ctx cmp</span>
            <span class="legend-item"><span class="pill good"><span class="fit-star">&#9733;</span>Good</span> 50+ turns</span>
            <span class="legend-item"><span class="pill improve"><span class="fit-arrow">&uarr;</span>Improve</span> keep chatting before donating</span>
          </div>
        </div>
        <div id="pager" class="session-pager">
          <button id="prevPage" class="secondary">&lsaquo; Previous</button>
          <span id="pageInfo" class="muted"></span>
          <button id="nextPage" class="secondary">Next &rsaquo;</button>
        </div>
      </div>
    </div>
    <div id="sessionMenu" class="session-menu" role="menu" aria-label="Session actions"></div>
    <div class="bottom-nav">
      <div class="tip"><strong>Tip:</strong> Context compactions are detected from agent logs; Codex may record them internally without a visible progress bar.</div>
      <button id="pickNext" class="next-button" disabled>Next: Redact  -&gt;</button>
    </div>
  </section>

  <section id="step2" class="card step">
    <div class="step-headline">
      <div id="redactStepBubble" class="step-bubble">2</div>
      <div>
        <h2>Redact + Verify</h2>
        <p class="muted">Review your session details, choose a redaction mode, and verify the output.</p>
      </div>
    </div>
    <div id="selectedCard" class="selected-card"></div>
    <div class="danger">Only donate personal, internal tooling, or open-source sessions. <span style="font-weight:500">Do not donate client-confidential/NDA data.</span></div>
    <div class="redact-section-title">Choose your redaction level <span class="inline-note">(ContextEcho analyzes assistant behavior, not donor personality. Choose how much of your own wording to keep.)</span></div>
    <div class="privacy-options">
      <label class="privacy-card"><input type="radio" name="privacyTier" value="full_redacted" checked><div class="privacy-icon">✣</div><div><strong>Full redacted <span class="pill best">Recommended</span></strong><div class="hint">Default. Keeps task flow after PII/secrets/custom terms are removed.<br>Highest scientific fidelity.</div></div></label>
      <label class="privacy-card"><input type="radio" name="privacyTier" value="user_minimized"><div class="privacy-icon">♢</div><div><strong>User-minimized</strong><div class="hint">Selectively masks sensitive donor text after redaction.<br>Coding task context remains; stronger privacy.</div></div></label>
    </div>
    <div class="redact-info-strip"><strong>Automatic redaction covers:</strong> paths, usernames, emails, names, phone numbers, IPs, URLs, API keys, tokens, and credential-like strings.</div>
    <label><input id="safeConfirm" type="checkbox" style="width:auto"> I confirm this session is safe to donate.</label>
    <div class="row redact-action-row">
      <button id="redactBtn" disabled>Redact and Verify</button>
    </div>
    <div id="redactProgress" class="progress"><div></div></div>
    <div class="redact-review-grid">
      <div id="redactResult" class="result"></div>
      <div id="searchPanel" class="search-panel compact-search">
        <label>Check whether a private word is still present <span class="muted">(optional)</span></label>
        <div class="row">
          <input id="searchTerms" placeholder="word or phrase to check" style="flex:1; min-width:260px" />
          <button id="searchBtn" class="secondary">Check File</button>
        </div>
        <div id="searchProgress" class="progress"><div></div></div>
        <div id="searchResult" class="result"></div>
      </div>
    </div>
    <div class="inline-status" id="redactStatus"></div>
    <label style="margin-top:14px"><input id="reviewConfirm" type="checkbox" style="width:auto" disabled> I reviewed the verify output and redacted file path; it is ready to submit.</label>
    <div class="row actions">
      <button id="redactPrev" class="secondary">Previous</button>
      <button id="redactNext" disabled>Next: Submit</button>
    </div>
  </section>

  <section id="step3" class="card step">
    <h2>3. Submit</h2>
    <p class="muted">Contributor info is required for credit, leaderboard accounting, duplicate handling, and release acknowledgments.</p>
    <p class="muted">The tool writes manifest + consent, confirms the verified redacted artifact, uploads it, and saves a local receipt.</p>
    <div class="submit-grid">
      <div><label>Name or GitHub/HF handle <span class="muted">(required)</span></label><input id="contributorName" placeholder="your name or handle" required /></div>
      <div><label>Email <span class="muted">(required)</span></label><input id="contributorEmail" type="email" list="emailSuggestions" placeholder="you@example.com" required /><datalist id="emailSuggestions"></datalist></div>
      <div><label>Institute <span class="muted">(required)</span></label><input id="contributorInstitute" placeholder="University / company / independent" required /></div>
    </div>
    <div id="supportRequestBox" class="result" style="display:none">
      <label>Problem type</label>
      <select id="supportReason">
        <option value="reset_for_resubmit">Reset so I can resubmit</option>
        <option value="remove_submission">Remove my submission</option>
        <option value="wrong_session">Wrong session</option>
        <option value="duplicate">Duplicate submission</option>
        <option value="other">Other</option>
      </select>
      <label>Message to maintainer</label>
      <textarea id="supportMessage" rows="4" placeholder="Describe what should happen and why."></textarea>
    </div>
    <div id="submitLeaderboardPreview" class="submit-leaderboard"></div>
    <div class="row actions">
      <button id="submitPrev" class="secondary">Previous</button>
      <button id="submitBtn" disabled>Submit Donation</button>
      <button id="metadataUpdateBtn" class="secondary" disabled>Send Info Update</button>
      <button id="supportRequestBtn" class="secondary" disabled style="display:none">Send Report</button>
      <button id="metadataBackBtn" class="secondary" style="display:none">Back to Sessions</button>
    </div>
    <div id="submitProgress" class="progress"><div></div></div>
    <div id="submitResult" class="result"></div>
    <div class="inline-status" id="submitStatus"></div>
  </section>
</main>
<script>
let sessions = [];
let selected = null;
let redacted = null;
let metadataUpdateSubmissionId = '';
let metadataUpdateSession = null;
let metadataUpdateComplete = false;
let supportRequestSubmissionId = '';
let supportRequestSession = null;
let supportRequestComplete = false;
let appliedScrubTerms = [];
let redactionCache = new Map();
let submitted = false;
let activeOperation = false;
let page = 0;
let sessionSort = {key:'', dir:'desc'};
let sessionSearchQuery = '';
const pageSize = 4;
const $ = id => document.getElementById(id);
const donatedPaths = new Set(JSON.parse(localStorage.getItem('contextechoDonatedPaths') || '[]'));
let donatedRecords = JSON.parse(localStorage.getItem('contextechoDonatedRecordsV1') || '{}');
let publicStats = {};
let leaderboardPreviewPage = null;
const statIcons = {
  star: '<svg viewBox="0 0 24 24" aria-hidden="true"><path class="icon-fill" d="M12 2.4l2.95 5.98 6.6.96-4.78 4.66 1.13 6.57L12 17.47l-5.9 3.1 1.13-6.57-4.78-4.66 6.6-.96L12 2.4z"/></svg>',
  download: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 3v11"/><path d="M7.5 9.5L12 14l4.5-4.5"/><path d="M5 17.5V20h14v-2.5"/></svg>',
  heart: '<svg viewBox="0 0 24 24" aria-hidden="true"><path class="icon-fill" d="M12 21s-7.25-4.45-9.35-8.7C.93 8.82 3.05 5 6.9 5c2.05 0 3.47 1.08 4.1 2.02C11.63 6.08 13.05 5 15.1 5c3.85 0 5.97 3.82 4.25 7.3C19.25 16.55 12 21 12 21z"/></svg>',
  users: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M16 21v-2a4 4 0 0 0-4-4H7a4 4 0 0 0-4 4v2"/><circle cx="9.5" cy="7" r="4"/><path d="M22 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/></svg>',
  handHeart: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M11 14h2.8a2 2 0 1 0 0-4H11"/><path d="M4 20V9a2 2 0 0 1 2-2h3.5"/><path d="M6 18h8l5.4-5.4a2 2 0 0 1 2.8 2.8L17 20H6"/><path d="M12 5.2C12.7 4.3 14.2 4 15.2 5c1 1 1 2.5 0 3.5L12 11.5 8.8 8.5a2.5 2.5 0 0 1 0-3.5c1-1 2.5-.7 3.2.2z"/></svg>',
  institute: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M3 21h18"/><path d="M5 10h14"/><path d="M6 10v9"/><path d="M10 10v9"/><path d="M14 10v9"/><path d="M18 10v9"/><path d="M12 3l8 5H4z"/></svg>',
  bot: '<svg viewBox="0 0 24 24" aria-hidden="true"><rect x="5" y="8" width="14" height="10" rx="3"/><path d="M12 8V4"/><path d="M8.5 13h.01"/><path d="M15.5 13h.01"/><path d="M9 18v2"/><path d="M15 18v2"/></svg>',
  cube: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 2.5 21 7.5v9L12 21.5 3 16.5v-9z"/><path d="M12 12.5v9"/><path d="m3 7.5 9 5 9-5"/><path d="m7.5 5 9 5"/></svg>',
  grid: '<svg viewBox="0 0 24 24" aria-hidden="true"><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/></svg>'
};
function iconSvg(name){ return statIcons[name] || ''; }
function saveDonatedPaths(){ localStorage.setItem('contextechoDonatedPaths', JSON.stringify([...donatedPaths])); }
function saveDonatedRecords(){ localStorage.setItem('contextechoDonatedRecordsV1', JSON.stringify(donatedRecords)); }
function sessionLocalKey(s){
  return [s?.path || '', s?.records || '', s?.turns || '', s?.compactions || '', s?.last_active || s?.modified || ''].join('|');
}
function sessionPathKey(s){ return String(s?.path || ''); }
function normalizeSubmissionId(value){
  return String(value || '').replace(/^pending\//, '').replace(/\/$/, '');
}
function localContributorRecord(s){
  const pathKey = sessionPathKey(s);
  const record = pathKey ? donatedRecords[pathKey] : {};
  return {
    creditName: s?.local_credit_name || record?.credit_name || '',
    email: s?.local_contributor_email || record?.contributor_email || '',
    institute: s?.local_institute || record?.institute || '',
    publicAnonymous: !!(s?.local_public_anonymous || record?.public_anonymous)
  };
}
function prefillContributorFields(session, overwrite=false){
  const info = localContributorRecord(session || {});
  if(info.creditName && (overwrite || !$('contributorName').value)) $('contributorName').value = info.creditName;
  if(info.email && (overwrite || !$('contributorEmail').value)) $('contributorEmail').value = info.email;
  if(info.institute && (overwrite || !$('contributorInstitute').value)) $('contributorInstitute').value = info.institute;
  if($('publicAnonymous') && (overwrite || info.publicAnonymous)) $('publicAnonymous').checked = info.publicAnonymous;
}
function setContributorFieldsLocked(locked){
  ['contributorName','contributorEmail','contributorInstitute'].forEach(id => { $(id).disabled = !!locked; });
  if($('publicAnonymous')) $('publicAnonymous').disabled = !!locked;
}
function resetMetadataUpdateUi(){
  metadataUpdateSubmissionId = '';
  metadataUpdateSession = null;
  metadataUpdateComplete = false;
  supportRequestSubmissionId = '';
  supportRequestSession = null;
  supportRequestComplete = false;
  setContributorFieldsLocked(false);
  $('metadataUpdateBtn').textContent = 'Send Info Update';
  $('metadataUpdateBtn').style.display = '';
  $('supportRequestBtn').textContent = 'Send Report';
  $('supportRequestBtn').style.display = 'none';
  $('supportRequestBox').style.display = 'none';
  $('supportReason').disabled = false;
  $('supportMessage').disabled = false;
  $('supportMessage').value = '';
  $('metadataBackBtn').style.display = 'none';
}
function localDonationInfo(s){
  const relayRecordId = normalizeSubmissionId(s?.relay_submission_id || '');
  const supportId = relayRecordId.startsWith('submission-') ? relayRecordId : '';
  const previousTurns = Number(s?.donated_turns || 0);
  const newTurns = Math.max(0, Number(s?.new_turns || 0));
  const updateReady = !!s?.update_ready;
  const exactDonated = !!s?.donated;
  const donatedBefore = exactDonated || !!s?.donated_before;
  return {exactDonated, donatedBefore, previousTurns, newTurns, updateReady, supportId, localRecordId:''};
}
function beginMetadataUpdate(session, submissionId){
  metadataUpdateComplete = false;
  supportRequestSubmissionId = '';
  supportRequestSession = null;
  supportRequestComplete = false;
  setContributorFieldsLocked(false);
  $('metadataUpdateBtn').textContent = 'Send Info Update';
  $('metadataUpdateBtn').style.display = '';
  $('supportRequestBtn').style.display = 'none';
  $('supportRequestBox').style.display = 'none';
  $('metadataBackBtn').style.display = 'none';
  metadataUpdateSubmissionId = submissionId;
  metadataUpdateSession = session || null;
  selected = session || selected;
  goStep(3);
  prefillContributorFields(session, true);
  renderSubmitLeaderboardPreview();
  status('submitStatus', `Editing contributor info for ${submissionId}. Change the fields above, then click Send Info Update.`);
  refreshButtons();
  $('contributorName').focus();
}
function beginSupportRequest(session, submissionId){
  resetMetadataUpdateUi();
  supportRequestSubmissionId = submissionId;
  supportRequestSession = session || null;
  selected = session || selected;
  goStep(3);
  prefillContributorFields(session, true);
  $('supportRequestBox').style.display = 'block';
  $('metadataUpdateBtn').style.display = 'none';
  $('supportRequestBtn').style.display = '';
  $('metadataBackBtn').style.display = '';
  renderSubmitLeaderboardPreview();
  status('submitStatus', `Reporting a problem for ${submissionId}. Choose a reason, add a note, then click Send Report.`);
  refreshButtons();
  $('supportMessage').focus();
}
async function sendMetadataUpdate(){
  if(!metadataUpdateSubmissionId) return;
  const name = ($('contributorName')?.value || '').trim();
  const email = ($('contributorEmail')?.value || '').trim();
  const institute = ($('contributorInstitute')?.value || '').trim();
  if(!name && !email && !institute){
    status('submitStatus', 'Edit at least one contributor field before sending an info update.');
    return;
  }
  const btn = $('metadataUpdateBtn');
  const originalText = btn.textContent;
  btn.disabled = true;
  btn.textContent = 'Sending...';
  status('submitStatus', 'Sending info update request to the maintainer relay...', '');
  try {
    const result = await post('/api/metadata_update', {
      submission_id: metadataUpdateSubmissionId,
      credit_name: name,
      contributor_email: email,
      contributor_institute: institute,
      public_anonymous: !!$('publicAnonymous')?.checked,
      source_session_id: metadataUpdateSession?.source_session_id || '',
      conversation_fingerprint: metadataUpdateSession?.conversation_fingerprint || ''
    });
    const requestText = result.request_id ? ` Request ID: ${result.request_id}.` : '';
    status('submitStatus', `Update sent successfully.${requestText} The maintainer will review it before changing the public record.`, 'ok');
    metadataUpdateSubmissionId = '';
    metadataUpdateSession = null;
    metadataUpdateComplete = true;
    setContributorFieldsLocked(true);
    btn.textContent = 'Update Sent';
    $('metadataBackBtn').style.display = '';
  } catch(e) {
    btn.textContent = originalText;
    status('submitStatus', 'ERROR: '+friendlyRequestError(e, 'info update request'), 'error');
  } finally {
    refreshButtons();
    $('submitStatus').scrollIntoView({behavior:'smooth', block:'nearest'});
  }
}
async function sendSupportRequest(){
  if(!supportRequestSubmissionId) return;
  const btn = $('supportRequestBtn');
  const originalText = btn.textContent;
  btn.disabled = true;
  btn.textContent = 'Sending...';
  status('submitStatus', 'Sending support request to the maintainer relay...', '');
  try {
    const result = await post('/api/support_request', {
      submission_id: supportRequestSubmissionId,
      reason: $('supportReason').value,
      message: $('supportMessage').value,
      source_session_id: supportRequestSession?.source_session_id || '',
      conversation_fingerprint: supportRequestSession?.conversation_fingerprint || ''
    });
    const requestText = result.support_id ? ` Request ID: ${result.support_id}.` : '';
    status('submitStatus', `Report sent successfully.${requestText} A maintainer will review it before making changes.`, 'ok');
    supportRequestSubmissionId = '';
    supportRequestSession = null;
    supportRequestComplete = true;
    setContributorFieldsLocked(true);
    $('supportReason').disabled = true;
    $('supportMessage').disabled = true;
    btn.textContent = 'Report Sent';
  } catch(e) {
    btn.textContent = originalText;
    status('submitStatus', 'ERROR: '+friendlyRequestError(e, 'support request'), 'error');
  } finally {
    refreshButtons();
    $('submitStatus').scrollIntoView({behavior:'smooth', block:'nearest'});
  }
}
function redactionCacheKey(){
  if(!selected) return '';
  return [
    sessionLocalKey(selected),
    privacyTier()
  ].join('\u001e');
}
function restoreCachedRedaction(){
  const cached = redactionCache.get(redactionCacheKey());
  if(!cached) return false;
  redacted = cached.data;
  appliedScrubTerms = [...cached.appliedScrubTerms];
  submitted = false;
  $('reviewConfirm').checked = false;
  renderRedactResult(redacted);
  status('redactStatus', 'Restored the verified result for this privacy mode. Review it, then check the review box to continue.');
  refreshButtons();
  return true;
}
function allSessionsDonated(){
  return sessions.length > 0 && sessions.every(s => {
    const info = localDonationInfo(s);
    return info.exactDonated || (info.donatedBefore && !info.updateReady);
  });
}
function relayStatusChecked(){
  return sessions.length > 0 && sessions.every(s => !!s.relay_checked);
}
function allSessionsDonatedMessage(){
  return `Thank you for donating all your scanned session data. ${sessions.length} session${sessions.length === 1 ? '' : 's'} on this machine are marked donated.`;
}
function noSessionsMessage(){
  return 'Thanks for considering a ContextEcho donation. We did not find any Claude Code or Codex sessions on this machine yet. Feel free to keep using your coding agent and come back later; we will continue collecting donations.';
}
function privacyTier(){ return document.querySelector('input[name="privacyTier"]:checked')?.value || 'full_redacted'; }
function parseScrubTerms(value){
  return [...new Set(String(value || '').split(',').map(x => x.trim()).filter(Boolean))];
}
function newScrubTerms(){
  return [];
}
function hasDetectSecretsFailure(data){
  const blocking = ((data || {}).verify_report || {}).blocking || {};
  return Array.isArray(blocking.detect_secrets) && blocking.detect_secrets.length > 0;
}
function goStep(n){
  if(n !== 3) resetMetadataUpdateUi();
  const pct = Math.round((n / 3) * 100);
  for(let i=1;i<=3;i++){
    $('step'+i).classList.toggle('active', i===n);
    $('pill'+i).classList.toggle('active', i===n);
    $('pill'+i).classList.toggle('done', i<n);
  }
  $('stepLabel').textContent = `Step ${n} of 3`;
  $('stepPercentText').textContent = `${pct}% complete`;
  $('progressRing').style.setProperty('--pct', pct);
  $('progressRingText').textContent = `${pct}%`;
  if(n === 3) {
    prefillContributorFields(selected, false);
    renderSubmitLeaderboardPreview();
  }
}
function refreshButtons(){
  if(activeOperation) return;
  const selectedInfo = selected ? localDonationInfo(selected) : null;
  const selectedDonated = !!(selectedInfo && (selectedInfo.exactDonated || (selectedInfo.donatedBefore && !selectedInfo.updateReady)));
  const selectedImprove = selected ? fit(selected) === 'improve' : false;
  const canSubmitArtifact = !!(redacted && redacted.verify_passed) && !submitted && !selectedDonated && !metadataUpdateSubmissionId && !metadataUpdateComplete && !supportRequestSubmissionId && !supportRequestComplete;
  const contributorComplete = contributorFieldsComplete();
  $('pickNext').disabled = !selected || selectedDonated || selectedImprove;
  $('redactBtn').disabled = !(selected && $('safeConfirm').checked);
  $('reviewConfirm').disabled = !(redacted && redacted.verify_passed);
  $('redactNext').disabled = !(redacted && redacted.verify_passed && $('reviewConfirm').checked);
  $('submitBtn').disabled = !canSubmitArtifact || !contributorComplete;
  $('metadataUpdateBtn').disabled = !metadataUpdateSubmissionId;
  $('supportRequestBtn').disabled = !supportRequestSubmissionId;
  $('submitBtn').title = '';
}
function setUiProcessing(on){
  activeOperation = !!on;
  document.body.classList.toggle('is-processing', activeOperation);
  document.querySelectorAll('button').forEach(btn => {
    if(activeOperation){
      btn.dataset.wasDisabled = btn.disabled ? '1' : '0';
      btn.disabled = true;
    } else if(btn.dataset.wasDisabled === '0'){
      btn.disabled = false;
      delete btn.dataset.wasDisabled;
    } else {
      delete btn.dataset.wasDisabled;
    }
  });
  if(!activeOperation) refreshButtons();
}
function fit(s){ const t=+s.turns||0,c=+s.compactions||0; return t>=100&&c>=2?'best':(t>=50?'good':'improve'); }
function sessionReady(s){ return fit(s) !== 'improve'; }
function fitCounts(){
  return sessions.reduce((acc, s) => {
    acc[fit(s)] = (acc[fit(s)] || 0) + 1;
    return acc;
  }, {best:0, good:0, improve:0});
}
function agentFamilyCounts(){
  return sessions.reduce((acc, s) => {
    const agent = String(s.agent || '').toLowerCase();
    if(agent.includes('claude')) acc.claude += 1;
    else if(agent.includes('codex')) acc.codex += 1;
    else acc.other += 1;
    return acc;
  }, {claude:0, codex:0, other:0});
}
function compactNumber(n){ n=+n||0; return n>=1000 ? (n/1000).toFixed(1)+'k' : String(n); }
function compactionNote(s){
  const agent = String(s.agent || '').toLowerCase();
  if(agent.includes('codex')) return 'Internal Codex context compaction events from the local JSONL log.';
  if(agent.includes('claude')) return 'Claude Code context summary/compaction events detected in the local log.';
  return 'Context compaction events detected in the local agent log.';
}
function status(id, text, kind=''){
  const el = $(id);
  el.textContent = text;
  el.classList.toggle('ok', kind === 'ok');
  el.classList.toggle('error', kind === 'error');
}
function contributorFieldsComplete(){
  return ['contributorName','contributorEmail','contributorInstitute'].every(id => ($(id).value || '').trim());
}
const commonEmailDomains = [
  'gmail.com', 'outlook.com', 'hotmail.com', 'yahoo.com', 'icloud.com',
  'proton.me', 'protonmail.com', 'aol.com', 'live.com', 'msn.com',
  'me.com', 'mac.com', 'qq.com', '163.com'
];
function emailParts(){
  const input = $('contributorEmail');
  const value = (input.value || '').trim();
  const at = value.indexOf('@');
  if(at <= 0) return {local:'', domain:'', value};
  return {local:value.slice(0, at), domain:value.slice(at + 1).toLowerCase(), value};
}
function updateEmailSuggestions(){
  const list = $('emailSuggestions');
  const {local, domain} = emailParts();
  if(!local){
    list.innerHTML = '';
    return;
  }
  const matches = commonEmailDomains
    .filter(d => !domain || d.startsWith(domain))
    .slice(0, 8);
  list.innerHTML = matches.map(d => `<option value="${escapeHtml(local)}@${escapeHtml(d)}"></option>`).join('');
}
function completeEmailDomain(){
  const input = $('contributorEmail');
  const {local, domain} = emailParts();
  if(!local || !domain || domain.includes('.')) return;
  const matches = commonEmailDomains.filter(d => d.startsWith(domain));
  if(matches.length === 1){
    input.value = `${local}@${matches[0]}`;
    status('submitStatus', `Email domain completed with @${matches[0]}. Edit it if needed.`);
    updateEmailSuggestions();
    renderSubmitLeaderboardPreview();
    refreshButtons();
  }
}
function fmtStat(n){
  if(n === null || n === undefined || n === '') return '—';
  n = Number(n);
  if(!Number.isFinite(n)) return '—';
  return Math.trunc(n).toLocaleString();
}
function escapeHtml(s){
  return String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}
function renderProjectStats(){
  const cards = [
    ['download', 'Total Downloads', publicStats.dataset_total_downloads, 'https://huggingface.co/datasets/contextecho2026/persona-drift-contextecho'],
    ['star', 'GitHub Stars', publicStats.github_stars, 'https://github.com/Accenture/ContextEcho'],
    ['heart', 'Dataset Likes', publicStats.dataset_likes, 'https://huggingface.co/datasets/contextecho2026/persona-drift-contextecho'],
  ];
  $('projectStats').innerHTML = cards.map(([icon, label, value, href]) => `
    <a class="stat-card" href="${escapeHtml(href)}" target="_blank" rel="noopener noreferrer">
      <div class="stat-icon" data-icon="${escapeHtml(icon)}">${iconSvg(icon)}</div>
      <div class="stat-value">${escapeHtml(fmtStat(value))}</div>
      <div class="stat-label">${escapeHtml(label)}</div>
    </a>
  `).join('');
  renderDatasetComposition();
}
function compositionMetric(label, key, target, icon, note=''){
  const coverage = publicStats.coverage || {};
  const raw = Number(coverage[key] || 0);
  const value = Number.isFinite(raw) ? raw : 0;
  return {label, key, value, target, icon, note, pct: Math.max(0, Math.min(100, (value / target) * 100))};
}
function renderDatasetComposition(){
  const target = $('datasetComposition');
  if(!target) return;
  const metrics = [
    compositionMetric('Sessions', 'sessions', 12, 'users'),
    compositionMetric('Donors', 'contributors', 12, 'handHeart'),
    compositionMetric('Institutes', 'institutions', 8, 'institute'),
    compositionMetric('Total turns', 'turns', 50000, 'grid'),
  ];
  target.innerHTML = `
    <div class="composition-head">
      <div class="composition-title">Dataset Composition</div>
      <div class="composition-subtitle">Breakdown of key public coverage metrics.</div>
    </div>
    <div class="composition-list">
      ${metrics.map(m => `
        <div class="composition-row">
          <div class="composition-icon">${iconSvg(m.icon)}</div>
          <div class="composition-label">${escapeHtml(m.label)}${m.note ? `<small>${escapeHtml(m.note)}</small>` : ''}</div>
          <div class="composition-track" aria-hidden="true"><div class="composition-fill" style="width:${m.pct.toFixed(1)}%"></div></div>
          <div class="composition-value">${escapeHtml(fmtStat(m.value))}</div>
        </div>
      `).join('')}
    </div>
  `;
}
function verifyFailureSummary(data){
  const blocking = ((data || {}).verify_report || {}).blocking || {};
  const entries = Object.entries(blocking);
  if(!entries.length) return 'Verify failed. Re-run redaction; if it repeats, inspect the redacted file with Test search.';
  const labels = entries.map(([k,v]) => `${k} (${(v || []).length})`).join(', ');
	  if(blocking.detect_secrets){
	    return `Verify failed: residual ${labels}. Automatic cleanup could not safely remove every credential-shaped finding. Reveal the redacted file, remove the flagged line or token, then run Redact and Verify again.`;
	  }
	  if(blocking.malformed_jsonl){
	    return `Verify failed: ${labels}. The redacted output has a formatting issue, not a private word. Click Redact and Verify again to regenerate or normalize the redacted file.`;
	  }
	  return `Verify failed: residual ${labels}. Click Redact and Verify Again for the shown private word(s), or use Check File after redaction.`;
	}
function suggestedScrubTerms(data){
  const blocking = ((data || {}).verify_report || {}).blocking || {};
  const terms = [];
  Object.entries(blocking).forEach(([category, values]) => {
	    if(category === 'detect_secrets' || category === 'malformed_jsonl') return;
    (values || []).forEach(v => {
      const term = String(v || '').trim();
      if(term) terms.push(term);
    });
  });
  return [...new Set(terms)];
}
async function loadProjectStats(){
  renderProjectStats();
  try {
    const r = await fetch('/api/project_stats');
    if(!r.ok) return;
    publicStats = await r.json();
    renderProjectStats();
    renderSubmitLeaderboardPreview();
  } catch(e) {
    renderProjectStats();
  }
}
function renderRedactResult(data){
  const stats = data.stats || {};
  const autoStats = {};
  const privateStats = {};
  Object.entries(stats).forEach(([k,v]) => {
    const value = Number(v || 0);
    if(!value) return;
    if(k.startsWith('private_word:')){
      const term = k.slice('private_word:'.length);
      const noisy = term.length > 24 || /PRIVATE KEY|BEGIN |[A-Z0-9]{20,}|[@=]/.test(term);
      const label = noisy ? 'credential/private patterns' : term;
      privateStats[label] = (privateStats[label] || 0) + value;
      return;
    }
    if(k === 'scrub_term' && Object.keys(stats).some(name => name.startsWith('private_word:'))) return;
    autoStats[k] = (autoStats[k] || 0) + value;
  });
  const autoEntries = Object.entries(autoStats).sort((a,b)=>b[1]-a[1]);
  const privateEntries = Object.entries(privateStats).sort((a,b)=>b[1]-a[1]);
  const entries = [...autoEntries, ...privateEntries];
  const autoMetrics = autoEntries.length
    ? autoEntries.map(([k,v]) => `<span class="metric">${escapeHtml(k)}: <strong>${v}</strong></span>`).join('')
    : '<span class="metric">No automatic detector matches</span>';
  const privateMetrics = privateEntries.length
    ? privateEntries.map(([k,v]) => `<span class="metric">${escapeHtml(k)}: <strong>${v}</strong></span>`).join('')
    : '<span class="metric">No detector matches</span>';
  const verify = data.verify_report || {};
  const blocking = verify.blocking || {};
  const blockingEntries = Object.entries(blocking);
  const suggestedTerms = suggestedScrubTerms(data);
  const suggestedText = suggestedTerms.join(', ');
  const failDetails = blockingEntries.length
    ? blockingEntries.map(([k,v]) => {
        const samples = (v || []).slice(0, 3).map(x => `<code>${escapeHtml(String(x))}</code>`).join(', ');
        return `<li><strong>${escapeHtml(k)}</strong>${samples ? ': ' + samples : ''}</li>`;
      }).join('')
    : '<li>Verifier returned a non-clean result. Re-run redaction; if it repeats, use Test search to inspect the redacted file.</li>';
  const failureBox = data.verify_passed ? '' : `
    <div class="verify-fail-box">
      <strong>Why it failed</strong>
      <ul>${failDetails}</ul>
      ${suggestedTerms.length ? `
        <div class="scrub-suggestion">
          <code>${escapeHtml(suggestedText)}</code>
          <button class="secondary" type="button" id="useSuggestedScrub">Redact and Verify Again</button>
        </div>
      ` : ''}
      <div class="hint"><strong>Next:</strong> ${blocking.detect_secrets ? 'the tool already tried automatic credential cleanup. Reveal the redacted file and remove the credential-shaped line or token manually, then run Redact and Verify again.' : 'click Redact and Verify Again for the shown private word(s). For paths, use the username/project part, not the full path.'}</div>
    </div>
  `;
  const removedCount = entries.reduce((acc, item) => acc + Number(item[1] || 0), 0);
  $('redactResult').innerHTML = `
    <div class="result-title">
      <div class="result-title-main">
        <span class="status-dot ${data.verify_passed ? 'pass' : 'fail'}">${data.verify_passed ? '✓' : '×'}</span>
        <span>${data.verify_passed ? 'Verified clean' : 'Verify failed'}</span>
      </div>
      <div class="muted">${data.privacy_tier === 'user_minimized' ? 'Redaction + user minimization complete' : 'Redaction complete'} ${data.verify_passed ? '✓' : ''}</div>
    </div>
    ${failureBox}
    <div class="field"><div class="field-label">Redacted file</div><div class="redacted-path-row"><div class="pathbox">${escapeHtml(data.redacted_file)}</div><button class="copy-file-btn" type="button" id="copyRedactedPath">Copy</button></div></div>
    <div class="row" style="margin-top:8px"><button class="secondary" id="revealRedactedFile">Reveal File</button></div>
    <div class="field"><div class="field-label">Already redacted in this output ${removedCount ? `<span class="removed-count">(${removedCount})</span>` : ''}</div><div class="field-label" style="margin-top:10px">Automatic redaction</div><div class="metrics">${autoMetrics}</div>${privateEntries.length ? `<div class="field-label" style="margin-top:10px">Private words you asked to redact</div><div class="metrics">${privateMetrics}</div>` : ''}<div class="removed-note">These chips are a summary of what the tool already redacted. They are not terms to type.</div></div>
  `;
  $('redactResult').className = `result show redact-card ${data.verify_passed ? 'pass-card' : 'fail-card'}`;
  $('searchPanel').classList.add('show');
  $('searchResult').classList.remove('show');
  $('revealRedactedFile').onclick = () => post('/api/open_path', {path:data.redacted_file, reveal:true}).catch(e => status('redactStatus','ERROR: '+e.message));
  $('copyRedactedPath').onclick = () => navigator.clipboard?.writeText(data.redacted_file).catch(()=>{});
  const suggestedBtn = $('useSuggestedScrub');
  if(suggestedBtn){
    suggestedBtn.onclick = () => {
      const existing = $('scrub').value.split(',').map(x => x.trim()).filter(Boolean);
      const merged = [...new Set([...existing, ...suggestedTerms])];
      $('scrub').value = merged.join(', ');
      $('reviewConfirm').checked = false;
      status('redactStatus', 'Private words added. Click Redact and Verify again to remove them from the file.');
      refreshButtons();
    };
  }
}
function mergeRedactionStats(previousStats, nextStats){
  const merged = {...(previousStats || {})};
  Object.entries(nextStats || {}).forEach(([key, value]) => {
    merged[key] = Number(merged[key] || 0) + Number(value || 0);
  });
  return merged;
}
function renderSelectedCard(s, idx){
  $('selectedCard').innerHTML = `
    <div class="selected-card-layout">
      <div class="selected-card-main">
        <div class="result-head">
          <div><strong>Selected #${idx + 1}: ${escapeHtml(s.project || 'unknown project')}</strong></div>
          <span class="pill ${fit(s)}">${fit(s).charAt(0).toUpperCase() + fit(s).slice(1)}</span>
        </div>
        <div class="metrics">
          <span class="metric">Agent: <strong>${escapeHtml(s.agent || '?')}</strong></span>
          <span class="metric">Model: <strong>${escapeHtml(s.model || '?')}</strong></span>
          <span class="metric">User turns: <strong>${compactNumber(s.turns)}</strong></span>
          <span class="metric">Records: <strong>${compactNumber(s.records || s.turns)}</strong></span>
          <span class="metric">Compactions: <strong>${s.compactions || 0}</strong></span>
          <span class="metric">Last active: <strong>${escapeHtml(s.last_active || s.modified || '?')}</strong></span>
        </div>
        <div class="hint">${escapeHtml(compactionNote(s))}</div>
      </div>
      <div class="selected-card-action"><button class="secondary" id="revealSourceFile">Reveal Source File</button></div>
    </div>
  `;
  $('selectedCard').classList.add('show');
  $('revealSourceFile').onclick = () => post('/api/open_path', {path:s.path, reveal:true}).catch(e => status('redactStatus','ERROR: '+e.message));
}
function renderSearchResult(data){
  const hits = data.results || [];
  const anyHit = hits.some(x => x.count > 0);
  const matchedTerms = [...new Set(hits
    .filter(x => x.count > 0)
    .map(x => String(x.term || '').trim())
    .filter(Boolean))];
  const matchedText = matchedTerms.join(', ');
  const metrics = hits.length
    ? hits.map(x => `<span class="metric">${escapeHtml(x.term)}: <strong>${x.count}</strong></span>`).join('')
    : '<span class="metric">No terms entered</span>';
  $('searchResult').innerHTML = `
    <div class="result-head">
      <div><span class="badge ${anyHit ? 'fail' : 'pass'}">${anyHit ? 'Still present' : 'Not found'}</span></div>
      <div class="muted">${anyHit ? 'Run Redact and Verify again here for the matched word(s), then this check will refresh.' : 'The checked word(s) were not found in the redacted file.'}</div>
    </div>
    <div class="metrics">${metrics}</div>
    ${matchedTerms.length ? `
      <div class="scrub-suggestion">
        <button class="redact-primary" type="button" id="repairSearchTerms">Redact and Verify Again</button>
      </div>
    ` : ''}
  `;
  $('searchResult').classList.add('show');
  const repairBtn = $('repairSearchTerms');
  if(repairBtn){
    repairBtn.onclick = async () => {
      repairBtn.disabled = true;
      repairBtn.textContent = 'Redacting...';
      setBusy('searchProgress', true, 60);
      $('searchResult').innerHTML = `
        <div class="result-head">
          <div><span class="badge fail">Redacting</span></div>
        </div>
        <div class="metrics">${metrics}</div>
      `;
      try {
        await runRedactVerify(matchedTerms, {fromSearch:true});
        const refreshed = await post('/api/search_redacted', {redacted_file:redacted.redacted_file, terms:matchedText});
        renderSearchResult(refreshed);
        const remaining = (refreshed.results || []).reduce((acc, row) => acc + Number(row.count || 0), 0);
        status('redactStatus', remaining ? 'Redaction ran, but the checked word is still present. Inspect the redacted file or try a more exact term.' : 'Redaction complete. The checked word is now found 0 times.');
      } catch(e) {
        status('redactStatus','ERROR: '+e.message);
        setBusy('searchProgress', false);
      }
    };
  }
}
function receiptEmailHref(receipt, receiptPath){
  const email = receipt.contributor_email || '';
  const publicId = (receipt.submission || '').replace(/^pending\//, '').replace(/\/$/, '') || 'not available';
  const subject = `ContextEcho donation receipt ${publicId}`.trim();
  const body = [
    'ContextEcho donation receipt',
    '',
    `Submission ID: ${publicId}`,
    `Credit name: ${receipt.credit_name || 'anonymous'}`,
    `Public leaderboard: ${receipt.public_anonymous ? 'anonymous' : (receipt.credit_name || 'anonymous')}`,
    `Agent/model: ${(receipt.agent || '')} / ${(receipt.model || '')}`,
    `Privacy tier: ${receipt.privacy_tier || 'full_redacted'}`,
    `User turns: ${receipt.turns || ''}`,
    `Records: ${receipt.records || ''}`,
    `Context compactions: ${receipt.compactions || ''}`,
    `Receipt file: ${receiptPath || ''}`,
    '',
    'Status: pending maintainer review. Credit is awarded after acceptance.'
  ].join('\n');
  return `mailto:${encodeURIComponent(email)}?subject=${encodeURIComponent(subject)}&body=${encodeURIComponent(body)}`;
}
function publicCreditLabel(creditName, publicAnonymous, publicId='pending'){
  const cleanName = String(creditName || '').trim() || 'anonymous';
  if(!publicAnonymous) return cleanName;
  const suffix = String(publicId || '').replace(/^submission-/, '') || 'pending';
  return `Anonymous donor ${suffix}`;
}
function pendingLeaderboardModel(publicCreditName, publicAnonymous, turns, compactions, localPending = {}, pendingDisplayName = ''){
  const highValue = Number(turns || 0) >= 50 || Number(compactions || 0) >= 1;
  const pendingPointsLow = highValue ? 3 : 2;
  const pendingPointsHigh = highValue ? 5 : 4;
  const localPendingSessions = Number(localPending.sessions || 1);
  const localPendingLow = Number(localPending.points_low || pendingPointsLow);
  const localPendingHigh = Number(localPending.points_high || pendingPointsHigh);
  const localPendingTurns = Number(localPending.turns || turns || 0);
  const acceptedLeaders = publicStats.leaderboard || [];
  const sameName = row => !publicAnonymous && String(row.contributor || '').toLowerCase() === publicCreditName.toLowerCase();
  const mergedWithExisting = acceptedLeaders.some(sameName);
  const simulatedLeaders = acceptedLeaders.map(row => {
    const basePoints = Number(row.points_num || 0);
    const baseSessions = Number(row.sessions_num || 0);
    const baseTurns = Number(row.turns_num || 0);
    if(!sameName(row)) return {
      name: row.contributor || '',
      points: basePoints,
      pointsLow: basePoints,
      pointsHigh: basePoints,
      sessions: baseSessions,
      turns: baseTurns,
      pending: false,
      pendingExisting: false,
    };
    return {
      name: publicCreditName,
      displayName: pendingDisplayName || publicCreditName,
      points: basePoints + localPendingLow,
      pointsLow: basePoints + localPendingLow,
      pointsHigh: basePoints + localPendingHigh,
      sessions: baseSessions + localPendingSessions,
      turns: baseTurns + localPendingTurns,
      pending: true,
      pendingExisting: true,
    };
  });
  if(!mergedWithExisting) simulatedLeaders.push({
    name: publicCreditName,
    displayName: pendingDisplayName || publicCreditName,
    points: localPendingLow,
    pointsLow: localPendingLow,
    pointsHigh: localPendingHigh,
    sessions: localPendingSessions,
    turns: localPendingTurns,
    pending: true,
    pendingExisting: false,
  });
  simulatedLeaders.sort((a,b) =>
    (Number(b.pointsLow || b.points || 0) - Number(a.pointsLow || a.points || 0)) ||
    ((a.pending && !a.pendingExisting ? 1 : 0) - (b.pending && !b.pendingExisting ? 1 : 0)) ||
    (b.sessions - a.sessions) ||
    (b.turns - a.turns) ||
    a.name.localeCompare(b.name)
  );
  return {
    highValue,
    localPendingSessions,
    localPendingLow,
    localPendingHigh,
    simulatedLeaders,
    estimatedRank: Math.max(1, simulatedLeaders.findIndex(row => row.pending) + 1),
  };
}
function leaderboardDefaultPage(model, pageSize){
  return Math.max(0, Math.floor((Math.max(1, model.estimatedRank) - 1) / pageSize));
}
function leaderboardPreviewHtml(model){
  const totalDonorsEstimate = model.simulatedLeaders.length;
  const rankLabel = `${model.estimatedRank}/${totalDonorsEstimate}`;
  const pageSize = 5;
  const totalPages = Math.max(1, Math.ceil(model.simulatedLeaders.length / pageSize));
  const defaultPage = leaderboardDefaultPage(model, pageSize);
  if(leaderboardPreviewPage === null || leaderboardPreviewPage >= totalPages) leaderboardPreviewPage = defaultPage;
  const currentPage = Math.max(0, Math.min(Number(leaderboardPreviewPage || 0), totalPages - 1));
  const windowStart = currentPage * pageSize;
  const displayRank = rank => ({1:'🥇', 2:'🥈', 3:'🥉'}[rank] || String(rank));
  const leaderboardRows = model.simulatedLeaders.slice(windowStart, windowStart + pageSize).map((row, offset) => {
    const rank = windowStart + offset + 1;
    const sessionText = row.pending
      ? `${row.sessions} pending`
      : `${row.sessions} session${row.sessions === 1 ? '' : 's'}`;
    const pointsText = row.pending
      ? `${row.pointsLow}–${row.pointsHigh} pts`
      : `${row.points} pts`;
    return `
    <div class="leaderboard-row ${row.pending ? 'pending' : ''}">
      <span>${escapeHtml(displayRank(rank))}</span>
      <span class="leader-person">${escapeHtml(row.displayName || row.name || 'anonymous')}</span>
      <span>${escapeHtml(sessionText)}</span>
      <span>${escapeHtml(pointsText)}</span>
    </div>
  `;
  }).join('');
  return `
    <div class="leaderboard-title"><span class="leaderboard-title-main">♙ <span>Leaderboard preview</span></span><span class="leaderboard-rank-badge">Estimated rank: ${escapeHtml(rankLabel)}</span></div>
    <div class="leaderboard-head"><span>#</span><span>Contributor</span><span>Sessions</span><span>Points</span></div>
    ${leaderboardRows || '<div class="leaderboard-row"><span>—</span><span class="leader-person">Accepted leaderboard loads after release</span><span>—</span><span>—</span></div>'}
    ${totalPages > 1 ? `
      <div class="leaderboard-pager">
        <button type="button" id="leaderPrev" ${currentPage <= 0 ? 'disabled' : ''}>Previous</button>
        <span>Page ${currentPage + 1} of ${totalPages}</span>
        <button type="button" id="leaderNext" ${currentPage >= totalPages - 1 ? 'disabled' : ''}>Next</button>
      </div>
    ` : ''}
  `;
}
function bindLeaderboardPager(model){
  const pageSize = 5;
  const totalPages = Math.max(1, Math.ceil((model.simulatedLeaders || []).length / pageSize));
  const prev = $('leaderPrev');
  const next = $('leaderNext');
  if(prev) prev.onclick = () => {
    leaderboardPreviewPage = Math.max(0, Number(leaderboardPreviewPage || 0) - 1);
    renderSubmitLeaderboardPreview();
  };
  if(next) next.onclick = () => {
    leaderboardPreviewPage = Math.min(totalPages - 1, Number(leaderboardPreviewPage || 0) + 1);
    renderSubmitLeaderboardPreview();
  };
}
function scrollToSubmitResult(){
  requestAnimationFrame(() => $('submitResult').scrollIntoView({behavior:'smooth', block:'start'}));
}
function renderSubmitLeaderboardPreview(){
  const target = $('submitLeaderboardPreview');
  if(!target) return;
  const creditName = ($('contributorName').value || 'anonymous').trim();
  const publicAnonymous = !!$('publicAnonymous')?.checked;
  const publicName = publicCreditLabel(creditName, publicAnonymous, 'pending');
  const previewName = publicAnonymous ? 'You (anonymous)' : publicName;
  const model = pendingLeaderboardModel(publicName, publicAnonymous, selected?.turns || 0, selected?.compactions || 0, {}, previewName);
  target.innerHTML = `
    <div class="leader-note"><span><strong>Pending score: ${model.localPendingLow}–${model.localPendingHigh} points if accepted.</strong> This preview shows the public leaderboard name before submission.</span></div>
    <div class="leaderboard-preview">
      ${leaderboardPreviewHtml(model)}
      <div class="public-credit-option">
        <label><input id="publicAnonymous" type="checkbox" ${publicAnonymous ? 'checked' : ''}> Show me anonymously on the public leaderboard</label>
        <div class="hint">Default is public credit. If selected, your row appears as ${escapeHtml(previewName)} in this preview; maintainers can still see the name, email, and institute above for review and support.</div>
      </div>
    </div>
  `;
  bindLeaderboardPager(model);
  $('publicAnonymous').onchange = renderSubmitLeaderboardPreview;
}
function renderSubmitResult(data){
  const receipt = data.receipt || {};
  const duplicate = !!data.duplicate || !!receipt.duplicate;
  const duplicateDetail = (data.duplicate_detail || receipt.duplicate_detail || '').trim();
  const duplicateText = duplicateDetail
    ? `The maintainer relay rejected this repeat attempt: ${duplicateDetail}.`
    : 'The maintainer relay recognized this redacted artifact or source session as already received.';
  const publicId = (receipt.submission || '').replace(/^pending\//, '').replace(/\/$/, '') || 'not available';
  const idHint = receipt.submission
    ? 'Save this ID for support. Maintainers can use it to find your private staging submission.'
    : 'The receipt was saved locally, but no staging submission ID was returned.';
  const creditName = (receipt.credit_name || receipt.contributor || $('contributorName').value || 'Contributor').trim();
  const publicAnonymous = !!receipt.public_anonymous;
  const publicCreditName = publicCreditLabel(creditName, publicAnonymous, publicId);
  const firstName = creditName.split(/\s+/)[0] || 'Contributor';
  const turns = Number(receipt.turns || 0);
  const compactions = Number(receipt.compactions || 0);
  const localPending = data.local_pending || {};
  const model = pendingLeaderboardModel(publicCreditName, publicAnonymous, turns, compactions, localPending, publicCreditName);
  const highValue = model.highValue;
  const localPendingSessions = model.localPendingSessions;
  const localPendingLow = model.localPendingLow;
  const localPendingHigh = model.localPendingHigh;
  const localPendingRange = `${localPendingLow}–${localPendingHigh}`;
  const uploads = (receipt.uploads || [])
    .map(m => `<div class="file-pill">${escapeHtml(m.source)}</div>`)
    .join('');
  const emailHref = receipt.contributor_email ? receiptEmailHref(receipt, data.receipt_path) : '';
  if(duplicate){
    $('submitResult').innerHTML = `
      <div class="success-layout">
        <div class="success-main">
          <div class="success-hero">
            <div class="success-check">✓</div>
            <div>
              <div class="success-title">Already submitted</div>
              <div class="success-subtitle">${escapeHtml(duplicateText)} We marked it donated locally to prevent repeat uploads.</div>
            </div>
          </div>
          <div class="leader-note"><span><strong>No new donation was needed.</strong> This repeat attempt will not be counted again until the same source session has enough new research signal.</span></div>
          ${data.receipt_path ? `<div class="receipt-card"><div class="receipt-head">Local duplicate receipt</div><div class="copybox"><span>${escapeHtml(data.receipt_path)}</span><button class="copy-mini" type="button" id="copyReceiptPath">Copy</button></div><div class="hint">This receipt records that the duplicate was detected locally; it is not a new donation.</div></div>` : ''}
        </div>
        <aside class="success-detail-card">
          <div class="detail-section">
            <div class="detail-heading"><span class="detail-icon">◷</span><span>Status</span></div>
            <div class="detail-chip">Duplicate detected</div>
          </div>
          <div class="detail-section">
            <div class="detail-heading"><span class="detail-icon">▧</span><span>Donation result</span></div>
            <div class="detail-value">Already received</div>
            <div class="hint">The maintainer relay recognized this redacted session content and skipped the repeat submission.</div>
          </div>
          ${data.receipt_path ? `<div class="detail-section"><div class="detail-heading"><span class="detail-icon">▤</span><span>Receipt</span></div><div class="row"><button id="revealReceipt" type="button">Reveal Receipt</button>${emailHref ? `<a href="${escapeHtml(emailHref)}"><button class="secondary" type="button">Email Receipt</button></a>` : ''}</div><div class="hint">${emailHref ? 'Email opens your mail app with receipt details; no email is sent by the local tool.' : 'No email was provided, so the duplicate receipt was saved locally only.'}</div></div>` : ''}
          <button id="submitAnother" class="secondary" style="width:100%">＋ Submit another session</button>
        </aside>
      </div>
    `;
    $('submitResult').classList.add('show', 'success-panel');
    if(data.receipt_path && $('revealReceipt')) $('revealReceipt').onclick = () => post('/api/open_path', {path:data.receipt_path, reveal:true}).catch(e => status('submitStatus','ERROR: '+e.message));
    if($('copyReceiptPath') && data.receipt_path) $('copyReceiptPath').onclick = () => navigator.clipboard?.writeText(data.receipt_path).catch(()=>{});
    $('submitAnother').onclick = () => { resetSessionArtifacts(); goStep(1); };
    scrollToSubmitResult();
    return;
  }
  $('submitResult').innerHTML = `
    <div class="success-layout">
      <div class="success-main">
        <div class="success-hero">
          <div class="success-check">✓</div>
          <div>
            <div class="success-title">${duplicate ? 'Already received' : `Thank you, ${escapeHtml(firstName)}.`}</div>
            <div class="success-subtitle">${duplicate ? 'This verified redacted session was already submitted. We marked it donated locally to prevent repeat uploads.' : (publicAnonymous ? 'Your verified redacted session is submitted for maintainer review. Public credit will appear under an anonymous donor label.' : 'Your verified redacted session is submitted for maintainer review and release credit.')}</div>
          </div>
        </div>
        <div class="credit-scoreboard">
          <div class="credit-card"><div class="credit-icon">☆</div><div><strong>+2</strong><span>Base points if accepted</span></div></div>
          <div class="credit-card"><div class="credit-icon">◇</div><div><strong>${highValue ? '+1' : '+0'}</strong><span>${highValue ? 'High-value session bonus' : 'High-value bonus pending'}</span></div></div>
          <div class="credit-card"><div class="credit-icon">▣</div><div><strong>+1</strong><span>Possible coverage / usability bonus</span></div></div>
        </div>
        <div class="leader-note"><span><strong>Pending score: ${localPendingRange} points across ${localPendingSessions} pending session${localPendingSessions === 1 ? '' : 's'}.</strong> Accepted donations appear on the contributor leaderboard and release acknowledgments.</span></div>
        <div class="leaderboard-preview">
          ${leaderboardPreviewHtml(model)}
        </div>
        ${data.receipt_path ? `<div class="receipt-card"><div class="receipt-head">Receipt</div><div class="copybox"><span>${escapeHtml(data.receipt_path)}</span><button class="copy-mini" type="button" id="copyReceiptPath">Copy</button></div><div class="hint">Email opens your mail app with the receipt details; no email is sent by the local tool.</div></div>` : ''}
      </div>
      <aside class="success-detail-card">
        <div class="detail-section">
          <div class="detail-heading"><span class="detail-icon">◷</span><span>Status</span></div>
          <div class="detail-chip">${duplicate ? 'Already submitted' : 'Pending maintainer review'}</div>
        </div>
        <div class="detail-section">
          <div class="detail-heading"><span class="detail-icon">♙</span><span>Public credit</span></div>
          <div class="detail-value">${escapeHtml(publicCreditName)}</div>
          ${publicAnonymous ? `<div class="hint">Maintainers can still see your submitted name, email, and institute for review and support.</div>` : ''}
        </div>
        <div class="detail-section">
          <div class="detail-heading"><span class="detail-icon">▧</span><span>Submission ID</span></div>
          <div class="copybox"><span>${escapeHtml(publicId)}</span><button class="copy-mini" type="button" id="copySubmissionId">Copy</button></div>
          <div class="hint">${escapeHtml(idHint)}</div>
        </div>
        ${data.receipt_path ? `<div class="detail-section"><div class="detail-heading"><span class="detail-icon">▤</span><span>Receipt</span></div><div class="row"><button id="revealReceipt" type="button">Reveal Receipt</button>${emailHref ? `<a href="${escapeHtml(emailHref)}"><button class="secondary" type="button">Email Receipt</button></a>` : ''}</div><div class="hint">${emailHref ? 'Email opens your mail app with receipt details; no email is sent by the local tool.' : 'No email was provided, so the receipt was saved locally only.'}</div></div>` : ''}
        ${uploads ? `<div class="detail-section"><div class="detail-heading"><span class="detail-icon">▱</span><span>Submitted files</span></div><div class="file-list">${uploads}</div></div>` : ''}
        <button id="submitAnother" class="secondary" style="width:100%">＋ Submit another session</button>
      </aside>
    </div>
  `;
  $('submitResult').classList.add('show', 'success-panel');
  if(data.receipt_path) $('revealReceipt').onclick = () => post('/api/open_path', {path:data.receipt_path, reveal:true}).catch(e => status('submitStatus','ERROR: '+e.message));
  if($('copySubmissionId')) $('copySubmissionId').onclick = () => navigator.clipboard?.writeText(publicId).catch(()=>{});
  if($('copyReceiptPath') && data.receipt_path) $('copyReceiptPath').onclick = () => navigator.clipboard?.writeText(data.receipt_path).catch(()=>{});
  $('submitAnother').onclick = () => { resetSessionArtifacts(); goStep(1); };
  scrollToSubmitResult();
}
function resetSessionArtifacts(){
  selected = null;
  redacted = null;
  appliedScrubTerms = [];
  redactionCache = new Map();
  submitted = false;
  document.querySelectorAll('.session-row.selected').forEach(x=>x.classList.remove('selected'));
  ['selectedCard','redactResult','submitResult','searchResult'].forEach(id => {
    $(id).innerHTML = '';
    $(id).classList.remove('show', 'success-panel');
  });
  $('searchPanel').classList.remove('show');
  $('reviewConfirm').checked = false;
  $('safeConfirm').checked = false;
  ['redactStatus','submitStatus','discoverStatus'].forEach(id => status(id, ''));
  ['redactProgress','submitProgress','searchProgress'].forEach(id => setBusy(id, false));
  refreshButtons();
}
function setProgress(pct){
  const clamped = Math.max(0, Math.min(100, pct));
  $('discoverProgress').style.display = 'block';
  $('discoverProgress').firstElementChild.style.width = clamped + '%';
  if(progressTimers.discoverProgress) progressTimers.discoverProgress.pct = clamped;
  updateProgressTime('discoverProgress');
}
const progressTimers = {};
function fmtElapsed(ms){
  const seconds = Math.max(0, Math.floor(ms / 1000));
  const mins = Math.floor(seconds / 60);
  const secs = seconds % 60;
  return mins ? `${mins}m ${secs}s` : `${secs}s`;
}
function progressTimeEl(id){
  const progress = $(id);
  let el = progress.nextElementSibling;
  if(!el || !el.classList.contains('progress-time')){
    el = document.createElement('div');
    el.className = 'progress-time';
    progress.insertAdjacentElement('afterend', el);
  }
  return el;
}
function updateProgressTime(id, text='', opts={}){
  const timer = progressTimers[id];
  const el = progressTimeEl(id);
  const pct = timer ? timer.pct : opts.percent;
  const prefix = Number.isFinite(pct) ? `${Math.round(pct)}% · ` : '';
  if(!timer){
    if(opts.keep && text){
      el.textContent = prefix + text;
      el.style.display = 'block';
    } else {
      el.style.display = 'none';
      el.textContent = '';
    }
    return;
  }
  const elapsed = fmtElapsed(Date.now() - timer.start);
  const displayText = text || (timer.refreshText ? timer.refreshText() : timer.text) || `Elapsed ${elapsed}`;
  const longNote = Date.now() - timer.start > 120000 ? ' · still running on a large session' : '';
  timer.text = displayText;
  el.textContent = prefix + displayText.replace(/Elapsed \d+m? ?\d*s?/, `Elapsed ${elapsed}`) + longNote;
  el.style.display = 'block';
}
function progressBreakdown(parts){
  return Object.entries(parts || {})
    .filter(([name,ms]) => ms >= 1000 && !['starting','done'].includes(name))
    .map(([name,ms]) => `${name}: ${fmtElapsed(ms)}`)
    .join(' · ');
}
function setBusy(id, on, pct=35, opts={}){
  const el = $(id);
  const previousPct = progressTimers[id] ? progressTimers[id].pct : undefined;
  let clamped = Math.max(0, Math.min(100, pct));
  if(on && Number.isFinite(previousPct)) clamped = Math.max(clamped, previousPct);
  el.style.display = on ? 'block' : 'none';
  el.firstElementChild.style.width = on ? clamped + '%' : '0%';
  if(on && !progressTimers[id]){
    progressTimers[id] = {start: Date.now(), text: ''};
    progressTimers[id].interval = setInterval(() => updateProgressTime(id), 1000);
  }
  if(on && progressTimers[id]) progressTimers[id].pct = clamped;
  if(!on && progressTimers[id]){
    if(progressTimers[id].interval) clearInterval(progressTimers[id].interval);
    delete progressTimers[id];
  }
  updateProgressTime(id, opts.finalText || '', {keep:opts.keepTime, percent:opts.percent ?? previousPct});
}
async function post(url, body){
  const r = await fetch(url, {method:'POST', headers:{'content-type':'application/json'}, body:JSON.stringify(body)});
  const data = await r.json();
  if(!r.ok) throw new Error(data.error || r.statusText);
  return data;
}
async function postStream(url, body, onEvent){
  const r = await fetch(url, {method:'POST', headers:{'content-type':'application/json'}, body:JSON.stringify(body)});
  if(!r.ok){
    let data = {};
    try { data = await r.json(); } catch(e) {}
    throw new Error(data.error || r.statusText);
  }
  const reader = r.body.getReader();
  const decoder = new TextDecoder();
  let buf = '';
  while(true){
    const {value, done} = await reader.read();
    if(done) break;
    buf += decoder.decode(value, {stream:true});
    const lines = buf.split('\n');
    buf = lines.pop();
    for(const line of lines){
      if(!line.trim()) continue;
      const ev = JSON.parse(line);
      onEvent(ev);
      if(ev.event === 'error') throw new Error(ev.error || 'redaction failed');
    }
  }
  if(buf.trim()){
    const ev = JSON.parse(buf);
    onEvent(ev);
    if(ev.event === 'error') throw new Error(ev.error || 'redaction failed');
  }
}
function friendlyRequestError(e, action='operation'){
  const msg = String(e && e.message || e || '');
  const low = msg.toLowerCase();
  if(low.includes('network error') || low.includes('failed to fetch') || low.includes('load failed')){
    return `The local browser connection was interrupted during this ${action}. Keep this tab open and the computer awake, then click the button again.`;
  }
  return msg;
}
function sessionTableHead(){
  const arrow = key => sessionSort.key === key ? `<span class="sort-arrow">${sessionSort.dir === 'asc' ? '&uarr;' : '&darr;'}</span>` : '<span class="sort-arrow"></span>';
  const aria = key => sessionSort.key === key ? (sessionSort.dir === 'asc' ? 'ascending' : 'descending') : 'none';
  return `<div class="session-table-head">
    <div>#</div>
    <div><button type="button" class="sort-header" data-sort-key="session" aria-sort="${aria('session')}">Session${arrow('session')}</button></div>
    <div><button type="button" class="sort-header" data-sort-key="last_active" aria-sort="${aria('last_active')}">Last active${arrow('last_active')}</button></div>
    <div><button type="button" class="sort-header" data-sort-key="turns" aria-sort="${aria('turns')}">User turns${arrow('turns')}</button></div>
    <div><button type="button" class="sort-header" data-sort-key="compactions" aria-sort="${aria('compactions')}">Ctx cmp<span class="header-footnote">1</span>${arrow('compactions')}</button></div>
    <div><button type="button" class="sort-header" data-sort-key="fit" aria-sort="${aria('fit')}">Fit${arrow('fit')}</button></div>
    <div></div>
  </div>`;
}
function fitLabel(value){
  return value ? value.charAt(0).toUpperCase() + value.slice(1) : '';
}
function sortableDateValue(s){
  const raw = String(s?.last_active_ts || s?.modified_ts || s?.last_active || s?.modified || '');
  const time = Date.parse(raw);
  return Number.isFinite(time) ? time : 0;
}
function sessionGroupValue(s){
  return [
    s?.agent || '',
    s?.model || '',
    s?.session_label || s?.project || '',
    s?.path || ''
  ].map(x => String(x).toLowerCase()).join('\u0000');
}
function sessionSearchText(s){
  const donationInfo = localDonationInfo(s);
  return [
    s?.agent || '',
    s?.model || '',
    s?.session_label || '',
    s?.project || '',
    s?.last_active || '',
    s?.modified || '',
    s?.path || '',
    s?.relay_submission_id || '',
    donationInfo.supportId || '',
    fitLabel(fit(s)),
  ].map(x => String(x).toLowerCase()).join(' ');
}
function filteredSessionItems(){
  const q = sessionSearchQuery.trim().toLowerCase();
  return sessions.map((s, idx) => ({s, idx})).filter(item => !q || sessionSearchText(item.s).includes(q));
}
function fitSortValue(s){
  return {best:3, good:2, improve:1}[fit(s)] || 0;
}
function sessionSortValue(s, key){
  if(key === 'session') return sessionGroupValue(s);
  if(key === 'last_active') return sortableDateValue(s);
  if(key === 'turns') return Number(s?.turns || 0);
  if(key === 'compactions') return Number(s?.compactions || 0);
  if(key === 'fit') return fitSortValue(s);
  return 0;
}
function sortedSessions(){
  const items = filteredSessionItems();
  if(!sessionSort.key) return items;
  const dir = sessionSort.dir === 'asc' ? 1 : -1;
  return items.sort((a, b) => {
    const av = sessionSortValue(a.s, sessionSort.key);
    const bv = sessionSortValue(b.s, sessionSort.key);
    if(av < bv) return -1 * dir;
    if(av > bv) return 1 * dir;
    return a.idx - b.idx;
  });
}
function bindSessionSortHeaders(){
  document.querySelectorAll('[data-sort-key]').forEach(btn => {
    btn.onclick = event => {
      event.stopPropagation();
      const key = btn.dataset.sortKey || '';
      const defaultDir = key === 'last_active' || key === 'turns' || key === 'compactions' || key === 'fit' ? 'desc' : 'asc';
      sessionSort = sessionSort.key === key
        ? {key, dir: sessionSort.dir === 'desc' ? 'asc' : 'desc'}
        : {key, dir: defaultDir};
      page = 0;
      renderSessions();
    };
  });
}
function hideSessionMenu(){
  const menu = $('sessionMenu');
  if(menu) menu.classList.remove('show');
}
function showSessionMenu(event, session, donationInfo){
  const menu = $('sessionMenu');
  if(!menu || !donationInfo?.supportId || !donationInfo?.donatedBefore) return;
  event.preventDefault();
  event.stopPropagation();
  menu.innerHTML = `
    <button type="button" role="menuitem" data-session-action="update">Update info</button>
    <button type="button" role="menuitem" data-session-action="problem" class="danger">Report problem</button>
  `;
  menu.querySelector('[data-session-action="update"]').onclick = e => {
    e.stopPropagation();
    hideSessionMenu();
    beginMetadataUpdate(session, donationInfo.supportId);
  };
  menu.querySelector('[data-session-action="problem"]').onclick = e => {
    e.stopPropagation();
    hideSessionMenu();
    beginSupportRequest(session, donationInfo.supportId);
  };
  menu.classList.add('show');
  const rect = menu.getBoundingClientRect();
  const margin = 10;
  const left = Math.min(event.clientX, window.innerWidth - rect.width - margin);
  const top = Math.min(event.clientY, window.innerHeight - rect.height - margin);
  menu.style.left = `${Math.max(margin, left)}px`;
  menu.style.top = `${Math.max(margin, top)}px`;
}
function renderSessions(){
  const list = $('sessionList');
  list.innerHTML = '';
  const sorted = sortedSessions();
  const totalVisible = sorted.length;
  const totalPages = Math.max(1, Math.ceil(totalVisible / pageSize));
  if(page >= totalPages) page = totalPages - 1;
  const start = page * pageSize;
  const rows = sorted.slice(start, start + pageSize);
  const allDonated = allSessionsDonated();
  const counts = fitCounts();
  const readyCount = (counts.best || 0) + (counts.good || 0);
  const donatedTotal = sessions.reduce((count, s) => {
    const info = localDonationInfo(s);
    return count + (info.exactDonated || info.donatedBefore ? 1 : 0);
  }, 0);
  const agentCounts = agentFamilyCounts();
  const sessionSummaryTitle = `Claude: ${agentCounts.claude}\nCodex: ${agentCounts.codex}\nOther: ${agentCounts.other}`;
  const readySummaryTitle = `Best: ${counts.best || 0}\nGood: ${counts.good || 0}`;
  $('sessionCount').dataset.tooltip = sessionSummaryTitle;
  $('sessionCount').setAttribute('aria-label', sessionSummaryTitle);
  $('sessionCount').innerHTML = `<strong>${sessions.length}</strong><span>found</span>`;
  $('fitSummary').innerHTML = sessions.length
    ? `<span class="fit-chip donated">Donated ${donatedTotal}</span><span class="fit-chip ready" data-tooltip="${escapeHtml(readySummaryTitle)}" aria-label="${escapeHtml(readySummaryTitle)}">Ready ${readyCount}</span><span class="fit-chip improve" data-tooltip="Not ready yet: needs more turns or a context compaction" aria-label="Not ready yet: needs more turns or a context compaction">Keep chatting ${counts.improve || 0}</span>`
    : '';
  if(!rows.length){
    const searched = $('discoverProgress').style.display === 'block';
    const emptyText = searched
      ? `<div class="empty-sessions thanks">Thanks for considering a ContextEcho donation.<span>We did not find any Claude Code or Codex sessions on this machine yet. Feel free to keep using your coding agent and come back later; we will continue collecting donations.</span></div>`
      : (sessions.length && sessionSearchQuery.trim()
        ? '<div class="empty-sessions">No sessions match this search.</div>'
        : '<div class="empty-sessions">No sessions found yet. Click Discover Sessions to scan this machine.</div>');
    list.innerHTML = `${sessionTableHead()}${emptyText}`;
    bindSessionSortHeaders();
  }
  if(rows.length){
    list.innerHTML = sessionTableHead();
    bindSessionSortHeaders();
  }
  rows.forEach(({s}, i) => {
    const idx = start + i;
    const row = document.createElement('div');
    const donationInfo = localDonationInfo(s);
    const donated = donationInfo.exactDonated || (donationInfo.donatedBefore && !donationInfo.updateReady);
    const ready = sessionReady(s);
    const updateReady = donationInfo.updateReady;
    const statusPill = updateReady
      ? `<span class="pill best">update ready · +${escapeHtml(compactNumber(donationInfo.newTurns))} turns</span>`
      : (donationInfo.donatedBefore
        ? `<span class="pill donated">donated${donationInfo.newTurns ? ` · +${escapeHtml(compactNumber(donationInfo.newTurns))} turns` : ''}</span>`
        : '');
    const supportPill = donationInfo.supportId
      ? `<span class="pill support-id" data-copy-submission="${escapeHtml(donationInfo.supportId)}" title="Click to copy maintainer reset ID">ID ${escapeHtml(donationInfo.supportId)}</span>`
      : '';
    row.className = donated ? 'session-row donated-row' : (ready ? 'session-row' : 'session-row improve-row');
    if(donationInfo.donatedBefore) row.classList.add('donated-history-row');
    const currentFit = fit(s);
    const hasMenuActions = donationInfo.supportId && donationInfo.donatedBefore;
    const chipLine = [statusPill, supportPill].filter(Boolean).join(' ');
    row.innerHTML = `
      <div class="session-icon">${idx + 1}</div>
      <div class="session-main">
        <div class="session-title-line">${escapeHtml(s.agent || 'Session')} &middot; ${escapeHtml(s.session_label || s.project || 'unknown project')}</div>
        <div class="session-chip-row">${chipLine}</div>
      </div>
      <div class="session-date">${escapeHtml(s.last_active || s.modified || '?')}</div>
      <div class="session-turns"><div class="session-num">${compactNumber(s.turns)}</div></div>
      <div class="session-cmp"><div class="session-num">${s.compactions || 0}</div></div>
      <div class="session-fit"><span class="pill ${currentFit}">${currentFit === 'improve' ? '<span class="fit-arrow">&uarr;</span>' : '<span class="fit-star">&#9733;</span>'}${fitLabel(currentFit)}</span></div>
      <button type="button" class="session-chevron" aria-label="${hasMenuActions ? 'Open session actions' : 'Session details'}" ${hasMenuActions ? 'title="Actions: update info or report problem"' : 'disabled'}>&rsaquo;</button>
    `;
    row.querySelectorAll('[data-copy-submission]').forEach(el => {
      el.onclick = event => {
        event.stopPropagation();
        const id = el.dataset.copySubmission || '';
        navigator.clipboard?.writeText(id).catch(()=>{});
        status('discoverStatus', `Copied maintainer reset ID: ${id}`);
      };
    });
    row.oncontextmenu = event => showSessionMenu(event, s, donationInfo);
    const chevron = row.querySelector('.session-chevron');
    if(chevron && hasMenuActions) chevron.onclick = event => showSessionMenu(event, s, donationInfo);
    if (selected && selected.path === s.path && !donated && ready) row.classList.add('selected');
    row.onclick = () => {
      if(!ready){
        status('discoverStatus', 'This session is not ready to donate yet. Keep working until it reaches 50+ turns.');
        return;
      }
      if(donated){
        status('discoverStatus', donationInfo.newTurns ? `This session was donated before and has ${compactNumber(donationInfo.newTurns)} new turns, but it is below the update threshold. Keep working until it has at least 50 new turns or 20% growth.` : 'This session is already marked donated. Click the ID pill to copy the maintainer reset ID if support is needed.');
        return;
      }
      document.querySelectorAll('.session-row.selected').forEach(x=>x.classList.remove('selected'));
      row.classList.add('selected'); selected = s;
      redacted = null; appliedScrubTerms = []; redactionCache = new Map(); submitted = !!donated;
      renderSelectedCard(s, idx);
      status('redactStatus', '');
      status('discoverStatus', updateReady ? `This session was donated before and now has ${compactNumber(donationInfo.newTurns)} new turns. You can submit an updated version for maintainer review.` : '');
      refreshButtons();
    };
    list.appendChild(row);
  });
  if(allDonated && totalVisible === sessions.length){
    selected = null;
    redacted = null;
    submitted = false;
    $('selectedCard').innerHTML = '';
    $('selectedCard').classList.remove('show');
    $('reviewConfirm').checked = false;
    list.insertAdjacentHTML('beforeend', `<div class="all-donated-note">Thank you for donating all your scanned session data.<span>All ${sessions.length} discovered session${sessions.length === 1 ? '' : 's'} are already marked donated. Click an ID pill to copy the maintainer reset ID if support is needed.</span></div>`);
  }
  $('pageInfo').textContent = `Page ${page + 1} of ${totalPages} · showing ${totalVisible ? start + 1 : 0}-${Math.min(start + pageSize, totalVisible)} of ${totalVisible}${totalVisible !== sessions.length ? ` filtered from ${sessions.length}` : ''}`;
  $('prevPage').disabled = page <= 0;
  $('nextPage').disabled = page >= totalPages - 1;
  $('pager').style.display = totalVisible > pageSize ? 'flex' : 'none';
  refreshButtons();
}
$('discoverBtn').onclick = async () => {
  $('discoverBtn').disabled = true;
  status('discoverStatus','Scanning local session logs. This can take a minute for large histories...');
  progressTimers.discoverProgress = {start: Date.now()};
  setProgress(2);
  let discoverTiming = '';
  try {
    const max = '50';
    const r = await fetch('/api/discover_stream?max_per_agent=' + max);
    if(!r.ok) throw new Error(await r.text());
    const reader = r.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let final = null;
    while(true){
      const {done, value} = await reader.read();
      if(done) break;
      buffer += decoder.decode(value, {stream:true});
      const lines = buffer.split('\n');
      buffer = lines.pop();
      for(const line of lines){
        if(!line.trim()) continue;
        const ev = JSON.parse(line);
        if(ev.event === 'adapter_start'){
          status('discoverStatus', `Scanning ${ev.agent}...`);
        } else if(ev.event === 'inspect'){
          const limit = ev.adapter_limit || 50;
          const pct = max === 'all' ? Math.min(90, 5 + ev.inspected) : Math.min(90, ((ev.inspected || 0) / (limit * 2)) * 90);
          setProgress(pct);
          status('discoverStatus', `${ev.agent}: inspected ${ev.adapter_inspected}${ev.adapter_limit ? '/' + ev.adapter_limit : ''}; found ${ev.found} sessions so far.`);
        } else if(ev.event === 'adapter_done'){
          status('discoverStatus', `${ev.agent}: done. ${ev.found} sessions so far.`);
        } else if(ev.event === 'done'){
          final = ev;
          setProgress(100);
        }
      }
    }
    sessions = (final && final.sessions) || [];
    page = 0;
    discoverTiming = `Completed in ${fmtElapsed(Date.now() - progressTimers.discoverProgress.start)}`;
    status('discoverStatus', sessions.length === 0
      ? noSessionsMessage()
      : (!relayStatusChecked()
        ? `Found ${sessions.length} sessions. Donation status could not be checked with the relay, so previously donated sessions may not be marked here. You can still pick a session to donate.`
        : (allSessionsDonated() ? allSessionsDonatedMessage() : `Found ${sessions.length} sessions. Click a row to select.`)));
    renderSessions();
  } catch(e) { status('discoverStatus','ERROR: '+friendlyRequestError(e, 'discovery scan')); }
  finally {
    if(!discoverTiming && progressTimers.discoverProgress) discoverTiming = `Stopped after ${fmtElapsed(Date.now() - progressTimers.discoverProgress.start)}`;
    $('discoverBtn').disabled = false;
    delete progressTimers.discoverProgress;
    updateProgressTime('discoverProgress', discoverTiming, {keep:!!discoverTiming});
  }
};
$('prevPage').onclick = () => { if(page > 0){ page--; renderSessions(); } };
$('nextPage').onclick = () => { if((page + 1) * pageSize < sortedSessions().length){ page++; renderSessions(); } };
$('sessionSearch').oninput = () => {
  sessionSearchQuery = $('sessionSearch').value || '';
  page = 0;
  renderSessions();
};
document.addEventListener('click', hideSessionMenu);
document.addEventListener('keydown', event => { if(event.key === 'Escape') hideSessionMenu(); });
window.addEventListener('resize', hideSessionMenu);
window.addEventListener('scroll', hideSessionMenu, true);
$('safeConfirm').onchange = refreshButtons;
$('reviewConfirm').onchange = refreshButtons;
document.querySelectorAll('input[name="privacyTier"]').forEach(el => {
  el.onchange = () => {
    redacted = null;
    appliedScrubTerms = [];
    $('reviewConfirm').checked = false;
    $('redactResult').classList.remove('show');
    $('searchPanel').classList.remove('show');
    $('searchResult').classList.remove('show');
    if(!restoreCachedRedaction()) status('redactStatus', 'Privacy mode changed. Click Redact and Verify again before moving on.');
    refreshButtons();
  };
});
$('pickNext').onclick = () => goStep(2);
$('redactPrev').onclick = () => goStep(1);
$('redactNext').onclick = () => goStep(3);
$('submitPrev').onclick = () => goStep(2);
$('metadataUpdateBtn').onclick = () => sendMetadataUpdate();
$('supportRequestBtn').onclick = () => sendSupportRequest();
$('metadataBackBtn').onclick = () => goStep(1);
['contributorName','contributorEmail','contributorInstitute'].forEach(id => {
  $(id).oninput = () => {
    if(id === 'contributorEmail') updateEmailSuggestions();
    renderSubmitLeaderboardPreview();
    refreshButtons();
  };
});
$('contributorEmail').onblur = completeEmailDomain;
$('searchBtn').onclick = async () => {
  if(!redacted) return;
  setUiProcessing(true);
  setBusy('searchProgress', true, 55);
  let searchTiming = '';
  try {
    const data = await post('/api/search_redacted', {redacted_file:redacted.redacted_file, terms:$('searchTerms').value});
    setBusy('searchProgress', true, 100);
    searchTiming = `Completed in ${fmtElapsed(Date.now() - progressTimers.searchProgress.start)}`;
    updateProgressTime('searchProgress', searchTiming);
    renderSearchResult(data);
  } catch(e) { status('redactStatus','ERROR: '+friendlyRequestError(e, 'private-word check')); }
  finally {
    setBusy('searchProgress', false, 35, {keepTime:!!searchTiming, finalText:searchTiming});
    setUiProcessing(false);
  }
};
async function runRedactVerify(extraTerms = [], opts = {}){
  if(!selected) return;
  setUiProcessing(true);
  const progressId = opts.fromSearch ? 'searchProgress' : 'redactProgress';
  const stageTimes = {};
  let stageName = 'starting';
  let stageLabel = opts.fromSearch ? 'Redacting checked word' : 'Starting redaction';
  let stageStart = Date.now();
  const markStage = (next, label) => {
    const now = Date.now();
    stageTimes[stageName] = (stageTimes[stageName] || 0) + (now - stageStart);
    stageName = next;
    stageLabel = label || next;
    stageStart = now;
  };
  let progressTimingText = '';
  const showTiming = (label='Elapsed') => {
    const live = {...stageTimes, [stageName]: (stageTimes[stageName] || 0) + (Date.now() - stageStart)};
    const total = progressTimers[progressId] ? fmtElapsed(Date.now() - progressTimers[progressId].start) : '0s';
    const breakdown = progressBreakdown(live);
    progressTimingText = breakdown ? `${stageLabel} · ${label} ${total} · ${breakdown}` : `${stageLabel} · ${label} ${total}`;
    if(progressTimers[progressId]){
      progressTimers[progressId].refreshText = () => {
        const liveNow = {...stageTimes, [stageName]: (stageTimes[stageName] || 0) + (Date.now() - stageStart)};
        const totalNow = fmtElapsed(Date.now() - progressTimers[progressId].start);
        const breakdownNow = progressBreakdown(liveNow);
        return breakdownNow ? `${stageLabel} · ${label} ${totalNow} · ${breakdownNow}` : `${stageLabel} · ${label} ${totalNow}`;
      };
    }
    updateProgressTime(progressId, progressTimingText);
  };
  $('redactBtn').disabled = true;
  $('redactResult').classList.remove('show');
  if(!opts.fromSearch){
    $('searchPanel').classList.remove('show');
    $('searchResult').classList.remove('show');
  }
  setBusy(opts.fromSearch ? 'redactProgress' : 'searchProgress', false);
  setBusy(progressId, true, 30);
  status('redactStatus', '');
  try {
    let finalData = null;
    const directTerms = [...new Set((extraTerms || []).map(x => String(x || '').trim()).filter(Boolean))];
    const pendingScrubTerms = [...new Set([...newScrubTerms(), ...directTerms])];
    const canRepair = !!(
      redacted &&
      redacted.redacted_file &&
      redacted.privacy_tier === privacyTier() &&
      (pendingScrubTerms.length || hasDetectSecretsFailure(redacted))
    );
    const previousStats = canRepair ? {...(redacted.stats || {})} : null;
    const scrubForRun = canRepair ? pendingScrubTerms.join(', ') : '';
    if(redacted && redacted.verify_passed && !canRepair && redacted.privacy_tier === privacyTier() && !pendingScrubTerms.length){
      status('redactStatus', 'No new private words to redact. Review the current redacted file or use Check File for another word.');
      renderRedactResult(redacted);
      refreshButtons();
      return;
    }
    await postStream('/api/redact_stream', {
      path:selected.path,
      scrub:scrubForRun,
      auto:selected,
      confirm_safe:$('safeConfirm').checked,
      privacy_tier:privacyTier(),
      repair_allowed: canRepair,
      previous_redacted_file: canRepair ? redacted.redacted_file : ''
    }, ev => {
      if(ev.event === 'start'){
        markStage('preparing', 'Preparing redaction');
        setBusy(progressId, true, 5);
      } else if(ev.event === 'repair'){
        markStage('repair', 'Redacting checked word');
        setBusy(progressId, true, ev.percent || 55);
      } else if(ev.event === 'engine'){
        markStage('engine', 'Loading redaction engine');
        setBusy(progressId, true, 8);
      } else if(ev.event === 'progress'){
        if(stageName !== 'redacting') markStage('redacting', 'Redacting locally');
        const pct = Math.max(5, Math.min(92, ev.percent || 5));
        setBusy(progressId, true, pct);
      } else if(ev.event === 'minimize'){
        markStage('minimizing', 'Applying user-minimized mode');
        setBusy(progressId, true, 94);
      } else if(ev.event === 'verify'){
        markStage('verifying', ev.message || 'Verifying redacted file');
        setBusy(progressId, true, ev.percent || 96);
      } else if(ev.event === 'done'){
        markStage('done', 'Redaction complete');
        finalData = ev.result;
      }
      showTiming();
    });
    if(!finalData) throw new Error('redaction did not return a result');
    if(previousStats) finalData.stats = mergeRedactionStats(previousStats, finalData.stats || {});
    redacted = finalData;
    setBusy(progressId, true, 100);
    submitted = false;
    if(redacted.verify_passed){
      appliedScrubTerms = canRepair
        ? [...new Set([...appliedScrubTerms, ...pendingScrubTerms])]
        : [];
      redactionCache.set(redactionCacheKey(), {data:redacted, appliedScrubTerms:[...appliedScrubTerms]});
    }
    $('reviewConfirm').checked = false;
    renderRedactResult(redacted);
    status('redactStatus', redacted.verify_passed ? 'Review the result above. If a private word remains, use Check File to redact it. Otherwise check the review box to continue.' : verifyFailureSummary(redacted));
    refreshButtons();
  } catch(e) { status('redactStatus','ERROR: '+friendlyRequestError(e, 'redaction and verification')); }
  finally {
    if(progressTimers[progressId]) showTiming(progressTimingText ? 'Completed in' : 'Stopped after');
    setBusy(progressId, false, 35, {keepTime:!!progressTimingText, finalText:progressTimingText});
    setUiProcessing(false);
    refreshButtons();
  }
}
$('redactBtn').onclick = () => runRedactVerify();
$('submitBtn').onclick = async () => {
  if(!redacted) return;
  if(!contributorFieldsComplete()){
    refreshButtons();
    return;
  }
  if(!confirm('Upload verified redacted artifacts as a PR?')) return;
  setUiProcessing(true);
  $('submitBtn').disabled = true;
  $('submitResult').classList.remove('show');
  setBusy('submitProgress', true, 10);
  status('submitStatus','Preparing upload...');
  const submitStages = {};
  let submitStage = 'preparing';
  let submitStageLabel = 'Preparing upload';
  let submitStageStart = Date.now();
  let submitTimingText = '';
  const markSubmitStage = (next, label) => {
    const now = Date.now();
    submitStages[submitStage] = (submitStages[submitStage] || 0) + (now - submitStageStart);
    submitStage = next;
    submitStageLabel = label || next;
    submitStageStart = now;
  };
  const showSubmitTiming = (label='Elapsed') => {
    const live = {...submitStages, [submitStage]: (submitStages[submitStage] || 0) + (Date.now() - submitStageStart)};
    const total = progressTimers.submitProgress ? fmtElapsed(Date.now() - progressTimers.submitProgress.start) : '0s';
    const breakdown = progressBreakdown(live);
    submitTimingText = breakdown ? `${submitStageLabel} · ${label} ${total} · ${breakdown}` : `${submitStageLabel} · ${label} ${total}`;
    updateProgressTime('submitProgress', submitTimingText);
  };
  try {
    const payload = {
      redacted_file:redacted.redacted_file,
      source_path:selected ? selected.path : '',
      auto:selected,
      privacy_tier:redacted.privacy_tier || privacyTier(),
      contributor:$('contributorName').value,
      email:$('contributorEmail').value,
      institute:$('contributorInstitute').value,
      public_anonymous:$('publicAnonymous').checked
    };
    const started = await post('/api/submit_job', payload);
    let data = null;
    while(true){
      await new Promise(resolve => setTimeout(resolve, 1000));
      const job = await post('/api/submit_status', {job_id: started.job_id});
      if(job.status === 'error') throw new Error(job.error || job.message || 'submit failed');
      markSubmitStage(job.status === 'done' ? 'done' : 'uploading', job.message || 'Submitting donation');
      setBusy('submitProgress', true, job.percent || 45);
      status('submitStatus', '');
      showSubmitTiming();
      if(job.status === 'done'){
        data = job.result;
        showSubmitTiming();
        break;
      }
    }
    if(!data) throw new Error('submit did not return a result');
    setBusy('submitProgress', true, 100);
    submitted = true;
    if(selected && selected.path){
      selected.donated = true;
      selected.donated_before = true;
      selected.update_ready = false;
      selected.new_turns = 0;
      selected.donated_turns = Number(selected.turns || 0);
      donatedPaths.add(sessionLocalKey(selected));
      donatedRecords[sessionPathKey(selected)] = {
        turns: Number(selected.turns || 0),
        records: Number(selected.records || 0),
        compactions: Number(selected.compactions || 0),
        submission: data.receipt?.submission || '',
        submission_id: normalizeSubmissionId(data.receipt?.submission || ''),
        credit_name: data.receipt?.credit_name || data.receipt?.contributor || $('contributorName').value || '',
        contributor_email: data.receipt?.contributor_email || $('contributorEmail').value || '',
        institute: data.receipt?.institute || $('contributorInstitute').value || '',
        public_anonymous: !!data.receipt?.public_anonymous,
        submitted_at: new Date().toISOString()
      };
      saveDonatedPaths();
      saveDonatedRecords();
      renderSessions();
    }
    renderSubmitResult(data);
    status('submitStatus', allSessionsDonated() ? allSessionsDonatedMessage() : (data.duplicate ? 'This session was already received. It is now marked donated locally.' : 'Submission marked donated locally. Pick another session to submit more.'));
    refreshButtons();
  }
  catch(e) { status('submitStatus','ERROR: '+friendlyRequestError(e, 'submission')); }
  finally {
    if(progressTimers.submitProgress) showSubmitTiming(submitTimingText ? 'Completed in' : 'Stopped after');
    setBusy('submitProgress', false, 35, {keepTime:!!submitTimingText, finalText:submitTimingText});
    setUiProcessing(false);
    refreshButtons();
  }
};
loadProjectStats();
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    server_version = "ContextEchoDonateWeb/0.1"

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("[web] " + (fmt % args) + "\n")

    def _write_body(self, body: bytes, *, stream: bool = False) -> bool:
        try:
            self.wfile.write(body)
            if stream:
                self.wfile.flush()
            return True
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError) as exc:
            if stream:
                raise ClientDisconnected() from exc
            return False
        except OSError as exc:
            if exc.errno in CLIENT_DISCONNECT_ERRNOS:
                if stream:
                    raise ClientDisconnected() from exc
                return False
            raise

    def _json(self, payload: dict, status: int = 200, *, cors: bool = False) -> None:
        body = json.dumps(payload, indent=2).encode()
        try:
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            if cors:
                self.send_header("access-control-allow-origin", "*")
                self.send_header("access-control-allow-methods", "GET, OPTIONS")
                self.send_header("access-control-allow-headers", "content-type")
            self.end_headers()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            return
        except OSError as exc:
            if exc.errno in CLIENT_DISCONNECT_ERRNOS:
                return
            raise
        self._write_body(body)

    def _read_json(self) -> dict:
        n = int(self.headers.get("content-length", "0"))
        return json.loads(self.rfile.read(n).decode() or "{}")

    def _start_ndjson(self) -> None:
        try:
            self.send_response(200)
            self.send_header("content-type", "application/x-ndjson")
            self.send_header("cache-control", "no-store")
            self.end_headers()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError) as exc:
            raise ClientDisconnected() from exc
        except OSError as exc:
            if exc.errno in CLIENT_DISCONNECT_ERRNOS:
                raise ClientDisconnected() from exc
            raise

    def _event(self, payload: dict) -> None:
        self._write_body((json.dumps(payload) + "\n").encode(), stream=True)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            body = INDEX_HTML.encode()
            self.send_response(200)
            self.send_header("content-type", "text/html; charset=utf-8")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self._write_body(body)
            return
        if parsed.path == "/api/health":
            self._json({"ok": True, "app": "ContextEcho donation wizard"}, cors=True)
            return
        if parsed.path == "/api/discover":
            qs = parse_qs(parsed.query)
            raw_max = qs.get("max_per_agent", ["50"])[0]
            max_per_agent = None if raw_max == "all" else int(raw_max)
            sessions = discover_mod.discover(max_per_agent=max_per_agent, progress=False)
            self._json({"sessions": annotate_donated(sessions)})
            return
        if parsed.path == "/api/project_stats":
            self._json(project_stats())
            return
        if parsed.path == "/api/discover_stream":
            qs = parse_qs(parsed.query)
            raw_max = qs.get("max_per_agent", ["50"])[0]
            max_per_agent = None if raw_max == "all" else int(raw_max)
            self.send_response(200)
            self.send_header("content-type", "application/x-ndjson")
            self.send_header("cache-control", "no-store")
            self.end_headers()
            try:
                for event in discover_mod.discover_iter(max_per_agent=max_per_agent):
                    if event.get("event") == "done":
                        event = dict(event)
                        event["sessions"] = annotate_donated(list(event.get("sessions") or []))
                    self._write_body((json.dumps(event) + "\n").encode(), stream=True)
            except ClientDisconnected:
                return
            return
        self._json({"error": "not found"}, 404)

    def do_OPTIONS(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/health":
            self.send_response(204)
            self.send_header("access-control-allow-origin", "*")
            self.send_header("access-control-allow-methods", "GET, OPTIONS")
            self.send_header("access-control-allow-headers", "content-type")
            self.end_headers()
            return
        self.send_response(404)
        self.end_headers()

    def do_POST(self) -> None:
        try:
            if self.path == "/api/redact":
                self._handle_redact()
            elif self.path == "/api/redact_stream":
                self._handle_redact_stream()
            elif self.path == "/api/describe":
                self._handle_describe()
            elif self.path == "/api/describe_stream":
                self._handle_describe_stream()
            elif self.path == "/api/submit":
                self._handle_submit()
            elif self.path == "/api/submit_stream":
                self._handle_submit_stream()
            elif self.path == "/api/submit_job":
                self._handle_submit_job()
            elif self.path == "/api/submit_status":
                self._handle_submit_status()
            elif self.path == "/api/open_path":
                self._handle_open_path()
            elif self.path == "/api/search_redacted":
                self._handle_search_redacted()
            elif self.path == "/api/metadata_update":
                self._handle_metadata_update()
            elif self.path == "/api/support_request":
                self._handle_support_request()
            else:
                self._json({"error": "not found"}, 404)
        except ClientDisconnected:
            return
        except Exception as e:
            try:
                self._json({"error": str(e)}, 400)
            except ClientDisconnected:
                return

    def _handle_redact(self) -> None:
        data = self._read_json()
        self._json(self._redact_payload(data))

    def _redact_payload(self, data: dict, emit=None) -> dict:
        if not data.get("confirm_safe"):
            raise ValueError("safety confirmation is required")
        scrub_terms = {t.strip() for t in str(data.get("scrub", "")).split(",") if t.strip()}
        privacy_tier = str(data.get("privacy_tier") or "full_redacted")
        if privacy_tier not in {"full_redacted", "user_minimized"}:
            raise ValueError("invalid privacy tier")
        previous = Path(data.get("previous_redacted_file", "")).expanduser()
        if data.get("repair_allowed") and previous.exists() and is_redacted_artifact(previous):
            if not previous.resolve().is_relative_to(DONATION_ROOT.resolve()):
                raise ValueError("repair is only allowed for local donation output files")
            if emit:
                emit({
                    "event": "repair",
                    "percent": 45,
                    "message": "Applying new private words and credential cleanup to the existing redacted file...",
                })
            stats = redact_mod.apply_scrub_terms_to_file(previous, previous, scrub_terms)
            if emit:
                emit({
                    "event": "repair",
                    "percent": 82,
                    "message": "Fast repair complete. Preparing verify gate...",
                })
                emit({
                    "event": "verify",
                    "percent": 90,
                    "message": "Verify 1/2: scanning for residual emails, paths, API keys, and secrets...",
                })
            verify_report = verify_mod.verify_session(previous)
            verify_report, stats, auto_repair_passes = _auto_repair_until_verified(
                previous,
                verify_report,
                stats,
                emit=emit,
            )
            if emit:
                emit({
                    "event": "verify",
                    "percent": 99,
                    "message": "Verify 2/2: checking final result...",
                })
            if verify_report.get("passed"):
                submit_mod.write_verify_cache(previous, verify_report)
            return {
                "redacted_file": str(previous),
                "output_dir": str(previous.parent),
                "stats": stats,
                "privacy_tier": privacy_tier,
                "verify_passed": bool(verify_report.get("passed")),
                "verify_report": verify_report,
                "repair_used": True,
                "auto_repair_passes": auto_repair_passes,
                "repair_terms": sorted(scrub_terms),
            }

        src = Path(data.get("path", "")).expanduser()
        if not src.exists():
            raise ValueError(f"not found: {src}")
        if is_redacted_artifact(src):
            raise ValueError("selected file already looks redacted; choose the original session log")
        auto = data.get("auto") or discover_mod.inspect_session(src)
        out_dir = donation_output_dir(auto)
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / redacted_output_name(src)
        try:
            with src.open("r", encoding="utf-8", errors="replace") as f:
                total = sum(1 for _ in f)
        except Exception:
            total = 0
        if emit:
            emit({"event": "start", "total": total})
        last_pct = -1

        def on_progress(current: int, total_records: int) -> None:
            nonlocal last_pct
            if not emit:
                return
            pct = int((current / total_records) * 90) if total_records else 90
            if pct != last_pct or current == total_records:
                last_pct = pct
                emit({
                    "event": "progress",
                    "current": current,
                    "total": total_records,
                    "percent": pct,
                })

        if emit:
            emit({"event": "engine"})
        stats = redact_mod.redact_file_with_progress(
            src,
            out,
            scrub_terms,
            progress=False,
            progress_callback=on_progress if emit else None,
        )
        if privacy_tier == "user_minimized":
            if emit:
                emit({"event": "minimize"})
            min_stats = minimize_mod.minimize_file(out, out)
            stats.update({f"minimize_{k}": v for k, v in min_stats.items()})
        if emit:
            emit({
                "event": "verify",
                "percent": 96,
                "message": "Verify 1/2: scanning for residual emails, paths, API keys, and secrets...",
            })
        verify_report = verify_mod.verify_session(out)
        verify_report, stats, auto_repair_passes = _auto_repair_until_verified(
            out,
            verify_report,
            stats,
            emit=emit,
        )
        if emit:
            emit({
                "event": "verify",
                "percent": 99,
                "message": "Verify 2/2: checking final result...",
            })
        verify_ok = bool(verify_report.get("passed"))
        if verify_ok:
            submit_mod.write_verify_cache(out, verify_report)
        return {
            "redacted_file": str(out),
            "output_dir": str(out_dir),
            "stats": stats,
            "privacy_tier": privacy_tier,
            "verify_passed": verify_ok,
            "verify_report": verify_report,
            "repair_used": False,
            "auto_repair_passes": auto_repair_passes,
            "scrub_terms": sorted(scrub_terms),
        }

    def _handle_redact_stream(self) -> None:
        data = self._read_json()
        self._start_ndjson()
        try:
            result = self._redact_payload(data, emit=self._event)
            self._event({"event": "done", "result": result})
        except ClientDisconnected:
            return
        except Exception as exc:
            tb = traceback.format_exc()
            print(tb, file=sys.stderr, end="")
            support_id = report_wizard_error(data, exc, "Redaction", tb)
            error = stream_error_message(exc, "Redaction")
            if support_id:
                error = f"{error} A sanitized diagnostic was sent to the maintainer relay as {support_id}."
            self._event({"event": "error", "error": error})

    def _handle_describe(self) -> None:
        data = self._read_json()
        self._json(self._describe_payload(data))

    def _describe_payload(self, data: dict, emit=None) -> dict:
        session = Path(data.get("redacted_file", "")).expanduser()
        if not session.exists():
            raise ValueError(f"not found: {session}")
        contributor_fields = required_contributor_fields(data)
        if emit:
            emit({"event": "progress", "percent": 20, "message": "Checking verified redacted file..."})
        auto = metadata_for_redacted_artifact(data, session)
        if emit:
            emit({"event": "progress", "percent": 45, "message": "Inferring manifest metadata..."})
        manifest, consent, _ = describe_mod.write_manifest_and_consent(
            session=session,
            auto=auto,
            domain=infer_domain(auto),
            language=infer_language(auto),
            contributor=contributor_fields["contributor"],
            email=contributor_fields["email"],
            institute=contributor_fields["institute"],
            privacy_tier=str(data.get("privacy_tier") or "full_redacted"),
            public_anonymous=bool(data.get("public_anonymous")),
        )
        if emit:
            emit({"event": "progress", "percent": 80, "message": "Writing manifest and consent files..."})
        return {"manifest": str(manifest), "consent": str(consent)}

    def _handle_describe_stream(self) -> None:
        data = self._read_json()
        self._start_ndjson()
        try:
            result = self._describe_payload(data, emit=self._event)
            self._event({"event": "progress", "percent": 100, "message": "Manifest and consent ready."})
            self._event({"event": "done", "result": result})
        except ClientDisconnected:
            return
        except Exception as exc:
            tb = traceback.format_exc()
            print(tb, file=sys.stderr, end="")
            support_id = report_wizard_error(data, exc, "Manifest preparation", tb)
            error = stream_error_message(exc, "Manifest preparation")
            if support_id:
                error = f"{error} A sanitized diagnostic was sent to the maintainer relay as {support_id}."
            self._event({"event": "error", "error": error})

    def _handle_submit(self) -> None:
        data = self._read_json()
        self._json(self._submit_payload(data))

    def _submit_payload(self, data: dict, emit=None) -> dict:
        session = Path(data.get("redacted_file", "")).expanduser()
        if not session.exists():
            raise ValueError(f"not found: {session}")
        if emit:
            emit({"event": "progress", "percent": 15, "message": "Checking local submission files..."})
        source_path = data.get("source_path")
        if already_submitted(source_path, session):
            raise ValueError(
                "This session or redacted artifact is already marked submitted locally. "
                "Pick another session, or use Clear local donated labels only if the previous "
                "submission truly failed."
            )
        auto = metadata_for_redacted_artifact(data, session)
        if not donation_ready(auto.get("turns", 0), auto.get("compactions", 0)):
            raise ValueError("This session is not ready to donate yet. Keep working until it reaches 50+ turns.")
        if emit:
            emit({"event": "progress", "percent": 30, "message": "Writing manifest and consent files..."})
        describe_result = self._describe_payload(data)
        if emit:
            emit({"event": "progress", "percent": 50, "message": "Manifest and consent ready."})
        if emit:
            emit({"event": "progress", "percent": 65, "message": "Confirming verified artifact, then uploading donation..."})
        rc, output = run_submit_with_heartbeats(session, emit=emit)
        if rc != 0 and is_duplicate_submit_output(output):
            duplicate_detail = duplicate_submit_detail(output)
            receipt_path, receipt = write_receipt(session, source_path or "", "[submit] Submission ID: submission-already-received")
            receipt["duplicate"] = True
            receipt["duplicate_detail"] = duplicate_detail
            save_donation_record(source_path=source_path or "", artifact_path=session, output="[submit] Submission ID: submission-already-received", receipt=receipt)
            if emit:
                emit({"event": "progress", "percent": 95, "message": "Local duplicate receipt saved."})
            return {
                "duplicate": True,
                "duplicate_detail": duplicate_detail,
                "output": output,
                "receipt_path": str(receipt_path),
                "receipt": receipt,
                "local_pending": local_pending_summary(receipt),
                "manifest": describe_result.get("manifest"),
                "consent": describe_result.get("consent"),
            }
        if rc != 0:
            raise ValueError(friendly_submit_error(output or f"submit failed with code {rc}"))
        receipt_path, receipt = write_receipt(session, source_path or "", output)
        save_donation_record(source_path=source_path or "", artifact_path=session, output=output, receipt=receipt)
        if emit:
            emit({"event": "progress", "percent": 95, "message": "Local receipt saved."})
        return {
            "output": output,
            "receipt_path": str(receipt_path),
            "receipt": receipt,
            "local_pending": local_pending_summary(receipt),
            "manifest": describe_result.get("manifest"),
            "consent": describe_result.get("consent"),
        }

    def _handle_submit_stream(self) -> None:
        data = self._read_json()
        self._start_ndjson()
        try:
            result = self._submit_payload(data, emit=self._event)
            self._event({"event": "progress", "percent": 100, "message": "Submission complete."})
            self._event({"event": "done", "result": result})
        except ClientDisconnected:
            return
        except Exception as exc:
            tb = traceback.format_exc()
            print(tb, file=sys.stderr, end="")
            support_id = report_wizard_error(data, exc, "Submission", tb)
            error = stream_error_message(exc, "Submission")
            if support_id:
                error = f"{error} A sanitized diagnostic was sent to the maintainer relay as {support_id}."
            self._event({"event": "error", "error": error})

    def _handle_submit_job(self) -> None:
        data = self._read_json()
        job_id = uuid.uuid4().hex[:12]
        update_submit_job(
            job_id,
            id=job_id,
            status="running",
            percent=5,
            message="Preparing submission...",
            started_at=time.time(),
        )

        def emit_job(event: dict) -> None:
            if event.get("event") == "progress":
                update_submit_job(
                    job_id,
                    percent=int(event.get("percent") or 0),
                    message=str(event.get("message") or ""),
                )

        def worker() -> None:
            try:
                result = self._submit_payload(data, emit=emit_job)
                update_submit_job(
                    job_id,
                    status="done",
                    percent=100,
                    message="Submission complete.",
                    result=result,
                )
            except Exception as exc:
                update_submit_job(
                    job_id,
                    status="error",
                    error=str(exc),
                    message=str(exc),
                )

        threading.Thread(target=worker, daemon=True).start()
        self._json({"job_id": job_id})

    def _handle_submit_status(self) -> None:
        data = self._read_json()
        job_id = str(data.get("job_id") or "")
        job = get_submit_job(job_id)
        if not job:
            raise ValueError("submit job not found")
        self._json(job)

    def _handle_metadata_update(self) -> None:
        data = self._read_json()
        payload = {
            "submission_id": str(data.get("submission_id") or "").strip(),
            "credit_name": str(data.get("credit_name") or "").strip(),
            "contributor_email": str(data.get("contributor_email") or "").strip(),
            "contributor_institute": str(data.get("contributor_institute") or "").strip(),
            "public_anonymous": bool(data.get("public_anonymous")),
            "source_session_id": str(data.get("source_session_id") or "").strip(),
            "conversation_fingerprint": str(data.get("conversation_fingerprint") or "").strip(),
            "reason": "donor requested contributor metadata update",
        }
        if not payload["submission_id"]:
            raise ValueError("Missing submission ID")
        if not any([payload["credit_name"], payload["contributor_email"], payload["contributor_institute"]]):
            raise ValueError("Edit at least one contributor field before sending an info update")
        self._json(relay_metadata_update(payload))

    def _handle_support_request(self) -> None:
        data = self._read_json()
        payload = {
            "submission_id": str(data.get("submission_id") or "").strip(),
            "reason": str(data.get("reason") or "other").strip(),
            "message": str(data.get("message") or "").strip(),
            "source_session_id": str(data.get("source_session_id") or "").strip(),
            "conversation_fingerprint": str(data.get("conversation_fingerprint") or "").strip(),
        }
        if not payload["submission_id"]:
            raise ValueError("Missing submission ID")
        self._json(relay_support_request(payload))

    def _handle_open_path(self) -> None:
        data = self._read_json()
        path = Path(data.get("path", "")).expanduser()
        if not path.exists():
            raise ValueError(f"not found: {path}")
        if sys.platform == "darwin":
            cmd = ["open", "-R", str(path)] if data.get("reveal") else ["open", str(path)]
        elif sys.platform.startswith("linux"):
            target = path.parent if data.get("reveal") and path.is_file() else path
            cmd = ["xdg-open", str(target)]
        elif sys.platform.startswith("win"):
            cmd = ["explorer", f"/select,{path}"] if data.get("reveal") else ["explorer", str(path)]
        else:
            raise ValueError(f"open path is unsupported on {sys.platform}")
        subprocess.run(cmd, check=False)
        self._json({"ok": True})

    def _handle_search_redacted(self) -> None:
        data = self._read_json()
        path = Path(data.get("redacted_file", "")).expanduser()
        if not path.exists():
            raise ValueError(f"not found: {path}")
        if not is_redacted_artifact(path):
            raise ValueError("search is only allowed on redacted donation artifacts")
        terms = [t.strip() for t in str(data.get("terms", "")).split(",") if t.strip()]
        text = path.read_text(encoding="utf-8", errors="replace")
        self._json({"results": [{"term": term, "count": text.count(term)} for term in terms]})


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run the local ContextEcho donation web wizard.")
    p.add_argument("--host", default="127.0.0.1", help="bind host; default is local-only")
    p.add_argument("--port", type=int, default=8766)
    p.add_argument("--no-open", action="store_true", help="do not open a browser")
    args = p.parse_args(argv)

    try:
        server, actual_port = create_server(args.host, args.port)
    except OSError as exc:
        if exc.errno == errno.EADDRINUSE:
            print(f"[web] ERROR: port {args.port} is already in use.")
            print(f"[web] Try: python3 -m donate --web --web-port {args.port + 1}")
            return 2
        raise

    if actual_port != args.port:
        print(f"[web] port {args.port} is already in use; using {actual_port} instead.")
    url = f"http://{args.host}:{actual_port}/"
    print(f"[web] ContextEcho donation wizard: {url}")
    print("[web] Raw sessions stay local. Press Ctrl-C to stop.")
    if not args.no_open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[web] stopped.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
