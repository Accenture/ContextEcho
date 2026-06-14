"""Shared primitives for agent-specific session discovery adapters."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable, Protocol


MODEL_RE = re.compile(r'"model"\s*:\s*"([^"]+)"')
FINGERPRINT_ROWS = 48


class SessionAdapter(Protocol):
    """Adapter contract for one coding-agent log format."""

    agent: str
    roots: list[Path]

    def discover_paths(self) -> Iterable[Path]:
        """Yield candidate session files."""

    def inspect(self, path: Path) -> dict:
        """Return normalized metadata for a candidate path."""

    def can_inspect_path(self, path: Path) -> bool:
        """Whether this adapter is the best default for a manual path."""


def guess_org(model: str) -> str:
    m = model.lower()
    table = [
        ("claude", "Anthropic"),
        ("gpt", "OpenAI"),
        ("o1", "OpenAI"),
        ("o3", "OpenAI"),
        ("o4", "OpenAI"),
        ("gemini", "Google"),
        ("deepseek", "DeepSeek"),
        ("llama", "Meta"),
        ("qwen", "Alibaba"),
        ("mistral", "Mistral"),
        ("command", "Cohere"),
        ("kimi", "Moonshot"),
        ("nemotron", "NVIDIA"),
    ]
    for key, org in table:
        if key in m:
            return org
    return "unknown"


def is_redacted_artifact(path: Path) -> bool:
    """Whether a JSONL path looks like an output from this donation tool."""
    parts = {p.lower() for p in path.parts}
    name = path.name.lower()
    return (
        ".redacted." in name
        or name.endswith(".redacted.jsonl")
        or "contextecho_donations" in parts
    )


def safe_project_name_from_path(value: str | Path) -> str:
    """Display-only project hint that avoids showing user/org path scaffolding."""
    raw = str(value)
    if not raw:
        return "(unknown project)"
    # Handle normal paths and Claude's dash-flattened path slugs.
    tokens = [t for t in re.split(r"[/\\-]+", raw) if t]
    skip = {
        "users",
        "home",
        "documents",
        "library",
        "cloudstorage",
        "desktop",
        "downloads",
        "projects",
        "onedrive",
        "sessions",
        ".codex",
        ".claude",
    }
    lowered = [t.lower() for t in tokens]
    try:
        ui = lowered.index("users")
        if ui + 1 < len(tokens):
            skip.add(tokens[ui + 1].lower())
    except ValueError:
        pass
    meaningful = [t for t in tokens if t.lower() not in skip]
    hint = "-".join(meaningful[-3:]) if meaningful else Path(raw).stem or raw
    return hint[:48] or "(unknown project)"


def modified_date(path: Path) -> str:
    try:
        return dt.date.fromtimestamp(path.stat().st_mtime).isoformat()
    except Exception:
        return "?"


def date_from_timestamp(value: str) -> str | None:
    text = value.strip()
    if not text:
        return None
    try:
        normalized = text.replace("Z", "+00:00")
        return dt.datetime.fromisoformat(normalized).date().isoformat()
    except Exception:
        match = re.match(r"(\d{4}-\d{2}-\d{2})", text)
        return match.group(1) if match else None


def iter_jsonl(path: Path) -> Iterable[tuple[str, Any | None]]:
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                yield line, json.loads(line)
            except Exception:
                yield line, None
    except Exception:
        return


def walk_values(obj: Any) -> Iterable[tuple[str | None, Any]]:
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield str(key), value
            yield from walk_values(value)
    elif isinstance(obj, list):
        for value in obj:
            yield None, value
            yield from walk_values(value)


def first_value_for_keys(obj: Any | None, keys: set[str]) -> str:
    if obj is None:
        return ""
    for key, value in walk_values(obj):
        if key and key.lower() in keys and isinstance(value, str) and value:
            return value[:128]
    return ""


def collect_model_counts(line: str, obj: Any | None, models: dict[str, int]) -> None:
    if obj is not None:
        for key, value in walk_values(obj):
            if key and key.lower() == "model" and isinstance(value, str) and value:
                models[value] = models.get(value, 0) + 1
    if not models:
        for model in MODEL_RE.findall(line):
            models[model] = models.get(model, 0) + 1


def content_size(value: Any) -> int:
    if isinstance(value, str):
        return len(value)
    if isinstance(value, dict):
        return sum(content_size(v) for v in value.values())
    if isinstance(value, list):
        return sum(content_size(v) for v in value)
    return 0


def content_bucket(value: Any) -> str:
    size = content_size(value)
    if size <= 0:
        return "0"
    if size < 64:
        return "s"
    if size < 512:
        return "m"
    if size < 4096:
        return "l"
    return "xl"


def structural_record_signature(line: str, obj: Any | None) -> str:
    if not isinstance(obj, dict):
        return f"raw:{len(line) // 256}"
    payload = obj.get("payload") if isinstance(obj.get("payload"), dict) else {}
    message = obj.get("message") if isinstance(obj.get("message"), dict) else {}
    record_type = str(obj.get("type") or payload.get("type") or message.get("type") or "")[:48]
    role = str(obj.get("role") or payload.get("role") or message.get("role") or "")[:24]
    model = first_value_for_keys(obj, {"model"})
    timestamp = first_value_for_keys(obj, {"timestamp", "created_at", "createdat", "time"})[:32]
    content = obj.get("content") or obj.get("text") or payload.get("content") or payload.get("message") or message.get("content")
    turn = "u" if looks_like_human_turn(obj) else "n"
    compact = "c" if looks_like_compaction(line, obj) else "n"
    return "|".join([record_type, role, model, timestamp, content_bucket(content), turn, compact])


def conversation_fingerprint(path: Path, max_rows: int = FINGERPRINT_ROWS) -> str:
    h = hashlib.sha256()
    rows = 0
    for line, obj in iter_jsonl(path):
        h.update(structural_record_signature(line, obj).encode("utf-8"))
        h.update(b"\n")
        rows += 1
        if rows >= max_rows:
            break
    if rows == 0:
        try:
            fallback = f"{path.name}|{path.stat().st_size}"
        except OSError:
            fallback = str(path)
        h.update(fallback.encode("utf-8"))
    return f"conv-{h.hexdigest()[:16]}"


def dominant_model(models: dict[str, int]) -> str:
    if not models:
        return "unknown"
    dom = max(models, key=lambda k: models[k])
    if len(models) <= 1:
        return dom
    fam = re.sub(r"-\d+$", "", dom)
    return f"{fam}.x (mixed)" if "-" in dom else f"{dom} (mixed)"


def looks_like_compaction(line: str, obj: Any | None) -> bool:
    if '"isCompactSummary":true' in line:
        return True
    if obj is not None:
        for key, value in walk_values(obj):
            key_l = (key or "").lower()
            if isinstance(value, bool) and value and "compact" in key_l:
                return True
            if isinstance(value, str):
                val_l = value.lower()
                # Claude emits attachment rows pointing at compacted files; those
                # are references to a compaction, not additional compaction events.
                if val_l == "compact_file_reference":
                    continue
                if key_l in {"type", "event", "name", "kind"} and "compact" in val_l:
                    return True
    return False


def first_path_hint(obj: Any | None) -> str | None:
    if obj is None:
        return None
    keys = {"cwd", "workdir", "working_dir", "workspace", "project_path", "repo_path"}
    for key, value in walk_values(obj):
        if key and key.lower() in keys and isinstance(value, str) and value:
            return value
    return None


def timestamp_values(obj: Any | None) -> Iterable[str]:
    if obj is None:
        return []
    keys = {"timestamp", "created_at", "createdat", "time"}
    values: list[str] = []
    for key, value in walk_values(obj):
        if key and key.lower() in keys and isinstance(value, str) and value:
            values.append(value)
    return values


def content_has_human_input(content: Any) -> bool:
    if isinstance(content, str):
        return bool(content.strip())
    if isinstance(content, dict):
        item_type = str(content.get("type") or "").lower()
        if item_type in {"text", "input_text", "image", "image_url"}:
            return True
        text = content.get("text")
        return isinstance(text, str) and bool(text.strip())
    if isinstance(content, list):
        return any(content_has_human_input(item) for item in content)
    return False


def looks_like_human_turn(obj: Any | None) -> bool:
    """Whether a JSONL record is a human/user prompt turn.

    A turn here follows the multi-turn-conversation convention: user input that
    can elicit an assistant response. Tool results, assistant messages, system
    prompts, function calls, and bookkeeping events are not counted.
    """
    if not isinstance(obj, dict):
        return False

    payload = obj.get("payload")
    if isinstance(payload, dict):
        payload_type = str(payload.get("type") or "").lower()
        if payload_type == "user_message":
            return True
        if payload_type == "message" and payload.get("role") == "user":
            return content_has_human_input(payload.get("content"))

    message = obj.get("message")
    if isinstance(message, dict) and message.get("role") == "user":
        return content_has_human_input(message.get("content"))

    if obj.get("role") == "user":
        return content_has_human_input(obj.get("content") or obj.get("text"))

    return False


class GenericJsonlAdapter:
    """Best-effort fallback for unknown JSONL coding-agent logs."""

    agent = "Unknown agent"
    roots: list[Path] = []

    def discover_paths(self) -> Iterable[Path]:
        return []

    def can_inspect_path(self, path: Path) -> bool:
        return path.suffix == ".jsonl" and not is_redacted_artifact(path)

    def inspect(self, path: Path) -> dict:
        models: dict[str, int] = {}
        records = 0
        turns = 0
        compactions = 0
        project_hint: str | None = None
        event_dates: list[str] = []
        for line, obj in iter_jsonl(path):
            records += 1
            if looks_like_human_turn(obj):
                turns += 1
            collect_model_counts(line, obj, models)
            if looks_like_compaction(line, obj):
                compactions += 1
            if project_hint is None:
                project_hint = first_path_hint(obj)
            for timestamp in timestamp_values(obj):
                date = date_from_timestamp(timestamp)
                if date:
                    event_dates.append(date)

        model_label = dominant_model(models)
        started = min(event_dates) if event_dates else modified_date(path)
        last_active = max(event_dates) if event_dates else modified_date(path)
        return {
            "path": str(path),
            "project": safe_project_name_from_path(project_hint or path),
            "conversation_fingerprint": conversation_fingerprint(path),
            "fingerprint_version": "structure-v1",
            "modified": last_active,
            "started": started,
            "last_active": last_active,
            "records": records,
            "turns": turns,
            "compactions": compactions,
            "model": model_label,
            "org": guess_org(model_label),
            "models_seen": models,
            "agent": self.agent,
            "source_format": "generic-jsonl",
            "confidence": {
                "agent": "low",
                "model": "medium" if models else "low",
                "records": "high",
                "turns": "high" if turns else "low",
                "compactions": "medium",
            },
        }
