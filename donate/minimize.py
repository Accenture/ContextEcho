"""Optional donor-privacy minimization for redacted session JSONL files.

This pass runs after normal PII/secret redaction. It keeps assistant/tool
behavior available for ContextEcho while masking sensitive donor-authored text.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

PERSONAL_RE = re.compile(
    r"\b("
    r"feel|felt|feeling|stressed|stress|anxious|anxiety|sad|angry|upset|"
    r"worried|scared|afraid|personal|private|family|relationship|health|"
    r"depressed|depression|overwhelmed|burned out|burnt out|panic|lonely|"
    r"grief|trauma|therapy|therapist|suicide|self-harm"
    r")\b",
    re.IGNORECASE,
)
PRIVATE_LIFE_RE = re.compile(
    r"\b("
    r"medical|doctor|diagnosis|medication|hospital|illness|disability|"
    r"salary|debt|bankruptcy|mortgage|rent|visa|immigration|lawsuit|"
    r"divorce|pregnant|pregnancy|child|children|parent|spouse|partner"
    r")\b",
    re.IGNORECASE,
)
IDENTITY_RE = re.compile(
    r"\b("
    r"my age is|i am \d{1,3}|i'm \d{1,3}|my gender|my race|my religion|"
    r"my nationality|my disability|my political"
    r")\b",
    re.IGNORECASE,
)
TOXIC_RE = re.compile(
    r"\b("
    r"fuck|shit|bitch|asshole|bastard|idiot|stupid|hate|kill myself|"
    r"slur"
    r")\b",
    re.IGNORECASE,
)
CONFIDENTIAL_RE = re.compile(
    r"\b("
    r"nda|confidential|client[- ]confidential|do not share|internal only|"
    r"secret project|codename|password|token|api key|credential"
    r")\b",
    re.IGNORECASE,
)
TEXT_KEYS = {"content", "message", "text", "prompt", "input", "text_elements"}
SENSITIVE_PATTERNS = [
    ("personal_feeling", PERSONAL_RE, "<USER_PRIVATE_FEELING_REDACTED>"),
    ("private_life", PRIVATE_LIFE_RE, "<USER_PRIVATE_DETAIL_REDACTED>"),
    ("identity_disclosure", IDENTITY_RE, "<USER_IDENTITY_DISCLOSURE_REDACTED>"),
    ("toxic_language", TOXIC_RE, "<USER_TOXIC_LANGUAGE_REDACTED>"),
    ("confidential", CONFIDENTIAL_RE, "<USER_CONFIDENTIAL_DETAIL_REDACTED>"),
]


def approx_tokens(text: str) -> int:
    return max(1, len(text.split()))


def minimize_user_text(text: str, stats: dict[str, int]) -> str:
    """Mask sensitive spans while keeping task/code semantics."""
    out = text
    for tag, pattern, replacement in SENSITIVE_PATTERNS:
        out, n = pattern.subn(replacement, out)
        if n:
            stats["user_sensitive_spans_minimized"] = stats.get("user_sensitive_spans_minimized", 0) + n
            stats[f"user_{tag}_spans_minimized"] = stats.get(f"user_{tag}_spans_minimized", 0) + n
    if out != text:
        stats["user_strings_minimized"] = stats.get("user_strings_minimized", 0) + 1
    return out


def is_user_turn(obj: dict[str, Any]) -> bool:
    role = str(obj.get("role", "")).lower()
    typ = str(obj.get("type", "")).lower()
    payload = obj.get("payload") if isinstance(obj.get("payload"), dict) else {}
    payload_role = str(payload.get("role", "")).lower()
    payload_type = str(payload.get("type", "")).lower()
    return (
        role == "user"
        or typ == "user"
        or payload_type == "user_message"
        or (payload_type == "message" and payload_role == "user")
    )


def minimize_value(value: Any, stats: dict[str, int]) -> Any:
    if isinstance(value, str):
        return minimize_user_text(value, stats)
    if isinstance(value, list):
        return [minimize_value(v, stats) for v in value]
    if isinstance(value, dict):
        return {
            k: minimize_value(v, stats) if k in TEXT_KEYS else minimize_text_fields(v, stats)
            for k, v in value.items()
        }
    return value


def minimize_text_fields(value: Any, stats: dict[str, int]) -> Any:
    """Preserve JSON schema while masking user-authored textual fields."""
    if isinstance(value, list):
        return [minimize_text_fields(v, stats) for v in value]
    if isinstance(value, dict):
        return {
            k: minimize_value(v, stats) if k in TEXT_KEYS else minimize_text_fields(v, stats)
            for k, v in value.items()
        }
    return value


def minimize_user_turn(obj: dict[str, Any], stats: dict[str, int]) -> dict[str, Any]:
    out = dict(obj)
    stats["user_turns_minimized"] = stats.get("user_turns_minimized", 0) + 1
    for key in ("content", "message", "text", "prompt"):
        if key in out:
            out[key] = minimize_value(out[key], stats)
    if isinstance(out.get("payload"), dict):
        payload = dict(out["payload"])
        out["payload"] = minimize_text_fields(payload, stats)
    out["privacy_tier"] = "user_minimized"
    return out


def minimize_file(src: Path, dst: Path) -> dict[str, int]:
    stats: dict[str, int] = {}
    lines_out: list[str] = []
    for line in src.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            lines_out.append(line)
            continue
        try:
            obj = json.loads(line)
        except Exception:
            stats["raw_lines_minimized"] = stats.get("raw_lines_minimized", 0) + 1
            lines_out.append(json.dumps(minimize_user_text(line, stats), ensure_ascii=False, separators=(",", ":")))
            continue
        if isinstance(obj, dict) and is_user_turn(obj):
            obj = minimize_user_turn(obj, stats)
        lines_out.append(json.dumps(obj, ensure_ascii=False, separators=(",", ":")))
    dst.write_text("\n".join(lines_out) + "\n", encoding="utf-8")
    return stats
