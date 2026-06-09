"""Optional donor-privacy minimization for redacted session JSONL files.

This pass runs after normal PII/secret redaction. It keeps assistant/tool
behavior available for ContextEcho while masking donor-authored free text.
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
    r"depressed|depression|overwhelmed|burned out|burnt out"
    r")\b",
    re.IGNORECASE,
)
CODE_HINT_RE = re.compile(r"(```|def |class |function |const |let |var |import |from |SELECT |<[^>]+>)")


def approx_tokens(text: str) -> int:
    return max(1, len(text.split()))


def classify_user_text(text: str) -> dict[str, Any]:
    tags: list[str] = []
    if PERSONAL_RE.search(text):
        tags.append("personal")
    if CODE_HINT_RE.search(text):
        tags.append("contains_code")
    if not tags:
        tags.append("task_or_instruction")
    summary = "personal non-task statement" if "personal" in tags else "user task or instruction"
    return {
        "content": "<USER_PERSONAL_TEXT_REDACTED>" if "personal" in tags else "<USER_TEXT_REDACTED>",
        "summary": summary,
        "tags": tags,
        "approx_tokens": approx_tokens(text),
    }


def is_user_turn(obj: dict[str, Any]) -> bool:
    role = str(obj.get("role", "")).lower()
    typ = str(obj.get("type", "")).lower()
    return role == "user" or typ == "user"


def minimize_value(value: Any, stats: dict[str, int]) -> Any:
    if isinstance(value, str):
        stats["user_strings_minimized"] = stats.get("user_strings_minimized", 0) + 1
        if PERSONAL_RE.search(value):
            stats["personal_user_strings_minimized"] = stats.get("personal_user_strings_minimized", 0) + 1
        return classify_user_text(value)
    if isinstance(value, list):
        return [minimize_value(v, stats) for v in value]
    if isinstance(value, dict):
        return {k: minimize_value(v, stats) for k, v in value.items()}
    return value


def minimize_user_turn(obj: dict[str, Any], stats: dict[str, int]) -> dict[str, Any]:
    out = dict(obj)
    stats["user_turns_minimized"] = stats.get("user_turns_minimized", 0) + 1
    for key in ("content", "message", "text", "prompt"):
        if key in out:
            out[key] = minimize_value(out[key], stats)
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
            lines_out.append(json.dumps(classify_user_text(line), ensure_ascii=False, separators=(",", ":")))
            continue
        if isinstance(obj, dict) and is_user_turn(obj):
            obj = minimize_user_turn(obj, stats)
        lines_out.append(json.dumps(obj, ensure_ascii=False, separators=(",", ":")))
    dst.write_text("\n".join(lines_out) + "\n", encoding="utf-8")
    return stats
