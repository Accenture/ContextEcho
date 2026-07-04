"""Claude Code session discovery."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

from donate.adapters.base import (
    GenericJsonlAdapter,
    first_path_hint,
    is_redacted_artifact,
    iter_jsonl,
    safe_project_name_from_path,
    session_label,
)


UUID_RE = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")


def resume_session_id_from_path(path: Path) -> str:
    match = UUID_RE.fullmatch(path.stem) or UUID_RE.search(path.stem)
    return match.group(0).lower() if match else ""


def _existing_path_from_slug_tokens(tokens: list[str]) -> Path | None:
    def walk(root: Path, remaining: list[str]) -> Path | None:
        if not remaining:
            return root if root.exists() else None
        for end in range(1, len(remaining) + 1):
            candidate = root / "-".join(remaining[:end])
            if not candidate.exists():
                continue
            resolved = walk(candidate, remaining[end:])
            if resolved is not None:
                return resolved
        return None

    return walk(Path("/"), tokens)


def resume_dir_from_project_slug(slug: str) -> str:
    tokens = [part for part in slug.split("-") if part]
    if not tokens:
        return ""
    existing = _existing_path_from_slug_tokens(tokens)
    if existing is not None and existing.is_dir():
        return str(existing)
    if len(tokens) >= 2 and tokens[0].lower() == "users":
        prefix = Path("/") / tokens[0] / tokens[1]
        rest = tokens[2:]
        if rest and rest[0].lower() in {"desktop", "documents", "downloads", "library", "projects"}:
            prefix /= rest[0]
            rest = rest[1:]
        return str(prefix / "-".join(rest)) if rest else str(prefix)
    return ""


def first_resume_dir_from_records(path: Path) -> str:
    for _, obj in iter_jsonl(path):
        resume_dir = first_path_hint(obj)
        if resume_dir:
            return resume_dir
    return ""


class ClaudeCodeAdapter(GenericJsonlAdapter):
    agent = "Claude Code"
    roots = [Path.home() / ".claude" / "projects"]

    def discover_paths(self) -> Iterable[Path]:
        for root in self.roots:
            if root.exists():
                yield from (
                    p for p in sorted(root.rglob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
                    if not is_redacted_artifact(p) and safe_project_name_from_path(p.parent.name).lower() != "subagents"
                )

    def can_inspect_path(self, path: Path) -> bool:
        if is_redacted_artifact(path):
            return False
        path_s = str(path.expanduser())
        return "/.claude/projects/" in path_s or any(root in path.parents for root in self.roots)

    def inspect(self, path: Path) -> dict:
        info = super().inspect(path)
        info["agent"] = self.agent
        info["project"] = safe_project_name_from_path(path.parent.name)
        resume_id = resume_session_id_from_path(path)
        info["session_label"] = session_label(
            info["project"],
            str(info.get("conversation_fingerprint") or ""),
            path,
            resume_id,
        )
        info["resume_dir"] = first_resume_dir_from_records(path) or resume_dir_from_project_slug(path.parent.name)
        if resume_id:
            info["resume_session_id"] = resume_id
            info["resume_command"] = f"claude --resume {resume_id}"
        info["source_format"] = "claude-code-jsonl"
        info["confidence"]["agent"] = "high"
        return info
