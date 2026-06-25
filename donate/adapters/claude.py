"""Claude Code session discovery."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from donate.adapters.base import GenericJsonlAdapter, is_redacted_artifact, safe_project_name_from_path, session_label


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
        info["session_label"] = session_label(info["project"], str(info.get("conversation_fingerprint") or ""), path)
        info["source_format"] = "claude-code-jsonl"
        info["confidence"]["agent"] = "high"
        return info
