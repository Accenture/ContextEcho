"""Codex CLI session discovery."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

from donate.adapters.base import GenericJsonlAdapter, first_path_hint, is_redacted_artifact, iter_jsonl, session_label


UUID_RE = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")


def resume_session_id_from_path(path: Path) -> str:
    matches = list(UUID_RE.finditer(path.stem))
    return matches[-1].group(0).lower() if matches else ""


class CodexCliAdapter(GenericJsonlAdapter):
    agent = "Codex CLI"
    roots = [Path.home() / ".codex" / "sessions"]

    def discover_paths(self) -> Iterable[Path]:
        for root in self.roots:
            if root.exists():
                yield from (
                    p for p in sorted(root.rglob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
                    if not is_redacted_artifact(p)
                )

    def can_inspect_path(self, path: Path) -> bool:
        if is_redacted_artifact(path):
            return False
        path_s = str(path.expanduser())
        return "/.codex/sessions/" in path_s or any(root in path.parents for root in self.roots)

    def inspect(self, path: Path) -> dict:
        info = super().inspect(path)
        info["agent"] = self.agent
        info["source_format"] = "codex-cli-jsonl"
        resume_id = resume_session_id_from_path(path)
        if resume_id:
            info["resume_session_id"] = resume_id
            info["resume_command"] = f"codex resume {resume_id}"
            info["session_label"] = session_label(
                str(info.get("project") or ""),
                str(info.get("conversation_fingerprint") or ""),
                path,
                resume_id,
            )
        for _, obj in iter_jsonl(path):
            resume_dir = first_path_hint(obj)
            if resume_dir:
                info["resume_dir"] = resume_dir
                break
        # Codex emits explicit top-level `compacted` records. Generic "compact"
        # text in messages/summaries is too noisy to count.
        info["compactions"] = sum(
            1 for _, obj in iter_jsonl(path)
            if isinstance(obj, dict) and obj.get("type") == "compacted"
        )
        info["confidence"]["agent"] = "high"
        info["confidence"]["compactions"] = "high"
        return info
