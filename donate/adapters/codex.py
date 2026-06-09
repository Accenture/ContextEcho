"""Codex CLI session discovery."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from donate.adapters.base import GenericJsonlAdapter, is_redacted_artifact


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
        info["confidence"]["agent"] = "high"
        return info
