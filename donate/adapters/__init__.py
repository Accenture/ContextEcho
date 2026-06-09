"""Agent-specific session discovery adapters."""

from __future__ import annotations

from donate.adapters.base import GenericJsonlAdapter, SessionAdapter
from donate.adapters.claude import ClaudeCodeAdapter
from donate.adapters.codex import CodexCliAdapter


ADAPTERS: list[SessionAdapter] = [
    ClaudeCodeAdapter(),
    CodexCliAdapter(),
]


__all__ = [
    "ADAPTERS",
    "ClaudeCodeAdapter",
    "CodexCliAdapter",
    "GenericJsonlAdapter",
    "SessionAdapter",
]
