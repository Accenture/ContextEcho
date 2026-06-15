"""ContextEcho donation — multi-agent session discovery + inspection.

Finds coding-agent session logs on the contributor's machine and auto-extracts
the metadata we care about (agent, model, turns, compactions), so the
contributor types as little as possible.

Supported discovery adapters:
    * Claude Code   ~/.claude/projects/**/*.jsonl
    * Codex CLI     ~/.codex/sessions/**/*.jsonl
    * Generic JSONL for manually supplied paths

Usage:
    python -m donate.discover            # list discovered sessions
    python -m donate.discover --json     # machine-readable
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from donate.adapters import ADAPTERS, GenericJsonlAdapter
from donate.adapters.base import guess_org as guess_org
from donate.adapters.base import safe_project_name_from_path as project_hint

MIN_RESEARCH_TURNS = 20


def adapter_for_path(path: Path):
    """Pick the best adapter for a manually supplied session path."""
    expanded = path.expanduser()
    for adapter in ADAPTERS:
        try:
            if adapter.can_inspect_path(expanded):
                return adapter
        except Exception:
            continue
    return GenericJsonlAdapter()


def inspect_session(path: Path) -> dict:
    """Read a JSONL session and auto-extract metadata. Best-effort."""
    return adapter_for_path(path).inspect(path.expanduser())


def _progress(msg: str, enabled: bool) -> None:
    if not enabled:
        return
    print(f"\r[discover] {msg}", end="", flush=True)


def is_research_candidate(info: dict, min_turns: int = MIN_RESEARCH_TURNS) -> bool:
    """Return whether a discovered session has enough signal to request donation."""
    turns = int(info.get("turns") or 0)
    compactions = int(info.get("compactions") or 0)
    return turns >= min_turns or compactions >= 1


def discover_iter(max_per_agent: int | None = 50):
    """Yield discovery progress events and a final result event."""
    found = []
    seen: set[Path] = set()
    total_inspected = 0
    for adapter in ADAPTERS:
        inspected = 0
        yield {"event": "adapter_start", "agent": adapter.agent, "inspected": total_inspected, "found": len(found)}
        for jsonl in adapter.discover_paths():
            if max_per_agent is not None and inspected >= max_per_agent:
                break
            resolved = jsonl.expanduser()
            if resolved in seen:
                continue
            seen.add(resolved)
            inspected += 1
            total_inspected += 1
            yield {
                "event": "inspect",
                "agent": adapter.agent,
                "adapter_inspected": inspected,
                "adapter_limit": max_per_agent,
                "inspected": total_inspected,
                "found": len(found),
                "path": str(resolved),
            }
            info = adapter.inspect(resolved)
            # Skip sessions with too little benchmark signal. Compactions are
            # accepted because they indicate long-context behavior even when
            # the visible user-turn count is modest.
            if not is_research_candidate(info):
                continue
            found.append(info)
        yield {
            "event": "adapter_done",
            "agent": adapter.agent,
            "adapter_inspected": inspected,
            "inspected": total_inspected,
            "found": len(found),
        }
    found.sort(key=lambda x: (-int(x.get("turns") or 0), x.get("agent", ""), x.get("path", "")))
    yield {"event": "done", "inspected": total_inspected, "found": len(found), "sessions": found}


def discover(max_per_agent: int | None = 50, progress: bool = False) -> list[dict]:
    sessions: list[dict] = []
    inspected = 0
    found = 0
    for event in discover_iter(max_per_agent=max_per_agent):
        kind = event.get("event")
        if kind == "adapter_start":
            _progress(f"scanning {event['agent']}...", progress)
        elif kind == "inspect":
            inspected = int(event.get("inspected") or 0)
            found = int(event.get("found") or 0)
            limit = event.get("adapter_limit")
            limit_txt = "" if limit is None else f"/{limit}"
            _progress(
                f"{event['agent']}: inspecting {event['adapter_inspected']}{limit_txt} candidates, found {found} usable",
                progress,
            )
        elif kind == "adapter_done":
            _progress(
                f"{event['agent']}: inspected {event['adapter_inspected']}, found {event['found']} usable total",
                progress,
            )
        elif kind == "done":
            inspected = int(event.get("inspected") or 0)
            found = int(event.get("found") or 0)
            sessions = list(event.get("sessions") or [])
    if progress:
        print(f"\r[discover] done: inspected {inspected} candidates, found {found} usable session(s).")
    return sessions


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description="Discover coding-agent sessions on this machine.")
    p.add_argument("--json", action="store_true", help="machine-readable output")
    p.add_argument("--max-per-agent", type=int, default=50,
                   help="inspect at most this many recent logs per adapter (default: 50)")
    p.add_argument("--all", action="store_true", help="inspect every discovered log")
    args = p.parse_args(argv)

    sessions = discover(max_per_agent=None if args.all else args.max_per_agent, progress=not args.json)
    if args.json:
        print(json.dumps(sessions, indent=2))
        return 0

    if not sessions:
        looked = ", ".join(root.as_posix() for adapter in ADAPTERS for root in adapter.roots)
        print(f"[discover] No sessions found (looked in {looked}).")
        print("[discover] Point the tool at a file directly: python -m donate.redact <file>")
        return 0

    print(f"[discover] Found {len(sessions)} session(s):\n")
    for i, s in enumerate(sessions, 1):
        print(f"  [{i}] {s['project']}  ({s['modified']})")
        print(
            f"      {s['agent']} · {s['model']} · {s['turns']:,} user turns · "
            f"{s.get('records', '?')} records · {s['compactions']} context compactions"
        )
        print(f"      {s['path']}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
