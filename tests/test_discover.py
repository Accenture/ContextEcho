from __future__ import annotations

import json
import contextlib
import io
import tempfile
import unittest
import os
import time
from pathlib import Path

from donate.discover import MIN_RESEARCH_TURNS, discover, discover_iter, inspect_session, is_research_candidate
from donate.adapters.base import date_from_timestamp, is_redacted_artifact


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


class DiscoverTests(unittest.TestCase):
    def test_timestamp_dates_are_displayed_in_local_timezone(self) -> None:
        old_tz = os.environ.get("TZ")
        try:
            os.environ["TZ"] = "America/Los_Angeles"
            if hasattr(time, "tzset"):
                time.tzset()
            self.assertEqual(date_from_timestamp("2026-06-23T01:25:00Z"), "2026-06-22")
        finally:
            if old_tz is None:
                os.environ.pop("TZ", None)
            else:
                os.environ["TZ"] = old_tz
            if hasattr(time, "tzset"):
                time.tzset()

    def test_redacted_artifact_detection(self) -> None:
        self.assertTrue(is_redacted_artifact(Path("session.redacted.jsonl")))
        self.assertTrue(is_redacted_artifact(Path("/tmp/ContextEcho_donations/run/session.jsonl")))
        self.assertFalse(is_redacted_artifact(Path("session.jsonl")))

    def test_codex_manual_path_is_classified_and_inspected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".codex" / "sessions" / "rollout.jsonl"
            rows = [
                {
                    "timestamp": "2026-01-02T03:04:05.000Z",
                    "type": "session_meta",
                    "payload": {"cwd": "/Users/alice/Documents/work/agent-project"},
                },
                {
                    "timestamp": "2026-01-03T03:04:05.000Z",
                    "type": "event_msg",
                    "payload": {"type": "user_message", "message": "please help with the bug"},
                },
                {
                    "timestamp": "2026-01-03T03:04:06.000Z",
                    "type": "turn_context",
                    "payload": {"model": "gpt-5", "summary": "ordinary context summary"},
                },
                {
                    "timestamp": "2026-01-04T03:04:05.000Z",
                    "type": "response_item",
                    "payload": {"type": "message", "role": "assistant", "content": []},
                },
            ]
            write_jsonl(path, rows)

            info = inspect_session(path)

        self.assertEqual(info["agent"], "Codex CLI")
        self.assertEqual(info["source_format"], "codex-cli-jsonl")
        self.assertEqual(info["model"], "gpt-5")
        self.assertEqual(info["org"], "OpenAI")
        self.assertEqual(info["records"], 4)
        self.assertEqual(info["turns"], 1)
        self.assertEqual(info["compactions"], 0)
        self.assertEqual(info["started"], "2026-01-01")
        self.assertEqual(info["last_active"], "2026-01-03")
        self.assertEqual(info["modified"], "2026-01-03")
        self.assertEqual(info["project"], "work-agent-project")
        self.assertRegex(info["session_label"], r"^work-agent-project · [0-9a-f]{4}$")
        self.assertTrue(info["conversation_fingerprint"].startswith("conv-"))
        self.assertEqual(info["fingerprint_version"], "structure-v1")

    def test_conversation_fingerprint_is_stable_when_session_is_appended(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".codex" / "sessions" / "rollout.jsonl"
            rows = [
                {
                    "timestamp": "2026-01-02T03:04:05.000Z",
                    "type": "session_meta",
                    "payload": {"cwd": "/Users/alice/Documents/work/agent-project"},
                }
            ] + [
                {
                    "timestamp": f"2026-01-03T03:{i:02d}:05.000Z",
                    "type": "event_msg",
                    "payload": {"type": "user_message", "message": f"please help with bug {i}"},
                }
                for i in range(50)
            ]
            write_jsonl(path, rows)
            first = inspect_session(path)["conversation_fingerprint"]
            write_jsonl(path, rows + [
                {
                    "timestamp": "2026-01-04T03:04:05.000Z",
                    "type": "event_msg",
                    "payload": {"type": "user_message", "message": "now add more tests"},
                },
            ])
            second = inspect_session(path)["conversation_fingerprint"]

        self.assertEqual(first, second)

    def test_codex_does_not_count_generic_compact_text_as_compaction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".codex" / "sessions" / "rollout.jsonl"
            rows = [
                {"type": "session_meta", "payload": {"cwd": "/Users/alice/Documents/work/agent-project"}},
                {"type": "turn_context", "payload": {"model": "gpt-5", "summary": "compact prior context"}},
                {"type": "response_item", "payload": {"type": "context_compaction", "content": []}},
                {"isCompactSummary": True, "model": "gpt-5"},
            ]
            write_jsonl(path, rows)

            info = inspect_session(path)

        self.assertEqual(info["agent"], "Codex CLI")
        self.assertEqual(info["records"], 4)
        self.assertEqual(info["turns"], 0)
        self.assertEqual(info["compactions"], 0)
        self.assertEqual(info["confidence"]["compactions"], "high")

    def test_codex_counts_only_explicit_compacted_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".codex" / "sessions" / "rollout.jsonl"
            rows = [
                {"type": "session_meta", "payload": {"cwd": "/Users/alice/Documents/work/agent-project"}},
                {"type": "event_msg", "payload": {"type": "user_message", "message": "fix failing tests"}},
                {"type": "compacted", "payload": {"message": {}, "replacement_history": []}},
                {"type": "response_item", "payload": {"type": "message", "role": "assistant", "content": []}},
            ]
            write_jsonl(path, rows)

            info = inspect_session(path)

        self.assertEqual(info["agent"], "Codex CLI")
        self.assertEqual(info["records"], 4)
        self.assertEqual(info["turns"], 1)
        self.assertEqual(info["compactions"], 1)
        self.assertEqual(info["confidence"]["compactions"], "high")

    def test_claude_manual_path_is_classified_and_counts_explicit_compaction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = (
                Path(tmp)
                / ".claude"
                / "projects"
                / "-Users-alice-Documents-client-safe-repo"
                / "session.jsonl"
            )
            rows = [
                {"model": "claude-opus-4-7", "message": {"role": "user", "content": [{"type": "text", "text": "fix failing tests"}]}},
                {"model": "claude-opus-4-8", "isCompactSummary": True},
                {"type": "compact_file_reference"},
            ]
            write_jsonl(path, rows)

            info = inspect_session(path)

        self.assertEqual(info["agent"], "Claude Code")
        self.assertEqual(info["source_format"], "claude-code-jsonl")
        self.assertEqual(info["org"], "Anthropic")
        self.assertEqual(info["records"], 3)
        self.assertEqual(info["turns"], 1)
        self.assertEqual(info["compactions"], 1)
        self.assertEqual(info["project"], "client-safe-repo")
        self.assertRegex(info["session_label"], r"^client-safe-repo · [0-9a-f]{4}$")

    def test_same_folder_sessions_have_distinct_display_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / ".claude" / "projects" / "-Users-alice-Documents-client-safe-repo"
            path_a = folder / "session-a.jsonl"
            path_b = folder / "session-b.jsonl"
            write_jsonl(path_a, [{"message": {"role": "user", "content": [{"type": "text", "text": "fix tests"}]}}])
            write_jsonl(path_b, [{"message": {"role": "user", "content": [{"type": "text", "text": "build dashboard"}]}}])

            info_a = inspect_session(path_a)
            info_b = inspect_session(path_b)

        self.assertEqual(info_a["project"], info_b["project"])
        self.assertNotEqual(info_a["session_label"], info_b["session_label"])
        self.assertRegex(info_a["session_label"], r"^client-safe-repo · [0-9a-f]{4}$")
        self.assertRegex(info_b["session_label"], r"^client-safe-repo · [0-9a-f]{4}$")

    def test_claude_tool_result_user_records_are_not_turns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = (
                Path(tmp)
                / ".claude"
                / "projects"
                / "-Users-alice-Documents-client-safe-repo"
                / "session.jsonl"
            )
            rows = [
                {"type": "user", "message": {"role": "user", "content": [{"type": "tool_result", "content": "ok"}]}},
                {"type": "user", "message": {"role": "user", "content": [{"type": "text", "text": "now fix it"}]}},
                {"type": "assistant", "message": {"role": "assistant", "content": [{"type": "text", "text": "done"}]}},
            ]
            write_jsonl(path, rows)

            info = inspect_session(path)

        self.assertEqual(info["records"], 3)
        self.assertEqual(info["turns"], 1)

    def test_unknown_jsonl_falls_back_to_generic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "custom-agent.jsonl"
            write_jsonl(path, [{"model": "qwen3-32b"}, {"type": "context_compaction"}])

            info = inspect_session(path)

        self.assertEqual(info["agent"], "Unknown agent")
        self.assertEqual(info["source_format"], "generic-jsonl")
        self.assertEqual(info["org"], "Alibaba")
        self.assertEqual(info["records"], 2)
        self.assertEqual(info["turns"], 0)
        self.assertEqual(info["compactions"], 1)

    def test_research_candidate_requires_turns_or_compaction(self) -> None:
        self.assertFalse(is_research_candidate({"turns": MIN_RESEARCH_TURNS - 1, "compactions": 0, "records": 500}))
        self.assertTrue(is_research_candidate({"turns": MIN_RESEARCH_TURNS, "compactions": 0}))
        self.assertTrue(is_research_candidate({"turns": 1, "compactions": 1}))

    def test_discover_progress_can_be_disabled_for_json_callers(self) -> None:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            sessions = discover(max_per_agent=0, progress=False)

        self.assertEqual(sessions, [])
        self.assertEqual(buf.getvalue(), "")

    def test_discover_iter_emits_done_event(self) -> None:
        events = list(discover_iter(max_per_agent=0))
        self.assertTrue(events)
        self.assertEqual(events[-1]["event"], "done")
        self.assertEqual(events[-1]["sessions"], [])


if __name__ == "__main__":
    unittest.main()
