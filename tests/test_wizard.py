from __future__ import annotations

import unittest
import contextlib
import io
from pathlib import Path
from unittest import mock

from donate.__main__ import choose_session, compact_label, donation_output_dir, format_turns, quality_tag, redacted_output_name, safe_slug


class WizardTests(unittest.TestCase):
    def test_safe_slug_removes_path_unfriendly_characters(self) -> None:
        self.assertEqual(safe_slug("Codex CLI / Project Name"), "Codex-CLI-Project-Name")
        self.assertEqual(safe_slug(""), "session")

    def test_compact_table_helpers(self) -> None:
        self.assertEqual(compact_label("abcdef", 4), "abc…")
        self.assertEqual(format_turns(59482), "59.5k")
        self.assertEqual(format_turns(480), "480")
        self.assertEqual(quality_tag({"turns": 1000, "compactions": 1}), "best")
        self.assertEqual(quality_tag({"turns": 1000, "compactions": 0}), "long")
        self.assertEqual(quality_tag({"turns": 999, "compactions": 5}), "short")

    def test_default_output_dir_uses_downloads(self) -> None:
        with mock.patch("pathlib.Path.home", return_value=Path("/home/tester")):
            out = donation_output_dir({"agent": "Codex CLI", "project": "demo/project"})

        self.assertEqual(out.parts[:4], ("/", "home", "tester", "Downloads"))
        self.assertEqual(out.parts[4], "ContextEcho_donations")
        self.assertIn("codex-cli", out.name)
        self.assertIn("demo-project", out.name)

    def test_redacted_output_name_does_not_double_redacted(self) -> None:
        self.assertEqual(
            redacted_output_name(Path("agent.redacted.jsonl")),
            "agent.redacted.jsonl",
        )

    def test_choose_session_accepts_numbers_beyond_first_page(self) -> None:
        sessions = [
            {"path": f"/tmp/session-{i}.jsonl", "agent": "Codex CLI", "turns": i, "compactions": 0}
            for i in range(1, 21)
        ]
        with mock.patch("donate.__main__.ask", return_value="20"):
            with contextlib.redirect_stdout(io.StringIO()):
                chosen, path = choose_session(sessions)

        self.assertEqual(chosen, sessions[19])
        self.assertEqual(path, Path("/tmp/session-20.jsonl"))

    def test_choose_session_more_advances_page(self) -> None:
        sessions = [
            {"path": f"/tmp/session-{i}.jsonl", "agent": "Codex CLI", "turns": i, "compactions": 0}
            for i in range(1, 21)
        ]
        with mock.patch("donate.__main__.ask", side_effect=["more", "16"]):
            with contextlib.redirect_stdout(io.StringIO()):
                chosen, path = choose_session(sessions)

        self.assertEqual(chosen, sessions[15])
        self.assertEqual(path, Path("/tmp/session-16.jsonl"))


if __name__ == "__main__":
    unittest.main()
