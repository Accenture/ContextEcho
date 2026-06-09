from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from donate import describe


class DescribeTests(unittest.TestCase):
    def test_minimal_prompts_auto_accepts_detected_agent_model_org(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            session = root / "session.redacted.jsonl"
            session.write_text("{}\n", encoding="utf-8")
            auto = root / "auto.json"
            auto.write_text(json.dumps({
                "agent": "Codex CLI",
                "model": "gpt-5",
                "org": "OpenAI",
                "turns": 42,
                "compactions": 0,
                "source_format": "codex-cli-jsonl",
            }), encoding="utf-8")

            answers = iter(["agentic-coding", "Python", "donor-handle"])
            with mock.patch("builtins.input", side_effect=lambda _prompt: next(answers)):
                with contextlib.redirect_stdout(io.StringIO()):
                    rc = describe.main([
                        "--session", str(session),
                        "--auto", str(auto),
                        "--minimal-prompts",
                    ])

            manifest = json.loads((root / "session.manifest.json").read_text())

        self.assertEqual(rc, 0)
        self.assertEqual(manifest["agent"], "Codex CLI")
        self.assertEqual(manifest["model"], "gpt-5")
        self.assertEqual(manifest["org"], "OpenAI")
        self.assertEqual(manifest["domain"], "agentic-coding")
        self.assertEqual(manifest["language"], "Python")
        self.assertEqual(manifest["contributor"], "donor-handle")
        self.assertEqual(manifest["source_format"], "codex-cli-jsonl")
        self.assertEqual(manifest["privacy_tier"], "full_redacted")
        self.assertIn("persona_drift_benchmarking", manifest["allowed_uses"])


if __name__ == "__main__":
    unittest.main()
