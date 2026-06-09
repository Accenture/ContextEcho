import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from experiments.e18_session_validation.run import load_transcript_lines, position_plan


class SessionValidationTests(unittest.TestCase):
    def test_load_transcript_lines_reads_codex_response_items(self):
        with TemporaryDirectory() as td:
            path = Path(td) / "codex.redacted.jsonl"
            rows = [
                {"type": "session_meta", "payload": {"cwd": "/tmp/project"}},
                {
                    "type": "response_item",
                    "payload": {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "Please fix the parser."}],
                    },
                },
                {
                    "type": "response_item",
                    "payload": {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "I will update it."}],
                    },
                },
                {
                    "type": "response_item",
                    "payload": {
                        "type": "message",
                        "role": "user",
                        "content": "Run the test.",
                    },
                },
            ]
            path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

            parsed = load_transcript_lines(path)

        self.assertEqual([r["role"] for r in parsed], ["user", "assistant", "user"])
        self.assertEqual(parsed[0]["content"], "Please fix the parser.")
        self.assertEqual(parsed[1]["content"], "I will update it.")
        self.assertEqual(parsed[2]["content"], "Run the test.")
        self.assertEqual(len(position_plan(len(parsed), 3)), 3)


if __name__ == "__main__":
    unittest.main()
