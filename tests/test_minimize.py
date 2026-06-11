from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from donate.minimize import minimize_file


class MinimizeTests(unittest.TestCase):
    def test_minimize_masks_user_text_and_keeps_assistant(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "session.redacted.jsonl"
            dst = root / "session.min.jsonl"
            src.write_text(
                "\n".join([
                    json.dumps({"type": "user", "message": "I feel very stressed about this deadline."}),
                    json.dumps({"type": "assistant", "message": "I can help break it into steps."}),
                ]) + "\n",
                encoding="utf-8",
            )

            stats = minimize_file(src, dst)
            rows = [json.loads(line) for line in dst.read_text().splitlines()]

        self.assertEqual(stats["user_turns_minimized"], 1)
        self.assertEqual(rows[0]["message"]["content"], "<USER_PERSONAL_TEXT_REDACTED>")
        self.assertIn("personal", rows[0]["message"]["tags"])
        self.assertEqual(rows[0]["privacy_tier"], "user_minimized")
        self.assertEqual(rows[1]["message"], "I can help break it into steps.")

    def test_minimize_masks_codex_response_item_user_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "session.redacted.jsonl"
            dst = root / "session.min.jsonl"
            src.write_text(
                "\n".join([
                    json.dumps({
                        "type": "response_item",
                        "payload": {
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_text", "text": "Please fix this parser bug."}],
                        },
                    }),
                    json.dumps({
                        "type": "response_item",
                        "payload": {
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": "I will inspect the parser."}],
                        },
                    }),
                ]) + "\n",
                encoding="utf-8",
            )

            stats = minimize_file(src, dst)
            rows = [json.loads(line) for line in dst.read_text().splitlines()]

        self.assertEqual(stats["user_turns_minimized"], 1)
        self.assertEqual(rows[0]["payload"]["type"], "message")
        self.assertEqual(rows[0]["payload"]["role"], "user")
        self.assertEqual(rows[0]["payload"]["content"][0]["type"], "input_text")
        self.assertEqual(rows[0]["payload"]["content"][0]["text"]["content"], "<USER_TEXT_REDACTED>")
        self.assertEqual(rows[0]["privacy_tier"], "user_minimized")
        self.assertEqual(rows[1]["payload"]["content"][0]["text"], "I will inspect the parser.")

    def test_minimize_masks_codex_event_user_message(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "session.redacted.jsonl"
            dst = root / "session.min.jsonl"
            src.write_text(
                json.dumps({
                    "type": "event_msg",
                    "payload": {
                        "type": "user_message",
                        "message": "I feel stressed about this broken test.",
                        "images": [],
                    },
                }) + "\n",
                encoding="utf-8",
            )

            stats = minimize_file(src, dst)
            row = json.loads(dst.read_text().splitlines()[0])

        self.assertEqual(stats["user_turns_minimized"], 1)
        self.assertEqual(row["payload"]["type"], "user_message")
        self.assertEqual(row["payload"]["message"]["content"], "<USER_PERSONAL_TEXT_REDACTED>")
        self.assertIn("personal", row["payload"]["message"]["tags"])
        self.assertEqual(row["payload"]["images"], [])


if __name__ == "__main__":
    unittest.main()
