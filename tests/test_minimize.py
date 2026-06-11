from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from donate.minimize import minimize_file


class MinimizeTests(unittest.TestCase):
    def test_minimize_masks_sensitive_user_text_and_keeps_task_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "session.redacted.jsonl"
            dst = root / "session.min.jsonl"
            src.write_text(
                "\n".join([
                    json.dumps({"type": "user", "message": "I feel very stressed about this deadline. Please fix parser.py."}),
                    json.dumps({"type": "assistant", "message": "I can help break it into steps."}),
                ]) + "\n",
                encoding="utf-8",
            )

            stats = minimize_file(src, dst)
            rows = [json.loads(line) for line in dst.read_text().splitlines()]

        self.assertEqual(stats["user_turns_minimized"], 1)
        self.assertIn("<USER_PRIVATE_FEELING_REDACTED>", rows[0]["message"])
        self.assertIn("Please fix parser.py.", rows[0]["message"])
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
        self.assertNotIn("user_strings_minimized", stats)
        self.assertEqual(rows[0]["payload"]["type"], "message")
        self.assertEqual(rows[0]["payload"]["role"], "user")
        self.assertEqual(rows[0]["payload"]["content"][0]["type"], "input_text")
        self.assertEqual(rows[0]["payload"]["content"][0]["text"], "Please fix this parser bug.")
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
        self.assertIn("<USER_PRIVATE_FEELING_REDACTED>", row["payload"]["message"])
        self.assertIn("this broken test", row["payload"]["message"])
        self.assertEqual(row["payload"]["images"], [])

    def test_minimize_masks_confidential_and_toxic_spans_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "session.redacted.jsonl"
            dst = root / "session.min.jsonl"
            src.write_text(
                json.dumps({
                    "type": "user",
                    "message": "This is NDA client-confidential work. The flaky test is in auth_flow.py. This is stupid.",
                }) + "\n",
                encoding="utf-8",
            )

            stats = minimize_file(src, dst)
            row = json.loads(dst.read_text().splitlines()[0])

        self.assertEqual(stats["user_turns_minimized"], 1)
        self.assertIn("<USER_CONFIDENTIAL_DETAIL_REDACTED>", row["message"])
        self.assertIn("<USER_TOXIC_LANGUAGE_REDACTED>", row["message"])
        self.assertIn("The flaky test is in auth_flow.py.", row["message"])

    def test_minimize_masks_identity_disclosure_span_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "session.redacted.jsonl"
            dst = root / "session.min.jsonl"
            src.write_text(
                json.dumps({
                    "type": "user",
                    "message": "I am 37 and debugging the CLI upload path.",
                }) + "\n",
                encoding="utf-8",
            )

            stats = minimize_file(src, dst)
            row = json.loads(dst.read_text().splitlines()[0])

        self.assertEqual(stats["user_turns_minimized"], 1)
        self.assertIn("<USER_IDENTITY_DISCLOSURE_REDACTED>", row["message"])
        self.assertIn("debugging the CLI upload path.", row["message"])


if __name__ == "__main__":
    unittest.main()
