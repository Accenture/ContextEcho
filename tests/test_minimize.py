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


if __name__ == "__main__":
    unittest.main()
