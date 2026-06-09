from __future__ import annotations

import json
import importlib.util
import tempfile
import unittest
from pathlib import Path

from donate.redact import redact_file


class RedactTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("presidio_analyzer"), "presidio_analyzer not installed")
    def test_redact_preserves_jsonl_when_redacting_nested_url_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "session.jsonl"
            dst = root / "session.redacted.jsonl"
            nested = json.dumps({"links": [{"title": "Example", "url": "https://example.com/private?q=1"}]})
            src.write_text(json.dumps({"type": "assistant", "message": {"content": nested}}) + "\n")

            redact_file(src, dst, scrub_terms=set(), progress=False)

            rows = [json.loads(line) for line in dst.read_text().splitlines()]
        self.assertEqual(rows[0]["type"], "assistant")
        self.assertNotIn("https://example.com", rows[0]["message"]["content"])


if __name__ == "__main__":
    unittest.main()
