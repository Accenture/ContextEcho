from __future__ import annotations

import json
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from donate.redact import redact_file, redact_file_with_progress


HAS_LOCAL_PRESIDIO = (
    importlib.util.find_spec("presidio_analyzer") is not None
    and importlib.util.find_spec("en_core_web_lg") is not None
)


class RedactTests(unittest.TestCase):
    @unittest.skipUnless(HAS_LOCAL_PRESIDIO, "local Presidio spaCy model not installed")
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

    def test_redact_reports_record_progress(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "session.jsonl"
            dst = root / "session.redacted.jsonl"
            src.write_text(
                "\n".join([
                    json.dumps({"type": "user", "message": "hello"}),
                    json.dumps({"type": "assistant", "message": "hi"}),
                    json.dumps({"type": "user", "message": "bye"}),
                ]) + "\n",
                encoding="utf-8",
            )
            events = []

            with mock.patch("donate.redact.build_analyzer", return_value=object()):
                with mock.patch("donate.redact.redact_text", side_effect=lambda text, *_args, **_kwargs: text):
                    redact_file_with_progress(
                        src,
                        dst,
                        scrub_terms=set(),
                        progress=False,
                        progress_callback=lambda current, total: events.append((current, total)),
                    )

        self.assertEqual(events, [(1, 3), (2, 3), (3, 3)])


if __name__ == "__main__":
    unittest.main()
