from __future__ import annotations

import json
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from donate.redact import (
    apply_scrub_terms_to_file,
    build_analyzer,
    redact_file,
    redact_file_with_progress,
    redact_text,
)


HAS_LOCAL_PRESIDIO = (
    importlib.util.find_spec("presidio_analyzer") is not None
    and importlib.util.find_spec("en_core_web_lg") is not None
)


class RedactTests(unittest.TestCase):
    def test_build_analyzer_uses_no_download_nlp_engine(self) -> None:
        with mock.patch("presidio_analyzer.AnalyzerEngine") as analyzer_cls:
            analyzer = mock.Mock()
            analyzer_cls.return_value = analyzer

            build_analyzer()

        kwargs = analyzer_cls.call_args.kwargs
        self.assertEqual(kwargs["supported_languages"], ["en"])
        self.assertEqual(kwargs["nlp_engine"].get_supported_languages(), ["en"])

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

    def test_fast_scrub_repair_expands_home_path_terms(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "session.redacted.jsonl"
            path.write_text(
                '{"text":"/Users/alice/project -Users-alice-Library alice"}\n',
                encoding="utf-8",
            )

            stats = apply_scrub_terms_to_file(path, path, {"/Users/alice"})
            text = path.read_text(encoding="utf-8")

        self.assertGreaterEqual(stats.get("scrub_term", 0), 3)
        self.assertNotIn("alice", text)
        self.assertNotIn("/Users/alice", text)
        self.assertNotIn("-Users-alice", text)

    def test_fast_scrub_repair_removes_private_key_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "session.redacted.jsonl"
            path.write_text(
                '{"text":"-----BEGIN PRIVATE KEY-----\\nabc123\\n-----END PRIVATE KEY-----"}\n',
                encoding="utf-8",
            )

            stats = apply_scrub_terms_to_file(path, path, set())
            text = path.read_text(encoding="utf-8")

        self.assertNotIn("BEGIN PRIVATE KEY", text)
        self.assertNotIn("abc123", text)
        self.assertGreaterEqual(stats.get("api_key", 0), 1)

    def test_redact_text_removes_basic_auth_credentials(self) -> None:
        class NoopAnonymizer:
            @property
            def text(self):
                return self._text

        class NoopEngine:
            def analyze(self, text, language, entities):
                return []

        with mock.patch("presidio_anonymizer.AnonymizerEngine") as anonymizer_cls:
            anonymizer = mock.Mock()
            anonymizer.anonymize.side_effect = lambda text, analyzer_results, operators: type("Result", (), {"text": text})()
            anonymizer_cls.return_value = anonymizer
            stats = {}
            text = (
                "curl https://user:pass@example.com/private "
                "Authorization: Basic dXNlcjpwYXNzd29yZA=="
            )

            redacted = redact_text(text, NoopEngine(), set(), stats)

        self.assertNotIn("user:pass", redacted)
        self.assertNotIn("dXNlcjpwYXNzd29yZA", redacted)
        self.assertGreaterEqual(stats.get("api_key", 0), 2)


if __name__ == "__main__":
    unittest.main()
