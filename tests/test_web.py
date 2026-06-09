import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from donate.web import (
    _parse_donated_sessions,
    already_submitted,
    annotate_donated,
    artifact_key,
    load_donated_artifact_keys,
    parse_submit_output,
    save_donation_record,
    session_key,
    write_receipt,
)


class WebTests(unittest.TestCase):
    def test_parse_redacted_donor_sessions_from_readme(self):
        self.assertEqual(_parse_donated_sessions("3 redacted donor sessions"), 3)
        self.assertEqual(_parse_donated_sessions("1,234 donated sessions"), 1234)
        self.assertIsNone(_parse_donated_sessions("no donation count here"))

    def test_annotate_donated_marks_known_source_key(self):
        path = "/tmp/example-session.jsonl"
        with mock.patch("donate.web.load_donated_keys", return_value={session_key(path)}):
            rows = annotate_donated([{"path": path}, {"path": "/tmp/other.jsonl"}])
        self.assertTrue(rows[0]["donated"])
        self.assertFalse(rows[1]["donated"])

    def test_save_donation_record_tracks_artifact_and_blocks_duplicates(self):
        with TemporaryDirectory() as td:
            registry = Path(td) / ".donated_sessions.json"
            source = Path(td) / "source.jsonl"
            artifact = Path(td) / "session.redacted.jsonl"
            source.write_text('{"type":"user"}\n')
            artifact.write_text('{"type":"user","message":"<PERSON>"}\n')

            with mock.patch("donate.web.DONATION_ROOT", Path(td)), mock.patch("donate.web.DONATION_REGISTRY", registry):
                save_donation_record(
                    source_path=source,
                    artifact_path=artifact,
                    output="[submit] submission  : pending/submission-abc12345/",
                )
                self.assertIn(artifact_key(artifact), load_donated_artifact_keys())
                self.assertTrue(already_submitted(source, artifact))
                self.assertTrue(already_submitted("", artifact))

    def test_write_receipt_records_submission_details(self):
        with TemporaryDirectory() as td:
            root = Path(td)
            session = root / "session.redacted.jsonl"
            manifest = root / "session.manifest.json"
            session.write_text('{"type":"user"}\n')
            manifest.write_text(
                '{"contributor":"donor","credit_name":"donor","contributor_email":"d@example.com",'
                '"agent":"Codex CLI","model":"gpt-5","turns":"123","compactions":"1"}'
            )
            output = "\n".join([
                "[submit] target repo : contextecho2026/persona-drift-staging (private)",
                "[submit] submission  : pending/submission-abc12345/",
                "[submit]   session.redacted.jsonl -> pending/submission-abc12345/session.redacted.jsonl",
                "[submit] https://huggingface.co/datasets/x/y/discussions/1",
            ])
            receipt_path, receipt = write_receipt(session, "/tmp/source.jsonl", output)
            self.assertTrue(receipt_path.exists())
            self.assertEqual(receipt["submission"], "pending/submission-abc12345/")
            self.assertEqual(receipt["contributor_email"], "d@example.com")
            self.assertIn("pending/submission-abc12345/", receipt_path.read_text())

    def test_parse_submit_output(self):
        parsed = parse_submit_output(
            "[submit] target repo : repo (private)\n"
            "[submit] submission  : pending/submission-abc12345/\n"
            "[submit]   a -> pending/submission-abc12345/a\n"
            "[submit] https://example.com/pr\n"
        )
        self.assertEqual(parsed["submission"], "pending/submission-abc12345/")
        self.assertEqual(parsed["url"], "https://example.com/pr")
        self.assertEqual(parsed["uploads"][0]["source"], "a")


if __name__ == "__main__":
    unittest.main()
