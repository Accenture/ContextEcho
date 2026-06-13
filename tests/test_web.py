import unittest
import errno
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from donate.verify import verify_session
from donate.web import (
    INDEX_HTML,
    _auto_repair_until_verified,
    _load_contributors_markdown,
    _parse_contributor_leaderboard,
    _parse_donated_sessions,
    already_submitted,
    annotate_donated,
    artifact_key,
    clear_donation_registry,
    create_server,
    friendly_submit_error,
    is_duplicate_submit_output,
    load_donated_artifact_keys,
    local_pending_summary,
    parse_submit_output,
    save_donation_record,
    session_key,
    write_receipt,
)


class WebTests(unittest.TestCase):
    def test_submit_step_previews_public_leaderboard_identity(self):
        self.assertIn("submitLeaderboardPreview", INDEX_HTML)
        self.assertIn("Leaderboard preview", INDEX_HTML)
        self.assertIn("Show me anonymously on the public leaderboard", INDEX_HTML)
        self.assertIn("Default is public credit", INDEX_HTML)
        self.assertNotIn("What becomes public after maintainer acceptance?", INDEX_HTML)

    def test_search_panel_can_run_cleanup_directly(self):
        self.assertIn("Redact and Verify Again", INDEX_HTML)
        self.assertIn("redact-primary", INDEX_HTML)
        self.assertIn("Running Redact and Verify again for the matched word", INDEX_HTML)
        self.assertIn("Redaction complete. The checked word is now found 0 times", INDEX_HTML)
        self.assertIn("Already redacted in this output", INDEX_HTML)

    def test_private_word_input_uses_redact_language(self):
        self.assertIn("Private words to redact", INDEX_HTML)
        self.assertIn("Use this only if a private word remains", INDEX_HTML)
        self.assertNotIn("Private words to remove on the next redaction run", INDEX_HTML)

    def test_auto_repair_removes_detect_secrets_value(self):
        with TemporaryDirectory() as td:
            path = Path(td) / "session.redacted.jsonl"
            secret = "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY"
            path.write_text(f'aws_secret_access_key = "{secret}"\n', encoding="utf-8")
            report = verify_session(path)

            repaired_report, stats, passes = _auto_repair_until_verified(path, report, {})
            repaired_text = path.read_text(encoding="utf-8")

        self.assertGreaterEqual(passes, 1)
        self.assertTrue(repaired_report["passed"])
        self.assertGreaterEqual(stats.get("scrub_term", 0), 1)
        self.assertNotIn(secret, repaired_text)

    def test_parse_redacted_donor_sessions_from_readme(self):
        self.assertEqual(_parse_donated_sessions("3 redacted donor sessions"), 3)
        self.assertEqual(_parse_donated_sessions("1,234 donated sessions"), 1234)
        self.assertIsNone(_parse_donated_sessions("no donation count here"))

    def test_parse_contributor_leaderboard_stops_before_session_ledger(self):
        rows = _parse_contributor_leaderboard(
            "\n".join([
                "| Rank | Contributor | Sessions | Turns | Agents | Models | Points |",
                "|:----:|-------------|:--------:|------:|--------|--------|:------:|",
                "| 🥇 | Founding donors | 3 | 18,380 | Claude Code | Opus 4.x | — |",
                "",
                "| ID | Agent / Harness | Model | Org | Domain | Language | Turns |",
                "|----|-----------------|-------|-----|--------|----------|------:|",
                "| S1 | Claude Code | Opus | Anthropic | coding | Python | 9,716 |",
            ])
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["contributor"], "Founding donors")
        self.assertEqual(rows[0]["sessions_num"], 3)
        self.assertEqual(rows[0]["turns_num"], 18380)

    def test_load_contributors_markdown_falls_back_to_github_when_packaged(self):
        remote_text = "| Rank | Contributor | Sessions | Turns | Agents | Models | Points |\n"
        with TemporaryDirectory() as td:
            missing = Path(td) / "CONTRIBUTORS.md"
            with mock.patch("donate.web._fetch_text", return_value=remote_text) as fetch_text:
                text = _load_contributors_markdown(missing)

        self.assertEqual(text, remote_text)
        fetch_text.assert_called_once_with(
            "https://raw.githubusercontent.com/Accenture/ContextEcho/main/CONTRIBUTORS.md"
        )

    def test_create_server_falls_back_when_port_is_busy(self):
        fake_server = mock.Mock()
        fake_server.server_address = ("127.0.0.1", 8767)
        busy = OSError(errno.EADDRINUSE, "address in use")
        with mock.patch("donate.web.ThreadingHTTPServer", side_effect=[busy, fake_server]) as server_cls:
            server, actual_port = create_server("127.0.0.1", 8766, attempts=3)

        self.assertIs(server, fake_server)
        self.assertEqual(actual_port, 8767)
        self.assertEqual(server_cls.call_args_list[0].args[0], ("127.0.0.1", 8766))
        self.assertEqual(server_cls.call_args_list[1].args[0], ("127.0.0.1", 8767))

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

    def test_clear_donation_registry_removes_local_duplicate_memory(self):
        with TemporaryDirectory() as td:
            registry = Path(td) / ".donated_sessions.json"
            source = Path(td) / "source.jsonl"
            source.write_text('{"type":"user"}\n')

            with mock.patch("donate.web.DONATION_ROOT", Path(td)), mock.patch("donate.web.DONATION_REGISTRY", registry):
                save_donation_record(source_path=source)
                self.assertTrue(already_submitted(source))
                self.assertTrue(clear_donation_registry())
                self.assertFalse(already_submitted(source))
                self.assertFalse(clear_donation_registry())

    def test_local_pending_summary_merges_full_identity(self):
        with TemporaryDirectory() as td:
            registry = Path(td) / ".donated_sessions.json"
            source1 = Path(td) / "source1.jsonl"
            source2 = Path(td) / "source2.jsonl"
            artifact1 = Path(td) / "session1.redacted.jsonl"
            artifact2 = Path(td) / "session2.redacted.jsonl"
            for path in [source1, source2, artifact1, artifact2]:
                path.write_text('{"type":"user"}\n')
            receipt = {
                "credit_name": "Xianzhong Ding",
                "contributor_email": "xding5@ucmerced.edu",
                "institute": "UC Merced",
                "turns": "18",
                "compactions": "0",
            }

            with mock.patch("donate.web.DONATION_ROOT", Path(td)), mock.patch("donate.web.DONATION_REGISTRY", registry):
                save_donation_record(source1, artifact1, "[submit] Submission ID: submission-a", receipt=receipt)
                save_donation_record(source2, artifact2, "[submit] Submission ID: submission-b", receipt=receipt)
                summary = local_pending_summary(receipt)

        self.assertEqual(summary["sessions"], 2)
        self.assertEqual(summary["points_low"], 4)
        self.assertEqual(summary["points_high"], 8)

    def test_write_receipt_records_submission_details(self):
        with TemporaryDirectory() as td:
            root = Path(td)
            session = root / "session.redacted.jsonl"
            manifest = root / "session.manifest.json"
            session.write_text('{"type":"user"}\n')
            manifest.write_text(
                '{"contributor":"donor","credit_name":"donor","contributor_email":"d@example.com",'
                '"public_anonymous":true,'
                '"agent":"Codex CLI","model":"gpt-5","turns":"42","records":"123","compactions":"1"}'
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
            self.assertTrue(receipt["public_anonymous"])
            self.assertEqual(receipt["turns"], "42")
            self.assertEqual(receipt["records"], "123")
            text = receipt_path.read_text()
            self.assertIn("pending/submission-abc12345/", text)
            self.assertIn("User turns: 42", text)
            self.assertIn("Public leaderboard: anonymous", text)
            self.assertIn("Records: 123", text)
            self.assertIn("Context compactions: 1", text)
            self.assertNotIn("persona-drift-staging", text)
            self.assertNotIn("huggingface.co", text)

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

    def test_parse_relay_submit_output(self):
        parsed = parse_submit_output(
            "[submit] upload mode  : relay\n"
            "[submit] relay       : http://127.0.0.1:8088\n"
            "[submit]   session.redacted.jsonl     -> session.redacted.jsonl\n"
            "[submit] Submitted for maintainer review.\n"
            "[submit] Submission ID: submission-abc12345\n"
        )
        self.assertEqual(parsed["submission"], "submission-abc12345")
        self.assertEqual(parsed["uploads"][0]["source"], "session.redacted.jsonl")

    def test_duplicate_relay_submit_output_is_detected(self):
        self.assertTrue(is_duplicate_submit_output(
            '[submit] relay upload failed: HTTP 409 {"detail":"duplicate redacted session artifact"}'
        ))
        self.assertFalse(is_duplicate_submit_output("[submit] relay upload failed: HTTP 500"))

    def test_friendly_submit_error_explains_missing_relay_or_token(self):
        msg = friendly_submit_error(
            "401 Client Error. Repository Not Found for url: "
            "https://huggingface.co/api/datasets/contextecho2026/persona-drift-staging"
        )
        self.assertIn("Upload is not configured for public donors yet", msg)
        self.assertIn("CONTEXTECHO_RELAY_URL", msg)


if __name__ == "__main__":
    unittest.main()
