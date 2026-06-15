from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from donate.verify import _write_detect_secrets_candidate_file, verify_session


class VerifyTests(unittest.TestCase):
    def test_private_key_prose_is_not_blocking(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "session.redacted.jsonl"
            path.write_text(
                '{"text":"Check for BEGIN RSA PRIVATE KEY and BEGIN OPENSSH PRIVATE KEY patterns."}\n',
                encoding="utf-8",
            )

            report = verify_session(path)

        self.assertNotIn("detect_secrets", report["blocking"])

    def test_real_private_key_delimiter_is_blocking(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "session.redacted.jsonl"
            path.write_text(
                '{"text":"-----BEGIN PRIVATE KEY----- abc -----END PRIVATE KEY-----"}\n',
                encoding="utf-8",
            )

            report = verify_session(path)

        self.assertIn("detect_secrets", report["blocking"])

    def test_detect_secrets_report_exposes_type_not_secret_value(self) -> None:
        secret = "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "session.redacted.jsonl"
            path.write_text(f'aws_secret_access_key = "{secret}"\n', encoding="utf-8")

            report = verify_session(path)

        self.assertIn("detect_secrets", report["blocking"])
        self.assertIn("AWS Access Key", report["blocking"]["detect_secrets"])
        self.assertNotIn(secret, str(report))

    def test_detect_secrets_prefilter_keeps_only_suspicious_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "session.redacted.jsonl"
            path.write_text(
                "\n".join([
                    '{"text":"ordinary coding discussion"}',
                    '{"text":"still ordinary"}',
                    '{"text":"discuss token handling, secret scanning, and password reset UI"}',
                    'aws_secret_access_key = "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY"',
                    '{"text":"another normal line"}',
                ])
                + "\n",
                encoding="utf-8",
            )

            candidate, line_map = _write_detect_secrets_candidate_file(path)
            self.addCleanup(lambda: candidate and candidate.unlink(missing_ok=True))

        self.assertIsNotNone(candidate)
        self.assertEqual(line_map, {1: 4})
        self.assertIn("aws_secret_access_key", candidate.read_text(encoding="utf-8"))
        self.assertNotIn("ordinary coding discussion", candidate.read_text(encoding="utf-8"))
        self.assertNotIn("secret scanning", candidate.read_text(encoding="utf-8"))

    def test_verify_session_reads_source_once_for_large_logs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "session.redacted.jsonl"
            path.write_text(
                '{"text":"ordinary redacted coding discussion"}\n' * 500
                + 'aws_secret_access_key = "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY"\n'
                + '{"text":"more ordinary redacted coding discussion"}\n' * 500,
                encoding="utf-8",
            )
            source_opens = []
            original_open = Path.open

            def counted_open(self, *args, **kwargs):
                if self == path:
                    source_opens.append(args[0] if args else kwargs.get("mode", "r"))
                return original_open(self, *args, **kwargs)

            with (
                mock.patch("pathlib.Path.open", counted_open),
                mock.patch(
                    "donate.verify._detect_secret_findings_in_candidate",
                    return_value=[{"type": "AWS Access Key"}],
                ),
            ):
                report = verify_session(path)

        self.assertIn("detect_secrets", report["blocking"])
        self.assertEqual(source_opens, ["r"])

    def test_long_record_windowing_preserves_blocking_detections(self) -> None:
        padding = " ordinary redacted coding discussion " * 2500
        long_record = (
            padding
            + "/Users/alice/project "
            + padding
            + "alice@example.com "
            + padding
            + "hf_gRdux2IUk42ASAx0GGFKHighABP1Fe1Ep0V2fdTBJ96Y43F4JVH9XD1hhNq3 "
            + padding
            + "-----BEGIN PRIVATE KEY----- abc -----END PRIVATE KEY-----"
            + padding
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "session.redacted.jsonl"
            path.write_text(long_record + "\n", encoding="utf-8")

            report = verify_session(path)

        self.assertIn("home_path", report["blocking"])
        self.assertIn("/Users/alice", report["blocking"]["home_path"])
        self.assertIn("email", report["blocking"])
        self.assertIn("alice@example.com", report["blocking"]["email"])
        self.assertIn("api_key", report["blocking"])
        self.assertIn("detect_secrets", report["blocking"])


if __name__ == "__main__":
    unittest.main()
