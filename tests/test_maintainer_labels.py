import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.promote_donation import default_label as promote_default_label
from scripts.promote_donation import normalize_manifest
from scripts.promote_donation import sha256_file
from scripts.review_donation import default_label as review_default_label


class MaintainerLabelTests(unittest.TestCase):
    def test_default_labels_include_submission_id(self):
        manifest = {"credit_name": "Carlos"}
        submission = Path("hf_staging_download/pending/submission-abc12345")

        self.assertEqual(review_default_label(manifest, submission), "Carlos-submission-abc12345")
        self.assertEqual(promote_default_label(manifest, submission), "Carlos-submission-abc12345")

    def test_public_anonymous_default_labels_do_not_include_name(self):
        manifest = {"credit_name": "Carlos", "public_anonymous": True}
        submission = Path("hf_staging_download/pending/submission-abc12345")

        self.assertEqual(review_default_label(manifest, submission), "anonymous-donor-submission-abc12345")
        self.assertEqual(promote_default_label(manifest, submission), "anonymous-donor-submission-abc12345")

    def test_sha256_file_hashes_session_content(self):
        with TemporaryDirectory() as td:
            path = Path(td) / "session.redacted.jsonl"
            path.write_text("abc\n", encoding="utf-8")

            self.assertEqual(
                sha256_file(path),
                "edeaaff3f1774ad2888673770c6d64097e391bc362d7d6fb34982ddf0efd18cb",
            )

    def test_promote_normalizes_legacy_manifest_metadata(self):
        with TemporaryDirectory() as td:
            root = Path(td)
            session = root / "session.redacted.jsonl"
            session.write_text("{}\n", encoding="utf-8")
            submission = root / "pending" / "submission-abc12345"

            manifest = normalize_manifest(
                {
                    "session_id": "S?",
                    "domain": "agentic-coding",
                    "language": "unknown",
                    "records": "6088",
                    "turns": "449",
                    "compactions": "5",
                },
                submission,
                session,
            )

        self.assertEqual(manifest["session_id"], "submission-abc12345")
        self.assertEqual(manifest["language"], "mixed")
        self.assertEqual(manifest["records"], 6088)
        self.assertEqual(manifest["turns"], 449)
        self.assertEqual(manifest["compactions"], 5)
        self.assertEqual(manifest["donor_domain"], "agentic-coding")
        self.assertEqual(manifest["reviewed_domain"], "agentic-coding")
        self.assertEqual(manifest["reviewed_submission_id"], "submission-abc12345")
        self.assertIn("session_sha256", manifest)


if __name__ == "__main__":
    unittest.main()
