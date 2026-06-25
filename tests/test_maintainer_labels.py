import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.promote_donation import default_label as promote_default_label
from scripts.promote_donation import normalize_manifest
from scripts.promote_donation import sha256_file
from scripts.review_donation import default_label as review_default_label
from scripts.sync_approved_metadata_updates import apply_approved_updates


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

    def test_sync_approved_metadata_updates_patches_promoted_release(self):
        with TemporaryDirectory() as td:
            root = Path(td)
            donation_dir = root / "data" / "donations" / "Carlos-submission-abc12345"
            donation_dir.mkdir(parents=True)
            manifest_path = donation_dir / "manifest.json"
            manifest_path.write_text(
                '{"credit_name":"Carlos","contributor_email":"old@example.com","contributor_institute":"Old Lab","public_anonymous":false}\n',
                encoding="utf-8",
            )
            ledger = root / "data" / "donations" / "ledger.jsonl"
            ledger.write_text(
                '{"submission_id":"submission-abc12345","manifest_path":"data/donations/Carlos-submission-abc12345/manifest.json","credit_name":"Carlos","contributor_email":"old@example.com","institute":"Old Lab","public_anonymous":false,"decision":"ACCEPTABLE"}\n',
                encoding="utf-8",
            )

            result = apply_approved_updates(
                root,
                [
                    {
                        "request_id": "metadata-one",
                        "status": "approved",
                        "approved_utc": "2026-06-25T23:45:00+00:00",
                        "submission_id": "submission-abc12345",
                        "credit_name": "Anonymous donor",
                        "contributor_email": "new@example.com",
                        "contributor_institute": "New Lab",
                        "public_anonymous": True,
                    }
                ],
            )

            self.assertEqual(result["ledger"], 1)
            self.assertEqual(result["manifests"], 1)
            ledger_text = ledger.read_text(encoding="utf-8")
            manifest_text = manifest_path.read_text(encoding="utf-8")
            self.assertIn('"credit_name": "Anonymous donor"', ledger_text)
            self.assertIn('"institute": "New Lab"', ledger_text)
            self.assertIn('"public_anonymous": true', ledger_text)
            self.assertIn('"contributor_email": "new@example.com"', manifest_text)
            self.assertIn('"metadata_update_request_id": "metadata-one"', manifest_text)


if __name__ == "__main__":
    unittest.main()
