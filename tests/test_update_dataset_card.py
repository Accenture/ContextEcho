import json
import tempfile
import unittest
from pathlib import Path

from scripts.update_dataset_card import load_ledger_counts, main, render_dataset_card


class UpdateDatasetCardTests(unittest.TestCase):
    def test_card_summarizes_public_ledger_without_emails(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "data_archive_release_v2"
            donation = root / "data" / "donations" / "donor-one"
            donation.mkdir(parents=True)
            manifest = donation / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "public_anonymous": False,
                        "contributor_email": "donor@example.com",
                        "contributor_institute": "Example University",
                        "redacted_file": "session_donor-one.jsonl",
                    }
                ),
                encoding="utf-8",
            )
            ledger = root / "data" / "donations" / "ledger.jsonl"
            ledger.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "submission_id": "submission-one",
                                "decision": "ACCEPTABLE",
                                "manifest_path": "data/donations/donor-one/manifest.json",
                                "contributor": "Donor One",
                                "credit_name": "Donor One",
                                "public_anonymous": False,
                                "institute": "Example University",
                                "agent": "Codex CLI",
                                "model": "GPT-5",
                                "org": "OpenAI",
                                "records": 1000,
                                "turns": 120,
                                "compactions": 2,
                                "domain": "agentic-coding",
                                "language": "Python",
                                "privacy_tier": "full_redacted",
                                "session_sha256": "abc",
                                "promoted_utc": "2026-06-15T00:00:00+00:00",
                            }
                        ),
                        json.dumps(
                            {
                                "submission_id": "submission-old",
                                "decision": "SUPERSEDED",
                                "superseded_by": "submission-one",
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            card = render_dataset_card(root)

        self.assertIn("Institution coverage | 1 institution", card)
        self.assertNotIn("Example University", card)
        self.assertIn("Codex CLI (1)", card)
        self.assertIn("ACCEPTABLE | 1", card)
        self.assertIn("SUPERSEDED | 1", card)
        self.assertNotIn("donor@example.com", card)

    def test_card_counts_public_anonymous_institution_coverage(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "data_archive_release_v2"
            donation = root / "data" / "donations" / "donor-one"
            donation.mkdir(parents=True)
            (donation / "manifest.json").write_text(
                json.dumps(
                    {
                        "public_anonymous": True,
                        "contributor_email": "donor@example.com",
                        "contributor_institute": "Private Lab",
                    }
                ),
                encoding="utf-8",
            )
            ledger = root / "data" / "donations" / "ledger.jsonl"
            ledger.write_text(
                json.dumps(
                    {
                        "submission_id": "submission-one",
                        "decision": "ACCEPTABLE",
                        "manifest_path": "data/donations/donor-one/manifest.json",
                        "contributor": "Named Donor",
                        "credit_name": "Named Donor",
                        "public_anonymous": True,
                        "institute": "Private Lab",
                        "agent": "Codex CLI",
                        "model": "GPT-5",
                        "org": "OpenAI",
                        "turns": 120,
                        "compactions": 2,
                        "domain": "agentic-coding",
                        "language": "Python",
                        "session_sha256": "abc",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            card = render_dataset_card(root)

        self.assertIn("Institution coverage | 1 institution", card)
        self.assertNotIn("Private Lab", card)
        self.assertNotIn("donor@example.com", card)

    def test_check_mode_reports_stale_card(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "data_archive_release_v2"
            (root / "data" / "donations").mkdir(parents=True)
            out = Path(td) / "DATASET_CARD.md"
            out.write_text("stale\n", encoding="utf-8")
            status = main(["--dataset-root", str(root), "--out", str(out), "--check"])
        self.assertEqual(status, 1)

    def test_ledger_counts_handles_missing_ledger(self):
        with tempfile.TemporaryDirectory() as td:
            counts = load_ledger_counts(Path(td) / "missing")
        self.assertEqual(counts.rows, 0)
        self.assertEqual(counts.acceptable, 0)


if __name__ == "__main__":
    unittest.main()
