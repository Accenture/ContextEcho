import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.intake_donations import (
    append_review_record,
    known_session_lineage,
    known_session_hashes,
    load_review_registry,
    promoted_submission_ids,
    sha256_file,
    submission_fingerprint,
    submission_lineage,
    submission_session_hash,
)


class IntakeDonationTests(unittest.TestCase):
    def test_promoted_submission_ids_reads_acceptable_ledger_rows(self):
        with TemporaryDirectory() as td:
            root = Path(td)
            ledger = root / "data" / "donations" / "ledger.jsonl"
            ledger.parent.mkdir(parents=True)
            ledger.write_text(
                "\n".join(
                    [
                        json.dumps({"submission_id": "submission-accepted", "decision": "ACCEPTABLE"}),
                        json.dumps({"submission_id": "submission-check", "decision": "CHECK_REQUIRED"}),
                        "not json",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            self.assertEqual(promoted_submission_ids(root), {"submission-accepted"})

    def test_review_registry_records_failed_or_accepted_submissions(self):
        with TemporaryDirectory() as td:
            root = Path(td)
            append_review_record(root, {
                "submission_id": "submission-failed",
                "fingerprint": "abc",
                "decision": "CHECK_REQUIRED",
            })
            append_review_record(root, {
                "submission_id": "submission-accepted",
                "fingerprint": "def",
                "decision": "ACCEPTABLE",
            })

            registry = load_review_registry(root)
            self.assertEqual(registry["submission-failed"]["decision"], "CHECK_REQUIRED")
            self.assertEqual(registry["submission-accepted"]["fingerprint"], "def")

    def test_submission_fingerprint_changes_when_files_change(self):
        with TemporaryDirectory() as td:
            sub = Path(td) / "submission-abc"
            sub.mkdir()
            (sub / "session.redacted.jsonl").write_text('{"type":"user"}\n', encoding="utf-8")
            (sub / "manifest.json").write_text('{"turns":1}\n', encoding="utf-8")
            (sub / "CONSENT.md").write_text("I consent.\n", encoding="utf-8")

            before = submission_fingerprint(sub)
            (sub / "manifest.json").write_text('{"turns":2}\n', encoding="utf-8")
            after = submission_fingerprint(sub)

            self.assertNotEqual(before, after)

    def test_session_hash_tracks_duplicate_redacted_artifacts(self):
        with TemporaryDirectory() as td:
            root = Path(td)
            sub1 = root / "pending" / "submission-one"
            sub2 = root / "pending" / "submission-two"
            sub1.mkdir(parents=True)
            sub2.mkdir(parents=True)
            for sub in (sub1, sub2):
                (sub / "session.redacted.jsonl").write_text('{"type":"user","content":"same"}\n', encoding="utf-8")
                (sub / "manifest.json").write_text("{}\n", encoding="utf-8")
                (sub / "CONSENT.md").write_text("consent\n", encoding="utf-8")

            session_hash = submission_session_hash(sub1)
            self.assertEqual(session_hash, submission_session_hash(sub2))
            append_review_record(root, {
                "submission_id": "submission-one",
                "fingerprint": submission_fingerprint(sub1),
                "session_sha256": session_hash,
                "decision": "ACCEPTABLE",
            })

            self.assertEqual(known_session_hashes(root)[session_hash], "submission-one")

    def test_session_lineage_tracks_same_source_variants(self):
        with TemporaryDirectory() as td:
            root = Path(td)
            sub = root / "pending" / "submission-one"
            sub.mkdir(parents=True)
            (sub / "manifest.json").write_text(
                json.dumps({
                    "source_session_id": "source-abc",
                    "conversation_fingerprint": "conv-abc",
                })
                + "\n",
                encoding="utf-8",
            )

            self.assertEqual(
                submission_lineage(sub),
                {"source_session_id": "source-abc", "conversation_fingerprint": "conv-abc"},
            )

            append_review_record(root, {
                "submission_id": "submission-one",
                "fingerprint": "abc",
                "source_session_id": "source-abc",
                "conversation_fingerprint": "conv-abc",
                "decision": "ACCEPTABLE",
            })

            lineage = known_session_lineage(root)
            self.assertEqual(lineage["source_session_id:source-abc"], "submission-one")
            self.assertEqual(lineage["conversation_fingerprint:conv-abc"], "submission-one")

    def test_known_session_hashes_reads_legacy_ledger_session_path(self):
        with TemporaryDirectory() as td:
            root = Path(td)
            session = root / "data" / "sessions" / "session_old.jsonl"
            ledger = root / "data" / "donations" / "ledger.jsonl"
            session.parent.mkdir(parents=True)
            ledger.parent.mkdir(parents=True)
            session.write_text('{"type":"user","content":"old"}\n', encoding="utf-8")
            ledger.write_text(
                json.dumps({
                    "submission_id": "submission-old",
                    "decision": "ACCEPTABLE",
                    "session_path": "data/sessions/session_old.jsonl",
                })
                + "\n",
                encoding="utf-8",
            )

            self.assertEqual(known_session_hashes(root), {sha256_file(session): "submission-old"})


if __name__ == "__main__":
    unittest.main()
