import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.intake_donations import promoted_submission_ids


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


if __name__ == "__main__":
    unittest.main()
