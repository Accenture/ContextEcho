import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.promote_donation import default_label as promote_default_label
from scripts.promote_donation import sha256_file
from scripts.review_donation import default_label as review_default_label


class MaintainerLabelTests(unittest.TestCase):
    def test_default_labels_include_submission_id(self):
        manifest = {"credit_name": "Carlos"}
        submission = Path("hf_staging_download/pending/submission-abc12345")

        self.assertEqual(review_default_label(manifest, submission), "Carlos-submission-abc12345")
        self.assertEqual(promote_default_label(manifest, submission), "Carlos-submission-abc12345")

    def test_sha256_file_hashes_session_content(self):
        with TemporaryDirectory() as td:
            path = Path(td) / "session.redacted.jsonl"
            path.write_text("abc\n", encoding="utf-8")

            self.assertEqual(
                sha256_file(path),
                "edeaaff3f1774ad2888673770c6d64097e391bc362d7d6fb34982ddf0efd18cb",
            )


if __name__ == "__main__":
    unittest.main()
