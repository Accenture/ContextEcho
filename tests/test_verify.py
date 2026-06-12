from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from donate.verify import verify_session


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


if __name__ == "__main__":
    unittest.main()
