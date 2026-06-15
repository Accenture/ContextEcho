from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from donate import submit


class SubmitTests(unittest.TestCase):
    def test_verify_cache_skips_reverify_when_artifact_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            session = Path(td) / "session.redacted.jsonl"
            session.write_text('{"text":"clean"}\n', encoding="utf-8")
            submit.write_verify_cache(session, {"passed": True})

            with mock.patch("donate.submit.subprocess.run") as run:
                self.assertTrue(submit.verify_passed(session))

        run.assert_not_called()

    def test_verify_cache_does_not_skip_reverify_when_artifact_changed(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            session = Path(td) / "session.redacted.jsonl"
            session.write_text('{"text":"clean"}\n', encoding="utf-8")
            submit.write_verify_cache(session, {"passed": True})
            session.write_text('{"text":"changed"}\n', encoding="utf-8")

            completed = mock.Mock(returncode=0, stdout="[verify] PASS\n", stderr="")
            with mock.patch("donate.submit.subprocess.run", return_value=completed) as run:
                self.assertTrue(submit.verify_passed(session))

        run.assert_called_once()

    def test_gather_artifacts_includes_verify_cache_when_present(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            session = root / "session.redacted.jsonl"
            session.write_text('{"text":"clean"}\n', encoding="utf-8")
            (root / "session.manifest.json").write_text("{}", encoding="utf-8")
            (root / "CONSENT.md").write_text("consent", encoding="utf-8")
            submit.write_verify_cache(session, {"passed": True})

            names = [name for _path, name in submit.gather_artifacts(session)]

        self.assertIn("verify.json", names)


if __name__ == "__main__":
    unittest.main()
