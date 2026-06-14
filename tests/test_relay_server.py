from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

try:
    from donate import relay_server
except ModuleNotFoundError as exc:
    if exc.name != "fastapi":
        raise
    relay_server = None  # type: ignore[assignment]


@unittest.skipIf(relay_server is None, "relay dependencies are not installed")
class RelayServerTests(unittest.TestCase):
    def test_session_prefixes_preserve_nested_submission_paths(self) -> None:
        files = [
            "pending/submission-abc/session.redacted.jsonl",
            "pending/submission-abc/manifest.json",
            "session.redacted.jsonl",
        ]

        self.assertEqual(relay_server._session_prefixes(files), ["", "pending/submission-abc"])

    def test_backfill_seen_hashes_records_existing_sessions(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            state_dir = root / "state"
            session = root / "session.redacted.jsonl"
            manifest = root / "manifest.json"
            session.write_text(
                '{"timestamp":"2026-01-01T00:00:00Z","type":"event_msg","payload":{"type":"user_message","message":"<REDACTED>"}}\n',
                encoding="utf-8",
            )
            manifest.write_text(
                json.dumps({
                    "session_id": "donation-old",
                    "records": 1,
                    "turns": 1,
                }),
                encoding="utf-8",
            )

            api = mock.Mock()
            api.list_repo_files.return_value = [
                "pending/submission-old/session.redacted.jsonl",
                "pending/submission-old/manifest.json",
            ]

            def fake_download(*, filename: str, **_kwargs: object) -> str:
                return str(manifest if filename.endswith("manifest.json") else session)

            with (
                mock.patch("donate.relay_server.HfApi", return_value=api),
                mock.patch("donate.relay_server.hf_hub_download", side_effect=fake_download),
                mock.patch("donate.relay_server.BACKFILL_REPOS", ["owner/repo"]),
                mock.patch("donate.relay_server.STATE_DIR", state_dir),
                mock.patch("donate.relay_server.SEEN_HASHES", state_dir / "seen_artifact_hashes.jsonl"),
            ):
                result = relay_server._backfill_seen_hashes_from_hf()
                records = relay_server._read_seen_records()

        self.assertEqual(result["scanned"], 1)
        self.assertEqual(result["added"], 1)
        self.assertEqual(records[0]["submission_id"], "donation-old")
        self.assertTrue(records[0]["artifact_hash"])
        self.assertTrue(records[0]["conversation_fingerprint"].startswith("conv-"))
        self.assertEqual(records[0]["fingerprint_version"], "structure-v1")


if __name__ == "__main__":
    unittest.main()
