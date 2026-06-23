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

    def test_ensure_lfs_jsonl_rule_adds_stable_gitattributes_rule(self) -> None:
        api = mock.Mock()

        with mock.patch("donate.relay_server._read_staging_gitattributes", return_value="*.zip filter=lfs diff=lfs merge=lfs -text\n"):
            relay_server._ensure_lfs_jsonl_rule(api, "hf_token")

        api.create_commit.assert_called_once()
        op = api.create_commit.call_args.kwargs["operations"][0]
        self.assertEqual(op.path_in_repo, ".gitattributes")
        self.assertIn(relay_server.LFS_JSONL_RULE.encode("utf-8"), op.path_or_fileobj)

    def test_ensure_lfs_jsonl_rule_noops_when_present(self) -> None:
        api = mock.Mock()

        with mock.patch("donate.relay_server._read_staging_gitattributes", return_value=relay_server.LFS_JSONL_RULE + "\n"):
            relay_server._ensure_lfs_jsonl_rule(api, "hf_token")

        api.create_commit.assert_not_called()

    def test_copy_upload_limited_writes_temp_file_and_enforces_limit(self) -> None:
        class FakeUpload:
            def __init__(self, chunks):
                self.chunks = list(chunks)

            async def read(self, _size):
                return self.chunks.pop(0) if self.chunks else b""

        async def run_ok():
            path, total = await relay_server._copy_upload_limited(
                FakeUpload([b'{"text":"', b'clean"}\n']),
                100,
                "session.redacted.jsonl",
            )
            try:
                self.assertEqual(total, len(b'{"text":"clean"}\n'))
                self.assertEqual(path.read_bytes(), b'{"text":"clean"}\n')
            finally:
                path.unlink(missing_ok=True)

        async def run_too_large():
            with self.assertRaises(relay_server.HTTPException) as cm:
                await relay_server._copy_upload_limited(
                    FakeUpload([b"12345", b"67890"]),
                    6,
                    "session.redacted.jsonl",
                )
            self.assertEqual(cm.exception.status_code, 413)

        import asyncio

        asyncio.run(run_ok())
        asyncio.run(run_too_large())

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
        self.assertEqual(records[0]["submission_id"], "submission-old")
        self.assertTrue(records[0]["artifact_hash"])
        self.assertTrue(records[0]["conversation_fingerprint"].startswith("conv-"))
        self.assertEqual(records[0]["fingerprint_version"], "structure-v1")

    def test_backfill_seen_hashes_reads_public_release_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            state_dir = root / "state"
            session = root / "session_public.jsonl"
            manifest = root / "manifest.json"
            ledger = root / "ledger.jsonl"
            session.write_text(
                '{"timestamp":"2026-01-01T00:00:00Z","type":"event_msg","payload":{"type":"user_message","message":"<REDACTED>"}}\n',
                encoding="utf-8",
            )
            manifest.write_text(
                json.dumps({
                    "reviewed_submission_id": "submission-public",
                    "records": 1,
                    "turns": 1,
                }),
                encoding="utf-8",
            )
            ledger.write_text(
                json.dumps({
                    "submission_id": "submission-public",
                    "session_path": "data/sessions/session_public.jsonl",
                    "manifest_path": "data/donations/donor/manifest.json",
                })
                + "\n",
                encoding="utf-8",
            )

            api = mock.Mock()
            api.list_repo_files.return_value = [
                "data/donations/ledger.jsonl",
                "data/donations/donor/manifest.json",
                "data/sessions/session_public.jsonl",
            ]

            def fake_download(*, filename: str, **_kwargs: object) -> str:
                if filename.endswith("ledger.jsonl"):
                    return str(ledger)
                if filename.endswith("manifest.json"):
                    return str(manifest)
                return str(session)

            with (
                mock.patch("donate.relay_server.HfApi", return_value=api),
                mock.patch("donate.relay_server.hf_hub_download", side_effect=fake_download),
                mock.patch("donate.relay_server.BACKFILL_REPOS", ["owner/public"]),
                mock.patch("donate.relay_server.STATE_DIR", state_dir),
                mock.patch("donate.relay_server.SEEN_HASHES", state_dir / "seen_artifact_hashes.jsonl"),
            ):
                result = relay_server._backfill_seen_hashes_from_hf()
                records = relay_server._read_seen_records()

        self.assertEqual(result["scanned"], 1)
        self.assertEqual(result["added"], 1)
        self.assertEqual(records[0]["submission_id"], "submission-public")
        self.assertTrue(records[0]["conversation_fingerprint"].startswith("conv-"))

    def test_backfill_seen_hashes_reads_public_session_files_without_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            state_dir = root / "state"
            session = root / "session_chainassemble.jsonl"
            session.write_text(
                '{"timestamp":"2026-01-01T00:00:00Z","type":"event_msg","payload":{"type":"user_message","message":"<REDACTED>"}}\n',
                encoding="utf-8",
            )

            api = mock.Mock()
            api.list_repo_files.return_value = ["data/sessions/session_chainassemble.jsonl"]

            def fake_download(*, filename: str, **_kwargs: object) -> str:
                self.assertEqual(filename, "data/sessions/session_chainassemble.jsonl")
                return str(session)

            with (
                mock.patch("donate.relay_server.HfApi", return_value=api),
                mock.patch("donate.relay_server.hf_hub_download", side_effect=fake_download),
                mock.patch("donate.relay_server.BACKFILL_REPOS", ["owner/public"]),
                mock.patch("donate.relay_server.STATE_DIR", state_dir),
                mock.patch("donate.relay_server.SEEN_HASHES", state_dir / "seen_artifact_hashes.jsonl"),
            ):
                result = relay_server._backfill_seen_hashes_from_hf()
                records = relay_server._read_seen_records()

        self.assertEqual(result["scanned"], 1)
        self.assertEqual(result["added"], 1)
        self.assertEqual(records[0]["submission_id"], "public-session-chainassemble")
        self.assertTrue(records[0]["conversation_fingerprint"].startswith("conv-"))

    def test_lineage_status_reports_received_and_update_ready(self) -> None:
        seen = [
            {
                "submission_id": "submission-old",
                "source_session_id": "source-1",
                "conversation_fingerprint": "conv-1",
                "turns": 100,
                "records": 200,
            }
        ]

        received = relay_server._lineage_status(
            {"source_session_id": "source-1", "conversation_fingerprint": "conv-1", "turns": 110, "records": 205},
            seen,
        )
        update_ready = relay_server._lineage_status(
            {"source_session_id": "source-1", "conversation_fingerprint": "conv-1", "turns": 160, "records": 260},
            seen,
        )

        self.assertTrue(received["received"])
        self.assertFalse(received["update_ready"])
        self.assertEqual(received["new_turns"], 10)
        self.assertEqual(received["submission_id"], "submission-old")
        self.assertTrue(update_ready["received"])
        self.assertTrue(update_ready["update_ready"])
        self.assertEqual(update_ready["new_turns"], 60)

    def test_remove_seen_records_removes_only_matching_submission(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            state_dir = Path(td)
            with (
                mock.patch("donate.relay_server.STATE_DIR", state_dir),
                mock.patch("donate.relay_server.SEEN_HASHES", state_dir / "seen_artifact_hashes.jsonl"),
                mock.patch("donate.relay_server.SUBMISSION_EVENTS", state_dir / "submission_events.jsonl"),
            ):
                relay_server._append_seen_record({
                    "artifact_hash": "hash-1",
                    "submission_id": "submission-one",
                    "source_session_id": "source-1",
                    "conversation_fingerprint": "conv-1",
                    "turns": 100,
                    "records": 200,
                })
                relay_server._append_seen_record({
                    "artifact_hash": "hash-2",
                    "submission_id": "submission-two",
                    "source_session_id": "source-2",
                    "conversation_fingerprint": "conv-2",
                    "turns": 120,
                    "records": 240,
                })

                result = relay_server._remove_seen_records({"submission_id": "submission-one"})
                records = relay_server._read_seen_records()

        self.assertEqual(result["removed"], 1)
        self.assertEqual(result["remaining"], 1)
        self.assertEqual(result["removed_submission_ids"], ["submission-one"])
        self.assertEqual(records[0]["submission_id"], "submission-two")
        events = [json.loads(line) for line in (state_dir / "submission_events.jsonl").read_text().splitlines()]
        self.assertEqual(events[0]["event"], "reset_one")
        self.assertEqual(events[0]["removed"], 1)
        self.assertEqual(events[0]["removed_submission_ids"], ["submission-one"])

    def test_remove_seen_records_requires_specific_match_key(self) -> None:
        with self.assertRaises(relay_server.HTTPException) as cm:
            relay_server._remove_seen_records({"unknown": "value"})

        self.assertEqual(cm.exception.status_code, 400)

    def test_submission_events_are_read_newest_first_by_admin_endpoint(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            state_dir = Path(td)
            with (
                mock.patch("donate.relay_server.STATE_DIR", state_dir),
                mock.patch("donate.relay_server.SUBMISSION_EVENTS", state_dir / "submission_events.jsonl"),
                mock.patch("donate.relay_server.ADMIN_TOKEN", "secret"),
            ):
                relay_server._append_submission_event("submitted", submission_id="submission-one")
                relay_server._append_submission_event("reset_one", removed_submission_ids=["submission-one"])
                result = relay_server.admin_submission_events(x_admin_token="secret")

        self.assertEqual([row["event"] for row in result["events"]], ["reset_one", "submitted"])

    def test_metadata_update_request_is_persisted_and_audited(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            state_dir = Path(td)
            with (
                mock.patch("donate.relay_server.STATE_DIR", state_dir),
                mock.patch("donate.relay_server.METADATA_UPDATES", state_dir / "metadata_updates.jsonl"),
                mock.patch("donate.relay_server.SUBMISSION_EVENTS", state_dir / "submission_events.jsonl"),
            ):
                record = relay_server._metadata_update_request({
                    "submission_id": "submission-abc12345",
                    "credit_name": "New Name",
                    "contributor_email": "new@example.com",
                    "contributor_institute": "New Institute",
                    "public_anonymous": True,
                })
                requests = relay_server._read_jsonl(state_dir / "metadata_updates.jsonl")
                events = relay_server._read_jsonl(state_dir / "submission_events.jsonl")

        self.assertEqual(record["status"], "pending")
        self.assertEqual(requests[0]["submission_id"], "submission-abc12345")
        self.assertEqual(requests[0]["credit_name"], "New Name")
        self.assertEqual(events[0]["event"], "metadata_update_requested")

    def test_metadata_update_request_allows_partial_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            state_dir = Path(td)
            with (
                mock.patch("donate.relay_server.STATE_DIR", state_dir),
                mock.patch("donate.relay_server.METADATA_UPDATES", state_dir / "metadata_updates.jsonl"),
                mock.patch("donate.relay_server.SUBMISSION_EVENTS", state_dir / "submission_events.jsonl"),
            ):
                record = relay_server._metadata_update_request({
                    "submission_id": "submission-abc12345",
                    "contributor_institute": "Updated Institute",
                })

        self.assertEqual(record["credit_name"], "")
        self.assertEqual(record["contributor_email"], "")
        self.assertEqual(record["contributor_institute"], "Updated Institute")

    def test_seen_record_summary_counts_records_and_totals(self) -> None:
        summary = relay_server._seen_record_summary([
            {
                "submission_id": "submission-one",
                "source_session_id": "source-1",
                "conversation_fingerprint": "conv-1",
                "turns": 100,
                "records": 200,
            },
            {
                "submission_id": "submission-two",
                "source_session_id": "source-2",
                "conversation_fingerprint": "conv-2",
                "turns": "25",
                "records": "50",
            },
        ])

        self.assertEqual(summary["records"], 2)
        self.assertEqual(summary["submissions"], 2)
        self.assertEqual(summary["source_sessions"], 2)
        self.assertEqual(summary["conversations"], 2)
        self.assertEqual(summary["turns"], 125)
        self.assertEqual(summary["jsonl_records"], 250)

    def test_pending_submissions_from_hf_reads_manifest_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            manifest = Path(td) / "manifest.json"
            manifest.write_text(
                json.dumps({
                    "agent": "Codex CLI",
                    "model": "gpt-5",
                    "turns": 120,
                    "records": 456,
                    "compactions": 3,
                    "source_session_id": "source-abc",
                    "conversation_fingerprint": "conv-abc",
                    "credit_name": "Donor",
                    "contributor_email": "donor@example.com",
                    "contributor_institute": "Institute",
                    "privacy_tier": "full_redacted",
                }),
                encoding="utf-8",
            )
            api = mock.Mock()
            api.list_repo_files.return_value = [
                "pending/submission-abc/session.redacted.jsonl",
                "pending/submission-abc/manifest.json",
                "pending/submission-abc/CONSENT.md",
            ]

            def fake_download(*, filename: str, **_kwargs: object) -> str:
                self.assertEqual(filename, "pending/submission-abc/manifest.json")
                return str(manifest)

            with (
                mock.patch("donate.relay_server.HfApi", return_value=api),
                mock.patch("donate.relay_server.hf_hub_download", side_effect=fake_download),
                mock.patch("donate.relay_server.STAGING_REPO", "owner/staging"),
            ):
                result = relay_server._pending_submissions_from_hf()

        self.assertTrue(result["ok"])
        self.assertEqual(len(result["submissions"]), 1)
        row = result["submissions"][0]
        self.assertEqual(row["submission_id"], "submission-abc")
        self.assertEqual(row["agent"], "Codex CLI")
        self.assertEqual(row["turns"], 120)
        self.assertTrue(row["has_session"])
        self.assertTrue(row["has_manifest"])
        self.assertTrue(row["has_consent"])


if __name__ == "__main__":
    unittest.main()
