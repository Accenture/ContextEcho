import unittest
import errno
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from donate.verify import verify_session
from donate.web import (
    DEFAULT_RELAY_URL,
    INDEX_HTML,
    _auto_repair_until_verified,
    _load_contributors_markdown,
    _parse_dataset_card_coverage,
    _parse_contributor_leaderboard,
    _parse_donated_sessions,
    already_submitted,
    annotate_donated,
    artifact_key,
    clear_donation_record,
    clear_donation_registry,
    create_server,
    duplicate_submit_detail,
    donation_fit,
    donation_ready,
    friendly_submit_error,
    is_duplicate_submit_output,
    load_donated_artifact_keys,
    local_pending_summary,
    metadata_for_redacted_artifact,
    parse_submit_output,
    relay_url,
    required_contributor_fields,
    save_donation_record,
    session_update_ready,
    session_key,
    sanitize_diagnostic_text,
    stream_error_message,
    submit_auto_metadata,
    wizard_error_report_payload,
    write_receipt,
)


class WebTests(unittest.TestCase):
    def test_relay_url_defaults_to_official_relay(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertEqual(relay_url(), DEFAULT_RELAY_URL)

    def test_stream_error_message_does_not_hide_blank_exceptions(self):
        self.assertEqual(stream_error_message(AssertionError(), "Redaction"), "Redaction failed: AssertionError")
        missing = ModuleNotFoundError("No module named 'presidio_analyzer'", name="presidio_analyzer")
        message = stream_error_message(missing, "Redaction")
        self.assertIn("missing presidio_analyzer", message)
        self.assertIn("rerun the install command", message)

    def test_wizard_error_report_is_sanitized_for_maintainers(self):
        text = "/Users/jane.doe/project/file.py hf_abcdefghijklmnopqrstuvwxyz jane@example.com"
        sanitized = sanitize_diagnostic_text(text)
        self.assertNotIn("jane.doe", sanitized)
        self.assertNotIn("abcdefghijklmnopqrstuvwxyz", sanitized)
        self.assertNotIn("jane@example.com", sanitized)
        self.assertIn("/Users/<USER>", sanitized)
        self.assertIn("<SECRET>", sanitized)
        self.assertIn("<EMAIL>", sanitized)

        payload = wizard_error_report_payload(
            {"auto": {"agent": "Codex CLI", "model": "gpt-5", "turns": 120, "records": 300}},
            RuntimeError("bad path /Users/jane.doe/session.jsonl"),
            "Redaction",
            "Traceback /Users/jane.doe/session.jsonl",
        )
        self.assertEqual(payload["submission_id"], "wizard-error")
        self.assertEqual(payload["reason"], "wizard_error")
        self.assertIn("Codex CLI", payload["message"])
        self.assertNotIn("jane.doe", payload["message"])

    def test_submit_step_previews_public_leaderboard_identity(self):
        self.assertIn("submitLeaderboardPreview", INDEX_HTML)
        self.assertIn("Leaderboard preview", INDEX_HTML)
        self.assertIn("Show me anonymously on the public leaderboard", INDEX_HTML)
        self.assertIn("Default is public credit", INDEX_HTML)
        self.assertIn("You (anonymous)", INDEX_HTML)
        self.assertIn("leaderboardPreviewPage", INDEX_HTML)
        self.assertIn("Page ${currentPage + 1} of ${totalPages}", INDEX_HTML)
        self.assertIn("bindLeaderboardPager(model)", INDEX_HTML)
        self.assertIn("scrollToSubmitResult", INDEX_HTML)
        self.assertIn("scrollIntoView({behavior:'smooth', block:'start'})", INDEX_HTML)
        self.assertIn("pendingLeaderboardModel(publicCreditName, publicAnonymous, turns, compactions, localPending, publicCreditName)", INDEX_HTML)

    def test_donated_rows_show_copyable_support_submission_id(self):
        self.assertIn("support-id", INDEX_HTML)
        self.assertNotIn("local-record", INDEX_HTML)
        self.assertNotIn("Local receipt only; relay does not currently have this submission", INDEX_HTML)
        self.assertIn("data-copy-submission", INDEX_HTML)
        self.assertIn("sessionMenu", INDEX_HTML)
        self.assertIn("data-session-action=\"update\"", INDEX_HTML)
        self.assertIn("data-session-action=\"problem\"", INDEX_HTML)
        self.assertIn("showSessionMenu", INDEX_HTML)
        self.assertIn("/api/metadata_update", INDEX_HTML)
        self.assertIn("Send Info Update", INDEX_HTML)
        self.assertIn("Editing contributor info for", INDEX_HTML)
        self.assertIn("prefillContributorFields(selected, false)", INDEX_HTML)
        self.assertIn("prefillContributorFields(session, true)", INDEX_HTML)
        self.assertIn("localContributorRecord", INDEX_HTML)
        self.assertIn("credit_name: data.receipt?.credit_name", INDEX_HTML)
        self.assertIn("Update sent successfully", INDEX_HTML)
        self.assertIn("Back to Sessions", INDEX_HTML)
        self.assertIn("setContributorFieldsLocked", INDEX_HTML)
        self.assertIn("Report problem", INDEX_HTML)
        self.assertIn("/api/support_request", INDEX_HTML)
        self.assertIn("Send Report", INDEX_HTML)
        self.assertIn("Report sent successfully", INDEX_HTML)
        self.assertNotIn("New public/credit name for this donation", INDEX_HTML)
        self.assertIn("Copied maintainer reset ID", INDEX_HTML)
        self.assertIn("normalizeSubmissionId", INDEX_HTML)
        self.assertNotIn("localPathDonated", INDEX_HTML)
        self.assertNotIn("record && !relayChecked", INDEX_HTML)
        self.assertNotIn("What becomes public after maintainer acceptance?", INDEX_HTML)

    def test_maintainer_metadata_updates_can_be_approved(self):
        html = Path("docs/maintainer/index.html").read_text(encoding="utf-8")
        self.assertIn("/api/admin/metadata-updates/approve", html)
        self.assertIn("data-approve-metadata", html)
        self.assertIn("function approveMetadata", html)
        self.assertIn("Relay records updated", html)
        self.assertIn("Staging manifest updated", html)
        self.assertIn("Run make update-release-metadata locally to sync approved metadata into public files", html)
        self.assertIn("function formatMinute", html)
        self.assertIn("formatMinute(row.submitted_utc)", html)
        self.assertIn("formatMinute(row.approved_utc)", html)
        self.assertIn("formatMinute(row.ts)", html)
        self.assertIn("Support Requests", html)
        self.assertIn("/api/admin/support-requests", html)
        self.assertIn("function resolveSupport", html)
        self.assertIn("Mark resolved", html)

    def test_maintainer_pending_view_separates_promoted_from_needs_validation(self):
        html = Path("docs/maintainer/index.html").read_text(encoding="utf-8")
        self.assertIn("pendingStatusFilter:'needs_validation'", html)
        self.assertIn("Needs validation", html)
        self.assertIn("Promoted", html)
        self.assertIn("All staging", html)
        self.assertIn("review_status", html)
        self.assertIn("function reviewStatusPill", html)
        self.assertIn("promoted</span>", html)
        self.assertIn("needs validation</span>", html)
        self.assertIn("Run maintainer intake locally", html)
        self.assertIn("Loaded ${counts.all} staging submission", html)

    def test_submit_requires_contributor_identity_fields(self):
        self.assertIn("Contributor info is required", INDEX_HTML)
        self.assertIn('id="contributorName" placeholder="your name or handle" required', INDEX_HTML)
        self.assertIn('id="contributorEmail" type="email" list="emailSuggestions" placeholder="you@example.com" required', INDEX_HTML)
        self.assertIn('datalist id="emailSuggestions"', INDEX_HTML)
        self.assertIn('id="contributorInstitute" placeholder="University / company / independent" required', INDEX_HTML)
        self.assertIn("function contributorFieldsComplete()", INDEX_HTML)
        self.assertIn("$('submitBtn').disabled = !canSubmitArtifact || !contributorComplete", INDEX_HTML)
        self.assertNotIn("missingContributorMessage", INDEX_HTML)
        self.assertNotIn("Please fill ${missing.join(', ')} before submitting.", INDEX_HTML)
        self.assertIn("function completeEmailDomain()", INDEX_HTML)
        self.assertIn("commonEmailDomains", INDEX_HTML)
        self.assertIn("function updateEmailSuggestions()", INDEX_HTML)
        self.assertIn("'outlook.com'", INDEX_HTML)
        self.assertIn("if(matches.length === 1)", INDEX_HTML)
        self.assertIn("$('contributorEmail').onblur = completeEmailDomain", INDEX_HTML)
        with self.assertRaisesRegex(ValueError, "Missing: email"):
            required_contributor_fields({"contributor": "Donor", "email": "", "institute": "Lab"})
        self.assertEqual(
            required_contributor_fields({"contributor": " Donor ", "email": "d@example.com", "institute": " Lab "}),
            {"contributor": "Donor", "email": "d@example.com", "institute": "Lab"},
        )

    def test_search_panel_can_run_cleanup_directly(self):
        self.assertIn("Redact and Verify Again", INDEX_HTML)
        self.assertIn("redact-primary", INDEX_HTML)
        self.assertIn("Redacting checked word", INDEX_HTML)
        self.assertNotIn("Running Redact and Verify again for the matched word", INDEX_HTML)
        self.assertIn("Redaction complete. The checked word is now found 0 times", INDEX_HTML)
        self.assertIn("Already redacted in this output", INDEX_HTML)
        self.assertIn("Automatic redaction", INDEX_HTML)
        self.assertIn("Private words you asked to redact", INDEX_HTML)
        self.assertIn("mergeRedactionStats", INDEX_HTML)
        self.assertIn("minmax(180px,1fr) 120px 100px", INDEX_HTML)
        self.assertIn("progress-time", INDEX_HTML)
        self.assertIn("progressBreakdown", INDEX_HTML)
        self.assertIn("Elapsed", INDEX_HTML)
        self.assertIn("${Math.round(pct)}% · ", INDEX_HTML)
        self.assertIn("The local browser connection was interrupted", INDEX_HTML)
        self.assertIn("Keep this tab open and the computer awake", INDEX_HTML)
        self.assertIn("sessionLocalKey", INDEX_HTML)

    def test_private_word_redaction_uses_post_verify_check(self):
        self.assertNotIn("Private words to redact", INDEX_HTML)
        self.assertNotIn("Use this only if a private word remains", INDEX_HTML)
        self.assertNotIn("Private words to remove on the next redaction run", INDEX_HTML)
        self.assertNotIn("removal box", INDEX_HTML)
        self.assertIn("Check whether a private word is still present", INDEX_HTML)
        self.assertIn("use Check File to redact it", INDEX_HTML)

    def test_malformed_jsonl_failure_uses_regenerate_language(self):
        self.assertIn("not a private word", INDEX_HTML)
        self.assertIn("Click Redact and Verify again to regenerate or normalize", INDEX_HTML)
        self.assertIn("category === 'detect_secrets' || category === 'malformed_jsonl'", INDEX_HTML)

    def test_duplicate_submit_view_explains_no_new_upload(self):
        self.assertIn("The maintainer relay rejected this repeat attempt", INDEX_HTML)
        self.assertIn("No new donation was needed", INDEX_HTML)
        self.assertIn("Already received", INDEX_HTML)
        self.assertIn("Local duplicate receipt", INDEX_HTML)
        self.assertNotIn("new folder in Hugging Face", INDEX_HTML)

    def test_donated_sessions_do_not_offer_local_clear_action(self):
        self.assertNotIn("Clear all local donated labels", INDEX_HTML)
        self.assertNotIn("clearDonatedBtn", INDEX_HTML)
        self.assertNotIn("Clear the local donated label for this session", INDEX_HTML)
        self.assertNotIn("/api/clear_donated_label", INDEX_HTML)
        self.assertNotIn("/api/clear_donated_labels", Path("donate/web.py").read_text(encoding="utf-8"))
        self.assertIn("Click the ID pill to copy the maintainer reset ID", INDEX_HTML)
        self.assertIn("update ready", INDEX_HTML)
        self.assertIn("new turns", INDEX_HTML)
        self.assertIn("contextechoDonatedRecordsV1", INDEX_HTML)
        self.assertIn("at least 50 new turns or 20% growth", INDEX_HTML)
        self.assertNotIn(">Clear label<", INDEX_HTML)
        self.assertNotIn(">Retry failed upload<", INDEX_HTML)

    def test_pick_session_explains_research_value_fit(self):
        self.assertIn("Ready sessions can be donated now", INDEX_HTML)
        self.assertIn("keep chatting sessions need more turns or a context compaction", INDEX_HTML)
        self.assertIn(".count-badge[data-tooltip]:hover:after", INDEX_HTML)
        self.assertIn(".fit-chip[data-tooltip]:hover:after", INDEX_HTML)
        self.assertIn("white-space:pre-line", INDEX_HTML)
        self.assertIn('<span class="fit-chip ready" data-tooltip="${escapeHtml(readySummaryTitle)}" aria-label="${escapeHtml(readySummaryTitle)}">Ready ${readyCount}</span>', INDEX_HTML)
        self.assertIn('<span class="fit-chip improve" data-tooltip="Not ready yet: needs more turns or a context compaction" aria-label="Not ready yet: needs more turns or a context compaction">Keep chatting ${counts.improve || 0}</span>', INDEX_HTML)
        self.assertIn("const readyCount = (counts.best || 0) + (counts.good || 0) + (counts.long || 0)", INDEX_HTML)
        self.assertIn("const readySummaryTitle = `Best: ${counts.best || 0}\\nBetter: ${counts.good || 0}\\nGood: ${counts.long || 0}`", INDEX_HTML)
        self.assertIn("function agentFamilyCounts()", INDEX_HTML)
        self.assertIn("const sessionSummaryTitle = `Claude: ${agentCounts.claude}\\nCodex: ${agentCounts.codex}\\nOther: ${agentCounts.other}`", INDEX_HTML)
        self.assertIn("$('sessionCount').dataset.tooltip = sessionSummaryTitle", INDEX_HTML)
        self.assertIn("$('sessionCount').setAttribute('aria-label', sessionSummaryTitle)", INDEX_HTML)
        self.assertIn(".pill.good { background:#dff1d9; color:#13552f; }", INDEX_HTML)
        self.assertIn(".pill.long { background:#dff1d9; color:#13552f; }", INDEX_HTML)
        self.assertNotIn('class="fit-chip donated"', INDEX_HTML)
        self.assertNotIn("Best ${counts.best || 0}", INDEX_HTML)
        self.assertIn("100+ turns and 2+ ctx cmp", INDEX_HTML)
        self.assertIn("50+ turns and 1+ ctx cmp", INDEX_HTML)
        self.assertIn('<span class="pill good"><span class="fit-star">&#9733;</span>Better</span> 50+ turns and 1+ ctx cmp', INDEX_HTML)
        self.assertIn('<span class="pill long"><span class="fit-star">&#9733;</span>Good</span> 100+ turns', INDEX_HTML)
        self.assertIn("if(value === 'good') return 'Better'", INDEX_HTML)
        self.assertIn("if(value === 'long') return 'Good'", INDEX_HTML)
        self.assertIn("currentFit === 'improve' ? '<span class=\"fit-arrow\">&uarr;</span>' : '<span class=\"fit-star\">&#9733;</span>'", INDEX_HTML)
        self.assertIn("keep chatting before donating", INDEX_HTML)
        self.assertIn("t>=100?'long':'improve'", INDEX_HTML)
        self.assertIn("sessionReady(s)", INDEX_HTML)
        self.assertIn("This session is not ready to donate yet", INDEX_HTML)
        self.assertIn("donation_ready(auto.get", Path("donate/web.py").read_text(encoding="utf-8"))
        self.assertIn("s.session_label || s.project", INDEX_HTML)
        self.assertIn('id="sessionSearch"', INDEX_HTML)
        self.assertIn('placeholder="Search sessions, agent, model, project"', INDEX_HTML)
        self.assertIn('id="sessionSortMode"', INDEX_HTML)
        self.assertIn('<option value="original">Discovery order</option>', INDEX_HTML)
        self.assertIn('<option value="model">Group by model</option>', INDEX_HTML)
        self.assertIn("let sessionSortMode = 'original'", INDEX_HTML)
        self.assertIn("let sessionSearchQuery = ''", INDEX_HTML)
        self.assertIn("function filteredSessionItems()", INDEX_HTML)
        self.assertIn("const donationInfo = localDonationInfo(s)", INDEX_HTML)
        self.assertIn("s?.relay_submission_id || ''", INDEX_HTML)
        self.assertIn("donationInfo.supportId || ''", INDEX_HTML)
        self.assertIn("if(!sessionSort.key && sessionSortMode !== 'model') return items", INDEX_HTML)
        self.assertIn("No sessions match this search", INDEX_HTML)
        self.assertIn("filtered from ${sessions.length}", INDEX_HTML)
        self.assertNotIn("Choose session file manually", INDEX_HTML)
        self.assertNotIn("/api/import_session", INDEX_HTML)
        self.assertNotIn("manualSessionFile", INDEX_HTML)
        self.assertNotIn("loadDiscoveryCache", INDEX_HTML)
        self.assertNotIn("contextechoDiscoveryCacheV1", INDEX_HTML)
        self.assertIn("/api/health", Path("donate/web.py").read_text(encoding="utf-8"))
        self.assertIn("access-control-allow-origin", Path("donate/web.py").read_text(encoding="utf-8"))

    def test_donation_fit_thresholds(self):
        self.assertEqual(donation_fit(100, 2), "best")
        self.assertEqual(donation_fit(99, 2), "good")
        self.assertEqual(donation_fit(50, 1), "good")
        self.assertEqual(donation_fit(100, 0), "long")
        self.assertEqual(donation_fit(49, 1), "improve")
        self.assertTrue(donation_ready(50, 1))
        self.assertTrue(donation_ready(100, 0))

    def test_privacy_switch_restores_cached_verified_redaction(self):
        self.assertIn("let redactionCache = new Map()", INDEX_HTML)
        self.assertIn("function redactionCacheKey()", INDEX_HTML)
        self.assertIn("function restoreCachedRedaction()", INDEX_HTML)
        self.assertIn("redactionCache.set(redactionCacheKey()", INDEX_HTML)
        self.assertIn("Restored the verified result for this privacy mode", INDEX_HTML)
        self.assertIn("if(!restoreCachedRedaction()) status('redactStatus', 'Privacy mode changed.", INDEX_HTML)

    def test_top_stats_are_embedded_in_support_card(self):
        self.assertNotIn('<a class="github" href="https://github.com/Accenture/ContextEcho"', INDEX_HTML)
        self.assertNotIn('<a class="dataset" href="https://huggingface.co/datasets/contextecho2026/persona-drift-contextecho"', INDEX_HTML)
        self.assertNotIn('<button class="github" type="button">Star on GitHub</button>', INDEX_HTML)
        self.assertNotIn('<button class="dataset" type="button">Like Dataset</button>', INDEX_HTML)
        self.assertNotIn("Star on GitHub</a>", INDEX_HTML)
        self.assertNotIn("Like Dataset</a>", INDEX_HTML)
        self.assertIn("Total Downloads", INDEX_HTML)
        self.assertNotIn("Downloads Last Month", INDEX_HTML)
        self.assertIn("GitHub Stars", INDEX_HTML)
        self.assertIn("Dataset Likes", INDEX_HTML)
        self.assertIn('<a class="stat-card" href="https://github.com/Accenture/ContextEcho"', INDEX_HTML)
        self.assertIn('<a class="stat-card" href="https://huggingface.co/datasets/contextecho2026/persona-drift-contextecho"', INDEX_HTML)
        self.assertIn('class="ranking" href="https://github.com/Accenture/ContextEcho/blob/main/CONTRIBUTORS.md"', INDEX_HTML)
        self.assertIn(">Ranking</a>", INDEX_HTML)
        self.assertNotIn('id="rankingBtn"', INDEX_HTML)
        self.assertNotIn("rankingPopover", INDEX_HTML)
        self.assertNotIn("showRankingPopover", INDEX_HTML)
        self.assertNotIn("hideRankingPopover", INDEX_HTML)
        self.assertIn('<div id="projectStats" class="stats" aria-live="polite">', INDEX_HTML)
        self.assertLess(INDEX_HTML.index('id="projectStats"'), INDEX_HTML.index('id="step1"'))
        self.assertIn(".hero-flow { display:grid; grid-template-columns:minmax(560px,1fr) minmax(420px,.8fr)", INDEX_HTML)
        self.assertIn(".stat-card { min-height:78px", INDEX_HTML)
        self.assertIn("background:#fff; border:1px solid #e3e7df", INDEX_HTML)
        self.assertIn("box-shadow:0 5px 14px", INDEX_HTML)
        self.assertNotIn("['gift', 'Donated Sessions'", INDEX_HTML)
        self.assertNotIn('<div class="stat-card"><div class="stat-icon" data-icon="gift"', INDEX_HTML)

    def test_donor_summary_distinguishes_unchecked_relay_status(self):
        self.assertIn("function relayStatusChecked()", INDEX_HTML)
        self.assertIn("Donation status could not be checked with the relay", INDEX_HTML)
        self.assertNotIn("Donated ?", INDEX_HTML)

    def test_pick_session_shows_public_dataset_composition(self):
        self.assertIn("datasetComposition", INDEX_HTML)
        self.assertIn("renderDatasetComposition", INDEX_HTML)
        self.assertIn("Dataset Composition", INDEX_HTML)
        self.assertIn("Breakdown of key public coverage metrics", INDEX_HTML)
        self.assertIn("composition-track", INDEX_HTML)
        self.assertIn("Institutes", INDEX_HTML)
        self.assertIn("Total turns", INDEX_HTML)
        self.assertNotIn("compositionMetric('Agents'", INDEX_HTML)
        self.assertNotIn("compositionMetric('Models'", INDEX_HTML)
        self.assertNotIn("compositionMetric('Ctx cmp'", INDEX_HTML)
        self.assertNotIn("Coverage radar chart", INDEX_HTML)
        self.assertNotIn("renderCoverageRadar", INDEX_HTML)

    def test_auto_repair_removes_detect_secrets_value(self):
        with TemporaryDirectory() as td:
            path = Path(td) / "session.redacted.jsonl"
            secret = "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY"
            path.write_text(
                f'{{"text":"-----BEGIN PRIVATE KEY----- {secret} -----END PRIVATE KEY-----"}}\n',
                encoding="utf-8",
            )
            report = verify_session(path)
            events = []

            repaired_report, stats, passes = _auto_repair_until_verified(path, report, {}, emit=events.append)
            repaired_text = path.read_text(encoding="utf-8")

        self.assertGreaterEqual(passes, 1)
        self.assertTrue(repaired_report["passed"])
        self.assertGreaterEqual(stats.get("scrub_term", 0), 1)
        self.assertGreaterEqual(stats.get("credential_pattern", 0), 1)
        self.assertFalse(any(key.startswith("private_word:") for key in stats))
        self.assertIn("redacting residual private patterns", " ".join(str(e.get("message", "")) for e in events))
        self.assertIn("Verifying after auto-repair", " ".join(str(e.get("message", "")) for e in events))
        self.assertNotIn(secret, repaired_text)

    def test_auto_repair_wraps_malformed_jsonl(self):
        with TemporaryDirectory() as td:
            path = Path(td) / "session.redacted.jsonl"
            path.write_text('{"text":"bad escape \\s but already <REDACTED>"}\n', encoding="utf-8")
            report = verify_session(path)
            events = []

            repaired_report, stats, passes = _auto_repair_until_verified(path, report, {}, emit=events.append)
            repaired_lines = path.read_text(encoding="utf-8").splitlines()

        self.assertGreaterEqual(passes, 1)
        self.assertTrue(repaired_report["passed"])
        self.assertEqual(stats.get("malformed_jsonl_wrapped"), 1)
        self.assertEqual(len(repaired_lines), 1)
        self.assertIn('"type":"redacted_raw_line"', repaired_lines[0])
        self.assertIn("normalizing malformed redacted JSONL lines", " ".join(str(e.get("message", "")) for e in events))

    def test_parse_redacted_donor_sessions_from_readme(self):
        self.assertEqual(_parse_donated_sessions("3 redacted donor sessions"), 3)
        self.assertEqual(_parse_donated_sessions("1,234 donated sessions"), 1234)
        self.assertIsNone(_parse_donated_sessions("no donation count here"))

    def test_parse_contributor_leaderboard_stops_before_session_ledger(self):
        rows = _parse_contributor_leaderboard(
            "\n".join([
                "| Rank | Contributor | Sessions | Turns | Agents | Models | Points |",
                "|:----:|-------------|:--------:|------:|--------|--------|:------:|",
                "| 🥇 | Founding donors | 3 | 18,380 | Claude Code | Opus 4.x | — |",
                "",
                "| ID | Agent / Harness | Model | Org | Domain | Language | Turns |",
                "|----|-----------------|-------|-----|--------|----------|------:|",
                "| S1 | Claude Code | Opus | Anthropic | coding | Python | 9,716 |",
            ])
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["contributor"], "Founding donors")
        self.assertEqual(rows[0]["sessions_num"], 3)
        self.assertEqual(rows[0]["turns_num"], 18380)

    def test_parse_contributor_leaderboard_keeps_all_ranked_rows(self):
        lines = [
            "| Rank | Contributor | Sessions | Turns | Agents | Models | Points |",
            "|:----:|-------------|:--------:|------:|--------|--------|:------:|",
        ]
        lines.extend(
            f"| {i} | Donor {i} | 1 | {1000 - i} | Claude Code | Opus | 5 |"
            for i in range(1, 10)
        )
        rows = _parse_contributor_leaderboard("\n".join(lines))
        self.assertEqual(len(rows), 9)
        self.assertEqual(rows[-1]["contributor"], "Donor 9")

    def test_parse_dataset_card_coverage(self):
        coverage = _parse_dataset_card_coverage(
            "\n".join([
                "## Dataset Summary",
                "| Field | Value |",
                "|-------|-------|",
                "| Active public/candidate sessions tracked locally | 5 |",
                "| Active public/candidate user turns tracked locally | 20,380 |",
                "| Active public/candidate context compactions tracked locally | 21 |",
                "| Public contributors in leaderboard | 4 |",
                "",
                "## Composition",
                "| Axis | Values |",
                "|------|--------|",
                "| Agent / harness | Claude Code (3), Codex CLI (2) |",
                "| Model family | Opus 4.x (mixed) (3), GPT-5 (2) |",
                "| Model organization | Anthropic (3), OpenAI (2) |",
                "| Task domain | agentic-coding (4), research (1) |",
                "| Primary language | Python (3), mixed (2) |",
                "| Public contributor institutions | UC Merced (2), independent (1) |",
            ])
        )
        self.assertEqual(coverage["sessions"], 5)
        self.assertEqual(coverage["contributors"], 4)
        self.assertEqual(coverage["institutions"], 2)
        self.assertEqual(coverage["agents"], 2)
        self.assertEqual(coverage["models"], 2)
        self.assertEqual(coverage["compactions"], 21)
        self.assertEqual(coverage["turns"], 20380)

    def test_load_contributors_markdown_falls_back_to_github_when_packaged(self):
        remote_text = "| Rank | Contributor | Sessions | Turns | Agents | Models | Points |\n"
        with TemporaryDirectory() as td:
            missing = Path(td) / "CONTRIBUTORS.md"
            with mock.patch("donate.web._fetch_text", return_value=remote_text) as fetch_text:
                text = _load_contributors_markdown(missing)

        self.assertEqual(text, remote_text)
        fetch_text.assert_called_once_with(
            "https://raw.githubusercontent.com/Accenture/ContextEcho/main/CONTRIBUTORS.md"
        )

    def test_load_contributors_markdown_prefers_live_github_by_default(self):
        remote_text = "| Rank | Contributor | Sessions | Turns | Agents | Models | Points |\n"
        with mock.patch("donate.web._fetch_text", return_value=remote_text) as fetch_text:
            text = _load_contributors_markdown()

        self.assertEqual(text, remote_text)
        fetch_text.assert_called_once_with(
            "https://raw.githubusercontent.com/Accenture/ContextEcho/main/CONTRIBUTORS.md"
        )

    def test_create_server_falls_back_when_port_is_busy(self):
        fake_server = mock.Mock()
        fake_server.server_address = ("127.0.0.1", 8767)
        busy = OSError(errno.EADDRINUSE, "address in use")
        with mock.patch("donate.web.ThreadingHTTPServer", side_effect=[busy, fake_server]) as server_cls:
            server, actual_port = create_server("127.0.0.1", 8766, attempts=3)

        self.assertIs(server, fake_server)
        self.assertEqual(actual_port, 8767)
        self.assertEqual(server_cls.call_args_list[0].args[0], ("127.0.0.1", 8766))
        self.assertEqual(server_cls.call_args_list[1].args[0], ("127.0.0.1", 8767))

    def test_annotate_donated_ignores_local_source_key_status(self):
        path = "/tmp/example-session.jsonl"
        with mock.patch("donate.web.load_donated_keys", return_value={session_key(path)}):
            rows = annotate_donated([{"path": path}, {"path": "/tmp/other.jsonl"}])
        self.assertFalse(rows[0]["donated"])
        self.assertFalse(rows[1]["donated"])

    def test_session_update_ready_uses_relay_growth_threshold(self):
        self.assertFalse(session_update_ready(119, 100))
        self.assertTrue(session_update_ready(120, 100))
        self.assertTrue(session_update_ready(150, 120))

    def test_annotate_donated_ignores_local_source_update_ready(self):
        path = "/tmp/example-session.jsonl"
        record = {"source_path_key": "unused", "turns": 100}
        with (
            mock.patch("donate.web.load_donated_keys", return_value=set()),
            mock.patch("donate.web.load_donated_source_records", return_value={"path-key": record}),
            mock.patch("donate.web.source_path_key", return_value="path-key"),
        ):
            rows = annotate_donated([{"path": path, "turns": 130}])

        self.assertFalse(rows[0]["donated"])
        self.assertFalse(rows[0]["donated_before"])
        self.assertEqual(rows[0]["donated_turns"], 0)
        self.assertEqual(rows[0]["new_turns"], 0)
        self.assertFalse(rows[0]["update_ready"])

    def test_annotate_donated_ignores_local_source_submission_id(self):
        path = "/tmp/example-session.jsonl"
        record = {
            "source_path_key": "unused",
            "turns": 100,
            "submission": "pending/submission-local123/",
            "credit_name": "Local Donor",
            "contributor_email": "local@example.com",
            "institute": "Local Institute",
            "public_anonymous": True,
        }
        with (
            mock.patch("donate.web.load_donated_keys", return_value=set()),
            mock.patch("donate.web.load_donated_source_records", return_value={"path-key": record}),
            mock.patch("donate.web.source_path_key", return_value="path-key"),
        ):
            rows = annotate_donated([{"path": path, "turns": 110}])

        self.assertFalse(rows[0]["donated"])
        self.assertFalse(rows[0]["donated_before"])
        self.assertEqual(rows[0]["new_turns"], 0)
        self.assertFalse(rows[0]["update_ready"])
        self.assertEqual(rows[0]["relay_submission_id"], "")
        self.assertEqual(rows[0]["local_credit_name"], "")
        self.assertEqual(rows[0]["local_contributor_email"], "")
        self.assertEqual(rows[0]["local_institute"], "")
        self.assertFalse(rows[0]["local_public_anonymous"])

    def test_annotate_donated_clears_status_when_relay_not_received(self):
        path = "/tmp/example-session.jsonl"
        record = {"source_path_key": "unused", "turns": 100, "submission": "pending/submission-stale/"}
        with (
            mock.patch("donate.web.load_donated_keys", return_value=set()),
            mock.patch("donate.web.load_donated_source_records", return_value={"path-key": record}),
            mock.patch("donate.web.source_path_key", return_value="path-key"),
            mock.patch(
                "donate.web.relay_donation_status",
                return_value=[{"received": False, "update_ready": False, "new_turns": 0, "new_records": 0}],
            ),
        ):
            rows = annotate_donated([{"path": path, "turns": 110, "records": 200}])

        self.assertTrue(rows[0]["relay_checked"])
        self.assertFalse(rows[0]["donated"])
        self.assertFalse(rows[0]["donated_before"])
        self.assertEqual(rows[0]["relay_submission_id"], "")

    def test_annotate_donated_uses_relay_lineage_status(self):
        path = "/tmp/example-session.jsonl"
        with (
            mock.patch("donate.web.load_donated_keys", return_value=set()),
            mock.patch("donate.web.load_donated_source_records", return_value={}),
            mock.patch(
                "donate.web.relay_donation_status",
                return_value=[{
                    "received": True,
                    "turns": 100,
                    "new_turns": 10,
                    "update_ready": False,
                    "submission_id": "submission-old",
                    "credit_name": "Existing Donor",
                    "contributor_email": "donor@example.com",
                    "contributor_institute": "Existing Lab",
                    "public_anonymous": True,
                }],
            ),
        ):
            rows = annotate_donated([{"path": path, "turns": 110, "records": 200}])

        self.assertTrue(rows[0]["donated"])
        self.assertTrue(rows[0]["donated_before"])
        self.assertEqual(rows[0]["donated_turns"], 100)
        self.assertEqual(rows[0]["new_turns"], 10)
        self.assertTrue(rows[0]["relay_received"])
        self.assertEqual(rows[0]["relay_submission_id"], "submission-old")
        self.assertEqual(rows[0]["local_credit_name"], "Existing Donor")
        self.assertEqual(rows[0]["local_contributor_email"], "donor@example.com")
        self.assertEqual(rows[0]["local_institute"], "Existing Lab")
        self.assertTrue(rows[0]["local_public_anonymous"])

    def test_annotate_donated_does_not_use_public_session_id_for_support(self):
        path = "/tmp/example-session.jsonl"
        with (
            mock.patch("donate.web.load_donated_keys", return_value=set()),
            mock.patch("donate.web.load_donated_source_records", return_value={}),
            mock.patch(
                "donate.web.relay_donation_status",
                return_value=[{
                    "received": True,
                    "turns": 100,
                    "new_turns": 0,
                    "update_ready": False,
                    "submission_id": "public-session-raw_transcript",
                }],
            ),
        ):
            rows = annotate_donated([{"path": path, "turns": 180, "records": 300}])

        self.assertTrue(rows[0]["relay_received"])
        self.assertEqual(rows[0]["relay_submission_id"], "")
        self.assertEqual(rows[0]["relay_public_session_id"], "public-session-raw_transcript")
        self.assertFalse(rows[0]["update_ready"])

    def test_annotate_donated_uses_relay_update_ready_status(self):
        path = "/tmp/example-session.jsonl"
        with (
            mock.patch("donate.web.load_donated_keys", return_value=set()),
            mock.patch("donate.web.load_donated_source_records", return_value={}),
            mock.patch(
                "donate.web.relay_donation_status",
                return_value=[{"received": True, "turns": 100, "new_turns": 60, "update_ready": True}],
            ),
        ):
            rows = annotate_donated([{"path": path, "turns": 160, "records": 200}])

        self.assertFalse(rows[0]["donated"])
        self.assertTrue(rows[0]["donated_before"])
        self.assertTrue(rows[0]["update_ready"])
        self.assertEqual(rows[0]["new_turns"], 60)

    def test_save_donation_record_tracks_artifact_and_blocks_duplicates(self):
        with TemporaryDirectory() as td:
            registry = Path(td) / ".donated_sessions.json"
            source = Path(td) / "source.jsonl"
            artifact = Path(td) / "session.redacted.jsonl"
            source.write_text('{"type":"user"}\n')
            artifact.write_text('{"type":"user","message":"<PERSON>"}\n')

            with mock.patch("donate.web.DONATION_ROOT", Path(td)), mock.patch("donate.web.DONATION_REGISTRY", registry):
                save_donation_record(
                    source_path=source,
                    artifact_path=artifact,
                    output="[submit] submission  : pending/submission-abc12345/",
                )
                self.assertIn(artifact_key(artifact), load_donated_artifact_keys())
                self.assertTrue(already_submitted(source, artifact))
                self.assertTrue(already_submitted("", artifact))
                saved = json.loads(registry.read_text())
                self.assertEqual(saved["submissions"][0]["submission_id"], "submission-abc12345")

    def test_clear_donation_registry_removes_local_duplicate_memory(self):
        with TemporaryDirectory() as td:
            registry = Path(td) / ".donated_sessions.json"
            source = Path(td) / "source.jsonl"
            source.write_text('{"type":"user"}\n')

            with mock.patch("donate.web.DONATION_ROOT", Path(td)), mock.patch("donate.web.DONATION_REGISTRY", registry):
                save_donation_record(source_path=source)
                self.assertTrue(already_submitted(source))
                self.assertTrue(clear_donation_registry())
                self.assertFalse(already_submitted(source))
                self.assertFalse(clear_donation_registry())

    def test_clear_donation_record_removes_only_one_local_duplicate(self):
        with TemporaryDirectory() as td:
            registry = Path(td) / ".donated_sessions.json"
            source1 = Path(td) / "source1.jsonl"
            source2 = Path(td) / "source2.jsonl"
            artifact1 = Path(td) / "session1.redacted.jsonl"
            artifact2 = Path(td) / "session2.redacted.jsonl"
            for path in [source1, source2, artifact1, artifact2]:
                path.write_text('{"type":"user"}\n')

            with mock.patch("donate.web.DONATION_ROOT", Path(td)), mock.patch("donate.web.DONATION_REGISTRY", registry):
                save_donation_record(source1, artifact1, "[submit] Submission ID: submission-a")
                save_donation_record(source2, artifact2, "[submit] Submission ID: submission-b")

                self.assertTrue(clear_donation_record(source_path=source1, artifact_path=artifact1))
                self.assertFalse(already_submitted(source1, artifact1))
                self.assertTrue(already_submitted(source2, artifact2))
                self.assertFalse(clear_donation_record(source_path=source1, artifact_path=artifact1))

    def test_session_key_changes_when_source_log_changes(self):
        with TemporaryDirectory() as td:
            source = Path(td) / "source.jsonl"
            source.write_text('{"type":"user","message":"one"}\n')
            first = session_key(source)
            source.write_text('{"type":"user","message":"one"}\n{"type":"user","message":"two"}\n')
            second = session_key(source)

        self.assertNotEqual(first, second)

    def test_submit_auto_metadata_falls_back_to_source_path(self):
        with TemporaryDirectory() as td:
            root = Path(td)
            source = root / "source.jsonl"
            session = root / "session.redacted.jsonl"
            source.write_text('{"type":"user"}\n', encoding="utf-8")
            session.write_text('{"type":"user","message":"<PERSON>"}\n', encoding="utf-8")

            with mock.patch(
                "donate.web.discover_mod.inspect_session",
                return_value={"agent": "Codex CLI", "model": "gpt-5", "turns": 42},
            ) as inspect_session:
                auto = submit_auto_metadata({"source_path": str(source)}, session)

        self.assertEqual(auto["agent"], "Codex CLI")
        self.assertEqual(auto["model"], "gpt-5")
        self.assertEqual(auto["turns"], 42)
        inspect_session.assert_called_once_with(source)

    def test_metadata_for_redacted_artifact_refreshes_record_count(self):
        with TemporaryDirectory() as td:
            root = Path(td)
            session = root / "session.redacted.jsonl"
            session.write_text(
                '{"type":"user","message":"one"}\n'
                '{"type":"assistant","message":"two"}\n',
                encoding="utf-8",
            )
            original_auto = {"agent": "Claude Code", "model": "claude", "turns": 1, "records": 1, "compactions": 0}
            auto = metadata_for_redacted_artifact({"auto": original_auto}, session)

        self.assertEqual(auto["records"], 2)
        self.assertEqual(original_auto["records"], 1)

    def test_local_pending_summary_merges_full_identity(self):
        with TemporaryDirectory() as td:
            registry = Path(td) / ".donated_sessions.json"
            source1 = Path(td) / "source1.jsonl"
            source2 = Path(td) / "source2.jsonl"
            artifact1 = Path(td) / "session1.redacted.jsonl"
            artifact2 = Path(td) / "session2.redacted.jsonl"
            for path in [source1, source2, artifact1, artifact2]:
                path.write_text('{"type":"user"}\n')
            receipt = {
                "credit_name": "Xianzhong Ding",
                "contributor_email": "xding5@ucmerced.edu",
                "institute": "UC Merced",
                "turns": "18",
                "compactions": "0",
            }

            with mock.patch("donate.web.DONATION_ROOT", Path(td)), mock.patch("donate.web.DONATION_REGISTRY", registry):
                save_donation_record(source1, artifact1, "[submit] Submission ID: submission-a", receipt=receipt)
                save_donation_record(source2, artifact2, "[submit] Submission ID: submission-b", receipt=receipt)
                summary = local_pending_summary(receipt)

        self.assertEqual(summary["sessions"], 2)
        self.assertEqual(summary["points_low"], 4)
        self.assertEqual(summary["points_high"], 8)

    def test_write_receipt_records_submission_details(self):
        with TemporaryDirectory() as td:
            root = Path(td)
            session = root / "session.redacted.jsonl"
            manifest = root / "session.manifest.json"
            session.write_text('{"type":"user"}\n')
            manifest.write_text(
                '{"contributor":"donor","credit_name":"donor","contributor_email":"d@example.com",'
                '"public_anonymous":true,'
                '"agent":"Codex CLI","model":"gpt-5","turns":"42","records":"123","compactions":"1"}'
            )
            output = "\n".join([
                "[submit] target repo : contextecho2026/persona-drift-staging (private)",
                "[submit] submission  : pending/submission-abc12345/",
                "[submit]   session.redacted.jsonl -> pending/submission-abc12345/session.redacted.jsonl",
                "[submit] https://huggingface.co/datasets/x/y/discussions/1",
            ])
            receipt_path, receipt = write_receipt(session, "/tmp/source.jsonl", output)
            self.assertTrue(receipt_path.exists())
            self.assertEqual(receipt["submission"], "pending/submission-abc12345/")
            self.assertEqual(receipt["contributor_email"], "d@example.com")
            self.assertTrue(receipt["public_anonymous"])
            self.assertEqual(receipt["turns"], "42")
            self.assertEqual(receipt["records"], "123")
            text = receipt_path.read_text()
            self.assertIn("pending/submission-abc12345/", text)
            self.assertIn("User turns: 42", text)
            self.assertIn("Public leaderboard: anonymous", text)
            self.assertIn("Records: 123", text)
            self.assertIn("Context compactions: 1", text)
            self.assertNotIn("persona-drift-staging", text)
            self.assertNotIn("huggingface.co", text)

    def test_parse_submit_output(self):
        parsed = parse_submit_output(
            "[submit] target repo : repo (private)\n"
            "[submit] submission  : pending/submission-abc12345/\n"
            "[submit]   a -> pending/submission-abc12345/a\n"
            "[submit] https://example.com/pr\n"
        )
        self.assertEqual(parsed["submission"], "pending/submission-abc12345/")
        self.assertEqual(parsed["url"], "https://example.com/pr")
        self.assertEqual(parsed["uploads"][0]["source"], "a")

    def test_parse_relay_submit_output(self):
        parsed = parse_submit_output(
            "[submit] upload mode  : relay\n"
            "[submit] relay       : http://127.0.0.1:8088\n"
            "[submit]   session.redacted.jsonl     -> session.redacted.jsonl\n"
            "[submit] Submitted for maintainer review.\n"
            "[submit] Submission ID: submission-abc12345\n"
        )
        self.assertEqual(parsed["submission"], "submission-abc12345")
        self.assertEqual(parsed["uploads"][0]["source"], "session.redacted.jsonl")

    def test_duplicate_relay_submit_output_is_detected(self):
        self.assertTrue(is_duplicate_submit_output(
            '[submit] relay upload failed: HTTP 409 {"detail":"duplicate redacted session artifact"}'
        ))
        self.assertTrue(is_duplicate_submit_output(
            '[submit] relay upload failed: HTTP 409 {"detail":"same source session changed too little since prior submission"}'
        ))
        self.assertFalse(is_duplicate_submit_output("[submit] relay upload failed: HTTP 500"))

    def test_duplicate_submit_detail_extracts_relay_reason(self):
        output = (
            '[submit] relay upload failed: HTTP 409\n'
            '{"detail":"same source session changed too little since prior submission '
            '(turns +5, records +220; require >= 20% growth or >= 50 new turns)"}'
        )
        self.assertEqual(
            duplicate_submit_detail(output),
            "same source session changed too little since prior submission "
            "(turns +5, records +220; require >= 20% growth or >= 50 new turns)",
        )

    def test_friendly_submit_error_explains_missing_relay_or_token(self):
        msg = friendly_submit_error(
            "401 Client Error. Repository Not Found for url: "
            "https://huggingface.co/api/datasets/contextecho2026/persona-drift-staging"
        )
        self.assertIn("Upload is not configured for public donors yet", msg)
        self.assertIn("CONTEXTECHO_RELAY_URL", msg)


if __name__ == "__main__":
    unittest.main()
