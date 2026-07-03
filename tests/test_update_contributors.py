import unittest

from scripts.update_contributors import (
    SessionEntry,
    anonymous_ledger_name,
    accepted_submission_ids,
    display_name,
    group_contributors,
    render_project_stats,
    score_sessions,
)


class UpdateContributorsTests(unittest.TestCase):
    def test_anonymous_sessions_without_private_identity_do_not_merge(self):
        sessions = [
            SessionEntry(sid="S4", contributor="Anonymous donor S4", agent="Codex", model="gpt", org="OpenAI", domain="coding", language="Python", turns=100, compactions=1, source_key="a"),
            SessionEntry(sid="S5", contributor="Anonymous donor S5", agent="Codex", model="gpt", org="OpenAI", domain="coding", language="Python", turns=100, compactions=1, source_key="b"),
        ]
        score_sessions(sessions)
        contributors = group_contributors(sessions)
        self.assertEqual(len(contributors), 2)
        self.assertEqual({c.name for c in contributors}, {"Anonymous donor S4", "Anonymous donor S5"})

    def test_public_anonymous_sessions_with_same_private_identity_merge(self):
        sessions = [
            SessionEntry(sid="S4", contributor="Anonymous donor S4", identity_name="Dana Contributor", email="d@example.com", institute="Lab", public_anonymous=True, agent="Codex", model="gpt", org="OpenAI", domain="coding", language="Python", turns=100, compactions=1, source_key="a"),
            SessionEntry(sid="S5", contributor="Anonymous donor S5", identity_name="Dana Contributor", email="d@example.com", institute="Lab", public_anonymous=True, agent="Claude", model="opus", org="Anthropic", domain="docs", language="mixed", turns=200, compactions=1, source_key="b"),
        ]
        score_sessions(sessions)
        contributors = group_contributors(sessions)
        self.assertEqual(len(contributors), 1)
        self.assertEqual(contributors[0].name, "Anonymous donor S4")
        self.assertEqual(len(contributors[0].counted_sessions), 2)
        self.assertGreater(contributors[0].points, 5)

    def test_manual_contributor_group_merges_different_private_identity_fields(self):
        sessions = [
            SessionEntry(sid="S4", contributor="Anonymous donor controlled", contributor_group="Anonymous donor controlled", identity_name="Dana Contributor", email="d@example.com", institute="Lab", public_anonymous=True, agent="Codex", model="gpt", org="OpenAI", domain="coding", language="Python", turns=100, compactions=1, source_key="a"),
            SessionEntry(sid="S5", contributor="Anonymous donor controlled", contributor_group="Anonymous donor controlled", identity_name="Dana C.", email="other@example.com", institute="Other Lab", public_anonymous=True, agent="Claude", model="opus", org="Anthropic", domain="docs", language="mixed", turns=200, compactions=1, source_key="b"),
        ]
        score_sessions(sessions)
        contributors = group_contributors(sessions)
        self.assertEqual(len(contributors), 1)
        self.assertEqual(contributors[0].name, "Anonymous donor controlled")
        self.assertEqual(len(contributors[0].counted_sessions), 2)

    def test_future_anonymous_donor_uses_submission_id(self):
        fallback = anonymous_ledger_name({"submission_id": "submission-d51e3f33"}, "S4")
        self.assertEqual(display_name({"credit_name": "anonymous"}, fallback), "Anonymous donor d51e3f33")

    def test_public_anonymous_donor_uses_submission_id(self):
        fallback = anonymous_ledger_name({"submission_id": "submission-d51e3f33"}, "S4")
        self.assertEqual(display_name({"credit_name": "Named Donor", "public_anonymous": True}, fallback), "Anonymous donor d51e3f33")

    def test_matching_name_email_and_institute_merge(self):
        sessions = [
            SessionEntry(sid="S4", contributor="Dana Contributor", email="d@example.com", institute="Lab", agent="Codex", model="gpt", org="OpenAI", domain="coding", language="Python", turns=100, compactions=1, source_key="a"),
            SessionEntry(sid="S5", contributor="Dana Contributor", email="d@example.com", institute="Lab", agent="Claude", model="opus", org="Anthropic", domain="docs", language="mixed", turns=200, compactions=1, source_key="b"),
        ]
        score_sessions(sessions)
        contributors = group_contributors(sessions)
        self.assertEqual(len(contributors), 1)
        self.assertEqual(len(contributors[0].counted_sessions), 2)

    def test_duplicate_source_variant_does_not_double_count_even_with_different_identity(self):
        sessions = [
            SessionEntry(sid="S4", contributor="Dana Contributor", email="d@example.com", institute="Lab", agent="Codex", model="gpt", org="OpenAI", domain="coding", language="Python", turns=100, compactions=1, source_key="same"),
            SessionEntry(sid="S5", contributor="Dana Contributor", email="other@example.com", institute="Lab", agent="Codex", model="gpt", org="OpenAI", domain="coding", language="Python", turns=100, compactions=1, source_key="same"),
        ]
        score_sessions(sessions)
        contributors = group_contributors(sessions)
        self.assertEqual(len(contributors), 1)
        self.assertEqual(len(contributors[0].counted_sessions), 1)
        self.assertFalse(sessions[1].counted)
        self.assertEqual(sessions[1].points, 0)

    def test_project_stats_preserves_downloads_and_publishes_accepted_submission_ids(self):
        sessions = [
            SessionEntry(sid="S4", contributor="Dana Contributor", submission_id="submission-b", source_key="b"),
            SessionEntry(sid="S5", contributor="Dana Contributor", submission_id="submission-a", source_key="a"),
            SessionEntry(sid="S6", contributor="Founding", source_key="founding"),
        ]
        self.assertEqual(accepted_submission_ids(sessions), ["submission-a", "submission-b"])

        rendered = render_project_stats({"dataset_total_downloads": 47350}, sessions)

        self.assertIn('"dataset_total_downloads": 47350', rendered)
        self.assertIn('"accepted_submission_count": 2', rendered)
        self.assertIn('"submission-a"', rendered)
        self.assertIn('"submission-b"', rendered)


if __name__ == "__main__":
    unittest.main()
