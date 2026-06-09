import unittest
from unittest import mock

from donate.web import _parse_donated_sessions, annotate_donated, session_key


class WebTests(unittest.TestCase):
    def test_parse_redacted_donor_sessions_from_readme(self):
        self.assertEqual(_parse_donated_sessions("3 redacted donor sessions"), 3)
        self.assertEqual(_parse_donated_sessions("1,234 donated sessions"), 1234)
        self.assertIsNone(_parse_donated_sessions("no donation count here"))

    def test_annotate_donated_marks_known_source_key(self):
        path = "/tmp/example-session.jsonl"
        with mock.patch("donate.web.load_donated_keys", return_value={session_key(path)}):
            rows = annotate_donated([{"path": path}, {"path": "/tmp/other.jsonl"}])
        self.assertTrue(rows[0]["donated"])
        self.assertFalse(rows[1]["donated"])


if __name__ == "__main__":
    unittest.main()
