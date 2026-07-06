import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from analysis.analyze_session_validation import summarize
from experiments.e18_session_validation.run import load_transcript_lines, position_plan


class SessionValidationTests(unittest.TestCase):
    def test_load_transcript_lines_reads_codex_response_items(self):
        with TemporaryDirectory() as td:
            path = Path(td) / "codex.redacted.jsonl"
            rows = [
                {"type": "session_meta", "payload": {"cwd": "/tmp/project"}},
                {
                    "type": "response_item",
                    "payload": {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "Please fix the parser."}],
                    },
                },
                {
                    "type": "response_item",
                    "payload": {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "I will update it."}],
                    },
                },
                {
                    "type": "response_item",
                    "payload": {
                        "type": "message",
                        "role": "user",
                        "content": "Run the test.",
                    },
                },
            ]
            path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

            parsed = load_transcript_lines(path)

        self.assertEqual([r["role"] for r in parsed], ["user", "assistant", "user"])
        self.assertEqual(parsed[0]["content"], "Please fix the parser.")
        self.assertEqual(parsed[1]["content"], "I will update it.")
        self.assertEqual(parsed[2]["content"], "Run the test.")
        self.assertEqual(len(position_plan(len(parsed), 3)), 3)

    def test_summary_rejects_invalid_judge_scores(self):
        with TemporaryDirectory() as td:
            root = Path(td)
            for arm in ("claude_session", "filler"):
                cell = root / "P00" / arm / "C01.json"
                cell.parent.mkdir(parents=True, exist_ok=True)
                score = -1 if arm == "filler" else 1
                cell.write_text(json.dumps({
                    "position": "P00",
                    "arm": arm,
                    "probe_id": "C01",
                    "score": score,
                }), encoding="utf-8")

            report = summarize(root, expected_positions=1, probes=1)

        self.assertEqual(report["invalid_scores"], 1)
        self.assertEqual(report["scored_cells"], 1)
        self.assertFalse(report["acceptable"])


if __name__ == "__main__":
    unittest.main()
