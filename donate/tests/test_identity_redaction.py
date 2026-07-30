"""Tests for the identity-seeded redaction engine.

SYNTHETIC identities only — no real donor data may ever appear in this file.
"""
from __future__ import annotations

import base64
import json
import random
import string
import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from donate import identity_redaction as ir
from donate.redact import pseudonym

NAME = "Jane Marisol Doereski"
EMAIL = "jane.doereski@synthetic-example.org"
INSTITUTE = "Fictional Institute of Synthetic Data"


@pytest.fixture()
def terms():
    return ir.build_identity_terms(NAME, EMAIL, INSTITUTE, extra_handles=["jdoereski42"])


# ---------------------------------------------------------------------------
# build_identity_terms
# ---------------------------------------------------------------------------

def test_terms_cover_expected_forms(terms):
    keys = set(terms)
    assert "jane marisol doereski" in keys       # full name
    assert "jane" in keys and "doereski" in keys  # parts (len>=3)
    assert "jane-doereski" in keys                # First-Last
    assert "jane.doereski" in keys                # first.last
    assert "janedoereski" in keys                 # firstlast / FirstLast (ci)
    assert EMAIL in keys                          # full email
    assert "jane.doereski" in keys                # email local-part
    assert "jdoereski42" in keys                  # handle
    assert "fictional institute of synthetic data" in keys
    assert "fictional" in keys                    # distinctive institute word
    # generic words never become terms
    assert "institute" not in keys
    assert "the" not in keys and "of" not in keys


def test_terms_replacements_follow_pipeline_scheme(terms):
    assert terms[EMAIL] == "<EMAIL>"
    assert terms["fictional institute of synthetic data"] == "<REDACTED>"
    # name-like terms use the SAME salted pseudonym scheme as donate/redact.py
    assert terms["doereski"] == pseudonym("doereski")
    assert terms["doereski"].startswith("<USER_")


def test_short_and_generic_parts_skipped():
    t = ir.build_identity_terms("Al Bo", "al@independent.com", "Independent")
    assert "al" not in t and "bo" not in t          # len < 3
    assert "independent" not in t                    # generic


# ---------------------------------------------------------------------------
# scrub_text — matching contexts
# ---------------------------------------------------------------------------

def test_git_log_author_line(terms):
    line = f"commit abc123\nAuthor: Jane Doereski <{EMAIL}>\nDate: today"
    out, counts = ir.scrub_text(line, terms)
    assert "Jane" not in out and "Doereski" not in out
    assert EMAIL not in out
    assert counts["identity_term"] + counts["structured:person_email"] > 0


def test_person_email_construct_with_unknown_nickname(terms):
    # Nickname is NOT in the term set; donor email adjacency identifies it.
    line = f"Co-authored-by: JayDee Nine <{EMAIL}>"
    out, _ = ir.scrub_text(line, terms)
    assert EMAIL not in out
    assert "JayDee Nine" not in out
    assert "<PERSON> <EMAIL>" in out


def test_users_path_segment(terms):
    out, _ = ir.scrub_text("cwd=/Users/doereski/projects/x.py", terms)
    assert "/Users/doereski/" not in out
    assert f"/Users/{pseudonym('doereski')}/projects/x.py" in out

    out2, _ = ir.scrub_text(r"C:\Users\doereski\code", terms)
    assert "doereski" not in out2.lower()


def test_slug_and_join_forms(terms):
    text = (
        "workspace hf.co/jane-doereski/persona and dir "
        "-Users-doereski-Library plus Jane-Doereski-submission-01 and "
        "user janedoereski pushed"
    )
    out, _ = ir.scrub_text(text, terms)
    assert "doereski" not in out.lower()


def test_email_local_part_as_workspace_name(terms):
    out, _ = ir.scrub_text("https://huggingface.co/jane.doereski/space", terms)
    assert "jane.doereski" not in out.lower()


def test_no_partial_word_match(terms):
    # 'jane' must not fire inside an alphanumeric word.
    out, counts = ir.scrub_text("the trajanesque column stands", terms)
    assert out == "the trajanesque column stands"
    assert counts["identity_term"] == 0


# ---------------------------------------------------------------------------
# high-entropy (blob) guard
# ---------------------------------------------------------------------------

def _blob_containing(word: str, side: int = 24) -> str:
    rng = random.Random(7)
    alphabet = string.ascii_letters + string.digits + "+/"
    left = "".join(rng.choice(alphabet) for _ in range(side))
    right = "".join(rng.choice(alphabet) for _ in range(side))
    return left + word + right


def test_base64_blob_false_positive_skipped(terms):
    # Name flanked by alphanumerics: the boundary rule already refuses it.
    blob_alnum = _blob_containing("jane")
    out, counts = ir.scrub_text(f"payload: {blob_alnum} end", terms)
    assert blob_alnum in out
    assert counts["identity_term"] == 0

    # Name flanked by base64 symbol chars (+ /): boundaries would fire, so the
    # high-entropy guard must catch it.
    blob_sym = _blob_containing("+jane/")
    out, counts = ir.scrub_text(f"payload: {blob_sym} end", terms)
    assert blob_sym in out                 # untouched
    assert counts["blob_guard_skips"] >= 1
    assert counts["identity_term"] == 0


def test_hex_blob_skipped(terms):
    blob = "9f8a7c" * 3 + "jane" + "0b1c2d" * 3  # hex-ish run, name interior
    # force pure-hex context around the name
    blob = "abcdef0123456789abcdef" + "0123" + "fedcba9876543210fedcba"
    text_with_name = blob[:20] + "jane" + blob[20:]
    out, counts = ir.scrub_text(text_with_name, terms)
    # 'jane' is strictly inside a >=24-char [A-Za-z0-9...] run of hex chars
    assert "jane" in out
    assert counts["identity_term"] == 0


def test_name_at_blob_edge_still_scrubbed(terms):
    # Slug that merely LOOKS long: name at the edge must still be scrubbed.
    out, _ = ir.scrub_text("dir Jane-Doereski-submission-4f3a9b7c1d2e distributed", terms)
    assert "doereski" not in out.lower()


def test_real_path_not_mistaken_for_blob(terms):
    out, _ = ir.scrub_text("/Users/doereski/Library/CloudStorage/OneDrive-Somewhere/x", terms)
    assert "doereski" not in out.lower()


# ---------------------------------------------------------------------------
# JSONL safety + idempotence
# ---------------------------------------------------------------------------

def test_jsonl_line_stays_valid_json(terms):
    obj = {
        "type": "user",
        "text": f'git log says Author: Jane Doereski <{EMAIL}> ok "quoted"',
        "nested": {"cwd": "/Users/doereski/proj", "n": 3, "flag": True,
                   "arr": ["Jane Marisol Doereski", 42, None]},
    }
    line = json.dumps(obj, ensure_ascii=False)
    out, counts = ir.scrub_jsonl_line(line, terms)
    parsed = json.loads(out)  # must not raise
    assert counts["identity_term"] > 0
    dumped = json.dumps(parsed)
    assert "doereski" not in dumped.lower()
    assert parsed["nested"]["n"] == 3 and parsed["nested"]["flag"] is True
    assert parsed["nested"]["arr"][1] == 42 and parsed["nested"]["arr"][2] is None


def test_json_keys_scrubbed_too(terms):
    # Real leak shape: Claude Code snapshot.trackedFileBackups keys are
    # absolute file paths that can carry the donor name.
    obj = {"snapshot": {"trackedFileBackups": {
        "/Users/doereski/proj/src/main.py": {"backupId": "b1"},
        "/Users/doereski/proj/README.md": {"backupId": "b2"},
    }}}
    line = json.dumps(obj)
    out, counts = ir.scrub_jsonl_line(line, terms)
    parsed = json.loads(out)
    assert "doereski" not in out.lower()
    backups = parsed["snapshot"]["trackedFileBackups"]
    assert len(backups) == 2                      # no entry silently dropped
    assert {b["backupId"] for b in backups.values()} == {"b1", "b2"}
    assert counts["identity_term"] >= 2


def test_json_key_collision_after_scrub(terms):
    obj = {"m": {"jane-a": 1, "Jane-a": 2}}      # collide once case-folded to pseudonym
    out, _ = ir.scrub_jsonl_line(json.dumps(obj), terms)
    parsed = json.loads(out)
    assert len(parsed["m"]) == 2                  # deterministic suffix keeps both
    assert sorted(parsed["m"].values()) == [1, 2]
    assert "jane" not in out.lower()


def test_non_matching_line_passes_through_unchanged(terms):
    line = '{"a": "no identity here", "b": [1, 2]}'
    out, counts = ir.scrub_jsonl_line(line, terms)
    assert out == line
    assert sum(counts.values()) == 0


def test_idempotence(terms):
    text = (
        f"Author: Jane Doereski <{EMAIL}>\n"
        "cwd /Users/doereski/x and hf.co/jane-doereski\n"
        "Your Action Items (Jane Marisol Doereski leads the rollout)."
    )
    once, c1 = ir.scrub_text(text, terms)
    twice, c2 = ir.scrub_text(once, terms)
    assert once == twice
    assert c2["identity_term"] == 0 and c2["structured:person_email"] == 0


def test_scrub_file_roundtrip(tmp_path, terms):
    src = tmp_path / "s.jsonl"
    rows = [
        {"type": "user", "text": f"Author: Jane Doereski <{EMAIL}>"},
        {"type": "assistant", "text": "no names at all"},
        {"type": "tool", "text": "path /Users/doereski/repo"},
    ]
    src.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    dst = tmp_path / "out.jsonl"
    counts = ir.scrub_file(src, dst, terms)
    assert counts["identity_term"] >= 1
    lines = dst.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    for l in lines:
        json.loads(l)
        assert "doereski" not in l.lower()
    # untouched line is byte-identical
    assert lines[1] == json.dumps(rows[1])


# ---------------------------------------------------------------------------
# find_identity_hits (verify-side matcher parity)
# ---------------------------------------------------------------------------

def test_find_hits_agrees_with_scrub(terms):
    text = f"Author: Jane Doereski <{EMAIL}> in /Users/doereski/ plus blob " + _blob_containing("jane")
    hits = ir.find_identity_hits(text, terms)
    assert sum(hits.values()) > 0
    assert "jane" in hits or "doereski" in hits
    scrubbed, _ = ir.scrub_text(text, terms)
    assert sum(ir.find_identity_hits(scrubbed, terms).values()) == 0


def test_mask_term_never_leaks():
    assert ir.mask_term("doereski") == "do***(8 chars)"
    masked = ir.mask_term("jane.doereski@synthetic-example.org")
    assert masked.startswith("ja***@")
    assert "doereski" not in masked


# ---------------------------------------------------------------------------
# third-party person detection (review-only)
# ---------------------------------------------------------------------------

PROSE = (
    "Here is the meeting summary you asked for. Your Action Items are as follows. "
    "Carlos Ventimiglia agreed to review the deployment pipeline by Friday. "
    "Please sync with Priya Ramachandran about the vendor contract before then. "
    "The rest of the notes cover scheduling and are not urgent for this sprint. "
    "Overall the team felt the milestone was achievable within the quarter."
)

CODE = (
    "def summarize(items):\n    return {k: v for k, v in items.items() if v}\n"
    "# Carlos Ventimiglia wrote this helper\n" + "x = [i ** 2 for i in range(100)]\n" * 10
)


def test_third_party_persons_found_in_prose(terms):
    result = ir.find_third_party_persons(PROSE, exclude_terms=terms)
    names = {c["name"] for c in result["candidates"]}
    assert any("Ventimiglia" in n for n in names)
    assert any("Ramachandran" in n for n in names)
    for c in result["candidates"]:
        assert c["count"] >= 1 and c["snippets"]


def test_third_party_skips_code_and_short_fields(terms):
    assert ir.find_third_party_persons(CODE, exclude_terms=terms)["candidates"] == []
    assert ir.find_third_party_persons("Short note from Carlos Ventimiglia.", exclude_terms=terms)["candidates"] == []


def test_third_party_excludes_donor(terms):
    text = PROSE + " Finally, Jane Doereski signed off on the plan as discussed. "
    names = {c["name"] for c in ir.find_third_party_persons(text, exclude_terms=terms)["candidates"]}
    assert not any("Doereski" in n for n in names)


def test_third_party_never_modifies():
    # API returns candidates only; scrubbing is a separate, human-approved step.
    result = ir.find_third_party_persons(PROSE)
    assert set(result) == {"method", "engine_version", "candidates"}


# ---------------------------------------------------------------------------
# structured git-config rule
# ---------------------------------------------------------------------------

def test_git_config_email_line(terms):
    out, counts = ir.scrub_text(f"user.email={EMAIL}\nuser.name=Jane Doereski", terms)
    assert EMAIL not in out
    assert "doereski" not in out.lower()
