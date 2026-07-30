"""ContextEcho donation — identity-seeded redaction engine.

Why this exists (audit 2026-07-28): the generic redaction pipeline
(`donate/redact.py`) misses donor identity strings that appear in *content*
rather than in path/email/credential shapes — git-log `Author:` lines,
`/Users/<name>/` cwd segments, workspace/org slugs (e.g. an email local-part
reused as a Hugging Face workspace name), and model-written prose that names
the donor. The consent form already collects the donor's name, email, and
institute; this module turns those fields into scrub input.

Design:
  * `build_identity_terms()` expands the consent fields into the term set a
    donor's identity actually leaks as (name parts, First-Last / first.last /
    firstlast joins, email local-part, institute short forms, handles), each
    mapped to a replacement consistent with the existing pipeline (salted
    `<USER_xxxxxx>` pseudonyms via `donate.redact.pseudonym`, `<EMAIL>`,
    `<REDACTED>`).
  * `scrub_text()` matches case-insensitively on word boundaries that INCLUDE
    path/slug separators (`/Users/<name>/`, `C:\\Users\\<name>\\`,
    `<name>-submission`, dots/dashes/underscores), with a high-entropy guard so
    short names strictly inside base64/hex blobs do not false-positive.
  * `scrub_jsonl_line()` parses each JSONL line and scrubs inside string values
    only, so JSON validity is preserved by construction.
  * `find_third_party_persons()` surfaces PERSON candidates in prose-like
    values for HUMAN review — it never auto-scrubs.

The identity terms are runtime-only scrub input: they must never be written to
any repo file, manifest stat key, log, or report. Callers that report counts
must aggregate (see `mask_term`).
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

ENGINE_VERSION = "identity-redaction-v1"

# ---------------------------------------------------------------------------
# Pseudonyms — reuse the exact salted scheme of donate/redact.py so one donor
# maps to ONE stable pseudonym across both engines. Lazy import avoids a
# circular import (redact.py imports this module).
# ---------------------------------------------------------------------------


def _pseudonym(term: str) -> str:
    try:
        from donate.redact import pseudonym  # type: ignore
    except ImportError:  # direct invocation from inside the package dir
        from redact import pseudonym  # type: ignore
    return pseudonym(term)


# Words that are too generic to be identity evidence on their own. Includes
# institute boilerplate and common English words that double as name parts.
GENERIC_TERMS = {
    # articles / conjunctions / stop-ish words
    "the", "and", "for", "von", "van", "der", "den", "del", "does", "not",
    "one", "two", "new", "old", "all", "any", "may", "can", "will", "did",
    "you", "our", "who", "how", "get", "set", "run", "use", "his", "her",
    # institute / org boilerplate
    "university", "institute", "institution", "college", "school", "labs",
    "lab", "inc", "llc", "ltd", "gmbh", "corp", "corporation", "company",
    "co", "group", "team", "dept", "department", "center", "centre",
    "research", "science", "sciences", "technology", "technologies", "tech",
    "national", "state", "independent", "personal", "none", "n/a", "self",
    "freelance", "student", "engineering", "computer", "consulting",
    # generic account words
    "user", "users", "admin", "root", "test", "demo", "info", "mail",
    "email", "contact", "hello", "anonymous", "unknown", "home", "shared",
    "public", "example",
}

_SPLIT_RE = re.compile(r"[\s\-_.]+")
_LOCAL_SPLIT_RE = re.compile(r"[._\-+]+")


def _clean(term: str) -> str:
    return (term or "").strip().strip("\"'`,;:()[]{}")


def build_identity_terms(
    name: str,
    email: str = "",
    institute: str = "",
    extra_handles: Iterable[str] = (),
) -> dict[str, str]:
    """Expand donor consent fields into {lowercased term -> replacement}.

    Replacements follow the existing pipeline: name-like terms -> salted
    stable pseudonym; email -> <EMAIL>; institute -> <REDACTED>.
    """
    terms: dict[str, str] = {}

    def add(term: str, replacement: str | None = None, min_len: int = 3) -> None:
        t = _clean(term)
        if len(t) < min_len:
            return
        low = t.lower()
        if low in GENERIC_TERMS or low.startswith("<"):
            return
        terms.setdefault(low, replacement if replacement is not None else _pseudonym(low))

    full = _clean(name)
    if full:
        add(full)
        parts = [p for p in _SPLIT_RE.split(full) if p]
        for part in parts:
            add(part)  # name parts, len >= 3
        if len(parts) >= 2:
            first, last = parts[0], parts[-1]
            # Common join forms a name leaks as in slugs/usernames/workspaces:
            # First-Last, first.last, first_last, firstlast (and full joins).
            for join in ("-", ".", "_", ""):
                add(join.join(parts))
                add(f"{first}{join}{last}")

    em = _clean(email).lower()
    if em and "@" in em:
        add(em, "<EMAIL>")
        local = em.split("@", 1)[0]
        add(local)  # email local-part (often reused as a workspace/handle)
        for piece in _LOCAL_SPLIT_RE.split(local):
            add(piece, min_len=4)

    inst = _clean(institute)
    if inst and inst.lower() not in GENERIC_TERMS:
        add(inst, "<REDACTED>")
        words = [w for w in re.split(r"[\s,]+", inst) if w]
        if len(words) >= 2:
            acronym = "".join(w[0] for w in words if w and w[0].isalpha())
            add(acronym, "<REDACTED>")  # obvious short form (e.g. initials)
        for w in words:
            add(w, "<REDACTED>", min_len=5)

    for handle in extra_handles or ():
        add(handle)

    return terms


# ---------------------------------------------------------------------------
# Matching. Boundaries: anything that is not a letter/digit counts as a
# boundary, so matches fire inside `/Users/<name>/`, `C:\Users\<name>\`,
# `<name>-submission`, `first.last`, `-Users-<name>-...` slugs, etc.
# ---------------------------------------------------------------------------

_BOUNDARY_L = r"(?<![A-Za-z0-9])"
_BOUNDARY_R = r"(?![A-Za-z0-9])"

# A base64/hex-looking run: >=24 chars of the blob alphabet, no spaces.
# NOTE: '.' is NOT in the alphabet, so dotted names/paths break runs.
_BLOB_RUN_RE = re.compile(r"[A-Za-z0-9+/=_\-]{24,}")
_HEX_RE = re.compile(r"[0-9a-fA-F]{24,}")
_BLOB_INTERIOR_MARGIN = 6  # term must sit >= this many chars inside the run


def _shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = Counter(s)
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def _looks_like_blob(run: str) -> bool:
    """True for base64/hex-like high-entropy runs; False for paths/slugs."""
    if len(run) < 24 or " " in run:
        return False
    if _HEX_RE.fullmatch(run):
        return True
    # Path-like runs contain many '/' separators; base64 uses '/' sparsely.
    if run.count("/") > max(1, len(run) // 20):
        return False
    digits = sum(c.isdigit() for c in run)
    if digits / len(run) < 0.08:
        return False
    # Base64 is case-balanced; human slugs (First-Last-submission-...) are
    # mostly lowercase with sparse capitals.
    upper = sum(c.isupper() for c in run)
    lower = sum(c.islower() for c in run)
    letters = upper + lower
    if letters and min(upper, lower) / letters < 0.25:
        return False
    return _shannon_entropy(run) >= 4.2


def _blob_spans(text: str) -> list[tuple[int, int]]:
    return [m.span() for m in _BLOB_RUN_RE.finditer(text) if _looks_like_blob(m.group(0))]


def _inside_blob(spans: list[tuple[int, int]], start: int, end: int) -> bool:
    for s, e in spans:
        if s + _BLOB_INTERIOR_MARGIN <= start and end <= e - _BLOB_INTERIOR_MARGIN:
            return True
        if s > end:
            break
    return False


# Compiled-alternation cache: terms_maps are built once per donor/file and
# then used for millions of string values, so recompiling per call would
# dominate runtime. Keyed by id() with a length sanity check (maps are not
# mutated after build).
_RX_CACHE: dict[int, tuple[int, re.Pattern[str] | None]] = {}


def compile_terms(terms_map: dict[str, str]) -> re.Pattern[str] | None:
    """One alternation over all terms, longest-first, boundary-guarded."""
    if not terms_map:
        return None
    cached = _RX_CACHE.get(id(terms_map))
    if cached is not None and cached[0] == len(terms_map):
        return cached[1]
    alternatives = sorted(terms_map, key=len, reverse=True)
    body = "|".join(re.escape(t) for t in alternatives)
    rx = re.compile(f"{_BOUNDARY_L}(?:{body}){_BOUNDARY_R}", re.IGNORECASE)
    _RX_CACHE[id(terms_map)] = (len(terms_map), rx)
    return rx


# --- Structured-context rules ----------------------------------------------

# `Name Surname <email>` constructs (git log Author:/Committer:/Co-authored-by,
# changelog credits, package metadata). Scrubbed when the email is the donor's.
_PERSON_EMAIL_RE = re.compile(
    r"(?P<name>[A-Za-z][\w'\-]*(?:[ \t]+[A-Za-z][\w'\-]*){0,3})[ \t]*"
    r"<(?P<email>[^<>\s]+@[^<>\s]+)>"
)
# git-config style assignments: user.name=..., user.email=..., "name": "..."
_GIT_CONFIG_RE = re.compile(
    r"(?im)^(?P<prefix>\s*(?:user\.(?:name|email)|author(?:_name|_email)?|committer(?:_name|_email)?)"
    r"\s*[=:]\s*[\"']?)(?P<value>[^\"'\r\n]+)"
)


def _donor_emails(terms_map: dict[str, str]) -> set[str]:
    return {t for t in terms_map if "@" in t}


def _scrub_structured(text: str, terms_map: dict[str, str], counts: Counter) -> str:
    emails = _donor_emails(terms_map)
    if not emails:
        return text

    def person_email_repl(m: re.Match[str]) -> str:
        if m.group("email").lower() in emails:
            counts["structured:person_email"] += 1
            return "<PERSON> <EMAIL>"
        return m.group(0)

    if "<" in text and "@" in text:
        text = _PERSON_EMAIL_RE.sub(person_email_repl, text)

    def git_config_repl(m: re.Match[str]) -> str:
        value = m.group("value").strip().lower()
        if value in emails:
            counts["structured:git_config"] += 1
            return m.group("prefix") + "<EMAIL>"
        return m.group(0)

    lowered = text.lower()
    if "user.name" in lowered or "user.email" in lowered or "author" in lowered or "committer" in lowered:
        text = _GIT_CONFIG_RE.sub(git_config_repl, text)
    return text


def scrub_text(text: str, terms_map: dict[str, str], counts: Counter | None = None) -> tuple[str, Counter]:
    """Scrub donor identity terms from a plain text string.

    Case-insensitive, path-segment-aware boundaries, blob guard. Returns the
    scrubbed text and a Counter of replacement counts. Counter keys never
    contain the terms themselves (aggregate keys only), so counts are safe to
    log/report.
    """
    if counts is None:
        counts = Counter()
    rx = compile_terms(terms_map)
    if rx is None or not text:
        return text, counts

    # Fast path: no candidate at all (structured rules key off donor emails,
    # which are themselves terms, so the alternation covers them).
    if not rx.search(text):
        return text, counts

    # Structured pass FIRST: it needs the donor email still intact to decide
    # whether an adjacent free-form name (possibly not in the term set, e.g. a
    # git-config nickname) belongs to the donor.
    text = _scrub_structured(text, terms_map, counts)

    spans = _blob_spans(text)

    def repl(m: re.Match[str]) -> str:
        if spans and _inside_blob(spans, m.start(), m.end()):
            counts["blob_guard_skips"] += 1
            return m.group(0)
        replacement = terms_map.get(m.group(0).lower(), "<REDACTED>")
        counts["identity_term"] += 1
        return replacement

    text = rx.sub(repl, text)
    return text, counts


# ---------------------------------------------------------------------------
# JSONL-safe scrubbing: parse each line, walk string values, re-serialize.
# ---------------------------------------------------------------------------


def scrub_json_value(value: Any, terms_map: dict[str, str], counts: Counter) -> Any:
    if isinstance(value, str):
        scrubbed, _ = scrub_text(value, terms_map, counts)
        return scrubbed
    if isinstance(value, list):
        return [scrub_json_value(v, terms_map, counts) for v in value]
    if isinstance(value, dict):
        # Keys are scrubbed too: coding-agent logs use FILE PATHS as dict keys
        # (e.g. Claude Code snapshot.trackedFileBackups), and those paths can
        # carry the donor username/name. Post-scrub key collisions get a
        # deterministic suffix so no entry is silently dropped.
        out: dict = {}
        for k, v in value.items():
            new_k = k
            if isinstance(k, str):
                new_k, _ = scrub_text(k, terms_map, counts)
                if new_k != k and new_k in out:
                    base, n = new_k, 2
                    while new_k in out:
                        new_k = f"{base}~{n}"
                        n += 1
            out[new_k] = scrub_json_value(v, terms_map, counts)
        return out
    return value


def scrub_jsonl_line(line: str, terms_map: dict[str, str], counts: Counter | None = None) -> tuple[str, Counter]:
    """Scrub one JSONL line without breaking JSON validity.

    The line is parsed and only string values are scrubbed; key order and
    non-string values are preserved. Non-JSON lines fall back to raw-text
    scrubbing (the replacements contain no JSON metacharacters).
    """
    if counts is None:
        counts = Counter()
    stripped = line.rstrip("\r\n")
    if not stripped.strip():
        return line, counts
    rx = compile_terms(terms_map)
    if rx is None:
        return line, counts
    # Fast pre-check on the raw line: skip the JSON parse when nothing can
    # match. Structured rules only fire on donor emails, which are terms in
    # the map, so the alternation search covers them too.
    if not rx.search(stripped):
        return line, counts
    try:
        obj = json.loads(stripped)
    except Exception:
        scrubbed, _ = scrub_text(stripped, terms_map, counts)
        return scrubbed + line[len(stripped):], counts
    scrubbed_obj = scrub_json_value(obj, terms_map, counts)
    out = json.dumps(scrubbed_obj, ensure_ascii=False, separators=(",", ":"))
    return out + line[len(stripped):], counts


def scrub_file(src: Path, dst: Path, terms_map: dict[str, str]) -> Counter:
    """Scrub a whole JSONL file line-by-line. src and dst may be the same path."""
    counts: Counter = Counter()
    out_lines: list[str] = []
    with src.open("r", encoding="utf-8", errors="replace") as fin:
        for raw in fin:
            has_newline = raw.endswith("\n")
            scrubbed, _ = scrub_jsonl_line(raw.rstrip("\n"), terms_map, counts)
            out_lines.append(scrubbed + ("\n" if has_newline else ""))
    dst.write_text("".join(out_lines), encoding="utf-8")
    return counts


def find_identity_hits(text: str, terms_map: dict[str, str]) -> Counter:
    """Count REAL (non-blob) identity matches without modifying the text.

    Same matching rules as scrub_text, so verify and scrub agree on what
    counts as a leak. Counter keys are the matched terms (lowercased) — callers
    must mask them (see mask_term) before display.
    """
    counts: Counter = Counter()
    rx = compile_terms(terms_map)
    if rx is None or not text:
        return counts
    if not rx.search(text):
        return counts
    spans = _blob_spans(text)
    for m in rx.finditer(text):
        if spans and _inside_blob(spans, m.start(), m.end()):
            continue
        counts[m.group(0).lower()] += 1
    return counts


def mask_term(term: str) -> str:
    """Display-safe form of an identity term (never print terms verbatim)."""
    if "@" in term:
        local, _, dom = term.partition("@")
        return f"{local[:2]}***@{dom}"
    return f"{term[:2]}***({len(term)} chars)"


# ---------------------------------------------------------------------------
# Third-party person detection (review-only; never auto-scrubbed).
# ---------------------------------------------------------------------------

_PROSE_MIN_LEN = 200
_CODE_SYMBOLS = set("{}[]()<>=;|\\`$#&*%")
_SENTENCE_RE = re.compile(r"[.!?][\s\"')\]]")
_SNIPPET_WINDOW = 60
_MAX_CANDIDATES = 40
_MAX_SNIPPETS = 3

# Capitalized-bigram fallback stoplist (either token disqualifies).
_NAME_STOPWORDS = {
    "The", "This", "That", "These", "Those", "Your", "Our", "Their", "His",
    "Her", "New", "Next", "First", "Last", "Then", "When", "While", "After",
    "Before", "Action", "Items", "Item", "Step", "Steps", "Note", "Notes",
    "Summary", "Meeting", "Review", "Type", "Error", "None", "True", "False",
    "Open", "Close", "Read", "Write", "File", "Files", "User", "Users",
    "Data", "Test", "Tests", "Run", "Set", "Get", "Add", "Api", "App",
    "Code", "Claude", "Codex", "Cursor", "Python", "Java", "Please", "Thanks",
    "Thank", "Hello", "Best", "Kind", "Regards", "Dear", "Also", "However",
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday",
    "Sunday", "January", "February", "March", "April", "June", "July",
    "August", "September", "October", "November", "December",
}
_NAME_BIGRAM_RE = re.compile(r"\b([A-Z][a-z]{2,})[ \t]+([A-Z][a-z]{2,})\b")


def looks_like_prose(text: str) -> bool:
    """Heuristic: long, sentence-structured, low symbol density (not code)."""
    if len(text) <= _PROSE_MIN_LEN:
        return False
    sample = text[:20000]
    symbols = sum(1 for c in sample if c in _CODE_SYMBOLS)
    if symbols / len(sample) > 0.04:
        return False
    if len(_SENTENCE_RE.findall(sample)) < 2:
        return False
    words = sample.split()
    return len(words) >= 20


def _person_candidates_regex(text: str) -> list[tuple[str, int, int]]:
    out = []
    for m in _NAME_BIGRAM_RE.finditer(text):
        first, last = m.group(1), m.group(2)
        if first in _NAME_STOPWORDS or last in _NAME_STOPWORDS:
            continue
        out.append((m.group(0), m.start(), m.end()))
    return out


_PRESIDIO_ANALYZER = None
_PRESIDIO_STATE = "unloaded"  # unloaded | ready | unavailable


def _presidio_person_analyzer():
    """Presidio + a REAL spaCy NER model (the pipeline's blank engine has no
    NER, so PERSON detection needs an installed en_core_web_* model)."""
    global _PRESIDIO_ANALYZER, _PRESIDIO_STATE
    if _PRESIDIO_STATE != "unloaded":
        return _PRESIDIO_ANALYZER
    try:
        import spacy
        from presidio_analyzer import AnalyzerEngine
        from presidio_analyzer.nlp_engine import NlpEngineProvider

        model = None
        for candidate in ("en_core_web_lg", "en_core_web_md", "en_core_web_sm"):
            try:
                spacy.load(candidate)
                model = candidate
                break
            except Exception:
                continue
        if model is None:
            raise RuntimeError("no spaCy NER model installed")
        provider = NlpEngineProvider(nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": model}],
        })
        _PRESIDIO_ANALYZER = AnalyzerEngine(
            nlp_engine=provider.create_engine(), supported_languages=["en"]
        )
        _PRESIDIO_STATE = "ready"
    except Exception:
        _PRESIDIO_ANALYZER = None
        _PRESIDIO_STATE = "unavailable"
    return _PRESIDIO_ANALYZER


def _person_candidates(text: str) -> tuple[list[tuple[str, int, int]], str]:
    analyzer = _presidio_person_analyzer()
    if analyzer is not None:
        try:
            results = analyzer.analyze(text=text[:100_000], language="en", entities=["PERSON"])
            hits = [
                (text[r.start:r.end], r.start, r.end)
                for r in results
                if r.score >= 0.5 and 0 <= r.start < r.end <= len(text)
            ]
            return hits, "presidio"
        except Exception:
            pass
    return _person_candidates_regex(text), "heuristic"


def _iter_string_values(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for v in value:
            yield from _iter_string_values(v)
    elif isinstance(value, dict):
        for v in value.values():
            yield from _iter_string_values(v)


def find_third_party_persons(
    texts: Iterable[str] | str,
    exclude_terms: dict[str, str] | None = None,
    max_candidates: int = _MAX_CANDIDATES,
) -> dict:
    """PERSON candidates in prose-like strings, for HUMAN review.

    Only fields that look like prose (>200 chars, sentence structure, low
    symbol density) are analyzed; code-looking content is skipped. Candidates
    already covered by the donor's own identity terms are excluded. Returns
    {"method": ..., "candidates": [{"name", "count", "snippets"}]} and never
    modifies anything — a human decides keep vs scrub per candidate.
    """
    if isinstance(texts, str):
        texts = [texts]
    exclude = set(exclude_terms or {})
    found: dict[str, dict] = {}
    method = "heuristic"
    for text in texts:
        if not isinstance(text, str) or not looks_like_prose(text):
            continue
        candidates, method = _person_candidates(text)
        for name, start, end in candidates:
            key = name.strip()
            if not key or len(key) < 4:
                continue
            if key.lower() in exclude or any(p.lower() in exclude for p in key.split()):
                continue
            entry = found.setdefault(key, {"name": key, "count": 0, "snippets": []})
            entry["count"] += 1
            if len(entry["snippets"]) < _MAX_SNIPPETS:
                lo = max(0, start - _SNIPPET_WINDOW)
                hi = min(len(text), end + _SNIPPET_WINDOW)
                snippet = text[lo:hi].replace("\n", " ").strip()
                entry["snippets"].append(("..." if lo else "") + snippet + ("..." if hi < len(text) else ""))
        if len(found) >= max_candidates:
            break
    ordered = sorted(found.values(), key=lambda e: -e["count"])[:max_candidates]
    return {"method": method, "engine_version": ENGINE_VERSION, "candidates": ordered}


def find_third_party_persons_in_jsonl(path: Path, exclude_terms: dict[str, str] | None = None) -> dict:
    """Convenience: run the person scan over all string values of a JSONL file."""

    def strings():
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    yield line
                    continue
                yield from _iter_string_values(obj)

    return find_third_party_persons(strings(), exclude_terms=exclude_terms)
