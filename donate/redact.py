"""ContextEcho donation — local PII redactor.

Redacts a coding-agent session log (Claude Code / Codex / Cursor JSONL) on the
contributor's own machine. Raw data never leaves the machine; only the redacted
output is ever submitted.

Design (research-backed, see CONTRIBUTING.md):
  Layer A — auto-detected, ZERO contributor input:
    * Presidio built-ins: PERSON, EMAIL_ADDRESS, IP_ADDRESS, PHONE_NUMBER,
      CREDIT_CARD, URL, CRYPTO
    * custom recognizers: HOME_PATH (and the username *inside* it, auto-extracted),
      API_KEY / token shapes
    * detect-secrets: high-entropy strings / secrets Presidio misses
  Layer B — optional, local-only:
    * --scrub "handle,codename,employer" — extra identifiers the contributor
      chooses to remove. Used to REMOVE info, never to collect it.

Anonymization:
  * usernames  -> salted hash, so the same user maps to ONE stable pseudonym
                  across the whole session  (/Users/jane -> /Users/<USER_a1b2c3>)
  * emails/names/paths -> <EMAIL> / <PERSON> / <HOME_PATH>
  * secrets    -> <SECRET>

Usage:
    python -m donate.redact path/to/session.jsonl
    python -m donate.redact session.jsonl --scrub "jbob,Falcon migration,AcmeCorp"
    python -m donate.redact session.jsonl --out session.redacted.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Custom detectors (the pieces Presidio does not ship)
# ---------------------------------------------------------------------------

# /Users/<name>, /home/<name>, C:\Users\<name>  — capture the username group.
HOME_PATH_RE = re.compile(
    r"(?P<prefix>/Users/|/home/|[A-Za-z]:\\Users\\)(?P<user>[^/\\\s\"']+)"
)
# Dash-flattened slug form Claude Code uses for internal project paths:
#   '-Users-<name>-Library-...'  — the username appears after '-Users-'.
SLUG_PATH_RE = re.compile(r"(?P<prefix>-Users-)(?P<user>[A-Za-z0-9._]+)")

# Common credential / token shapes. detect-secrets handles entropy; these pin
# the well-known prefixes so we never depend on entropy alone for known keys.
API_KEY_RES = [
    re.compile(r"sk-[A-Za-z0-9_\-]{16,}"),          # OpenAI / Anthropic-style
    re.compile(r"sk-ant-[A-Za-z0-9_\-]{16,}"),      # Anthropic
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),            # GitHub PAT
    re.compile(r"gho_[A-Za-z0-9]{20,}"),            # GitHub OAuth
    re.compile(r"AKIA[0-9A-Z]{16}"),                # AWS access key id
    re.compile(r"AIza[0-9A-Za-z_\-]{35}"),          # Google API key
    re.compile(r"xox[baprs]-[0-9A-Za-z\-]{10,}"),   # Slack
    re.compile(r"Bearer\s+[A-Za-z0-9._\-]{20,}"),   # bearer tokens
    re.compile(r"hf_[A-Za-z0-9]{20,}"),             # HuggingFace
]


def _salt() -> str:
    """Per-run salt so pseudonyms are stable within a session but not reversible
    across runs without the salt. Deterministic within one invocation."""
    return "contextecho-donate-v1"


def pseudonym(name: str) -> str:
    h = hashlib.sha256((_salt() + name.lower()).encode()).hexdigest()[:6]
    return f"<USER_{h}>"


# ---------------------------------------------------------------------------
# Presidio engine (lazy import so --help works without the heavy deps)
# ---------------------------------------------------------------------------

def build_analyzer():
    from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer

    try:
        analyzer = AnalyzerEngine()
    except Exception as exc:
        raise RuntimeError(
            "Could not initialize the local redaction engine. Run "
            "`make setup-donate` from the repository root. If Presidio reports "
            "a missing English model, run `python3 -m spacy download en_core_web_lg`."
        ) from exc

    api_key = PatternRecognizer(
        supported_entity="API_KEY",
        patterns=[Pattern("api_key", r"(sk-ant-|sk-|ghp_|gho_|AKIA|AIza|xox[baprs]-|hf_)[A-Za-z0-9._\-]{12,}", 0.9)],
    )
    analyzer.registry.add_recognizer(api_key)
    return analyzer


# ---------------------------------------------------------------------------
# Redaction passes
# ---------------------------------------------------------------------------

def discover_usernames(text: str) -> set[str]:
    """Auto-extract usernames from home paths (slash AND dash-slug form)."""
    users = set()
    for rx in (HOME_PATH_RE, SLUG_PATH_RE):
        for m in rx.finditer(text):
            u = m.group("user")
            if u and u.lower() not in {"<user>", "user", "shared", "public", "root"}:
                users.add(u)
    return users


def redact_text(
    text: str,
    analyzer,
    scrub_terms: set[str],
    stats: dict,
    known_usernames: set[str] | None = None,
) -> str:
    from presidio_anonymizer import AnonymizerEngine
    from presidio_anonymizer.entities import OperatorConfig

    anonymizer = AnonymizerEngine()

    # 1. Auto-discovered usernames -> stable pseudonyms (do first; before paths).
    usernames = set(known_usernames or set()) | discover_usernames(text)
    for user in sorted(usernames, key=len, reverse=True):
        if user in scrub_terms:
            continue
        n = text.count(user)
        if n:
            text = text.replace(user, pseudonym(user))
            stats["username"] = stats.get("username", 0) + n

    # 2. Contributor-supplied scrub terms (local-only safety valve).
    for term in sorted(scrub_terms, key=len, reverse=True):
        if not term:
            continue
        n = text.count(term)
        if n:
            text = text.replace(term, "<REDACTED>")
            stats["scrub_term"] = stats.get("scrub_term", 0) + n

    # 3. Home-path prefixes left over -> generic (slash and dash-slug forms).
    text, n = HOME_PATH_RE.subn(lambda m: m.group("prefix") + "<USER>", text)
    if n:
        stats["home_path"] = stats.get("home_path", 0) + n
    text, n = SLUG_PATH_RE.subn(lambda m: m.group("prefix") + "<USER>", text)
    if n:
        stats["home_path"] = stats.get("home_path", 0) + n

    # 4. Known API-key/token shapes (pre-Presidio so they never slip through).
    for rx in API_KEY_RES:
        text, n = rx.subn("<SECRET>", text)
        if n:
            stats["api_key"] = stats.get("api_key", 0) + n

    # 5. Presidio: names, emails, IPs, phones, cards, URLs, crypto.
    # spaCy/NER caps input at ~1M chars and uses ~1GB RAM per 100k chars, so
    # process in chunks — real agent turns (huge tool outputs/file dumps) can be
    # millions of chars on a single JSONL line.
    operators = {
        "DEFAULT": OperatorConfig("replace", {"new_value": "<REDACTED>"}),
        "EMAIL_ADDRESS": OperatorConfig("replace", {"new_value": "<EMAIL>"}),
        "PERSON": OperatorConfig("replace", {"new_value": "<PERSON>"}),
        "IP_ADDRESS": OperatorConfig("replace", {"new_value": "<IP>"}),
        "PHONE_NUMBER": OperatorConfig("replace", {"new_value": "<PHONE>"}),
        "CREDIT_CARD": OperatorConfig("replace", {"new_value": "<CARD>"}),
        "URL": OperatorConfig("replace", {"new_value": "<URL>"}),
        "CRYPTO": OperatorConfig("replace", {"new_value": "<CRYPTO>"}),
        "API_KEY": OperatorConfig("replace", {"new_value": "<SECRET>"}),
    }
    entities = ["PERSON", "EMAIL_ADDRESS", "IP_ADDRESS", "PHONE_NUMBER",
                "CREDIT_CARD", "URL", "CRYPTO", "API_KEY"]
    CHUNK = 200_000  # well under spaCy's 1M limit; bounds NER memory

    out_parts = []
    for start in range(0, len(text), CHUNK):
        chunk = text[start:start + CHUNK]
        results = analyzer.analyze(text=chunk, language="en", entities=entities)
        if results:
            for r in results:
                stats[r.entity_type] = stats.get(r.entity_type, 0) + 1
            chunk = anonymizer.anonymize(
                text=chunk, analyzer_results=results, operators=operators
            ).text
        out_parts.append(chunk)

    return "".join(out_parts)


def redact_json_value(value: Any, analyzer, scrub_terms: set[str], stats: dict, usernames: set[str]) -> Any:
    """Redact string leaves while preserving JSON structure."""
    if isinstance(value, str):
        return redact_text(value, analyzer, scrub_terms, stats, known_usernames=usernames)
    if isinstance(value, list):
        return [redact_json_value(v, analyzer, scrub_terms, stats, usernames) for v in value]
    if isinstance(value, dict):
        return {k: redact_json_value(v, analyzer, scrub_terms, stats, usernames) for k, v in value.items()}
    return value


def _progress_iter(items, total, show):
    """Yield items with a progress bar. Uses tqdm if present, else a plain bar."""
    if not show:
        yield from items
        return
    try:
        from tqdm import tqdm
        yield from tqdm(items, total=total, desc="[redact] turns", unit="turn", ncols=70)
        return
    except Exception:
        pass
    # Lightweight fallback bar (no dependency).
    width = 30
    for i, item in enumerate(items, 1):
        if i == 1 or i % 50 == 0 or i == total:
            filled = int(width * i / total) if total else width
            bar = "#" * filled + "-" * (width - filled)
            pct = (100 * i // total) if total else 100
            sys.stderr.write(f"\r[redact] [{bar}] {pct:3d}%  {i}/{total} turns")
            sys.stderr.flush()
        yield item
    sys.stderr.write("\n")
    sys.stderr.flush()


def redact_file(src: Path, dst: Path, scrub_terms: set[str], progress: bool = False) -> dict:
    analyzer = build_analyzer()
    stats: dict = {}
    lines = src.read_text(encoding="utf-8", errors="replace").splitlines()
    lines_out = []
    for line in _progress_iter(lines, len(lines), progress):
        if not line.strip():
            lines_out.append(line)
            continue
        usernames = discover_usernames(line)
        try:
            obj = json.loads(line)
        except Exception:
            # Unknown/non-JSON logs are still handled as raw text.
            lines_out.append(redact_text(line, analyzer, scrub_terms, stats, known_usernames=usernames))
            continue
        redacted_obj = redact_json_value(obj, analyzer, scrub_terms, stats, usernames)
        lines_out.append(json.dumps(redacted_obj, ensure_ascii=False, separators=(",", ":")))
    dst.write_text("\n".join(lines_out) + "\n", encoding="utf-8")
    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description="Locally redact a coding-agent session for donation.")
    p.add_argument("session", type=Path, help="Path to the session .jsonl")
    p.add_argument("--out", type=Path, default=None, help="Output path (default: <session>.redacted.jsonl)")
    p.add_argument("--scrub", type=str, default="", help="Optional comma-separated extra terms to remove (local-only)")
    args = p.parse_args(argv)

    if not args.session.exists():
        print(f"[error] not found: {args.session}", file=sys.stderr)
        return 2

    out = args.out or args.session.with_suffix(".redacted.jsonl")
    scrub_terms = {t.strip() for t in args.scrub.split(",") if t.strip()}

    print(f"[redact] {args.session}  ->  {out}")
    if scrub_terms:
        print(f"[redact] extra scrub terms (local-only): {sorted(scrub_terms)}")
    stats = redact_file(args.session, out, scrub_terms)

    print("\n[redact] removed:")
    if stats:
        for k, v in sorted(stats.items(), key=lambda x: -x[1]):
            print(f"   {k:16s} {v}")
    else:
        print("   (nothing matched — review the diff carefully)")
    print(f"\n[redact] wrote {out}")
    print("[next] run:  python -m donate.verify", out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
