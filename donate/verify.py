"""ContextEcho donation — fail-closed PII verifier.

Re-scans an ALREADY-REDACTED session and exits NONZERO if any PII detector
still fires. This is the gate: a contribution cannot be submitted until verify
passes clean. Analogue of `tb tasks check`.

Checks:
  1. Residual-PII scan  — re-run the same detectors used in redaction; any hit
                          is a failure.
  2. Entropy scan       — flag high-entropy tokens that look like missed secrets.
  3. detect-secrets     — Yelp's secret scanner as an independent second opinion.

Usage:
    python -m donate.verify session.redacted.jsonl
    echo $?      # 0 = clean, 1 = residual PII found
"""
from __future__ import annotations

import argparse
import hashlib
import math
import re
import sys
from collections import Counter
from pathlib import Path

# Reuse the same key/path detectors the redactor used, so verify checks exactly
# what redact targeted. Works both as `python -m donate.verify` and direct run.
try:
    from donate.redact import API_KEY_RES  # type: ignore
except ImportError:  # direct invocation from inside the package dir
    from redact import API_KEY_RES  # type: ignore

# Email: local part must START with alphanumeric (rejects code like
# `@click.option` and `something-@x`), and the TLD must be a real-looking
# alphabetic suffix. Reject common code-attribute tails (.option, .py, etc.).
EMAIL_RE = re.compile(
    r"\b[A-Za-z0-9][A-Za-z0-9._%+\-]*@[A-Za-z0-9][A-Za-z0-9.\-]*\.[A-Za-z]{2,24}\b"
)
# TLDs that are really code/file tails, not email domains.
_CODE_TAILS = {"option", "py", "js", "ts", "json", "md", "txt", "sh", "yaml",
               "yml", "command", "group", "argument", "setter", "getter",
               "property", "staticmethod", "classmethod", "dataclass"}

# A residual home path: real '/Users/<name>' (slash form) still un-redacted.
RESIDUAL_HOME_RE = re.compile(
    r"(?:/Users/|/home/|[A-Za-z]:\\Users\\)(?!<USER)(?!<REDACTED)[^/\\\s\"']+"
)
# Dash-flattened slug form used in Claude Code internal paths:
#   '-Users-<name>-Library-...'  — these carry the username and must be redacted.
RESIDUAL_SLUG_RE = re.compile(r"-Users-(?!<USER)[A-Za-z0-9._]+")


def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = Counter(s)
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


# High-entropy tokens that are NON-secret structural IDs, ubiquitous in LLM
# session logs. These are not PII and not credentials — whitelist them so the
# entropy check doesn't cry wolf on every message/tool/request id.
ID_PREFIXES = (
    "msg_", "agent_msg_", "toolu_", "req_", "chatcmpl-", "run_", "thread_",
    "call_", "fc_", "resp_", "evt_", "asst_", "file-", "ftjob-", "batch_",
    "sess_", "span_", "trace_", "ws_", "conv_",
)
UUID_RE = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
                     r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")


def _is_known_non_secret(tok: str) -> bool:
    if UUID_RE.match(tok):
        return True
    low = tok.lower()
    return any(low.startswith(p) for p in ID_PREFIXES)


def find_high_entropy(text: str, min_len: int = 20, max_len: int = 64,
                      threshold: float = 4.5) -> list[str]:
    """Flag SHORT high-entropy tokens that look like real keys/secrets.

    LLM session logs are full of high-entropy NON-secrets: message/tool/request
    IDs (msg_, agent_msg_, toolu_, ...), UUIDs, and base64 content. Those are
    whitelisted. Long encoded BLOBS are excluded by the length bound. What
    remains is genuinely key-shaped. Note: the authoritative secret detection is
    the known-shape API_KEY_RES patterns; this entropy pass is a backstop.
    """
    hits = []
    for tok in re.findall(r"[A-Za-z0-9_\-/+=]+", text):
        if not (min_len <= len(tok) <= max_len):
            continue
        if tok.startswith("<") or tok.endswith(">"):
            continue
        if _is_known_non_secret(tok):
            continue
        if shannon_entropy(tok) >= threshold:
            hits.append(tok)
    return hits


def mask_token(tok: str) -> str:
    digest = hashlib.sha256(tok.encode("utf-8", errors="ignore")).hexdigest()[:8]
    return f"<HIGH_ENTROPY len={len(tok)} sha256={digest}>"


def verify_text(text: str) -> dict[str, list[str]]:
    findings: dict[str, list[str]] = {}

    emails = [e for e in EMAIL_RE.findall(text)
              if e.rsplit(".", 1)[-1].lower() not in _CODE_TAILS]
    if emails:
        findings["email"] = sorted(set(emails))[:10]

    homes = RESIDUAL_HOME_RE.findall(text) + RESIDUAL_SLUG_RE.findall(text)
    if homes:
        findings["home_path"] = sorted(set("".join(h) if isinstance(h, tuple) else h for h in homes))[:10]

    keys = []
    for rx in API_KEY_RES:
        keys.extend(rx.findall(text))
    if keys:
        findings["api_key"] = sorted(set(keys))[:10]

    return findings


# Categories that BLOCK submission (genuine PII / named credentials) vs.
# categories that are ADVISORY (shown for the user to eyeball, but don't fail —
# entropy can't tell a base64 secret from a base64 content fragment in an LLM
# log, so blocking on it makes verify never pass).
BLOCKING = {"email", "home_path", "api_key", "detect_secrets"}


# detect-secrets plugins that fire on ANY high-entropy/encoded string. In LLM
# logs these are overwhelmingly message IDs / base64 content, not credentials,
# so we ignore them and keep only the NAMED-provider key detectors.
_NOISY_DS_TYPES = {
    "Base64 High Entropy String", "Hex High Entropy String",
    "Secret Keyword",  # keyword heuristic — too noisy on prose
}


def detect_secret_findings(path: Path) -> list[dict[str, str | int]]:
    """Independent second opinion via Yelp detect-secrets, if available.

    Keeps only NAMED-provider detectors (AWS, GitHub, Stripe, etc.); drops the
    pure-entropy detectors that flag every base64/hex string and message ID.
    """
    try:
        from detect_secrets.settings import default_settings
        from detect_secrets.core import scan
    except Exception:
        return []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        with default_settings():
            findings: list[dict[str, str | int]] = []
            for s in scan.scan_file(str(path)):
                if s.type in _NOISY_DS_TYPES:
                    continue
                if s.type == "Private Key":
                    line = lines[s.line_number - 1] if 0 < s.line_number <= len(lines) else ""
                    # detect-secrets flags prose like "BEGIN RSA PRIVATE KEY".
                    # Only block real PEM delimiters that still expose key data.
                    if "-----BEGIN" not in line or "PRIVATE KEY-----" not in line:
                        continue
                findings.append({
                    "type": s.type,
                    "line_number": s.line_number,
                    "secret_value": getattr(s, "secret_value", "") or "",
                    "secret_hash": getattr(s, "secret_hash", "") or "",
                })
            return findings
    except Exception:
        return []


def run_detect_secrets(path: Path) -> list[str]:
    """Return only safe-to-display detect-secrets finding types."""
    return [str(item["type"]) for item in detect_secret_findings(path)]


def verify_session(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    findings = verify_text(text)
    findings["high_entropy"] = [mask_token(tok) for tok in sorted(set(find_high_entropy(text)))[:10]]
    ds = run_detect_secrets(path)
    if ds:
        findings["detect_secrets"] = sorted(set(ds))

    blocking = {k: v for k, v in findings.items() if k in BLOCKING and v}
    advisory = {k: v for k, v in findings.items() if k not in BLOCKING and v}
    return {
        "passed": not blocking,
        "findings": findings,
        "blocking": blocking,
        "advisory": advisory,
    }


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description="Fail-closed PII verifier for a redacted session.")
    p.add_argument("session", type=Path, help="Path to the redacted .jsonl")
    args = p.parse_args(argv)

    if not args.session.exists():
        print(f"[error] not found: {args.session}", file=sys.stderr)
        return 2

    report = verify_session(args.session)
    blocking = report["blocking"]
    advisory = report["advisory"]

    print(f"[verify] {args.session}")

    # Advisory findings: shown for the user to eyeball, never block.
    if advisory:
        print("\n[verify] NOTE — high-entropy tokens found (often message IDs or")
        print("[verify] base64 content, NOT secrets). Skim these to be sure:")
        for category, samples in advisory.items():
            for s in samples[:5]:
                print(f"      ? {s[:60]}")

    if not blocking:
        print("\n[verify] PASS — no residual emails, paths, or named credentials.")
        if advisory:
            print("[verify] (Review the high-entropy notes above; they are not blocking.)")
        return 0

    print("\n[verify] FAIL — residual PII found. DO NOT submit until resolved:\n")
    for category, samples in blocking.items():
        print(f"   {category}:")
        for s in samples:
            print(f"      - {s}")
    print("\n[verify] Re-run redact with --scrub for any missed identifiers,")
    print("[verify] or edit the file manually, then verify again.")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
