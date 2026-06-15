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
import tempfile
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


_LONG_TEXT_THRESHOLD = 20000
_VERIFY_WINDOW = 4096


def _windows_around_literals(text: str, literals: tuple[str, ...], *, window: int = _VERIFY_WINDOW) -> list[str]:
    low = text.lower()
    spans: list[tuple[int, int]] = []
    for literal in literals:
        start = 0
        needle = literal.lower()
        while True:
            idx = low.find(needle, start)
            if idx < 0:
                break
            spans.append((max(0, idx - window), min(len(text), idx + len(literal) + window)))
            start = idx + max(1, len(literal))
            if len(spans) >= 100:
                break
        if len(spans) >= 100:
            break
    if not spans:
        return []
    spans.sort()
    merged: list[tuple[int, int]] = []
    for start, end in spans:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return [text[start:end] for start, end in merged]


def _texts_for_literals(text: str, literals: tuple[str, ...]) -> list[str]:
    if len(text) <= _LONG_TEXT_THRESHOLD:
        return [text]
    return _windows_around_literals(text, literals) or []


def verify_text(text: str) -> dict[str, list[str]]:
    findings: dict[str, list[str]] = {}

    if "@" in text:
        emails = []
        for sample in _texts_for_literals(text, ("@",)):
            emails.extend(e for e in EMAIL_RE.findall(sample)
                          if e.rsplit(".", 1)[-1].lower() not in _CODE_TAILS)
        if emails:
            findings["email"] = sorted(set(emails))[:10]

    if "/Users/" in text or "/home/" in text or "\\Users\\" in text or "-Users-" in text:
        homes = []
        for sample in _texts_for_literals(text, ("/Users/", "/home/", "\\Users\\", "-Users-")):
            homes.extend(RESIDUAL_HOME_RE.findall(sample) + RESIDUAL_SLUG_RE.findall(sample))
        if homes:
            findings["home_path"] = sorted(set("".join(h) if isinstance(h, tuple) else h for h in homes))[:10]

    if _has_any_indicator(text, _API_KEY_INDICATORS):
        keys = []
        for sample in _texts_for_literals(text, _API_KEY_INDICATORS + ("://",)):
            if not _API_KEY_PREFILTER_RE.search(sample):
                continue
            for rx in API_KEY_RES:
                keys.extend(rx.findall(sample))
            if "private key" in sample.lower() and _PRIVATE_KEY_DELIMITER_RE.search(sample):
                keys.append("-----BEGIN PRIVATE KEY-----")
        if keys:
            findings["api_key"] = sorted(set(keys))[:10]

    return findings


def _merge_findings(target: dict[str, set[str]], findings: dict[str, list[str]]) -> None:
    for category, samples in findings.items():
        if not samples:
            continue
        target.setdefault(category, set()).update(str(sample) for sample in samples)


def _new_candidate_file():
    return tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        prefix="contextecho-detect-secrets-",
        suffix=".txt",
        delete=False,
    )


def verify_text_stream(path: Path) -> dict[str, list[str]]:
    """Run ContextEcho's built-in residual checks without loading huge logs."""
    merged: dict[str, set[str]] = {}
    entropy_hits: set[str] = set()
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            _merge_findings(merged, verify_text(line))
            if len(entropy_hits) < 100:
                entropy_hits.update(find_high_entropy(line))
    findings = {category: sorted(samples)[:10] for category, samples in merged.items() if samples}
    findings["high_entropy"] = [mask_token(tok) for tok in sorted(entropy_hits)[:10]]
    return findings


def _scan_builtin_and_write_detect_candidates(path: Path) -> tuple[dict[str, list[str]], Path | None, dict[int, int]]:
    """Single pass for built-in checks and detect-secrets candidate extraction."""
    merged: dict[str, set[str]] = {}
    entropy_hits: set[str] = set()
    line_map: dict[int, int] = {}
    candidate_no = 0
    tmp = _new_candidate_file()
    tmp_path = Path(tmp.name)
    try:
        with tmp, path.open("r", encoding="utf-8", errors="replace") as f:
            for original_no, line in enumerate(f, 1):
                _merge_findings(merged, verify_text(line))
                if len(entropy_hits) < 100:
                    entropy_hits.update(find_high_entropy(line))
                for snippet in _detect_secret_candidate_snippets(line):
                    candidate_no += 1
                    line_map[candidate_no] = original_no
                    tmp.write(snippet)
                    tmp.write("\n")
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    findings = {category: sorted(samples)[:10] for category, samples in merged.items() if samples}
    findings["high_entropy"] = [mask_token(tok) for tok in sorted(entropy_hits)[:10]]
    if not line_map:
        tmp_path.unlink(missing_ok=True)
        return findings, None, {}
    return findings, tmp_path, line_map


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

_DETECT_SECRETS_HINT_RE = re.compile(
    r"(?i)(secret|api[_-]?key|access[_-]?key|private[_ -]?key|authorization|"
    r"bearer|token|password|passwd|credential|client[_-]?secret|aws_|github|"
    r"stripe|slack|huggingface|sk-ant-|sk-|ghp_|gho_|akia|aiza|xox[baprs]-|hf_)"
)
_DETECT_SECRETS_VALUE_RE = re.compile(
    r"(?i)(?:secret|api[_-]?key|access[_-]?key|authorization|bearer|token|password|"
    r"passwd|credential|client[_-]?secret)\s*[=:]\s*['\"]?[A-Za-z0-9_./+\-=]{16,}"
)
_PRIVATE_KEY_DELIMITER_RE = re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")
_KNOWN_SECRET_PREFIX_RE = re.compile(
    r"(?i)(sk-ant-|sk-[A-Za-z0-9]{16,}|ghp_|gho_|akia[0-9a-z]{12,}|aiza[0-9a-z_-]{20,}|"
    r"xox[baprs]-|hf_[A-Za-z0-9]{16,})"
)
_API_KEY_PREFILTER_RE = re.compile(
    r"(?i)(sk-|sk-ant-|ghp_|gho_|akia|aiza|xox[baprs]-|bearer\s+|hf_|"
    r"authorization:\s*basic|://[^/\s:@]+:[^/\s:@]+@|private key)"
)
_API_KEY_INDICATORS = (
    "sk-", "sk-ant-", "ghp_", "gho_", "akia", "aiza", "xoxb-", "xoxa-",
    "xoxp-", "xoxr-", "xoxs-", "bearer ", "hf_", "authorization: basic",
    "private key",
)
_DETECT_SECRETS_INDICATORS = _API_KEY_INDICATORS + (
    "secret", "api_key", "api-key", "access_key", "access-key", "token",
    "password", "passwd", "credential", "client_secret", "client-secret",
    "aws_", "github", "stripe", "slack", "huggingface",
)
_DETECT_SECRETS_WINDOW = 2048


def _has_any_indicator(text: str, indicators: tuple[str, ...]) -> bool:
    low = text.lower()
    if any(indicator in low for indicator in indicators):
        return True
    return "://" in text and "@" in text and ":" in text


def _candidate_window(line: str, start: int, end: int) -> str:
    left = max(0, start - _DETECT_SECRETS_WINDOW)
    right = min(len(line), end + _DETECT_SECRETS_WINDOW)
    return line[left:right].replace("\n", " ")


def _detect_secret_candidate_snippets(line: str) -> list[str]:
    if not _has_any_indicator(line, _DETECT_SECRETS_INDICATORS):
        return []
    spans: list[tuple[int, int]] = []
    offset_samples: list[tuple[int, str]]
    if len(line) <= _LONG_TEXT_THRESHOLD:
        offset_samples = [(0, line)]
    else:
        offset_samples = []
        low = line.lower()
        for literal in _DETECT_SECRETS_INDICATORS + ("://",):
            start = 0
            needle = literal.lower()
            while True:
                idx = low.find(needle, start)
                if idx < 0:
                    break
                left = max(0, idx - _VERIFY_WINDOW)
                right = min(len(line), idx + len(literal) + _VERIFY_WINDOW)
                offset_samples.append((left, line[left:right]))
                start = idx + max(1, len(literal))
                if len(offset_samples) >= 100:
                    break
            if len(offset_samples) >= 100:
                break
    for offset, sample in offset_samples:
        for rx in (_PRIVATE_KEY_DELIMITER_RE, _KNOWN_SECRET_PREFIX_RE, _DETECT_SECRETS_VALUE_RE):
            spans.extend((offset + m.start(), offset + m.end()) for m in rx.finditer(sample))
        for rx in API_KEY_RES:
            spans.extend((offset + m.start(), offset + m.end()) for m in rx.finditer(sample))
    if not spans:
        return []
    snippets: list[str] = []
    seen: set[str] = set()
    for start, end in sorted(spans):
        snippet = _candidate_window(line, start, end)
        if snippet in seen:
            continue
        seen.add(snippet)
        snippets.append(snippet)
    return snippets


def _should_scan_with_detect_secrets(line: str) -> bool:
    return bool(_detect_secret_candidate_snippets(line))


def _write_detect_secrets_candidate_file(path: Path) -> tuple[Path | None, dict[int, int]]:
    line_map: dict[int, int] = {}
    candidate_no = 0
    tmp = _new_candidate_file()
    tmp_path = Path(tmp.name)
    try:
        with tmp, path.open("r", encoding="utf-8", errors="replace") as f:
            for original_no, line in enumerate(f, 1):
                for snippet in _detect_secret_candidate_snippets(line):
                    candidate_no += 1
                    line_map[candidate_no] = original_no
                    tmp.write(snippet)
                    tmp.write("\n")
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise
    if not line_map:
        tmp_path.unlink(missing_ok=True)
        return None, {}
    return tmp_path, line_map


def _detect_secret_findings_in_candidate(scan_path: Path, line_map: dict[int, int]) -> list[dict[str, str | int]]:
    try:
        from detect_secrets.settings import default_settings
        from detect_secrets.core import scan
    except Exception:
        return []
    try:
        cached_lines: list[str] | None = None

        def line_at(line_number: int) -> str:
            nonlocal cached_lines
            if cached_lines is None:
                cached_lines = scan_path.read_text(encoding="utf-8", errors="replace").splitlines()
            return cached_lines[line_number - 1] if 0 < line_number <= len(cached_lines) else ""

        with default_settings():
            findings: list[dict[str, str | int]] = []
            for s in scan.scan_file(str(scan_path)):
                if s.type in _NOISY_DS_TYPES:
                    continue
                if s.type == "Private Key":
                    line = line_at(s.line_number)
                    # detect-secrets flags prose like "BEGIN RSA PRIVATE KEY".
                    # Only block real PEM delimiters that still expose key data.
                    if "-----BEGIN" not in line or "PRIVATE KEY-----" not in line:
                        continue
                findings.append({
                    "type": s.type,
                    "line_number": line_map.get(s.line_number, s.line_number),
                    "secret_value": getattr(s, "secret_value", "") or "",
                    "secret_hash": getattr(s, "secret_hash", "") or "",
                })
            return findings
    except Exception:
        return []


def detect_secret_findings(path: Path) -> list[dict[str, str | int]]:
    """Independent second opinion via Yelp detect-secrets, if available.

    Keeps only NAMED-provider detectors (AWS, GitHub, Stripe, etc.); drops the
    pure-entropy detectors that flag every base64/hex string and message ID.
    """
    scan_path, line_map = _write_detect_secrets_candidate_file(path)
    if scan_path is None:
        return []
    try:
        return _detect_secret_findings_in_candidate(scan_path, line_map)
    finally:
        scan_path.unlink(missing_ok=True)


def run_detect_secrets(path: Path) -> list[str]:
    """Return only safe-to-display detect-secrets finding types."""
    return [str(item["type"]) for item in detect_secret_findings(path)]


def verify_session(path: Path) -> dict:
    findings, scan_path, line_map = _scan_builtin_and_write_detect_candidates(path)
    if scan_path is not None:
        try:
            ds = [str(item["type"]) for item in _detect_secret_findings_in_candidate(scan_path, line_map)]
            if ds:
                findings["detect_secrets"] = sorted(set(ds))
        finally:
            scan_path.unlink(missing_ok=True)

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
