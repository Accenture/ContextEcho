#!/usr/bin/env python3
"""Sanitizing release-export for the ContextEcho v2 public release.

Produces a clean, publishable tree under dist/release_v2/ (gitignored) from the
private local archive data_archive_release_v2/ — WITHOUT modifying the archive.

Why this exists (release blocker, PERSONA_DRIFT_PLAN.md 2026-07-28):
donation manifest.json files and some ledger.jsonl rows carry donor identity
fields (contributor, contributor_email, contributor_institute, credit_name,
institute, ...) even though every row is public_anonymous=true. DONOR_PRIVACY.md
and DATASET_CARD.md promise donor emails / donor-to-institution links are never
published. The archive must keep those fields locally (generators, review, and
consent records join on them), so the fix is an EXPORT step with a strict field
ALLOWLIST plus a post-export verification scan.

Usage:
    python3 scripts/export_release.py --dry-run          # keep/drop table + inventory, no writes
    python3 scripts/export_release.py                    # export + RE-REDACT sessions + verify
    python3 scripts/export_release.py --no-reredact      # legacy mode (sessions symlinked as-is)
    python3 scripts/export_release.py --copy             # force real file copies
    python3 scripts/export_release.py --skip-generators  # skip generator-regeneration diff
    python3 scripts/export_release.py --third-party-scan # also report third-party PERSON candidates

Design rules:
  * ALLOWLIST, not denylist: every field must be explicitly listed as KEEP or
    DROP; unknown fields abort the export so new archive fields get a decision.
  * Donor identity strings (names, emails, institutes) are loaded from the live
    archive AT RUNTIME for the re-redaction pass and the verification scan.
    They are never written to the export tree, to any file in the repo, or
    embedded in this script; reports carry COUNTS only.
  * RE-REDACTION (identity-seeded, 2026-07-28): exported session copies are
    scrubbed with the donate/identity_redaction engine, seeded with the union
    of all donors' name/email identity terms from the private archive (union,
    because at least one session echoes ledger rows naming OTHER donors) plus
    the maintainer git identity for the founding sessions (their donor
    identity is not recorded in the v1 archive). Sessions are always COPIED in
    this mode. Institutes stay a metadata-only concern (session-content scrub
    by institute would over-redact shared employer names).
  * The archive (data_archive_release_v2/, data_archive_release/) is read-only:
    only files under the --out tree are ever rewritten.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from donate import identity_redaction  # noqa: E402  (engine shared with the wizard)
ARCHIVE = REPO / "data_archive_release_v2"
V1_ARCHIVE = REPO / "data_archive_release"
DONATIONS = ARCHIVE / "data" / "donations"
SESSIONS = ARCHIVE / "data" / "sessions"
RESULTS = ARCHIVE / "results" / "session_validation"
DEFAULT_OUT = REPO / "dist" / "release_v2"

# Root governance docs shipped with the release (copied verbatim).
ROOT_DOCS = [
    "README.md",
    "DATASET_CARD.md",
    "DATASHEET.md",
    "croissant.json",
    "DATA_USE_POLICY.md",
    "DONOR_PRIVACY.md",
    "LICENSE",
]

# Derived results CSVs that croissant.json declares as distributions.
RESULT_CSVS = ["summary_v2.0.csv", "figure_inputs_v2.0.csv", "compactions_rederived.csv"]

# ---------------------------------------------------------------------------
# Field allowlists. Every field observed in the archive MUST appear in exactly
# one of KEEP / DROP below (audited 2026-07-28 over all 55 manifests + ledger).
# "why" strings document the decision; --dry-run prints them as a table.
# ---------------------------------------------------------------------------

LEDGER_KEEP = {
    "label": "join key for analyses (predict_drift_v2 extracts submission-id); croissant field",
    "submission_id": "public submission id; croissant field; generators + analyses join on it",
    "decision": "ACCEPTABLE/SUPERSEDED state; generators count populations from it; croissant field",
    "agent": "harness axis; generators/analyses/croissant",
    "source_format": "session JSONL schema selector; analyses/croissant",
    "model": "model axis; generators/analyses/croissant",
    "org": "org axis; generators/analyses/croissant",
    "domain": "task-domain axis; generators/analyses/croissant",
    "language": "language axis; generators/analyses/croissant",
    "records": "session scale; analyses (predict_drift) + croissant",
    "turns": "user-turn count; DATASET_CARD 12,337 total derives from it; croissant",
    "compactions": "compaction count; generators/croissant",
    "privacy_tier": "declared tier; generators/croissant",
    "public_anonymous": "anonymity flag (all true); generators/croissant",
    "session_path": "relative path to exported redacted session; croissant field",
    "session_sha256": "content hash of redacted session; dedup + croissant",
    "conversation_fingerprint": "structure fingerprint for lineage/dedup; croissant field",
    "fingerprint_version": "fingerprint algorithm version; non-identifying provenance",
    "promoted_utc": "promotion timestamp; generators sort on it; croissant field",
    "source_session_id": "16-hex lineage hash of the source session; dedup provenance, non-identifying",
    "metadata_confidence": "per-field confidence declared by donor tooling; documented in DATASHEET",
    "manifest_path": "relative path to exported sanitized manifest",
    "consent_path": "relative path to exported public consent summary",
    "superseded_by": "lineage pointer between public rows (submission ids only)",
    "supersedes_submission": "lineage pointer between public rows (submission ids only)",
}
LEDGER_DROP = {
    "contributor": "donor real name; public_anonymous=true on all rows (release blocker)",
    "credit_name": "donor real name (credit field); rows are anonymous so it is never displayed; NOTE croissant.json declares it — croissant needs the field removed",
    "contributor_email": "donor email (2 rows); DONOR_PRIVACY.md: never published",
    "contributor_institute": "donor-to-institution link (2 rows); DONOR_PRIVACY.md: never published",
    "institute": "donor-to-institution link (all 55 rows); DATASET_CARD only ever showed the aggregate count",
    "review_report_path": "target review_report.json is not exported (contains maintainer-local absolute paths); avoid dangling path",
}

MANIFEST_KEEP = {
    "session_id": "public donation id",
    "source_session_id": "16-hex lineage hash; non-identifying",
    "conversation_fingerprint": "structure fingerprint; lineage/dedup",
    "fingerprint_version": "fingerprint algorithm version",
    "agent": "harness axis",
    "model": "model axis",
    "org": "org axis",
    "domain": "task-domain axis",
    "donor_domain": "domain as declared by donor (kept alongside reviewed_domain)",
    "reviewed_domain": "domain after maintainer review",
    "language": "language axis",
    "records": "session scale",
    "turns": "user-turn count",
    "compactions": "compaction count",
    "public_anonymous": "anonymity flag",
    "privacy_tier": "declared tier",
    "allowed_uses": "consent scope",
    "disallowed_uses": "consent scope",
    "tool_version": "donation tool provenance",
    "source_format": "session JSONL schema",
    "metadata_confidence": "per-field confidence",
    "submitted_utc": "submission timestamp",
    "redacted_file": "original redacted artifact filename (random session UUID, non-identifying); update_contributors uses it as the dedup source_key — dropping it would change scoring",
    "session_sha256": "content hash of redacted session",
    "reviewed_submission_id": "public staging id",
    "metadata_update_request_id": "approved-metadata-update provenance (opaque id)",
    "metadata_updated_utc": "approved-metadata-update timestamp",
    "maintenance_redaction_updated_utc": "timestamp of last maintenance redaction pass (term-free)",
}
MANIFEST_DROP = {
    "donor_device_id": "one-way machine hash grouping a donor's submissions (maintainer-only; DONOR_PRIVACY: never published)",
    "contributor": "donor real name (release blocker)",
    "contributor_email": "donor email in ALL 55 manifests (release blocker)",
    "contributor_institute": "donor-to-institution link in ALL 55 manifests (release blocker)",
    "credit_name": "donor real name; never displayed for anonymous rows",
    "source_session_label": "donor-local project/workspace label — embeds donor surnames and private project names the redaction pass scrubbed from the sessions",
    "source_display_id": "prefix of the donor-local session UUID; redundant with redacted_file and only useful with the dropped source_session_label",
    "maintenance_redaction_terms": "the literal private words scrubbed from the session — publishing them defeats the redaction",
    "maintenance_redaction_stats": "counter keys embed the scrubbed terms (private_word:<term>)",
    "maintenance_redaction_history": "history entries embed the same term-bearing stats",
}

EMAIL_RE = re.compile(rb"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
DROPPED_KEY_NAMES = sorted(set(LEDGER_DROP) | set(MANIFEST_DROP))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Step 1: audit — field inventory + runtime-collected identity strings.
# ---------------------------------------------------------------------------

def name_variants(name: str) -> set[str]:
    """Case-insensitive scan variants of a donor name (space/hyphen/underscore)."""
    n = name.strip()
    out = {n}
    if " " in n:
        out.add(n.replace(" ", "-"))
        out.add(n.replace(" ", "_"))
    return {v.lower() for v in out if len(v) >= 4}


def audit(donation_dirs: list[Path]) -> dict:
    ledger_rows = iter_jsonl(DONATIONS / "ledger.jsonl")
    reviewed_rows = iter_jsonl(DONATIONS / "reviewed_submissions.jsonl")
    manifests = {d.name: read_json(d / "manifest.json") for d in donation_dirs}

    ledger_fields = Counter(k for r in ledger_rows for k in r)
    manifest_fields = Counter(k for m in manifests.values() for k in m)
    reviewed_fields = Counter(k for r in reviewed_rows for k in r)

    names: set[str] = set()
    emails: set[str] = set()
    institutes: set[str] = set()
    for src in list(ledger_rows) + list(manifests.values()):
        for key in ("contributor", "credit_name"):
            if src.get(key):
                names |= name_variants(str(src[key]))
        for key in ("contributor_email",):
            if src.get(key):
                email = str(src[key]).strip().lower()
                emails.add(email)
                # Distinctive email local-parts also identify the donor when the
                # domain is stripped (e.g. a handle reused as a workspace name).
                local = email.split("@", 1)[0]
                if len(local) >= 6:
                    names.add(local)
        for key in ("institute", "contributor_institute"):
            if src.get(key):
                institutes.add(str(src[key]).strip().lower())

    # Consent files: check whether they carry donor identity lines.
    consent_findings: list[str] = []
    for d in donation_dirs:
        consent = d / "CONSENT.md"
        text = consent.read_text(encoding="utf-8") if consent.exists() else ""
        hit = []
        if EMAIL_RE.search(text.encode()):
            hit.append("email")
        low = text.lower()
        if any(v in low for v in names):
            hit.append("donor-name")
        if any(i in low for i in institutes):
            hit.append("institute")
        if hit:
            consent_findings.append(f"{d.name}/CONSENT.md: {'+'.join(hit)}")

    return {
        "ledger_rows": ledger_rows,
        "reviewed_rows": reviewed_rows,
        "manifests": manifests,
        "ledger_fields": ledger_fields,
        "manifest_fields": manifest_fields,
        "reviewed_fields": reviewed_fields,
        "names": names,
        "emails": emails,
        "institutes": institutes,
        "consent_findings": consent_findings,
    }


def check_allowlist_complete(audit_data: dict) -> list[str]:
    problems = []
    for field in audit_data["ledger_fields"]:
        if field not in LEDGER_KEEP and field not in LEDGER_DROP:
            problems.append(f"ledger field {field!r} has no keep/drop decision")
    for field in audit_data["manifest_fields"]:
        if field not in MANIFEST_KEEP and field not in MANIFEST_DROP:
            problems.append(f"manifest field {field!r} has no keep/drop decision")
    return problems


# ---------------------------------------------------------------------------
# Step 2: sanitizers.
# ---------------------------------------------------------------------------

ANON_LABEL_RE = re.compile(r"^anonymous-donor-(submission-[0-9a-f]+)$")


def sanitized_label(label: str, submission_id: str) -> str:
    """Public directory/file label. Any label that is not already the anonymous
    form (e.g. it embeds a donor credit name) is rewritten to the anonymous form."""
    if ANON_LABEL_RE.match(label):
        return label
    return f"anonymous-donor-{submission_id}"


def sanitize_ledger_row(row: dict) -> dict:
    old_label = str(row.get("label", ""))
    new_label = sanitized_label(old_label, str(row.get("submission_id", "")))
    out = {}
    for key, value in row.items():
        if key in LEDGER_DROP:
            continue
        if key in ("manifest_path", "consent_path", "session_path", "label") and old_label != new_label:
            value = str(value).replace(old_label, new_label)
        out[key] = value
    return out


def sanitize_manifest(manifest: dict) -> dict:
    return {k: v for k, v in manifest.items() if k not in MANIFEST_DROP}


IDENTITY_LINE_RE = re.compile(r"^\*\*(Contributor|Email|Institute)\b.*$", re.MULTILINE)


def consent_summary(text: str, label: str) -> str:
    """Public consent summary: full attestation text minus donor identity lines."""
    body = IDENTITY_LINE_RE.sub("", text)
    body = re.sub(r"\n{3,}", "\n\n", body).rstrip() + "\n"
    note = (
        "\n---\n"
        "*Public consent summary. The donor chose to appear publicly as anonymous\n"
        "(`public_anonymous: true`); the contributor name and optional maintainer-contact\n"
        "email/institute fields of the signed consent are retained only in the private\n"
        "maintainer archive, per `DONOR_PRIVACY.md`. Donation label: "
        f"`{label}`.*\n"
    )
    return body + note


# ---------------------------------------------------------------------------
# Step 2b: identity-seeded re-redaction (donate/identity_redaction engine).
# ---------------------------------------------------------------------------

def maintainer_identity() -> tuple[str, str]:
    """Maintainer name/email from the repo git config (founding-session donor
    identity is not recorded in the v1 archive — noted in the export report)."""
    def cfg(key: str) -> str:
        try:
            return subprocess.run(
                ["git", "-C", str(REPO), "config", key],
                capture_output=True, text=True, check=False,
            ).stdout.strip()
        except Exception:
            return ""
    return cfg("user.name"), cfg("user.email")


def build_session_scrub_map(audit_data: dict) -> tuple[dict[str, str], int]:
    """Union of all donors' identity terms (names + emails; NOT institutes),
    plus the maintainer git identity for the founding sessions.

    Union rather than per-donor because at least one archived session echoes
    ledger rows that name OTHER donors; verify scans the union anyway, so
    scrub and verify agree by construction.
    """
    identities: set[tuple[str, str]] = set()
    for src in list(audit_data["ledger_rows"]) + list(audit_data["manifests"].values()):
        for name_key in ("contributor", "credit_name"):
            name = str(src.get(name_key) or "").strip()
            email = str(src.get("contributor_email") or "").strip()
            if name and name.lower() not in {"anonymous", "unknown"}:
                identities.add((name, email))
            elif email:
                identities.add(("", email))
    m_name, m_email = maintainer_identity()
    if m_name or m_email:
        identities.add((m_name, m_email))
    scrub_map: dict[str, str] = {}
    for name, email in sorted(identities):
        scrub_map.update(identity_redaction.build_identity_terms(name, email, ""))
    return scrub_map, len(identities)


def reredact_session(src: Path, dst: Path, scrub_map: dict[str, str]) -> dict:
    """Copy src -> dst with identity scrubbing; JSONL-safe; archive untouched.

    Returns per-session counts only (never the terms): pre-scrub REAL hits
    (engine matcher: word/path boundaries + blob guard), distinct terms hit,
    replacements made, and blob-guard skips.
    """
    assert src.resolve() != dst.resolve(), "re-redaction must never write to the archive"
    rx = identity_redaction.compile_terms(scrub_map)
    pre_hits: Counter = Counter()
    counts: Counter = Counter()
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    with src.open("r", encoding="utf-8", errors="replace") as fin, \
            dst.open("w", encoding="utf-8") as fout:
        for line in fin:
            stripped = line.rstrip("\n")
            if stripped and rx is not None and rx.search(stripped):
                pre_hits.update(identity_redaction.find_identity_hits(stripped, scrub_map))
                scrubbed, _ = identity_redaction.scrub_jsonl_line(stripped, scrub_map, counts)
                fout.write(scrubbed + ("\n" if line.endswith("\n") else ""))
            else:
                fout.write(line)
    return {
        "real_hits_pre_scrub": sum(pre_hits.values()),
        "distinct_terms_hit": len(pre_hits),
        "replacements": int(counts.get("identity_term", 0))
        + int(counts.get("structured:person_email", 0))
        + int(counts.get("structured:git_config", 0)),
        "blob_guard_skips": int(counts.get("blob_guard_skips", 0)),
    }


# ---------------------------------------------------------------------------
# Step 3: export.
# ---------------------------------------------------------------------------

def place(src: Path, dst: Path, copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    if copy:
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src.resolve())


def export(
    out: Path,
    audit_data: dict,
    copy: bool,
    scrub_map: dict[str, str] | None = None,
) -> tuple[list[str], dict[str, dict]]:
    """Export the sanitized tree. When scrub_map is set, every session is
    COPIED and re-redacted with the identity engine; returns (inventory,
    per-session re-redaction stats keyed by exported filename)."""
    inventory: list[str] = []
    reredact_stats: dict[str, dict] = {}
    if out.exists():
        shutil.rmtree(out)
    out_archive = out / "data_archive_release_v2"
    out_donations = out_archive / "data" / "donations"
    out_sessions = out_archive / "data" / "sessions"
    out_results = out_archive / "results" / "session_validation"
    out_donations.mkdir(parents=True)
    out_sessions.mkdir(parents=True)
    out_results.mkdir(parents=True)

    # Root docs (verbatim copies).
    for doc in ROOT_DOCS:
        shutil.copy2(REPO / doc, out / doc)
        inventory.append(f"doc      {doc}")

    # contributor_groups.json (already pseudonymous — audited clean).
    shutil.copy2(DONATIONS / "contributor_groups.json", out_donations / "contributor_groups.json")
    inventory.append("meta     data_archive_release_v2/data/donations/contributor_groups.json")

    # Per-donation manifests + consent summaries; sessions (renamed if needed).
    # Sessions are processed FIRST so re-redacted content hashes can be written
    # into the sanitized manifests and ledger (session_sha256 must describe the
    # published bytes, not the private-archive bytes).
    renamed = 0
    new_hashes: dict[str, str] = {}  # dirname -> sha256 of exported session
    for dirname, manifest in sorted(audit_data["manifests"].items()):
        row = next(
            (r for r in audit_data["ledger_rows"] if r.get("label") == dirname), {}
        )
        new_label = sanitized_label(dirname, str(row.get("submission_id", manifest.get("reviewed_submission_id", "")).strip()))
        if new_label != dirname:
            renamed += 1
        ddir = out_donations / new_label
        ddir.mkdir(parents=True, exist_ok=True)
        # Session file. Renamed sessions are ALWAYS copied so no symlink target
        # string can leak the original (name-bearing) filename; re-redaction
        # mode ALWAYS copies (the archive is never rewritten).
        src_session = SESSIONS / f"session_{dirname}.jsonl"
        dst_session = out_sessions / f"session_{new_label}.jsonl"
        if scrub_map is not None:
            stats = reredact_session(src_session, dst_session, scrub_map)
            reredact_stats[dst_session.name] = stats
            new_hashes[dirname] = sha256_file(dst_session)
        else:
            place(src_session, dst_session, copy=copy or (new_label != dirname))
        sanitized = sanitize_manifest(manifest)
        if dirname in new_hashes and sanitized.get("session_sha256"):
            sanitized["session_sha256"] = new_hashes[dirname]
        (ddir / "manifest.json").write_text(
            json.dumps(sanitized, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        consent_src = DONATIONS / dirname / "CONSENT.md"
        (ddir / "CONSENT.md").write_text(
            consent_summary(consent_src.read_text(encoding="utf-8"), new_label), encoding="utf-8"
        )
        reredacted_tag = " [RE-REDACTED]" if scrub_map is not None else ""
        inventory.append(f"donation {new_label} (manifest + consent summary + session{' [RENAMED]' if new_label != dirname else ''}{reredacted_tag})")

    # Sanitized ledger (session_sha256 refreshed for re-redacted sessions).
    rows = []
    for r in audit_data["ledger_rows"]:
        row_out = sanitize_ledger_row(r)
        label = str(r.get("label", ""))
        if label in new_hashes and row_out.get("session_sha256"):
            row_out["session_sha256"] = new_hashes[label]
        rows.append(row_out)
    ledger_out = out_donations / "ledger.jsonl"
    ledger_out.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8")
    inventory.append(f"ledger   {ledger_out.relative_to(out)} ({len(rows)} rows"
                     f"{'; session_sha256 refreshed post-re-redaction' if new_hashes else ''})")

    # Derived results CSVs declared by croissant.json.
    for csv in RESULT_CSVS:
        shutil.copy2(RESULTS / csv, out_results / csv)
        inventory.append(f"results  data_archive_release_v2/results/session_validation/{csv}")

    # v1 founding sessions (croissant declares them at data_archive_release/...).
    for f in sorted((V1_ARCHIVE / "data" / "sessions").glob("*.jsonl")):
        dst = out / "data_archive_release" / "data" / "sessions" / f.name
        if scrub_map is not None:
            stats = reredact_session(f, dst, scrub_map)
            reredact_stats[f"v1/{f.name}"] = stats
            inventory.append(f"v1       data_archive_release/data/sessions/{f.name} [RE-REDACTED]")
        else:
            place(f, dst, copy=copy)
            inventory.append(f"v1       data_archive_release/data/sessions/{f.name}")

    inventory.append(f"(renamed {renamed} name-bearing donation label(s) to the anonymous form)")
    return inventory, reredact_stats


# ---------------------------------------------------------------------------
# Step 4: verification scan.
# ---------------------------------------------------------------------------

def scan_file(path: Path, pattern: re.Pattern[bytes]) -> Counter:
    """Stream-scan a (possibly multi-GB) file; returns Counter of matched groups."""
    hits: Counter = Counter()
    overlap = 4096
    with open(path, "rb") as f:
        tail = b""
        while True:
            chunk = f.read(8 << 20)
            if not chunk:
                break
            buf = tail + chunk
            for m in pattern.finditer(buf):
                hits[m.group(0).lower()] += 1
            tail = buf[-overlap:]
    return hits


def mask(term: bytes | str) -> str:
    s = term.decode("utf-8", "replace") if isinstance(term, bytes) else term
    if "@" in s:
        local, _, dom = s.partition("@")
        return f"{local[:2]}***@{dom}"
    return f"{s[:3]}***({len(s)} chars)"


def scan_session_identity(path: Path, scrub_map: dict[str, str]) -> Counter:
    """Engine-matcher identity scan for session CONTENT: same word/path
    boundaries and base64/hex blob guard as the re-redaction pass, so verify
    counts real leaks — not substring false positives inside encoded blobs."""
    hits: Counter = Counter()
    rx = identity_redaction.compile_terms(scrub_map)
    if rx is None:
        return hits
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if rx.search(line):
                hits.update(identity_redaction.find_identity_hits(line, scrub_map))
    return hits


def verify(out: Path, audit_data: dict, scrub_map: dict[str, str] | None = None) -> int:
    print("\n=== VERIFICATION REPORT ===")
    identity_terms = sorted(audit_data["names"] | audit_data["emails"])
    identity_re = re.compile(
        b"|".join(re.escape(t.encode()) for t in identity_terms), re.IGNORECASE
    )
    donor_emails = {e.encode() for e in audit_data["emails"]}
    # Institute names can be short common words; require word boundaries so a
    # 4-char institute does not match inside e.g. "metadata_confidence".
    institute_re = re.compile(
        rb"\b(?:" + b"|".join(re.escape(t.encode()) for t in sorted(audit_data["institutes"])) + rb")\b",
        re.IGNORECASE,
    ) if audit_data["institutes"] else None
    key_re = re.compile(
        b"|".join(rb'"%s"\s*:' % k.encode() for k in DROPPED_KEY_NAMES)
    )

    metadata_files: list[Path] = []
    session_files: list[Path] = []
    doc_files: list[Path] = []
    for p in sorted(out.rglob("*")):
        if not p.is_file() and not p.is_symlink():
            continue
        if p.is_dir():
            continue
        rel = p.relative_to(out).as_posix()
        if "/data/sessions/" in rel:
            session_files.append(p)
        elif "/data/donations/" in rel:
            metadata_files.append(p)
        else:
            doc_files.append(p)

    failures = 0

    def report(kind: str, path: Path, hits: Counter, fatal: bool) -> None:
        nonlocal failures
        if not hits:
            return
        failures += 1 if fatal else 0
        tag = "VIOLATION" if fatal else "FINDING"
        terms = ", ".join(f"{mask(t)} x{c}" for t, c in hits.most_common(5))
        print(f"  [{tag}:{kind}] {path.relative_to(out)}: {terms}")

    # (a) dropped field names as JSON keys in sanitized metadata.
    print(f"[1/3] dropped-field-name scan over {len(metadata_files)} sanitized metadata files")
    for p in metadata_files:
        report("dropped-key", p, scan_file(p, key_re), fatal=True)

    # (b)+(c) emails + donor identity strings, everywhere.
    print(f"[2/3] identity + email scan: {len(metadata_files)} metadata, {len(doc_files)} docs, {len(session_files)} session files")
    for p in metadata_files:
        report("identity", p, scan_file(p, identity_re), fatal=True)
        report("email", p, scan_file(p, EMAIL_RE), fatal=True)
        if institute_re:
            report("institute", p, scan_file(p, institute_re), fatal=True)
    for p in doc_files:
        # Donor-name strings in root docs can be the arXiv author citation
        # (authors may also be donors). Reported as non-fatal findings for a
        # human to confirm; emails in docs are always fatal.
        report("identity(doc; may be paper-author citation — confirm)", p,
               scan_file(p, identity_re), fatal=False)
        report("email(doc)", p, scan_file(p, EMAIL_RE), fatal=True)
    for p in session_files:
        # Session content: donor identity hits are fatal. When a scrub map is
        # available (re-redaction mode), the scan uses the SAME engine matcher
        # as the scrub pass (boundaries + blob guard) so post-scrub verify
        # counts real leaks and not blob substrings; otherwise fall back to
        # the raw substring scan. Generic email-shaped strings (Python
        # decorators like @click.option, git@github.com, alice@example.com
        # fixtures) are reported non-fatally; exact donor emails are fatal.
        if scrub_map is not None:
            report("identity(session-content)", p, scan_session_identity(p, scrub_map), fatal=True)
        else:
            report("identity(session-content)", p, scan_file(p, identity_re), fatal=True)
        email_hits = scan_file(p, EMAIL_RE)
        donor_hits = Counter({t: c for t, c in email_hits.items() if t in donor_emails})
        other_hits = Counter({t: c for t, c in email_hits.items() if t not in donor_emails})
        report("donor-email(session-content)", p, donor_hits, fatal=True)
        report("email-like(session-content; review)", p, other_hits, fatal=False)

    # Structural checks on the sanitized ledger.
    print("[3/3] structural checks")
    rows = iter_jsonl(out / "data_archive_release_v2" / "data" / "donations" / "ledger.jsonl")
    turns = sum(int(r.get("turns", 0)) for r in rows)
    print(f"  ledger rows={len(rows)} total user turns={turns:,}")
    for r in rows:
        for key in ("session_path", "manifest_path", "consent_path"):
            target = out / "data_archive_release_v2" / str(r.get(key, ""))
            if not (target.exists() or target.is_symlink()):
                failures += 1
                print(f"  [VIOLATION:dangling-path] {r.get('submission_id')}: {key} -> {r.get(key)}")

    status = "FAILED" if failures else "PASSED"
    print(f"verification {status}: {failures} violating file(s)/check(s)")
    return failures


# ---------------------------------------------------------------------------
# Step 5: croissant consistency.
# ---------------------------------------------------------------------------

def croissant_check(out: Path) -> list[str]:
    print("\n=== CROISSANT CONSISTENCY ===")
    notes: list[str] = []
    cro = read_json(out / "croissant.json")
    rows = iter_jsonl(out / "data_archive_release_v2" / "data" / "donations" / "ledger.jsonl")
    ledger_keys = set().union(*[set(r) for r in rows])

    for rs in cro.get("recordSet", []):
        rs_id = rs.get("@id", "")
        for field in rs.get("field", []):
            src = field.get("source", {})
            col = (src.get("extract", {}) or {}).get("column") or (src.get("extract", {}) or {}).get("jsonPath", "")
            col = str(col).lstrip("$.")
            if rs_id == "donation-ledger" and col and col.split(".")[0] not in ledger_keys:
                notes.append(
                    f"croissant.json declares donation-ledger field '$.{col}' but the sanitized "
                    "ledger drops it -> croissant.json needs this field REMOVED before v2 publish (not edited here)"
                )
            if rs_id in ("session-drift-summary", "session-figure-inputs", "compactions-rederived") and col:
                csv_map = {
                    "session-drift-summary": "summary_v2.0.csv",
                    "session-figure-inputs": "figure_inputs_v2.0.csv",
                    "compactions-rederived": "compactions_rederived.csv",
                }
                csv_path = out / "data_archive_release_v2" / "results" / "session_validation" / csv_map[rs_id]
                header = csv_path.read_text(encoding="utf-8").splitlines()[0].split(",")
                if col not in header:
                    notes.append(f"croissant field {rs_id}/{col} missing from {csv_map[rs_id]} header")

    for dist in cro.get("distribution", []):
        url = dist.get("contentUrl", "")
        if not url or url.startswith("http"):
            continue
        p = out / url
        if not (p.exists() or p.is_symlink()):
            notes.append(f"croissant contentUrl not present in export: {url}")
            continue
        declared_sha = dist.get("sha256", "")
        if declared_sha and len(declared_sha) == 64:
            actual = sha256_file(p)
            if actual != declared_sha:
                notes.append(
                    f"croissant sha256 mismatch for {url}: file was sanitized/regenerated -> "
                    "croissant.json sha256/contentSize need refresh at publish time"
                )

    if notes:
        for n in notes:
            print(f"  [croissant] {n}")
    else:
        print("  all declared fields/files consistent")
    return notes


# ---------------------------------------------------------------------------
# Step 6: generator regeneration check (sanitized data -> same public stats?).
# ---------------------------------------------------------------------------

def generators_check(out: Path) -> list[str]:
    print("\n=== GENERATOR REGENERATION CHECK (sanitized data) ===")
    diffs: list[str] = []
    with tempfile.TemporaryDirectory(prefix="ctxecho_regen_") as td:
        tmp = Path(td)
        jobs = [
            (
                "DATASET_CARD.md",
                [sys.executable, str(REPO / "scripts" / "update_dataset_card.py"),
                 "--dataset-root", str(out / "data_archive_release_v2"),
                 "--out", str(tmp / "DATASET_CARD.md")],
                REPO / "DATASET_CARD.md", tmp / "DATASET_CARD.md",
            ),
            (
                "CONTRIBUTORS.md",
                [sys.executable, str(REPO / "scripts" / "update_contributors.py"),
                 "--dataset-root", str(out / "data_archive_release_v2"),
                 "--out", str(tmp / "CONTRIBUTORS.md"),
                 "--stats-out", str(tmp / "project_stats.json")],
                REPO / "CONTRIBUTORS.md", tmp / "CONTRIBUTORS.md",
            ),
        ]
        for name, cmd, ref, gen in jobs:
            proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
            if proc.returncode != 0:
                diffs.append(f"{name}: generator FAILED: {proc.stderr.strip()[:400]}")
                continue
            ref_text = ref.read_text(encoding="utf-8").splitlines()
            gen_text = gen.read_text(encoding="utf-8").splitlines()
            delta = [
                l for l in difflib.unified_diff(ref_text, gen_text, lineterm="", n=0)
                if l.startswith(("+", "-")) and not l.startswith(("+++", "---"))
            ]
            if delta:
                diffs.append(f"{name}: {len(delta)} changed line(s) vs repo copy")
                for l in delta[:30]:
                    diffs.append(f"    {l}")
            else:
                diffs.append(f"{name}: IDENTICAL to repo copy")
    for d in diffs:
        print(f"  {d}")
    return diffs


# ---------------------------------------------------------------------------
# Reporting.
# ---------------------------------------------------------------------------

def print_field_table(audit_data: dict) -> None:
    print("=== FIELD KEEP/DROP TABLE ===")
    for scope, counts, keep, drop in (
        ("ledger.jsonl", audit_data["ledger_fields"], LEDGER_KEEP, LEDGER_DROP),
        ("manifest.json", audit_data["manifest_fields"], MANIFEST_KEEP, MANIFEST_DROP),
    ):
        print(f"\n[{scope}]  ({sum(counts.values())} field instances over {len(counts)} distinct fields)")
        for field in sorted(counts):
            decision = "KEEP" if field in keep else "DROP"
            why = keep.get(field) or drop.get(field) or "?"
            print(f"  {decision}  {field:38s} (n={counts[field]:2d})  {why}")
    print("\n[reviewed_submissions.jsonl] NOT exported (not part of the public tree); fields observed:")
    print("  " + ", ".join(sorted(audit_data["reviewed_fields"])))
    print("\n[CONSENT.md] identity check:")
    if audit_data["consent_findings"]:
        print(f"  {len(audit_data['consent_findings'])} consent file(s) contain donor identity "
              "(name/email/institute) -> exporting PUBLIC CONSENT SUMMARIES instead of raw files")
    else:
        print("  clean")


def print_reredact_report(reredact_stats: dict[str, dict], n_identities: int, scrub_map: dict[str, str]) -> None:
    print("\n=== IDENTITY RE-REDACTION REPORT (counts only — terms are never printed) ===")
    print(f"  scrub seed: union of {n_identities} donor/maintainer identities "
          f"-> {len(scrub_map)} identity terms (names+emails; institutes excluded from session scrub)")
    with_hits = {k: v for k, v in reredact_stats.items() if v["real_hits_pre_scrub"]}
    total_hits = sum(v["real_hits_pre_scrub"] for v in reredact_stats.values())
    total_repl = sum(v["replacements"] for v in reredact_stats.values())
    total_blob = sum(v["blob_guard_skips"] for v in reredact_stats.values())
    print(f"  sessions scanned: {len(reredact_stats)}")
    print(f"  sessions with REAL (non-blob) identity hits before scrubbing: {len(with_hits)}")
    print(f"  total real identity hits: {total_hits:,}; replacements made: {total_repl:,}; "
          f"blob-guard false-positive skips: {total_blob:,}")
    for name, v in sorted(with_hits.items(), key=lambda kv: -kv[1]["real_hits_pre_scrub"]):
        print(f"    {name}: hits={v['real_hits_pre_scrub']} distinct_terms={v['distinct_terms_hit']} "
              f"replacements={v['replacements']} blob_skips={v['blob_guard_skips']}")


def third_party_scan(out: Path) -> None:
    print("\n=== THIRD-PARTY PERSON CANDIDATES (review-only; masked; never auto-scrubbed) ===")
    session_files = sorted(out.rglob("data/sessions/*.jsonl"))
    total = 0
    sessions_with = 0
    examples: list[str] = []
    method = "heuristic"
    for p in session_files:
        try:
            result = identity_redaction.find_third_party_persons_in_jsonl(p)
        except Exception as exc:
            print(f"  [skip] {p.name}: {exc}")
            continue
        candidates = result.get("candidates") or []
        method = result.get("method", method)
        if candidates:
            sessions_with += 1
            total += len(candidates)
            print(f"  {p.name}: {len(candidates)} candidate name(s)")
            for c in candidates[:3]:
                if len(examples) < 10:
                    examples.append(f"{identity_redaction.mask_term(str(c['name']))} x{c['count']}")
    print(f"  detector: {method}; sessions with candidates: {sessions_with}/{len(session_files)}; "
          f"total candidate names: {total}")
    if examples:
        print("  masked examples: " + ", ".join(examples))
    print("  -> human review required before any scrub decision (these may include public figures,")
    print("     library authors, or names the donor is fine keeping).")


def gitignore_check() -> None:
    gi = (REPO / ".gitignore").read_text(encoding="utf-8").splitlines()
    if any(line.strip() in ("dist/", "dist") for line in gi):
        print("[.gitignore] dist/ already ignored — export tree will not be committed")
    else:
        print("[.gitignore] WARNING: dist/ is NOT ignored; add it before exporting")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--copy", action="store_true", help="copy session files instead of symlinking")
    ap.add_argument("--dry-run", action="store_true", help="print keep/drop table + inventory; write nothing")
    ap.add_argument("--skip-generators", action="store_true", help="skip DATASET_CARD/CONTRIBUTORS regeneration diff")
    ap.add_argument("--no-reredact", action="store_true",
                    help="skip the identity-seeded session re-redaction (legacy export; sessions may be symlinked)")
    ap.add_argument("--third-party-scan", action="store_true",
                    help="also report third-party PERSON candidates in exported sessions (counts + masked examples)")
    args = ap.parse_args()

    donation_dirs = sorted(d for d in DONATIONS.iterdir() if d.is_dir())
    audit_data = audit(donation_dirs)

    problems = check_allowlist_complete(audit_data)
    if problems:
        print("ABORT — archive contains fields with no keep/drop decision:")
        for p in problems:
            print(f"  {p}")
        return 2

    print_field_table(audit_data)
    gitignore_check()
    print(f"\nidentity strings collected at runtime for the verify scan: "
          f"{len(audit_data['names'])} name variants, {len(audit_data['emails'])} emails, "
          f"{len(audit_data['institutes'])} institutes (values never written anywhere)")

    reredact = not args.no_reredact
    scrub_map: dict[str, str] = {}
    n_identities = 0
    if reredact:
        scrub_map, n_identities = build_session_scrub_map(audit_data)
        m_name, m_email = maintainer_identity()
        print(f"re-redaction scrub map: union of {n_identities} identities -> {len(scrub_map)} terms "
              f"(founding-session donor identity is not recorded in the v1 archive; using the "
              f"maintainer git identity as the known term source"
              f"{' — git config found' if (m_name or m_email) else ' — WARNING: git config empty'})")

    if args.dry_run:
        print("\n=== DRY RUN — planned export inventory ===")
        mode = "copied + re-redacted" if reredact else ("copied" if args.copy else "symlinked")
        print(f"out: {args.out}  (sessions {mode})")
        for doc in ROOT_DOCS:
            print(f"  doc      {doc}")
        for d in donation_dirs:
            row = next((r for r in audit_data["ledger_rows"] if r.get("label") == d.name), {})
            nl = sanitized_label(d.name, str(row.get("submission_id", "")))
            print(f"  donation {nl}{' [RENAMED from name-bearing label]' if nl != d.name else ''}")
        print(f"  ledger   ledger.jsonl ({len(audit_data['ledger_rows'])} rows) + contributor_groups.json")
        print(f"  results  {', '.join(RESULT_CSVS)}")
        print(f"  v1       {len(list((V1_ARCHIVE / 'data' / 'sessions').glob('*.jsonl')))} founding session files")
        return 0

    inventory, reredact_stats = export(
        args.out, audit_data, copy=args.copy or reredact,
        scrub_map=scrub_map if reredact else None,
    )
    print(f"\n=== EXPORTED {len(inventory)} item(s) to {args.out} ===")
    if reredact:
        print_reredact_report(reredact_stats, n_identities, scrub_map)

    failures = verify(args.out, audit_data, scrub_map=scrub_map if reredact else None)
    croissant_check(args.out)
    if args.third_party_scan:
        third_party_scan(args.out)
    if not args.skip_generators:
        generators_check(args.out)

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
