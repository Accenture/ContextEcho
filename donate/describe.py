"""ContextEcho donation — metadata + consent capture.

Collects the session metadata (auto-filled from discovery; contributor only
confirms/fills gaps) and a signed consent record, then writes a submission
manifest that travels with the redacted session.

The contributor's NAME here is for CREDIT (leaderboard / acknowledgments) and is
OPT-IN — they may use a GitHub handle or pseudonym. It is unrelated to redaction.

Usage (interactive):
    python -m donate.describe --session session.redacted.jsonl --auto auto.json

Produces: <session-stem>.manifest.json  +  CONSENT.md
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

COVERAGE_GAPS = {
    "agent": ["Codex CLI", "Cursor", "Aider", "Windsurf", "Cline", "Continue"],
    "org": ["OpenAI", "Google", "Meta", "DeepSeek", "Alibaba", "Mistral", "Cohere", "NVIDIA", "Moonshot"],
    "language": ["TypeScript", "Rust", "Go", "Java", "C++", "SQL"],
}

CONSENT_TEMPLATE = """# ContextEcho Donor Consent — {session_id}

I, the contributor identified below, consent to donating the redacted session
described in `{manifest}` to the ContextEcho benchmark dataset.

I attest that:

- [x] I own this session, or have the right to donate it.
- [x] It contains **no client-confidential** code/data, no material under NDA,
      and no other person's personal data.
- [x] I have run the local redaction + verification step and reviewed the diff;
      to my knowledge it contains no PII or secrets.
- [x] I agree to release the redacted session under **CC-BY-SA-4.0**.

**Contributor (for credit; may be a handle/pseudonym):** {contributor}
**Email (optional, maintainer contact only):** {email}
**Institute (optional):** {institute}
**Date:** {date}
**Tool version:** {tool_version}
"""

TOOL_VERSION = "contextecho-donate 0.3 (multi-agent-discovery)"


def ask(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    try:
        v = input(f"{prompt}{suffix}: ").strip()
    except EOFError:
        v = ""
    return v or default


def write_manifest_and_consent(
    session: Path,
    auto: dict,
    domain: str,
    language: str,
    contributor: str,
    email: str = "",
    institute: str = "",
) -> tuple[Path, Path, dict]:
    """Write the donation manifest and consent file next to a redacted session."""
    session_id = "S?"  # assigned by maintainer on merge
    manifest = {
        "session_id": session_id,
        "agent": auto.get("agent", "unknown"),
        "model": auto.get("model", "unknown"),
        "org": auto.get("org", "unknown"),
        "domain": domain,
        "language": language,
        "turns": str(auto.get("turns", "")),
        "compactions": str(auto.get("compactions", "")),
        "contributor": contributor or "anonymous",
        "credit_name": contributor or "anonymous",
        "contributor_email": email,
        "contributor_institute": institute,
        "tool_version": TOOL_VERSION,
        "source_format": auto.get("source_format", "unknown"),
        "metadata_confidence": auto.get("confidence", {}),
        "submitted_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "redacted_file": session.name,
    }

    stem = session.stem.replace(".redacted", "")
    manifest_path = session.with_name(f"{stem}.manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    consent_path = session.with_name("CONSENT.md")
    consent_path.write_text(CONSENT_TEMPLATE.format(
        session_id=session_id, manifest=manifest_path.name,
        contributor=manifest["contributor"],
        email=email or "",
        institute=institute or "",
        date=dt.date.today().isoformat(),
        tool_version=TOOL_VERSION,
    ))
    return manifest_path, consent_path, manifest


def gap_note(field: str, value: str) -> str:
    gaps = COVERAGE_GAPS.get(field, [])
    if value and value not in ("Anthropic", "Claude Code", "Python") and any(
        value.lower() == g.lower() for g in gaps
    ):
        return "  ⭐ fills a coverage gap — +1 novelty bonus"
    return ""


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description="Capture session metadata + consent.")
    p.add_argument("--session", type=Path, required=True, help="The redacted session file")
    p.add_argument("--auto", type=Path, default=None, help="Auto-detected metadata JSON from discover")
    p.add_argument("--non-interactive", action="store_true", help="Use auto values + defaults, no prompts")
    p.add_argument("--minimal-prompts", action="store_true",
                   help="auto-accept confident detected fields; ask only missing human metadata")
    args = p.parse_args(argv)

    auto: dict = {}
    if args.auto and args.auto.exists():
        auto = json.loads(args.auto.read_text())

    print("\n=== ContextEcho donation — describe your session ===")
    print("(Press Enter to accept the auto-detected value in [brackets].)\n")

    def field(name, label, default):
        if args.non_interactive:
            return default
        if args.minimal_prompts and name in {"agent", "model", "org"} and default and default != "unknown":
            print(f"{label}: {default}  [auto]")
            note = gap_note(name, default)
            if note:
                print(note)
            return default
        v = ask(label, default)
        note = gap_note(name, v)
        if note:
            print(note)
        return v

    agent    = field("agent",    "Agent / harness",        auto.get("agent", "Claude Code"))
    model    = field("model",    "Model",                  auto.get("model", "unknown"))
    org      = field("org",      "Organization",           auto.get("org", "unknown"))
    domain   = field("domain",   "Task domain (e.g. web-frontend, data-science, infra)", "")
    language = field("language", "Primary language (e.g. Python, TypeScript, Rust)",     "")
    turns    = str(auto.get("turns", ""))
    compactions = str(auto.get("compactions", ""))

    print("\n--- Credit (optional; a handle/pseudonym is fine) ---")
    contributor = field("contributor", "Name/handle for the contributor list", "anonymous")

    auto_for_write = dict(auto)
    auto_for_write.update({
        "agent": agent,
        "model": model,
        "org": org,
        "turns": turns,
        "compactions": compactions,
    })
    manifest_path, consent_path, manifest = write_manifest_and_consent(
        args.session, auto_for_write, domain, language, contributor
    )

    print("\n[describe] wrote:")
    print(f"   {manifest_path}")
    print(f"   {consent_path}")
    print("\n[describe] metadata:")
    for k in ("agent", "model", "org", "domain", "language", "turns", "compactions", "source_format"):
        print(f"   {k:12s} {manifest[k]}")
    print(f"\n[next] submit:  python -m donate.submit {args.session}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
