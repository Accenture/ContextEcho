"""ContextEcho donation — the one-key wizard.

    python -m donate

Chains the five stages into a single guided flow:
    discover -> redact (local) -> verify (fail-closed) -> describe + consent -> submit

Privacy: redaction happens entirely on your machine; only the verified-clean,
redacted session is ever uploaded. A pull request is opened for maintainer
review — nothing is public until accepted.
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
from pathlib import Path


def reexec_repo_venv_if_available() -> None:
    """Let `python3 -m donate --web` use the venv created by `make setup-donate`."""
    repo_root = Path(__file__).resolve().parent.parent
    venv_dir = repo_root / ".venv"
    venv_python = venv_dir / "bin" / "python"
    if not venv_python.exists():
        return
    try:
        in_repo_venv = Path(sys.prefix).resolve() == venv_dir.resolve()
    except OSError:
        return
    if in_repo_venv or os.environ.get("CONTEXTECHO_DONATE_NO_REEXEC"):
        return
    os.execv(str(venv_python), [str(venv_python), "-m", "donate", *sys.argv[1:]])


reexec_repo_venv_if_available()

from donate.adapters.base import is_redacted_artifact
from donate import describe as describe_mod
from donate import discover as discover_mod
from donate import redact as redact_mod
from donate import submit as submit_mod


def ask(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    try:
        return input(f"{prompt}{suffix}: ").strip() or default
    except EOFError:
        return default


def banner(txt: str) -> None:
    print("\n" + "=" * 64)
    print(f"  {txt}")
    print("=" * 64)


def safe_slug(text: str, default: str = "session") -> str:
    cleaned = "".join(c if c.isalnum() or c in ("-", "_") else "-" for c in text)
    cleaned = "-".join(part for part in cleaned.split("-") if part)
    return (cleaned[:64] or default).strip("-_") or default


def compact_label(text: object, width: int) -> str:
    s = str(text or "")
    return s if len(s) <= width else s[: max(0, width - 1)] + "…"


def format_turns(value: object) -> str:
    try:
        n = int(value)
    except Exception:
        return "?"
    if n >= 1000:
        return f"{n / 1000:.1f}k"
    return str(n)


def quality_tag(session: dict) -> str:
    turns = int(session.get("turns") or 0)
    compactions = int(session.get("compactions") or 0)
    if turns >= 100 and compactions > 0:
        return "best"
    if turns >= 100:
        return "long"
    return "short"


def print_session_table(sessions: list[dict], start: int = 0, limit: int = 15) -> None:
    shown = sessions[start:start + limit]
    print("  #   Fit    Agent        UserT  CCmp Last active  Project")
    print("  --  -----  -----------  -----  ---  -----------  ------------------------------")
    for i, s in enumerate(shown, start + 1):
        agent = compact_label(s.get("agent", "?").replace("Claude Code", "Claude").replace("Codex CLI", "Codex"), 11)
        date = compact_label(s.get("last_active") or s.get("modified", "?"), 11)
        project = compact_label(s.get("project", "?"), 30)
        print(
            f"  {i:>2}  {quality_tag(s):<5}  {agent:<11}  "
            f"{format_turns(s.get('turns')):>5}  {int(s.get('compactions') or 0):>3}  "
            f"{date:<11}  {project}"
        )
    end = min(start + limit, len(sessions))
    print(f"\n  Showing {start + 1}-{end} of {len(sessions)} usable sessions.")
    print("  UserT = human/user prompt turns. CCmp = context compactions detected in local logs.")
    print("  Minimum shown: 20+ user turns, or any detected context compaction.")
    print("  Fit: best = 100+ user turns with context compactions.")
    if end < len(sessions):
        print("  Type 'more' to show more, a number to select, or paste a session path.")


def choose_session(sessions: list[dict], page_size: int = 15) -> tuple[dict | None, Path]:
    start = 0
    while True:
        print_session_table(sessions, start=start, limit=page_size)
        default = "1" if start == 0 else str(start + 1)
        pick = ask("\nPick a session number, 'more', or paste a path", default)
        pick_l = pick.lower()
        if pick_l in {"more", "m", "next", "n"}:
            if start + page_size >= len(sessions):
                print("[discover] Already showing the last page.")
            else:
                start += page_size
            continue
        if pick_l in {"top", "first"}:
            start = 0
            continue
        if pick.isdigit() and 1 <= int(pick) <= len(sessions):
            chosen = sessions[int(pick) - 1]
            return chosen, Path(chosen["path"])
        return None, Path(pick).expanduser()


def donation_output_dir(chosen: dict) -> Path:
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    agent = safe_slug(str(chosen.get("agent", "agent")).lower())
    project = safe_slug(str(chosen.get("project", "session")))
    root = Path.home() / "Downloads" / "ContextEcho_donations"
    return root / f"{stamp}-{agent}-{project}"


def redacted_output_name(src: Path) -> str:
    stem = src.stem
    if stem.endswith(".redacted"):
        stem = stem[: -len(".redacted")]
    return f"{safe_slug(stem)}.redacted.jsonl"


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="One-command ContextEcho donation wizard.",
        epilog=(
            "Default path: run this with no flags, press Enter for the suggested "
            "session, confirm the safety checkpoint, and the wizard handles "
            "redact -> verify -> describe -> submit."
        ),
    )
    p.add_argument("--max-per-agent", type=int, default=50,
                   help="inspect at most this many recent logs per agent during discovery")
    p.add_argument("--all", action="store_true",
                   help="inspect every discovered log instead of only recent candidates")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="where to write redacted donation artifacts (default: ~/Downloads/ContextEcho_donations/<run>)")
    p.add_argument("--web", action="store_true", help="launch the local browser wizard instead of terminal prompts")
    p.add_argument("--web-port", type=int, default=8766, help="port for --web")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.web:
        from donate import web as web_mod
        return web_mod.main(["--port", str(args.web_port)])

    banner("ContextEcho — donate a coding-agent session")
    print(
        "Your session is redacted ON YOUR MACHINE. Only the verified-clean,\n"
        "redacted version is uploaded, as a pull request maintainers review.\n"
        "Nothing becomes public until it is accepted.\n"
    )

    # 1) DISCOVER ----------------------------------------------------------
    banner("1/5  Discover")
    sessions = discover_mod.discover(
        max_per_agent=None if args.all else args.max_per_agent,
        progress=True,
    )
    chosen: dict | None = None
    if sessions:
        chosen, src = choose_session(sessions)
    else:
        src = Path(ask("No sessions auto-found. Paste a session .jsonl path")).expanduser()

    if not src.exists():
        print(f"[error] not found: {src}")
        return 2
    if is_redacted_artifact(src):
        print(f"[error] selected file already looks redacted: {src}")
        print("[wizard] Please select the original agent session log, not a previous donation output.")
        print("[wizard] Auto-discovery now hides .redacted.jsonl files; re-run `python -m donate`.")
        return 2
    if chosen is None:
        chosen = discover_mod.inspect_session(src)

    # Confidentiality checkpoint — enforce the "no client/NDA work" rule.
    print(f"\n  Selected: {chosen.get('project', src.name)} · {chosen.get('turns', '?')} user turns")
    print("\n  ⚠️  Only donate sessions from PERSONAL projects, internal tooling, or")
    print("      open-source work. Do NOT donate client-confidential or NDA work,")
    print("      or anything containing another person's data.")
    if ask("  Confirm this session is safe to donate (y/N)", "N").lower() not in ("y", "yes"):
        print("[wizard] Stopped. Nothing was processed.")
        return 0

    # 2) REDACT ------------------------------------------------------------
    banner("2/5  Redact (local)")
    print("Names, emails, file paths, usernames, and API keys are removed")
    print("AUTOMATICALLY — you don't need to list them.")
    print("Optionally add anything auto-detection can't know: a nickname/handle,")
    print("an internal codename, an employer/client word.")
    extra = ask("Extra terms to scrub (comma-separated, or press Enter to skip)", "")

    # Treat skip-words as "nothing" rather than literal terms to redact.
    SKIP_WORDS = {"", "no", "none", "n", "skip", "nothing", "na", "n/a", "-"}
    raw_terms = [t.strip() for t in extra.split(",") if t.strip()]
    scrub_terms = {t for t in raw_terms if t.lower() not in SKIP_WORDS}

    # Acknowledge the input immediately, BEFORE the (possibly slow) NER pass,
    # so the user knows their input registered and what's about to happen.
    if scrub_terms:
        print(f"\n[redact] Got it. Will also remove these terms: {', '.join(sorted(scrub_terms))}")
    elif raw_terms:
        print("\n[redact] Read that as 'nothing extra' — using auto-detection only.")
    else:
        print("\n[redact] No extra terms — using auto-detection only.")

    records = chosen.get("records") or chosen.get("turns", 0) if isinstance(chosen, dict) else 0
    print(f"[redact] Redacting {int(records):,} records locally — your data never leaves this machine."
          if records else "[redact] Redacting locally — your data never leaves this machine.")

    out_dir = (args.output_dir.expanduser() if args.output_dir else donation_output_dir(chosen))
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / redacted_output_name(src)
    print(f"[redact] Donation files will be saved in: {out_dir}")
    stats = redact_mod.redact_file(src, out, scrub_terms, progress=True)
    print("[redact] Done.")

    # Per-term feedback: confirm each requested term was found (and how often).
    for term in sorted(scrub_terms, key=len, reverse=True):
        # count occurrences in the ORIGINAL to report honestly
        try:
            hits = src.read_text(encoding="utf-8", errors="replace").count(term)
        except Exception:
            hits = 0
        if hits:
            print(f"[redact]   '{term}': removed in {hits} place(s) ✓")
        else:
            print(f"[redact]   '{term}': not found (0 places) — nothing to remove")

    print("[redact] auto-removed:", ", ".join(f"{k}={v}" for k, v in sorted(stats.items(), key=lambda x: -x[1])) or "(nothing)")
    print(f"[redact] wrote {out}")
    print(f"[redact] Easy-to-find folder: {out_dir}")
    print("[redact] Please open the redacted file and skim it — you are the final reviewer.")

    # 3) VERIFY ------------------------------------------------------------
    banner("3/5  Verify (fail-closed)")
    if not submit_mod.verify_passed(out):
        print("\n[wizard] Verify FAILED. Re-run with extra --scrub terms, or edit the file, then retry.")
        return 1

    # 4) DESCRIBE + CONSENT ------------------------------------------------
    banner("4/5  Describe + consent")
    import json, tempfile
    auto = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    json.dump(chosen, auto); auto.close()
    rc = describe_mod.main(["--session", str(out), "--auto", auto.name, "--minimal-prompts"])
    if rc != 0:
        return rc

    # 5) SUBMIT ------------------------------------------------------------
    banner("5/5  Submit")
    print("This uploads ONLY the redacted session + manifest + consent, as a PR.")
    if ask("Proceed with upload? (y/N)", "N").lower() not in ("y", "yes"):
        print("[wizard] Stopped before upload. Your redacted files are saved locally:")
        print(f"   {out}")
        print("   Run `python -m donate.submit` later to upload.")
        return 0
    return submit_mod.main([str(out)])


if __name__ == "__main__":
    sys.exit(main())
