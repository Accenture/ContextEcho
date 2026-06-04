"""Single-pass redactor for the public ContextEcho release.

Reads every text-like file under SRC_ROOTS, applies the substitution map
loaded from `.redaction_patterns.json`, and writes results to DST_ROOT
preserving directory structure. After the run, greps the output tree
for the donor-1 surface forms (canonical-placeholder `<USER>` etc.) and
reports zero hits as the release verification check.

The pattern list is loaded at runtime from a gitignored config file so
that the literal pattern strings (which would otherwise reveal donor
identity to a casual reader of the public anonymous-review code mirror)
do not ship inside this script. See `.redaction_patterns.json.example`
for the format.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SRC_ROOTS = [PROJECT_ROOT / "data", PROJECT_ROOT / "results"]
DST_ROOT = PROJECT_ROOT / "data_archive_release"
PATTERNS_PATH = Path(__file__).resolve().parent / ".redaction_patterns.json"

TEXT_SUFFIXES = {
    ".json", ".jsonl", ".md", ".txt", ".csv", ".log", ".tsv",
    ".yaml", ".yml", ".py", ".sh", ".lock",
}


def load_substitutions() -> list[tuple[re.Pattern[str], str]]:
    """Load and compile the substitution panel from the gitignored config.

    Order matters: longer/more-specific patterns first so we don't
    double-substitute inside an already-substituted run (e.g., the email
    must be replaced before `<author>.<surname>` is substituted, otherwise
    the local-part collapses to <USER> and the @-tail dangles).
    """
    if not PATTERNS_PATH.exists():
        sys.exit(
            f"[error] {PATTERNS_PATH.name} not found.\n"
            f"        Copy {PATTERNS_PATH.name}.example to "
            f"{PATTERNS_PATH.name} and fill in the donor-specific\n"
            f"        patterns before running the redactor.\n"
            f"        See REPRODUCE.md for the methodology."
        )
    cfg = json.loads(PATTERNS_PATH.read_text())
    compiled: list[tuple[re.Pattern[str], str]] = []
    for entry in cfg.get("patterns", []):
        flags = re.IGNORECASE if entry.get("ignore_case", False) else 0
        compiled.append((re.compile(entry["regex"], flags), entry["replacement"]))
    return compiled


SUBSTITUTIONS = load_substitutions()


def redact(text: str) -> str:
    for pattern, replacement in SUBSTITUTIONS:
        text = pattern.sub(replacement, text)
    return text


def should_redact(path: Path) -> bool:
    return path.suffix.lower() in TEXT_SUFFIXES


def relpath_from_src(path: Path) -> Path:
    for src_root in SRC_ROOTS:
        try:
            return Path(src_root.name) / path.relative_to(src_root)
        except ValueError:
            continue
    raise RuntimeError(f"path {path} not under any SRC_ROOT")


def process_file(src: Path, dst: Path) -> tuple[int, int]:
    """Returns (bytes_in, bytes_out_changed_count)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not should_redact(src):
        # Binary or non-text — copy as-is.
        shutil.copy2(src, dst)
        return (src.stat().st_size, 0)
    raw = src.read_text(encoding="utf-8", errors="replace")
    redacted = redact(raw)
    dst.write_text(redacted, encoding="utf-8")
    return (len(raw), len(raw) - len(redacted))  # rough delta


def walk_files(root: Path):
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if name == ".DS_Store":
                continue
            yield Path(dirpath) / name


def run_verification() -> int:
    """Audit-only mode: grep the release tree for the substitution panel.

    Returns 0 if every pattern reports 0 hits; non-zero on any leak.
    """
    if not DST_ROOT.exists():
        print(f"[error] {DST_ROOT} not found — run the redactor first", file=sys.stderr)
        return 2
    cfg = json.loads(PATTERNS_PATH.read_text())
    audit_terms = sorted({entry["regex"] for entry in cfg.get("patterns", [])})

    print("=== Verification: grep release tree for redaction surface forms ===")
    leak_count = 0
    for pattern in audit_terms:
        with os.popen(f"grep -rIlF {pattern!r} {DST_ROOT!s} 2>/dev/null | wc -l") as p:
            count = int(p.read().strip() or "0")
        marker = "OK" if count == 0 else "LEAK"
        print(f"  [{marker}] '{pattern}': {count} files")
        leak_count += count
    print()
    print("[ok] all surface forms clean" if leak_count == 0 else f"[fail] {leak_count} leaks")
    return 0 if leak_count == 0 else 1


def main(argv: list[str]) -> int:
    if "--verify-only" in argv:
        return run_verification()

    DST_ROOT.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    total_files = 0
    total_bytes_in = 0
    redacted_files = 0
    for src_root in SRC_ROOTS:
        if not src_root.exists():
            print(f"[warn] {src_root} missing, skipping", file=sys.stderr)
            continue
        for src in walk_files(src_root):
            rel = relpath_from_src(src)
            dst = DST_ROOT / rel
            bytes_in, delta = process_file(src, dst)
            total_files += 1
            total_bytes_in += bytes_in
            if delta != 0:
                redacted_files += 1
            if total_files % 500 == 0:
                print(f"  ...processed {total_files} files ({total_bytes_in / 1e6:.1f} MB)", flush=True)

    elapsed = time.time() - t0
    print(f"\n[ok] processed {total_files} files, {total_bytes_in / 1e6:.1f} MB in {elapsed:.1f}s")
    print(f"[ok] {redacted_files} files had at least one byte change")

    print()
    rc = run_verification()
    if rc == 0:
        print(f"\n[done] release tree: {DST_ROOT}")
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
