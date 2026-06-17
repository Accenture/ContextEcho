"""Regenerate DATASET_CARD.md from public release metadata."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

try:
    from update_contributors import (
        FOUNDING_SESSIONS,
        SessionEntry,
        group_contributors,
        iter_jsonl,
        load_ledger_sessions,
        md_escape,
        norm,
        score_sessions,
    )
except ModuleNotFoundError:
    from scripts.update_contributors import (
        FOUNDING_SESSIONS,
        SessionEntry,
        group_contributors,
        iter_jsonl,
        load_ledger_sessions,
        md_escape,
        norm,
        score_sessions,
    )

FOUNDING_CELL_JSONS = 41921
FOUNDING_DATA_SIZE = "310 MB redacted sessions + 705 MB per-cell evaluations"


@dataclass
class LedgerCounts:
    rows: int = 0
    acceptable: int = 0
    superseded: int = 0
    check_required: int = 0
    duplicate: int = 0


def load_ledger_counts(dataset_root: Path) -> LedgerCounts:
    counts = LedgerCounts()
    for row in iter_jsonl(dataset_root / "data" / "donations" / "ledger.jsonl"):
        counts.rows += 1
        decision = norm(row.get("decision"))
        if decision == "ACCEPTABLE":
            counts.acceptable += 1
        elif decision == "SUPERSEDED":
            counts.superseded += 1
        elif decision == "CHECK_REQUIRED":
            counts.check_required += 1
        elif decision == "DUPLICATE":
            counts.duplicate += 1
    return counts


def value_counts(values: list[str], limit: int = 8) -> str:
    clean = [v for v in values if v]
    if not clean:
        return "none yet"
    counts = Counter(clean)
    parts = [f"{md_escape(value)} ({count})" for value, count in counts.most_common(limit)]
    if len(counts) > limit:
        parts.append(f"+{len(counts) - limit} more")
    return ", ".join(parts)


def coverage_count(values: list[str], singular: str, plural: str) -> str:
    count = len(set(v for v in values if v))
    label = singular if count == 1 else plural
    return f"{count} {label}"


def institution_coverage(sessions: list[SessionEntry]) -> list[str]:
    # Emails and donor-level identity links stay out of the public card. The
    # aggregate coverage count can still include public-anonymous donations.
    return [s.institute for s in sessions if s.institute]


def render_dataset_card(dataset_root: Path = Path("data_archive_release_v2")) -> str:
    founding = [SessionEntry(**vars(s)) for s in FOUNDING_SESSIONS]
    promoted = load_ledger_sessions(dataset_root)
    sessions = founding + promoted
    score_sessions(sessions)
    contributors = group_contributors(sessions)
    counted = [s for s in sessions if s.counted]
    counts = load_ledger_counts(dataset_root)
    v2_note = (
        f"{counts.acceptable} promoted accepted donation(s)"
        if counts.acceptable
        else "No promoted v2 candidate donations in the local public ledger yet"
    )

    lines: list[str] = [
        "# ContextEcho Dataset Card",
        "",
        "This card summarizes the public ContextEcho dataset release and the rolling",
        "donation pipeline used to build later release candidates. It is generated",
        "from public release metadata; donor emails and private reviewer fields are",
        "not included.",
        "",
        "## Dataset Summary",
        "",
        "| Field | Value |",
        "|-------|-------|",
        "| Name | ContextEcho persona-drift benchmark |",
        "| Repository | https://github.com/Accenture/ContextEcho |",
        "| Dataset host | https://huggingface.co/datasets/contextecho2026/persona-drift-contextecho |",
        "| License | CC-BY-SA-4.0 for data; Apache-2.0 for code |",
        "| Public v1 founding sessions | 3 |",
        f"| Public v1 per-cell evaluations | {FOUNDING_CELL_JSONS:,} |",
        f"| Public v1 data size | {FOUNDING_DATA_SIZE} |",
        f"| Active public/candidate sessions tracked locally | {len(counted)} |",
        f"| Active public/candidate user turns tracked locally | {sum(s.turns for s in counted):,} |",
        f"| Active public/candidate context compactions tracked locally | {sum(s.compactions for s in counted):,} |",
        f"| Public contributors in leaderboard | {len(contributors)} |",
        f"| V2 promotion ledger status | {v2_note} |",
        "",
        "## Composition",
        "",
        "| Axis | Values |",
        "|------|--------|",
        f"| Agent / harness | {md_escape(value_counts([s.agent for s in counted]))} |",
        f"| Model family | {md_escape(value_counts([s.model for s in counted]))} |",
        f"| Model organization | {md_escape(value_counts([s.org for s in counted]))} |",
        f"| Task domain | {md_escape(value_counts([s.domain for s in counted]))} |",
        f"| Primary language | {md_escape(value_counts([s.language for s in counted]))} |",
        f"| Privacy tier | {md_escape(value_counts([s.privacy_tier for s in counted if s.privacy_tier]))} |",
        f"| Institution coverage | {md_escape(coverage_count(institution_coverage(counted), 'institution', 'institutions'))} |",
        "",
        "## Donation And Promotion Pipeline",
        "",
        "1. Donors run the local browser wizard and select a real coding-agent session.",
        "2. Redaction and verification run on the donor machine before upload.",
        "3. The relay accepts only verified redacted artifacts and opens private staging submissions.",
        "4. Maintainers run technical review, PII checks, consent checks, and quick scientific validation.",
        "5. Accepted donations are promoted into `data_archive_release_v2/` and appended to the public ledger.",
        "6. `CONTRIBUTORS.md` and this `DATASET_CARD.md` are regenerated from the same ledger.",
        "",
        "## Lineage And Deduplication",
        "",
        "The intake pipeline tracks `source_session_id` and `conversation_fingerprint`",
        "where available. Exact duplicate redacted artifacts are rejected. Same-lineage",
        "updates are accepted only when the session has grown substantially; when a",
        "new accepted update supersedes an older public row, the older ledger row is",
        "marked `SUPERSEDED` and stops counting as an active session.",
        "",
        "## Public Credit And Privacy",
        "",
        "Donors may submit maintainer-visible name, email, and institute fields while",
        "choosing to appear publicly as anonymous. Public leaderboard names and this",
        "dataset card never publish donor email addresses or donor-to-institution",
        "links. Institution coverage is reported only as aggregate dataset",
        "composition.",
        "",
        "## Current Ledger Counts",
        "",
        "| Ledger state | Count |",
        "|--------------|------:|",
        f"| Total rows | {counts.rows} |",
        f"| ACCEPTABLE | {counts.acceptable} |",
        f"| SUPERSEDED | {counts.superseded} |",
        f"| CHECK_REQUIRED | {counts.check_required} |",
        f"| DUPLICATE | {counts.duplicate} |",
        "",
        "## Maintenance",
        "",
        "This file is auto-generated. Do not edit it by hand. Regenerate public",
        "release metadata after promotion with:",
        "",
        "```bash",
        "make update-release-metadata",
        "```",
        "",
        "Verify it with:",
        "",
        "```bash",
        "make check-release-metadata",
        "```",
        "",
    ]
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Regenerate DATASET_CARD.md from public release metadata.")
    p.add_argument("--dataset-root", type=Path, default=Path("data_archive_release_v2"))
    p.add_argument("--out", type=Path, default=Path("DATASET_CARD.md"))
    p.add_argument("--check", action="store_true", help="fail if the output file is not up to date")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rendered = render_dataset_card(args.dataset_root)
    if args.check:
        current = args.out.read_text(encoding="utf-8") if args.out.exists() else ""
        if current != rendered:
            print(f"[dataset-card] stale: {args.out}")
            return 1
        print(f"[dataset-card] up to date: {args.out}")
        return 0
    args.out.write_text(rendered, encoding="utf-8")
    print(f"[dataset-card] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
