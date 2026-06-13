"""Regenerate CONTRIBUTORS.md from accepted donation ledgers."""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class SessionEntry:
    sid: str
    contributor: str
    email: str = ""
    institute: str = ""
    agent: str = ""
    model: str = ""
    org: str = ""
    domain: str = ""
    language: str = ""
    turns: int = 0
    compactions: int = 0
    status: str = ""
    submission_id: str = ""
    source_key: str = ""
    privacy_tier: str = ""
    promoted_utc: str = ""
    points: int = 0
    counted: bool = True


@dataclass
class Contributor:
    key: tuple[str, ...]
    name: str
    email: str = ""
    institute: str = ""
    sessions: list[SessionEntry] = field(default_factory=list)

    @property
    def counted_sessions(self) -> list[SessionEntry]:
        return [s for s in self.sessions if s.counted]

    @property
    def points(self) -> int:
        return sum(s.points for s in self.counted_sessions)

    @property
    def turns(self) -> int:
        return sum(s.turns for s in self.counted_sessions)


FOUNDING_SESSIONS = [
    SessionEntry(
        sid="S1",
        contributor="Anonymous donor S1",
        agent="Claude Code",
        model="Opus 4.x (mixed)",
        org="Anthropic",
        domain="agentic-coding",
        language="Python",
        turns=9716,
        compactions=6,
        status="v1.0",
        source_key="founding-s1",
        promoted_utc="2026-01-01T00:00:00+00:00",
    ),
    SessionEntry(
        sid="S2",
        contributor="Anonymous donor S2",
        agent="Claude Code",
        model="Opus 4.x (mixed)",
        org="Anthropic",
        domain="manuscript-writing",
        language="mixed",
        turns=3746,
        compactions=3,
        status="v1.0",
        source_key="founding-s2",
        promoted_utc="2026-01-02T00:00:00+00:00",
    ),
    SessionEntry(
        sid="S3",
        contributor="Anonymous donor S3",
        agent="Claude Code",
        model="Opus 4.x (mixed)",
        org="Anthropic",
        domain="non-coding-docs",
        language="mixed",
        turns=4918,
        compactions=4,
        status="v1.0",
        source_key="founding-s3",
        promoted_utc="2026-01-03T00:00:00+00:00",
    ),
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def norm(value: Any) -> str:
    return str(value or "").strip()


def display_name(record: dict[str, Any], fallback: str) -> str:
    if bool(record.get("public_anonymous")):
        return fallback
    name = norm(record.get("credit_name") or record.get("contributor"))
    if not name or name.lower() in {"anonymous", "anon", "donor"}:
        return fallback
    return name


def anonymous_ledger_name(row: dict[str, Any], sid: str) -> str:
    submission = norm(row.get("submission_id"))
    if submission.startswith("submission-"):
        return f"Anonymous donor {submission.removeprefix('submission-')}"
    if submission:
        return f"Anonymous donor {submission}"
    return f"Anonymous donor {sid}"


def merge_key(name: str, email: str, institute: str, unique: str) -> tuple[str, ...]:
    """Merge only when name, email, and institute are all present and equal."""
    if name and email and institute and not name.lower().startswith("anonymous donor"):
        return ("identified", name.lower(), email.lower(), institute.lower())
    return ("unique", unique)


def load_ledger_sessions(dataset_root: Path) -> list[SessionEntry]:
    ledger = dataset_root / "data" / "donations" / "ledger.jsonl"
    rows = iter_jsonl(ledger)
    out: list[SessionEntry] = []
    for i, row in enumerate(rows, start=4):
        manifest: dict[str, Any] = {}
        manifest_path = dataset_root / norm(row.get("manifest_path"))
        if manifest_path.exists():
            manifest = read_json(manifest_path)
        sid = f"S{i}"
        public_record = dict(row)
        if manifest.get("public_anonymous"):
            public_record["public_anonymous"] = True
        name = display_name(public_record, anonymous_ledger_name(row, sid))
        email = norm(manifest.get("contributor_email") or row.get("contributor_email"))
        institute = norm(row.get("institute") or manifest.get("contributor_institute"))
        source_key = norm(manifest.get("redacted_file")) or norm(row.get("session_sha256")) or norm(row.get("submission_id"))
        out.append(
            SessionEntry(
                sid=sid,
                contributor=name,
                email=email,
                institute=institute,
                agent=norm(row.get("agent")),
                model=norm(row.get("model")),
                org=norm(row.get("org")),
                domain=norm(row.get("domain")),
                language=norm(row.get("language")) or "mixed",
                turns=as_int(row.get("turns")),
                compactions=as_int(row.get("compactions")),
                status="v2 candidate",
                submission_id=norm(row.get("submission_id")),
                source_key=source_key,
                privacy_tier=norm(row.get("privacy_tier")),
                promoted_utc=norm(row.get("promoted_utc")),
            )
        )
    return out


def score_sessions(sessions: list[SessionEntry]) -> None:
    seen_axes: set[tuple[str, str]] = set()
    counted_sources: set[str] = set()
    for session in sorted(sessions, key=lambda s: (s.promoted_utc, s.sid)):
        if session.source_key in counted_sources:
            session.counted = False
            session.points = 0
        else:
            counted_sources.add(session.source_key)
            high_value = session.turns >= 100 or session.compactions >= 1
            axes = {
                ("agent", session.agent.lower()),
                ("model", session.model.lower()),
                ("org", session.org.lower()),
                ("domain", session.domain.lower()),
                ("language", session.language.lower()),
            }
            new_coverage = any(value and axis not in seen_axes for axis, value in axes)
            usability = bool(session.agent and session.model and session.domain and session.language and session.turns)
            session.points = 2 + int(high_value) + int(new_coverage) + int(usability)
        for axis in (
            ("agent", session.agent.lower()),
            ("model", session.model.lower()),
            ("org", session.org.lower()),
            ("domain", session.domain.lower()),
            ("language", session.language.lower()),
        ):
            if axis[1]:
                seen_axes.add(axis)


def group_contributors(sessions: list[SessionEntry]) -> list[Contributor]:
    grouped: dict[tuple[str, ...], Contributor] = {}
    for session in sessions:
        key = merge_key(session.contributor, session.email, session.institute, session.sid)
        if key not in grouped:
            grouped[key] = Contributor(key=key, name=session.contributor, email=session.email, institute=session.institute)
        grouped[key].sessions.append(session)
    ranked = [c for c in grouped.values() if c.counted_sessions]
    return sorted(ranked, key=lambda c: (-c.points, -len(c.counted_sessions), -c.turns, c.name.lower()))


def short_set(values: list[str], fallback: str = "—") -> str:
    clean = sorted({v for v in values if v})
    if not clean:
        return fallback
    if len(clean) <= 2:
        return " · ".join(clean)
    return " · ".join(clean[:2]) + f" · +{len(clean) - 2}"


def medal(rank: int) -> str:
    return {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, str(rank))


def md_escape(text: str) -> str:
    return text.replace("|", "\\|")


def title_case_axis(value: str) -> str:
    return value.replace("-", " ").title() if value else "—"


def compact_model(value: str) -> str:
    return value.replace("Opus 4.x (mixed)", "Opus 4.x") if value else "—"


def compact_domain(value: str) -> str:
    return value or "—"


def compact_status(session: SessionEntry) -> str:
    if not session.counted:
        return "v2 dup"
    return session.status.replace(" candidate", "") or "—"


def render_contributors(contributors: list[Contributor], sessions: list[SessionEntry]) -> str:
    counted = [s for s in sessions if s.counted]
    total_sessions = len(counted)
    total_turns = sum(s.turns for s in counted)
    lines: list[str] = [
        "# ContextEcho Contributors",
        "",
        "ContextEcho grows with every real session the community donates. This page",
        "credits everyone who has contributed to the corpus, ranked by accepted",
        "points. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for how to join and",
        "what you get.",
        "",
        "> Authorship of the dataset paper is separate from this list: it is reserved",
        "> for contributors who clear the points threshold in",
        "> [`CONTRIBUTING.md`](CONTRIBUTING.md). Everyone here is credited in the",
        "> release acknowledgments.",
        "",
        "---",
        "",
        "## Contributor Leaderboard",
        "",
        "Ranked by accepted points, then accepted unique sessions, then total user turns.",
        "Points follow the scale in [`CONTRIBUTING.md`](CONTRIBUTING.md).",
        "",
        "| Rank | Contributor | Sessions | Turns | Agents | Models | Points |",
        "|:----:|-------------|:--------:|------:|--------|--------|:------:|",
    ]
    for i, contributor in enumerate(contributors, start=1):
        counted_sessions = contributor.counted_sessions
        lines.append(
            "| {rank} | {name} | {sessions} | {turns:,} | {agents} | {models} | {points} |".format(
                rank=medal(i),
                name=md_escape(contributor.name),
                sessions=len(counted_sessions),
                turns=contributor.turns,
                agents=md_escape(short_set([s.agent for s in counted_sessions])),
                models=md_escape(short_set([s.model for s in counted_sessions])),
                points=contributor.points,
            )
        )
    lines.extend([
        "",
        f"*Corpus total: **{total_sessions} sessions · {total_turns:,} user turns**.*",
        "",
        "> Anonymous donors are assigned stable session nicknames unless they provide",
        "> name, email, and institute. Contributions are merged only when all three",
        "> identity fields match exactly after normalization.",
        "",
        "---",
        "",
        "## Session Ledger",
        "",
        "Each donated session declares the **agent/harness** it was driven by, the",
        "**model** it ran, the model's **organization**, the **task domain** and primary",
        "**language**, and its **scale** (turns / compactions). Duplicate privacy-tier",
        "variants can be accepted for analysis, but only the first unique source session",
        "per contributor counts toward points.",
        "",
        "| ID | Donor | Agent | Model | Org | Domain | Lang | Turns | Cmp | Pts | Status |",
        "|----|-------|-------|-------|-----|--------|------|------:|:---:|:---:|--------|",
    ])
    for session in sessions:
        lines.append(
            "| {sid} | {contributor} | {agent} | {model} | {org} | {domain} | {language} | {turns:,} | {compactions} | {points} | {status} |".format(
                sid=session.sid,
                contributor=md_escape(session.contributor),
                agent=md_escape(session.agent or "—"),
                model=md_escape(compact_model(session.model)),
                org=md_escape(session.org or "—"),
                domain=md_escape(compact_domain(session.domain)),
                language=md_escape(session.language or "—"),
                turns=session.turns,
                compactions=session.compactions,
                points=session.points if session.counted else 0,
                status=md_escape(compact_status(session)),
            )
        )
    axes = {
        "Agent / harness": [s.agent for s in counted],
        "Model": [s.model for s in counted],
        "Organization": [s.org for s in counted],
        "Domain": [title_case_axis(s.domain) for s in counted],
        "Language": [s.language for s in counted],
    }
    lines.extend([
        "",
        "---",
        "",
        "## Coverage Map",
        "",
        "The benchmark's value is in its diversity. Donating a session that fills a",
        "new coverage gap can earn a novelty bonus.",
        "",
        "| Axis | Covered so far | Wanted |",
        "|------|----------------|--------|",
    ])
    wanted = {
        "Agent / harness": "Cursor · Aider · Windsurf · Cline · Continue · custom harnesses",
        "Model": "Gemini · DeepSeek · Llama · Qwen · Mistral · Kimi · any frontier model",
        "Organization": "Google · Meta · DeepSeek · Alibaba · Mistral · Cohere · NVIDIA · Moonshot",
        "Domain": "data science · web/frontend · infra/DevOps · debugging · research · refactoring",
        "Language": "TypeScript/JS · Rust · Go · Java · C++ · SQL · non-English natural language",
    }
    for axis, values in axes.items():
        lines.append(f"| **{axis}** | {md_escape(short_set(values))} | {wanted[axis]} |")
    lines.extend([
        "",
        "---",
        "",
        "## How This List Is Maintained",
        "",
        "This file is **auto-generated** from `data_archive_release_v2/data/donations/ledger.jsonl`",
        "plus the anonymized v1 founding-session metadata. Do not edit it by hand.",
        "Regenerate it with:",
        "",
        "```bash",
        "make update-contributors",
        "```",
        "",
    ])
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Regenerate CONTRIBUTORS.md from accepted donations.")
    p.add_argument("--dataset-root", type=Path, default=Path("data_archive_release_v2"))
    p.add_argument("--out", type=Path, default=Path("CONTRIBUTORS.md"))
    p.add_argument("--check", action="store_true", help="fail if the output file is not up to date")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    sessions = [SessionEntry(**vars(s)) for s in FOUNDING_SESSIONS]
    sessions.extend(load_ledger_sessions(args.dataset_root))
    score_sessions(sessions)
    contributors = group_contributors(sessions)
    rendered = render_contributors(contributors, sessions)
    out = args.out
    if args.check:
        current = out.read_text(encoding="utf-8") if out.exists() else ""
        if current != rendered:
            print(f"[contributors] stale: {out}")
            return 1
        print(f"[contributors] up to date: {out}")
        return 0
    out.write_text(rendered, encoding="utf-8")
    print(f"[contributors] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
