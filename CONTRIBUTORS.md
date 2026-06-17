# ContextEcho Contributors

ContextEcho grows with every real session the community donates. This page
credits everyone who has contributed to the corpus, ranked by accepted
points. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for how to join and
what you get.

> Authorship of the dataset paper is separate from this list: it is reserved
> for contributors who clear the points threshold in
> [`CONTRIBUTING.md`](CONTRIBUTING.md). Everyone here is credited in the
> release acknowledgments.

---

## Contributor Leaderboard

Ranked by accepted points, then accepted unique sessions, then total user turns.
Points follow the scale in [`CONTRIBUTING.md`](CONTRIBUTING.md).

| Rank | Contributor | Sessions | Turns | Agents | Models | Points |
|:----:|-------------|:--------:|------:|--------|--------|:------:|
| 🥇 | Anonymous donor S1 | 1 | 9,716 | Claude Code | Opus 4.x (mixed) | 5 |
| 🥈 | Anonymous donor S3 | 1 | 4,918 | Claude Code | Opus 4.x (mixed) | 5 |
| 🥉 | Anonymous donor S2 | 1 | 3,746 | Claude Code | Opus 4.x (mixed) | 5 |
| 4 | Anonymous donor 95c3332b | 1 | 2,509 | Claude Code | claude-opus-4.x (mixed) | 5 |
| 5 | Anonymous donor d016759e | 1 | 1,611 | Claude Code | claude-opus-4.x (mixed) | 5 |
| 6 | Anonymous donor ae4ddeac | 1 | 829 | Claude Code | claude-opus-4.x (mixed) | 5 |
| 7 | Anonymous donor 97289703 | 1 | 564 | Codex CLI | gpt-5.5 | 5 |
| 8 | Anonymous donor 5102291c | 1 | 449 | Codex CLI | gpt-5.4 | 5 |
| 9 | Anonymous donor d9b5dd55 | 1 | 307 | Codex CLI | gpt-5.5 | 5 |

*Corpus total: **9 sessions · 24,649 user turns**.*

> Anonymous donors are assigned stable session nicknames unless they provide
> name, email, and institute. Contributions are merged only when all three
> identity fields match exactly after normalization.

---

## Session Ledger

Each donated session declares the **agent/harness** it was driven by, the
**model** it ran, the model's **organization**, the **task domain** and primary
**language**, and its **scale** (turns / compactions). Duplicate privacy-tier
variants can be accepted for analysis, but only the first unique source session
per contributor counts toward points.

| ID | Donor | Agent | Model | Org | Domain | Lang | Turns | Cmp | Pts | Status |
|----|-------|-------|-------|-----|--------|------|------:|:---:|:---:|--------|
| S1 | Anonymous donor S1 | Claude Code | Opus 4.x | Anthropic | agentic-coding | Python | 9,716 | 6 | 5 | v1.0 |
| S2 | Anonymous donor S2 | Claude Code | Opus 4.x | Anthropic | manuscript-writing | mixed | 3,746 | 3 | 5 | v1.0 |
| S3 | Anonymous donor S3 | Claude Code | Opus 4.x | Anthropic | non-coding-docs | mixed | 4,918 | 4 | 5 | v1.0 |
| S4 | Anonymous donor 5102291c | Codex CLI | gpt-5.4 | OpenAI | agentic-coding | mixed | 449 | 5 | 5 | v2 |
| S5 | Anonymous donor 95c3332b | Claude Code | claude-opus-4.x (mixed) | Anthropic | agentic-coding | mixed | 2,509 | 17 | 5 | v2 |
| S6 | Anonymous donor 97289703 | Codex CLI | gpt-5.5 | OpenAI | agentic-coding | mixed | 564 | 8 | 5 | v2 |
| S7 | Anonymous donor ae4ddeac | Claude Code | claude-opus-4.x (mixed) | Anthropic | agentic-coding | mixed | 829 | 10 | 5 | v2 |
| S8 | Anonymous donor d016759e | Claude Code | claude-opus-4.x (mixed) | Anthropic | web-frontend | mixed | 1,611 | 15 | 5 | v2 |
| S9 | Anonymous donor d9b5dd55 | Codex CLI | gpt-5.5 | OpenAI | agentic-coding | mixed | 307 | 5 | 5 | v2 |

---

## Coverage Map

The benchmark's value is in its diversity. Donating a session that fills a
new coverage gap can earn a novelty bonus.

| Axis | Covered so far | Wanted |
|------|----------------|--------|
| **Agent / harness** | Claude Code · Codex CLI | Cursor · Aider · Windsurf · Cline · Continue · custom harnesses |
| **Model** | Opus 4.x (mixed) · claude-opus-4.x (mixed) · +2 | Gemini · DeepSeek · Llama · Qwen · Mistral · Kimi · any frontier model |
| **Organization** | Anthropic · OpenAI | Google · Meta · DeepSeek · Alibaba · Mistral · Cohere · NVIDIA · Moonshot |
| **Domain** | Agentic Coding · Manuscript Writing · +2 | data science · web/frontend · infra/DevOps · debugging · research · refactoring |
| **Language** | Python · mixed | TypeScript/JS · Rust · Go · Java · C++ · SQL · non-English natural language |

---

## How This List Is Maintained

This file is **auto-generated** from `data_archive_release_v2/data/donations/ledger.jsonl`
plus the anonymized v1 founding-session metadata. Do not edit it by hand.
Regenerate it with:

```bash
make update-contributors
```
