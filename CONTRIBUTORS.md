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
| 4 | Xianzhong Ding | 1 | 449 | Codex CLI | gpt-5.4 | 5 |

*Corpus total: **4 sessions · 18,829 user turns**.*

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
| S4 | Xianzhong Ding | Codex CLI | gpt-5.4 | OpenAI | agentic-coding | mixed | 449 | 5 | 5 | v2 |
| S5 | Xianzhong Ding | Codex CLI | gpt-5.4 | OpenAI | agentic-coding | mixed | 449 | 5 | 0 | v2 dup |
| S6 | Xianzhong Ding | Codex CLI | gpt-5.4 | OpenAI | agentic-coding | mixed | 449 | 5 | 0 | v2 dup |

---

## Coverage Map

The benchmark's value is in its diversity. Donating a session that fills a
new coverage gap can earn a novelty bonus.

| Axis | Covered so far | Wanted |
|------|----------------|--------|
| **Agent / harness** | Claude Code · Codex CLI | Cursor · Aider · Windsurf · Cline · Continue · custom harnesses |
| **Model** | Opus 4.x (mixed) · gpt-5.4 | Gemini · DeepSeek · Llama · Qwen · Mistral · Kimi · any frontier model |
| **Organization** | Anthropic · OpenAI | Google · Meta · DeepSeek · Alibaba · Mistral · Cohere · NVIDIA · Moonshot |
| **Domain** | Agentic Coding · Manuscript Writing · +1 | data science · web/frontend · infra/DevOps · debugging · research · refactoring |
| **Language** | Python · mixed | TypeScript/JS · Rust · Go · Java · C++ · SQL · non-English natural language |

---

## How This List Is Maintained

This file is **auto-generated** from `data_archive_release_v2/data/donations/ledger.jsonl`
plus the anonymized v1 founding-session metadata. Do not edit it by hand.
Regenerate it with:

```bash
make update-contributors
```
