# ContextEcho Contributors

ContextEcho grows with every real session the community donates. This page
credits everyone who has contributed to the corpus, ranked by accepted
contributions. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for how to join and
what you get.

> Authorship of the dataset paper is separate from this list: it is reserved
> for contributors who clear the points threshold in
> [`CONTRIBUTING.md`](CONTRIBUTING.md). Everyone here is credited in the
> release acknowledgments.

---

## Contributor leaderboard

Ranked by accepted sessions, then total turns. Points follow the scale in
[`CONTRIBUTING.md`](CONTRIBUTING.md).

| Rank | Contributor | Sessions | Turns | Agents | Models | Points |
|:----:|-------------|:--------:|------:|--------|--------|:------:|
| 🥇 | Founding donors | 3 | 18,380 | Claude Code | Opus 4.x | — |

*Corpus total: **3 sessions · 18,380 turns**.*

> The founding sessions (S1–S3) were donated by anonymized contributors under
> the project's consent terms; they seed the corpus released with the v1.0
> paper. As new sessions are merged, this leaderboard regenerates from the
> dataset manifest.

---

## Session ledger

Each donated session declares the **agent/harness** it was driven by, the
**model** it ran, the model's **organization**, the **task domain** and primary
**language**, and its **scale** (turns / compactions). These axes are what make
ContextEcho a *coverage* benchmark, not just a pile of logs — we want breadth
across agents, models, organizations, domains, and languages.

| ID | Agent / Harness | Model | Org | Domain | Language | Turns | Compactions | Status |
|----|-----------------|-------|-----|--------|----------|------:|:-----------:|--------|
| S1 | Claude Code | Opus 4.x (mixed) | Anthropic | Agentic coding | Python | 9,716 | 6 | ✅ v1.0 |
| S2 | Claude Code | Opus 4.x (mixed) | Anthropic | Manuscript writing | — | 3,746 | 3 | ✅ v1.0 |
| S3 | Claude Code | Opus 4.x (mixed) | Anthropic | Non-coding docs | — | 4,918 | 4 | ✅ v1.0 |

*Your session could be S4. See [`CONTRIBUTING.md`](CONTRIBUTING.md).*

---

## Coverage map

The benchmark's value is in its diversity. These are the axes we most want to
expand — donating a session that fills a **gap** (a new agent, model, org,
domain, or language) is worth more than another session in an already-covered
cell (see the novelty bonus in [`CONTRIBUTING.md`](CONTRIBUTING.md)).

| Axis | Covered so far | Wanted |
|------|----------------|--------|
| **Agent / harness** | Claude Code | Codex CLI · Cursor · Aider · Windsurf · Cline · Continue · custom harnesses |
| **Model** | claude-opus-4-7 | GPT-5 · Gemini 2.5 Pro · DeepSeek · Llama · Qwen · Mistral · Kimi · any frontier model |
| **Organization** | Anthropic | OpenAI · Google · Meta · DeepSeek · Alibaba · Mistral · Cohere · NVIDIA · Moonshot |
| **Domain** | agentic coding · manuscript writing · non-coding docs | data science · web/frontend · infra/DevOps · debugging · research · refactoring |
| **Language** | Python | TypeScript/JS · Rust · Go · Java · C++ · SQL · non-English natural language |
| **Scale** | 3.7k–9.7k turns | shorter (<2k) and longer (>15k) sessions |

---

## What to declare when you submit

When you open a PR with a session, include these fields in the session
metadata (the donation tool prompts for them; see
[`CONTRIBUTING.md`](CONTRIBUTING.md)):

| Field | Example | Why it matters |
|-------|---------|----------------|
| `agent` | `Codex CLI`, `Claude Code`, `Cursor`, `Aider` | The harness shapes the system prompt + tool loop that drives drift. |
| `model` | `gpt-5`, `claude-sonnet-4-6`, `gemini-2.5-pro` | The unit of analysis — which model's persona drifted. |
| `org` | `OpenAI`, `Anthropic`, `Google` | Supports the cross-organization generality claim. |
| `domain` | `web-frontend`, `data-science`, `infra` | Task type; affects how drift surfaces. |
| `language` | `TypeScript`, `Rust`, `Python` | Primary programming (or natural) language of the session. |
| `turns` | `6204` | Session length. |
| `compactions` | `4` | Number of context-compaction events (a key drift axis). |

---

## How this list is maintained

This file is **auto-generated** from the dataset manifest on each merged
contribution — do not edit it by hand. Contributor names and session metadata
come from the merged consent + datasheet entries. Annotations and engineering
contributions (which do not add a session) are credited in the acknowledgments
of the next release and counted toward each contributor's points.
