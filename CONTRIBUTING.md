# Contributing to ContextEcho

ContextEcho is a **living benchmark**. Its scientific value grows with every
real coding session it covers — more donors, more domains, more model
families, more session lengths. We actively welcome contributions, and we
credit them.

This document explains **how to contribute** and, just as importantly,
**what you get for contributing**.

---

## TL;DR — what you get

| You do | You get |
|--------|---------|
| Run ContextEcho on your own session | A **free persona-drift report** on your own AI tooling — even if you never contribute. |
| Contribute a qualifying session or annotation | A spot on the [**contributor leaderboard**](CONTRIBUTORS.md) + acknowledgment in the next release. |
| Reach the contribution threshold (see below) | **Co-authorship on the next dataset release** (author order by contribution). |

Credit is awarded on a transparent, points-based scale modeled on
[Terminal-Bench](https://www.tbench.ai/docs/contributing) and
[BIG-bench](https://github.com/google/BIG-bench). No contribution goes
unrecognized; substantial contributions earn authorship.

---

## The free self-audit (start here)

Before you contribute anything, run ContextEcho on one of your own Claude Code
sessions. You get back a personalized report:

> *"Your session showed 14× verbosity inflation by turn 5,000. Your model
> stopped honoring 'no-preamble' instructions after the second compaction.
> Here is the ~110-token anchor that restores the trained register for your
> workflow."*

This is yours to keep whether or not you donate the session. See
[`REPRODUCE.md`](REPRODUCE.md) for how to run it.

---

## Ways to contribute

You do **not** have to donate a whole session. Contribution is decomposed so
anyone can find a low-effort entry point:

| Contribution type | Points | Notes |
|-------------------|:------:|-------|
| **Donate a qualifying session** | 2 | A real agentic-coding (or related) session that passes the PII + in-scope review. New domains / languages / model families are especially valuable. |
| **Build a provider adapter** | 4 | Wire a new chat-completions API target into the harness, with a parity check. |
| **Annotate an existing session** | 1 | Label drift onset, rank drift severity, or validate the redaction of a donated session. |
| **Engineering / analysis** | 1–4 | New probes, scorers, figures, or analysis. Counts toward authorship only with meaningful effort (≈15+ hours). |

**Points are awarded only after your contribution is reviewed and merged.**

---

## Credit tiers

| Total points | Recognition |
|:------------:|-------------|
| Any (≥1) | Listed on the [contributor leaderboard](CONTRIBUTORS.md) + named in the release acknowledgments. |
| **≥ 6** | **Co-authorship on the next dataset release** (e.g. the v2 / living-benchmark paper). Author order is set by point total plus intangible contributions. |

> **Note on authorship and review timelines.** Co-authorship applies to the
> **next** dataset release, never to a paper currently under peer review.
> Donating a session does not by itself confer authorship — authorship is
> reserved for substantial contributions that clear the threshold above. This
> keeps the author line meaningful and the contributor list welcoming to
> everyone.

We also follow a **rolling re-authorship** model: each versioned release
(v2, v3, …) invites the contributors who cleared the threshold *for that
release* as authors. Missing the first paper does not close the door.

---

## How to submit a session (the pipeline)

> **Privacy is local-first.** You redact your session **on your own machine**
> and only ever upload already-clean data. We never receive your raw session.
> A pull request is public the moment it is opened, so all scrubbing must
> happen **before** you open it.

1. **Capture** — export the Claude Code session you want to donate.
2. **Redact locally** — run the redaction tool. It scrubs home paths, emails,
   API keys, tokens, IPs, and a scrub-list of your own identifiers.
3. **Verify** — run the PII verifier. It must report **zero residual PII**
   before you proceed. If it flags anything, resolve it first.
4. **Consent** — fill in the donor consent form (CC-BY-SA-4.0).
5. **Submit** — fork the repo, add your redacted session + consent under a new
   session ID, and open a pull request.

A maintainer (and CI) re-verifies PII on the PR. **Two independent reviewers**
confirm the session is in-scope and clean before it is merged and counts
toward your points.

### ⚠️ Confidentiality — read before donating

**Only donate sessions from personal projects, internal tooling, or
open-source work.** Do **not** donate sessions that contain client-confidential
code or data, material under NDA, or anyone else's personal data. When in
doubt, don't — and ask a maintainer.

---

## Quality and anti-gaming

To keep contributor credit meaningful:

- Points are awarded **only after merge**, never for opened-but-rejected PRs.
- Every donated session requires **two independent reviewers** to confirm it is
  in-scope and PII-clean.
- Annotations/validations from brand-new accounts do **not** count toward
  anyone's score (to prevent vote/score farming).
- Sessions must add genuine coverage; near-duplicate or padded sessions are not
  accepted.

---

## Questions

Open an issue on the repository, or reach the maintainers via the contact
listed in the [README](README.md). Thank you for helping make ContextEcho a
benchmark the whole community can rely on.
