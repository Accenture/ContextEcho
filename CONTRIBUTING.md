# Contributing to ContextEcho

ContextEcho is a **living benchmark**. Its scientific value grows with every
real coding session it covers — more donors, more domains, more model
families, more session lengths. We actively welcome contributions, and we
credit them.

This document explains **how to contribute** and, just as importantly,
**what you get for contributing**.

> **Donor privacy.** ContextEcho analyzes assistant behavior, not donor
> personality. The default mode is **full redacted**, which removes PII,
> secrets, paths, and custom scrub terms while preserving task flow. Donors can
> choose **user-minimized** mode to selectively mask sensitive donor-authored
> spans after redaction. See [`DONOR_PRIVACY.md`](DONOR_PRIVACY.md) and
> [`DATA_USE_POLICY.md`](DATA_USE_POLICY.md).

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
| **Donate a qualifying session** | 2–5 | A real agentic-coding (or related) session that passes the PII + in-scope review. See the session scoring rule below. |
| **Build a provider adapter** | 4 | Wire a new chat-completions API target into the harness, with a parity check. |
| **Annotate an existing session** | 1 | Label drift onset, rank drift severity, or validate the redaction of a donated session. |
| **Engineering / analysis** | 1–4 | New probes, scorers, figures, or analysis. Counts toward authorship only with meaningful effort (≈15+ hours). |

**Points are awarded only after your contribution is reviewed and merged.**

### Session scoring rule

Accepted donated sessions are scored from the public donation ledger, not from
private staging uploads:

| Session outcome | Points | Rule |
|-----------------|:------:|------|
| **Accepted unique session** | +2 | Passes technical review, redaction verify, consent, manifest/session match, and quick validation. |
| **High-value bonus** | +1 | Long session or context-compaction-rich session, e.g. `turns >= 100` user prompts or `compactions >= 1`. |
| **New coverage bonus** | +1 | Adds a useful new axis: agent, model family, organization, task domain, language, or session type. |
| **Usability bonus** | +1 | Clean metadata and no maintainer repair required. |
| **Duplicate / rejected / unsafe** | 0 | Same redacted-session hash, failed privacy checks, missing consent, or out-of-scope/confidential content. |

The normal accepted short session is worth **2 points**. A high-value session
can earn up to **5 points**. Maintainers may apply judgment for unusual
high-impact contributions, but the ledger remains the default source of truth.

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
> Uploads go to a private maintainer staging area first; nothing becomes public
> until maintainers accept it into a release.

Fastest donor path:

```bash
curl -Ls https://raw.githubusercontent.com/Accenture/ContextEcho/main/scripts/run-donate.sh | bash
```

This bootstraps a private `uv` runner with `python3` if needed, then launches
the local browser wizard from GitHub with Python 3.11. Verified donations
upload through the official ContextEcho relay:

```text
https://contextecho2026-context-echo-donation-relay.hf.space
```

If you already have `pipx` and Python 3.10+, you can run the package directly:

```bash
CONTEXTECHO_RELAY_URL=https://contextecho2026-context-echo-donation-relay.hf.space \
pipx run --no-cache --spec git+https://github.com/Accenture/ContextEcho.git contextecho-donate
```

From a cloned checkout:

```bash
make setup-donate
python3 -m donate --web
```

The browser wizard runs locally and guides you through:

1. **Discover** — find local Claude Code and Codex CLI sessions.
2. **Pick** — choose one session to donate.
3. **Redact + verify** — scrub names, emails, paths, usernames, URLs, API keys,
   tokens, and optional extra terms on your machine.
4. **Review** — reveal the redacted file and optionally search for missed
   private terms before continuing.
5. **Describe + consent** — write `manifest.json` and `CONSENT.md` with credit
   information and CC-BY-SA-4.0 consent.
6. **Submit** — upload only the verified redacted session, manifest, and consent
   as a private staging pull request for maintainer review.

The terminal fallback is:

```bash
python3 -m donate
```

Maintainers re-run PII/secret checks, JSONL validation, consent checks, and a
quick 30-cell scientific validation before a session counts toward points.

ContextEcho does not use donations for donor profiling, psychological analysis,
sentiment analysis of donors, or deanonymization; see
[`DATA_USE_POLICY.md`](DATA_USE_POLICY.md).
Accepted sessions are promoted into the next public dataset candidate, not
directly into the live public dataset:

```bash
make intake-donations RUN_QUICK=1 PROMOTE=1
```

This writes release-ready files under `data_archive_release_v2/`, including the
redacted session, manifest, consent, review report, and public donation ledger.
See [`MAINTAINER_DONATION_WORKFLOW.md`](MAINTAINER_DONATION_WORKFLOW.md) for
the full donor-to-ledger workflow, contributor ranking rules, and maintainer
checklist.

Before opening a real collection round after testing, maintainers can reset
local test intake state:

```bash
make reset-donation-test-state
```

The reset command archives local test artifacts first, then clears
`data_archive_release_v2/`, `hf_staging_download/`, and
`results_v2_candidate/`. It does not delete private Hugging Face staging
submissions.

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
