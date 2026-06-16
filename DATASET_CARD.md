# ContextEcho Dataset Card

This card summarizes the public ContextEcho dataset release and the rolling
donation pipeline used to build later release candidates. It is generated
from public release metadata; donor emails and private reviewer fields are
not included.

## Dataset Summary

| Field | Value |
|-------|-------|
| Name | ContextEcho persona-drift benchmark |
| Repository | https://github.com/Accenture/ContextEcho |
| Dataset host | https://huggingface.co/datasets/contextecho2026/persona-drift-contextecho |
| License | CC-BY-SA-4.0 for data; Apache-2.0 for code |
| Public v1 founding sessions | 3 |
| Public v1 per-cell evaluations | 41,921 |
| Public v1 data size | 310 MB redacted sessions + 705 MB per-cell evaluations |
| Active public/candidate sessions tracked locally | 3 |
| Active public/candidate user turns tracked locally | 18,380 |
| Active public/candidate context compactions tracked locally | 13 |
| Public contributors in leaderboard | 3 |
| V2 promotion ledger status | No promoted v2 candidate donations in the local public ledger yet |

## Composition

| Axis | Values |
|------|--------|
| Agent / harness | Claude Code (3) |
| Model family | Opus 4.x (mixed) (3) |
| Model organization | Anthropic (3) |
| Task domain | agentic-coding (1), manuscript-writing (1), non-coding-docs (1) |
| Primary language | mixed (2), Python (1) |
| Privacy tier | none yet |
| Public contributor institutions | none yet |

## Donation And Promotion Pipeline

1. Donors run the local browser wizard and select a real coding-agent session.
2. Redaction and verification run on the donor machine before upload.
3. The relay accepts only verified redacted artifacts and opens private staging submissions.
4. Maintainers run technical review, PII checks, consent checks, and quick scientific validation.
5. Accepted donations are promoted into `data_archive_release_v2/` and appended to the public ledger.
6. `CONTRIBUTORS.md` and this `DATASET_CARD.md` are regenerated from the same ledger.

## Lineage And Deduplication

The intake pipeline tracks `source_session_id` and `conversation_fingerprint`
where available. Exact duplicate redacted artifacts are rejected. Same-lineage
updates are accepted only when the session has grown substantially; when a
new accepted update supersedes an older public row, the older ledger row is
marked `SUPERSEDED` and stops counting as an active session.

## Public Credit And Privacy

Donors may submit maintainer-visible name, email, and institute fields while
choosing to appear publicly as anonymous. Public leaderboard names and this
dataset card never publish donor email addresses. Institution counts include
only public non-anonymous contributor rows with an institution value.

## Current Ledger Counts

| Ledger state | Count |
|--------------|------:|
| Total rows | 0 |
| ACCEPTABLE | 0 |
| SUPERSEDED | 0 |
| CHECK_REQUIRED | 0 |
| DUPLICATE | 0 |

## Maintenance

This file is auto-generated. Do not edit it by hand. Regenerate public
release metadata after promotion with:

```bash
make update-release-metadata
```

Verify it with:

```bash
make check-release-metadata
```
