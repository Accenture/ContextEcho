# Datasheet: ContextEcho v2.0

A datasheet for the ContextEcho corpus and benchmark, following the structure
of *Datasheets for Datasets* (Gebru et al., 2021). It covers the v2.0 release:
the 54 accepted community-donated sessions in `data_archive_release_v2/`
(the directory and ledger also carry one superseded duplicate submission,
retained for provenance and excluded from all analyses) plus the 3
founding sessions from the v1 release in `data_archive_release/` — 57
analysis sessions in total — together with the donation ledger and the
derived baseline results tables.

Answers below are synthesized from the project's governance documents —
[`DONOR_PRIVACY.md`](DONOR_PRIVACY.md), [`DATA_USE_POLICY.md`](DATA_USE_POLICY.md),
[`MAINTAINER_DONATION_WORKFLOW.md`](MAINTAINER_DONATION_WORKFLOW.md),
[`DONATION_RELAY.md`](DONATION_RELAY.md), [`CONTRIBUTING.md`](CONTRIBUTING.md),
the per-donation `CONSENT.md` files under `data_archive_release_v2/data/donations/`,
the v1 datasheet (`data_archive_release/DATASHEET.md`), and the paper appendix.
Where a standard datasheet question has no documented answer, this file says
**"Not documented"** and states what would answer it, rather than inventing one.

- Machine-readable metadata, file hashes, and field schemas: see
  [`croissant.json`](croissant.json) (MLCommons Croissant 1.0).
- Composition statistics and ledger counts: see the auto-generated
  [`DATASET_CARD.md`](DATASET_CARD.md).

---

## 1. Motivation

**For what purpose was the dataset created?**
ContextEcho measures whether a frontier LLM's trained Assistant persona
survives long agentic-coding sessions (thousands of tool-using turns, hours
of continuous use, including in-session context compactions) — the regime
real deployed coding agents run in, which short-dialog persona studies do not
cover. v1 (arXiv:2605.24279) established the phenomenon on 3 founding
sessions across a 23-model panel. v2.0 scales the corpus to 57 consented real
sessions (54 accepted community donations + 3 founding) via a community
donation pipeline, making it a contamination-resistant,
versioned, **extensible** behavioral benchmark: new donations drop in through
the same pipeline, only new cells are measured, and results append.

**Who created the dataset and on behalf of which entity?**
The authors of the ContextEcho paper (Ding, Yu, Liu, and Zhao, per the
citation in [`README.md`](README.md)) built the harness and pipeline; the
corpus itself is contributed by a pseudonymous community of donors credited
in [`CONTRIBUTORS.md`](CONTRIBUTORS.md) (all current donors chose anonymous
public credit, shown as "Anonymous donor `<id>`"). Contributors who clear the
points threshold in [`CONTRIBUTING.md`](CONTRIBUTING.md) earn co-authorship
on the next dataset release ("rolling re-authorship").

**Who funded the creation of the dataset?**
The v1 datasheet states no external funding, with compute provided by author
institutions. Not documented separately for v2 — a statement covering the v2
API evaluation costs and relay hosting would answer this.

---

## 2. Composition

**What do the instances represent?**
Three kinds of instances:

1. **Session transcripts** (`data_archive_release_v2/data/sessions/`, 55
   files; `data_archive_release/data/sessions/`, 3 founding files). One JSONL
   file per real donated coding-agent session, in the harness's native record
   schema, post-redaction. Two schemas coexist:
   - *Claude Code JSONL*: one record per event with top-level `type`
     (`user`, `assistant`, `system`, `attachment`, `file-history-snapshot`,
     ...), `sessionId`, `uuid`/`parentUuid` (conversation DAG), `timestamp`,
     `isSidechain` (subagent threads), and `message` payloads whose
     `tool_use` blocks are the tool-call events. Compactions are marked by
     `type=system, subtype=compact_boundary`.
   - *Codex CLI JSONL*: every record is `{type, timestamp, payload}` with
     `type` in `session_meta | turn_context | event_msg | response_item |
     compacted`; tool calls are `response_item` records with
     `payload.type=function_call` / `custom_tool_call`. Compactions are
     marked by `type=compacted`.

2. **Donation ledger** (`data_archive_release_v2/data/donations/ledger.jsonl`).
   One append-only row per promoted donation: label, harness, model, org,
   domain, privacy tier, record/turn/compaction counts, per-field metadata
   confidence, SHA-256, structural conversation fingerprint, review decision
   (`ACCEPTABLE`/`SUPERSEDED`), and paths to that donation's `manifest.json`,
   `CONSENT.md`, and `review_report.json`.

3. **Derived baseline results** (`data_archive_release_v2/results/session_validation/`):
   `summary_v2.0.csv` (per-session mean drift gap from the compact_validation_v1
   run), `figure_inputs_v2.0.csv` (per-session lexical drift proxies and
   session-shape covariates), `compactions_rederived.csv` (compaction-count
   cross-validation across JSONL markers, manifests, and figure inputs — all
   sources agree).

**How many instances are there in total?**
**57 analysis sessions**: 54 accepted community donations (29 Claude Code +
25 Codex CLI) plus the 3 founding sessions (Claude Code), i.e. 32 Claude
Code + 25 Codex CLI. Per session, 28–2,484 user turns (one user + assistant
round per turn); 12,210 user turns in total; 42 of the 57 sessions contain at
least one in-session context compaction (169 compaction events). These are
the figures used in the paper; source: `figure_inputs_v2.0.csv` (57 rows).

Intake artifacts additionally include one **superseded duplicate submission**
(`submission-607d8b75`, decision `SUPERSEDED`; superseded by a later
submission from the same donor). It is kept in the ledger and session
directory for provenance and excluded from every analysis. Intake-level
counts over all 55 community files (54 accepted + 1 superseded) are 225,570
JSONL records, 40,787 tool-call records, and a ledger `turns` field summing
to 12,337 (12,281 over the 54 accepted rows); the ledger `turns` field is an
intake count under a broader turn definition than the paper's and should not
be compared with the 12,210 figure above. Compaction markers over the 55
files: 157 (155 over the 54 accepted). Full composition breakdown (models,
domains, privacy tiers, institutions): see
[`DATASET_CARD.md`](DATASET_CARD.md).

**Does the dataset contain all possible instances, or is it a sample?**
A convenience sample: donors are self-selected users of Claude Code and
Codex CLI (24 institutions covered). It is not a random sample of coding-agent
sessions, and the composition skews toward agentic-coding and web-frontend
work in English. The corpus is designed to keep growing via donations.

**Is there a label or target associated with each instance?**
Sessions themselves are unlabeled transcripts. The benchmark produces labels:
per-session drift gaps (real-context arm minus length-matched filler arm)
ship in `summary_v2.0.csv`; judge-free lexical proxies ship in
`figure_inputs_v2.0.csv`.

**Is any information missing from individual instances?**
Yes, deliberately: all PII, secrets, private paths, URLs, and usernames are
replaced with placeholders (see §4). Three sessions additionally use the
`user_minimized` tier, which masks sensitive donor-authored spans. Some
manifest fields (`model`, `compactions`) carry `metadata_confidence:
"medium"` — donor-reported values that maintainers could not fully verify;
compaction counts were independently re-derived and reconcile exactly
(`compactions_rederived.csv`).

**Are relationships between instances made explicit?**
Yes. Ledger rows join to session files via `session_path` and to per-donation
consent/review artifacts via `manifest_path`/`consent_path`/`review_report_path`.
Same-conversation re-donations are linked by a privacy-preserving structural
`conversation_fingerprint`; when a grown re-donation supersedes an older row,
the old row is marked `SUPERSEDED` (`supersedes_submission`/`superseded_by`).

Donor-level linkage: each submission (tool version ≥ 0.5) carries a one-way
hash of the donating machine's OS identifier (`donor_device_id`,
sha256 of hardware UUID + username). It is disclosed in the per-donation
consent, visible to maintainers only, used to group one donor's submissions
even when their typed name/email/institute differ and to show donors their
own donation history, and is **stripped from the public release** (the
release exporter classifies it as a dropped field). It cannot be reversed
to identify a machine. This is the mechanism behind donor-level clustering
statistics reported for the corpus.

**Are there recommended data splits?**
No train/test split — this is an evaluation corpus. Using it as training data
is out of scope for the benchmark (and share-alike licensed; see §6).

**Are there errors, sources of noise, or redundancies?**
- One ledger row is `SUPERSEDED` and does not count as an active session
  (population 55 rows / 54 accepted).
- Donor-reported manifest fields are confidence-tagged; grep-based compaction
  counting is known to over-count on sessions that discuss ContextEcho itself —
  the released counts use the top-level-type detector instead.
- The judge-scored baseline uses single samples per cell (integer judge
  scores, no per-cell CIs); treat per-session means accordingly.

**Is the dataset self-contained, or does it rely on external resources?**
Self-contained: transcripts, ledger, consent records, and derived tables ship
in the release; the harness and analysis code ship in this repository.

**Does the dataset contain data that might be considered confidential?**
By consent attestation, no: every donor certifies in their `CONSENT.md` that
the session contains no client-confidential code or data, no material under
NDA, and no other person's personal data, and that they ran and reviewed the
local redaction + verification step. Maintainer review re-checks PII before
promotion.

**Does the dataset contain data that might be offensive, insulting, or
threatening?**
Sessions are real coding work; they may contain mild frustration language.
The `user_minimized` tier exists specifically to mask donor-authored spans
such as private feelings and toxic language ([`DONOR_PRIVACY.md`](DONOR_PRIVACY.md)).
No systematic toxicity audit of the corpus is documented — a corpus-wide
toxicity scan report would answer this.

**Does the dataset relate to people? Does it identify sub-populations?**
The dataset's object of study is **assistant behavior, not donors**
([`DONOR_PRIVACY.md`](DONOR_PRIVACY.md)): it explicitly does not study donor
personality, sentiment, mental health, work performance, or private beliefs.
No demographic attributes are collected or released. Donors appear publicly
only as anonymous pseudonyms; institution coverage is reported only as an
aggregate count. Third parties mentioned inside sessions are handled by (a)
the donor's consent attestation that no other person's personal data is
included, and (b) the automated redaction pass (person names, emails, phone
numbers, IPs are detected and replaced; see §4).

**Is it possible to identify individuals from the dataset?**
The pipeline is designed to prevent this: local redaction with fail-closed
verification, salted username pseudonymization, maintainer PII review, and an
explicit disallowed-use ban on deanonymization attempts
([`DATA_USE_POLICY.md`](DATA_USE_POLICY.md)). Residual risk: transcripts are
long-form natural text and stylometric re-identification can never be fully
excluded; the disallowed-use policy is the backstop. **Known gap:** donation
`manifest.json` files and some ledger rows in the release archive retain
maintainer-facing identity fields (`contributor`, `contributor_email`,
`contributor_institute`) even for donors who selected anonymous public
credit — [`DONOR_PRIVACY.md`](DONOR_PRIVACY.md) states these fields are "not
intended for public dataset rows." These fields should be stripped from the
public archive before external distribution (see §7).

---

## 3. Collection process

**How was the data associated with each instance acquired?**
Sessions are naturally occurring work logs: donors' own Claude Code / Codex
CLI sessions, generated by their normal work, not by prompted tasks. The
donor runs a local browser/terminal wizard (`python3 -m donate --web`) that
discovers local sessions, lets the donor pick one, redacts and verifies it
**on the donor's machine**, and writes `session.redacted.jsonl`,
`manifest.json`, and `CONSENT.md`. Raw session history never leaves the donor
machine ([`MAINTAINER_DONATION_WORKFLOW.md`](MAINTAINER_DONATION_WORKFLOW.md)).

**What mechanisms or procedures were used to collect the data?**
Only the verified redacted artifacts are uploaded, through a server-side
relay ([`DONATION_RELAY.md`](DONATION_RELAY.md)) that holds the staging token
as a server secret, enforces artifact-presence/JSONL/manifest/consent checks
and upload size limits, rejects exact duplicates by SHA-256, links
same-conversation updates by structural fingerprint (rejected until they grow
≥20% or 50 turns), and writes an append-only audit log. Maintainer intake
(`make maintainer-intake`) then runs technical review (files, JSONL, PII,
consent, metadata), a quick 30-cell scientific validation, promotion into
`data_archive_release_v2/`, ledger append, and regeneration of the public
metadata ([`DATASET_CARD.md`](DATASET_CARD.md), [`CONTRIBUTORS.md`](CONTRIBUTORS.md)).

**Who was involved in the collection process and how were they compensated?**
Donors (community members) and the maintainer team. No monetary compensation:
donors receive a free persona-drift report on their own session, public
leaderboard credit on a transparent points scale, release acknowledgment, and
co-authorship on the next dataset release above a points threshold
([`CONTRIBUTING.md`](CONTRIBUTING.md)).

**Over what timeframe was the data collected?**
Community donations in the v2.0 archive were promoted between 2026-06-25 and
2026-07-15 (ledger `promoted_utc`); the underlying sessions predate their
donation dates. Founding sessions span November 2025 – April 2026 (v1
datasheet).

**Were any ethical review processes conducted (e.g., by an IRB)?**
No IRB review was sought. The documented rationale (v1 datasheet §3; paper
appendix "Consent and IRB"): the data subject is the LLM, not the donor —
donors are curators of their own LLM-interaction logs, comparable to API
users sharing their own session logs, and no donor demographics, behavior, or
attitudes are studied. Not documented: an independent ethics assessment of
the v2 community-collection phase specifically; a short IRB-exemption
determination or ethics-board memo would answer it.

**Did the individuals consent to the collection and use of their data?**
Yes — explicit, per-donation, written consent. Every promoted donation
includes a `CONSENT.md` in which the donor attests to: ownership/right to
donate; absence of client-confidential material, NDA material, and other
people's personal data; having run and reviewed the redaction diff; and
release of the redacted session under CC-BY-SA-4.0. The consent records ship
in the archive (`data_archive_release_v2/data/donations/<label>/CONSENT.md`).

**Was a mechanism for revoking consent provided?**
Partially documented. [`DATA_USE_POLICY.md`](DATA_USE_POLICY.md): maintainers
may remove a donation "if a donor requests withdrawal before public release."
The v1 consent process additionally documented withdrawal at any time,
honored within 30 days (v1 datasheet §3). Not documented for v2: a
post-public-release withdrawal/takedown procedure with a stated SLA — a
takedown section in `DATA_USE_POLICY.md` covering already-released versions
(and how prior tagged versions are handled) would answer it. See §7.

---

## 4. Preprocessing / cleaning / labeling (anonymization)

**Was any preprocessing of the data done?**
Yes — a layered, local-first redaction pipeline (`donate/redact.py`,
described in [`CONTRIBUTING.md`](CONTRIBUTING.md) and the donor docs):

1. **Layer A (automatic, zero donor input):** Presidio built-in detectors
   (PERSON, EMAIL_ADDRESS, IP_ADDRESS, PHONE_NUMBER, CREDIT_CARD, URL,
   CRYPTO); custom recognizers for home paths (including Claude Code's
   dash-flattened path slugs) with the username auto-extracted and replaced
   by a **salted stable pseudonym** (`/Users/jane → /Users/<USER_a1b2c3>`, so
   one user maps to one pseudonym within a session); known API-key/token
   shapes (OpenAI/Anthropic, GitHub, AWS, Google, Slack, Hugging Face, bearer
   and basic-auth credentials, private-key blocks); and a detect-secrets
   entropy scan for everything else. The maintainers' employer name is a
   built-in default scrub term.
2. **Layer B (optional, local-only):** donor-supplied scrub terms
   (`--scrub "handle,codename,employer"`) — used only to remove information,
   never collected.
2b. **Identity-seeded scrub (tool 0.4, 2026-07-28,
   `donate/identity_redaction.py`):** the consent-form fields the wizard
   already collects (name, email, institute) seed an additional scrub pass:
   name parts and join forms (`First-Last`, `first.last`, `firstlast`), the
   email local-part, and institute short forms are replaced with the same
   salted pseudonyms, with path/slug-aware boundaries (git `Author:` lines,
   `/Users/<name>/` segments, workspace slugs), a base64/hex false-positive
   guard, and JSONL-safe rewriting. A "People detected in content" review
   list additionally surfaces third-party PERSON candidates in prose for a
   per-item keep/scrub decision (default scrub); nothing is auto-scrubbed
   without donor confirmation. Sessions exported before tool 0.4 receive the
   same engine retroactively at export time (`scripts/export_release.py`
   re-redaction stage, seeded from the private consent records; the private
   archive itself is read-only).
3. **Fail-closed local verification:** the wizard verifies the redacted
   output before allowing submission; the donor reviews the diff and can add
   scrub terms and re-run. Since tool 0.4 the verifier also blocks
   submission while any donor identity term remains in the artifact
   (`donor_identity` category; samples always masked).
4. **Privacy tiers:** `full_redacted` (default) preserves transcript
   structure and task semantics; `user_minimized` additionally masks
   sensitive donor-authored spans (private feelings, private-life details,
   toxic language, confidentiality markers) after full redaction
   ([`DONOR_PRIVACY.md`](DONOR_PRIVACY.md)).
5. **Independent maintainer PII/secrets review** before promotion
   ([`MAINTAINER_DONATION_WORKFLOW.md`](MAINTAINER_DONATION_WORKFLOW.md)).
   For the founding sessions, the release additionally shipped a verifiable
   13-pattern grep audit returning 0 hits (`make verify-pii`; v1 datasheet §8).

**Was the "raw" data saved in addition to the preprocessed data?**
Raw session history never leaves the donor's machine; maintainers and the
release only ever hold the redacted artifacts. (For the v1 founding sessions,
donor pre-redacted inputs were retained by the authors for withdrawal
handling; v1 datasheet §4.)

**Is the software used to preprocess the instances available?**
Yes — the full wizard ships in this repository (`donate/`, Apache-2.0), and
anyone can run it on their own Claude Code / Codex CLI session.

**Labeling.** No human labeling is applied to transcripts. Benchmark scores
(LLM-judge drift gaps, lexical proxies) are produced deterministically by the
released harness and shipped as derived tables (§2).

---

## 5. Uses

**Has the dataset been used for any tasks already?**
Yes. v1 (founding sessions): the 23-model panel-wide drift result, compaction
non-reset, single-shot anchor mitigation, and mode-dependent downstream-cost
findings (arXiv:2605.24279). v2 corpus: baseline drift-gap validation across
the 57-session accepted population (`summary_v2.0.csv`; mean drift gap
+0.1696), compaction-count cross-validation, and session-level drift
predictor analyses.

**What (other) tasks could the dataset be used for?**
Per [`DATA_USE_POLICY.md`](DATA_USE_POLICY.md), allowed uses are: persona
drift benchmarking; model behavior analysis in long agentic-coding sessions;
redaction, validation, and aggregate dataset-quality checks; and reproduction
of ContextEcho figures and claims. The snapshot-then-probe harness runs any
chat-completions target against any session, so new models can be evaluated
on the fixed corpus and new sessions can extend the corpus.

**Are there tasks for which the dataset should not be used?**
Yes — the disallowed uses in [`DATA_USE_POLICY.md`](DATA_USE_POLICY.md)
(mirrored in every donation manifest's `disallowed_uses`):

- **Donor profiling.**
- **Psychological or sentiment analysis of donors.**
- **Employment, productivity, or performance evaluation of donors.**
- **Attempts to deanonymize donors, organizations, repositories, or private
  paths.**

Additional documented misuse cautions (v1 datasheet §5): drift gap measures
output register, not latent persona-state — do not use it to certify
alignment properties or to rank one model as "safer" than another; do not
extrapolate long-context drift magnitudes to fresh-task capability.

**Is there anything about the composition or collection that might impact
future uses?**
Self-selected donor population, two harnesses, mostly English coding work;
redaction removes paths/URLs/identifiers, so analyses that need them are
impossible by design. Maintainers may reject or remove donations that violate
the use policy.

---

## 6. Distribution

**Will the dataset be distributed to third parties? How?**
Yes — public release. Code and governance docs in this repository
(github.com/Accenture/ContextEcho, public release subject to the
organization's OSS governance process per the paper appendix). Data hosting:
the v1.0 artifact (3 founding sessions + 41,921 per-cell evaluations) is live
on Hugging Face (`contextecho2026/persona-drift-contextecho`); the v2.0 data
release described in this datasheet (57 analysis sessions, 51,089 per-cell
evaluations) is pending the organization's OSS governance review and will be
published to the same Hugging Face repository under a new revision tag,
accompanied by this [`croissant.json`](croissant.json) metadata. Donations
flow in through the public relay and hosted donor page (see
[`README.md`](README.md)).

**When will it be distributed?**
v1 has been public since June 2026 alongside the arXiv preprint. The v2.0
archive is the current release candidate; Not documented — an exact v2.0
publication date (it will be set by the release/versioning step).

**How is it licensed?**
Data: **CC-BY-SA-4.0**, per each donor's `CONSENT.md` attestation. Code:
**Apache-2.0** (`LICENSE`). Note the v1 datasheet (§6) still says the code is
CC-BY-4.0 — superseded by the current `LICENSE` and README (see
inconsistencies note in §7).

**Have any third parties imposed IP-based or other restrictions?**
Donors attest that sessions contain no material under NDA and no
client-confidential content. For the founding sessions, third-party
copyrighted text was preserved only when openly licensed, otherwise replaced
with `<REDACTED_QUOTE>` (v1 datasheet §6). Not documented for v2: whether the
`<REDACTED_QUOTE>` practice is enforced mechanically by the wizard — the
consent attestation is the operative control.

**Do any export controls or other regulatory restrictions apply?**
None documented (v1 datasheet: none known). Not documented for v2
specifically.

---

## 7. Maintenance

**Who supports/hosts/maintains the dataset?**
The ContextEcho maintainer team, via the GitHub repository and the Hugging
Face dataset org. The maintainer console, relay audit log, and intake
tooling are documented in [`MAINTAINER_DONATION_WORKFLOW.md`](MAINTAINER_DONATION_WORKFLOW.md)
and [`DONATION_RELAY.md`](DONATION_RELAY.md).

**How can the owner/curator be contacted?**
Via GitHub issues on the repository. Not documented: a dedicated maintainer
contact email for privacy/takedown requests — adding one to
`DATA_USE_POLICY.md` would answer this.

**Will the dataset be updated? How often?**
Yes — ContextEcho is explicitly a **living benchmark**: donations are
reviewed and promoted on a rolling basis, appended to the ledger, and public
metadata is regenerated from the ledger (`make update-release-metadata`).
Releases are versioned (v1 founding archive; v2 community archive; the
corpus note in [`croissant.json`](croissant.json) marks it `isLiveDataset`).
Superseded same-lineage donations are retained as `SUPERSEDED` ledger rows
rather than silently replaced.

**Is there an erratum / changelog?**
Not documented — there is no `CHANGELOG.md` in the repository; the ledger and
git history are the de-facto record. A release changelog would answer this.

**Will older versions continue to be supported/hosted?**
The v1 archive ships alongside v2 in this repository; the v1 datasheet
commits to Hugging Face revision tags for versioning. Not documented: a
formal retention policy for old versions after a takedown (see below).

**If others want to extend/augment/build on the dataset, is there a
mechanism?**
Yes — this is the core design. Anyone can donate a session through the local
wizard and relay; accepted donations are promoted through review and quick
validation into the next release candidate, credited on the leaderboard, and
eligible for rolling co-authorship ([`CONTRIBUTING.md`](CONTRIBUTING.md)).
Provider adapters, annotations, and analyses are also credited contribution
types.

**Takedown / repair.**
Documented: maintainers may reject or remove a donation that violates the use
policy or on donor withdrawal request before public release
([`DATA_USE_POLICY.md`](DATA_USE_POLICY.md)); the relay supports one-record
resets and maintains an append-only audit log; `make backfill-promoted-validation`
repairs incompletely promoted sessions. Not documented: a post-release
takedown SLA and the handling of withdrawn sessions in already-tagged
versions.

**Known governance inconsistencies (recorded for repair):**
1. Donation `manifest.json` files (and some ledger rows) in the public
   archive contain maintainer-facing identity fields (`contributor`,
   `contributor_email`, `contributor_institute`) for donors with
   `public_anonymous: true`, contrary to the intent stated in
   [`DONOR_PRIVACY.md`](DONOR_PRIVACY.md) that these fields not appear in
   public dataset rows. They should be stripped or moved to a private
   sidecar before external distribution.
2. Withdrawal policy: `DATA_USE_POLICY.md` covers withdrawal *before* public
   release; the v1 consent process documented any-time withdrawal honored
   within 30 days. The two should be unified.
3. Code license: v1 `data_archive_release/DATASHEET.md` §6 says CC-BY-4.0 for
   code; the repository `LICENSE` and README say Apache-2.0 (current).
4. Turn totals: three turn counts exist and use different definitions —
   the paper's user turns (one user + assistant round; 12,210 over the 57
   analysis sessions, `figure_inputs_v2.0.csv`), the ledger intake `turns`
   field (12,337 over 55 rows / 12,281 over 54 accepted), and the
   `CONTRIBUTORS.md` leaderboard turns, which do not sum to either (known
   reconciliation item).
