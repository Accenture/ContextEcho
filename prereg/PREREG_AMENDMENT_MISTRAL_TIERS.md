# Pre-Registration Amendment: Mistral-Family Tier Stratification + Reasoning Model

**Project**: NeurIPS 2026 D&B submission, "The ContextEcho Probe Suite: Measuring Recency-Content Persona Drift in Long Agentic Coding Sessions"
**Amendment date**: 2026-04-29
**Author**: Anonymous (NeurIPS 2026 D&B submission, under review)
**Status**: LOCKED before any Magistral data collection. SHA-256 hash in §10. No post-hoc edits permitted; deviations logged in §6.

> **Signing**: After review, append `- [author signature, date, time]` and timestamp via `shasum -a 256 PREREG_AMENDMENT_MAGISTRAL.md`.

> **Relationship to primary pre-registration and prior amendments**: This amendment **does not modify** any primary pre-registered analysis or prior amendments. The original `PREREG.md`, `PREREG_PATH_A.md`, `WEEK1_PREREG.md`, `PREREG_AMENDMENT_GEMINI.md` (sha256 `411b248ab959...`), `PREREG_AMENDMENT_MISTRAL.md` (sha256 `187d0c602134...`), and `PREREG_AMENDMENT_KIMI.md` (sha256 `b5516bfab228...`) all retain their original byte content. Results from this amendment are reported in the paper as **post-hoc panel extension**, not primary findings.

---

## 1. Motivation

A novel pattern is emerging from the panel-extension data that was **not visible in the original 12-target panel**:

| Target | Scratch baseline (Sonnet judge) | Notable |
|---|---|---|
| Original 12-target panel (typical) | 2.85 – 2.88 | typical |
| Gemini 2.5 Pro | 2.32 | reasoning-mode model, first observation |
| Kimi K2.6 | 2.48 | second observation |
| Mistral Large 2512 | 2.00 | third, lowest in panel |

All three panel-extension targets so far show **anomalously low scratch baselines** — i.e., elaborated, persona-flavored probe responses *with no prepended context at all*. The original 12-target panel had no targets with this pattern.

The most parsimonious common factor is that **all three are frontier 2025/2026 models with verbose elaborated default behavior**, and Pro is explicitly a reasoning-mode model. Whether this **verbose-baseline pattern** is specifically a property of reasoning-mode models — or a more general property of frontier 2025/2026 models — is testable.

`magistral-medium-latest` (Mistral's reasoning-class model, mid-tier) is the cheapest direct test: same organization as Mistral Large (so we control for organizational training pipeline), explicitly reasoning-mode (so we test the reasoning hypothesis directly).

This amendment adds Magistral Medium to the panel under the **same protocol**, **same probes**, **same rubric**, **same judges** — only target addition.

## 2. What's new

### 2.1 New targets

| Target | Provider | Tier | Identifier |
|---|---|---|---|
| Magistral Medium | Mistral la Plateforme | **Reasoning**, mid-tier | `magistral-medium-latest` (resolves to `magistral-medium-2509` as of 2026-04-29) |
| Mistral Medium | Mistral la Plateforme | Mid-tier non-reasoning | `mistral-medium-latest` (resolves to `mistral-medium-2508`) |
| Mistral Small | Mistral la Plateforme | Small-tier non-reasoning | `mistral-small-latest` (resolves to `mistral-small-2603`) |

All three accessed via the same Mistral API (`https://api.mistral.ai/v1`) with the same `MISTRAL_API_KEY`. Reuses `harness/clients_mistral.py::call_mistral`.

The two non-reasoning targets (Mistral Medium, Mistral Small) provide tier-stratified within-organization data (analogous to Gemini Pro vs Flash). Magistral Medium is explicitly a reasoning-class model and tests whether the verbose-baseline pattern observed in Pro / Kimi / Mistral Large is specifically a property of reasoning-mode models.

**Wrapper bug fix logged 2026-04-29**: `harness/clients_mistral.py` originally extracted `choices[0].message.content` as a string. Magistral reasoning models return content as a `list` of structured blocks (text + reasoning). Wrapper now handles both string and list-content shapes; reasoning/thinking blocks are excluded from the extracted text. See deviation §6.

### 2.2 No changes to instrument

- **Probes**: identical 25-probe suite from `harness/probes.py`.
- **Conditions**: identical 5-condition content-position protocol.
- **Judges**: Sonnet 4.6 (primary) + GPT-5 (cross-replication audit), held constant.
- **Rubric**: identical 0-3 hedge-compliance rubric.
- **$c_{\text{pre}}$**: same donated Claude-derived context (Phase 2) and GPT-5-derived $c_{\text{pre}}$ (Phase 3 if added later).

### 2.3 Sampling parameters

- `temperature = 0.0` (deterministic)
- `max_tokens = 4096` (matches panel-extension Convention B)
- `request timeout = 120s`

These match `PREREG_AMENDMENT_MISTRAL.md` §2.3 byte-identically.

## 3. Hypotheses

### H1 (Q1 extension): per-target drift on Claude-derived $c_{\text{pre}}$

> Per-target Δ(recent3K vs scratch) on Magistral Medium, Mistral Medium, Mistral Small. Tested by paired permutation on per-probe differences with Holm correction over the 3 new targets, two-sided α = 0.05.

### H1' (verbose-baseline hypothesis, exploratory)

> Per-target scratch baseline mean under Sonnet judge for each of the three new targets, compared against the typical original-panel range (2.85–2.88). Reported descriptively, not as a primary statistical claim — exploratory test of whether the verbose-baseline pattern observed in Pro / Kimi / Mistral Large generalizes.

Possible outcomes:
- **All three show verbose baseline** (scratch < 2.60): pattern is robust across Mistral tiers and includes reasoning + non-reasoning models. Suggests it's a frontier-2025/2026 generation-effect rather than reasoning-mode-specific.
- **Only Magistral shows verbose baseline**: pattern is specifically reasoning-mode. Strongest support for reasoning-as-cause hypothesis.
- **None show verbose baseline**: pattern is specific to Pro / Kimi / Mistral Large for reasons we haven't identified — unifying explanation gets harder.
- **Mixed**: document descriptively.

### H1'' (within-Mistral tier stratification, exploratory)

> Comparison of scratch baseline and Δ(recent3K − scratch) across Mistral Large 2512, Mistral Medium, and Mistral Small under Sonnet judge. Reported descriptively. Tests whether Mistral's verbose-baseline pattern is uniform across capability tiers (analogous to Gemini Pro/Flash analysis) or specific to the frontier tier.

### H2 (Q2 extension, deferred)

Same-target context-source ablation on Magistral × GPT-5-derived $c_{\text{pre}}$ is **not part of this amendment**. If H1 finds Magistral drifts on Claude-derived $c_{\text{pre}}$, a follow-up amendment may add the ablation. Out of scope here to keep the amendment narrow.

## 4. Sample size & statistical procedure

### 4.1 Sample size

25 probes per condition, paired by probe ID. No new seeds (deterministic decoding).

### 4.2 Primary test

Paired permutation (10,000 resamples, two-sided), byte-identical to `scripts/cross_judge_12model_analyze.py`. Holm correction within the family of new tests added by this amendment (1 test for H1 → Holm-of-1 = raw p). α = 0.05.

### 4.3 Cross-judge replication

Each Magistral cell evaluated under both Sonnet 4.6 and GPT-5 judges on full responses. Cross-judge robustness reported per the same gate as primary protocol.

### 4.4 Reporting label

Results labeled **"post-hoc panel extension"** in the paper. The verbose-baseline observation (H1') is labeled **exploratory** and reported alongside Pro / Kimi / Mistral Large baselines as a four-target descriptive pattern.

## 5. Cost cap & kill conditions

### 5.1 Cost cap

Hard stop at **$50 cumulative API spend** across all three new targets, Phase 2 only (Phase 3 not in scope of this amendment).

### 5.2 Kill conditions

Same as `PREREG_AMENDMENT_MISTRAL.md` §5.2:
- Empty-output rate > 5% per cell after first 5 probes → halt that cell.
- API error rate > 5% → halt, diagnose.
- Δ outside [-1.5, +0.5] → halt, verify probe responses.
- Cross-judge agreement (Spearman ρ) < 0.4 → flag.

Mistral API rate-limit caveat: free-tier-style 429s observed on `mistral-large-latest` Phase 2 (errors on probes 5/6 of `scratch` and probe 5 of `recent3K_earlier`, n_valid 22-24/25). Same wrapper, same retry policy applies; expect n_valid in 22-25/25 range per cell.

## 6. Permitted deviations log

| Date | Deviation | Reason | Affected analysis |
|---|---|---|---|
| 2026-04-29 | `harness/clients_mistral.py` updated to handle `choices[0].message.content` as either a string (regular models) or a list of structured blocks (reasoning models like Magistral). Reasoning/thinking blocks are excluded from the extracted visible text. | Magistral Medium smoke test 2026-04-29 raised `AttributeError: 'list' object has no attribute 'strip'`. Fix is a content-shape unification, not a sampling-parameter change. Documented before any data collection. | All Magistral cells; existing Mistral Large cells unaffected because the wrapper preserves string-content path byte-identically. |
| 2026-04-29 | Amendment expanded from Magistral Medium only to also include `mistral-medium-latest` and `mistral-small-latest` (added before any Magistral data collection began). | User direction to test Option 2 (Mistral tier stratification) alongside the original reasoning-model test. The three new targets share infrastructure (same wrapper, same key, same protocol) so consolidating into a single amendment keeps pre-registration discipline cleaner. Documented before any data collection. | All three new Mistral targets. |

## 7. What's NOT permitted post-hoc

- Adding Mistral-reasoning targets beyond `magistral-medium-latest` specified here.
- Changing probes, rubric, judges, or $c_{\text{pre}}$ for Magistral cells.
- Re-running Magistral cells with adjusted parameters after seeing results.
- Promoting any Magistral result to "primary" status.
- Reporting H1' as a primary finding — it's locked as exploratory.

## 8. Files affected

- **New code**: `scripts/phase2_mistral_tiers.py` (template-derived from `scripts/phase2_mistral_main.py` with target list of three; reuses `harness/clients_mistral.py` with the list-content fix).
- **New data**: `data_archive/mistral_tiers_panel/phase2_main/` with subdirectories per target.
- **Pre-registration documents** (untouched): all prior amendments retain original byte content and SHA-256 hashes.

## 9. Phase plan (informational)

- **Phase 0 (now)**: this amendment + reuse existing Mistral wrapper. No additional smoke testing needed since `harness/clients_mistral.py` was validated 2026-04-29 against `mistral-large-latest`. A 1-call confirmation that `magistral-medium-latest` works will run before Phase 2.
- **Phase 1 (skipped)**: probe pipeline already validated.
- **Phase 2**: Magistral Medium × 5 conditions × 25 probes × 2 judges. Estimated ~20 min, ~$3-5.
- **Phase 3 (deferred)**: same-target ablation not in scope here.
- **Phase 4 (deferred)**: stats analysis combined with the existing Mistral Large stats run, since both share the same organization and allow tier-stratification reporting.

## 10. Signature

Signed by: Anonymous (NeurIPS 2026 D&B submission, under review).
Date: 2026-04-29.
SHA-256 of this file at signing (pre-signature byte content):
`1dfceb4fadc57763ef3959f0a41cfd092862ab2208eb99dd64ec2bf5aed0ce23`
Reviewers can verify by removing the lines from "SHA-256 of this file
at signing" through end-of-file and rehashing.

Updated `harness/clients_mistral.py` SHA-256:
`785998698908c2e9cb16137c45e54a95c1de880257236cc76207070cf24e687c`
(was `95d4e2424caa10f490ce9c80946aef84d290d10831d5bcfcbd51fe2b3128fe6d`
before the list-content fix; original Mistral Large cells unaffected
because the string-content code path is preserved byte-identically.)

Git SHA at signing: see commit message.
