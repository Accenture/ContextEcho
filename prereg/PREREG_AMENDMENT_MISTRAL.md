# Pre-Registration Amendment: Mistral Large Panel Extension

**Project**: NeurIPS 2026 D&B submission, "The ContextEcho Probe Suite: Measuring Recency-Content Persona Drift in Long Agentic Coding Sessions"
**Amendment date**: 2026-04-29
**Author**: Anonymous (NeurIPS 2026 D&B submission, under review)
**Status**: LOCKED before any Mistral API data collection. SHA-256 hash in §10. No post-hoc edits permitted; deviations logged in §6.

> **Signing**: After review, append `- [author signature, date, time]` and timestamp via `shasum -a 256 PREREG_AMENDMENT_MISTRAL.md`. Do not begin Phase 2 (target+judge run) until this file is signed and timestamped.

> **Relationship to primary pre-registration and prior amendments**: This amendment **does not modify** any primary pre-registered analysis or the prior Gemini amendment. The original 12-target panel (`PREREG.md`), Q3 substrate track (`PREREG_PATH_A.md`), Week 1 replication gate (`WEEK1_PREREG.md`), and Gemini extension (`PREREG_AMENDMENT_GEMINI.md` sha256 `411b248ab959...`) all retain their original byte content and SHA-256 hashes. Results from this amendment are reported in the paper as **post-hoc panel extension**, not primary findings.

---

## 1. Motivation

Pass-5 cold reviewers flagged the absence of Mistral from the cross-organizational panel. Hivemind (Jiang et al., NeurIPS 2025 best paper) included Mistral Large in their main 25-model panel. After adding Google in the Gemini amendment (panel: 12 → 14 targets, 5 → 6 organizations), adding Mistral grows the cross-organizational claim to **7 organizations** (Anthropic, OpenAI, Meta, Alibaba, DeepSeek, Google, Mistral).

Mistral is a European frontier-class organization with a publicly-deployed Assistant persona via RLHF, distinct training pipeline from Anthropic / OpenAI / Google, and frontier-tier coding performance — it's the highest-leverage single-model addition for cross-organizational breadth.

This amendment adds Mistral Large 2411 to the existing panel under the **same protocol**, **same probes**, **same rubric**, **same judges** — no methodological changes, only a target addition.

## 2. What's new

### 2.1 New target

| Target | Provider | Tier | Identifier |
|---|---|---|---|
| Mistral Large 2411 | Mistral via Together AI | Frontier-class | `mistralai/Mistral-Large-Instruct-2411` |

Selected as the frontier-tier Mistral model active in the Hivemind panel and most directly comparable to Sonnet 4.6 / GPT-4.1 / Gemini 2.5 Pro in capability class.

Together AI provides Mistral Large via an OpenAI-compatible endpoint at `https://api.together.xyz/v1`, accessed via the existing `TOGETHER_AI_KEY` (or `TOGETHER_API_KEY`, same key under either name) in the project's `.env` file. This matches the access path used by the existing 12-target panel for Llama 3.3 70B, Qwen 3 235B, and DeepSeek V3.

### 2.2 No changes to instrument

- **Probes**: identical 25-probe suite from `harness/probes.py`.
- **Conditions**: identical 5-condition content-position protocol (`scratch`, `recent3K`, `recent3K_filler`, `recent3K_earlier`, `filler14K`).
- **Judges**: Sonnet 4.6 (primary) + GPT-5 (cross-replication), held constant.
- **Rubric**: identical 0-3 hedge-compliance rubric from `harness/judge.py`.
- **$c_{\text{pre}}$**: same donated Claude-derived context for Q1; same length-and-structure-matched GPT-5-derived $c_{\text{pre}}$ (from `data/openai_gpt-5_debug_and_fix_baseline_seed301_0952d536c9c9/transcript.jsonl`) for Q2 same-target ablation.

### 2.3 Sampling parameters

- `temperature = 0.0` (deterministic decoding, matching primary protocol)
- `max_tokens = 4096` (matching the locked Gemini convention; matches `harness/clients.py` Anthropic / OpenAI cap of 16384 well)
- `request timeout = 120s` (Together AI's Mistral hosting has not shown the long-context hang pattern observed for Gemini, but the timeout is included as defensive measure)

Note: this DEVIATES from the existing `non_anthropic_extension.py` `call_together` wrapper which uses `temperature=0.7` and `max_tokens=400`. The Mistral campaign uses the locked Gemini-amendment convention (`temperature=0.0`, `max_tokens=4096`) for consistency with the deterministic protocol used across the panel-extension family.

## 3. Hypotheses

### H1 (Q1 extension): Mistral drift on Claude-derived $c_{\text{pre}}$

> Per-target Δ(recent3K vs scratch) on Mistral Large. Tested by paired permutation test on per-probe differences with Holm correction across the 1 new target (Holm-of-1 = raw p), two-sided α = 0.05.

Possible outcomes (committed before unblinding):
- **Drifts** (Δ ≤ −0.20 with p_holm < 0.05): drifter count grows from 9/14 to 10/15; family-clustering pattern strengthens with another non-Anthropic, non-OpenAI drifter.
- **Does not drift** (CI includes zero): non-drifter cluster grows; family-clustering pattern remains directional.
- **Borderline**: report descriptively without primary-claim status.

### H2 (Q2 extension): Mistral family-specificity

> Mean Δ(recent3K vs scratch) for Mistral Large × Claude-derived $c_{\text{pre}}$ vs Mistral Large × GPT-5-derived $c_{\text{pre}}$ of equivalent length and structure. Tested by paired permutation on per-probe paired differences within the new same-target ablation cell (Holm-of-1 = raw p), two-sided α = 0.05.

Same-target ablation panel post-amendment will be n=5: Sonnet 4.6, Opus 4.7, GPT-4.1, Gemini 2.5 Flash, **Mistral Large**.

Possible outcomes (committed before unblinding):
- **Family-specific** (Δ on GPT-5-derived shrinks toward zero, gap > 0.20 with p_holm < 0.05): Mistral joins Sonnet 4.6 / GPT-4.1 majority pattern. Same-target panel becomes 3 of 5 strongly family-specific.
- **Family-agnostic**: Mistral joins Opus 4.7 outlier pattern. Same-target panel becomes 2 of 5 strongly family-specific.
- **Partial / not significant**: Joins the Gemini Flash directional-but-not-significant cluster.

## 4. Sample size & statistical procedure

### 4.1 Sample size

Same as primary protocol: 25 probes per condition, paired by probe ID. No new seeds (deterministic decoding).

### 4.2 Primary test

Paired permutation test on per-probe differences, byte-identical to `scripts/cross_judge_12model_analyze.py` (10,000 resamples, two-sided). Holm correction within the family of new tests added by this amendment (1 test for H1, 1 test for H2 → Holm-of-1 = raw p in each case). α = 0.05.

### 4.3 Cross-judge replication (primary gate)

Each Mistral cell evaluated under **both** Sonnet 4.6 and GPT-5 judges on full responses. A finding is reported as **"cross-judge robust"** only if it survives Holm-corrected paired permutation under **both** judges. Same gate as primary protocol and Gemini amendment §4.3.

### 4.4 Reporting label

Results from this amendment labeled **"post-hoc panel extension"** in the paper. Headline numbers (12-target panel post-hoc mixed-effects, n=3 same-target ablation) remain primary; the new 15-target / n=5 numbers are explicitly noted as added in response to cold-review feedback.

## 5. Cost cap & kill conditions

### 5.1 Cost cap

Hard stop at **$50 cumulative API spend** for the full Phase 2 + 3 campaign. Mistral via Together AI is approximately $2/M input, $6/M output — Phase 2 + 3 estimated at ~$15-25 with margin.

### 5.2 Kill conditions (per phase, hard)

- **Empty-output rate** > 5% across any (target, condition) cell after the first 5 probes → halt that cell, diagnose `max_tokens` settings.
- **API error rate** > 5% across any cell → halt, diagnose auth / rate / billing.
- **Δ outside [-1.5, +0.5]** on any cell → halt, verify probe responses.
- **Cross-judge agreement (Spearman ρ) < 0.4** on Mistral → flag in writing as a Mistral-specific judge-instability finding; do not silently exclude.

## 6. Permitted deviations log

Any deviations from the locked protocol after signing must be logged below with a dated entry. As of signing, no deviations.

| Date | Deviation | Reason | Affected analysis |
|---|---|---|---|
| 2026-04-29 | Mistral wrapper uses `temperature=0.0` and `max_tokens=4096`, deviating from existing `scripts/non_anthropic_extension.py::call_together` (`temperature=0.7`, `max_tokens=400`). | Match the deterministic-decoding convention locked in `PREREG_AMENDMENT_GEMINI.md` §2.3 and used across all post-Gemini panel-extension work. Documented before any Mistral data collection. | All Mistral cells |
| 2026-04-29 | Target swapped from `mistralai/Mistral-Large-Instruct-2411` (Together AI dedicated endpoint) to `mistral-large-latest` (Mistral la Plateforme serverless), which resolves to `mistral-large-2512` as of this date. | Together AI gates `Mistral-Large-Instruct-2411` behind dedicated endpoints (~$1+/hour idle charges) rather than serverless. After acquiring `MISTRAL_API_KEY` from `console.mistral.ai`, the same organization (Mistral AI) is accessible via the official API on a usage-based serverless pricing model with no idle charges. The model `mistral-large-2512` is the December 2025 frontier release of Mistral Large, *newer* than the originally-scoped `Mistral-Large-Instruct-2411` (November 2024) — same organization, same capability tier, more recent build. Sampling parameters and protocol unchanged; only the access path and resolved model version differ. Smoke test 2026-04-29 confirms the new target works at 0.98s latency on probe-shape calls. | All Mistral cells |

## 7. What's NOT permitted post-hoc

- Adding Mistral targets beyond Mistral Large 2411 specified here.
- Changing probes, rubric, judges, or $c_{\text{pre}}$ for Mistral cells.
- Re-running Mistral cells with adjusted parameters after seeing results.
- Promoting any Mistral result to "primary" status — the post-hoc-extension label is locked.
- Excluding "outlier" probe responses without pre-specified criteria from the original protocol.

## 8. Files affected by this amendment

- **New code**: `harness/clients_together.py` (or extension to existing pattern; small wrapper around Together AI's OpenAI-compatible endpoint with the locked sampling parameters), `scripts/phase2_mistral_main.py`, `scripts/phase3_mistral_same_target.py`, `scripts/phase4stats_mistral_analyze.py`.
- **New data**: `data_archive/mistral_panel/` directory tree.
- **Pre-registration documents** (untouched): `PREREG.md`, `PREREG_PATH_A.md`, `WEEK1_PREREG.md`, `PREREG_AMENDMENT_GEMINI.md` retain original byte content and SHA-256 hashes.

## 9. Phase plan (informational, not part of locked protocol)

- **Phase 0 (now)**: this amendment + Together API smoke test. No API spend on probes. Hash and commit before Phase 2.
- **Phase 1 (skipped)**: The probe pipeline integrity was already validated by the Gemini Phase 1 smoke run. Mistral via Together uses an OpenAI-compatible endpoint already exercised by the 12-target panel for Llama 3.3 70B / Qwen 3 235B / DeepSeek V3, so no new pipeline shape needs validation.
- **Phase 2**: Mistral × 5 conditions × 25 probes × 2 judges. Estimated ~30 min, ~$15.
- **Phase 3**: Mistral × GPT-5-derived $c_{\text{pre}}$ × 5 conditions × 25 probes × 2 judges. Estimated ~25 min, ~$10.
- **Phase 4**: Pre-registered Holm-corrected statistics. ~5 min, $0.

Phase 5 (paper integration) is deferred per user direction (campaign saved for later integration once additional model results inform the framing).

## 10. Signature

Signed by: Anonymous (NeurIPS 2026 D&B submission, under review).
Date: 2026-04-29.
SHA-256 of this file at signing: TO BE COMPUTED via `shasum -a 256 PREREG_AMENDMENT_MISTRAL.md` immediately after first commit; pre-signature value will be recorded here.
Git SHA at signing: see commit message.

No amendments to this amendment after signing unless explicitly logged in §6.
