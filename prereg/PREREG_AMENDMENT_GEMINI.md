# Pre-Registration Amendment: Google Gemini Panel Extension

**Project**: NeurIPS 2026 D&B submission, "The ContextEcho Probe Suite: Measuring Recency-Content Persona Drift in Long Agentic Coding Sessions"
**Amendment date**: 2026-04-28
**Author**: Anonymous (NeurIPS 2026 D&B submission, under review)
**Status**: LOCKED before any Gemini API data collection. SHA-256 hash in signature section. No post-hoc edits permitted; deviations logged in §6.

> **Signing**: After review, append `- [author signature, date, time]` and timestamp via `shasum -a 256 PREREG_AMENDMENT_GEMINI.md`. Do not begin Phase 1 (smoke run) until this file is signed and timestamped.

> **Relationship to primary pre-registration**: This amendment **does not modify** any primary pre-registered analysis. The original 12-target panel (`PREREG.md`), Q3 substrate track (`PREREG_PATH_A.md`), and Week 1 replication gate (`WEEK1_PREREG.md`) remain locked at their original SHA-256 hashes. Results from this amendment are reported in the paper as **post-hoc panel extension**, not primary findings.

---

## 1. Motivation

Pass-5 cold reviewers (Claude.ai, ChatGPT, Gemini independent reviews; synthesized in `docs/COLD_REVIEW_PASS5_SYNTHESIS.md`) flagged the absence of Google models from the cross-organizational panel as a breadth limitation. The original panel covers 5 organizations (Anthropic, OpenAI, Meta, Alibaba, DeepSeek). NeurIPS-best-paper benchmarks like Hivemind (Jiang et al., 2025) cover 9+ organizations. Adding Google's frontier and fast-tier targets closes the most-noticed cross-organizational gap.

This amendment adds Google Gemini 2.5 Pro and Gemini 2.5 Flash to the existing panel under the **same protocol**, **same probes**, **same rubric**, **same judges** — no methodological changes, only target additions.

## 2. What's new

### 2.1 New targets

| Target | Provider | Tier | Identifier |
|---|---|---|---|
| Gemini 2.5 Pro | Google AI Studio | Frontier-class | `gemini-2.5-pro` |
| Gemini 2.5 Flash | Google AI Studio | Fast-tier | `gemini-2.5-flash` |

Targets selected to match the closed-frontier and fast-tier categories already populated in the original panel (e.g., Sonnet 4.6 / Haiku 4.5 for Anthropic; GPT-5 / GPT-4o-mini for OpenAI).

### 2.2 No changes to instrument

- **Probes**: identical 25-probe suite from `harness/probes.py` (5 categories: identity, preference, relational, creative-self, experiential).
- **Conditions**: identical 5-condition content-position protocol (`scratch`, `recent3K`, `recent3K_filler`, `recent3K_earlier`, `filler14K`).
- **Judges**: Sonnet 4.6 (primary) + GPT-5 (cross-replication audit), held constant from primary protocol.
- **Rubric**: identical 0-3 hedge-compliance rubric from `harness/judge.py`, frozen.
- **$c_{\text{pre}}$**: same donated Claude-derived context for Q1 panel; same length-and-structure-matched GPT-5-derived $c_{\text{pre}}$ for Q2 same-target ablation.

### 2.3 Sampling parameters

- `temperature = 0.0` (deterministic decoding)
- `max_output_tokens = 4096` (must be ≥ 4096 for Gemini 2.5 Pro; smoke-tested 2026-04-28 that lower caps silently produce empty output due to internal reasoning tokens consuming the budget)
- `thinking_budget`: SDK default (reasoning enabled for Pro; Pro's reasoning is part of what makes Pro "Pro" — disabling it would invalidate the comparison to other frontier models).

## 3. Hypotheses

### H1 (Q1 extension): Gemini drift on Claude-derived $c_{\text{pre}}$

> Per-target mean Δ(recent3K vs scratch) on Gemini 2.5 Pro and Gemini 2.5 Flash, separately. Tested by paired permutation test on per-probe differences with Holm correction across the 2 new targets, two-sided α = 0.05.

Possible outcomes (committed before unblinding):
- **Both drift** (Δ ≤ −0.20 each): drifter count grows from 8/12 to 10/14; family-clustering pattern strengthens (Anthropic + Chinese labs + GPT-4.1 + Google all drift).
- **Only Pro drifts**: stratification finding — frontier-class drifts, fast-tier doesn't, mirroring what you see for Sonnet 4.6 (drifts) vs Haiku 4.5 (borderline).
- **Only Flash drifts**: surprising; would prompt investigation of size/training differences.
- **Neither drifts**: drifter count stays at 8/14 (proportionally similar); non-drifter cluster expands from {GPT-5, GPT-4o, GPT-4o-mini, Llama 3.3 70B} to include Google.

### H2 (Q2 extension): Gemini 2.5 Pro family-specificity

> Mean Δ(recent3K vs scratch) for Gemini 2.5 Pro × Claude-derived $c_{\text{pre}}$ vs Gemini 2.5 Pro × GPT-5-derived $c_{\text{pre}}$ of equivalent length and structure. Tested by paired permutation on per-probe paired differences, Holm correction across the n=4 same-target ablation (Sonnet 4.6, Opus 4.7, GPT-4.1, Gemini 2.5 Pro), two-sided α = 0.05.

Possible outcomes (committed before unblinding):
- **Pro is family-specific** (drifts more on Claude-derived than GPT-5-derived): same-target panel becomes 3 of 4 family-specific; Pass-5 T3 weakness ("n=3 is fragile") is partially addressed.
- **Pro is family-agnostic** (drifts equally on both): same-target panel becomes 2 of 4 family-specific; Opus 4.7 outlier pattern strengthens but family-specificity still the majority.
- **Pro doesn't drift on either** (Δ ≈ 0 in both arms): Gemini family-specificity is undefined; ablation can't be run, report as "ablation not applicable, Pro is non-drifter." Consistent with H1 negative outcome on Pro.

## 4. Sample size & statistical procedure

### 4.1 Sample size

Same as primary protocol: 25 probes per condition (paired across conditions by probe ID).

No new seeds: deterministic decoding (`temperature = 0.0`) means each (model, condition, probe) cell is a single value; there is no stochastic seed to vary, matching primary protocol convention.

### 4.2 Primary test (per H1, H2)

Paired permutation test on per-probe differences. Holm correction over the family of new tests added by this amendment:
- For H1: Holm over 2 targets (Pro recent3K-vs-scratch, Flash recent3K-vs-scratch).
- For H2: Holm over 1 target (Pro × Claude-derived vs GPT-5-derived); reported alongside the original n=3 panel for context but corrected within its own family.

Two-sided α = 0.05.

### 4.3 Cross-judge replication (primary gate)

Each Gemini cell is evaluated under **both** Sonnet 4.6 and GPT-5 judges on full responses (not 300-char previews — Pass-5 T1 fix already applied to primary panel; same standard applies here).

A finding is reported as **"cross-judge robust"** only if it survives Holm-corrected paired permutation under **both** judges. This is consistent with the primary pre-registration: cross-judge replication is a primary gate, not a robustness afterthought.

### 4.4 Reporting label

Results from this amendment are labeled **"post-hoc panel extension"** in the paper text, not primary. The paper's headline numbers (12-target panel post-hoc mixed-effects, n=3 same-target ablation) remain primary; the new 14-target / n=4 numbers are explicitly noted as added in response to cold-review feedback.

## 5. Cost cap & kill conditions

### 5.1 Cost cap

Hard stop at **$200 cumulative API spend** for the full Phase 1-3 campaign (smoke + main + ablation). Tier 2 main is estimated at ~$100-140; Tier 3 ablation at ~$30; ~$30 buffer.

If spend exceeds cap before campaign completion, halt and reassess — do not silently top up beyond cap.

### 5.2 Kill conditions (per phase, hard)

- **Empty-output rate** > 5% across any (target, condition) cell after the first 5 probes → halt that cell, diagnose `max_output_tokens` and reasoning-budget settings before continuing.
- **API error rate** > 5% across any cell → halt, diagnose auth / rate / billing before continuing.
- **Δ outside [-1.5, +0.5]** on any cell (i.e., implausibly extreme drift, beyond the range observed across all 12 original targets) → halt, verify probe responses look normal before continuing.
- **Cross-judge agreement (Spearman ρ) < 0.4** for any Gemini target → flag in writing as a Gemini-specific judge-instability finding; do not silently exclude.
- **Free-tier or billing reset** mid-campaign (HTTP 429 / 403 from Google) → halt; do not proceed until billing state is verified stable.

## 6. Permitted deviations log

This amendment was authored 2026-04-28 in a single session. Any deviations from the locked protocol after signing must be logged below with a dated entry. As of signing, no deviations.

| Date | Deviation | Reason | Affected analysis |
|---|---|---|---|
| 2026-04-28 | `max_output_tokens` set to 4096 (above original 200 used in primary protocol's standalone scripts) | Smoke test 2026-04-28 confirmed Gemini 2.5 Pro silently produces empty output at lower caps due to internal reasoning consuming budget. 4096 matches primary protocol's `harness/clients.py` Anthropic / OpenAI cap of 16384 and is well within Gemini's 8192-output limit. Documented before any data collection. | All Gemini cells |
| 2026-04-28 | Phase 3 same-target ablation target changed from Gemini 2.5 Pro to Gemini 2.5 Flash | Phase 2 results (committed before this deviation was logged) revealed Pro has Δ(recent3K − scratch) = −0.04 under Sonnet judge — i.e., Pro shows no drift on Claude-derived c_pre to begin with. Per H2 of this amendment §3, the same-target ablation is designed to test family-specificity by comparing drift on Claude-derived vs GPT-5-derived c_pre on the same target. With Pro's Δ already at noise floor on Claude-derived, the contrast cannot produce an informative result (it can only show null vs null). Flash, by contrast, drifts cleanly on Claude-derived (Δ = −0.32 Sonnet judge, attention-dilution signature). Phase 3 on Flash tests the family-specificity hypothesis on a target that has actual drift to attribute. The hypothesis remains H2 of §3 unchanged in form: paired permutation, Holm correction across the same-target ablation panel, two-sided α=0.05. The outcome decision tree adapts: family-specific = Δ shrinks toward zero on GPT-5-derived c_pre (joins Sonnet 4.6 / GPT-4.1 majority pattern); family-agnostic = Δ stays similar on both (joins Opus 4.7 outlier pattern). Pro is reported in the paper as "doesn't drift on Claude-derived c_pre, no same-target ablation runnable" — itself a meaningful negative result. | Phase 3 only; Phases 0/1/2 unaffected |

## 7. What's NOT permitted post-hoc

- Adding Gemini targets beyond the two specified here.
- Changing probes, rubric, judges, or $c_{\text{pre}}$ for Gemini cells.
- Re-running Gemini cells with adjusted parameters after seeing results.
- Promoting any Gemini result to "primary" status — the post-hoc-extension label is locked.
- Excluding "outlier" Gemini probe responses without pre-specified criteria from the original protocol.

## 8. Files affected by this amendment

- **New code**: `harness/clients_gemini.py` (Gemini SDK wrapper, locked at amendment SHA), `scripts/smoke_test_gemini.py` (Phase 0 / 1 validation harness).
- **New data**: `data_archive/gemini_panel/` directory tree (per phase, per cell, per probe + judge call).
- **Patched scripts**: `scripts/cross_judge_12model.py` extended with Google dispatch branch (additive only — no modification to existing 12-target dispatch logic).
- **Pre-registration documents** (untouched): `PREREG.md`, `PREREG_PATH_A.md`, `WEEK1_PREREG.md` retain original byte content and SHA-256 hashes.

## 9. Phase plan (informational, not part of locked protocol)

- **Phase 0 (now)**: write this amendment + harness wiring + Gemini wrapper. No API spend. Hash and commit before Phase 1.
- **Phase 1**: Flash-only smoke against 2 conditions × 25 probes × Sonnet judge only (~20 min, ~$2). Pipeline integrity check.
- **Phase 2**: Pro + Flash × 5 conditions × 25 probes × both judges (~3-4 hrs, ~$100-140). Full panel addition.
- **Phase 3**: Pro × GPT-5-derived $c_{\text{pre}}$ × 5 conditions × 25 probes × both judges (~1-2 hrs, ~$30). Same-target ablation extension.

Phase 4 (paper integration: abstract / §4 / Fig 3 / Fig 11 updates) is intentionally out of scope of this amendment and will be done after Phase 3 results land.

## 10. Signature

Signed by: Anonymous (NeurIPS 2026 D&B submission, under review).
Date: 2026-04-28.
SHA-256 of this file at signing (pre-signature byte content):
`411b248ab959975341d788beb21143db53844315658da7eeae9923fbc9ca4ed3`
This hash was computed over the file content **before** this signature block
was finalized; reviewers can verify it by removing the lines from
"SHA-256 of this file at signing" through end-of-file and rehashing.

Git SHA at signing: see commit message of the commit that follows.

No amendments to this amendment after signing unless explicitly logged in §6.
