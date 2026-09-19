# Pre-Registration Amendment: Moonshot AI Kimi K2.6 Panel Extension

**Project**: NeurIPS 2026 D&B submission, "The ContextEcho Probe Suite: Measuring Recency-Content Persona Drift in Long Agentic Coding Sessions"
**Amendment date**: 2026-04-29
**Author**: Anonymous (NeurIPS 2026 D&B submission, under review)
**Status**: LOCKED before any Kimi API data collection. SHA-256 hash in §10. No post-hoc edits permitted; deviations logged in §6.

> **Signing**: After review, append `- [author signature, date, time]` and timestamp via `shasum -a 256 PREREG_AMENDMENT_KIMI.md`.

> **Relationship to primary pre-registration and prior amendments**: This amendment **does not modify** any primary pre-registered analysis or prior amendments. The original `PREREG.md`, `PREREG_PATH_A.md`, `WEEK1_PREREG.md`, `PREREG_AMENDMENT_GEMINI.md` (sha256 `411b248ab959...`), and the un-launched `PREREG_AMENDMENT_MISTRAL.md` all retain their original byte content. Results from this amendment are reported in the paper as **post-hoc panel extension**, not primary findings.

---

## 1. Motivation

After the Gemini extension (panel 12 → 14 targets, 5 → 6 organizations), the next-highest-leverage organization to add is **Moonshot AI**. Three reasons:

1. **Strengthens an existing finding.** The current panel reports a family-clustering pattern where Chinese frontier labs (Qwen 3 235B, DeepSeek V3) drift while OpenAI-non-4.1 and Meta non-frontier do not. Moonshot AI is a third independent Chinese frontier lab; its drift behavior provides confirming or disconfirming evidence for the cluster claim.
2. **Frontier-2026 generation.** Kimi K2.6 is Moonshot's late-2025/early-2026 flagship, comparable in capability class to Sonnet 4.6, GPT-5, Opus 4.7, Gemini 2.5 Pro. No temporal asymmetry with the rest of the panel.
3. **Zero new account setup.** Available via Together AI's serverless endpoint at `moonshotai/Kimi-K2.6`, accessed by the existing `TOGETHER_AI_KEY` already wired up for Llama 3.3 70B, Qwen 3 235B, and DeepSeek V3 in the original 12-target panel.

Mistral Large (the originally-scoped 7th-org target) was deferred because Together AI gates it behind dedicated endpoints rather than serverless, and Mistral's own API requires new account setup. Kimi K2.6 produces equivalent breadth gain (one new frontier organization) at lower friction.

## 2. What's new

### 2.1 New target

| Target | Provider | Tier | Identifier |
|---|---|---|---|
| Kimi K2.6 | Moonshot AI via Together AI | Frontier-class | `moonshotai/Kimi-K2.6` |

Together AI lists `Kimi-K2.6` as a serverless `chat`-type model with 262K context window. OpenAI-compatible endpoint at `https://api.together.xyz/v1`.

### 2.2 No changes to instrument

- **Probes**: identical 25-probe suite from `harness/probes.py`.
- **Conditions**: identical 5-condition content-position protocol (`scratch`, `recent3K`, `recent3K_filler`, `recent3K_earlier`, `filler14K`).
- **Judges**: Sonnet 4.6 (primary) + GPT-5 (cross-replication audit), held constant.
- **Rubric**: identical 0-3 hedge-compliance rubric from `harness/judge.py`.
- **$c_{\text{pre}}$**: same donated Claude-derived context for Q1; same length-and-structure-matched GPT-5-derived $c_{\text{pre}}$ for Q2 same-target ablation.

### 2.3 Sampling parameters

- `temperature = 0.0` (deterministic decoding, matching panel-extension convention)
- `max_tokens = 4096` (matching the locked Gemini-amendment convention)
- `request timeout = 120s` (defensive measure consistent with panel-extension family)

These match the convention locked in `PREREG_AMENDMENT_GEMINI.md` §2.3.

## 3. Hypotheses

### H1 (Q1 extension): Kimi K2.6 drift on Claude-derived $c_{\text{pre}}$

> Per-target Δ(recent3K vs scratch) on Kimi K2.6. Tested by paired permutation on per-probe differences with Holm correction across the 1 new target (Holm-of-1 = raw p), two-sided α = 0.05.

Possible outcomes (committed before unblinding):
- **Drifts** (Δ ≤ −0.20 with p_holm < 0.05): drifter count grows; family-clustering pattern strengthens with a third Chinese-lab confirming target. Supports the "Chinese frontier models trained partly on synthetic Claude-style data exhibit family-matched drift" reading.
- **Does not drift** (CI includes zero, p > 0.05): non-drifter cluster grows. Weakens the simple "Chinese labs → drift" claim; suggests Moonshot's training pipeline is sufficiently distinct from Qwen/DeepSeek that the family-clustering pattern is not strictly geographic.
- **Borderline (cross-judge fails)**: report descriptively; document as another instance of the cross-judge collapse pattern documented for the original 12-target panel.

### H2 (Q2 extension): Kimi K2.6 family-specificity

> Mean Δ(recent3K vs scratch) for Kimi × Claude-derived $c_{\text{pre}}$ vs Kimi × GPT-5-derived $c_{\text{pre}}$ of equivalent length and structure. Tested by paired permutation on per-probe paired differences within the new same-target ablation cell, two-sided α = 0.05.

Same-target ablation panel post-amendment will be n=5: Sonnet 4.6, Opus 4.7, GPT-4.1, Gemini 2.5 Flash, **Kimi K2.6**.

Possible outcomes (committed before unblinding):
- **Family-specific** (gap > 0.20 with p_holm < 0.05): joins Sonnet 4.6 / GPT-4.1 majority pattern. Same-target panel becomes 3 of 5 strongly family-specific. Particularly informative if Kimi's drift on Claude-derived is large but on GPT-5-derived is small — would support distillation-contamination interpretation.
- **Family-agnostic** (gap ≈ 0): joins Opus 4.7 outlier pattern. Same-target panel stays at 2 of 5 strongly family-specific.
- **Partial / not significant**: joins Gemini Flash pattern (directional but not statistically validated).

## 4. Sample size & statistical procedure

### 4.1 Sample size

25 probes per condition, paired by probe ID. No new seeds (deterministic decoding).

### 4.2 Primary test

Paired permutation (10,000 resamples, two-sided), byte-identical to `scripts/cross_judge_12model_analyze.py`. Holm correction within the family of new tests added by this amendment. α = 0.05.

### 4.3 Cross-judge replication (primary gate)

Each Kimi cell evaluated under both Sonnet 4.6 and GPT-5 judges on full responses. A finding is **cross-judge robust** only if it survives Holm-corrected paired permutation under **both** judges.

### 4.4 Reporting label

Results labeled **"post-hoc panel extension"** in the paper. Headline numbers (12-target panel, n=3 same-target ablation) remain primary; the new 15-target / n=5 numbers are explicitly noted as added in response to cold-review feedback.

## 5. Cost cap & kill conditions

### 5.1 Cost cap

Hard stop at **$50 cumulative API spend** for Phase 2 + 3.

### 5.2 Kill conditions (per phase, hard)

- Empty-output rate > 5% across any cell after the first 5 probes → halt that cell.
- API error rate > 5% across any cell → halt, diagnose.
- Δ outside [-1.5, +0.5] → halt, verify probe responses.
- Cross-judge agreement (Spearman ρ) < 0.4 → flag in writing as a Kimi-specific judge-instability finding; do not silently exclude.

## 6. Permitted deviations log

| Date | Deviation | Reason | Affected analysis |
|---|---|---|---|
| 2026-04-29 | Pivoted target from Mistral Large 2411 (`PREREG_AMENDMENT_MISTRAL.md`) to Kimi K2.6. | Together AI gates Mistral Large behind dedicated endpoints (~$1+/hour) rather than serverless. Mistral's own API requires new account setup. Kimi K2.6 is available on Together's serverless tier, frontier-class, and adds Moonshot AI as a new organization at lower friction. The Mistral amendment is preserved un-launched for future reference. Documented before any Kimi data collection. | Kimi cells; the un-launched Mistral amendment is not affected. |

## 7. What's NOT permitted post-hoc

- Adding Kimi targets beyond `moonshotai/Kimi-K2.6` specified here.
- Changing probes, rubric, judges, or $c_{\text{pre}}$ for Kimi cells.
- Re-running Kimi cells with adjusted parameters after seeing results.
- Promoting any Kimi result to "primary" status.
- Excluding "outlier" probe responses without pre-specified criteria.

## 8. Files affected

- **New code**: `scripts/smoke_test_kimi.py`, `scripts/phase2_kimi_main.py`, `scripts/phase3_kimi_same_target.py`, `scripts/phase4stats_kimi_analyze.py`. Reuses existing `harness/clients_together.py` from the prepped Mistral wrapper.
- **New data**: `data_archive/kimi_panel/` directory tree.
- **Pre-registration documents** (untouched): `PREREG.md`, `PREREG_PATH_A.md`, `WEEK1_PREREG.md`, `PREREG_AMENDMENT_GEMINI.md`, `PREREG_AMENDMENT_MISTRAL.md` retain original byte content.

## 9. Phase plan (informational, not part of locked protocol)

- **Phase 0 (now)**: this amendment + Together API smoke test for Kimi. Hash and commit before Phase 2.
- **Phase 1 (skipped)**: probe-pipeline integrity already validated in Gemini Phase 1 and existing 12-target panel via Together for Llama/Qwen/DeepSeek.
- **Phase 2**: Kimi × 5 conditions × 25 probes × 2 judges. Estimated ~30 min, ~$15.
- **Phase 3**: Kimi × GPT-5-derived $c_{\text{pre}}$ × 5 conditions × 25 probes × 2 judges. Estimated ~25 min, ~$10.
- **Phase 4**: pre-registered Holm-corrected statistics. ~5 min, $0.

Phase 5 (paper integration) is deferred per user direction.

## 10. Signature

Signed by: Anonymous (NeurIPS 2026 D&B submission, under review).
Date: 2026-04-29.
SHA-256 of this file at signing (pre-signature byte content):
`b5516bfab228ae3deff458d2fae3f9adb704d15bb1627c8ea37ab053ccac6259`
Reviewers can verify this hash by removing the lines from "SHA-256 of
this file at signing" through end-of-file and rehashing.

Git SHA at signing: see commit message.

No amendments to this amendment after signing unless explicitly logged in §6.
