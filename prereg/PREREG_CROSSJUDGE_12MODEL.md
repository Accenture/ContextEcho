# Pre-Registration: Cross-Judge Replication on the 12-Model Panel

**Date locked**: 2026-04-25 (evening, post-cold-review-pass-4)
**SHA-256 to be recorded in Appendix B prior to data collection**
**Status**: LOCKED before any GPT-5 judge re-scoring on the 12-model panel.

## Background

Cold-reviewer pass-4 (Claude review W5, ChatGPT review W11/Q4) flagged that the 12-model behavioral panel uses a Sonnet 4.6 judge, with cross-judge replication only run for the 3-target length-control subset (Section 3.1 of the paper). The full 12-model "8 of 12 drift" claim has not been validated under a non-Anthropic judge. This pre-registration locks in a re-judgment of all 12 targets × 5 conditions × 25 probes = 1500 cells with GPT-5 judge before data collection.

## Targets

All 12 targets from the existing cross-organizational panel (Section 3.2 of the paper):
- Anthropic: Sonnet 4.5, Sonnet 4.6, Haiku 4.5, Opus 4.6, Opus 4.7
- OpenAI: GPT-5, GPT-4o, GPT-4o-mini, GPT-4.1
- Meta: Llama 3.3 70B Instruct Turbo
- Alibaba: Qwen 3 235B Instruct 2507-tput
- DeepSeek: V3

For each target we reuse the EXISTING per-probe target responses (already saved in `docs/CONTENT_POSITION_*.json` or `docs/OPTION_C_*.json` for that target), and only re-run the JUDGE step with GPT-5 instead of Sonnet 4.6.

## Protocol

### Inputs

For each (target, condition, probe) cell, we load the saved `response_preview` (full response text where available; preview if not) from the original Sonnet-judge run. We do NOT re-run target inference — only re-judge.

### Conditions

Five conditions, identical to the original study:
- `scratch` (no prior context)
- `recent3K` (donated $c_{\text{pre}}[-3K:]$)
- `recent3K_filler` (11K filler + recent3K)
- `recent3K_earlier` (11K earlier real + recent3K)
- `filler14K` (14K filler, no recent3K)

### Per cell

1. Load saved target response from existing JSON.
2. Score with GPT-5 judge (model `gpt-5`, frozen rubric, identical `JUDGE_SYSTEM_PROMPT` from `harness/judge.py`).
3. Save score, label, reason.

Re-judging only — no new target generation, no new conditions. Identical 25-probe suite.

### Sample

12 targets × 5 conditions × 25 probes = 1500 judge calls. At ~$0.015 per GPT-5 judge call, ~$22 total cost.

### Statistical analysis

Pre-registered: per-probe paired permutation test on each (target, condition) cell, identical methodology to the original Sonnet-judge analysis (10K sign-flip resamples, Holm-Bonferroni within target across 4 non-scratch contrasts). Bootstrap 95% CIs on each $\Delta$.

We will additionally compute per-(target, condition) inter-judge agreement: Spearman $\rho$ between Sonnet-judge and GPT-5-judge per-probe scores, and quadratic-weighted Cohen's $\kappa$ on the same paired data.

## Pre-registered hypotheses

**H_CJ1 (primary)**: Of the 7 targets that drift behaviorally under the Sonnet-judge pre-registered primary (Sonnet 4.5, Sonnet 4.6, Opus 4.6, Opus 4.7, GPT-4.1, Qwen 3 235B, DeepSeek V3 — i.e., those with Perm-Holm $p < 0.05$ or borderline), at least 4 still clear Holm-corrected $\alpha = 0.05$ under GPT-5 judge (paired permutation, within-target Holm). Pre-register decision: $\geq 4$ = SUCCESS (cross-judge robust); $\in \{1, 2, 3\}$ = PARTIAL (some judge-side bias narrows the claim but core finding survives); $0$ = COLLAPSE (the 12-model claim is judge-specific, retract to a Sonnet-judge-only claim).

**H_CJ2 (secondary)**: Inter-judge Spearman $\rho \geq 0.5$ on the paired per-probe scores across the full panel (1500 cells). If $\rho < 0.5$, the two judges disagree systematically and the cross-judge gate is informative; if $\rho \geq 0.5$, the judges substantially agree and the surviving-contrast count under H_CJ1 reflects power, not judge disagreement.

**H_CJ3 (secondary)**: The 4 non-drifting targets (GPT-5, GPT-4o, GPT-4o-mini, Llama 3.3 70B) remain non-drifting under GPT-5 judge (no contrast clears Perm-Holm). If a non-drifter starts drifting under GPT-5 judge, this would suggest the GPT-5 judge has its own family bias.

## Excluded analyses

- We will NOT cherry-pick a different Holm correction unit. Within-target, 4-contrast correction is locked.
- We will NOT exclude probes after seeing scores. The 25-probe suite is fixed.
- We will NOT re-judge with a third judge (e.g., a Llama judge) unless GPT-5 collapses the claim entirely; in that case, a third judge is the appropriate next step but is out of scope here.

## Decision rules

- **SUCCESS**: $\geq 4$ of 7 originally-drifting targets clear Perm-Holm under GPT-5 judge AND inter-judge $\rho \geq 0.5$. Result: paper claims "cross-judge robust drift on $\geq 4$ targets across $\geq 2$ organizations under both Anthropic and non-Anthropic judges."

- **PARTIAL**: 1-3 targets clear under GPT-5 judge. Result: paper reports the partial finding honestly. The Sonnet-judge claim narrows but is not fully retracted.

- **COLLAPSE**: 0 targets clear under GPT-5 judge. Result: paper retracts the 12-model panel to a Sonnet-judge-only claim and bounds it as "Sonnet-judge in-family bias may explain the apparent cross-organizational pattern." This is itself a publishable contribution about judge-side bias in cross-organizational studies.

All three outcomes are publishable. We pre-commit to honest reporting.

## Anticipated artifacts

- `docs/CROSS_JUDGE_12MODEL_RAW.json` — per-cell GPT-5 judge scores
- `docs/CROSS_JUDGE_12MODEL_ANALYSIS.json` — H_CJ1/H_CJ2/H_CJ3 statistics
- `docs/CROSS_JUDGE_12MODEL_ANALYSIS.md` — human-readable summary
- An appendix subsection (or new column in Table 1) reporting GPT-5 judge p-values alongside Sonnet judge.

## Code references

- `scripts/cross_judge_12model.py` (to be added; reuses `harness/judge.py` JUDGE_SYSTEM_PROMPT, `analyze_length_control.parse_judge`, OpenAI SDK)

## Locked-by

`Xianzhong.Ding@<institution>` 2026-04-25 evening, prior to any GPT-5 judge API call on the 12-model panel.
