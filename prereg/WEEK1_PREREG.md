# Week 1 Pre-Registration — Opus n=8 Replication

**Frozen on**: 2026-04-24
**Data collection starts**: after this doc is signed and committed
**Target model**: Claude Opus 4.7 (`claude-opus-4-7`)

## Research question

Does context compaction induce persona drift in long agentic coding sessions,
and is this effect separable from user-assigned role as measured by the role
labeler side channel?

## Design

### Conditions (3 × paired seeds)

For each of 8 seeds (seed_id ∈ {910, 911, 912, 913, 914, 915, 916, 917}):
- **none**: compaction disabled
- **plain**: plain compaction at threshold 30K
- **role_aware**: role-aware compaction at threshold 30K

Total: **24 sessions on Opus 4.7**.

### Profile

`debug_and_fix` (frozen at commit 486f51c). 10 bugs, 3 modules, 20 pytest tests.

### Session parameters (locked)

- `max_turns = 60`
- `compaction_threshold_tokens = 30_000`
- `compaction_keep_recent = 10`
- `track_role_labels = True` for all conditions (not just role_aware)
- `probe_checkpoints = (5, 20, 40, 55)`
- Target simulator: GPT-5, max_completion_tokens=4096
- Judge: claude-sonnet-4-6, max_tokens=400

### Per-seed pairing

Running the three conditions with the same seed ID means each condition uses
the **same simulator RNG state and same bug set**. This is the paired design
that enables paired permutation tests.

## Primary hypotheses (frozen)

### H1 — Plain compaction induces drift

> Mean Δ(5→55) for plain > Mean Δ(5→55) for none, tested by paired permutation
> test on (seed_i, plain) − (seed_i, none) differences.
> **Reject H0** if two-sided p < 0.05.

With n=8 pairs, paired permutation can reach p = 1/256 ≈ 0.004 in the
most extreme case.

### H2 — Role-aware compaction attenuates drift vs plain

> Mean Δ(5→55) for role_aware < Mean Δ(5→55) for plain, tested by paired
> permutation test on (seed_i, plain) − (seed_i, role_aware).
> **Reject H0** if two-sided p < 0.05.

### H3 — Role trajectory adds explanatory power beyond compaction count

Fit two OLS regressions on the 24 sessions:

- **Model A** (simple): Δ ~ intercept + n_compactions
- **Model B** (full): Δ ~ intercept + n_compactions + collab_fraction + tool_fraction

Compute Δ R² = R²(B) − R²(A). If Δ R² > 0.1 (rule of thumb for "meaningful
additional variance explained"), role trajectory is a non-redundant factor.

Note: n=24 supports ~4 parameters. Three covariates + intercept = 4; within budget.

## Secondary analyses

### Effect-size disentangling

Report per-coefficient CIs from Model B regression. Specifically the
`n_compactions` coefficient: if its bootstrapped 95% CI excludes zero, we have
evidence that compaction independently drives drift after controlling for role.

### Compaction-boundary probe (exploratory)

For each compacted session, compare probe-score at the checkpoint immediately
before a compaction vs the checkpoint immediately after. If post-compaction
scores drop sharply, it's direct evidence that compaction itself is the drift
event.

### Role-trajectory shift

Does the role trajectory itself shift post-compaction? Compare role labels in
the 5 turns before compaction vs the 5 turns after. If role distribution
drifts after compaction, compaction may be changing how the user simulator
treats the AI (confound worth documenting).

## What counts as each outcome

| Outcome | Criteria |
|---|---|
| **GREEN — strong C3-prime support** | H1 significant (p<0.05), H2 significant (p<0.05), Model B R² ≥ 0.5, n_compactions coef 95% CI excludes zero. |
| **YELLOW — partial support** | H1 significant but H2 not OR vice versa, Model B shows meaningful additional R² but some coefficients' CIs include zero. |
| **RED — C3-prime refuted** | H1 not significant (plain ≈ none), or Model B n_compactions coef 95% CI includes zero after controlling for role. Paper pivots to role-only story or kills entirely. |

## What's NOT permitted post-hoc

- Changing seed list (900-902/903-905/906-908 from feasibility are **not**
  part of this n=8; start fresh with 910-917 so analysis is independent of
  the contaminated pilot)
- Adding seeds after seeing results
- Switching analysis thresholds (p<0.05 is locked)
- Excluding "outlier" seeds without pre-specified criteria
- Changing compaction threshold away from 30K during data collection
- Modifying any of the frozen prompts in `docs/COMPACTION_PROMPTS_FROZEN.md`

## Permitted deviations (log them)

- Re-running failed sessions with the same seed (max_tokens crashes, 429
  storms, transient API failures) — document in deviations log
- Re-running with a larger context cap if a session fails due to context
  overflow — documented separately

## Cost cap

Week 1 stop condition: if total Week 1 spend exceeds $1,000 before the 24
sessions complete, stop and reassess. Rough projection: $600-800.

## Kill conditions (hard)

- Simulator returns empty content in >20% of turns across a sample of 3
  sessions → simulator regression; fix before continuing
- Role labeler returns "other" in >30% of turns across a sample of 3 sessions
  → labeler regression; investigate
- Compaction fires on 0 of 8 "plain" sessions → threshold too high OR logging
  bug; investigate before continuing

## Files affected

- Data: `~/persona_drift_neurips/data/anthropic_claude-opus-4-7_*_seed91*`
- Analysis output: `analysis/week1_opus.json`
- Verdict: `WEEK1_VERDICT.md` written after data collection

## Signature

Signed by: autonomous run per user mandate
Date: 2026-04-24
Git SHA at signing: (see commit that follows)

No amendments after signing unless explicitly logged in this file's deviations
section.
