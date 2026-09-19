# Pre-Registration Amendment: Coding-Session Continuation Downstream

**Project**: NeurIPS 2026 D&B submission, "The ContextEcho Probe Suite: Measuring Recency-Content Persona Drift in Long Agentic Coding Sessions"
**Amendment date**: 2026-04-29 (drafted ~22:50 PDT)
**Author**: Anonymous (NeurIPS 2026 D&B submission, under review)
**Status**: **DRAFT — pending review and signature.** Not yet locked. Computed SHA-256 in §10 once frozen. No data collected for this amendment may be promoted to the paper until signed.

> **Relationship to primary pre-registration**: This amendment **does not modify** any primary pre-registered analysis. The original 12-target panel (`PREREG.md`), Q3 substrate track (`PREREG_PATH_A.md`), Week 1 replication gate (`WEEK1_PREREG.md`), and the six existing panel-extension amendments retain their byte content and SHA-256 hashes.

> **Relationship to TerminalBench amendment**: `PREREG_AMENDMENT_TERMINALBENCH.md` (signed SHA `8365d3c8...`) yielded a **clean null** at n=3 on Sonnet 4.6 (sec/turn ratios 0.91-1.30 across 4 tasks; H1 p_holm = 0.626). That amendment tested whether recent3K injection degrades performance on **fresh unrelated coding tasks** (start a new astropy bug fix from scratch). The null result is consistent with a narrower drift mechanism: drift may live in *continuation of the same session*, not in *handling new task instructions*. This amendment tests that narrower hypothesis with a scope-correct design.

---

## 1. Motivation

The paper's contribution is "persona drift in long agentic coding sessions." The reviewer-facing question that *must* be answered is: **"if I deploy a Claude-Code-style agent and let it accumulate context, does its work degrade in a way I would notice?"**

The primary panel (Q1) shows drift on diagnostic probes (register/hedge-compliance scoring). Reviewers can correctly object: probe scoring is a measurement instrument, not a deployment outcome. The paper needs a downstream test where the unit of analysis is *the agent's continued coding work in the same session*, not register style.

We previously attempted this on TerminalBench (signed amendment SHA `8365d3c8`), where the agent is given a fresh unrelated coding task with recent3K as background noise. That amendment yielded a clean null at n=3, which we interpret as evidence that recent3K injection does not bleed into *fresh task instructions*. The narrower deployment-relevant question — *does recent3K degrade continuation of the same session?* — has not yet been tested.

This amendment locks the protocol for that narrower test using the donated Claude Code transcript (`data/session_raw_transcript.jsonl`) as the only existing scope-correct artifact. The transcript contains 8,401 assistant turns, of which 4,698 are `tool_use` invocations — providing abundant ground-truth "next action" labels at any cut point.

## 2. What's new

### 2.1 The cut-point continuation protocol

For each of n=25 pre-registered cut points in the donated session:

1. **Define the cut point**: a sequence index `k` in the transcript where the next assistant turn (`k+1`) is a `tool_use` content block. The tool_use's full payload (tool name + arguments) is the **ground-truth next action**.

2. **Build two arms**:
   - **Scratch arm**: the model receives only the most recent user message at index `k` (the immediate task framing) plus a system prompt instructing it to act as a Claude-Code-like agent and produce its next tool call.
   - **Recent3K arm**: same as scratch, but with the prior `recent3K` (≤3000 chars terminating exactly at the cut point) prepended as a single user message followed by an assistant acknowledgment, identical in structure to the primary-panel `recent3K` injection.

3. **Score continuation quality** on three pre-registered metrics (§3):
   - **Tool-name match (M1)**: did the model emit the same tool as ground truth? Boolean per cut point.
   - **Argument similarity (M2)**: a structured-similarity score on the tool's argument JSON (defined in §3.2). [0, 1] continuous.
   - **LLM-judge alignment (M3)**: a blinded Sonnet 4.6 judge, scoring "would this proposed action be a reasonable continuation of the session at this point?" on 0-3 rubric. The judge sees scratch and recent3K outputs in randomized order without knowing which is which.

### 2.2 Cut-point selection (LOCKED before unblinding)

Cut points are selected **deterministically** from the transcript, not by analyst judgment, to prevent cherry-picking:

- Walk the transcript in order.
- For each assistant turn that contains exactly one `tool_use` content block, record the index.
- From the resulting list of candidate cut points, pick every `(N // 25)`-th one starting from index 100 (skip the first 100 to ensure prior context exists for recent3K extraction). N = total candidates.

This gives 25 cut points spread evenly across the session. The script that selects them is `scripts/select_cutpoints.py` (to be committed alongside this amendment) and runs deterministically (no RNG seed needed).

### 2.3 Targets

| Target | Provider | Rationale |
|---|---|---|
| `anthropic/claude-sonnet-4-6` | Anthropic | primary panel headline drifter (Δ = -0.48); same target as TerminalBench Phase 2 |

**Sonnet 4.6 ONLY for this amendment.** If a signal exists on the strongest drifter, a follow-up amendment can extend to the panel. If null on Sonnet, no further targets justified — same Option-B logic as TerminalBench amendment §2.2.

### 2.4 Sampling parameters

- `temperature = 0.0` (deterministic decoding, matching primary protocol Convention B)
- `max_tokens = 4096`
- Tool-call schema: function-calling structured output via Anthropic native tool-use API

This is **Convention B compliant**, unlike TerminalBench amendment §2.4 which deviated to temperature 0.7 to match TB's reference.

## 3. Hypotheses

### H1 (PRIMARY): Tool-name match rate

> The proportion of cut points where the model emits the same tool as ground truth differs between scratch and recent3K. Tested by paired McNemar test (exact, mid-p) on the n=25 binary outcomes paired by cut point. Two-sided α = 0.05.

Possible outcomes (committed before unblinding):
- **Recent3K reduces match rate by ≥10pp**: drift causes the agent to pick *different tools* for the same continuation point. Strongest evidence of deployment-cost drift.
- **Recent3K matches scratch within ±5pp**: tool selection is robust to recent3K injection.
- **Recent3K *increases* match rate**: surprising; would prompt investigation of the cut-point selection (does recent3K resolve ambiguity that scratch lacks?).

### H2 (PRIMARY): LLM-judge alignment score

> Mean Sonnet-judge 0-3 alignment score differs between scratch and recent3K. Tested by paired permutation (10,000 resamples, seed=42, byte-identical to `cross_judge_12model_analyze.py`). Two-sided α = 0.05.

The judge is **blinded** (sees scratch and recent3K outputs in randomized order, no condition labels). The rubric prompt is locked at signing.

### H3 (SECONDARY): Argument similarity on matched tools

> Among cut points where both arms emit the same tool name (M1==1 in both), mean argument-JSON similarity differs between scratch and recent3K. Same paired permutation procedure as H2.

H3 conditions on H1==1 in both arms, so n_paired ≤ 25 and may be small. Reported descriptively if n_paired < 10.

### H4 (DESCRIPTIVE): No statistical test

Three additional metrics reported descriptively without inferential test:
- Output token count per cut point (does recent3K make the model more verbose in tool args / pre-tool reasoning?)
- LLM call wall-clock per cut point (does recent3K slow the response?)
- Tool argument JSON length (proxy for argument verbosity)

These are reported to **fail explicitly** the same per-turn-time inflation that TerminalBench failed, so the paper has parallel null evidence on continuation as well as fresh tasks.

## 4. Sample size & statistical procedure

### 4.1 Sample size

n = 25 cut points, paired by cut-point index. Single Sonnet 4.6 target. 25 × 2 conditions = 50 model calls.

This is small. Reviewer concern about power: anticipated and acknowledged. Defense: this is the only ground-truth-labeled coding-session-continuation data we have; the design is replicable on additional sessions (none currently ethics-cleared for release).

If H1 or H2 is significant at α=0.05 with n=25, that result will be reported as "preliminary on a single donated session, awaiting cross-session replication." If non-significant, also reported honestly.

### 4.2 Primary tests

- **H1**: paired McNemar exact mid-p test, two-sided α = 0.05.
- **H2, H3**: paired permutation, 10,000 resamples, seed=42, two-sided α = 0.05. Holm correction across {H2, H3} (m=2).

### 4.3 No cross-judge here

Single LLM judge (Sonnet 4.6). Reason: the donated transcript was generated by Sonnet/Opus, and using Sonnet 4.6 as judge avoids the cross-judge replication ambiguity from the primary panel. This is a deliberate scope choice for this amendment, documented here, not in §6.

### 4.4 Reporting label

Results are labeled **"single-session coding-continuation downstream pilot (post-hoc)"** in paper text. Headline numbers from primary panel (Q1) and same-target ablation (Q2) remain primary.

## 5. Cost cap & kill conditions

### 5.1 Cost cap

Hard stop at **$50 cumulative** for this amendment's complete run. Estimated cost: 50 × Sonnet 4.6 calls × ~$0.30/call avg (long context per cut point) = ~$15. Buffer absorbs the LLM-judge calls (50 × ~$0.10 = $5) and any retries. Realistic spend ~$25-30.

If spend exceeds $50, halt and reassess.

### 5.2 Kill conditions

- **Tool-call extraction failure rate > 20%** (model emits malformed tool call) → halt, diagnose schema before continuing.
- **Empty response rate > 10%** → halt, diagnose `max_tokens` before continuing.
- **Cut-point selection produces fewer than 25 candidates** → use all available (N < 25) and report n in results, do NOT introduce subjective backfill.

### 5.3 Resume-from-cache

The orchestrator caches per-cut-point results. Re-launch resumes at the first un-completed cut point.

## 6. Permitted deviations log

Authored 2026-04-29. As of this draft, no deviations from a strict reading of the locked protocol.

After signing, any further deviation must be logged below.

| Date | Deviation | Reason | Affected analysis |
|---|---|---|---|
| (none yet) | | | |

## 7. What's NOT permitted post-hoc

- Adding cut points beyond the deterministic 25 selected by `scripts/select_cutpoints.py`.
- Excluding cut points after seeing results without a pre-specified criterion (none specified — every selected cut point enters the analysis).
- Switching judge model.
- Re-running cells with adjusted parameters after seeing results.
- Promoting any result to "primary" status.

## 8. Files affected

- **New code**: `scripts/select_cutpoints.py` (deterministic cut-point selection), `scripts/run_downstream_continuation.py` (per-cut-point runner), `scripts/analyze_downstream_continuation.py` (paired McNemar + paired permutation).
- **New data**: `data_archive/downstream_coding/<target>/cutpoint-<i>/...` directory tree.
- **Pre-registration documents** (untouched): `PREREG.md`, `PREREG_PATH_A.md`, `WEEK1_PREREG.md`, all six panel-extension amendments, and `PREREG_AMENDMENT_TERMINALBENCH.md` retain their byte content and SHA-256 hashes.

## 9. Phase plan (informational)

- **Phase 0 (DONE 2026-04-29)**: TerminalBench null at n=3 on Sonnet 4.6 (signed amendment SHA `8365d3c8`).
- **Phase 1 (this amendment)**: cut-point selection + runner build + 1-cell wiring smoke (~$1, ~5 min).
- **Phase 2**: full n=25 cut-point run on Sonnet 4.6 (~$25-30, ~2 hr wall).
- **Phase 3**: write up findings; if signal, follow-up amendment for cross-target extension.

Phases 2-3 do not begin until this amendment is signed.

## 10. Signature

**[DRAFT — not yet signed.]**

Signed by: Anonymous (NeurIPS 2026 D&B submission, under review).
Date: pending.
SHA-256 of this file at signing (pre-signature byte content): pending — compute via `awk '/^SHA-256 of this file at signing/{exit} {print}' PREREG_AMENDMENT_DOWNSTREAM_CODING.md | shasum -a 256`.

Companion artifact hashes (locked at signing):
- `scripts/select_cutpoints.py`: pending
- `scripts/run_downstream_continuation.py`: pending
- `scripts/analyze_downstream_continuation.py`: pending
- `data/session_raw_transcript.jsonl`: pending (large file, hash before signing)
