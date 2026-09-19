# Pre-Registration Amendment: TerminalBench Validation of Token-Efficiency Drift

**Project**: NeurIPS 2026 D&B submission, "The ContextEcho Probe Suite: Measuring Recency-Content Persona Drift in Long Agentic Coding Sessions"
**Amendment date**: 2026-04-29
**Author**: Anonymous (NeurIPS 2026 D&B submission, under review)
**Status**: **DRAFT — pending review and signature.** Not yet locked. SHA-256 will be recorded in §10 once frozen. No data from this amendment may be promoted to the paper until the file is signed and committed.

> **Signing**: After review, append `- [author signature, date, time]` and timestamp via `shasum -a 256 PREREG_AMENDMENT_TERMINALBENCH.md`. Do not begin Phase 1 (multi-trial scale-up) until this file is signed and timestamped.

> **Relationship to primary pre-registration**: This amendment **does not modify** any primary pre-registered analysis. The original 12-target panel (`PREREG.md`), Q3 substrate track (`PREREG_PATH_A.md`), and Week 1 replication gate (`WEEK1_PREREG.md`) remain locked at their original SHA-256 hashes. Results from this amendment are reported in the paper as a **post-hoc public-benchmark validation** of the token-efficiency / wall-clock practical-significance argument, not as primary findings.

> **Relationship to Phase 0 smoke (already executed)**: The 10-cell smoke (Sonnet 4.6 × scratch/recent3K × 5 tasks, 0 failures) ran 2026-04-29 prior to this amendment. Smoke results are reported as **exploratory/methodological-validation only** and never enter any inferential test in the paper. Their sole purpose was to validate the wiring (`harness/terminus_drifted.py`) and demonstrate that token-efficiency / wall-clock signals exist before paying for the multi-trial scale-up. Smoke is committed at `data_archive/terminalbench/smoke/`; per the save-everything-for-release rule, smoke artifacts ship with the public mirror but are labeled "smoke / exploratory" in any paper text.

---

## 1. Motivation

The paper's existing downstream pilot (`Downstream pilot: practical-significance bound`, appendix) uses 25 questions and a 4-dimension register-shape rubric (refusal, hedge-frequency, length, correctness). Reviewers of comparable recent persona-drift / agent-evaluation work (Hivemind 2025, AgentBench, Terminal-Bench public reports) consistently push for:

1. **A public benchmark anyone can re-run** — currently the paper's downstream pilot uses a custom 25-question set without a public counterpart.
2. **Token-efficiency / wall-clock metrics**, not just accuracy or register shape — these are the metrics agentic-coding deployments actually optimize.
3. **Multi-turn task completion**, which the existing single-turn 25-question pilot does not exercise.

This amendment validates the paper's practical-significance claim on **TerminalBench v0.1.1** (`terminal-bench-core==0.1.1`), a public, multi-turn, agentic-coding benchmark. It uses TerminalBench's native instrumentation (`is_resolved`, `total_input_tokens`, `total_output_tokens`, `agent_started_at`, `agent_ended_at`) — no custom rubric, no judge LLM.

This amendment **adds** a benchmark; it does not replace the existing downstream pilot. Both ship in the paper.

## 2. What's new

### 2.1 New benchmark

| Component | Spec |
|---|---|
| Dataset | `terminal-bench-core==0.1.1` (pinned tag) |
| Agent | `harness/terminus_drifted.TerminusDrifted` — subclass of TerminalBench's stock `terminus_1.Terminus`. Pre-seeds `Chat._messages` with `{"role":"user","content":<recent3K>}` then `{"role":"assistant","content":"Acknowledged. How can I help continue this work?"}` when `condition_label=="recent3K"`. Otherwise behaves byte-identically to stock Terminus. |
| recent3K source | `data_archive/terminalbench/recent3K_claude.txt` (3017 bytes, last 3K of donated Claude session, identical to the recent3K used in primary panel Q1). |
| Conditions | `scratch` (no injection) and `recent3K` (injection). The 3 supplementary primary-panel conditions (`recent3K_filler`, `recent3K_earlier`, `filler14K`) are **not** run on TerminalBench. Justification: the position-control suite is the primary panel's job; TerminalBench's job is to validate that the recent3K-vs-scratch contrast generalizes to a public benchmark. |
| Trials per cell | **n = 3 baseline** (target × condition × task) with **auto-bump-to-n=5 rule** (see §4.1). TerminalBench has container-level scheduling and API-latency jitter; n=1 (smoke) cannot carry inferential weight. n=3 is the minimum for paired-difference CIs; n=5 is triggered for cells where the n=3 CI width exceeds the point estimate. |

### 2.2 Targets

The paper's "drift" / "no-drift" classification on the primary 12-target panel under Sonnet judge is treated as the ground truth for selecting TerminalBench targets. The panel below mixes drifters and non-drifters across **two organizations** (Anthropic, OpenAI, Google) so that the per-output-token-latency claim can be tested for **organization-specificity**. If the latency inflation reproduces only on Anthropic targets, the claim narrows to "Anthropic-specific reasoning amplification under recent3K injection." If it reproduces on Google or OpenAI targets too, the claim generalizes.

| Target | Provider | Primary-panel verdict | Rationale |
|---|---|---|---|
| `anthropic/claude-sonnet-4-6` | Anthropic | drifter (Δ = −0.48 Sonnet judge, paper headline) | strongest drift signal in primary panel; smoke-validated 3.5× sec/out_tok |
| `anthropic/claude-haiku-4-5` | Anthropic | borderline | within-Anthropic tier / within-organization control |
| `openai/gpt-5` | OpenAI | non-drifter | cross-organizational non-drifter control; tests whether latency inflation is Anthropic-specific |
| `google/gemini-2.5-pro` | Google | drifter on Claude-derived c_pre (panel-extension) | cross-organizational drifter control; tests whether latency inflation generalizes across reasoning-model families |

4 targets × 2 conditions × 3 trials × 5 tasks = 120 cells (baseline). With auto-bump to n=5 on borderline cells: up to 200 cells worst-case.

GPT-4.1 is **excluded** from this amendment despite its panel drift status: under the gpt-4 family it does not run the same internal-reasoning architecture as GPT-5 / o-series, so it cannot test the latent-reasoning-inflation hypothesis cleanly. If H1/H2 reproduce on GPT-5 + Gemini Pro, a follow-up amendment may add GPT-4.1 as an extension; that decision is out of scope here.

### 2.3 Tasks (subset of TerminalBench)

terminal-bench-core==0.1.1 ships ~80 tasks. Running all 80 × 4 targets × 2 conditions × 3 trials = 1920 cells is outside this amendment's cost cap. The subset is pre-registered here:

| Task | Difficulty (TB-stated) | Why included |
|---|---|---|
| `hello-world` | trivial | sanity / wiring check; recent3K injection cost should equal token delta on this task |
| `crack-7z-hash.easy` | easy | short multi-turn shell flow with concrete success criterion |
| `git-multibranch` | medium | git state + reasoning, multi-turn |
| `swe-bench-astropy-1` | hard / SWE-bench-style | real Python codebase + bug fix; the kind of task this paper's recent3K was extracted from |
| `count-dataset-tokens` is **excluded** (smoke showed both conditions unresolved → no correctness signal worth budget) |
| Optional 5th task slot reserved for one additional `swe-bench-*` if budget allows. |

Final task list locked at signing.

### 2.4 Sampling parameters

- TerminusDrifted inherits stock Terminus defaults: `temperature = 0.7` (TerminalBench's default; **not** Convention B's 0.0).
- Justification for not using Convention B: TerminalBench's evaluation is multi-turn task completion, not single-turn deterministic generation. The TB-stated baseline numbers in the public benchmark were collected at the agent's default temperature; switching to 0.0 would make our numbers non-comparable to TB's reference scores.
- This is a **deliberate deviation** from Convention B and is documented here, not in §6, because it is part of the locked protocol of this amendment from the start.
- `max_episodes = 50` (TB default).

## 3. Hypotheses

The smoke (`data_archive/terminalbench/smoke/SMOKE_USAGE_ANALYSIS.md`) showed visible-content metrics (turn-to-turn redundancy, self-reference, command repetition, response length) all moved <15% on resolved coding tasks, while per-output-token wall-clock inflated 2.6-3.5× and total per-turn wall-clock inflated 3-6×. The hypotheses below promote the per-output-token metric to **primary** because it is robust to turn-count variation across trials and is the cleanest available indirect signature of latent reasoning.

### H1 (PRIMARY, mechanistic): Per-output-token latency drift

> On tasks resolved (`is_resolved == True`) in **both** conditions, the per-trial mean of `sec_per_output_token` differs between scratch and recent3K. `sec_per_output_token` is computed per-turn as `(start_time(turn+1) − start_time(turn)) / output_tokens(turn)` from each episode's `debug.json`, then averaged across turns within a trial. Tested by paired permutation on per-trial paired differences (paired by `(target, task, trial_idx)`), Holm correction across 4 targets, two-sided α = 0.05.

Possible outcomes (committed before unblinding):
- **Per-output-token latency inflates on ≥2 targets**: latent-compute story validated cross-organization (or cross-Anthropic-tier). Strongest paper claim.
- **Inflates only on Anthropic targets**: claim narrows to "Anthropic-specific reasoning amplification under recent3K." Still publishable, narrower scope.
- **Inflates on no targets**: smoke result was n=1 noise; latent-compute claim retracted; paper retains the existing 4-dim register-shape downstream pilot only.

### H2 (PRIMARY, behavioral): Total per-turn wall-clock drift

> On tasks resolved in both conditions, the per-trial mean of `agent_sec_total / n_turns` differs between scratch and recent3K. Same paired-permutation procedure as H1; Holm correction across 4 targets, two-sided α = 0.05.

H2 is the deployment-cost behavioral metric (what someone running the agent in production would feel). It is expected to track H1 closely on resolved tasks (smoke ratio convergence: 5.6× total wall-clock vs 3.5× sec/out_tok on astropy). Reporting both lets the paper say "the latency a deployer feels is mostly invisible compute, not visible output."

### H3 (SECONDARY, descriptive): Output-token-per-turn drift

> On tasks resolved in both conditions, the per-trial mean of `output_tokens` per turn differs between scratch and recent3K. Same paired-permutation procedure; Holm correction across 4 targets, two-sided α = 0.05.

Smoke showed +15-20% growth in visible output tokens — much smaller than the 2.6-3.5× sec/out_tok inflation. H3 directly tracks "longer answers" so the paper can quantify how much of the latency is visible vs invisible. Outcome interpretation:
- **H1 significant + H3 significant + ratio H1/H3 > 2**: latency inflation is mostly latent (small visible-output growth, large per-token-time growth).
- **H1 significant + H3 significant + ratio H1/H3 ≈ 1**: visible-output growth fully explains the latency; "longer answers" is the right story, latent-reasoning is overclaimed.
- **H1 significant + H3 not significant**: cleanest possible "latent compute, no visible content change" claim.

### H4 (SECONDARY, replication): Primary-panel-drift correlation

> Per-target sign of (mean recent3K − mean scratch) on H1's metric (sec_per_output_token) agrees with primary-panel persona-drift verdict (drifter ↔ TB shows latency inflation; non-drifter ↔ no inflation). Tested by exact 4-target sign test (one-sided, α = 0.05). With n=4 the only significant outcome is 4/4 agreement (binomial p = 0.0625 under H₀ of 0.5; we report 4/4 as positive at α≈0.05 with the boundary noted).

Outcomes as in prior amendment H3.

## 4. Sample size & statistical procedure

### 4.1 Sample size & auto-bump rule

**Baseline n = 3 trials per cell**, paired by `(target, task, trial_idx)`. Per target that's 5 tasks × 3 trials = 15 paired trials; only trial-pairs where both conditions resolve enter H1/H2/H3 (others reported descriptively).

**Auto-bump-to-n=5 rule (locked before any data collection):**

After the n=3 pass completes, for each `(target, task)` cell where both conditions had ≥1 resolved trial, compute the **bootstrap 95% CI half-width** of the paired difference on the H1 primary metric (sec_per_output_token). If `CI_half_width >= |point_estimate|`, that cell auto-bumps to n=5 (i.e., 2 additional trials per condition).

The rule is **deterministic and pre-specified**: it triggers from a numerical comparison, not from the analyst's judgment. It does not look at significance, sign, or any specific value — only at whether the CI is wider than the estimate (a standard "underpowered cell" signal). Cells that already have a tight CI at n=3 do not get more trials, conserving budget for the cells that need it.

The auto-bumped n=5 trials are added to the same paired-permutation analysis. No reanalysis of n=3 data after bumping; the bump is a sample-size top-up, not a re-test.

Worst case: all 20 (target, task) cells trigger the bump → 4 targets × 5 tasks × 2 conditions × 5 trials = 200 cells (vs 120 baseline). Cost ceiling raised in §5.1 to absorb this.

### 4.2 Primary tests

- **H1, H2**: paired permutation test on per-trial differences, 10,000 resamples (matching `scripts/cross_judge_12model_analyze.py` byte-identically). Holm correction over 4 targets per hypothesis. Two-sided α = 0.05.
- **H3**: same procedure as H1/H2.
- **H4**: exact sign test, 4-of-4 directional agreement, one-sided α = 0.05.

### 4.3 No cross-judge here

TerminalBench has a deterministic test-suite verdict for each task (`is_resolved`); there is no LLM judge. The cross-judge replication gate is not applicable to this amendment.

### 4.4 Reporting label

Results are labeled **"public-benchmark practical-significance validation (post-hoc)"** in paper text. The headline numbers from the primary panel and same-target ablation remain primary; the TerminalBench numbers are an explicit response to reviewer concerns about token-efficiency and public-benchmark coverage.

### 4.5 Time-to-first-token (TTFT) capture

The smoke artifacts have `completion_start_time = None` because LiteLLM's default debug logger does not capture stream time-to-first-token. TTFT is the cleanest direct measurement of "thinking before first visible token" and would let H1's mechanistic claim (latent reasoning) be verified directly rather than inferred from the sec/out_tok ratio.

**Pre-registered: the scaled run will capture TTFT per turn.** Implementation: the harness will wrap each `chat.chat()` call to record the wall-clock between request send and first token received. This is added to each cell's `results.json` as a new `ttft_seconds_per_turn` array. If TTFT capture fails (e.g., LiteLLM streaming mode incompatibility), the harness falls back to gracefully omitting the field; H1/H2 are not affected.

TTFT is **not** a primary tested metric in this amendment (we don't yet have the smoke evidence to set a meaningful effect-size expectation). It is a **reported descriptive statistic** alongside H1: per-target mean TTFT under each condition. If a future amendment elevates TTFT to primary, that amendment will pre-register the test from these data; the n=3/n=5 trials collected here become its observational sample.

## 5. Cost cap & kill conditions

### 5.1 Cost cap

Hard stop at **$500 cumulative API spend** for this amendment's Phase 1-3 campaign (raised from $300 in earlier draft to absorb the auto-bump-to-n=5 worst case). Estimate based on smoke wall-clock and tokens:

- Smoke (Sonnet × 10 cells) consumed ~$8 of API at ~$0.80/cell average.
- **Baseline n=3**: 4 targets × 2 conditions × 3 trials × 5 tasks = 120 cells × ~$1.50/cell weighted (GPT-5 ~3× Sonnet, Gemini Pro ~1.5× Sonnet, Haiku ~0.3× Sonnet) ≈ **$200**.
- **Auto-bump worst case (all cells bumped to n=5)**: 200 cells × ~$1.50/cell ≈ **$330**. Buffer $170 absorbs swe-bench-astropy-1 outliers (~$5/cell on the 30-min recent3K cell in smoke).
- Realistic auto-bump rate per smoke variance: maybe 30-50% of cells trigger; expected total ~$280-330.

If spend exceeds cap before campaign completion, halt and reassess — do not silently top up beyond cap.

### 5.2 Kill conditions (per phase, hard)

- **Per-cell wall-clock > 60 minutes** (2× the 30-min smoke cap on swe-bench-astropy-1) → kill cell, mark as `failure_mode = "wall_clock_timeout"`, do NOT retry beyond the per-cell trial budget. Trial counts as failed; subsequent trials of the same cell may proceed.
- **Per-target failure-mode rate > 30%** (any failure_mode set) → halt that target, diagnose before continuing other targets.
- **Container/Docker daemon error rate > 10%** → halt full campaign, diagnose Colima before continuing.
- **API rate-limit or billing error** → halt that target only; do not silently top up billing past cost cap.

### 5.3 Resume-from-cache

The orchestrator (`scripts/run_terminalbench_smoke.py` and its scaled successor) uses per-cell caching: a cell is "done" iff `results.json` exists AND the trial has `is_resolved is not None`. Re-running after a kill resumes at the first un-done cell. This is part of the locked protocol; modifying the cache predicate post-launch counts as a deviation and must be logged in §6.

## 6. Permitted deviations log

This amendment was authored 2026-04-29. As of this draft, the only deviations from a strict reading of the primary protocol are those documented in §2.4 (temperature 0.7, not Convention B 0.0) and §2.1 (5-condition position-control suite reduced to scratch/recent3K). Both are part of the **locked design** of this amendment, not deviations from it.

After signing, any further deviation must be logged below with a dated entry.

| Date | Deviation | Reason | Affected analysis |
|---|---|---|---|
| (none yet) | | | |

## 7. What's NOT permitted post-hoc

- Adding TB targets beyond the 4 specified in §2.2.
- Adding TB tasks beyond the 4 (or 5 if optional slot used) specified in §2.3.
- Re-running cells with adjusted parameters after seeing results.
- Promoting any TB result to "primary" status — the post-hoc-validation label is locked.
- Switching agent class (`TerminusDrifted` is locked at the SHA-256 of `harness/terminus_drifted.py` recorded at signing).
- Changing recent3K source (`recent3K_claude.txt` SHA-256 recorded at signing).
- Excluding "outlier" trials without pre-specified criteria. Wall-clock outliers (e.g., cells hitting the 60-min cap) are reported as-is with `failure_mode` set; they are not silently dropped from the test.

## 8. Files affected by this amendment

- **New code** (already exists, locked at amendment SHA): `harness/terminus_drifted.py`, `scripts/run_terminalbench_smoke.py`, `scripts/analyze_smoke_redundancy.py`, `scripts/analyze_smoke_usage.py`.
- **New code** (TBD before Phase 1): `scripts/run_terminalbench_panel.py` (scaled multi-target multi-trial orchestrator with TTFT capture per §4.5 and auto-bump trigger per §4.1; thin extension of the smoke runner).
- **New analysis**: `scripts/analyze_terminalbench.py` (paired permutation per H1/H2/H3 + sign test per H4 + auto-bump CI computation).
- **New data**: `data_archive/terminalbench/panel/<target>/<condition>/<task>/trial-<i>/...` directory tree.
- **Pre-registration documents** (untouched): `PREREG.md`, `PREREG_PATH_A.md`, `WEEK1_PREREG.md`, all six existing panel-extension amendments retain original byte content and SHA-256 hashes.

## 9. Phase plan (informational, not part of locked protocol)

- **Phase 0 (DONE 2026-04-29)**: smoke (Sonnet 4.6 × scratch/recent3K × 5 tasks × n=1) committed at `data_archive/terminalbench/smoke/`. Validated wiring; surfaced sec/output_token inflation 2.6-3.5× and total per-turn wall-clock 3-6× on resolved tasks. Three follow-up digs (turn count, visible-content redundancy, per-turn Anthropic usage) ruled out turn-count thrashing, content redundancy, and prefix-length latency as causes — leaving latent compute as the most likely mechanism. Results are exploratory; no inferential test based on them.
- **Phase 1**: build `scripts/run_terminalbench_panel.py` (with TTFT capture + auto-bump trigger) and `scripts/analyze_terminalbench.py`; smoke them (1 cell each, scratch only, Sonnet only) for ~$5; verify analysis script byte-matches `cross_judge_12model_analyze.py` permutation procedure. ~30 min wall.
- **Phase 2 baseline**: 4-target × 2-condition × 5-task × 3-trial run = 120 cells. Estimated wall: ~30 hours sequential; estimated cost: ~$200.
- **Phase 2 auto-bump**: cells failing the n=3 CI rule (§4.1) get 2 additional trials each; expected 30-50% of cells trigger; expected additional cost ~$80-130.
- **Phase 3**: write up results in `data_archive/terminalbench/panel/PHASE_RESULTS.md`; integrate into paper (`§Practical Significance` subsection of appendix) — **after** Phase 2+bump land.

Phases 2-3 do not begin until this amendment is signed.

## 10. Signature

Signed by: Anonymous (NeurIPS 2026 D&B submission, under review).
Date: 2026-04-29.
SHA-256 of this file at signing (pre-signature byte content):
`8365d3c88e528737a4d88ab61d80adf0341be58dfdaa16f9b5cfad37253dd275`
This hash was computed over the file content **before** this signature block
was finalized; reviewers can verify it by removing the lines from
"SHA-256 of this file at signing" through end-of-file and rehashing.

Companion artifact hashes (locked at signing):
- `harness/terminus_drifted.py`: `4f2f55475f4aa5ff23d0699288f7b15dd03c605379dcf8a631bf2ed9edd60469`
- `scripts/run_terminalbench_smoke.py`: `855dd484b10274d2b76e1f5cb345b4fa7008cd3846aef5ff16daf9f8a8257fba`
- `data_archive/terminalbench/recent3K_claude.txt`: `896be6d0c6299422f02fbac74b2f94b080c01dcb0788172506f232412c1d1765`
- `scripts/analyze_smoke_redundancy.py`: `15d7b5e297ef5d298cd4e485401dcf4048f32752d0f86a042cf8944cde8a3ab4`
- `scripts/analyze_smoke_usage.py`: `9c0906c3ff574bdf9b7575a1b24dde1dc6710947105342787fd5e4c7cfef287a`

- [signed 2026-04-29 19:50 PDT]
