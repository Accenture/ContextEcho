# Pre-Registration: Path A — Joint Behavioral + Activation-Space Measurement

**Date locked**: 2026-04-25
**SHA-256 to be recorded in Appendix B prior to data collection**
**Status**: LOCKED before any GPU experiments are launched. No post-hoc changes permitted to this document; deviations during execution must be noted as deviations in the paper, not by editing this file.

## Background

Tier-2/3 of this project (commits `*` through `e422a75`) established a 12-target cross-organizational behavioral phenotype: 8 of 12 frontier models drift on a Claude-Code-derived recency-content protocol, 4 do not, and a same-target context-source ablation (Sonnet 4.6 on Claude- vs GPT-5-derived contexts) shows context source is a necessary causal variable. The behavioral evidence is API-only.

Path A adds activation-space readout for the open-weights drifting and non-drifting targets, using Lu et al.'s pre-computed Assistant Axis (arXiv:2601.10387, github.com/safety-research/assistant-axis, HuggingFace dataset `lu-christina/assistant-axis-vectors`).

## Targets

- **Qwen 3 32B** (`Qwen/Qwen3-32B`) — predicted drifter (Tier-3 API-level Δ on Qwen 3 235B Instruct via Together.ai: −0.80 on `recent3K`). Pre-computed axis: `qwen-3-32b/assistant_axis.pt`. Target layer per Lu et al.: 32 (of 64).
- **Llama 3.3 70B Instruct** (`meta-llama/Llama-3.3-70B-Instruct`) — predicted non-drifter (Tier-3 API-level Δ on Llama 3.3 70B Instruct Turbo via Together.ai: +0.08). Pre-computed axis: `llama-3.3-70b/assistant_axis.pt`. Target layer per Lu et al.: 40 (of 80).

We may run only Qwen 3 32B if Llama 70B requires unavailable HF gating or excess GPU resource. The Qwen-only run is the minimum-viable Path A.

## Protocol

The 5-condition content-position experiment from B2/B4 of this paper:
- `scratch`: no prior context
- `recent3K`: last 3K characters of `c_pre` (Claude-Code-derived donated session, same as B2)
- `recent3K_filler`: 11K of filler prepended to `recent3K`
- `recent3K_earlier`: 11K of earlier `c_pre` content prepended to `recent3K`
- `filler14K`: 14K of filler, no `c_pre`

For each (target, condition, probe) cell, we:
1. Format the conversation: system prompt (`DEFAULT_SYSTEM` from `analyze_length_control.py`), prior context messages per condition, and a user turn containing `PROBE_FRAMING + probe.text`.
2. Generate a model response (max 400 tokens, temperature 0.7, top-p 0.9, do_sample=True).
3. Capture the residual-stream activation at the **target layer** (Lu et al.'s recommended layer per `models.py`: 32 for Qwen 3 32B, 40 for Llama 3.3 70B), at the **last token of the generated response**.
4. Project onto the pre-computed Assistant Axis at the target layer (using Lu et al.'s `project()` function).
5. Score the response with the Sonnet 4.6 LLM judge using our existing `JUDGE_SYSTEM_PROMPT` (paper Appendix A.judge_prompt).
6. Record both the behavioral score and the axis projection per cell.

## Pre-registered hypotheses

**H1 (behavioral × activation correlation on the drifter, primary):**
For Qwen 3 32B, the per-cell axis projection $p_{i,c}$ is positively correlated with the per-cell behavioral score $s_{i,c}$ (higher behavioral score = more Assistant-like = higher axis projection). We pre-register Spearman $\rho \geq 0.5$ as a clear positive result; $\rho \in [0.3, 0.5]$ is borderline; $\rho < 0.3$ is a clear null. Tested on the union of all 5 conditions × 25 probes = 125 cells.

**H2 (mean-projection drop on drifter recent3K vs scratch, primary):**
For Qwen 3 32B, the mean axis projection across the 25 probes in the `recent3K` condition is significantly lower than the mean projection in the `scratch` condition. We pre-register a Welch t-test on the 25 paired projections; reject H0 at $p < 0.05$. Direction of effect: `scratch > recent3K`.

**H3 (no projection drop on non-drifter, primary):**
For Llama 3.3 70B, the mean axis projection in `recent3K` is NOT significantly different from `scratch` (paired Welch t-test, $p > 0.05$). This corroborates the API-level finding that Llama doesn't drift behaviorally.

**H4 (filler-kills-drift signature in projection, secondary):**
For drifters with sharp-dilution behavioral signature (Opus 4.7, GPT-4.1 had this; Qwen may), the axis projection in `recent3K_filler` returns toward `scratch` baseline more than the projection in `recent3K_earlier` does. This is a directional prediction; we will report both projection means and the difference.

**H5 (content-accumulation signature on Qwen, secondary):**
We observed Qwen 3 235B (sister of Qwen 3 32B) showed content-accumulation behaviorally (drift grows when more Claude content is prepended). H5 predicts the same on Qwen 3 32B in activation space: mean axis projection in `recent3K_earlier` is lower (more drifted) than mean axis projection in `recent3K`. This is a directional prediction.

## Excluded analyses

- We will NOT cherry-pick layers other than Lu et al.'s recommended `target_layer` for the primary tests. Other layers may be reported as exploratory.
- We will NOT exclude probes after seeing scores. The 25 probes are fixed (`harness/probes.py:ALL_PROBES`).
- We will NOT use a different judge for the behavioral score on Path A. Sonnet 4.6 with frozen rubric, same as B2.
- If activation extraction fails on >5 of 125 cells per target, we will note this explicitly and not impute.

## Decision rules

- **Path A succeeds** if H1 supports the correlation claim AND H2 supports the projection-drop claim AND H3 supports the null on Llama. Result: paper reframes around "behavioral phenotype + mechanism" with strong experimental evidence.
- **Path A partial** if any 2 of H1/H2/H3 support but the third does not. Result: paper reports the finding honestly with the caveat; mechanism story is weaker but not falsified.
- **Path A null** if H1 fails (correlation < 0.3) or H2 fails (no projection drop on Qwen 3 32B). Result: we explicitly report "behavioral drift on Qwen 3 32B does not correspond to detectable Assistant-Axis projection shift," which is itself an interesting finding (challenges the white-box mechanism story for closed-content-source contexts).

All three outcomes are publishable. We pre-commit to honest reporting regardless.

## Anticipated artifacts

After Path A completion, we will produce:
- `docs/PATH_A_QWEN3_32B.json` — per-cell behavioral score + axis projection
- `docs/PATH_A_LLAMA33_70B.json` — same, for Llama (if run)
- `docs/PATH_A_CORRELATION_ANALYSIS.json` — H1/H2/H3 statistics
- A new Section 5 in the paper: "Activation-space correlates of behavioral drift"

## Code references

- This pre-reg references implementation at:
  - `scripts/path_a_joint_behavioral_activation.py`
  - `infra/modal_path_a.py`
  - `analyze_length_control.py` (for `extract_verbatim_slice`, `PROBE_FRAMING`, `DEFAULT_SYSTEM`)
  - `harness/probes.py` (the 25 probes)
  - `harness/judge.py` (`JUDGE_SYSTEM_PROMPT`)
- And to Lu et al.'s released code:
  - `github.com/safety-research/assistant-axis` (master branch at the time of locking; specific SHA to be recorded in the paper Appendix B)
  - `huggingface.co/datasets/lu-christina/assistant-axis-vectors` (axis files)

## Locked-by

`Xianzhong.Ding@<institution>` 2026-04-25, prior to any GPU spend on `modal_path_a.py::run_*`.
