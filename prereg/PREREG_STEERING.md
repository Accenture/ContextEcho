# Pre-Registration: Activation-Steering Causal Test (Path Z)

**Date locked**: 2026-04-25 18:30 PT
**SHA-256 to be recorded in Appendix B prior to GPU data collection (computed at commit time)**
**Status**: LOCKED. Llama H3 has landed (2026-04-25 18:16 PT) and FAILS the predicted null: Llama also shows a significant Assistant Axis projection shift (Cohen d=+1.35 vs Qwen's d=+1.09). This makes Z MORE valuable, not less: Z is now the causal test of whether the axis is load-bearing for behavior, given that the axis is recruited on both targets.

## Background

Path A established that Lu et al.'s Assistant Axis projection shifts under the Claude-Code-derived recency-content protocol on BOTH Qwen 3 32B (Cohen d=+1.09, Welch p=0.0004) and Llama 3.3 70B (Cohen d=+1.35, Welch p<0.0001). The pre-registered Llama null fails: the axis is recruited on both targets, regardless of behavioral drift status. The simplest persona-selectivity reading of Lu et al.'s axis, extended naively from their plain-Q&A regime to long-agentic context, is therefore falsified.

Path A is a **measurement** of substrate recruitment. Path Z asks the **causal** follow-up: given that the axis is recruited on both targets but only one (the behavioral drifters in our 12-target panel) drifts behaviorally, **is the axis causally load-bearing for behavior, or is it a correlated readout?** We test this by intervening on the axis at inference time and measuring whether the behavioral output changes.

Specifically, on Qwen 3 32B (the open-weights drifter where we have the strongest data), we add $\alpha \cdot v$ to the residual stream at the target layer during every forward pass, where $v$ is the negative of the observed scratch-vs-recent3K projection delta along Lu et al.'s normalized axis (i.e., we *push* the residual stream back toward the scratch baseline projection). Three outcomes are pre-registered:

- **Z confirms causation**: steering at $\alpha = 1.0$ restores behavioral hedge-compliance toward scratch baseline → axis IS load-bearing → deployable mitigation primitive identified for open-weights drifters.
- **Z confirms dissociation**: steering at $\alpha = 1.0$ restores projection but leaves behavior unchanged → axis is recruited but NOT load-bearing → strengthens the mechanism-leads-behavior dissociation; behavior is buffered downstream.
- **Z fails sanity**: steering at $\alpha = 1.0$ does not restore projection → implementation failure; report honestly.

Combined with Path Y (surface-level re-anchoring mitigation), Z constitutes a causal-mechanism test: surface-level intervention vs substrate-level intervention. The pair clarifies whether mitigation is achievable conversationally, mechanistically, both, or neither.

## Target

**Qwen 3 32B** only. Llama 3.3 70B is the negative control in Path A; we do not steer Llama because (a) Path A predicts no projection drop on Llama in the first place, and (b) GPU budget is tight.

## Protocol

### Steering primitive

For each (condition, probe) cell, we register a forward hook on the target-layer residual stream (layer 32 of 64 for Qwen 3 32B, identical to Path A). At each generation step, the hook adds a fixed steering vector $\alpha \cdot v$ to the residual stream output of the target layer, where:

- $v$ is the **negative** of the observed recent3K-vs-scratch projection delta along the Assistant Axis: $v = -(p_{\text{recent3K}} - p_{\text{scratch}}) \cdot \hat{a}$, where $\hat{a}$ is the unit-normalized Assistant Axis at layer 32 from Lu et al.'s release. From Path A on Qwen 3 32B (`PATH_A_QWEN3_32B.json`, locked 2026-04-25 17:10 PT): $p_{\text{recent3K}} - p_{\text{scratch}} = -19.77 - (-11.12) = -8.65$, so $v = +8.65 \cdot \hat{a}$ (push back toward scratch baseline along the axis).
- $\alpha$ is a steering coefficient varying across conditions: $\alpha \in \{0, 0.5, 1.0, 1.5\}$ where $\alpha = 0$ is the no-steering baseline (sanity check, should reproduce the Path A behavior), $\alpha = 1.0$ is full subtraction of the observed projection delta, and $\alpha = 1.5$ is over-correction.

The same hook is applied at every generation step (every forward pass during autoregressive decoding), not just at the last input token. This is consistent with Lu et al.'s steering primitive in their released `assistant_axis` package.

### Conditions

For each (steering coefficient $\alpha$) cell we run the recent3K condition (Claude-Code-derived $c_{\text{pre}}[-3K:]$) as the prior context. We do not run scratch (no prior context) under steering — at $\alpha = 0$ scratch should be unchanged, and steering scratch is not informative for the question "can we mitigate drift with substrate-level intervention".

Five steering conditions:
1. `α=0` (recent3K, no steering) — sanity check, should match Path A recent3K
2. `α=0.5` (recent3K, half-strength steering)
3. `α=1.0` (recent3K, full-strength steering — subtracts the observed projection delta)
4. `α=1.5` (recent3K, over-correction)
5. `scratch` (no steering, for behavioral baseline reference)

### Per cell

1. Load Qwen 3 32B, load Lu et al.'s pre-computed Assistant Axis, compute the steering vector $v$ from the Path A `PATH_A_QWEN3_32B.json` per-condition mean projections (so $v$ is fixed and lockable in this pre-reg before Z runs).
2. Format the conversation per the recent3K protocol.
3. Register forward hook on layer 32 that adds $\alpha v$ to the residual stream.
4. Generate the target response (max 400 tokens, temperature 0.7, top-p 0.9, do_sample=True).
5. Capture the **post-steering** residual-stream activation at layer 32, last token (sanity: confirms steering moved the projection by approximately $\alpha \cdot 8.65$).
6. Score the response with the same Sonnet 4.6 LLM judge.

### Sample size

5 conditions × 25 probes = **125 cells**. We do NOT re-run multiple targets — only Qwen 3 32B. Estimated ~$15 GPU on H100:1 for 1-2 hours.

### Statistical analysis

Pre-registered primary: per-probe paired Welch test of $\Delta = \bar{s}(\alpha=1.0) - \bar{s}(\alpha=0)$ on the 25 probes, two-sided. We additionally report the dose-response: Spearman $\rho$ of mean behavioral score across $\alpha \in \{0, 0.5, 1.0, 1.5\}$ vs $\alpha$.

Sanity check: confirm that the post-steering projection at $\alpha = 1.0$ is approximately at the scratch baseline ($-11.12$ ± noise). If not, the steering primitive is not implementing what we think it implements; we report the failure honestly.

## Pre-registered hypotheses

**H_S1 (primary, dissociation prediction)**: At $\alpha = 1.0$ on Qwen 3 32B, the **post-steering residual-stream projection** is restored to approximately the scratch baseline ($-11.12 \pm 2$). If this fails, the steering primitive is broken and we report the implementation failure.

**H_S2 (primary, behavioral)**: At $\alpha = 1.0$, the **behavioral hedge-compliance score** changes by less than 0.2 from the $\alpha = 0$ baseline (i.e., remains essentially unchanged). This is the strong dissociation prediction: substrate-level intervention does NOT change behavioral output on a target whose behavior is already buffered. If H_S2 holds, we have causal evidence for the mechanism-leads-behavior dissociation.

**H_S3 (alternative)**: At $\alpha = 1.0$, the behavioral score increases by ≥ 0.2 toward scratch. This is the weak-dissociation prediction: the substrate does mediate behavior, just below the post-training threshold. If H_S3 holds, the mechanism story is "substrate-mediated, post-training-buffered."

**H_S4 (dose-response)**: Spearman $\rho$ between $\alpha$ and behavioral score is direction-positive across $\alpha \in \{0, 0.5, 1.0, 1.5\}$. Tested as a secondary directional prediction.

## Excluded analyses

- We will NOT cherry-pick the steering layer. Layer 32 is fixed (matches Path A target_layer).
- We will NOT cherry-pick $\alpha$ values after seeing results. Coefficient grid {0, 0.5, 1.0, 1.5} is locked in this document.
- We will NOT exclude probes after seeing scores. The 25 probes are fixed.
- We will NOT add additional model targets for steering. Qwen 3 32B only.

## Decision rules

- **Z SUCCESS, dissociation confirmed (H_S2 holds)**: substrate-level intervention restores the projection but does not change behavior. **This is the strong publishable finding.** It causally confirms the mechanism-leads-behavior dissociation: behavior on Qwen 3 32B is buffered downstream of the Assistant Axis, and the axis is mechanistically engaged but not behaviorally load-bearing on this target.

- **Z PARTIAL (H_S3 holds)**: substrate-level intervention restores both projection AND behavior. **Also publishable**. Mechanism is "substrate-mediated, post-training-buffered" — Qwen 32B's behavioral ceiling is breakable by intervening on the right axis. This provides a deployable mitigation primitive for open-weights drifters.

- **Z NULL (H_S1 fails — steering doesn't move the projection)**: implementation failure. We report honestly and remove Z from the paper, citing the failure as a limitation.

All three outcomes are publishable in their own way. We pre-commit to honest reporting.

## Anticipated artifacts

After Path Z completion:
- `docs/PATH_Z_QWEN3_32B.json` — per-cell behavioral score + post-steering projection per $\alpha$
- `docs/PATH_Z_ANALYSIS.json` — H_S1/H_S2/H_S3/H_S4 statistics with decision-rule output
- `docs/PATH_Z_ANALYSIS.md` — human-readable summary
- A new Section in the paper: "Causal test: activation steering"

## Code references

- `scripts/path_z_steering.py` (to be added; reuses `analyze_length_control.extract_verbatim_slice`, `harness.probes.ALL_PROBES`, `harness.judge`, the Path A activation pipeline; adds a steering hook on layer 32)
- `infra/modal_path_a.py::run_qwen_steering` (to be added; a new Modal function reusing the existing Qwen image)

## Locked-by

To be locked after Llama H3 result lands.

`Xianzhong.Ding@<institution>` 2026-04-25.
