# Pre-Registration Amendment: NVIDIA Nemotron Super 120B Panel Extension

**Project**: NeurIPS 2026 D&B submission, "The ContextEcho Probe Suite: Measuring Recency-Content Persona Drift in Long Agentic Coding Sessions"
**Amendment date**: 2026-04-29
**Author**: Anonymous (NeurIPS 2026 D&B submission, under review)
**Status**: LOCKED before any NVIDIA data collection. SHA-256 hash in §10. No post-hoc edits permitted; deviations logged in §6.

> **Signing**: After review, append `- [author signature, date, time]` and timestamp via `shasum -a 256 PREREG_AMENDMENT_NVIDIA.md`.

> **Relationship to primary pre-registration and prior amendments**: This amendment **does not modify** any primary pre-registered analysis or prior amendments. The original `PREREG.md`, `PREREG_PATH_A.md`, `WEEK1_PREREG.md`, `PREREG_AMENDMENT_GEMINI.md` (sha256 `411b248ab959...`), `PREREG_AMENDMENT_KIMI.md` (sha256 `b5516bfab228...`), `PREREG_AMENDMENT_MISTRAL.md` (sha256 `187d0c602134...`), and `PREREG_AMENDMENT_MISTRAL_TIERS.md` (sha256 `1dfceb4fadc5...`) all retain their original byte content. Results from this amendment are reported in the paper as **post-hoc panel extension**, not primary findings.

---

## 1. Motivation

After the Google + Moonshot + Mistral panel extensions (panel: 12 → 19 targets, 5 → 8 organizations), the next-highest-leverage organization to add is **NVIDIA**. NVIDIA's Nemotron family is a frontier-class effort, distinct training pipeline from prior orgs in the panel, and adds hardware-vendor representation. Available on Together AI's serverless tier — zero new account setup.

This amendment adds NVIDIA Nemotron Super 120B-A12B-FP8 to the panel under the **same protocol**, **same probes**, **same rubric**, **same judges** — only target addition.

## 2. What's new

### 2.1 New target

| Target | Provider | Tier | Identifier |
|---|---|---|---|
| Nemotron 3 Super 120B | NVIDIA NIM API | Frontier MoE 2025, 12B active, BF16 | `nvidia/nemotron-3-super-120b-a12b` |

Accessed via NVIDIA's NIM API at `https://integrate.api.nvidia.com/v1` (OpenAI-compatible), authenticated with `NVIDIA_API_KEY` from `me/projects/.env`. Wrapper module: `harness/clients_nvidia.py`. See §6 deviation log for the access-path change from the originally-scoped Together AI gating.

### 2.2 No changes to instrument

- **Probes**: identical 25-probe suite from `harness/probes.py`.
- **Conditions**: identical 5-condition content-position protocol.
- **Judges**: Sonnet 4.6 (primary) + GPT-5 (cross-replication audit), held constant.
- **Rubric**: identical 0-3 hedge-compliance rubric.
- **$c_{\text{pre}}$**: same donated Claude-derived context for Q1.

### 2.3 Sampling parameters

- `temperature = 0.0` (deterministic, panel-extension Convention B)
- `max_tokens = 4096` (matches Convention B)
- `request_timeout = 120s`

These match `PREREG_AMENDMENT_KIMI.md` §2.3 byte-identically (also a Together-hosted target).

## 3. Hypotheses

### H1 (Q1 extension): Nemotron drift on Claude-derived $c_{\text{pre}}$

> Per-target Δ(recent3K vs scratch). Tested by paired permutation on per-probe differences, Holm correction over the 1 new target (Holm-of-1 = raw p), two-sided α = 0.05.

Possible outcomes (committed before unblinding):
- **Drifts** (Δ ≤ −0.20 with p_holm < 0.05): drifter count grows; cross-organizational breadth claim strengthens.
- **Does not drift**: non-drifter cluster grows. NVIDIA's training pipeline produces more stable Assistant-class behavior on Claude-derived context.
- **Borderline (cross-judge fails)**: report descriptively.

### H1' (verbose-baseline hypothesis, exploratory)

> Nemotron's scratch baseline mean under Sonnet judge, compared to the typical original-panel range (2.85-2.88) and the panel-extension verbose-baseline cluster (Pro/Kimi/Mistral non-reasoning: 2.00-2.48). Reported descriptively.

Possible outcomes:
- **Verbose baseline** (scratch < 2.6): joins Pro/Kimi/Mistral cluster as 6th panel-extension target with the pattern. Suggests pattern is broader than Mistral-organization-specific.
- **Typical baseline** (scratch ≥ 2.85): joins Magistral / Gemini Flash / original-panel pattern.
- **Borderline**: document.

## 4. Sample size & statistical procedure

### 4.1 Sample size

25 probes per condition, paired by probe ID. No new seeds (deterministic).

### 4.2 Primary test

Paired permutation (10,000 resamples, two-sided), byte-identical to `scripts/cross_judge_12model_analyze.py`. Holm correction within the family of new tests added by this amendment (1 test for H1 → Holm-of-1 = raw p). α = 0.05.

### 4.3 Cross-judge replication

Sonnet 4.6 (primary) + GPT-5 (cross-replication) on full responses. Cross-judge robustness reported per same gate as primary protocol.

### 4.4 Reporting label

Results labeled **"post-hoc panel extension"**.

## 5. Cost cap & kill conditions

### 5.1 Cost cap

Hard stop at **$30 cumulative API spend** for Phase 2 only.

### 5.2 Kill conditions

- Empty-output rate > 5% per cell after first 5 probes → halt.
- API error rate > 5% → halt, diagnose.
- Δ outside [-1.5, +0.5] → halt, verify probe responses.
- Cross-judge agreement (Spearman ρ) < 0.4 → flag.

## 6. Permitted deviations log

| Date | Deviation | Reason | Affected analysis |
|---|---|---|---|
| 2026-04-29 | Target access path changed from Together AI (`nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8`) to NVIDIA NIM API (`nvidia/nemotron-3-super-120b-a12b`). Wrapper changed from `harness/clients_together.py` to a new `harness/clients_nvidia.py` (OpenAI-compatible against `https://integrate.api.nvidia.com/v1`). Auth changed from `TOGETHER_AI_KEY` to `NVIDIA_API_KEY`. | Together AI gates ALL NVIDIA Nemotron models (Super 120B FP8 + BF16, Nano 30B, Nano 9B, Llama-3.1-Nemotron-70B variants, NIM-prefixed Llama-3.3-Nemotron-49B) behind dedicated endpoints (~$1+/hour idle charges). The NIM API at `build.nvidia.com` provides serverless usage-based access to `nvidia/nemotron-3-super-120b-a12b` (the BF16-precision variant of the same architecture as the originally-scoped FP8 build, same parameter count and capability tier). Smoke test 2026-04-29 confirmed 1.24s probe-shape latency on the NIM-served model. Sampling parameters (Convention B: temperature=0.0, max_tokens=4096) and protocol unchanged; only access path and quantization-precision differ. | All NVIDIA cells |
| 2026-04-29 | Amendment scope expanded from `nvidia/nemotron-3-super-120b-a12b` (single target) to also include `nvidia/nemotron-3-nano-30b-a3b` (second NVIDIA target, same Nemotron-3 architecture family, smaller scale: 30B-A3B vs 120B-A12B). Hard cost cap raised from $30 to $50 to accommodate both targets. | After observing the within-Mistral inverse-size-drift pattern (Small −0.64 > Medium −0.48 > Large −0.30 on Sonnet judge), we want a clean within-architecture scaling test in a different organization. Nemotron-3 Super-120B and Nemotron-3 Nano-30B share architecture family, generation, and training pipeline — only size differs. This is a cleaner scaling test than the within-Mistral comparison (Mistral Large/Medium/Small have slightly different training). H1 extends to a 2-target Holm-corrected family within NVIDIA. The deviation is logged before launching Nano-30B Phase 2. | All NVIDIA cells, Holm correction across N=2 within NVIDIA |

## 7. What's NOT permitted post-hoc

- Adding NVIDIA targets beyond `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8`.
- Changing probes, rubric, judges, or $c_{\text{pre}}$ for Nemotron cells.
- Re-running Nemotron cells with adjusted parameters after seeing results.
- Promoting any Nemotron result to "primary" status.

## 8. Files affected

- **New code**: `scripts/phase2_nvidia_main.py` (template-derived from `scripts/phase2_kimi_main.py` with target swap; reuses `harness/clients_together.py` byte-identically).
- **New data**: `data_archive/nvidia_panel/phase2_main/`.
- **Pre-registration documents** (untouched): all prior amendments retain original byte content and SHA-256 hashes.

## 9. Phase plan (informational)

- **Phase 0 (now)**: this amendment + Together API smoke test for Nemotron.
- **Phase 1 (skipped)**: pipeline already validated in original 12-target panel and Gemini/Kimi extensions.
- **Phase 2**: Nemotron × 5 conditions × 25 probes × 2 judges. Estimated ~30 min, ~$15.
- **Phase 3 (deferred)**: same-target ablation not in scope.
- **Phase 4**: pre-registered Holm-corrected statistics combined with the existing
  `data_archive/PANEL_EXTENSION_PHASE4STATS.json` analysis.

## 10. Signature

Signed by: Anonymous.
Date: 2026-04-29.
SHA-256 of this file at signing (pre-signature byte content):
`6ffffb263eb5a8987492725c40108ebe698e661ce53b7f0b9e5abade319c9134`

Git SHA at signing: see commit message.
