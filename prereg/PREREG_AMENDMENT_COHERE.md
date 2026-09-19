# Pre-Registration Amendment: Cohere Command A Panel Extension

**Project**: NeurIPS 2026 D&B submission, "The ContextEcho Probe Suite: Measuring Recency-Content Persona Drift in Long Agentic Coding Sessions"
**Amendment date**: 2026-04-29
**Author**: Anonymous (NeurIPS 2026 D&B submission, under review)
**Status**: LOCKED before any Cohere data collection. SHA-256 hash in §10. No post-hoc edits permitted; deviations logged in §6.

> **Relationship to primary pre-registration and prior amendments**: This amendment **does not modify** any primary pre-registered analysis or prior amendments. The original `PREREG.md`, `PREREG_PATH_A.md`, `WEEK1_PREREG.md`, `PREREG_AMENDMENT_GEMINI.md` (411b248ab959…), `PREREG_AMENDMENT_KIMI.md` (b5516bfab228…), `PREREG_AMENDMENT_MISTRAL.md` (187d0c602134…), `PREREG_AMENDMENT_MISTRAL_TIERS.md` (1dfceb4fadc5…), and `PREREG_AMENDMENT_NVIDIA.md` (4adf5fe04cde…) all retain their original byte content. Results from this amendment are reported as **post-hoc panel extension**, not primary findings.

---

## 1. Motivation

After Google + Moonshot + Mistral + NVIDIA panel extensions (panel: 12 → 21 targets, 5 → 9 organizations), **Cohere** is the next-highest-leverage organization to add. Cohere's Command family is a frontier-class effort with a distinct training pipeline emphasizing RAG and grounded generation — different design philosophy from the assistant-conversational models in the existing panel.

Hivemind (Jiang et al., NeurIPS 2025 best paper) included Cohere's `c4ai-command-r-plus-08-2024` and `aya-expanse-32b` in their 25-model panel. Adding Cohere brings the cross-organizational claim to 10 organizations.

This amendment adds Cohere Command A to the panel under the **same protocol**, **same probes**, **same rubric**, **same judges** — only target addition.

## 2. What's new

### 2.1 New target

| Target | Provider | Tier | Identifier |
|---|---|---|---|
| Command A | Cohere | Frontier 2025, 288K context | `command-a-03-2025` |

Accessed via Cohere's OpenAI-compatible endpoint at `https://api.cohere.com/compatibility/v1`, authenticated with `COHERE_API_KEY` from `me/projects/.env`. Wrapper module: `harness/clients_cohere.py`.

`command-a-03-2025` is Cohere's March 2025 frontier release, the canonical "Cohere current flagship general-purpose model" — temporally comparable to Sonnet 4.6 / GPT-5 / Mistral Large 2512 / Gemini 2.5 Pro.

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

These match `PREREG_AMENDMENT_NVIDIA.md` §2.3 byte-identically.

## 3. Hypotheses

### H1 (Q1 extension): Command A drift on Claude-derived $c_{\text{pre}}$

> Per-target Δ(recent3K vs scratch). Tested by paired permutation on per-probe differences, Holm correction over the 1 new target (Holm-of-1 = raw p), two-sided α = 0.05.

Possible outcomes:
- **Drifts** (Δ ≤ −0.20 with p_holm < 0.05): drifter count grows; cross-organizational breadth claim strengthens.
- **Does not drift**: non-drifter cluster grows. Cohere's RAG-grounded training pipeline produces stable Assistant-class behavior on Claude-derived context. Particularly plausible because Cohere's commercial positioning emphasizes grounded / non-personable responses.

### H1' (verbose-baseline hypothesis, exploratory)

> Cohere Command A's scratch baseline mean under Sonnet judge, compared to the typical original-panel range (2.85-2.88) and the panel-extension verbose-baseline cluster.

Cohere is methodologically interesting because their commercial RAG-tuning emphasizes grounded, non-elaborative responses. **A priori prediction**: Command A is unlikely to show verbose-baseline pattern; expect scratch ≥ 2.80.

## 4. Sample size & statistical procedure

### 4.1 Sample size

25 probes per condition, paired by probe ID. No new seeds.

### 4.2 Primary test

Paired permutation (10,000 resamples, two-sided), byte-identical to `scripts/cross_judge_12model_analyze.py`. Holm correction within the family of new tests added by this amendment (1 test for H1 → Holm-of-1 = raw p). α = 0.05.

### 4.3 Cross-judge replication

Sonnet 4.6 (primary) + GPT-5 (cross-replication) on full responses.

### 4.4 Reporting label

Results labeled **"post-hoc panel extension"**.

## 5. Cost cap & kill conditions

### 5.1 Cost cap

Hard stop at **$30 cumulative API spend** for Phase 2 only.

### 5.2 Kill conditions

- Empty-output rate > 5% per cell after first 5 probes → halt.
- API error rate > 5% per cell → halt, diagnose.
- Δ outside [-1.5, +0.5] → halt, verify probe responses.
- Cross-judge agreement (Spearman ρ) < 0.4 → flag.

## 6. Permitted deviations log

| Date | Deviation | Reason | Affected analysis |
|---|---|---|---|
| 2026-04-29 | Amendment scope expanded from `command-a-03-2025` (single target) to also include `command-r7b-12-2024` (small-tier within-Cohere stratification target). Hard cost cap raised from $30 to $50. | Following the within-Mistral and within-NVIDIA tier-stratification design, adding Cohere's small-tier (Command R7B 7B) creates a within-Cohere tier comparison analogous to Gemini Pro/Flash, Mistral Large/Medium/Small, NVIDIA Super-120B/Nano-30B. Tests whether Cohere's preliminary anti-drift pattern (Command A: scratch ≈ 2.76, recent3K ≈ 2.92, Δ ≈ +0.16 — only positive Δ in the panel-extension family) is tier-stable or specific to the frontier flagship. Same wrapper, same protocol, same probes; only model ID differs. Documented before any R7B data collection. | All Cohere cells; Holm correction across N=2 within Cohere |

## 7. What's NOT permitted post-hoc

- Adding Cohere targets beyond `command-a-03-2025`.
- Changing probes, rubric, judges, or $c_{\text{pre}}$ for Cohere cells.
- Re-running Cohere cells with adjusted parameters after seeing results.
- Promoting any Cohere result to "primary" status.

## 8. Files affected

- **New code**: `harness/clients_cohere.py`, `scripts/phase2_cohere_main.py`.
- **New data**: `data_archive/cohere_panel/phase2_main/`.
- **Pre-registration documents** (untouched): all prior amendments retain original byte content and SHA-256 hashes.

## 9. Phase plan (informational)

- **Phase 0 (now)**: this amendment + smoke test. Hash and commit before Phase 2.
- **Phase 1 (skipped)**: probe pipeline already validated.
- **Phase 2**: Command A × 5 conditions × 25 probes × 2 judges. Estimated ~20 min, ~$10 (Cohere is fast — smoke at 0.5s/probe).
- **Phase 3 (deferred)**: same-target ablation not in scope.
- **Phase 4**: stats analysis combined with the existing
  `data_archive/PANEL_EXTENSION_PHASE4STATS.json` family.

## 10. Signature

Signed by: Anonymous.
Date: 2026-04-29.
SHA-256 of this file at signing (pre-signature byte content):
`50a418fae14d841b4e216d0f5b55bad497f761189d49ce59dc7033a36bb30aa4`

Git SHA at signing: see commit message.
