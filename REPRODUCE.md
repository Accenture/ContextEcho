# Reproducing ContextEcho

This document maps every paper claim, figure, and table to the **exact
command** that produces it from the released dataset. If a reviewer
wants to verify a specific number from the paper, find the row below
and run the command.

---

## Quick start: reproduce Fig. 2 (the headline forest plot) in 3 commands

```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Either symlink the released dataset, or download it
ln -s /path/to/data_archive_release/results results
ln -s /path/to/data_archive_release/data    data

# 3. Render Fig. 2
python3 plotting/fig2_forest_panelwide.py
# → writes paper/figures/fig2_forest_panelwide.pdf
```

That's the full reproduction loop. Every other claim follows the same
shape.

---

## Reproducing each paper artifact

The paper has **6 body figures** and **6 appendix figures**. Each row
below is the full command and expected output. All paths are relative
to the repository root.

### Body figures

| Paper artifact | What it shows | Data | Render command |
|---|---|---|---|
| **Fig. 1** (persona-space) | 2D PCA of behavioral fingerprint, drifted-coder vs. assistant clusters | `results/persona_space/` | `python3 plotting/fig_fig1_combined.py` |
| **Fig. 2** (probe taxonomy) | 25 probes × 5 categories with per-category drift gap | `results/probes_at_crosscompaction/` | `python3 plotting/fig_v2_probe_taxonomy.py` |
| **Fig. 2** (forest, panelwide) | 23 targets from 10 orgs, drift gap ± 95% CI | `results/probes_at_crosscompaction/` | `python3 plotting/fig2_forest_panelwide.py` |
| **Fig. 4** (surface-vs-substrate) | Qwen 3 32B steering dose-response | `results/persona_space/` | `python3 plotting/fig_v2_surface_vs_substrate.py` |
| **Fig. 5** (mitigation forest) | A-anchor restores Assistant register | `results/anchor_variants/` | `python3 plotting/fig_forest_mitigation.py` |
| **Fig. 6** (stressor forest) | Compliance breakdown + length inflation | `results/instruction_override/` | `python3 plotting/fig_forest_stressors.py` |

### Appendix figures

| Paper artifact | What it shows | Data | Render command |
|---|---|---|---|
| Fig. App-1 (full 25 probes) | 23-target × 25-probe forest | `results/probes_at_crosscompaction/` | `python3 plotting/fig_app_full25probes.py` |
| Fig. App-2 (cross-session) | Per-position trajectories on 3 sessions | `results/cross_session/` + `results/cross_compaction/` | `python3 plotting/fig_app_crosssession.py` |
| Fig. App-3 (anchor decay) | Anchor persistence ≥20 unanchored turns | `results/anchor_decay/` | `python3 plotting/fig_app_anchor_decay.py` |
| Fig. App-4 (anchor size) | Token-budget sweep for anchor recipe | `results/anchor_size_sweep/` | `python3 plotting/fig_app_anchor_size.py` |
| Fig. App-5 (cross-judge) | Sonnet vs GPT-5 paired audit | `results/crossjudge_audit/` | `python3 plotting/fig_app_crossjudge.py` |
| Fig. App-6 (drift onset) | Pre-C₁ turn sweep on 4 Anthropic targets | `results/drift_onset/` | `python3 plotting/fig_app_onset.py` |

### Key numerical claims in the paper

| Paper §  | Claim | Reproduction |
|---|---|---|
| §3 | "17 of 23 targets exceed \|Δ\|≥0.30" | `python3 analysis/consolidate_12model_perm.py` |
| §3 | "Negative-control Δ < 0.30 on all 22 audited targets" | inspect `results/negative_controls/` per-cell JSONs |
| §3 | "9 of 20 cross-compaction crossings RISE post-compaction" | `python3 analysis/analyze_v2_full_trajectory.py --report rise-counts` |
| §4.1 | "A-anchor reaches register ceiling on 8 of 23 rows" | `python3 analysis/analyze_anchor_variants.py --report ceiling-counts` |
| §4.1 | "Compliance restored to 93%-100% on all 4 Anthropic targets" | `python3 analysis/audit_compliance_all_cells.py` |
| §4.2 | "S₂ length 11.6×–31.8× inflation" | `python3 analysis/analyze_dual_surface_combined.py` |
| §4.3 | "SWE-Bench Δ argument fidelity = +0.147 on Sonnet 4.6" | `python3 analysis/analyze_terminalbench.py --report swebench-argfidelity` |

---

## Re-running an experiment from scratch

If you have provider API access and want to **re-collect** the per-cell
data (instead of using the released JSONs), each experiment has a
runner:

| Experiment | Runner | Cost (approx.) | Wall time |
|---|---|---:|---:|
| Cross-compaction trajectory (4 Anthropic × 12 positions) | `python3 experiments/e08_cross_compaction/run.py` | $50 | ~6 hr |
| 23-target panel at P5 | `python3 experiments/e10_cross_org/run_gap_fill.py` | $40 | ~3 hr |
| 25 probes × 12 positions × 23 targets | `python3 experiments/e15_probes_at_crosscompaction/run_12pos_with_a.py` | $200 | ~24 hr |
| Cross-session replication | `python3 experiments/e09_cross_session/run.py` | $30 | ~2 hr |
| A-anchor V0/V2/A_COMBINED ablation | `python3 experiments/e04_path_y/run_a_full_trajectory.py` | $25 | ~2 hr |
| Drift onset sweep (4 Anthropic × 8 turn-counts) | `python3 experiments/e17_drift_onset/run_onset.py` | $15 | ~1.5 hr |
| Stressor-surface S1/S2/S3/S4 | `python3 experiments/e11_instruction_override/run.py` | $10 | ~1 hr |
| SWE-Bench-style continuation (3 targets × 25 cutpoints) | `python3 experiments/e13_swebench/run.py` | $30 | ~2.5 hr |
| TerminalBench fresh-task null | `python3 experiments/e14_terminalbench/run_panel.py` | $20 | ~3 hr |
| Cross-judge audit (Sonnet vs GPT-5, n=190) | `python3 experiments/e15_probes_at_crosscompaction/run_crossjudge_audit.py` | $5 | ~30 min |
| Probe-framing ablation | `python3 experiments/e16_probe_framing_ablation/run.py` | $3 | ~20 min |

All runners are **idempotent** — re-running picks up where it stopped
(per-cell JSONs are atomic per `(target, position, paraphrase, arm)`).

### API keys required

Set these environment variables before running runners (only the
providers used in your runs):

```bash
export ANTHROPIC_API_KEY=...    # Anthropic models + judge
export OPENAI_API_KEY=...       # GPT-4o, GPT-4.1, GPT-5, GPT-5-mini, judge variants
export GOOGLE_API_KEY=...       # Gemini 2.5 Pro, Gemini 2.5 Flash
export OPENROUTER_API_KEY=...   # multiplexer for Llama, DeepSeek, Mistral, Qwen, Kimi
export TOGETHER_API_KEY=...     # alternate route for Llama / Mistral / Qwen
export NVIDIA_API_KEY=...       # Nemotron Super 120B / Nano 30B
export COHERE_API_KEY=...       # Command A / Command R7B
export MISTRAL_API_KEY=...      # Mistral Large / Medium / Small (direct)
```

You only need the keys for providers you intend to call.

---

## Verifying the released data

Three checks reviewers may want to run:

```bash
# 1. PII verification grep — should report 0 leaks
make verify-pii

# 2. Manifest cross-check — validates that every cell referenced in the
#    paper exists in the released tree
python3 -c "import json; m = json.loads(open('data_archive_release/results/MANIFEST.json').read()); \
    print(f'manifest cells: {len(m[\"cells\"])}'); \
    import pathlib; missing = [c for c in m['cells'] if not pathlib.Path('data_archive_release', c['path']).exists()]; \
    print(f'missing on disk: {len(missing)}')"

# 3. Smoke test — runs the harness on one cell and confirms the scorer
#    reproduces the recorded judge_score
make smoke-test
```

---

## Adding a new model to the panel

The harness is provider-agnostic. To extend the panel with a new
target:

1. Add the model id and provider to `harness/clients.py` (or use an
   existing provider client like `clients_openrouter.py` if the model
   is on OpenRouter).
2. Run `python3 experiments/e10_cross_org/run_gap_fill.py
   --target <provider/model_id>` to collect the panel-wide pilot at
   `P5_pre_C6`.
3. (Optional) `python3 experiments/e15_probes_at_crosscompaction/run_12pos_with_a.py
   --target <provider/model_id>` for the full 12-position run.
4. Re-render `Fig. 2` to include the new target.

Per-cell costs scale linearly with `n_probes × n_positions × n_paraphrases`.

---

## Adding a new donated session

The harness is also session-agnostic. To add a 4th donor session:

1. Have the donor sign the consent template at
   `archive/donor_consent_template.md`.
2. Have them run their own pre-redaction with `<USER>`, `<EMPLOYER>`,
   `<EMAIL>`, `<PROJECT>` placeholders.
3. Drop the donor's JSONL at `data/sessions/session_<name>.jsonl`.
4. Run `python3 scripts/anonymize_cell_jsons.py` to produce a
   verified-clean release-tree copy.
5. Run `python3 experiments/e09_cross_session/run.py
   --session <name>` to evaluate the panel on the new session.

The anonymizer's redaction-pattern panel is loaded at runtime from
`scripts/.redaction_patterns.json` (gitignored — copy
`scripts/.redaction_patterns.json.example` and fill in donor-specific
surface forms). Add new patterns there if the donor's pre-redaction
was incomplete.

---

## Repository layout (quick reference)

| Directory | What's there |
|---|---|
| `harness/` | snapshot-then-probe runtime, judge, scorer, multi-provider clients |
| `experiments/` | per-experiment runners (`run.py` per directory) |
| `analysis/` | aggregation, statistical tests, paper-claim auditors |
| `plotting/` | one `.py` per paper figure |
| `scripts/` | utilities (anonymizer, croissant generator, supervisor wrappers) |
| `archive/` | non-shipped scaffolding: consent template, internal notes (excluded from public mirror) |
| `data_archive_release/` | the released artifact (this directory itself; gitignored on public mirror) |

---

## Troubleshooting

**`ImportError: harness.judge`** — Run from the repository root, not
inside a subdirectory. The harness assumes the repo root is on
`sys.path`.

**`KeyError: 'response_text'` when reading a cell** — A small number
of cells (~30 of 41,921) have a `judge_parse_error` flag. Filter them
out via the manifest:

```python
cells = [c for c in manifest["cells"] if not c.get("judge_parse_error")]
```

**Provider rate-limit errors** — Runners retry with backoff. For
sustained rate limits, reduce `--concurrency` (default 4).

**Plot shows different numbers than the paper** — Check the harness
git commit. Each per-cell JSON's `provenance` field records the
harness commit it was produced at; plots use that data. Re-collecting
cells against newer model snapshots will produce different numbers.
