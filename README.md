# ContextEcho

[![arXiv](https://img.shields.io/badge/arXiv-2605.24279-b31b1b.svg)](https://arxiv.org/abs/2605.24279)
[![Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-contextecho2026-yellow)](https://huggingface.co/datasets/contextecho2026/persona-drift-contextecho)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)

Code release for:

> **ContextEcho: A Benchmark for Persona Drift in Long Agentic-Coding Sessions**
> Xianzhong Ding, Yangyang Yu, Changwei Liu, Bill Zhao. arXiv:2605.24279, 2026.

## News

- **June 2026** — ContextEcho is released alongside our [arXiv preprint](https://arxiv.org/abs/2605.24279), with the full harness, three donated sessions, and the per-cell evaluation corpus on [Hugging Face](https://huggingface.co/datasets/contextecho2026/persona-drift-contextecho).

ContextEcho measures whether a frontier LLM's trained Assistant persona
survives long agentic-coding sessions (thousands of tool-using turns,
hours of continuous use). It is a **25-probe identity suite + harness**
that snapshots a real Claude Code session prefix, forks the
conversation state, and probes any chat-completions API target on the
forked branch — without perturbing the main session.

<p align="center">
  <img src="images/context_echo_intro.png" alt="ContextEcho framework overview" width="100%" />
</p>

## Key findings

Measured across **24 frontier models from 10 organizations** on three
anonymized real Claude Code sessions (3,746–9,716 turns):

| # | Finding | Takeaway |
|---|---------|----------|
| 1 | **Drift is general, not family-specific** | Persona drift appears across organizations, not just one model family. |
| 2 | **Compaction does not reliably reset it** | In-session context compaction fails to restore the trained register. |
| 3 | **A single-shot anchor restores the persona** | One ~110-token anchor turn recovers the trained register across measured targets, persisting 20+ turns. |
| 4 | **Downstream effects are mode-dependent** | Drift can aid tool-using continuation, but in tool-free chat it breaks format contracts and inflates output length. |

See the [paper](https://arxiv.org/abs/2605.24279) for the full results and
per-target tables, and [`REPRODUCE.md`](REPRODUCE.md) for the
claim-by-claim reproduction.

This repository contains the **runtime, experiment runners, analysis,
and plotting code**. The donated session transcripts and the ~42K
per-cell JSON responses are released separately as the
[ContextEcho dataset](#the-released-dataset).

---

## Demo

**Drift in action** — the same model, asked the same question, answers very
differently late in a long session (left arm = real session context, right arm
= length-matched neutral control):

**Left: drift (no anchor) — Right: mitigation (A-anchor applied)**

https://github.com/user-attachments/assets/5ee5629d-ea0a-4e3f-bf67-0cc72652c148

**Try it live.** The repository ships an interactive, token-streamed
side-by-side demo — type any probe and watch the two arms diverge in real time,
with a live drift score. See [`demo_live/`](demo_live/):

```bash
python -m demo_live.server   # then open http://localhost:8765
```

---

## Reproducing the paper

The fastest path is the [`REPRODUCE.md`](REPRODUCE.md) document and the
[`Makefile`](Makefile). Every paper figure has a one-command reproduction:

```bash
# Install dependencies
make setup

# Verify the released data is PII-clean (audit greps for every redaction surface form)
make verify-pii

# Render the headline forest plot (Fig. 2)
make fig2-forest

# All body figures
make figs-body

# All appendix figures
make figs-app

# Show every available target
make help
```

For the claim-by-claim reproduction table, see [REPRODUCE.md](REPRODUCE.md).
For each command's data dependencies, expected output, and approximate
re-collection cost, see the same document.

---

## The released dataset

The per-cell JSONs and 3 donated session transcripts are hosted
separately on Hugging Face:
**[contextecho2026/persona-drift-contextecho](https://huggingface.co/datasets/contextecho2026/persona-drift-contextecho)**.

To use this code with the released data, point it at the dataset:

```bash
# Option 1: symlink the released data tree into the code repo
ln -s /path/to/data_archive_release/results results
ln -s /path/to/data_archive_release/data    data

# Option 2: download from Hugging Face
huggingface-cli download contextecho2026/persona-drift-contextecho \
    --repo-type=dataset --local-dir data_archive_release
ln -s data_archive_release/results results
ln -s data_archive_release/data    data
```

The released dataset includes:

- **3 redacted donor sessions** under `data/sessions/` (310 MB)
- **41,921 per-cell JSON evaluations** under `results/` (705 MB) covering
  the headline cross-compaction trajectory, the 23-target panel at $P_5$,
  the 25-probe × 12-position panel-extension, A-anchor mitigation,
  cross-judge audit, drift-onset sweep, stressor-surface compliance,
  SWE-Bench-style continuation, and TerminalBench fresh-task null
- **`DATASHEET.md`** (Datasheets-for-Datasets format)
- **`croissant.json`** (ML Commons Croissant 1.0 metadata)
- **`LICENSE-DATA`** (CC-BY-SA-4.0) and **`LICENSE-CODE`** (Apache-2.0
  reference)

PII redaction was verified via a 13-pattern grep audit returning 0 hits;
see `data_archive_release/DATASHEET.md` §8 and `make verify-pii`.

---

## Repository layout

| Directory | Contents |
|---|---|
| `harness/` | snapshot-then-probe runtime, multi-provider clients, judge, scorer, probe definitions, cost tracking |
| `experiments/` | per-experiment runners (`run.py` per directory, e.g. `e08_cross_compaction/run.py`) |
| `analysis/` | aggregation, statistical tests, paper-claim auditors |
| `plotting/` | one `.py` per paper figure (e.g. `fig2_forest_panelwide.py`) |
| `scripts/` | utilities (anonymizer, Croissant generator, per-experiment runner wrappers) |
| `donate/` | local browser/terminal wizard for donating a redacted coding-agent session |
| `schemas/` | donation manifest and public-ledger JSON schemas |
| `archive/` | consent template + pre-registration documents |
| `Makefile` | reproduction targets (`make help` for the full list) |
| `REPRODUCE.md` | claim-by-claim reproduction table |
| `DONOR_PRIVACY.md` | donor-facing privacy tiers and guarantees |
| `DATA_USE_POLICY.md` | allowed and disallowed uses for donated data |
| `requirements.txt` | Python dependencies |
| `requirements-donate.txt` | minimal dependencies for the donation wizard |

---

## Re-collecting cells from scratch

If you have provider API access and want to re-run an experiment from
zero (instead of using the released JSONs), each experiment is a self-
contained runner. From the repository root:

```bash
export ANTHROPIC_API_KEY=...    # or whichever provider you're targeting
python3 experiments/e08_cross_compaction/run.py
```

Runners are **idempotent** — they pick up where they stopped, so a
killed run can be resumed. See `REPRODUCE.md` §"Re-running an experiment
from scratch" for the full inventory of runners with cost and wall-time
estimates.

---

## Contributing

ContextEcho is a **living benchmark** — its value grows with every real coding
session it covers. We welcome contributions, and we credit them.

- **Run it on your own session** for a free persona-drift report (yours to keep,
  whether or not you donate).
- **Donate a session or annotation** to join the
  [contributor leaderboard](CONTRIBUTORS.md).
- **Clear the contribution threshold** for **co-authorship on the next dataset
  release** (rolling re-authorship — missing the first paper doesn't close the
  door).

Fastest donor path, after the package is published:

```bash
pipx run contextecho-donate
```

Until then, run directly from the GitHub repo:

```bash
pipx run --spec git+https://github.com/Accenture/ContextEcho.git contextecho-donate
```

Or donate from this cloned repository with the local browser wizard:

```bash
make setup-donate
python3 -m donate --web
```

The wizard discovers local Claude Code/Codex sessions, redacts and verifies
the selected session on your machine, writes `session.redacted.jsonl`,
`manifest.json`, and `CONSENT.md`, then submits only those redacted artifacts
to private maintainer review.

For public collection, maintainers should route uploads through the server-side
relay so the Hugging Face staging token is never shipped to donors. See
[`DONATION_RELAY.md`](DONATION_RELAY.md).

> **Donor privacy.** ContextEcho analyzes **assistant behavior**, not
> **donor personality**. The default donation mode is **full redacted**, which
> removes **PII, secrets, paths, and custom scrub terms** while preserving task
> flow. Donors can choose **user-minimized** mode to selectively mask sensitive
> donor-authored spans after redaction. See [`DONOR_PRIVACY.md`](DONOR_PRIVACY.md) and
> [`DATA_USE_POLICY.md`](DATA_USE_POLICY.md).

Maintainers convert accepted private staging submissions into the next public
dataset candidate with:

```bash
make intake-donations RUN_QUICK=1 PROMOTE=1
```

Accepted sessions are copied into `data_archive_release_v2/data/sessions/`,
with consent, manifests, review reports, and a donation ledger under
`data_archive_release_v2/data/donations/`.
See [`MAINTAINER_DONATION_WORKFLOW.md`](MAINTAINER_DONATION_WORKFLOW.md) for
the full donor-to-ledger workflow and maintainer checklist.

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the points scale, the local-first
redaction pipeline, session scoring rule, confidentiality rules, and exactly
what you get.

---

## Citation

```bibtex
@article{ding2026contextecho,
  title={ContextEcho: A Benchmark for Persona Drift in Long Agentic-Coding Sessions},
  author={Ding, Xianzhong and Yu, Yangyang and Liu, Changwei and Zhao, Bill},
  journal={arXiv preprint arXiv:2605.24279},
  year={2026}
}
```

---

## License

- **Code** (this repository): Apache-2.0
- **Data** (the released dataset, separate host): CC-BY-SA-4.0 per donor
  consent template

The dual license is standard for ML benchmarks that bundle data and
software — Apache-2.0 is the appropriate license for source code,
while CC-BY-SA-4.0 is the appropriate license for the dataset.

---

## Acknowledgments

- The three anonymized session donors, who consented to the release of
  their Claude Code transcripts under the project's donor consent terms.
- The maintainers of [SWE-bench](https://www.swebench.com/) and
  [Terminal-Bench](https://www.tbench.ai/), whose task suites the
  downstream-cost experiments build on.
- The model providers whose chat-completions APIs are evaluated as
  targets in the panel.

---

## Star History

<a href="https://star-history.com/#Accenture/ContextEcho&Date">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=Accenture/ContextEcho&type=Date&theme=dark" />
    <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=Accenture/ContextEcho&type=Date" />
    <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=Accenture/ContextEcho&type=Date" />
  </picture>
</a>
