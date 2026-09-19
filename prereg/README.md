# Pre-registration audit trail

The primary panel and each cross-organization panel extension in the
ContextEcho suite were pre-registered before the corresponding data
were collected. Each `.md` document specifies: the hypotheses tested,
the primary analysis plan, the sample size, and the conditions under
which the experiment would be reported as success / partial / null.

The cross-compaction trajectory, cross-session replication, and
anchor-persistence analyses were **not** separately pre-registered;
they apply the `PREREG.md` judge, stressors, and analysis plan to
additional positions and sessions.

`HASHES.txt` records the SHA-256 of the 15 pre-registration documents
at the time of release. 13 of the 15 carry an explicit lock date in
their header (24–29 April 2026); `PREREG.md` has no date field filled
in, and `PREREG_AMENDMENT_DOWNSTREAM_CODING.md` is marked as an
unsigned draft.

## Index

| Pre-reg file | Experiment | Lock date (from header) |
|---|---|---|
| `PREREG.md` | 12-target length-control panel (primary) | — (not filled in) |
| `WEEK1_PREREG.md` | Synthetic long-session matched-seed null | 2026-04-24 |
| `PREREG_PATH_A.md` | Activation-space readout (Qwen 3 32B + Llama 3.3 70B) | 2026-04-25 |
| `PREREG_STEERING.md` | Substrate steering causal test (Qwen 3 32B, α∈{0,0.5,1.0,1.5}) | 2026-04-25 |
| `PREREG_MITIGATION.md` | Path-Y surface re-anchoring (3 drifters × 3 anchor strengths) | 2026-04-25 |
| `PREREG_CROSSJUDGE_12MODEL.md` | Cross-judge replication (GPT-5 judge re-scoring) | 2026-04-25 |
| `PREREG_DOWNSTREAM_PILOT.md` | Construct-validity bug-fix pilot (3 drifters × 4 outcome dimensions) | 2026-04-25 |
| `PREREG_AMENDMENT_GEMINI.md` | Panel extension: Gemini 2.5 Pro / Flash | 2026-04-28 |
| `PREREG_AMENDMENT_KIMI.md` | Panel extension: Moonshot Kimi K2.6 (OpenRouter-routed) | 2026-04-29 |
| `PREREG_AMENDMENT_MISTRAL.md` | Panel extension: Mistral Large/Medium/Small | 2026-04-29 |
| `PREREG_AMENDMENT_MISTRAL_TIERS.md` | Mistral cross-tier sub-study | 2026-04-29 |
| `PREREG_AMENDMENT_NVIDIA.md` | Panel extension: NVIDIA Nemotron-3 | 2026-04-29 |
| `PREREG_AMENDMENT_COHERE.md` | Panel extension: Cohere Command A / R7B | 2026-04-29 |
| `PREREG_AMENDMENT_TERMINALBENCH.md` | TerminalBench fresh-task null | 2026-04-29 |
| `PREREG_AMENDMENT_DOWNSTREAM_CODING.md` | Downstream coding-continuation 4-arm | draft, unsigned |

`PREREG_AMENDMENT_RENAME.md` (2026-05-03) is a non-substantive
title-only note and is not part of `HASHES.txt`.

## Verification

Paths in `HASHES.txt` are relative to the repository root, so run the
check from there (not from inside `prereg/`):

```bash
shasum -a 256 -c prereg/HASHES.txt
```

Should print `OK` for all 15 lines.
