# PREREG amendment: title-only revision

**Date:** 2026-05-03
**Type:** non-substantive (title and brand only; no analysis change)

## Status

The brand name `ContextEcho` is **retained**. An earlier same-day rename
to `PersonaDrift` was reverted: the renamed title produced cognitive
overhead because "PersonaDrift" and the description "Register Drift"
forced reviewers to translate between two equivalent terms. The cleaner
solution is to keep `ContextEcho` as the brand and put the phenomenon
("Persona Drift") in the title.

**No analysis, metric, or data file changed in either direction.** All
per-cell JSON outputs, all SHA-256 hashes locked in earlier prereg
documents, and all numerical results remain identical.

## Final title

> **ContextEcho: A Benchmark for Persona Drift in Long Agentic-Coding Sessions**

The brand `ContextEcho` carries the artifact identity (the model echoes
back the family signal embedded in its context); the subtitle states the
phenomenon ("Persona Drift") and the regime ("Long Agentic-Coding
Sessions"). Brand and subtitle do non-overlapping work; no double
translation is required of the reader.

## What stays unchanged

Same as before any of these naming changes:

- All 25-probe identity battery items, their categories, and verbatim
  text (`harness/probes.py`).
- The `PROBE_FRAMING` string.
- The 4-point assistant-register rubric.
- The snapshot-then-probe primitive.
- All A-anchor variants (V0, V2, V0+V2 = A-anchor) and the size-sweep grid.
- All 4 stressors and the regex compliance scorer (`is_no_preamble`).
- All bootstrap CIs, paired permutation procedures, and Holm correction.
- All per-cell JSON file contents and paths under `results/`.
- All earlier prereg amendments and their SHA-256 hashes.

## Title history (for the record)

| Date    | Title                                                                                         |
|---------|------------------------------------------------------------------------------------------------|
| Initial | ContextEcho: Measuring Persona Drift in Long Agentic Coding Sessions                          |
| Revised | PersonaDrift: A Long Agentic-Coding Benchmark for Register Drift in Frontier LLMs (reverted)  |
| Final   | ContextEcho: A Benchmark for Persona Drift in Long Agentic-Coding Sessions                    |

The body text, plotting scripts, prereg amendments, and README were
search-and-replaced from `ContextEcho` → `PersonaDrift` and back, with
no semantic change in either direction. Backup `.bak` files from sed
were deleted after each pass.

## Verification

```
$ grep -rl "PersonaDrift" paper/main.tex paper/sections/ plotting/ \
    prereg/ README.md data/README.md analysis/ scripts/ \
    2>/dev/null | grep -vE "(_legacy|_v5_pretrim|_obsolete)"
(no output)
$ tectonic main.tex
(compiles cleanly, body p.9, refs p.10)
```
