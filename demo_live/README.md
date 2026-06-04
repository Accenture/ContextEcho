# ContextEcho live demo

An interactive, token-streamed **side-by-side comparison** of the
**session arm** (a real donated Claude Code session prefix) vs. a
**length-matched neutral control arm**, for any probe you type. Watch the two
arms diverge in real time, with a live drift score.

## Prerequisites

1. **The donated session data.** The demo replays real session prefixes, which
   are released separately as the
   [ContextEcho dataset](https://huggingface.co/datasets/contextecho2026/persona-drift-contextecho).
   Download it and make the `session_*.jsonl` files available to the harness:

   ```bash
   huggingface-cli download contextecho2026/persona-drift-contextecho \
       --repo-type=dataset --local-dir data_archive_release
   ln -s data_archive_release/data data    # so the demo finds session_*.jsonl
   ```

2. **An Anthropic API key.** Put it in a `.env` file (in the repo root or its
   parent) or export it:

   ```bash
   export ANTHROPIC_API_KEY=sk-...
   ```

## Run

```bash
python -m demo_live.server
# open http://localhost:8765
```

The default position is pre-warmed in the background so the first probe doesn't
pay the full prefix-build cost.

## What it does

- One `/probe` SSE endpoint per arm; the page opens two `EventSource`
  connections in parallel and streams both responses token-by-token.
- Anthropic prompt caching (`cache_control: ephemeral`) is set on the prefix
  block, so repeat probes at the same `(session, position, target)` pay ~10%
  input-token cost.
- The **length ratio** updates live as both responses complete.
- `/judge` scores each finished response on the published 0–3 drift rubric.
- `/config` enumerates sessions × positions, the target panel, the mitigations
  (none / V0 identity / V2 format / A-combined), and the 25-probe library.

## UI features

- **Session × position dropdowns** — each donated session has its own position
  list with turn counts.
- **Target dropdown** — multiple Anthropic + non-Anthropic targets.
- **Mitigation radios** — `none`, `V0` (identity reminder), `V2` (format demo),
  `A_COMBINED` (both), inserted between the prefix and the probe.
- **Probe library** — 25 canonical probes grouped by category
  (identity / experience / preference / relational / coding-self).
- **Live judge** — a pill renders `judge N/3 · label` once the stream ends.

## Cost / latency

- First call at a fresh `(position, target)`: ~$0.025, 2–6s to first token.
- Repeat calls (cache hit): ~$0.003, <1s to first token.
- Output budget: 1024 tokens per call.

## Customizing

Edit `SESSION_PATH`, `POSITIONS`, and `TARGETS` near the top of `server.py`;
the page discovers them via `/config` on load.
