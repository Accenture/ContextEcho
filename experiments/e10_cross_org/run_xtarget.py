"""Cross-organization extension of the cross-compaction probe.

Replicates the 12-position Dim 9 cross-compaction design on 2 non-Anthropic
drifters: Mistral Small (Q1 Δ=-0.64) and Kimi K2.6 (Q1 Δ=-0.40).

Same positions, same 10 S2_NO_PREAMBLE paraphrases, same 2 arms
(claude_session + filler) as scripts/run_cross_compaction_probe.py.
The only difference is the API client (Mistral la Plateforme + Together AI).

After this lands, the heatmap and small-multiples figures can be redrawn
with 6 rows (4 Anthropic + 2 non-Anthropic) for full cross-org coverage.

Total cells: 12 positions × 2 targets × 10 paraphrases × 2 arms = 480.
Cost projection: ~$10-15 (Mistral cheap, Kimi via Together comparable).
Wall: ~45-60 min.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.run_cross_compaction_probe import (  # type: ignore
    POSITIONS, COMPACT_TURNS, S2_VARIANTS, TRANSCRIPT_PATH,
    find_turn_to_line_index, extract_prefix_at_turn, load_transcript,
)
from scripts.run_downstream_clean_control import make_filler  # type: ignore
from scripts.run_instruction_override_probe import SYSTEM_PROMPT  # type: ignore
from harness.clients_mistral import make_mistral_client  # type: ignore
from harness.clients_together import make_together_client  # type: ignore

OUT_BASE = REPO_ROOT / "data_archive" / "cross_compaction"
ACK_MESSAGE = "Acknowledged. How can I help continue this work?"

# (model_id, target_safe, client_factory)
TARGETS = [
    ("mistral-small-latest", "mistral-small-latest", make_mistral_client),
    ("moonshotai/Kimi-K2.6", "moonshotai-Kimi-K2-6", make_together_client),
]


def call_no_tools(client, model_id, messages, max_tokens=1024):
    """Plain chat completion via OpenAI-compatible endpoint, no tools."""
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=model_id,
        messages=messages,
        temperature=0.0,
        max_tokens=max_tokens,
        timeout=180,
    )
    elapsed = time.perf_counter() - t0
    msg = resp.choices[0].message
    text = msg.content or ""
    # Some Mistral reasoning models return list-of-blocks; flatten if so
    if isinstance(text, list):
        parts = []
        for block in text:
            if isinstance(block, dict):
                btype = block.get("type")
                if btype == "text" and block.get("text"):
                    parts.append(block["text"])
                elif btype not in ("thinking", "reasoning") and "text" in block:
                    parts.append(block["text"])
        text = "\n".join(parts)
    text = (text or "").strip()
    usage = {
        "input_tokens": getattr(resp.usage, "prompt_tokens", None),
        "output_tokens": getattr(resp.usage, "completion_tokens", None),
    }
    return text, usage, elapsed


def build_messages(prefix: str, stressor: str) -> list[dict]:
    """OpenAI-compat messages: system + (prefix + ack)? + stressor."""
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    if prefix:
        msgs.append({"role": "user", "content": prefix})
        msgs.append({"role": "assistant", "content": ACK_MESSAGE})
    msgs.append({"role": "user", "content": stressor})
    return msgs


def run_one(client, model_id: str, prefix: str, stressor: str, out_path: Path) -> dict:
    if out_path.exists():
        try:
            return json.loads(out_path.read_text())
        except Exception:
            pass
    out_path.parent.mkdir(parents=True, exist_ok=True)
    msgs = build_messages(prefix, stressor)
    text, usage, elapsed = call_no_tools(client, model_id, msgs)
    metrics = {
        "response_text": text,
        "response_len": len(text),
        "wall_clock_sec": elapsed,
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
    }
    out_path.write_text(json.dumps(metrics, indent=2, default=str))
    return metrics


def main() -> int:
    if not TRANSCRIPT_PATH.exists():
        sys.exit(f"Transcript missing: {TRANSCRIPT_PATH}")
    if not os.environ.get("MISTRAL_API_KEY"):
        sys.exit("Set MISTRAL_API_KEY")
    if not (os.environ.get("TOGETHER_AI_KEY") or os.environ.get("TOGETHER_API_KEY")):
        sys.exit("Set TOGETHER_AI_KEY")

    print("Loading transcript & turn index...")
    rows = load_transcript()
    turn_to_line, total = find_turn_to_line_index(TRANSCRIPT_PATH)
    print(f"  {total} real turns indexed")

    # Pre-compute prefixes (same as Anthropic-family probe)
    prefixes = {}
    for turn, label in POSITIONS:
        p = extract_prefix_at_turn(rows, turn_to_line, turn, max_chars=30000)
        prefixes[label] = p
        print(f"  {label} (turn {turn}): prefix len = {len(p)} chars")

    OUT_BASE.mkdir(parents=True, exist_ok=True)

    started = time.time()
    n_total = len(POSITIONS) * len(TARGETS) * len(S2_VARIANTS) * 2
    n_done = 0

    for model_id, target_safe, client_factory in TARGETS:
        client = client_factory()
        target_dir = OUT_BASE / target_safe
        target_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n{'='*60}\nTarget: {target_safe}\n{'='*60}")

        for turn, pos_label in POSITIONS:
            prefix = prefixes[pos_label]
            filler = make_filler(len(prefix)) if prefix else ""

            for v_idx, variant in enumerate(S2_VARIANTS):
                # claude_session arm
                claude_path = target_dir / pos_label / f"v{v_idx:02d}" / "claude.json"
                if not claude_path.exists():
                    try:
                        m = run_one(client, model_id, prefix, variant, claude_path)
                        n_done += 1
                        if v_idx == 0:
                            print(f"  [{target_safe} {pos_label} v{v_idx} claude] "
                                  f"len={m['response_len']} in_tok={m['input_tokens']} "
                                  f"resp={m['response_text'][:50]!r}")
                    except Exception as e:
                        print(f"  [ERROR {target_safe} {pos_label} v{v_idx} claude]: {e}")
                else:
                    n_done += 1

                # filler arm
                filler_path = target_dir / pos_label / f"v{v_idx:02d}" / "filler.json"
                if not filler_path.exists():
                    try:
                        m = run_one(client, model_id, filler, variant, filler_path)
                        n_done += 1
                        if v_idx == 0:
                            print(f"  [{target_safe} {pos_label} v{v_idx} filler] "
                                  f"len={m['response_len']} in_tok={m['input_tokens']} "
                                  f"resp={m['response_text'][:50]!r}")
                    except Exception as e:
                        print(f"  [ERROR {target_safe} {pos_label} v{v_idx} filler]: {e}")
                else:
                    n_done += 1

        elapsed = time.time() - started
        print(f"  {target_safe} done ({n_done}/{n_total} total) "
              f"cum_elapsed={elapsed:.0f}s")

    elapsed = time.time() - started
    print(f"\n=== ALL DONE: {n_done}/{n_total} cells, {elapsed:.0f}s wall ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
