"""Cross-target instruction-override probe: Mistral Small + Kimi K2.6.

Replicates the Sonnet n=10 finding that recent3K causes verbose
non-compliance under format-restrictive instructions (S2_NO_PREAMBLE
Δ=+104 chars, p=0.027 clean drift test) on two more drifters.

Targets:
  mistral-small-latest   (panel-extension Q1 Δ = -0.64, strongest drifter)
  moonshotai/Kimi-K2.6   (panel-extension Q1 Δ = -0.40)

Same n=10 cuts, same 4 arms (scratch / filler3K / gpt5_3K / recent3K),
same 4 stressors (S1 ONE_WORD, S2 NO_PREAMBLE, S3 NO_ACTION, S4 STRICT_JSON).

Output: data_archive/instruction_override/<target>/cut-XXXXX/<arm>/<stressor>/metrics.json
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.run_downstream_continuation import (  # type: ignore
    CUTPOINTS_PATH, OUT_BASE, ACK_MESSAGE, load_transcript_indexed,
    extract_recent3K, get_immediate_context_at,
)
from scripts.run_downstream_continuation_mistral import (  # type: ignore
    call_mistral_with_tools,
)
from scripts.run_downstream_clean_control import make_filler  # type: ignore
from scripts.run_downstream_gpt5_arm import extract_gpt5_recent3K  # type: ignore
from scripts.run_instruction_override_probe import (  # type: ignore
    PROBE_CUT_INDICES, STRESSORS, SYSTEM_PROMPT, build_messages,
)
from harness.clients_mistral import make_mistral_client  # type: ignore
from harness.clients_together import make_together_client  # type: ignore

TARGET_CHARS = 3000

# (model_id, target_safe, client_factory, tool_choice_for_no_tools_call_path)
# For instruction-override probe we DON'T pass tools at all, so tool_choice
# is irrelevant — but Mistral SDK requires tools= to be present if tool_choice
# is passed. We use a minimal call path here (no tools).
TARGETS = [
    ("mistral-small-latest", "mistral-small-latest", make_mistral_client),
    ("moonshotai/Kimi-K2.6", "moonshotai-Kimi-K2-6", make_together_client),
]


def call_no_tools(client, model_id, messages, max_tokens=512):
    """Plain chat completion, no tools — for instruction-override probe."""
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=model_id,
        messages=messages,
        temperature=0.0,
        max_tokens=max_tokens,
        timeout=120,
    )
    elapsed = time.perf_counter() - t0
    msg = resp.choices[0].message
    text = msg.content or ""
    # Some Mistral reasoning models (Magistral) return content as a list of
    # blocks; flatten if needed.
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


def messages_for_openai_compat(arm, ctx_text, user_prior, stressor):
    """Build messages in OpenAI-compatible role/content format.
    Anthropic-style structured content blocks won't work for Mistral/Together;
    flatten to plain string content."""
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    if arm in ("filler3K", "gpt5_3K", "recent3K") and ctx_text:
        msgs.append({"role": "user", "content": ctx_text})
        msgs.append({"role": "assistant", "content": ACK_MESSAGE})
    if user_prior:
        msgs.append({"role": "user", "content": f"[Prior task context]\n{user_prior}"})
        msgs.append({"role": "assistant", "content": "Understood, continuing."})
    msgs.append({"role": "user", "content": stressor})
    return msgs


def main() -> int:
    if not CUTPOINTS_PATH.exists():
        sys.exit("Run scripts/select_cutpoints.py first")
    if not os.environ.get("MISTRAL_API_KEY"):
        sys.exit("Set MISTRAL_API_KEY")
    if not (os.environ.get("TOGETHER_AI_KEY") or os.environ.get("TOGETHER_API_KEY")):
        sys.exit("Set TOGETHER_AI_KEY")

    cuts = json.loads(CUTPOINTS_PATH.read_text())["cutpoints"]
    selected_cuts = [cuts[i] for i in PROBE_CUT_INDICES]
    print(f"Selected {len(selected_cuts)} cutpoints")

    rows = load_transcript_indexed()

    out_base = OUT_BASE.parent / "instruction_override"

    started_overall = time.time()
    for model_id, target_safe, client_factory in TARGETS:
        print(f"\n{'='*60}\nTarget: {target_safe}\n{'='*60}")
        client = client_factory()
        target_dir = out_base / target_safe
        target_dir.mkdir(parents=True, exist_ok=True)

        n_done = 0
        n_total = len(selected_cuts) * 4 * len(STRESSORS)

        for cut in selected_cuts:
            cut_idx = cut["cut_index"]
            recent3K_text = extract_recent3K(rows, cut_idx)
            user_prior = get_immediate_context_at(rows, cut_idx)
            filler3K = make_filler(TARGET_CHARS)
            gpt5_3K = extract_gpt5_recent3K()
            if len(gpt5_3K) > TARGET_CHARS:
                gpt5_3K = gpt5_3K[-TARGET_CHARS:]

            ctx_for = {
                "scratch": None,
                "filler3K": filler3K,
                "gpt5_3K": gpt5_3K,
                "recent3K": recent3K_text,
            }

            for arm in ["scratch", "filler3K", "gpt5_3K", "recent3K"]:
                ctx = ctx_for[arm]
                for label, stressor_text, scorer in STRESSORS:
                    cell_dir = target_dir / f"cut-{cut_idx}" / arm / label
                    metrics_path = cell_dir / "metrics.json"
                    if metrics_path.exists():
                        n_done += 1
                        continue
                    cell_dir.mkdir(parents=True, exist_ok=True)
                    msgs = messages_for_openai_compat(arm, ctx, user_prior, stressor_text)
                    try:
                        text, usage, elapsed = call_no_tools(client, model_id, msgs)
                        compliance = scorer(text)
                        metrics = {
                            "cut_index": cut_idx,
                            "arm": arm,
                            "stressor": label,
                            "response_text": text,
                            "response_len": len(text),
                            "compliance": int(compliance),
                            "wall_clock_sec": elapsed,
                            "input_tokens": usage.get("input_tokens"),
                            "output_tokens": usage.get("output_tokens"),
                        }
                        metrics_path.write_text(json.dumps(metrics, indent=2, default=str))
                        n_done += 1
                        if n_done % 20 == 0 or label == "S2_NO_PREAMBLE":
                            mark = "✓" if compliance else "✗"
                            print(f"  [{n_done:>3}/{n_total}] cut={cut_idx:>5} arm={arm:<9} "
                                  f"{label:<16} {mark} len={len(text)} "
                                  f"resp={text[:50]!r}")
                    except Exception as e:
                        print(f"  [ERROR] cut={cut_idx} arm={arm} {label}: {e}")
        print(f"\n  {target_safe}: {n_done}/{n_total} cells")

    elapsed = time.time() - started_overall
    print(f"\n=== ALL DONE: {elapsed:.0f}s wall ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
