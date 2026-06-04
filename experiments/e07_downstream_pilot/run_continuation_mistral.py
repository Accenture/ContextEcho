"""Downstream coding-session continuation runner for Mistral-family targets.

Cross-target replication of the Sonnet 4.6 downstream finding (DRAFT
PREREG_AMENDMENT_DOWNSTREAM_CODING). Tests whether the surprising
"recent3K helps argument fidelity" effect (Sonnet p=0.003) replicates
on other panel-extension drifters.

Targets:
  - mistral-large-latest (panel-extension Q1 Δ = -0.30)
  - magistral-medium-latest (panel-extension Q1 Δ = -0.17)

Both via Mistral la Plateforme OpenAI-compatible API. Same n=25 cut
points, same metric definitions as run_downstream_continuation.py.

Output: data_archive/downstream_coding/<target_safe>/cutpoint-XX/metrics.json
PII files (inputs/responses) gitignored.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Re-use the Sonnet runner's helpers + locked tool schema + system prompt.
from scripts.run_downstream_continuation import (  # type: ignore
    CUTPOINTS_PATH, OUT_BASE, ACK_MESSAGE, SYSTEM_PROMPT, TOOLS,
    load_transcript_indexed, extract_recent3K, get_immediate_context_at,
    jaccard_args,
)
from harness.clients_mistral import (  # type: ignore  # noqa: E402
    make_mistral_client, MISTRAL_DEFAULT_MAX_TOKENS, MISTRAL_DEFAULT_TIMEOUT_SEC,
)


TARGETS = [
    ("mistral-large-latest", "mistral-large-latest"),
    ("magistral-medium-latest", "magistral-medium-latest"),
]


def call_mistral_with_tools(client, model_id, messages, tools, max_tokens=4096,
                              tool_choice="any"):
    """Call OpenAI-compat API (Mistral, Together, etc.) with tool-calling.

    tool_choice: "any" for Mistral la Plateforme, "required" for Together AI
    and most OpenAI-compat endpoints. Both force the model to emit a tool call.

    Returns (text, [tool_calls], usage_dict, wall_clock_sec).
    """
    # Convert TOOLS (Anthropic tool_use schema) to OpenAI function-tool schema.
    openai_tools = []
    for t in tools:
        openai_tools.append({
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["input_schema"],
            },
        })

    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=model_id,
        messages=messages,
        tools=openai_tools,
        tool_choice=tool_choice,
        temperature=0.0,
        max_tokens=max_tokens,
        timeout=MISTRAL_DEFAULT_TIMEOUT_SEC,
    )
    elapsed = time.perf_counter() - t0

    msg = resp.choices[0].message
    text = msg.content or ""
    tool_calls = []
    for tc in (msg.tool_calls or []):
        try:
            args = json.loads(tc.function.arguments) if tc.function.arguments else {}
        except json.JSONDecodeError:
            args = {"_parse_error": True, "raw": tc.function.arguments}
        tool_calls.append({"id": tc.id, "name": tc.function.name, "input": args})

    usage = {
        "input_tokens": getattr(resp.usage, "prompt_tokens", None),
        "output_tokens": getattr(resp.usage, "completion_tokens", None),
    }
    return text, tool_calls, usage, elapsed


def run_one_cut_mistral(client, model_id, cut, recent3K_text, user_msg, out_dir: Path,
                          tool_choice="any") -> dict:
    metrics_path = out_dir / "metrics.json"
    if metrics_path.exists():
        try:
            return json.loads(metrics_path.read_text())
        except Exception:
            pass

    out_dir.mkdir(parents=True, exist_ok=True)
    gt_tool = cut["ground_truth_tool"]
    gt_args = cut["ground_truth_args"] or {}

    # Scratch
    scratch_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]
    s_text, s_calls, s_usage, s_sec = call_mistral_with_tools(
        client, model_id, scratch_messages, TOOLS, tool_choice=tool_choice,
    )

    # Recent3K
    recent3K_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": recent3K_text},
        {"role": "assistant", "content": ACK_MESSAGE},
        {"role": "user", "content": user_msg},
    ]
    r_text, r_calls, r_usage, r_sec = call_mistral_with_tools(
        client, model_id, recent3K_messages, TOOLS, tool_choice=tool_choice,
    )

    (out_dir / "inputs.json").write_text(json.dumps({
        "cut_index": cut["cut_index"],
        "user_msg_excerpt": user_msg[:500],
        "recent3K_excerpt": recent3K_text[:500],
        "recent3K_len": len(recent3K_text),
    }, indent=2))
    (out_dir / "scratch_response.json").write_text(json.dumps({
        "text": s_text, "tool_calls": s_calls, "usage": s_usage, "wall_clock_sec": s_sec,
    }, indent=2, default=str))
    (out_dir / "recent3K_response.json").write_text(json.dumps({
        "text": r_text, "tool_calls": r_calls, "usage": r_usage, "wall_clock_sec": r_sec,
    }, indent=2, default=str))

    s_tool = s_calls[0]["name"] if s_calls else None
    r_tool = r_calls[0]["name"] if r_calls else None
    s_args = s_calls[0]["input"] if s_calls else {}
    r_args = r_calls[0]["input"] if r_calls else {}

    metrics = {
        "cut_index": cut["cut_index"],
        "ground_truth_tool": gt_tool,
        "scratch_tool": s_tool,
        "recent3K_tool": r_tool,
        "M1_scratch_match": (s_tool == gt_tool) if s_tool else False,
        "M1_recent3K_match": (r_tool == gt_tool) if r_tool else False,
        "M2_scratch_arg_sim": jaccard_args(s_args, gt_args) if s_tool == gt_tool else None,
        "M2_recent3K_arg_sim": jaccard_args(r_args, gt_args) if r_tool == gt_tool else None,
        "scratch_wall_sec": s_sec,
        "recent3K_wall_sec": r_sec,
        "scratch_input_tokens": s_usage.get("input_tokens"),
        "scratch_output_tokens": s_usage.get("output_tokens"),
        "recent3K_input_tokens": r_usage.get("input_tokens"),
        "recent3K_output_tokens": r_usage.get("output_tokens"),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2, default=str))
    return metrics


def main() -> int:
    if not CUTPOINTS_PATH.exists():
        sys.exit(f"Run scripts/select_cutpoints.py first")
    if not os.environ.get("MISTRAL_API_KEY"):
        sys.exit("Set MISTRAL_API_KEY")

    cuts = json.loads(CUTPOINTS_PATH.read_text())["cutpoints"]
    print(f"Loaded {len(cuts)} cutpoints")

    print("Loading transcript...")
    rows = load_transcript_indexed()

    client = make_mistral_client()

    started_overall = time.time()
    for model_id, target_safe in TARGETS:
        print(f"\n{'='*60}\nTarget: {target_safe}\n{'='*60}")
        target_dir = OUT_BASE / target_safe
        target_dir.mkdir(parents=True, exist_ok=True)
        n_done = 0
        for i, cut in enumerate(cuts):
            out_dir = target_dir / f"cutpoint-{i:02d}"
            if (out_dir / "metrics.json").exists():
                print(f"  [skip {i}] cached")
                n_done += 1
                continue
            recent3K = extract_recent3K(rows, cut["cut_index"])
            user_msg = get_immediate_context_at(rows, cut["cut_index"])
            print(f"\n--- cut {i} (idx={cut['cut_index']}, gt={cut['ground_truth_tool']}) ---")
            try:
                m = run_one_cut_mistral(client, model_id, cut, recent3K, user_msg, out_dir)
                print(f"  scratch tool={m['scratch_tool']} match={m['M1_scratch_match']}  "
                      f"recent3K tool={m['recent3K_tool']} match={m['M1_recent3K_match']}")
                n_done += 1
            except Exception as e:
                print(f"  [ERROR] {e}")
        print(f"\n  {target_safe}: {n_done}/{len(cuts)} cells done")

    elapsed = time.time() - started_overall
    print(f"\n=== ALL DONE: {elapsed:.0f}s wall ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
