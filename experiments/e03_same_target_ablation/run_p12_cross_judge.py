"""P12: GPT-5 judge replication of the same-target context-source ablation.

The body's same-target ablation (Sonnet 4.6 × Claude-derived vs
GPT-5-derived) was scored by Sonnet 4.6 judge. Cross-judge replication
under GPT-5 judge is the appropriate robustness check.

We re-judge the existing scored responses (no new generations) using
GPT-5 as the judge, and compare the Δ on the recent3K-vs-scratch
contrast.

Cost: ~$3 (judge calls only).

Output:
  docs/P12_CROSS_JUDGE_SAME_TARGET.json
"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from harness.judge import Judge


def collect_responses_to_rejudge():
    """Pull the response_preview text from the existing same-target
    ablation files so we can re-judge them with GPT-5.

    For Sonnet 4.6:
      - Claude-derived: docs/CONTENT_POSITION_SONNET.json
      - GPT-5-derived: docs/A1_CONTEXT_SOURCE_GPT5_DERIVED.json
    """
    out = {}
    # Sonnet 4.6 × Claude-derived
    with open(REPO_ROOT / "docs/CONTENT_POSITION_SONNET.json") as f:
        d = json.load(f)
    out["sonnet46_claude_scratch"] = d["full_results"]["scratch"]["results"]
    out["sonnet46_claude_recent3K"] = d["full_results"]["recent3K"]["results"]
    # Sonnet 4.6 × GPT-5-derived
    with open(REPO_ROOT / "docs/A1_CONTEXT_SOURCE_GPT5_DERIVED.json") as f:
        d = json.load(f)
    fr = d.get("full_results", d)
    if isinstance(fr, dict) and "scratch" in fr:
        out["sonnet46_gpt5_scratch"] = fr["scratch"].get(
            "results", fr["scratch"]) if isinstance(fr["scratch"], dict) else fr["scratch"]
        out["sonnet46_gpt5_recent3K"] = fr["recent3K"].get(
            "results", fr["recent3K"]) if isinstance(fr["recent3K"], dict) else fr["recent3K"]
    return out


def rejudge_cell(judge: Judge, cell: dict) -> dict:
    """Score one (probe_text, response_preview) pair under GPT-5 judge."""
    probe_text = cell.get("probe_text") or cell.get("probe", "")
    response_text = cell.get("response_preview") or cell.get("response", "")
    if not probe_text or not response_text:
        return {**cell, "gpt5_score": -1, "gpt5_label": "missing_input"}
    try:
        j = judge.score(probe_text, response_text)
        return {**cell, "gpt5_score": j.score, "gpt5_label": j.label,
                "gpt5_reason": j.reason}
    except Exception as e:
        return {**cell, "gpt5_score": -1, "gpt5_label": "error",
                "gpt5_error": str(e)}


def run():
    cells = collect_responses_to_rejudge()
    print("Cell groups:", {k: len(v) for k, v in cells.items()}, flush=True)

    judge = Judge(provider="openai", model_id="gpt-5")

    rejudged = {}
    for group, items in cells.items():
        print(f"\n=== Re-judging {group} (n={len(items)}) ===", flush=True)
        rejudged[group] = []
        for i, cell in enumerate(items):
            r = rejudge_cell(judge, cell)
            rejudged[group].append(r)
            print(f"  {cell.get('probe_id', i)}: gpt5={r.get('gpt5_score')}",
                  flush=True)

    # Compute means under GPT-5 judge
    means = {}
    for group, items in rejudged.items():
        scores = [c["gpt5_score"] for c in items
                  if isinstance(c.get("gpt5_score"), int)
                  and c["gpt5_score"] in (0, 1, 2, 3)]
        means[group] = (sum(scores) / len(scores)) if scores else float("nan")

    delta_claude = means["sonnet46_claude_recent3K"] - means["sonnet46_claude_scratch"]
    delta_gpt5 = means["sonnet46_gpt5_recent3K"] - means["sonnet46_gpt5_scratch"]

    output = {
        "experiment": "p12_cross_judge_same_target_ablation",
        "judge_model": "gpt-5 (replication of body's Sonnet 4.6 judge)",
        "target_model": "claude-sonnet-4-6 (claude-sonnet-4-5 API)",
        "per_group_means_under_gpt5_judge": means,
        "delta_claude_derived": delta_claude,
        "delta_gpt5_derived": delta_gpt5,
        "comparison": {
            "sonnet_judge_delta_claude_derived": -0.48,
            "sonnet_judge_delta_gpt5_derived": -0.04,
            "gpt5_judge_delta_claude_derived": delta_claude,
            "gpt5_judge_delta_gpt5_derived": delta_gpt5,
        },
        "rejudged_cells": rejudged,
    }
    out_path = REPO_ROOT / "docs/P12_CROSS_JUDGE_SAME_TARGET.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nWrote {out_path}", flush=True)
    print(f"Sonnet 4.6 same-target ablation under GPT-5 judge:", flush=True)
    print(f"  Claude-derived recent3K Δ: {delta_claude:+.3f} "
          f"(Sonnet judge had -0.48)", flush=True)
    print(f"  GPT-5-derived recent3K Δ:  {delta_gpt5:+.3f} "
          f"(Sonnet judge had -0.04)", flush=True)
    if delta_gpt5 > -0.20 and delta_claude < -0.30:
        print("  -> Same-target ablation REPLICATES under GPT-5 judge.",
              flush=True)
    else:
        print("  -> Same-target ablation result changes under GPT-5 judge.",
              flush=True)


if __name__ == "__main__":
    run()
