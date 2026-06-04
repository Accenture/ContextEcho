"""Phase 2 main: NVIDIA Nemotron Nano 30B × 5 conditions × 25 probes × 2 judges.

Pre-registration: PREREG_AMENDMENT_NVIDIA.md
SHA-256 (pre-signature): 4adf5fe04cdeee4f21152e2db7f8cbb12c0955d09925c2c5b0efcde0f6044504

Phase 2A: target responses scored by Sonnet 4.6 (primary).
Phase 2B: same target responses re-scored by GPT-5 (cross-judge audit).

Target calls happen ONCE per (condition, probe) and are cached so 2B
re-uses responses from 2A — no double target spend.

Outputs:
  data_archive/nvidia_panel/phase2_main/
    PHASE2_RESULTS.json
    PHASE2_LOG.txt
    target__nvidia-nemotron-3-super-120b-a12b__<condition>.json
    sonnet__nvidia-nemotron-3-super-120b-a12b__<condition>.json
    gpt5__nvidia-nemotron-3-super-120b-a12b__<condition>.json

Hard cost cap: $50 (well above $25 estimate, under $50 amendment cap).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from harness.probes import ALL_PROBES, PROBE_FRAMING
from harness.judge import JUDGE_SYSTEM_PROMPT
from harness.clients_nvidia import call_nvidia, make_nvidia_client

from analyze_length_control import (
    load_events,
    extract_verbatim_slice,
    parse_judge,
)


FILLER_TEMPLATE = (
    "The following is filler placeholder content for an experimental "
    "control. Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
    "The quick brown fox jumps over the lazy dog. Pack my box with five "
    "dozen liquor jugs. The rain in Spain falls mainly on the plain. "
    "How vexingly quick daft zebras jump. The five boxing wizards jump "
    "quickly. Sphinx of black quartz, judge my vow. Two driven jocks "
    "help fax my big quiz. Cwm fjord bank glyphs vext quiz. "
)


def make_filler(target_chars: int) -> str:
    chunks = []
    total = 0
    while total < target_chars:
        chunks.append(FILLER_TEMPLATE)
        total += len(FILLER_TEMPLATE)
    return "".join(chunks)[:target_chars]


PHASE2_DIR = REPO_ROOT / "data_archive" / "nvidia_panel" / "phase2_nano"
PHASE2_RESULTS = PHASE2_DIR / "PHASE2_RESULTS.json"
PHASE2_LOG = PHASE2_DIR / "PHASE2_LOG.txt"

TARGET = "nvidia/nemotron-3-nano-30b-a3b"
SONNET_JUDGE = "claude-sonnet-4-6"
GPT5_JUDGE = "gpt-5"
CONDITIONS = ["scratch", "recent3K", "recent3K_filler", "recent3K_earlier", "filler14K"]

PHASE2_COST_CAP_USD = 50.0

# NVIDIA NIM API pricing for nvidia/nemotron-3-nano-30b-a3b is on a
# usage-based serverless tier; rough estimate ~$2/M input, ~$5/M output
# as of 2026-04-29. Used for in-script cost tracking only; real billing
# is from build.nvidia.com.
PRICING = {
    "nvidia/nemotron-3-nano-30b-a3b": {"in": 2.00, "out": 5.00},
    "claude-sonnet-4-6": {"in": 3.00, "out": 15.00},
    "gpt-5": {"in": 1.25, "out": 10.00},
}


_log_handle = None


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if _log_handle is not None:
        _log_handle.write(line + "\n")
        _log_handle.flush()


def estimate_cost(model_id: str, in_tok: int, out_tok: int) -> float:
    p = PRICING.get(model_id)
    if p is None:
        return 0.0
    return (in_tok / 1_000_000) * p["in"] + (out_tok / 1_000_000) * p["out"]


def build_conditions() -> dict[str, list[dict]]:
    """Build the 5 conditions byte-identical to Gemini Phase 2."""
    events = load_events()
    boundaries = [
        i for i, e in enumerate(events)
        if e.get("type") == "system" and e.get("subtype") == "compact_boundary"
    ]
    if not boundaries:
        raise RuntimeError("No compaction boundary in donated transcript.")
    target_boundary = boundaries[-1]
    verbatim_full = extract_verbatim_slice(events, target_boundary, 28000)
    recent_3K = verbatim_full[-3000:]
    earlier_11K = verbatim_full[-14000:-3000]
    filler_11K = make_filler(len(earlier_11K))
    filler_14K = make_filler(14000)
    ack = {
        "role": "assistant",
        "content": "Acknowledged. How can I help continue this work?",
    }
    return {
        "scratch": [],
        "recent3K": [{"role": "user", "content": recent_3K}, ack],
        "recent3K_filler": [{"role": "user", "content": filler_11K + recent_3K}, ack],
        "recent3K_earlier": [{"role": "user", "content": earlier_11K + recent_3K}, ack],
        "filler14K": [{"role": "user", "content": filler_14K}, ack],
    }


def call_sonnet_judge(client, probe_text: str, response_text: str) -> dict:
    user_msg = (
        f"PROBE:\n{probe_text}\n\nRESPONSE:\n{response_text}\n\n"
        f"Return the JSON now."
    )
    last_err: Exception = RuntimeError("sonnet judge failed without raising")
    for attempt in range(3):
        try:
            resp = client.messages.create(
                model=SONNET_JUDGE,
                system=JUDGE_SYSTEM_PROMPT,
                max_tokens=400,
                messages=[{"role": "user", "content": user_msg}],
            )
            text = "".join(b.text for b in resp.content if hasattr(b, "text"))
            parsed = parse_judge(text)
            parsed["_judge_input_tokens"] = resp.usage.input_tokens
            parsed["_judge_output_tokens"] = resp.usage.output_tokens
            return parsed
        except Exception as e:
            last_err = e
            time.sleep(2 * (2 ** attempt))
    raise last_err


def call_gpt5_judge(client, probe_text: str, response_text: str) -> dict:
    user_msg = (
        f"PROBE:\n{probe_text}\n\nRESPONSE:\n{response_text}\n\n"
        f"Return the JSON now."
    )
    last_err: Exception = RuntimeError("gpt-5 judge failed without raising")
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=GPT5_JUDGE,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                max_completion_tokens=400,
            )
            text = resp.choices[0].message.content or ""
            parsed = parse_judge(text)
            parsed["_judge_input_tokens"] = resp.usage.prompt_tokens
            parsed["_judge_output_tokens"] = resp.usage.completion_tokens
            return parsed
        except Exception as e:
            last_err = e
            time.sleep(2 * (2 ** attempt))
    raise last_err


def cell_path(prefix: str, condition: str) -> Path:
    safe_target = TARGET.replace("/", "-").replace(".", "-")
    return PHASE2_DIR / f"{prefix}__{safe_target}__{condition}.json"


def run_target_cell(target_client, condition, prior, cost_running) -> dict:
    cache = cell_path("target", condition)
    if cache.exists():
        log(f"  [target {condition}] cached")
        return json.loads(cache.read_text())

    log(f"--- target: {TARGET} / {condition} ({len(prior)} prior msgs) ---")
    results = []
    for i, probe in enumerate(ALL_PROBES):
        if cost_running[0] >= PHASE2_COST_CAP_USD:
            log(f"!! cost cap reached at probe {i}")
            results.append({"probe_id": probe.id, "category": probe.category, "error": "cost_cap"})
            break
        framed = f"{PROBE_FRAMING}\n\n{probe.text}"
        msgs = prior + [{"role": "user", "content": framed}]
        try:
            resp_text, in_tok, out_tok = call_nvidia(target_client, TARGET, msgs)
            cost_running[0] += estimate_cost(TARGET, in_tok, out_tok)
        except Exception as e:
            log(f"  [{condition} {i+1}/25] target ERR: {type(e).__name__}: {str(e)[:120]}")
            results.append({
                "probe_id": probe.id, "category": probe.category, "probe_text": probe.text,
                "error": f"target_call: {type(e).__name__}: {str(e)[:240]}",
            })
            continue
        results.append({
            "probe_id": probe.id, "category": probe.category, "probe_text": probe.text,
            "response_text": resp_text, "input_tokens": in_tok, "output_tokens": out_tok,
        })
        if (i + 1) % 5 == 0:
            log(f"  [{condition} {i+1}/25]  cumulative cost=${cost_running[0]:.3f}")

    cell = {
        "target_model": TARGET, "context_source": "claude_derived",
        "condition": condition, "n_probes": len(ALL_PROBES),
        "n_responses": sum(1 for r in results if "response_text" in r),
        "results": results,
    }
    cache.write_text(json.dumps(cell, indent=2))
    log(f"  saved → {cache.name}")
    return cell


def run_judge_cell(judge_name, judge_call, condition, target_cell, cost_running, judge_model_id):
    cache = cell_path(judge_name, condition)
    if cache.exists():
        log(f"  [{judge_name} {condition}] cached")
        return json.loads(cache.read_text())
    log(f"--- {judge_name}: {condition} ---")
    scored = []
    for i, r in enumerate(target_cell["results"]):
        if "error" in r:
            scored.append({"probe_id": r["probe_id"], "category": r["category"], "skipped": "target_error"})
            continue
        if cost_running[0] >= PHASE2_COST_CAP_USD:
            scored.append({"probe_id": r["probe_id"], "category": r["category"], "skipped": "cost_cap"})
            continue
        try:
            judgment = judge_call(r["probe_text"], r["response_text"])
            cost_running[0] += estimate_cost(judge_model_id,
                judgment.get("_judge_input_tokens", 0),
                judgment.get("_judge_output_tokens", 0))
        except Exception as e:
            log(f"  [{condition} {i+1}/25] {judge_name} ERR: {type(e).__name__}")
            scored.append({"probe_id": r["probe_id"], "category": r["category"],
                           "error": f"judge_call: {type(e).__name__}: {str(e)[:240]}"})
            continue
        scored.append({
            "probe_id": r["probe_id"], "category": r["category"],
            "score": judgment.get("score", -1), "label": judgment.get("label", ""),
            "reason": judgment.get("reason", "")[:240],
            "judge_input_tokens": judgment.get("_judge_input_tokens", 0),
            "judge_output_tokens": judgment.get("_judge_output_tokens", 0),
        })
        if (i + 1) % 5 == 0:
            valid = [s["score"] for s in scored if s.get("score", -1) in (0, 1, 2, 3)]
            mean = sum(valid) / max(len(valid), 1)
            log(f"  [{condition} {i+1}/25] running mean={mean:.2f}  cost=${cost_running[0]:.3f}")
    valid = [s["score"] for s in scored if s.get("score", -1) in (0, 1, 2, 3)]
    cell = {
        "judge": judge_name, "judge_model": judge_model_id,
        "target_model": TARGET, "context_source": "claude_derived",
        "condition": condition, "n_valid": len(valid),
        "mean_score": sum(valid) / max(len(valid), 1),
        "results": scored,
    }
    cache.write_text(json.dumps(cell, indent=2))
    log(f"  saved → {cache.name}  mean={cell['mean_score']:.4f}  n_valid={cell['n_valid']}/25")
    return cell


def main() -> int:
    if not os.environ.get("NVIDIA_API_KEY"):
        sys.exit("Set NVIDIA_API_KEY")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("Set ANTHROPIC_API_KEY")
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("Set OPENAI_API_KEY")

    PHASE2_DIR.mkdir(parents=True, exist_ok=True)
    global _log_handle
    _log_handle = open(PHASE2_LOG, "a")

    log(f"Phase 2: {TARGET} × {CONDITIONS}")
    log(f"Pre-reg: PREREG_AMENDMENT_NVIDIA.md "
        f"sha256=4adf5fe04cdeee4f21152e2db7f8cbb12c0955d09925c2c5b0efcde0f6044504")
    log(f"Output dir: {PHASE2_DIR}")

    target_client = make_nvidia_client()
    from anthropic import Anthropic
    from openai import OpenAI
    sonnet_client = Anthropic()
    gpt5_client = OpenAI()

    log("Building conditions...")
    conditions = build_conditions()
    for label, msgs in conditions.items():
        chars = len(msgs[0]["content"]) if msgs else 0
        log(f"  {label}: {chars} chars")

    cost_running = [0.0]
    started = time.time()

    target_cells: dict[str, dict] = {}
    sonnet_cells: dict[str, dict] = {}
    gpt5_cells: dict[str, dict] = {}

    log("\n=== PHASE 2A: targets + Sonnet judge ===")
    for cond in CONDITIONS:
        target_cells[cond] = run_target_cell(target_client, cond, conditions[cond], cost_running)
        sonnet_cells[cond] = run_judge_cell(
            "sonnet", lambda p, r: call_sonnet_judge(sonnet_client, p, r),
            cond, target_cells[cond], cost_running, SONNET_JUDGE,
        )

    log("\n=== PHASE 2B: GPT-5 cross-judge ===")
    for cond in CONDITIONS:
        gpt5_cells[cond] = run_judge_cell(
            "gpt5", lambda p, r: call_gpt5_judge(gpt5_client, p, r),
            cond, target_cells[cond], cost_running, GPT5_JUDGE,
        )

    elapsed = time.time() - started
    log(f"\nPhase 2 complete. wall={elapsed:.0f}s  cost=${cost_running[0]:.3f}")

    summary: dict = {}
    for cond in CONDITIONS:
        summary[cond] = {
            "sonnet_mean": round(sonnet_cells[cond]["mean_score"], 4),
            "sonnet_n_valid": sonnet_cells[cond]["n_valid"],
            "gpt5_mean": round(gpt5_cells[cond]["mean_score"], 4),
            "gpt5_n_valid": gpt5_cells[cond]["n_valid"],
        }
    summary["delta_recent3K_minus_scratch"] = {
        "sonnet": round(summary["recent3K"]["sonnet_mean"] - summary["scratch"]["sonnet_mean"], 4),
        "gpt5": round(summary["recent3K"]["gpt5_mean"] - summary["scratch"]["gpt5_mean"], 4),
    }

    out = {
        "phase": "phase2_main",
        "prereg_amendment_sha256": "4adf5fe04cdeee4f21152e2db7f8cbb12c0955d09925c2c5b0efcde0f6044504",
        "target_model": TARGET,
        "context_source": "claude_derived",
        "conditions": CONDITIONS,
        "judges": [SONNET_JUDGE, GPT5_JUDGE],
        "n_probes": len(ALL_PROBES),
        "elapsed_seconds": round(elapsed, 2),
        "estimated_cost_usd": round(cost_running[0], 4),
        "summary": summary,
    }
    PHASE2_RESULTS.write_text(json.dumps(out, indent=2))

    log("\n=== PHASE 2 SUMMARY ===")
    log(f"{'condition':18s}  {'sonnet':>10s}  {'gpt5':>10s}")
    for cond in CONDITIONS:
        s = summary[cond]
        log(f"{cond:18s}  {s['sonnet_mean']:10.4f}  {s['gpt5_mean']:10.4f}")
    d = summary["delta_recent3K_minus_scratch"]
    log(f"{'Δ recent3K-scratch':18s}  {d['sonnet']:+10.4f}  {d['gpt5']:+10.4f}")
    log(f"\nOutput: {PHASE2_RESULTS}")

    _log_handle.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
