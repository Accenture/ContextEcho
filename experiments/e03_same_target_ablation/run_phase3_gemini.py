"""Phase 3: Gemini 2.5 Flash same-target context-source ablation.

Pre-registration: PREREG_AMENDMENT_GEMINI.md (sha256 411b248ab959...)
Phase 3 deviation logged in §6 of the amendment: target swapped from
Pro → Flash because Phase 2 showed Pro Δ(recent3K − scratch) ≈ 0 on
Claude-derived c_pre, leaving no drift to family-attribute. Flash drifts
cleanly (Δ = -0.32 Sonnet judge, attention-dilution signature).

This script tests the family-specificity hypothesis on Flash:
  - Flash × Claude-derived c_pre: already known from Phase 2 (Δ = -0.32 Sonnet, -0.07 GPT-5)
  - Flash × GPT-5-derived c_pre: new — reuses the same GPT-5 session
    (data/openai_gpt-5_debug_and_fix_baseline_seed301) used by
    a1_context_source_ablation.py for Sonnet 4.6 / Opus 4.7 / GPT-4.1
    same-target ablations. Byte-identical context primitive.

Same protocol: 5 conditions × 25 probes × 2 judges (Sonnet 4.6 + GPT-5).

Outputs:
  data_archive/gemini_panel/phase3_ablation/
    PHASE3_RESULTS.json
    PHASE3_LOG.txt
    target__gemini-2-5-flash__<condition>.json
    sonnet__gemini-2-5-flash__<condition>.json
    gpt5__gemini-2-5-flash__<condition>.json

Hard cost cap: $50 (Phase 3 estimate ~$10 with Flash; cap leaves margin).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from harness.probes import ALL_PROBES, PROBE_FRAMING
from harness.judge import JUDGE_SYSTEM_PROMPT
from harness.clients_gemini import call_gemini

from analyze_length_control import parse_judge

# Reuse the exact GPT-5 c_pre extractor used by the original same-target
# ablation (a1_context_source_ablation.py). Pre-reg amendment §2 forbids
# methodological changes — this guarantees byte-identical context.
from a1_context_source_ablation import (
    extract_gpt5_verbatim,
    make_filler,
)


# Same GPT-5 source session used by a1_context_source_ablation.py
# (line 199 of that script). Byte-identical transcript path keeps Phase 3
# directly comparable to the existing same-target ablation panel.
GPT5_TRANSCRIPT = (
    REPO_ROOT
    / "data"
    / "openai_gpt-5_debug_and_fix_baseline_seed301_0952d536c9c9"
    / "transcript.jsonl"
)

PHASE3_DIR = REPO_ROOT / "data_archive" / "gemini_panel" / "phase3_ablation"
PHASE3_RESULTS = PHASE3_DIR / "PHASE3_RESULTS.json"
PHASE3_LOG = PHASE3_DIR / "PHASE3_LOG.txt"

TARGET = "gemini-2.5-flash"
SONNET_JUDGE = "claude-sonnet-4-6"
GPT5_JUDGE = "gpt-5"
CONDITIONS = ["scratch", "recent3K", "recent3K_filler", "recent3K_earlier", "filler14K"]

PHASE3_COST_CAP_USD = 50.0

PRICING = {
    "gemini-2.5-flash": {"in": 0.30, "out": 2.50},
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


def build_gpt5_conditions() -> dict[str, list[dict]]:
    """Build 5 conditions from the GPT-5-derived c_pre, byte-identical to
    a1_context_source_ablation.py."""
    if not GPT5_TRANSCRIPT.exists():
        raise FileNotFoundError(
            f"Expected GPT-5 session at {GPT5_TRANSCRIPT}; same source as "
            f"scripts/a1_context_source_ablation.py."
        )
    verbatim_full = extract_gpt5_verbatim(GPT5_TRANSCRIPT, target_chars=28000)
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
    return PHASE3_DIR / f"{prefix}__{TARGET.replace('.', '-')}__{condition}.json"


def run_target_cell(target_client, condition: str, prior, cost_running) -> dict:
    cache = cell_path("target", condition)
    if cache.exists():
        log(f"  [target {condition}] cached")
        return json.loads(cache.read_text())

    log(f"--- target: {TARGET} / {condition} ({len(prior)} prior msgs) ---")
    results = []
    for i, probe in enumerate(ALL_PROBES):
        if cost_running[0] >= PHASE3_COST_CAP_USD:
            log(f"!! cost cap reached at probe {i}")
            results.append({"probe_id": probe.id, "category": probe.category, "error": "cost_cap"})
            break
        framed = f"{PROBE_FRAMING}\n\n{probe.text}"
        msgs = prior + [{"role": "user", "content": framed}]
        try:
            resp_text, in_tok, out_tok = call_gemini(target_client, TARGET, msgs)
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
        "target_model": TARGET, "context_source": "gpt5_derived",
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
        if cost_running[0] >= PHASE3_COST_CAP_USD:
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
        "target_model": TARGET, "context_source": "gpt5_derived",
        "condition": condition, "n_valid": len(valid),
        "mean_score": sum(valid) / max(len(valid), 1),
        "results": scored,
    }
    cache.write_text(json.dumps(cell, indent=2))
    log(f"  saved → {cache.name}  mean={cell['mean_score']:.4f}  n_valid={cell['n_valid']}/25")
    return cell


def main() -> int:
    if not os.environ.get("GOOGLE_API_KEY"):
        sys.exit("Set GOOGLE_API_KEY")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("Set ANTHROPIC_API_KEY")
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("Set OPENAI_API_KEY")

    PHASE3_DIR.mkdir(parents=True, exist_ok=True)
    global _log_handle
    _log_handle = open(PHASE3_LOG, "a")

    log(f"Phase 3 same-target ablation: {TARGET} × GPT-5-derived c_pre")
    log(f"Pre-reg: PREREG_AMENDMENT_GEMINI.md sha256=411b248ab959975341d788beb21143db53844315658da7eeae9923fbc9ca4ed3")
    log(f"Pre-reg deviation logged 2026-04-28: target swapped Pro→Flash (see §6)")
    log(f"Output: {PHASE3_DIR}")
    log(f"Cost cap: ${PHASE3_COST_CAP_USD:.2f}")

    from google import genai
    from anthropic import Anthropic
    from openai import OpenAI

    target_client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    sonnet_client = Anthropic()
    gpt5_client = OpenAI()

    log("Building GPT-5-derived c_pre conditions...")
    conditions = build_gpt5_conditions()
    for label, msgs in conditions.items():
        chars = len(msgs[0]["content"]) if msgs else 0
        log(f"  {label}: {chars} chars")

    cost_running = [0.0]
    started = time.time()

    target_cells: dict[str, dict] = {}
    sonnet_cells: dict[str, dict] = {}
    gpt5_cells: dict[str, dict] = {}

    for cond in CONDITIONS:
        target_cells[cond] = run_target_cell(target_client, cond, conditions[cond], cost_running)
        sonnet_cells[cond] = run_judge_cell(
            "sonnet", lambda p, r: call_sonnet_judge(sonnet_client, p, r),
            cond, target_cells[cond], cost_running, SONNET_JUDGE,
        )

    log("\n--- Phase 3B: GPT-5 cross-judge ---")
    for cond in CONDITIONS:
        gpt5_cells[cond] = run_judge_cell(
            "gpt5", lambda p, r: call_gpt5_judge(gpt5_client, p, r),
            cond, target_cells[cond], cost_running, GPT5_JUDGE,
        )

    elapsed = time.time() - started

    summary = {}
    for cond in CONDITIONS:
        summary[cond] = {
            "sonnet_mean": round(sonnet_cells[cond]["mean_score"], 4),
            "sonnet_n_valid": sonnet_cells[cond]["n_valid"],
            "gpt5_mean": round(gpt5_cells[cond]["mean_score"], 4),
            "gpt5_n_valid": gpt5_cells[cond]["n_valid"],
        }
    delta_gpt5_derived = {
        "sonnet": round(summary["recent3K"]["sonnet_mean"] - summary["scratch"]["sonnet_mean"], 4),
        "gpt5": round(summary["recent3K"]["gpt5_mean"] - summary["scratch"]["gpt5_mean"], 4),
    }

    out = {
        "phase": "phase3_ablation",
        "prereg_amendment_sha256": "411b248ab959975341d788beb21143db53844315658da7eeae9923fbc9ca4ed3",
        "deviation": "target swapped Pro→Flash 2026-04-28; see §6 of amendment",
        "target_model": TARGET,
        "context_source": "gpt5_derived",
        "judges": [SONNET_JUDGE, GPT5_JUDGE],
        "n_probes": len(ALL_PROBES),
        "elapsed_seconds": round(elapsed, 2),
        "estimated_cost_usd": round(cost_running[0], 4),
        "summary_gpt5_derived": summary,
        "delta_gpt5_derived_recent3K_minus_scratch": delta_gpt5_derived,
        "delta_claude_derived_for_comparison_from_phase2": {
            "sonnet": -0.32, "gpt5": -0.0653,
        },
        "family_specificity_check": {
            "claude_minus_gpt5_sonnet":
                round(-0.32 - delta_gpt5_derived["sonnet"], 4),
            "claude_minus_gpt5_gpt5":
                round(-0.0653 - delta_gpt5_derived["gpt5"], 4),
        },
    }
    PHASE3_RESULTS.write_text(json.dumps(out, indent=2))

    log("\n==================== PHASE 3 SUMMARY ====================")
    log(f"  Flash × GPT-5-derived c_pre:")
    for cond in CONDITIONS:
        log(f"    {cond:18s} sonnet={summary[cond]['sonnet_mean']:.4f}  gpt5={summary[cond]['gpt5_mean']:.4f}")
    log(f"  Δ recent3K-scratch on GPT-5-derived:    sonnet={delta_gpt5_derived['sonnet']:+.4f}  gpt5={delta_gpt5_derived['gpt5']:+.4f}")
    log(f"  Δ recent3K-scratch on Claude-derived: sonnet=-0.3200  gpt5=-0.0653 (from Phase 2)")
    log(f"  Family-specificity gap (Claude-derived Δ minus GPT-5-derived Δ):")
    log(f"    Sonnet judge: {out['family_specificity_check']['claude_minus_gpt5_sonnet']:+.4f}")
    log(f"    GPT-5 judge:  {out['family_specificity_check']['claude_minus_gpt5_gpt5']:+.4f}")
    log(f"  Wall clock: {elapsed:.1f}s  Cost: ${cost_running[0]:.3f}")
    log(f"  Output: {PHASE3_RESULTS}")
    log("==========================================================")

    _log_handle.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
