"""Phase 2 main: Gemini 2.5 Pro + Flash × 5 conditions × 25 probes × 2 judges.

Pre-registration: PREREG_AMENDMENT_GEMINI.md
SHA-256 (pre-signature):
  411b248ab959975341d788beb21143db53844315658da7eeae9923fbc9ca4ed3

Phase 2A: target responses scored by Sonnet 4.6 (primary).
Phase 2B: same target responses re-scored by GPT-5 (cross-judge audit).

Target calls happen ONCE per (target, condition, probe) and are cached so
that 2B re-uses the responses from 2A — no double target spend.

Protocol identical to scripts/b2_content_position_crossmodel.py for the
existing 12-target panel — same probes, same judge prompts, same conditions,
same recent_3K = verbatim_full[-3000:] slicing, same ack message.

Outputs (preserved for the public mirror):
  data_archive/gemini_panel/phase2_main/
    PHASE2_RESULTS.json               — aggregated summary
    PHASE2_LOG.txt                    — chronological run log
    target__<model>__<condition>.json — per-cell target responses (cached)
    sonnet__<model>__<condition>.json — per-cell Sonnet judge scores
    gpt5__<model>__<condition>.json   — per-cell GPT-5 judge scores

Hard cost cap: $150 (well above ~$85 estimate, under $200 campaign cap from
PREREG_AMENDMENT_GEMINI §5.1).

Run:
  set -a && source ../.env && set +a && \\
    python scripts/phase2_gemini_main.py
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
from harness.clients_gemini import call_gemini

from analyze_length_control import (
    load_events,
    extract_verbatim_slice,
    parse_judge,
)


# Same filler template as scripts/b2_content_position_crossmodel.py.
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


PHASE2_DIR = REPO_ROOT / "data_archive" / "gemini_panel" / "phase2_main"
PHASE2_RESULTS = PHASE2_DIR / "PHASE2_RESULTS.json"
PHASE2_LOG = PHASE2_DIR / "PHASE2_LOG.txt"

TARGETS = ["gemini-2.5-flash", "gemini-2.5-pro"]
SONNET_JUDGE = "claude-sonnet-4-6"
GPT5_JUDGE = "gpt-5"
CONDITIONS = ["scratch", "recent3K", "recent3K_filler", "recent3K_earlier", "filler14K"]

# Hard cost cap inside the script. Above ~$85 estimate, under $200 campaign cap.
PHASE2_COST_CAP_USD = 150.0

# Pricing for cost tracking (USD per 1M tokens). Real billing comes from
# the providers' prepay/postpay accounts.
PRICING = {
    "gemini-2.5-flash": {"in": 0.30, "out": 2.50},
    "gemini-2.5-pro": {"in": 1.25, "out": 10.00},
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
    """Build all 5 conditions. Identical to b2_content_position_crossmodel.py."""
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


def cell_path(prefix: str, model: str, condition: str) -> Path:
    safe_model = model.replace("/", "_").replace(".", "-")
    return PHASE2_DIR / f"{prefix}__{safe_model}__{condition}.json"


def run_target_cell(
    target_client,
    target_model: str,
    condition: str,
    prior: list[dict],
    cost_running: list[float],
) -> dict:
    """Run all 25 probes against (target, condition). Returns per-probe responses.

    Caches result to disk so re-runs / Phase 2B don't re-invoke the target.
    """
    cache_path = cell_path("target", target_model, condition)
    if cache_path.exists():
        log(f"  [target {target_model} / {condition}] using cached responses at {cache_path.name}")
        return json.loads(cache_path.read_text())

    log(f"--- target: {target_model} / {condition} ({len(prior)} prior msg(s)) ---")
    results = []
    for i, probe in enumerate(ALL_PROBES):
        if cost_running[0] >= PHASE2_COST_CAP_USD:
            log(f"!! Cost cap ${PHASE2_COST_CAP_USD:.2f} reached at probe {i}; halting.")
            results.append({
                "probe_id": probe.id,
                "category": probe.category,
                "error": f"cost_cap_reached_at_${cost_running[0]:.2f}",
            })
            break

        framed = f"{PROBE_FRAMING}\n\n{probe.text}"
        msgs = prior + [{"role": "user", "content": framed}]

        try:
            resp_text, in_tok, out_tok = call_gemini(
                target_client, target_model, msgs, system=None,
            )
            target_cost = estimate_cost(target_model, in_tok, out_tok)
            cost_running[0] += target_cost
        except Exception as e:
            log(f"  [{condition} {i+1}/25] target ERROR: {type(e).__name__}: {str(e)[:120]}")
            results.append({
                "probe_id": probe.id,
                "category": probe.category,
                "probe_text": probe.text,
                "error": f"target_call: {type(e).__name__}: {str(e)[:240]}",
            })
            continue

        results.append({
            "probe_id": probe.id,
            "category": probe.category,
            "probe_text": probe.text,
            "response_text": resp_text,
            "input_tokens": in_tok,
            "output_tokens": out_tok,
        })
        if (i + 1) % 5 == 0:
            log(f"  [{condition} {i+1}/25]  cumulative cost=${cost_running[0]:.3f}")

    cell = {
        "target_model": target_model,
        "condition": condition,
        "n_probes": len(ALL_PROBES),
        "n_responses": sum(1 for r in results if "response_text" in r),
        "results": results,
    }
    cache_path.write_text(json.dumps(cell, indent=2))
    log(f"  saved → {cache_path.name}")
    return cell


def run_judge_cell(
    judge_name: str,
    judge_call,
    target_model: str,
    condition: str,
    target_cell: dict,
    cost_running: list[float],
    judge_model_id: str,
) -> dict:
    """Score the target_cell's responses with the given judge. Cached on disk."""
    cache_path = cell_path(judge_name, target_model, condition)
    if cache_path.exists():
        log(f"  [{judge_name} {target_model} / {condition}] using cached scores at {cache_path.name}")
        return json.loads(cache_path.read_text())

    log(f"--- {judge_name}: {target_model} / {condition} ---")
    scored = []
    for i, r in enumerate(target_cell["results"]):
        if "error" in r:
            scored.append({
                "probe_id": r["probe_id"],
                "category": r["category"],
                "skipped": "target_error",
            })
            continue
        if cost_running[0] >= PHASE2_COST_CAP_USD:
            log(f"!! Cost cap ${PHASE2_COST_CAP_USD:.2f} reached during {judge_name}; halting.")
            scored.append({
                "probe_id": r["probe_id"],
                "category": r["category"],
                "skipped": "cost_cap",
            })
            continue
        try:
            judgment = judge_call(r["probe_text"], r["response_text"])
            cost_running[0] += estimate_cost(
                judge_model_id,
                judgment.get("_judge_input_tokens", 0),
                judgment.get("_judge_output_tokens", 0),
            )
        except Exception as e:
            log(f"  [{condition} {i+1}/25] {judge_name} ERROR: {type(e).__name__}: {str(e)[:120]}")
            scored.append({
                "probe_id": r["probe_id"],
                "category": r["category"],
                "error": f"judge_call: {type(e).__name__}: {str(e)[:240]}",
            })
            continue
        scored.append({
            "probe_id": r["probe_id"],
            "category": r["category"],
            "score": judgment.get("score", -1),
            "label": judgment.get("label", ""),
            "reason": judgment.get("reason", "")[:240],
            "judge_input_tokens": judgment.get("_judge_input_tokens", 0),
            "judge_output_tokens": judgment.get("_judge_output_tokens", 0),
        })
        if (i + 1) % 5 == 0:
            valid = [s["score"] for s in scored if s.get("score", -1) in (0, 1, 2, 3)]
            mean = sum(valid) / max(len(valid), 1)
            log(f"  [{condition} {i+1}/25] running mean={mean:.2f}  cumulative cost=${cost_running[0]:.3f}")

    valid = [s["score"] for s in scored if s.get("score", -1) in (0, 1, 2, 3)]
    cell = {
        "judge": judge_name,
        "judge_model": judge_model_id,
        "target_model": target_model,
        "condition": condition,
        "n_valid": len(valid),
        "mean_score": sum(valid) / max(len(valid), 1),
        "results": scored,
    }
    cache_path.write_text(json.dumps(cell, indent=2))
    log(f"  saved → {cache_path.name}  mean={cell['mean_score']:.4f}  n_valid={cell['n_valid']}/25")
    return cell


def main() -> int:
    if not os.environ.get("GOOGLE_API_KEY"):
        sys.exit("Set GOOGLE_API_KEY (Gemini target) — source ../.env")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("Set ANTHROPIC_API_KEY (Sonnet judge) — source ../.env")
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("Set OPENAI_API_KEY (GPT-5 cross-judge) — source ../.env")

    PHASE2_DIR.mkdir(parents=True, exist_ok=True)
    global _log_handle
    _log_handle = open(PHASE2_LOG, "a")

    log(f"Phase 2 main: targets={TARGETS}, conditions={CONDITIONS}")
    log(f"Pre-reg: PREREG_AMENDMENT_GEMINI.md "
        f"sha256=411b248ab959975341d788beb21143db53844315658da7eeae9923fbc9ca4ed3")
    log(f"Output dir: {PHASE2_DIR}")
    log(f"Cost cap: ${PHASE2_COST_CAP_USD:.2f}")

    from google import genai
    from anthropic import Anthropic
    from openai import OpenAI

    target_client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    sonnet_client = Anthropic()
    gpt5_client = OpenAI()

    log("Building conditions from donated transcript...")
    conditions = build_conditions()
    for label, msgs in conditions.items():
        chars = len(msgs[0]["content"]) if msgs else 0
        log(f"  {label}: {chars} chars")

    cost_running = [0.0]
    started = time.time()

    # =================================================================
    # Phase 2A: target calls + Sonnet judge
    # =================================================================
    log("\n=================== PHASE 2A: targets + Sonnet judge ===================")
    target_cells: dict[tuple[str, str], dict] = {}
    sonnet_cells: dict[tuple[str, str], dict] = {}

    for target in TARGETS:
        for condition in CONDITIONS:
            tcell = run_target_cell(
                target_client, target, condition, conditions[condition], cost_running,
            )
            target_cells[(target, condition)] = tcell

            scell = run_judge_cell(
                "sonnet",
                lambda p, r: call_sonnet_judge(sonnet_client, p, r),
                target, condition, tcell, cost_running, SONNET_JUDGE,
            )
            sonnet_cells[(target, condition)] = scell

    log(f"\nPhase 2A complete. cumulative cost=${cost_running[0]:.3f}")

    # =================================================================
    # Phase 2B: GPT-5 cross-judge re-scores cached target responses
    # =================================================================
    log("\n=================== PHASE 2B: GPT-5 cross-judge ===================")
    gpt5_cells: dict[tuple[str, str], dict] = {}
    for target in TARGETS:
        for condition in CONDITIONS:
            tcell = target_cells[(target, condition)]
            gcell = run_judge_cell(
                "gpt5",
                lambda p, r: call_gpt5_judge(gpt5_client, p, r),
                target, condition, tcell, cost_running, GPT5_JUDGE,
            )
            gpt5_cells[(target, condition)] = gcell

    elapsed = time.time() - started
    log(f"\nPhase 2 complete. wall clock={elapsed:.0f}s, cumulative cost=${cost_running[0]:.3f}")

    # =================================================================
    # Aggregate summary
    # =================================================================
    summary: dict[str, dict] = {}
    for target in TARGETS:
        per_target: dict[str, dict] = {}
        for condition in CONDITIONS:
            per_target[condition] = {
                "sonnet_mean": round(sonnet_cells[(target, condition)]["mean_score"], 4),
                "sonnet_n_valid": sonnet_cells[(target, condition)]["n_valid"],
                "gpt5_mean": round(gpt5_cells[(target, condition)]["mean_score"], 4),
                "gpt5_n_valid": gpt5_cells[(target, condition)]["n_valid"],
            }
        # Δ vs scratch under each judge
        if "scratch" in per_target and "recent3K" in per_target:
            per_target["delta_recent3K_minus_scratch"] = {
                "sonnet": round(
                    per_target["recent3K"]["sonnet_mean"]
                    - per_target["scratch"]["sonnet_mean"], 4),
                "gpt5": round(
                    per_target["recent3K"]["gpt5_mean"]
                    - per_target["scratch"]["gpt5_mean"], 4),
            }
        summary[target] = per_target

    out = {
        "phase": "phase2_main",
        "prereg_amendment_sha256":
            "411b248ab959975341d788beb21143db53844315658da7eeae9923fbc9ca4ed3",
        "targets": TARGETS,
        "conditions": CONDITIONS,
        "judges": [SONNET_JUDGE, GPT5_JUDGE],
        "n_probes": len(ALL_PROBES),
        "elapsed_seconds": round(elapsed, 2),
        "estimated_cost_usd": round(cost_running[0], 4),
        "summary": summary,
    }
    PHASE2_RESULTS.write_text(json.dumps(out, indent=2))

    # Print summary table
    log("\n==================== PHASE 2 SUMMARY ====================")
    log(f"{'target':18s}  {'condition':18s}  {'sonnet':>8s}  {'gpt5':>8s}")
    for target in TARGETS:
        for condition in CONDITIONS:
            row = summary[target][condition]
            log(f"{target:18s}  {condition:18s}  {row['sonnet_mean']:>8.4f}  {row['gpt5_mean']:>8.4f}")
        delta = summary[target].get("delta_recent3K_minus_scratch", {})
        log(f"{target:18s}  {'Δ recent3K-scratch':18s}  "
            f"{delta.get('sonnet', float('nan')):+8.4f}  "
            f"{delta.get('gpt5', float('nan')):+8.4f}")
    log(f"\nWall clock: {elapsed:.1f}s")
    log(f"Estimated cost: ${cost_running[0]:.3f}")
    log(f"Output: {PHASE2_RESULTS}")
    log("==========================================================")

    _log_handle.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
