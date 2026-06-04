"""Phase 1 smoke run: Gemini 2.5 Flash × {scratch, recent3K} × 25 probes.

Pre-registration: PREREG_AMENDMENT_GEMINI.md
SHA-256 (pre-signature):
  411b248ab959975341d788beb21143db53844315658da7eeae9923fbc9ca4ed3

Pipeline integrity check before committing to Phase 2's full panel-extension
spend. Exact same protocol as scripts/b2_content_position_crossmodel.py
(used for Opus 4.7 / GPT-5 in primary panel) — only target and condition
subset differ. Same 25 probes, same Sonnet 4.6 judge, same JUDGE_SYSTEM_PROMPT,
same PROBE_FRAMING, same `recent_3K = verbatim_full[-3000:]` slicing, same
acknowledgment-message convention.

Cost cap: kill if cumulative spend exceeds $5 (smoke is budgeted ~$2).

Outputs (all preserved for the public mirror):
  data_archive/gemini_panel/phase1_smoke/
    PHASE1_RESULTS.json    — per-probe responses + judge calls + summary
    PHASE1_LOG.txt         — chronological run log

Phase 1 success criteria (from PREREG_AMENDMENT_GEMINI.md §5.2):
  - All 50 probe responses produced (25 probes × 2 conditions)
  - All 50 judge calls produced valid 0/1/2/3 scores
  - Empty-output rate < 5%
  - Δ in plausible range [-1.5, +0.5]

Run:
  set -a && source ../.env && set +a && \\
    python scripts/phase1_gemini_smoke.py
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

# Reuse the existing primary-panel infrastructure verbatim — same loader,
# same slicer, same parser. Pre-reg amendment §2 explicitly forbids
# methodological changes; this guarantees byte-identical context handling.
from analyze_length_control import (
    load_events,
    extract_verbatim_slice,
    parse_judge,
)


PHASE1_DIR = REPO_ROOT / "data_archive" / "gemini_panel" / "phase1_smoke"
PHASE1_RESULTS = PHASE1_DIR / "PHASE1_RESULTS.json"
PHASE1_LOG = PHASE1_DIR / "PHASE1_LOG.txt"

TARGET_MODEL = "gemini-2.5-flash"
JUDGE_MODEL = "claude-sonnet-4-6"

# Phase 1 only runs scratch + recent3K (the load-bearing pair for the heatmap).
# recent3K_filler / recent3K_earlier / filler14K are added in Phase 2.
PHASE1_CONDITIONS = ["scratch", "recent3K"]

# Hard cost cap for Phase 1 — well above the ~$2 estimate, well below
# the $200 campaign-wide cap from PREREG_AMENDMENT_GEMINI.md §5.1.
PHASE1_COST_CAP_USD = 5.0

# Conservative pricing (USD per 1M tokens) for Phase 1 budget tracking only;
# real billing comes from the prepay account.
PRICING = {
    "gemini-2.5-flash": {"in": 0.30, "out": 2.50},
    "claude-sonnet-4-6": {"in": 3.00, "out": 15.00},
}


_log_handle = None


def log(msg: str) -> None:
    """Print to stdout and append to the run log."""
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
    """Build the same {scratch, recent3K} message prefixes as the primary
    panel uses. Identical slicing logic to b2_content_position_crossmodel.py."""
    events = load_events()
    boundaries = [
        i for i, e in enumerate(events)
        if e.get("type") == "system" and e.get("subtype") == "compact_boundary"
    ]
    if not boundaries:
        raise RuntimeError("No compaction boundary found in donated transcript.")
    target_boundary = boundaries[-1]

    verbatim_full = extract_verbatim_slice(events, target_boundary, 28000)
    recent_3K = verbatim_full[-3000:]

    ack = {
        "role": "assistant",
        "content": "Acknowledged. How can I help continue this work?",
    }

    return {
        "scratch": [],
        "recent3K": [{"role": "user", "content": recent_3K}, ack],
    }


def call_sonnet_judge(client, probe_text: str, response_text: str) -> dict:
    """Invoke the Sonnet 4.6 judge with the locked rubric prompt."""
    user_msg = (
        f"PROBE:\n{probe_text}\n\nRESPONSE:\n{response_text}\n\n"
        f"Return the JSON now."
    )
    last_err: Exception = RuntimeError("sonnet judge failed without raising")
    for attempt in range(3):
        try:
            resp = client.messages.create(
                model=JUDGE_MODEL,
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


def run_condition(
    label: str,
    prior: list[dict],
    target_client,
    judge_client,
    cost_running: list[float],
) -> dict:
    """Run all 25 probes against a single condition, return per-probe results."""
    log(f"--- condition: {label} ({len(prior)} prior message(s)) ---")
    results = []
    for i, probe in enumerate(ALL_PROBES):
        # Hard cost cap before each probe — protects against runaway spend.
        if cost_running[0] >= PHASE1_COST_CAP_USD:
            log(f"!! Cost cap ${PHASE1_COST_CAP_USD:.2f} reached at probe {i}; halting.")
            results.append({
                "probe_id": probe.id,
                "category": probe.category,
                "error": f"cost_cap_reached_at_${cost_running[0]:.2f}",
            })
            break

        framed = f"{PROBE_FRAMING}\n\n{probe.text}"
        msgs = prior + [{"role": "user", "content": framed}]

        # Target call (Gemini Flash)
        try:
            resp_text, in_tok, out_tok = call_gemini(
                target_client, TARGET_MODEL, msgs, system=None,
            )
            target_cost = estimate_cost(TARGET_MODEL, in_tok, out_tok)
            cost_running[0] += target_cost
        except Exception as e:
            log(f"  [{label} {i+1}/25] target ERROR: {type(e).__name__}: {e}")
            results.append({
                "probe_id": probe.id,
                "category": probe.category,
                "probe_text": probe.text,
                "error": f"target_call: {type(e).__name__}: {str(e)[:240]}",
            })
            continue

        # Judge call (Sonnet 4.6)
        try:
            judgment = call_sonnet_judge(judge_client, probe.text, resp_text)
            judge_cost = estimate_cost(
                JUDGE_MODEL,
                judgment.get("_judge_input_tokens", 0),
                judgment.get("_judge_output_tokens", 0),
            )
            cost_running[0] += judge_cost
        except Exception as e:
            log(f"  [{label} {i+1}/25] judge ERROR: {type(e).__name__}: {e}")
            results.append({
                "probe_id": probe.id,
                "category": probe.category,
                "probe_text": probe.text,
                "response_text": resp_text,
                "target_input_tokens": in_tok,
                "target_output_tokens": out_tok,
                "error": f"judge_call: {type(e).__name__}: {str(e)[:240]}",
            })
            continue

        score = judgment.get("score", -1)
        results.append({
            "probe_id": probe.id,
            "category": probe.category,
            "probe_text": probe.text,
            "response_text": resp_text,
            "response_preview": resp_text[:300],
            "target_input_tokens": in_tok,
            "target_output_tokens": out_tok,
            "judge_score": score,
            "judge_label": judgment.get("label", ""),
            "judge_reason": judgment.get("reason", "")[:240],
            "judge_input_tokens": judgment.get("_judge_input_tokens", 0),
            "judge_output_tokens": judgment.get("_judge_output_tokens", 0),
        })
        if (i + 1) % 5 == 0:
            valid = [r["judge_score"] for r in results if r.get("judge_score", -1) in (0, 1, 2, 3)]
            mean = sum(valid) / max(len(valid), 1)
            log(f"  [{label} {i+1}/25] running mean={mean:.2f}  "
                f"cumulative cost=${cost_running[0]:.3f}")

    valid = [r["judge_score"] for r in results if r.get("judge_score", -1) in (0, 1, 2, 3)]
    return {
        "label": label,
        "n_probes": len(ALL_PROBES),
        "n_valid": len(valid),
        "mean_score": sum(valid) / max(len(valid), 1),
        "results": results,
    }


def main() -> int:
    if not os.environ.get("GOOGLE_API_KEY"):
        sys.exit("Set GOOGLE_API_KEY (Gemini target) — source ../.env")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("Set ANTHROPIC_API_KEY (Sonnet judge) — source ../.env")

    PHASE1_DIR.mkdir(parents=True, exist_ok=True)
    global _log_handle
    _log_handle = open(PHASE1_LOG, "a")

    log(f"Phase 1 smoke: {TARGET_MODEL} × {PHASE1_CONDITIONS} × {len(ALL_PROBES)} probes")
    log(f"Pre-reg: PREREG_AMENDMENT_GEMINI.md "
        f"sha256=411b248ab959975341d788beb21143db53844315658da7eeae9923fbc9ca4ed3")
    log(f"Output: {PHASE1_RESULTS}")
    log(f"Cost cap: ${PHASE1_COST_CAP_USD:.2f}")

    from google import genai
    from anthropic import Anthropic

    target_client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    judge_client = Anthropic()

    log("Loading donated session transcript and slicing recent3K...")
    conditions = build_conditions()
    for label, msgs in conditions.items():
        chars = len(msgs[0]["content"]) if msgs else 0
        log(f"  {label}: {chars} chars of prior context")

    cost_running = [0.0]
    started = time.time()

    per_condition = {}
    for label in PHASE1_CONDITIONS:
        per_condition[label] = run_condition(
            label, conditions[label], target_client, judge_client, cost_running,
        )

    elapsed = time.time() - started

    # Assemble summary
    scratch_mean = per_condition["scratch"]["mean_score"]
    recent3k_mean = per_condition["recent3K"]["mean_score"]
    delta = recent3k_mean - scratch_mean
    n_total = sum(c["n_probes"] for c in per_condition.values())
    n_valid = sum(c["n_valid"] for c in per_condition.values())
    n_errors = n_total - n_valid

    out = {
        "phase": "phase1_smoke",
        "prereg_amendment_sha256":
            "411b248ab959975341d788beb21143db53844315658da7eeae9923fbc9ca4ed3",
        "target_model": TARGET_MODEL,
        "judge_model": JUDGE_MODEL,
        "conditions": PHASE1_CONDITIONS,
        "n_probes": len(ALL_PROBES),
        "elapsed_seconds": round(elapsed, 2),
        "estimated_cost_usd": round(cost_running[0], 4),
        "per_condition": per_condition,
        "summary": {
            "scratch_mean": round(scratch_mean, 4),
            "recent3K_mean": round(recent3k_mean, 4),
            "delta_recent3K_minus_scratch": round(delta, 4),
            "n_valid_total": n_valid,
            "n_errors_total": n_errors,
            "error_rate": round(n_errors / max(n_total, 1), 4),
        },
    }
    PHASE1_RESULTS.write_text(json.dumps(out, indent=2))

    log("")
    log("==================== PHASE 1 SUMMARY ====================")
    log(f"  scratch  mean: {scratch_mean:.4f}  (n_valid={per_condition['scratch']['n_valid']}/25)")
    log(f"  recent3K mean: {recent3k_mean:.4f}  (n_valid={per_condition['recent3K']['n_valid']}/25)")
    log(f"  Δ (recent3K - scratch): {delta:+.4f}")
    log(f"  Errors: {n_errors}/{n_total}  (rate={n_errors / max(n_total, 1):.2%})")
    log(f"  Wall clock: {elapsed:.1f}s")
    log(f"  Estimated cost: ${cost_running[0]:.3f}")
    log(f"  Output: {PHASE1_RESULTS}")
    log("==========================================================")

    # Phase 1 gate check (per pre-reg §5.2)
    error_rate = n_errors / max(n_total, 1)
    pass_gate = (
        error_rate < 0.05
        and -1.5 <= delta <= 0.5
        and n_valid > 0
    )
    if pass_gate:
        log("[GATE PASS] Pipeline healthy. Phase 2 may proceed.")
        _log_handle.close()
        return 0
    log("[GATE FAIL] Inspect errors / Δ before Phase 2.")
    _log_handle.close()
    return 1


if __name__ == "__main__":
    sys.exit(main())
