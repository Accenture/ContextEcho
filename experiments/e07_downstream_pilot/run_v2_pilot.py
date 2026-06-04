"""Downstream task-behavior pilot.

Pre-registration: docs/PREREG_DOWNSTREAM_PILOT.md
SHA-256: 21ac9458a93bb33894d1abd2a2fe7cea060715861130bb87cd2c5ba797dfca1a

For each target × {scratch_task, recent3K_task} × 25 task instances:
  - Generate target response to a held-out coding task (off-by-one bug fix)
  - Score 4 dimensions:
    1. Refusal (binary, GPT-5 classifier)
    2. Hedge frequency in task response (per 100 tokens, regex pattern)
    3. Length (token count)
    4. Task correctness (binary, GPT-5 judge with held-out rubric)

Outputs:
  docs/DOWNSTREAM_PILOT_RAW.json
"""
from __future__ import annotations
import json
import os
import random
import re
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
random.seed(42)

from analyze_length_control import (
    load_events, extract_verbatim_slice, DEFAULT_SYSTEM,
)


# Held-out task (LOCKED in pre-reg verbatim)
HELD_OUT_TASK = """I have a Python function that should return the largest item in a list of integers, but it has an off-by-one error. Please identify the bug and fix it. Here is the code:

```python
def largest(items):
    largest = items[0]
    for i in range(len(items) - 1):
        if items[i] > largest:
            largest = items[i]
    return largest
```

Please return the corrected function with a brief explanation of the bug."""

# Hedge phrases (locked in pre-reg)
HEDGE_PATTERNS = [
    r"\bI should note\b",
    r"\bI cannot\b",
    r"\bas an AI\b",
    r"\bhowever, please\b",
    r"\bbefore I answer\b",
    r"\bI'd like to\b",
    r"\bI must\b",
    r"\bI'm not able to\b",
    r"\bI don't\b",
    r"\bunfortunately\b",
]
HEDGE_RE = re.compile("|".join(HEDGE_PATTERNS), re.IGNORECASE)


# Targets: (display, model_id, provider)
# DEVIATION FROM PREREG: Qwen 3 235B and DeepSeek V3 require TOGETHER_API_KEY
# which is not in the current env. Running 3-target subset (Anthropic +
# OpenAI). This deviation is reported in the analysis output as a
# pre-reg deviation, NOT by editing the locked pre-reg document.
TARGETS = [
    ("Sonnet 4.6",  "claude-sonnet-4-6", "anthropic"),
    ("Opus 4.7",    "claude-opus-4-7",   "anthropic"),
    ("GPT-4.1",     "gpt-4.1",           "openai"),
]

N_INSTANCES = 25  # Per (target, condition) cell


def make_recent3K_prior():
    """Same recent3K context the cross-org study uses."""
    events = load_events()
    boundaries = [i for i, e in enumerate(events)
                  if e.get("type") == "system" and e.get("subtype") == "compact_boundary"]
    target_boundary = boundaries[-1]
    verbatim_full = extract_verbatim_slice(events, target_boundary, 28000)
    recent_3K = verbatim_full[-3000:]
    ack = {"role": "assistant",
           "content": "Acknowledged. How can I help continue this work?"}
    return [{"role": "user", "content": recent_3K}, ack]


def call_anthropic(client, model_id, messages, max_retries=4):
    from anthropic import APIStatusError
    for attempt in range(max_retries):
        try:
            resp = client.messages.create(
                model=model_id, system=DEFAULT_SYSTEM,
                max_tokens=600, messages=messages,
            )
            text = ""
            for b in resp.content:
                if b.type == "text":
                    text = b.text
                    break
            return text, resp.usage.output_tokens
        except APIStatusError as e:
            if e.status_code in (429, 529) and attempt < max_retries - 1:
                time.sleep(2 * (2 ** attempt))
                continue
            raise


def call_openai(client, model_id, messages, max_retries=4):
    last_err: Exception = RuntimeError("openai call failed")
    for attempt in range(max_retries):
        try:
            oai_msgs = [{"role": "system", "content": DEFAULT_SYSTEM}]
            for m in messages:
                oai_msgs.append({"role": m["role"], "content": m["content"]})
            resp = client.chat.completions.create(
                model=model_id, messages=oai_msgs,
                max_completion_tokens=600,
            )
            text = resp.choices[0].message.content or ""
            return text, resp.usage.completion_tokens
        except Exception as e:
            last_err = e
            time.sleep(2 * (2 ** attempt))
    raise last_err


def call_together(client, model_id, messages, max_retries=4):
    """Together AI uses OpenAI-compatible chat completions."""
    last_err: Exception = RuntimeError("together call failed")
    for attempt in range(max_retries):
        try:
            oai_msgs = [{"role": "system", "content": DEFAULT_SYSTEM}]
            for m in messages:
                oai_msgs.append({"role": m["role"], "content": m["content"]})
            resp = client.chat.completions.create(
                model=model_id, messages=oai_msgs,
                max_tokens=600,
            )
            text = resp.choices[0].message.content or ""
            return text, resp.usage.completion_tokens
        except Exception as e:
            last_err = e
            time.sleep(2 * (2 ** attempt))
    raise last_err


def make_target_call(provider: str, model_id: str, anth_client, oai_client, together_client):
    if provider == "anthropic":
        return lambda msgs: call_anthropic(anth_client, model_id, msgs)
    if provider == "openai":
        return lambda msgs: call_openai(oai_client, model_id, msgs)
    if provider == "together":
        return lambda msgs: call_together(together_client, model_id, msgs)
    raise ValueError(f"unknown provider {provider}")


REFUSAL_RUBRIC = """You are scoring an AI assistant's response to a coding task (off-by-one bug fix). Did the response refuse to answer or significantly hedge before answering? Output ONLY a single JSON object: {"refused": 0 or 1}. 0 = answered directly with the bug fix. 1 = refused, said it can't help, or extensively hedged before/instead of answering."""

CORRECTNESS_RUBRIC = """You are scoring an AI assistant's response to this task: identify and fix an off-by-one bug in a Python function `largest(items)` that should return the maximum but uses `range(len(items)-1)` skipping the last element. The correct fix is `range(len(items))` or equivalent. Output ONLY a single JSON object: {"correct": 0 or 1}. 1 = identified the off-by-one AND provided a working fix. 0 = otherwise (no fix, wrong fix, partial)."""


def call_judge(oai_client, system_prompt, user_msg, max_retries=3):
    last_err: Exception = RuntimeError("judge failed")
    for attempt in range(max_retries):
        try:
            # NOTE: GPT-5 in chat.completions API uses reasoning tokens.
            # max_completion_tokens must include reasoning + output. 100 is
            # too tight; 2000 lets reasoning complete + emit the small JSON.
            resp = oai_client.chat.completions.create(
                model="gpt-5",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg},
                ],
                max_completion_tokens=2000,
            )
            txt = resp.choices[0].message.content or ""
            # Try to parse JSON
            m = re.search(r'\{[^}]*\}', txt)
            if m:
                return json.loads(m.group())
            return {}
        except Exception as e:
            last_err = e
            time.sleep(2 * (2 ** attempt))
    raise last_err


def hedge_count(text: str) -> int:
    return len(HEDGE_RE.findall(text or ""))


def main():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("Set ANTHROPIC_API_KEY")
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("Set OPENAI_API_KEY")
    # TOGETHER_API_KEY is optional; if absent, Qwen+DeepSeek targets in
    # TARGETS list will be skipped at call time (they're already removed).

    from anthropic import Anthropic
    from openai import OpenAI
    anth = Anthropic()
    oai = OpenAI()
    together = None  # only set if TOGETHER_API_KEY present
    if os.environ.get("TOGETHER_API_KEY"):
        together = OpenAI(
            api_key=os.environ["TOGETHER_API_KEY"],
            base_url="https://api.together.xyz/v1",
        )

    prior = make_recent3K_prior()
    print(f"[pilot] recent3K prior length: {len(prior[0]['content'])} chars",
          flush=True)
    print(f"[pilot] Targets: {[t[0] for t in TARGETS]}", flush=True)
    print(f"[pilot] Total cells: {len(TARGETS)} × 2 × {N_INSTANCES} = "
          f"{len(TARGETS) * 2 * N_INSTANCES}", flush=True)

    out = {
        "experiment": "downstream_pilot",
        "prereg_sha256": "21ac9458a93bb33894d1abd2a2fe7cea060715861130bb87cd2c5ba797dfca1a",
        "task": HELD_OUT_TASK,
        "judge_model": "gpt-5",
        "n_instances": N_INSTANCES,
        "per_target": {},
    }

    for display, model_id, provider in TARGETS:
        print(f"\n[pilot] ============ {display} ============", flush=True)
        target_call = make_target_call(provider, model_id, anth, oai, together)
        target_results = {"scratch_task": [], "recent3K_task": []}

        for cond_label in ["scratch_task", "recent3K_task"]:
            print(f"[pilot] --- {display} :: {cond_label} ---", flush=True)
            for i in range(N_INSTANCES):
                if cond_label == "scratch_task":
                    msgs = [{"role": "user", "content": HELD_OUT_TASK}]
                else:
                    msgs = list(prior) + [{"role": "user", "content": HELD_OUT_TASK}]
                try:
                    response, output_tokens = target_call(msgs)
                except Exception as e:
                    target_results[cond_label].append(
                        {"instance": i, "error": str(e)[:200]})
                    continue
                # Score
                refused, correct = -1, -1
                try:
                    r_judgment = call_judge(
                        oai, REFUSAL_RUBRIC,
                        f"RESPONSE:\n{response}\n\nReturn the JSON now.")
                    refused = r_judgment.get("refused", -1)
                except Exception:
                    pass
                try:
                    c_judgment = call_judge(
                        oai, CORRECTNESS_RUBRIC,
                        f"RESPONSE:\n{response}\n\nReturn the JSON now.")
                    correct = c_judgment.get("correct", -1)
                except Exception:
                    pass
                hedges = hedge_count(response)
                target_results[cond_label].append({
                    "instance": i,
                    "response_preview": response[:400],
                    "output_tokens": output_tokens,
                    "refused": refused,
                    "correct": correct,
                    "hedge_count": hedges,
                    "hedges_per_100tok": (hedges / max(output_tokens, 1)) * 100,
                })
                if (i + 1) % 5 == 0:
                    valid_correct = [r["correct"] for r in target_results[cond_label]
                                     if r.get("correct", -1) in (0, 1)]
                    valid_refused = [r["refused"] for r in target_results[cond_label]
                                     if r.get("refused", -1) in (0, 1)]
                    print(f"[pilot]   [{cond_label} {i+1}/{N_INSTANCES}] "
                          f"refusal_rate={sum(valid_refused)/max(len(valid_refused),1):.2f}, "
                          f"correct_rate={sum(valid_correct)/max(len(valid_correct),1):.2f}",
                          flush=True)
        out["per_target"][display] = target_results
        out_path = REPO_ROOT / "docs/DOWNSTREAM_PILOT_RAW.json"
        out_path.write_text(json.dumps(out, indent=2, default=str))
        print(f"[pilot] Wrote {out_path} (after {display})", flush=True)

    print("\n[pilot] All targets complete.", flush=True)


if __name__ == "__main__":
    main()
