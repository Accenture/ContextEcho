"""Cross-organization extension v2: 5 additional non-Anthropic targets on Session 1.

Mirrors scripts/run_cross_compaction_xtarget.py (Mistral + Kimi) but uses each
provider's *direct* API endpoint where the user has already paid for quota:

  - GPT-5            via OpenAI direct        (OPENAI_API_KEY)
  - GPT-4.1          via OpenAI direct        (OPENAI_API_KEY)
  - Gemini 2.5 Pro   via Google direct        (GOOGLE_API_KEY) — uses harness/clients_gemini.py
  - DeepSeek V3      via Together direct      (TOGETHER_AI_KEY)
  - Qwen3 235B       via Together direct      (TOGETHER_AI_KEY)
  - Llama 3.3 70B    via Together direct      (TOGETHER_AI_KEY)

Same probe shape: 12 positions × 10 paraphrases × 2 arms = 240 cells per target.
Total: 6 × 240 = 1440 cells. Cost projection: ~$25-50 (Gemini 2.5 Pro and GPT-5
are the expensive ones). Wall: ~60-90 min.

Output:
  data_archive/cross_compaction/{target_safe}/{position}/v{i}/{arm}.json
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
    POSITIONS, S2_VARIANTS, TRANSCRIPT_PATH,
    find_turn_to_line_index, extract_prefix_at_turn, load_transcript,
)
from scripts.run_downstream_clean_control import make_filler  # type: ignore
from scripts.run_instruction_override_probe import SYSTEM_PROMPT  # type: ignore

OUT_BASE = REPO_ROOT / "data_archive" / "cross_compaction"
ACK_MESSAGE = "Acknowledged. How can I help continue this work?"

# (provider, model_id, target_safe). target_safe = on-disk folder name.
TARGETS = [
    ("openai",   "gpt-5",                                              "gpt-5"),
    ("openai",   "gpt-4.1",                                            "gpt-4-1"),
    ("gemini",   "gemini-2.5-pro",                                     "gemini-2-5-pro"),
    ("together", "deepseek-ai/DeepSeek-V3",                            "deepseek-v3"),
    ("together", "Qwen/Qwen3-235B-A22B-Instruct-2507-tput",            "qwen3-235b"),
    ("together", "meta-llama/Llama-3.3-70B-Instruct-Turbo",            "llama-3-3-70b"),
]


def make_client(provider: str):
    if provider == "openai":
        from openai import OpenAI
        return OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    if provider == "together":
        from openai import OpenAI
        return OpenAI(api_key=os.environ["TOGETHER_AI_KEY"],
                      base_url="https://api.together.xyz/v1")
    if provider == "gemini":
        from google import genai  # type: ignore
        return genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    raise ValueError(f"Unknown provider: {provider}")


def call_oai_compat(client, model_id, messages, max_tokens=1024):
    """OpenAI-compatible call (works for OpenAI direct + Together)."""
    t0 = time.perf_counter()
    kwargs = dict(
        model=model_id, messages=messages, max_tokens=max_tokens,
        temperature=0.0, timeout=180,
    )
    # GPT-5 family: uses internal reasoning tokens that count against the
    # output budget. With long-context prefix, 1024 is fully consumed by
    # invisible reasoning, returning 0 visible. Bump to 8192 for headroom.
    if model_id.startswith("gpt-5"):
        kwargs.pop("temperature", None)
        kwargs["max_completion_tokens"] = 8192
        kwargs.pop("max_tokens", None)
    resp = client.chat.completions.create(**kwargs)
    elapsed = time.perf_counter() - t0
    msg = resp.choices[0].message
    text = msg.content or ""
    if isinstance(text, list):
        parts = []
        for block in text:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        text = "\n".join(parts)
    text = (text or "").strip()
    usage = {
        "input_tokens":  getattr(resp.usage, "prompt_tokens", None),
        "output_tokens": getattr(resp.usage, "completion_tokens", None),
    }
    return text, usage, elapsed


def call_gemini(client, model_id, messages, max_tokens=1024):
    """Gemini direct via harness/clients_gemini.py.

    The Gemini wrapper expects (system, messages) split; rebuild from oai-style.
    """
    from harness.clients_gemini import call_gemini as _call_gemini  # type: ignore
    sys_msg = next((m["content"] for m in messages if m["role"] == "system"), None)
    convo = [m for m in messages if m["role"] != "system"]
    t0 = time.perf_counter()
    # Gemini's call_gemini uses max_output_tokens with a hard floor of 4096
    text, in_tok, out_tok = _call_gemini(
        client, model_id, convo, system=sys_msg,
    )
    elapsed = time.perf_counter() - t0
    return text, {"input_tokens": in_tok, "output_tokens": out_tok}, elapsed


def build_messages(prefix: str, stressor: str) -> list[dict]:
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    if prefix:
        msgs.append({"role": "user", "content": prefix})
        msgs.append({"role": "assistant", "content": ACK_MESSAGE})
    msgs.append({"role": "user", "content": stressor})
    return msgs


def run_one(provider, client, model_id, prefix, stressor, out_path: Path) -> dict:
    if out_path.exists():
        try:
            return json.loads(out_path.read_text())
        except Exception:
            pass
    out_path.parent.mkdir(parents=True, exist_ok=True)
    msgs = build_messages(prefix, stressor)
    if provider == "gemini":
        text, usage, elapsed = call_gemini(client, model_id, msgs)
    else:
        text, usage, elapsed = call_oai_compat(client, model_id, msgs)
    metrics = {
        "response_text": text,
        "response_len": len(text),
        "wall_clock_sec": elapsed,
        "input_tokens":  usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "provider": provider,
    }
    out_path.write_text(json.dumps(metrics, indent=2, default=str))
    return metrics


def main() -> int:
    if not TRANSCRIPT_PATH.exists():
        sys.exit(f"Transcript missing: {TRANSCRIPT_PATH}")
    needed = {
        "openai":   "OPENAI_API_KEY",
        "together": "TOGETHER_AI_KEY",
        "gemini":   "GOOGLE_API_KEY",
    }
    for prov in {t[0] for t in TARGETS}:
        if not os.environ.get(needed[prov]):
            sys.exit(f"Set {needed[prov]} for provider {prov}")

    print("Loading transcript & turn index...")
    rows = load_transcript()
    turn_to_line, total = find_turn_to_line_index(TRANSCRIPT_PATH)
    print(f"  {total} real turns indexed")

    prefixes = {}
    for turn, label in POSITIONS:
        p = extract_prefix_at_turn(rows, turn_to_line, turn, max_chars=30000)
        prefixes[label] = p
        print(f"  {label} (turn {turn}): prefix len = {len(p)} chars")

    OUT_BASE.mkdir(parents=True, exist_ok=True)

    started = time.time()
    n_total = len(POSITIONS) * len(TARGETS) * len(S2_VARIANTS) * 2
    n_done = 0

    for provider, model_id, target_safe in TARGETS:
        try:
            client = make_client(provider)
        except Exception as e:
            print(f"  [SKIP {target_safe}]: client init failed: {e}")
            continue
        target_dir = OUT_BASE / target_safe
        target_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n{'='*60}\nTarget: {target_safe}  ({provider} :: {model_id})\n{'='*60}")

        for turn, pos_label in POSITIONS:
            prefix = prefixes[pos_label]
            filler = make_filler(len(prefix)) if prefix else ""

            for v_idx, variant in enumerate(S2_VARIANTS):
                claude_path = target_dir / pos_label / f"v{v_idx:02d}" / "claude.json"
                filler_path = target_dir / pos_label / f"v{v_idx:02d}" / "filler.json"

                for arm_path, arm_input, arm_label in [
                    (claude_path, prefix, "claude"),
                    (filler_path, filler, "filler"),
                ]:
                    if not arm_path.exists():
                        try:
                            m = run_one(provider, client, model_id, arm_input, variant, arm_path)
                            n_done += 1
                            if v_idx == 0:
                                print(f"  [{target_safe} {pos_label} v{v_idx} {arm_label}] "
                                      f"len={m['response_len']} in_tok={m['input_tokens']} "
                                      f"resp={m['response_text'][:50]!r}")
                        except Exception as e:
                            print(f"  [ERROR {target_safe} {pos_label} v{v_idx} {arm_label}]: {e}")
                    else:
                        n_done += 1

        elapsed = time.time() - started
        print(f"  {target_safe} done ({n_done}/{n_total} total) cum_elapsed={elapsed:.0f}s")

    elapsed = time.time() - started
    print(f"\n=== ALL DONE: {n_done}/{n_total} cells, {elapsed:.0f}s wall ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
