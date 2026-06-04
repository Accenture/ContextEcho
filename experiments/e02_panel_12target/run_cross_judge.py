"""Cross-judge replication on the 12-model panel.

Pre-registration: docs/PREREG_CROSSJUDGE_12MODEL.md
SHA-256: 22323d216e3d395cf518204c59da88312e22a2d2c7e7586c24cfce8bb8077bce

Reuses target-response data from existing CONTENT_POSITION_*.json /
OPTION_C_*.json files where full response text is available, and re-
scores with GPT-5 judge.

Where only response_preview (300 chars) is saved, we re-judge on the
preview and disclose the truncation. The original Sonnet judge scored on
full responses; the GPT-5 judge scores on the truncated preview. This is
a known limitation of the post-hoc cross-judge comparison and is
documented in the analysis output.

For runs where we have full responses (Modal-saved Path A files include
full text in 'response_preview' up to 300 chars), we use those. For
others, this script falls back to the preview.

Outputs:
  docs/CROSS_JUDGE_12MODEL_RAW.json
"""
from __future__ import annotations
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from harness.judge import JUDGE_SYSTEM_PROMPT
from analyze_length_control import parse_judge


# (display, content-position file, target_model_id_in_provider_form)
TARGETS = [
    ("Sonnet 4.5",    "OPTION_C_SONNET45.json",         "anthropic"),
    ("Sonnet 4.6",    "CONTENT_POSITION_SONNET.json",   "anthropic"),
    ("Haiku 4.5",     "OPTION_C_HAIKU45.json",          "anthropic"),
    ("Opus 4.6",      "OPTION_C_OPUS46.json",           "anthropic"),
    ("Opus 4.7",      "CONTENT_POSITION_OPUS.json",     "anthropic"),
    ("GPT-4o",        "CONTENT_POSITION_GPT4O.json",    "openai"),
    ("GPT-4o-mini",   "CONTENT_POSITION_GPT4O_MINI.json", "openai"),
    ("GPT-4.1",       "CONTENT_POSITION_GPT41.json",    "openai"),
    ("Llama 3.3 70B", "CONTENT_POSITION_LLAMA33_70B.json", "together"),
    ("Qwen 3 235B",   "CONTENT_POSITION_QWEN3_235B.json", "together"),
    ("DeepSeek V3",   "CONTENT_POSITION_DEEPSEEK_V3.json", "together"),
]

CONDITIONS = ["scratch", "recent3K", "recent3K_filler", "recent3K_earlier", "filler14K"]


def call_gpt5_judge(client, probe, response, max_retries=3):
    """Re-implement judge call via GPT-5 (OpenAI API) instead of Sonnet."""
    user_msg = f"PROBE:\n{probe}\n\nRESPONSE:\n{response}\n\nReturn the JSON now."
    last_err: Exception = RuntimeError("gpt-5 judge failed without raising")
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model="gpt-5",
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                max_completion_tokens=400,
            )
            text = resp.choices[0].message.content or ""
            return parse_judge(text)
        except Exception as e:
            last_err = e
            time.sleep(2 * (2 ** attempt))
    raise last_err


def extract_responses(d: dict, condition: str) -> list[dict]:
    """Return list of {probe_id, probe_text, response_text, original_score}
    for the given condition. Tries multiple schemas."""
    out = []
    for key in ("full_results", "per_condition"):
        if key in d:
            cond_data = d[key].get(condition)
            if cond_data and "results" in cond_data:
                for r in cond_data["results"]:
                    if r.get("score", -1) not in (0, 1, 2, 3):
                        continue
                    response_text = r.get("response_preview", "")
                    out.append({
                        "probe_id": r["probe_id"],
                        "probe_text": r.get("probe_text", ""),
                        "response_text": response_text,
                        "original_score": r["score"],
                        "original_label": r.get("label", ""),
                    })
                return out
    return out


def main():
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("Set OPENAI_API_KEY (GPT-5 judge runs via OpenAI API)")

    from openai import OpenAI
    client = OpenAI()

    out = {
        "experiment": "cross_judge_12model",
        "prereg_sha256": "22323d216e3d395cf518204c59da88312e22a2d2c7e7586c24cfce8bb8077bce",
        "judge_model": "gpt-5",
        "note": "Reuses target responses from existing Sonnet-judge runs. "
                "Where only response_preview (300 chars) is saved, GPT-5 "
                "judge scores on the preview. Original Sonnet judge scored "
                "on full responses; truncation may bias comparison.",
        "per_target": {},
    }

    n_total = 0
    for display, fname, _ in TARGETS:
        n_total += 5 * 25  # estimate
    print(f"[cross_judge] Target: {n_total} judge calls (~${n_total*0.015:.0f} estimated)",
          flush=True)
    print(f"[cross_judge] Judge: gpt-5", flush=True)

    for display, fname, _ in TARGETS:
        path = REPO_ROOT / "docs" / fname
        if not path.exists():
            print(f"[cross_judge] SKIP {display} (missing {fname})", flush=True)
            continue
        d = json.loads(path.read_text())
        print(f"\n[cross_judge] === {display} ===", flush=True)
        target_data = {"display": display, "source_file": fname, "per_condition": {}}
        for cond in CONDITIONS:
            cells = extract_responses(d, cond)
            if not cells:
                print(f"[cross_judge]   {cond}: NO DATA", flush=True)
                continue
            print(f"[cross_judge]   {cond}: {len(cells)} cells, scoring with gpt-5...",
                  flush=True)
            re_judged = []
            for i, cell in enumerate(cells):
                try:
                    judgment = call_gpt5_judge(
                        client, cell["probe_text"], cell["response_text"])
                    re_judged.append({
                        "probe_id": cell["probe_id"],
                        "gpt5_score": judgment.get("score", -1),
                        "gpt5_label": judgment.get("label", ""),
                        "gpt5_reason": (judgment.get("reason", "") or "")[:120],
                        "sonnet_score": cell["original_score"],
                    })
                except Exception as e:
                    re_judged.append({
                        "probe_id": cell["probe_id"],
                        "gpt5_score": -1,
                        "gpt5_label": "",
                        "gpt5_reason": f"err: {str(e)[:80]}",
                        "sonnet_score": cell["original_score"],
                    })
                if (i + 1) % 5 == 0:
                    valid = [r["gpt5_score"] for r in re_judged
                             if r["gpt5_score"] in (0, 1, 2, 3)]
                    print(f"[cross_judge]     [{cond} {i+1}/{len(cells)}] "
                          f"gpt5_mean={sum(valid)/max(len(valid),1):.2f}",
                          flush=True)
            target_data["per_condition"][cond] = re_judged
            # Incremental save after each condition
            out["per_target"][display] = target_data
            out_path = REPO_ROOT / "docs/CROSS_JUDGE_12MODEL_RAW.json"
            out_path.write_text(json.dumps(out, indent=2, default=str))

    print("\n[cross_judge] All targets complete.", flush=True)


if __name__ == "__main__":
    main()
