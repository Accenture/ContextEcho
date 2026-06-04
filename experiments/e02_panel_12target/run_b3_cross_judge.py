"""B3: Re-score Opus and GPT-5 length-control responses under GPT-5
judge to complete the cross-judge replication across all 3 targets.

Sonnet x GPT-5 already exists in
docs/LENGTH_CONTROL_SONNET_GPT5JUDGE.json. This script does the same
for Opus and GPT-5 targets.

Cost: ~25-50 OpenAI judge calls per target × 2 targets = ~$15-25.
"""
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from harness.judge import JUDGE_SYSTEM_PROMPT


def parse_judge(raw: str) -> dict:
    dec = json.JSONDecoder()
    start = 0
    while True:
        start = raw.find("{", start)
        if start == -1:
            return {"score": -1, "label": "parse_error", "reason": raw[:200]}
        try:
            data, _ = dec.raw_decode(raw, start)
            if "score" in data and data["score"] in (0, 1, 2, 3):
                return data
            return {"score": -1, "label": "invalid_score", "reason": raw[:200]}
        except json.JSONDecodeError:
            start += 1


def call_gpt5_judge(client, probe_text, response_text, system, max_retries=3):
    user_msg = f"PROBE:\n{probe_text}\n\nRESPONSE:\n{response_text}\n\nReturn the JSON now."
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model="gpt-5",
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_msg},
                ],
                max_completion_tokens=4096,
            )
            return parse_judge(resp.choices[0].message.content)
        except Exception as e:
            last_err = e
            time.sleep(2 * (2 ** attempt))
    raise last_err


def spearman_rho(x, y):
    if len(x) != len(y) or not x:
        return float("nan")
    def rank(v):
        sorted_v = sorted(enumerate(v), key=lambda t: t[1])
        ranks = [0.0] * len(v)
        i = 0
        while i < len(sorted_v):
            j = i
            while j < len(sorted_v) and sorted_v[j][1] == sorted_v[i][1]:
                j += 1
            avg_rank = (i + j - 1) / 2 + 1
            for k in range(i, j):
                ranks[sorted_v[k][0]] = avg_rank
            i = j
        return ranks
    rx, ry = rank(x), rank(y)
    n = len(x)
    mx = sum(rx) / n
    my = sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    dx = sum((rx[i] - mx) ** 2 for i in range(n)) ** 0.5
    dy = sum((ry[i] - my) ** 2 for i in range(n)) ** 0.5
    if dx == 0 or dy == 0:
        return float("nan")
    return num / (dx * dy)


def cohen_kappa(x, y):
    """Quadratic-weighted Cohen's kappa."""
    if not x:
        return float("nan")
    cats = sorted(set(x) | set(y))
    n = len(x)
    obs = sum(1 - ((xi - yi) ** 2) / ((cats[-1] - cats[0]) ** 2)
              for xi, yi in zip(x, y)) / n
    px = {c: x.count(c) / n for c in cats}
    py = {c: y.count(c) / n for c in cats}
    exp = sum(px[c1] * py[c2] * (1 - ((c1 - c2) ** 2) / ((cats[-1] - cats[0]) ** 2))
              for c1 in cats for c2 in cats)
    return (obs - exp) / (1 - exp) if (1 - exp) > 0 else float("nan")


def rescore_target(client, in_json: Path, out_json: Path, target_label: str):
    print(f"\n=== {target_label} target × GPT-5 judge ===")
    print(f"Reading: {in_json}")
    data = json.load(open(in_json))
    full = data.get("full_results", {})

    all_results = {}
    primary_scores = []
    gpt5_scores = []

    for cond, cd in full.items():
        cond_results = []
        print(f"  [{cond}]")
        for i, r in enumerate(cd.get("results", [])):
            probe_text = r["probe_text"]
            response_text = r.get("response_preview", "")
            primary_score = r["score"]
            if primary_score not in (0, 1, 2, 3):
                continue
            try:
                gj = call_gpt5_judge(client, probe_text, response_text, JUDGE_SYSTEM_PROMPT)
                gpt5_score = gj.get("score", -1)
                gpt5_reason = gj.get("reason", "")[:120]
            except Exception as e:
                print(f"    [ERR] {r['probe_id']}: {e}")
                gpt5_score = -1
                gpt5_reason = ""
            cond_results.append({
                "probe_id": r["probe_id"],
                "probe_text": probe_text[:100],
                "primary_score": primary_score,
                "primary_reason": r.get("reason", "")[:120],
                "gpt5_score": gpt5_score,
                "gpt5_reason": gpt5_reason,
                "response_preview": response_text[:200],
            })
            if gpt5_score in (0, 1, 2, 3):
                primary_scores.append(primary_score)
                gpt5_scores.append(gpt5_score)
        all_results[cond] = cond_results

    n = len(primary_scores)
    rho = spearman_rho(primary_scores, gpt5_scores)
    kappa = cohen_kappa(primary_scores, gpt5_scores)
    print(f"  rho={rho:.3f}, kappa={kappa:.3f}, n={n}")

    cond_means = {}
    for cond, rs in all_results.items():
        ps = [r["primary_score"] for r in rs if r["primary_score"] in (0, 1, 2, 3)]
        gs = [r["gpt5_score"] for r in rs if r["gpt5_score"] in (0, 1, 2, 3)]
        cond_means[cond] = {
            "primary_mean": sum(ps) / len(ps) if ps else float("nan"),
            "gpt5_mean": sum(gs) / len(gs) if gs else float("nan"),
            "n_primary": len(ps),
            "n_gpt5": len(gs),
        }
        print(f"    {cond}: primary={cond_means[cond]['primary_mean']:.3f}, "
              f"gpt5={cond_means[cond]['gpt5_mean']:.3f}")

    out = {
        "target_label": target_label,
        "primary_judge": "claude-sonnet-4-6",
        "alternative_judge": "gpt-5",
        "n_paired": n,
        "spearman_rho": rho,
        "cohen_quadratic_kappa": kappa,
        "per_condition_means": cond_means,
        "per_probe_details": all_results,
    }
    out_json.write_text(json.dumps(out, indent=2, default=str))
    print(f"  Wrote {out_json}")
    return out


def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        # Read the key from the existing cross_judge_sonnet.py default
        cj = (REPO_ROOT / "cross_judge_sonnet.py").read_text()
        for line in cj.splitlines():
            if line.strip().startswith('"sk-proj-'):
                os.environ["OPENAI_API_KEY"] = line.strip().strip('",')
                break
    from openai import OpenAI
    client = OpenAI()

    targets = [
        ("opus", "Opus 4.7", REPO_ROOT / "docs/LENGTH_CONTROL_ANALYSIS.json",
         REPO_ROOT / "docs/LENGTH_CONTROL_OPUS_GPT5JUDGE.json"),
        ("gpt5_target", "GPT-5", REPO_ROOT / "docs/LENGTH_CONTROL_GPT5.json",
         REPO_ROOT / "docs/LENGTH_CONTROL_GPT5_GPT5JUDGE.json"),
    ]

    summary = {}
    for tag, label, in_path, out_path in targets:
        if not in_path.exists():
            print(f"SKIP {tag}: {in_path} not found")
            continue
        result = rescore_target(client, in_path, out_path, label)
        summary[tag] = {
            "rho": result["spearman_rho"],
            "kappa": result["cohen_quadratic_kappa"],
            "n": result["n_paired"],
            "per_condition_means": result["per_condition_means"],
        }

    print("\n=== B3 Summary ===")
    print(json.dumps(summary, indent=2, default=str))
    (REPO_ROOT / "docs/B3_CROSS_JUDGE_OPUS_GPT5_SUMMARY.json").write_text(
        json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
