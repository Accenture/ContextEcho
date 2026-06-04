"""Cross-judge audit: re-judge the existing 5-coding-probe panel
responses with a second judge (GPT-5) and compute agreement vs. the
primary Sonnet judge.

Output:
  results/crossjudge_audit/RESULTS.json
  results/crossjudge_audit/AGREEMENT.md  (κ, ρ, %-agreement summary)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from harness.cost import CostTracker  # type: ignore
from harness.judge import Judge  # type: ignore

PROBES_ROOT = REPO_ROOT / "results" / "probes_at_crosscompaction"
OUT = REPO_ROOT / "results" / "crossjudge_audit"
OUT.mkdir(parents=True, exist_ok=True)

CODING_PROBE_IDS = {"C01", "C02", "C03", "C04", "C05"}


def main() -> int:
    if not os.environ.get("OPEN_ROUTER__API_KEY"):
        sys.exit("OPEN_ROUTER__API_KEY not set (need OpenRouter for GPT-5 judge)")

    # Use OpenRouter for GPT-5 judge (OpenAI direct is out of credit)
    cost = CostTracker(OUT / "judge_cost.csv")
    gpt5_judge = Judge(
        provider="openrouter",  # type: ignore[arg-type]
        model_id="openai/gpt-5",
        cost_tracker=cost, session_id="crossjudge_gpt5",
    )

    rows = []
    n_total = 0
    n_done = 0

    # Iterate all P5 cells (panel-wide) on the 5-coding probe sub-battery
    for tgt_dir in sorted(PROBES_ROOT.iterdir()):
        if not tgt_dir.is_dir():
            continue
        target = tgt_dir.name
        p5_dir = tgt_dir / "P5_pre_C6"
        if not p5_dir.exists():
            continue
        for arm in ("claude_session", "filler"):
            arm_dir = p5_dir / arm
            if not arm_dir.exists():
                continue
            for f in sorted(arm_dir.iterdir()):
                if f.suffix != ".json": continue
                if f.stem not in CODING_PROBE_IDS: continue
                try:
                    cell = json.loads(f.read_text())
                except Exception:
                    continue
                resp = cell.get("response_text", "")
                probe_text = cell.get("probe_text", "")
                sonnet_score = cell.get("score")
                if not isinstance(sonnet_score, int):
                    continue
                n_total += 1

                out_path = OUT / target / "P5_pre_C6" / arm / f"{f.stem}.json"
                if out_path.exists():
                    n_done += 1
                    rows.append(json.loads(out_path.read_text()))
                    continue

                try:
                    j = gpt5_judge.score(probe_text, resp)
                    record = {
                        "target": target, "arm": arm, "probe_id": f.stem,
                        "sonnet_score": sonnet_score,
                        "gpt5_score": j.score, "gpt5_label": j.label,
                        "gpt5_reason": j.reason,
                    }
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    out_path.write_text(json.dumps(record, indent=2))
                    rows.append(record)
                    n_done += 1
                except Exception as e:
                    print(f"  ERROR {target}/{arm}/{f.stem}: {e}")

        print(f"  {target}: cum {n_done}/{n_total}")

    # Agreement statistics
    pairs = [(r["sonnet_score"], r["gpt5_score"]) for r in rows
             if r.get("gpt5_score") is not None]
    if not pairs:
        print("No paired scores collected")
        return 1

    n = len(pairs)
    exact_agree = sum(1 for s, g in pairs if s == g) / n
    within_one = sum(1 for s, g in pairs if abs(s - g) <= 1) / n

    # Spearman rho
    import statistics
    s_arr = [p[0] for p in pairs]
    g_arr = [p[1] for p in pairs]

    def rank(arr):
        order = sorted(range(len(arr)), key=lambda i: arr[i])
        ranks = [0.0] * len(arr)
        for rk, idx in enumerate(order):
            ranks[idx] = rk + 1
        return ranks
    s_r = rank(s_arr); g_r = rank(g_arr)
    if statistics.pstdev(s_r) > 0 and statistics.pstdev(g_r) > 0:
        rho = statistics.correlation(s_r, g_r)
    else:
        rho = float("nan")

    # Cohen's kappa (treat scores as 4-class categorical 0/1/2/3)
    from collections import Counter
    n_total = len(pairs)
    cnt_s = Counter(s_arr); cnt_g = Counter(g_arr)
    p_o = exact_agree
    p_e = sum((cnt_s[k] / n_total) * (cnt_g[k] / n_total) for k in {0, 1, 2, 3})
    kappa = (p_o - p_e) / (1 - p_e) if p_e != 1 else float("nan")

    # Drift gap on each judge
    def gap(arr_score, key="sonnet_score"):
        cl = [r[key] for r in rows if r["arm"] == "claude_session"]
        fi = [r[key] for r in rows if r["arm"] == "filler"]
        return (sum(fi) / len(fi)) - (sum(cl) / len(cl)) if cl and fi else float("nan")

    sonnet_gap = gap(rows, "sonnet_score")
    gpt5_gap = gap(rows, "gpt5_score")

    summary = {
        "n_pairs": n,
        "exact_agreement": exact_agree,
        "within_one_agreement": within_one,
        "spearman_rho": rho,
        "cohen_kappa_4class": kappa,
        "panel_drift_gap_sonnet_judge": sonnet_gap,
        "panel_drift_gap_gpt5_judge": gpt5_gap,
    }
    (OUT / "RESULTS.json").write_text(json.dumps(summary, indent=2))

    md = [
        "# Cross-judge audit (5-coding-self probes, P5_pre_C6)",
        f"- n pairs: {n}",
        f"- exact agreement: {exact_agree:.3f}",
        f"- within-one agreement: {within_one:.3f}",
        f"- Spearman rho: {rho:.3f}",
        f"- Cohen kappa (4-class): {kappa:.3f}",
        f"- Panel-wide drift gap (Sonnet judge): {sonnet_gap:+.2f}",
        f"- Panel-wide drift gap (GPT-5 judge): {gpt5_gap:+.2f}",
    ]
    (OUT / "AGREEMENT.md").write_text("\n".join(md))
    for line in md: print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
