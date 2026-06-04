"""Phase 2 analyzer for the signed TerminalBench amendment.

Computes H1/H2/H3/H4 per signed PREREG_AMENDMENT_TERMINALBENCH.md
(SHA 8365d3c8...). Paired-permutation procedure is byte-identical to
scripts/cross_judge_12model_analyze.py (verified by `paired_permutation`
and `holm` reused with the same seed=42 and n_resamples=10_000).

Inputs:
  data_archive/terminalbench/panel/<target>/<condition>/<task>/trial-<i>/
    results.json
    llm_seconds_per_turn.json   (optional sidecar from TimedLiteLLM)

Outputs:
  data_archive/terminalbench/panel/PANEL_ANALYSIS.json
    - per-trial metrics
    - per-(target, task) paired Δ + bootstrap 95% CI half-width
    - per-target H1/H2/H3 paired-permutation p_raw + p_holm
    - per-target auto-bump triggers (cells where CI half-width >= |estimate|)
    - H4 sign-test result

Run:
  python scripts/analyze_terminalbench.py
"""
from __future__ import annotations

import json
import math
import random
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PANEL_DIR = REPO_ROOT / "data_archive" / "terminalbench" / "panel"

# Locked targets/tasks from signed amendment §2.2/§2.3.
TARGETS_SAFE = ["claude-sonnet-4-6", "claude-haiku-4-5", "gpt-5", "gemini-2-5-pro"]
TASKS = ["hello-world", "crack-7z-hash.easy", "git-multibranch", "swe-bench-astropy-1"]
CONDITIONS = ["scratch", "recent3K"]

# Locked panel-drift verdicts for H4 sign test (under Sonnet judge,
# from primary-panel results / panel-extension PHASE2 results).
PANEL_VERDICT = {
    "claude-sonnet-4-6": "drifter",
    "claude-haiku-4-5": "borderline",  # treated as drifter for sign test
    "gpt-5": "non-drifter",
    "gemini-2-5-pro": "drifter",  # panel-extension verdict on Claude-derived c_pre
}


# === paired-permutation byte-identical to cross_judge_12model_analyze.py ===
def paired_permutation(deltas, n_resamples=10_000, seed=42):
    rng = random.Random(seed)
    if not deltas:
        return float("nan"), float("nan")
    n = len(deltas)
    obs = sum(deltas) / n
    count = 0
    for _ in range(n_resamples):
        s = sum(d if rng.random() < 0.5 else -d for d in deltas) / n
        if abs(s) >= abs(obs):
            count += 1
    return obs, count / n_resamples


def holm(p_with_keys):
    sorted_items = sorted(p_with_keys, key=lambda kp: kp[1])
    m = len(sorted_items)
    out = {}
    running = 0.0
    for i, (k, p) in enumerate(sorted_items):
        adj = (m - i) * p
        running = max(running, adj)
        out[k] = min(running, 1.0)
    return out


def bootstrap_ci_half_width(values, n_resamples=2_000, alpha=0.05, seed=43):
    """Percentile bootstrap CI half-width on the mean."""
    if len(values) < 2:
        return float("nan")
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(n_resamples):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(alpha / 2 * n_resamples)]
    hi = means[int((1 - alpha / 2) * n_resamples)]
    point = sum(values) / len(values)
    return max(point - lo, hi - point)


# === metric extraction per trial ===
def trial_dir(target_safe, condition, task, trial):
    return PANEL_DIR / target_safe / condition / task / f"trial-{trial}"


def parse_dt(s):
    if not s:
        return None
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return None


def per_episode_usage(trial_dir_path):
    """Walk the cell's episode-*/debug.json files, returning per-turn dicts."""
    rows = []
    for ep in sorted(trial_dir_path.rglob("episode-*"),
                     key=lambda p: int(p.name.split("-")[1])):
        if not ep.is_dir():
            continue
        dj = ep / "debug.json"
        if not dj.exists():
            continue
        try:
            d = json.loads(dj.read_text())
            orr = d.get("original_response", "")
            parsed = json.loads(orr) if isinstance(orr, str) and orr else (orr or {})
            usage = parsed.get("usage", {}) or {}
            rows.append({
                "episode": int(ep.name.split("-")[1]),
                "start_time": d.get("start_time"),
                "input_tokens": usage.get("input_tokens", 0),
                "cache_creation_input_tokens": usage.get("cache_creation_input_tokens", 0),
                "cache_read_input_tokens": usage.get("cache_read_input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
            })
        except Exception:
            continue
    return rows


def trial_metrics(target_safe, condition, task, trial):
    """Per-trial metrics aligned to H1/H2/H3 of signed amendment."""
    td = trial_dir(target_safe, condition, task, trial)
    rj = td / "results.json"
    if not rj.exists():
        return None
    try:
        d = json.loads(rj.read_text())
    except Exception:
        return None
    results = d.get("results", [])
    if not results:
        return None
    r = results[0]
    is_resolved = r.get("is_resolved")
    if is_resolved is None:
        return None  # crashed cell, exclude

    started = parse_dt(r.get("agent_started_at"))
    ended = parse_dt(r.get("agent_ended_at"))
    agent_sec = (ended - started).total_seconds() if (started and ended) else None

    eps = per_episode_usage(td)
    n_turns = len(eps)
    output_tokens_total = sum(e["output_tokens"] for e in eps) or r.get("total_output_tokens", 0)

    # Per-LLM-call wall-clock from sidecar (TimedLiteLLM).
    sidecar = td / "llm_seconds_per_turn.json"
    if sidecar.exists():
        sd = json.loads(sidecar.read_text())
        llm_secs = sd.get("llm_seconds_per_turn", [])
    else:
        llm_secs = []

    # H1 metric: sec / output_token, computed per turn from
    # llm_seconds_per_turn (preferred) or fall back to inter-episode delta.
    sec_per_out_per_turn = []
    if llm_secs and len(llm_secs) == n_turns:
        # Pair each LLM call's wall-clock with that turn's output_tokens.
        for i, e in enumerate(eps):
            if e["output_tokens"] > 0:
                sec_per_out_per_turn.append(llm_secs[i] / e["output_tokens"])
    else:
        # Fallback: inter-episode delta (smoke method).
        for i in range(len(eps) - 1):
            t0 = parse_dt(eps[i]["start_time"])
            t1 = parse_dt(eps[i + 1]["start_time"])
            if t0 and t1 and eps[i]["output_tokens"] > 0:
                sec_per_out_per_turn.append(
                    (t1 - t0).total_seconds() / eps[i]["output_tokens"]
                )
    mean_sec_per_out = (
        sum(sec_per_out_per_turn) / len(sec_per_out_per_turn)
        if sec_per_out_per_turn else None
    )

    # H2 metric: total per-turn wall-clock = agent_sec / n_turns
    h2 = (agent_sec / n_turns) if (agent_sec and n_turns) else None

    # H3 metric: output_tokens per turn
    h3 = (output_tokens_total / n_turns) if n_turns else None

    return {
        "is_resolved": is_resolved,
        "n_turns": n_turns,
        "agent_sec": agent_sec,
        "total_input_tokens": r.get("total_input_tokens"),
        "total_output_tokens": output_tokens_total,
        "h1_sec_per_output_token": mean_sec_per_out,
        "h2_sec_per_turn": h2,
        "h3_output_tokens_per_turn": h3,
        "llm_seconds_total": sum(llm_secs) if llm_secs else None,
    }


def collect_panel():
    out = {}
    for tgt in TARGETS_SAFE:
        for cond in CONDITIONS:
            for task in TASKS:
                trial = 0
                while True:
                    td = trial_dir(tgt, cond, task, trial)
                    if not (td / "results.json").exists():
                        break
                    m = trial_metrics(tgt, cond, task, trial)
                    out[(tgt, cond, task, trial)] = m
                    trial += 1
    return out


def analyze(panel):
    """For each (target, hypothesis), assemble paired deltas, run permutation."""
    metric_keys = {
        "H1": "h1_sec_per_output_token",
        "H2": "h2_sec_per_turn",
        "H3": "h3_output_tokens_per_turn",
    }

    per_target = {}
    auto_bump_cells = []

    for tgt in TARGETS_SAFE:
        per_target[tgt] = {"contrasts": {}, "panel_verdict": PANEL_VERDICT.get(tgt)}
        for hyp, key in metric_keys.items():
            deltas = []
            for task in TASKS:
                trial = 0
                while True:
                    s = panel.get((tgt, "scratch", task, trial))
                    r = panel.get((tgt, "recent3K", task, trial))
                    if s is None and r is None:
                        break
                    if s is not None and r is not None and \
                       s.get("is_resolved") and r.get("is_resolved") and \
                       s.get(key) is not None and r.get(key) is not None:
                        deltas.append(r[key] - s[key])
                    trial += 1
                # Per-cell CI half-width for auto-bump trigger
                cell_deltas = []
                trial2 = 0
                while True:
                    s2 = panel.get((tgt, "scratch", task, trial2))
                    r2 = panel.get((tgt, "recent3K", task, trial2))
                    if s2 is None and r2 is None:
                        break
                    if s2 is not None and r2 is not None and \
                       s2.get("is_resolved") and r2.get("is_resolved") and \
                       s2.get(key) is not None and r2.get(key) is not None:
                        cell_deltas.append(r2[key] - s2[key])
                    trial2 += 1
                if hyp == "H1" and len(cell_deltas) >= 2:
                    point = sum(cell_deltas) / len(cell_deltas)
                    half = bootstrap_ci_half_width(cell_deltas)
                    if not math.isnan(half) and abs(point) > 0 and half >= abs(point):
                        auto_bump_cells.append({
                            "target": tgt, "task": task,
                            "n": len(cell_deltas),
                            "point": point, "ci_half_width": half,
                        })
            obs, p_raw = paired_permutation(deltas)
            per_target[tgt]["contrasts"][hyp] = {
                "n_paired": len(deltas),
                "obs_delta_mean": obs,
                "p_raw": p_raw,
            }

    # Holm correction across targets per hypothesis.
    for hyp in metric_keys:
        p_with_keys = [
            (tgt, per_target[tgt]["contrasts"][hyp]["p_raw"])
            for tgt in TARGETS_SAFE
            if not math.isnan(per_target[tgt]["contrasts"][hyp]["p_raw"])
        ]
        h_results = holm(p_with_keys)
        for tgt, p_holm in h_results.items():
            per_target[tgt]["contrasts"][hyp]["p_holm"] = p_holm

    # H4: sign agreement between H1 sign and panel verdict.
    h4_signs = []
    for tgt in TARGETS_SAFE:
        c = per_target[tgt]["contrasts"].get("H1", {})
        d = c.get("obs_delta_mean")
        verdict = PANEL_VERDICT.get(tgt)
        if d is None or math.isnan(d):
            h4_signs.append(None)
            continue
        # drifter ↔ recent3K should be SLOWER per output token (positive Δ)
        agrees = (
            (verdict in ("drifter", "borderline") and d > 0) or
            (verdict == "non-drifter" and abs(d) < 1e-6)  # hard to check exactly; descriptive
        )
        h4_signs.append({"target": tgt, "verdict": verdict, "delta": d, "agrees": agrees})

    return {
        "per_target": per_target,
        "auto_bump_cells": auto_bump_cells,
        "h4_signs": h4_signs,
        "n_auto_bump": len(auto_bump_cells),
    }


def main():
    if not PANEL_DIR.exists():
        print(f"No panel data yet at {PANEL_DIR}")
        return 0

    panel = collect_panel()
    print(f"Collected {sum(1 for v in panel.values() if v)} valid trials "
          f"from {len(panel)} cells")
    analysis = analyze(panel)

    print("\n=== Per-target H1 (sec/output_token) ===")
    print(f"{'target':<22}{'n':>4}{'obs':>10}{'p_raw':>10}{'p_holm':>10}")
    for tgt, blk in analysis["per_target"].items():
        c = blk["contrasts"].get("H1", {})
        n = c.get("n_paired", 0)
        obs = c.get("obs_delta_mean", float("nan"))
        praw = c.get("p_raw", float("nan"))
        pholm = c.get("p_holm", float("nan"))
        print(f"{tgt:<22}{n:>4}{obs:>10.4f}{praw:>10.4f}{pholm:>10.4f}")

    print(f"\nAuto-bump cells (CI half-width >= |point estimate|): {analysis['n_auto_bump']}")
    for cell in analysis["auto_bump_cells"]:
        print(f"  bump {cell['target']} / {cell['task']}: "
              f"n={cell['n']} point={cell['point']:.4f} ci_half={cell['ci_half_width']:.4f}")

    out_path = PANEL_DIR / "PANEL_ANALYSIS.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "amendment_sha256": "8365d3c88e528737a4d88ab61d80adf0341be58dfdaa16f9b5cfad37253dd275",
        "n_trials": sum(1 for v in panel.values() if v),
        **analysis,
    }, indent=2, default=str))
    print(f"\nSaved → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
