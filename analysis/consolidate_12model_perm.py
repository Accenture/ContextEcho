"""Consolidate paired-permutation primary results across all 12 targets'
content-position runs into one table.

Some files have pre-computed paired_vs_scratch (CONTENT_POSITION_*.json).
Others (OPTION_C_*.json) have raw per-probe scores in full_results; we
compute paired permutation ourselves.

Reports both:
  (a) Within-target Holm: per-target Holm across that target's 4
      non-scratch contrasts (matches the existing analysis).
  (b) Panel-level Holm: Holm across all 4 contrasts × 12 targets = 48
      contrasts in the panel (stricter).

Outputs:
  docs/CROSS_ORG_PERM_CONSOLIDATED.json
  docs/CROSS_ORG_PERM_CONSOLIDATED.md
"""
from __future__ import annotations
import json
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Map: display name, content-position file (None = no run)
TARGETS = [
    ("Sonnet 4.5",    "OPTION_C_SONNET45.json"),
    ("Sonnet 4.6",    "CONTENT_POSITION_SONNET.json"),
    ("Haiku 4.5",     "OPTION_C_HAIKU45.json"),
    ("Opus 4.6",      "OPTION_C_OPUS46.json"),
    ("Opus 4.7",      "CONTENT_POSITION_OPUS.json"),
    ("GPT-5",         None),
    ("GPT-4o",        "CONTENT_POSITION_GPT4O.json"),
    ("GPT-4o-mini",   "CONTENT_POSITION_GPT4O_MINI.json"),
    ("GPT-4.1",       "CONTENT_POSITION_GPT41.json"),
    ("Llama 3.3 70B", "CONTENT_POSITION_LLAMA33_70B.json"),
    ("Qwen 3 235B",   "CONTENT_POSITION_QWEN3_235B.json"),
    ("DeepSeek V3",   "CONTENT_POSITION_DEEPSEEK_V3.json"),
]

CONDITIONS = ["recent3K", "recent3K_filler", "recent3K_earlier", "filler14K"]


def paired_permutation(deltas: list[float], n_resamples=10_000, seed=42):
    """Per-probe paired diffs deltas; sign-flip permutation, two-sided p."""
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


def holm(p_with_keys: list[tuple[str, float]]) -> dict[str, float]:
    sorted_items = sorted(p_with_keys, key=lambda kp: kp[1])
    m = len(sorted_items)
    out = {}
    running = 0.0
    for i, (k, p) in enumerate(sorted_items):
        adj = (m - i) * p
        running = max(running, adj)
        out[k] = min(running, 1.0)
    return out


def extract_per_probe(d: dict, condition: str) -> dict[str, int]:
    """Return {probe_id: score} for the given condition. Tries multiple
    schemas: paired_vs_scratch (already computed) won't have this; instead
    we go through full_results or per_condition.results."""
    # Try full_results / per_condition pattern
    for key in ("full_results", "per_condition"):
        if key in d:
            cond_data = d[key].get(condition)
            if cond_data and "results" in cond_data:
                return {r["probe_id"]: r["score"] for r in cond_data["results"]
                        if r.get("score", -1) in (0, 1, 2, 3)}
    return {}


def compute_contrast(d: dict, condition: str, baseline: str = "scratch"):
    """Return (delta, p_perm, n) for condition vs baseline."""
    # First, prefer pre-computed if present
    pvs = d.get("paired_vs_scratch")
    if pvs and condition in pvs:
        c = pvs[condition]
        return c.get("mean_diff_vs_scratch"), c.get("p_raw"), c.get("n_pairs", 25)
    # Otherwise compute from raw
    base_scores = extract_per_probe(d, baseline)
    cond_scores = extract_per_probe(d, condition)
    common = sorted(set(base_scores) & set(cond_scores))
    deltas = [cond_scores[k] - base_scores[k] for k in common]
    obs, p = paired_permutation(deltas)
    return obs, p, len(deltas)


def main():
    rows = []
    for display, fname in TARGETS:
        if fname is None:
            for cond in CONDITIONS:
                rows.append({"target": display, "condition": cond,
                             "delta": None, "p_raw": None, "p_holm_within": None,
                             "n": 0, "note": "no content-position run"})
            continue
        path = REPO_ROOT / "docs" / fname
        if not path.exists():
            print(f"Skipping {display}: {path} missing")
            continue
        d = json.loads(path.read_text())
        # Compute all 4 contrasts; then within-target Holm
        target_contrasts = []
        for cond in CONDITIONS:
            delta, p, n = compute_contrast(d, cond)
            target_contrasts.append({"target": display, "condition": cond,
                                     "delta": delta, "p_raw": p, "n": n})
        # Within-target Holm (across 4 contrasts)
        within_holm = holm([(f'{c["condition"]}', c["p_raw"])
                            for c in target_contrasts if c["p_raw"] is not None])
        for c in target_contrasts:
            c["p_holm_within"] = within_holm.get(c["condition"]) if c["p_raw"] is not None else None
            rows.append(c)
        print(f"  {display:14}: recent3K Δ={target_contrasts[0]['delta']:+.2f} "
              f"p_raw={target_contrasts[0]['p_raw']:.4f} "
              f"p_holm_within={target_contrasts[0]['p_holm_within']:.4f}")

    # Panel-level Holm across all valid contrasts
    panel_keys = []
    for r in rows:
        if r.get("p_raw") is not None:
            panel_keys.append((f'{r["target"]}::{r["condition"]}', r["p_raw"]))
    panel_holm = holm(panel_keys)
    for r in rows:
        key = f'{r["target"]}::{r["condition"]}'
        r["p_holm_panel"] = panel_holm.get(key)

    out = {"experiment": "12_model_panel_paired_permutation_consolidated",
           "n_contrasts": len(panel_keys),
           "rows": rows}
    out_path = REPO_ROOT / "docs/CROSS_ORG_PERM_CONSOLIDATED.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nWrote {out_path}")

    # Markdown
    n_sig_within = sum(1 for r in rows if r.get("p_holm_within") is not None and r["p_holm_within"] < 0.05)
    n_sig_panel = sum(1 for r in rows if r.get("p_holm_panel") is not None and r["p_holm_panel"] < 0.05)

    md = ["# Cross-organizational paired-permutation consolidation\n\n",
          f"Total contrasts in panel: {len(panel_keys)}\n",
          f"Within-target Holm-significant: **{n_sig_within} / {len(panel_keys)}**\n",
          f"Panel-level Holm-significant: **{n_sig_panel} / {len(panel_keys)}**\n\n",
          "## recent3K-only headline contrast (each row is one target)\n\n",
          "| Target | Δ vs scratch | p_raw | p_holm (within-target) | p_holm (panel) | Sig within | Sig panel |\n",
          "|---|---|---|---|---|---|---|\n"]
    for r in rows:
        if r["condition"] == "recent3K":
            if r.get("delta") is None:
                md.append(f"| {r['target']} | — | — | — | — | — | — |\n")
                continue
            sig_within = "✓" if r["p_holm_within"] < 0.05 else ""
            sig_panel = "✓" if r["p_holm_panel"] is not None and r["p_holm_panel"] < 0.05 else ""
            md.append(f"| {r['target']} | {r['delta']:+.2f} | {r['p_raw']:.4f} | "
                      f"{r['p_holm_within']:.4f} | {r['p_holm_panel']:.4f} | {sig_within} | {sig_panel} |\n")
    md.append(f"\n## Full table (all 4 contrasts × 11 targets)\n\n")
    md.append("| Target | Condition | Δ | p_raw | p_holm (within) | p_holm (panel) |\n")
    md.append("|---|---|---|---|---|---|\n")
    for r in rows:
        if r.get("delta") is None:
            md.append(f"| {r['target']} | {r['condition']} | — | — | — | — |\n")
            continue
        md.append(f"| {r['target']} | {r['condition']} | {r['delta']:+.2f} | "
                  f"{r['p_raw']:.4f} | {r['p_holm_within']:.4f} | {r['p_holm_panel']:.4f} |\n")

    md_path = REPO_ROOT / "docs/CROSS_ORG_PERM_CONSOLIDATED.md"
    md_path.write_text("".join(md))
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
