"""Path A correlation analysis — runs after path_a_joint_behavioral_activation.py
produces docs/PATH_A_<MODEL>.json files.

Tests pre-registered hypotheses H1-H5 from PREREG_PATH_A.md:

  H1 (primary): Spearman rho >= 0.5 between per-cell axis projection
                and behavioral score on Qwen 3 32B (drifter).
  H2 (primary): mean axis projection on recent3K significantly
                lower than scratch on Qwen 3 32B (paired Welch t-test, p < 0.05).
  H3 (primary): no significant projection difference on Llama 3.3 70B
                (non-drifter), confirming behavioral null.
  H4 (secondary): filler-kills-drift signature visible in projections
                  (recent3K_filler closer to scratch than recent3K_earlier).
  H5 (secondary): content-accumulation on Qwen (recent3K_earlier projection
                  lower than recent3K projection).

Outputs:
  docs/PATH_A_CORRELATION_ANALYSIS.json
  docs/PATH_A_CORRELATION_ANALYSIS.md
"""
import json
import math
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def load_path_a(json_path: Path):
    if not json_path.exists():
        return None
    return json.loads(json_path.read_text())


def spearman_rho(xs: list[float], ys: list[float]) -> float:
    """Spearman rho without scipy."""
    if len(xs) != len(ys) or not xs:
        return float('nan')
    def rank(vs):
        sorted_v = sorted(enumerate(vs), key=lambda p: p[1])
        ranks = [0.0] * len(vs)
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
    rx, ry = rank(xs), rank(ys)
    n = len(xs)
    mx = sum(rx) / n
    my = sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    dx = math.sqrt(sum((rx[i] - mx) ** 2 for i in range(n)))
    dy = math.sqrt(sum((ry[i] - my) ** 2 for i in range(n)))
    if dx == 0 or dy == 0:
        return float('nan')
    return num / (dx * dy)


def welch_ttest(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """Welch's t-test (unequal variances). Returns (t, two-sided p)."""
    if not xs or not ys:
        return float('nan'), float('nan')
    nx, ny = len(xs), len(ys)
    mx = sum(xs) / nx
    my = sum(ys) / ny
    vx = sum((x - mx) ** 2 for x in xs) / (nx - 1) if nx > 1 else 0.0
    vy = sum((y - my) ** 2 for y in ys) / (ny - 1) if ny > 1 else 0.0
    se = math.sqrt(vx / nx + vy / ny)
    if se == 0:
        return float('nan'), float('nan')
    t = (mx - my) / se
    # Welch–Satterthwaite degrees of freedom
    if vx == 0 or vy == 0:
        df = max(nx - 1, ny - 1)
    else:
        df = (vx / nx + vy / ny) ** 2 / (
            (vx / nx) ** 2 / (nx - 1) + (vy / ny) ** 2 / (ny - 1)
        )
    # Two-sided p via t-distribution survival; we approximate with scipy if available
    try:
        from scipy import stats
        p = 2 * stats.t.sf(abs(t), df)
    except ImportError:
        # Fallback: normal approximation (large df)
        from math import erfc, sqrt
        p = erfc(abs(t) / sqrt(2))
    return t, p


def per_condition_data(d: dict, key: str = 'axis_projection'):
    """Returns {condition: [values_per_probe]}."""
    out = {}
    for cond, payload in d['per_condition'].items():
        out[cond] = [r[key] for r in payload['results'] if isinstance(r.get(key), (int, float))]
    return out


def per_cell_pairs(d: dict):
    """Returns (scores, projections) lists across all (cond, probe) cells with valid scores."""
    scores, projs = [], []
    for cond, payload in d['per_condition'].items():
        for r in payload['results']:
            score = r.get('score')
            proj = r.get('axis_projection')
            if score in (0, 1, 2, 3) and isinstance(proj, (int, float)):
                scores.append(score)
                projs.append(proj)
    return scores, projs


def analyze_target(label: str, json_path: Path):
    d = load_path_a(json_path)
    if d is None:
        return None
    print(f'\n=== {label} ===')
    print(f"target: {d.get('target_model')}, target_layer={d.get('target_layer')}")

    # H1: cross-cell Spearman correlation
    scores, projs = per_cell_pairs(d)
    rho = spearman_rho(scores, projs) if scores else float('nan')
    print(f"H1: Spearman rho between behavioral score and axis projection (n={len(scores)} cells): {rho:+.3f}")

    # H2: mean projection scratch vs recent3K (Welch t-test)
    proj_by_cond = per_condition_data(d, 'axis_projection')
    score_by_cond = per_condition_data(d, 'score')
    if 'scratch' in proj_by_cond and 'recent3K' in proj_by_cond:
        s, p = welch_ttest(proj_by_cond['scratch'], proj_by_cond['recent3K'])
        mean_scratch = sum(proj_by_cond['scratch']) / max(len(proj_by_cond['scratch']), 1)
        mean_recent = sum(proj_by_cond['recent3K']) / max(len(proj_by_cond['recent3K']), 1)
        h2_result = {'mean_scratch': mean_scratch, 'mean_recent3K': mean_recent,
                     'delta_proj': mean_recent - mean_scratch, 't': s, 'p': p}
        print(f"H2/H3: scratch projection {mean_scratch:+.3f} vs recent3K projection {mean_recent:+.3f}")
        print(f"        delta={mean_recent - mean_scratch:+.3f}, t={s:.3f}, p={p:.4f}")
    else:
        h2_result = None

    # H4 + H5: per-condition mean projection
    cond_means = {c: (sum(v) / max(len(v), 1)) for c, v in proj_by_cond.items()}
    print('per-condition mean axis projection:', cond_means)

    # Behavioral results (sanity check vs API-level Δ)
    score_means = {c: (sum(v) / max(len(v), 1)) for c, v in score_by_cond.items() if v}
    print('per-condition mean behavioral score:', score_means)

    return {
        'label': label,
        'target_model': d.get('target_model'),
        'target_layer': d.get('target_layer'),
        'h1_spearman_rho_score_vs_proj': rho,
        'h1_n_cells': len(scores),
        'h2_h3_scratch_vs_recent3K_proj': h2_result,
        'mean_projection_per_condition': cond_means,
        'mean_behavioral_per_condition': score_means,
    }


def main():
    out = {}
    out['qwen'] = analyze_target('Qwen 3 32B', REPO_ROOT / 'docs/PATH_A_QWEN3_32B.json')
    out['llama'] = analyze_target('Llama 3.3 70B', REPO_ROOT / 'docs/PATH_A_LLAMA33_70B.json')

    # Pre-registered decision rules
    decision = {}
    if out['qwen']:
        rho = out['qwen']['h1_spearman_rho_score_vs_proj']
        h2 = out['qwen']['h2_h3_scratch_vs_recent3K_proj']
        decision['H1_qwen_supported'] = rho >= 0.5 if rho == rho else False
        decision['H1_qwen_borderline'] = (0.3 <= rho < 0.5) if rho == rho else False
        decision['H2_qwen_supported'] = (h2 and h2['p'] < 0.05 and h2['delta_proj'] < 0) if h2 else False
        if 'recent3K_filler' in out['qwen']['mean_projection_per_condition'] and 'recent3K_earlier' in out['qwen']['mean_projection_per_condition']:
            f = out['qwen']['mean_projection_per_condition']['recent3K_filler']
            e = out['qwen']['mean_projection_per_condition']['recent3K_earlier']
            r = out['qwen']['mean_projection_per_condition'].get('recent3K')
            if r is not None:
                decision['H4_filler_kills_signature'] = (f - r) > (e - r)  # filler closer to scratch
                decision['H5_content_accumulation'] = e < r  # earlier prepended -> more drift
    if out['llama']:
        h2 = out['llama']['h2_h3_scratch_vs_recent3K_proj']
        decision['H3_llama_null_supported'] = (h2 and h2['p'] >= 0.05) if h2 else False

    # Coerce any numpy bool to native bool so json.dumps doesn't choke
    decision = {k: bool(v) if isinstance(v, (bool,)) or hasattr(v, 'item') else v
                for k, v in decision.items()}
    out['decision'] = decision
    print('\n=== Pre-registered decision rules ===')
    print(json.dumps(decision, indent=2, default=str))

    out_path = REPO_ROOT / 'docs/PATH_A_CORRELATION_ANALYSIS.json'
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f'\nWrote {out_path}')


if __name__ == '__main__':
    main()
