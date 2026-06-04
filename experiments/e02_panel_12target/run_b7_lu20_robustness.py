"""B7: Lu-derived 20-probe robustness re-analysis.

Drop the 5 coding-context probes (C01..C05) added during rubric
development; re-run all reported analyses on the 20 Lu-derived
probes only (I01-I04 identity, O01-O04 origin, E01-E04 experience,
P01-P04 preference, R01-R04 relational).

Tests whether the headline effects depend on probes added during
rubric iteration. Addresses Claude reviewer W4 + Q5
(probe-suite contamination concern).

Pure re-analysis: no new API calls.
"""
import json
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
random.seed(42)

LU20 = {f'I0{i}' for i in range(1, 5)} | {f'O0{i}' for i in range(1, 5)} | \
       {f'E0{i}' for i in range(1, 5)} | {f'P0{i}' for i in range(1, 5)} | \
       {f'R0{i}' for i in range(1, 5)}
ALL25 = LU20 | {f'C0{i}' for i in range(1, 6)}


def per_probe(full_results: dict, condition: str, probe_set: set[str]) -> dict[str, int]:
    out = {}
    for entry in full_results[condition]['results']:
        if entry['probe_id'] in probe_set and entry.get('score', -1) in (0, 1, 2, 3):
            out[entry['probe_id']] = entry['score']
    return out


def perm_p(scratch: dict, cond: dict, n_perm: int = 10000, seed: int = 42) -> tuple[float, float]:
    rng = random.Random(seed)
    common = sorted(set(scratch) & set(cond))
    paired = [(scratch[k], cond[k]) for k in common]
    n = len(paired)
    delta = sum(c - s for s, c in paired) / n
    abs_delta = abs(delta)
    extreme = 0
    for _ in range(n_perm):
        d = sum((c - s) if rng.random() < 0.5 else (s - c) for s, c in paired) / n
        if abs(d) >= abs_delta:
            extreme += 1
    p = (extreme + 1) / (n_perm + 1)
    return delta, p


def bootstrap_ci(scratch: dict, cond: dict, n_boot: int = 10000, seed: int = 42) -> tuple[float, float]:
    rng = random.Random(seed)
    common = sorted(set(scratch) & set(cond))
    paired = [(scratch[k], cond[k]) for k in common]
    n = len(paired)
    boot = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        sample = [paired[i] for i in idx]
        boot.append(sum(c - s for s, c in sample) / n)
    boot.sort()
    return boot[int(0.025 * n_boot)], boot[int(0.975 * n_boot) - 1]


def holm(p_dict: dict[str, float]) -> dict[str, float]:
    items = sorted(p_dict.items(), key=lambda x: x[1])
    n = len(items)
    out = {}
    last = 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, p * (n - i))
        adj = max(adj, last)
        out[k] = adj
        last = adj
    return out


def lme_all_conditions(per_probe_dict: dict[str, dict[str, int]]) -> dict[str, tuple[float, float]]:
    """Fit ONE LME with all conditions at once vs scratch reference,
    matching analyze_hierarchical.py. Returns {cond_name: (beta, p)}.

    per_probe_dict is {condition_name: {probe_id: score}}, with 'scratch'
    as one of the keys.
    """
    import warnings
    warnings.filterwarnings('ignore')
    import pandas as pd
    import statsmodels.formula.api as smf

    rows = []
    for cond_name, probes in per_probe_dict.items():
        for pid, score in probes.items():
            rows.append({'condition': cond_name, 'probe_id': pid, 'score': score})
    df = pd.DataFrame(rows)
    if 'scratch' not in df['condition'].unique():
        return {}
    other_conds = sorted(c for c in df['condition'].unique() if c != 'scratch')
    df['condition'] = pd.Categorical(df['condition'], categories=['scratch'] + other_conds)
    formula = "score ~ C(condition, Treatment(reference='scratch'))"
    out = {c: (float('nan'), float('nan')) for c in other_conds}
    for method in ('lbfgs', 'cg', 'bfgs', 'powell'):
        try:
            model = smf.mixedlm(formula, df, groups=df['probe_id'])
            fit = model.fit(method=method, disp=False)
            for name in fit.params.index:
                for cond_name in other_conds:
                    if f'T.{cond_name}]' in name:
                        out[cond_name] = (float(fit.params[name]), float(fit.pvalues[name]))
            return out
        except Exception:
            continue
    return out


def report_table(label: str, json_path: Path, probe_set: set[str], probe_set_name: str):
    print(f'\n=== {label} | probe set: {probe_set_name} (n={len(probe_set)}) ===')
    d = json.loads(json_path.read_text())
    fr = d['full_results']

    # Build per-condition probe dicts
    per_cond = {c: per_probe(fr, c, probe_set) for c in fr}
    scratch = per_cond['scratch']

    # Joint LME fit with all conditions vs scratch reference
    lme_results = lme_all_conditions(per_cond)

    contrasts = {}
    for cond_name in fr:
        if cond_name == 'scratch':
            continue
        cond = per_cond[cond_name]
        delta_p, p_perm = perm_p(scratch, cond)
        beta_lme, p_lme = lme_results.get(cond_name, (float('nan'), float('nan')))
        lo, hi = bootstrap_ci(scratch, cond)
        contrasts[cond_name] = {
            'delta': delta_p, 'ci_lo': lo, 'ci_hi': hi,
            'p_perm': p_perm, 'p_lme': p_lme,
        }

    perm_holm = holm({k: v['p_perm'] for k, v in contrasts.items()})
    # Only Holm-correct LME p's that are not NaN
    valid_lme = {k: v['p_lme'] for k, v in contrasts.items() if v['p_lme'] == v['p_lme']}  # NaN check
    lme_holm = holm(valid_lme) if valid_lme else {}
    for k in contrasts:
        if k not in lme_holm:
            lme_holm[k] = float('nan')

    print(f"{'Cond':<22}{'Δ':>8}{'95% CI':>22}{'PermP':>10}{'PermHolm':>11}{'LME':>10}{'LMEHolm':>11}")
    for k, v in contrasts.items():
        ph = perm_holm.get(k, float('nan'))
        lh = lme_holm.get(k, float('nan'))
        sig_p = '*' if ph == ph and ph < 0.05 else ''
        sig_l = '*' if lh == lh and lh < 0.05 else ''
        lme_str = f"{v['p_lme']:>10.4f}" if v['p_lme'] == v['p_lme'] else f"{'NaN':>10}"
        lh_str = f"{lh:>11.4f}" if lh == lh else f"{'NaN':>11}"
        print(f"{k:<22}{v['delta']:>+8.3f}  [{v['ci_lo']:+.3f}, {v['ci_hi']:+.3f}]"
              f"{v['p_perm']:>10.4f}{ph:>11.4f}{sig_p:<2}"
              f"{lme_str}{lh_str}{sig_l}")
    return {
        'json_path': str(json_path),
        'probe_set': probe_set_name,
        'n_probes': len(probe_set),
        'contrasts': contrasts,
        'perm_holm': perm_holm,
        'lme_holm': lme_holm,
    }


def main():
    out = {'experiment': 'B7_lu20_robustness',
           'description': 'Re-analyze all reported tables on Lu-derived 20 probes only',
           'lu20_probe_ids': sorted(LU20),
           'comparisons': []}

    for label, fname in [
        ('Sonnet length-control', 'LENGTH_CONTROL_SONNET.json'),
        ('Opus length-control', 'LENGTH_CONTROL_ANALYSIS.json'),
        ('GPT-5 length-control', 'LENGTH_CONTROL_GPT5.json'),
        ('Sonnet dose-response', 'DOSE_RESPONSE_SONNET.json'),
        ('Sonnet content-position', 'CONTENT_POSITION_SONNET.json'),
    ]:
        path = REPO_ROOT / 'docs' / fname
        all25 = report_table(label, path, ALL25, 'ALL25')
        lu20 = report_table(label, path, LU20, 'LU20')
        out['comparisons'].append({
            'label': label,
            'all25': all25,
            'lu20': lu20,
        })

    json_out = REPO_ROOT / 'docs/B7_LU20_ROBUSTNESS.json'
    json_out.write_text(json.dumps(out, indent=2))
    print(f'\nWrote {json_out}')


if __name__ == '__main__':
    main()
