"""Audit compliance on every per-cell response in the cross-compaction and
cross-session probes.

We measured response *length*; reviewers will ask whether the long responses
still answer the stressor or have derailed. This audit applies the existing
S2_NO_PREAMBLE compliance scorer (`_is_no_preamble` from
run_instruction_override_probe.py) post-hoc to every cell, then reports:

  - Compliance rate per (target, position, arm)
  - Conditional-on-compliance length ratio: compute the verbosity ratio using
    ONLY cells where both arms passed compliance. If the headline ratio
    survives this filter, the claim "compliant but more verbose" holds.
  - Examples of NON-compliant drifted responses for sanity inspection.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.run_instruction_override_probe import _is_no_preamble  # type: ignore

ROOTS = [
    ("Session 1", REPO_ROOT / "data_archive" / "cross_compaction"),
    ("Session 2", REPO_ROOT / "data_archive" / "cross_session" / "chainassemble"),
    ("Session 3", REPO_ROOT / "data_archive" / "cross_session" / "proeng"),
]
TARGETS = [
    "claude-sonnet-4-6", "claude-sonnet-4-5",
    "claude-opus-4-1",   "claude-haiku-4-5",
]


def is_s2_compliant(text: str) -> bool:
    """Same scorer used by run_instruction_override_probe.py for S2_NO_PREAMBLE."""
    return _is_no_preamble(text)


def audit_position(target_path: Path, pos_dir: Path):
    """Returns list of (v_idx, claude_len, claude_compl, filler_len, filler_compl, claude_text, filler_text)."""
    results = []
    for v in range(10):
        cp = pos_dir / f"v{v:02d}" / "claude.json"
        fp = pos_dir / f"v{v:02d}" / "filler.json"
        if not (cp.exists() and fp.exists()):
            continue
        try:
            cd = json.loads(cp.read_text())
            fd = json.loads(fp.read_text())
        except Exception:
            continue
        ct, ft = cd.get("response_text", ""), fd.get("response_text", "")
        results.append({
            "v": v,
            "claude_len": cd.get("response_len", 0),
            "claude_compl": is_s2_compliant(ct),
            "filler_len": fd.get("response_len", 0),
            "filler_compl": is_s2_compliant(ft),
            "claude_text": ct,
            "filler_text": ft,
        })
    return results


def main():
    print("=" * 70)
    print("S2_NO_PREAMBLE COMPLIANCE AUDIT")
    print("=" * 70)
    print()
    print("For every cell in the cross-compaction and cross-session probes,")
    print("apply the same compliance scorer used by the original instruction-")
    print("override probe (single line, no markdown, no leading explanation).")
    print()

    overall = {  # arm -> [n_pass, n_total]
        "claude": [0, 0],
        "filler": [0, 0],
    }
    per_target = {t: {"claude": [0, 0], "filler": [0, 0]} for t in TARGETS}
    per_session = {s[0]: {"claude": [0, 0], "filler": [0, 0]} for s in ROOTS}

    # Conditional-on-both-compliant ratio
    cond_pairs_per_session = {s[0]: [] for s in ROOTS}
    raw_pairs_per_session = {s[0]: [] for s in ROOTS}

    # Examples of non-compliant claude responses
    noncompliant_examples = []

    for sess_name, sess_root in ROOTS:
        if not sess_root.exists():
            print(f"[skip {sess_name}] missing {sess_root}")
            continue
        for target in TARGETS:
            target_path = sess_root / target
            if not target_path.exists():
                continue
            for pos_dir in sorted(target_path.iterdir()):
                if not pos_dir.is_dir():
                    continue
                rows = audit_position(target_path, pos_dir)
                for r in rows:
                    overall["claude"][1] += 1
                    overall["filler"][1] += 1
                    if r["claude_compl"]:
                        overall["claude"][0] += 1
                    if r["filler_compl"]:
                        overall["filler"][0] += 1
                    per_target[target]["claude"][1] += 1
                    per_target[target]["filler"][1] += 1
                    if r["claude_compl"]:
                        per_target[target]["claude"][0] += 1
                    if r["filler_compl"]:
                        per_target[target]["filler"][0] += 1
                    per_session[sess_name]["claude"][1] += 1
                    per_session[sess_name]["filler"][1] += 1
                    if r["claude_compl"]:
                        per_session[sess_name]["claude"][0] += 1
                    if r["filler_compl"]:
                        per_session[sess_name]["filler"][0] += 1

                    # Conditional ratio: only include if BOTH arms passed
                    if r["claude_len"] > 0 and r["filler_len"] > 0:
                        raw_pairs_per_session[sess_name].append(
                            (r["claude_len"], r["filler_len"]))
                        if r["claude_compl"] and r["filler_compl"]:
                            cond_pairs_per_session[sess_name].append(
                                (r["claude_len"], r["filler_len"]))

                    # Save a few non-compliant claude examples
                    if (not r["claude_compl"] and r["claude_len"] > 0
                            and len(noncompliant_examples) < 8):
                        noncompliant_examples.append({
                            "session": sess_name,
                            "target": target,
                            "position": pos_dir.name,
                            "v": r["v"],
                            "len": r["claude_len"],
                            "text_preview": r["claude_text"][:200],
                        })

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    def fmt(passed, total):
        if total == 0:
            return "n/a"
        return f"{passed}/{total} = {100 * passed / total:.1f}%"

    print(f"## Overall S2 compliance ({overall['claude'][1]} cells per arm)\n")
    print(f"  claude_session arm: {fmt(*overall['claude'])}")
    print(f"  filler arm:         {fmt(*overall['filler'])}")
    print(f"  Compliance gap:     {100 * (overall['filler'][0] / max(1,overall['filler'][1]) - overall['claude'][0] / max(1,overall['claude'][1])):+.1f}pp (filler − claude)")
    print()

    print("## By target (Anthropic family across all 3 sessions)\n")
    print(f"  {'target':<22} {'claude_arm':<22} {'filler_arm':<22} gap (pp)")
    for t in TARGETS:
        c = per_target[t]["claude"]
        f = per_target[t]["filler"]
        if c[1] == 0:
            continue
        cr = 100 * c[0] / c[1]
        fr = 100 * f[0] / f[1]
        print(f"  {t:<22} {fmt(*c):<22} {fmt(*f):<22} {fr - cr:+.1f}pp")
    print()

    print("## By session (all 4 targets pooled)\n")
    print(f"  {'session':<14} {'claude_arm':<22} {'filler_arm':<22} gap (pp)")
    for s_name, _ in ROOTS:
        c = per_session[s_name]["claude"]
        f = per_session[s_name]["filler"]
        if c[1] == 0:
            continue
        cr = 100 * c[0] / c[1]
        fr = 100 * f[0] / f[1]
        print(f"  {s_name:<14} {fmt(*c):<22} {fmt(*f):<22} {fr - cr:+.1f}pp")
    print()

    # ------------------------------------------------------------------
    # Conditional-on-compliance ratios
    # ------------------------------------------------------------------
    print("## Conditional-on-compliance verbosity ratio\n")
    print("Compute mean(claude_len) / mean(filler_len) using ONLY cells where")
    print("BOTH arms passed compliance. If the ratio survives this filter,")
    print("the verbosity inflation is REAL inflation, not just task derailment.\n")
    print(f"  {'session':<14} {'all pairs':<28} {'both-compliant only':<28}")
    for s_name, _ in ROOTS:
        raw = raw_pairs_per_session[s_name]
        cond = cond_pairs_per_session[s_name]
        if not raw:
            continue
        raw_ratio = sum(c for c, _ in raw) / sum(f for _, f in raw)
        cond_str = "n/a"
        if cond:
            cm = sum(c for c, _ in cond) / len(cond)
            fm = sum(f for _, f in cond) / len(cond)
            cond_ratio = cm / max(1e-9, fm)
            cond_str = f"{cond_ratio:.2f}× (n={len(cond)})"
        print(f"  {s_name:<14} {f'{raw_ratio:.2f}× (n={len(raw)})':<28} {cond_str:<28}")
    print()

    # ------------------------------------------------------------------
    # Non-compliant examples
    # ------------------------------------------------------------------
    print("## Examples of NON-compliant claude_session responses (first 8)\n")
    for ex in noncompliant_examples:
        print(f"  [{ex['session']} / {ex['target']} / {ex['position']} / v{ex['v']:02d}, len={ex['len']}]")
        print(f"    {ex['text_preview']!r}")
        print()


if __name__ == "__main__":
    main()
