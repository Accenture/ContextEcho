"""Appendix figure: panel-wide forest using the FULL 25-probe identity
battery. Robustness check referenced from §2 and §3.

Output: paper/figures/fig_app_full25probes.{png,pdf}
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Set env BEFORE importing the panelwide script
os.environ["PROBES_FULL25"] = "1"

# Now import its module-level state and override the filter
import plotting.fig2_forest_panelwide as base  # type: ignore

# Replace the 5-probe filter with all probes
base.CODING_PROBE_IDS = None  # type: ignore[assignment]

orig_load = base.load_arm_by_position


def load_all_probes(target: str, arm: str):
    out = {}
    for pos in base.POSITIONS:
        d = base.ROOT / target / pos / arm
        if not d.exists(): continue
        scores = []
        for f in sorted(d.iterdir()):
            if f.suffix != ".json": continue
            # NOTE: no probe-id filter — appendix uses all 25 probes
            try:
                data = json.loads(f.read_text())
            except Exception:
                continue
            s = data.get("score")
            if isinstance(s, int): scores.append(s)
        if scores:
            out[pos] = scores
    return out


base.load_arm_by_position = load_all_probes  # type: ignore[assignment]

# Override output paths
base.OUT_DATA = REPO_ROOT / "data_archive" / "fig_app"
base.OUT_DATA.mkdir(parents=True, exist_ok=True)


# Run the main function with patched output names
def main():
    # Monkey-patch the savefig calls by editing the function locals via a
    # wrapper. Easiest: just call main() but re-route paths after.
    rc = base.main()
    # base.main() saved to FOREST_PANELWIDE.{png,pdf} and fig2_forest_panelwide.{png,pdf}.
    # Copy / rename to appendix outputs.
    import shutil
    src_data_pdf = REPO_ROOT / "data_archive" / "fig2" / "FOREST_PANELWIDE.pdf"
    src_data_png = REPO_ROOT / "data_archive" / "fig2" / "FOREST_PANELWIDE.png"
    src_paper_pdf = REPO_ROOT / "paper" / "figures" / "fig2_forest_panelwide.pdf"
    src_paper_png = REPO_ROOT / "paper" / "figures" / "fig2_forest_panelwide.png"

    dst_app_pdf = REPO_ROOT / "paper" / "figures" / "fig_app_full25probes.pdf"
    dst_app_png = REPO_ROOT / "paper" / "figures" / "fig_app_full25probes.png"

    if src_paper_pdf.exists(): shutil.copy(src_paper_pdf, dst_app_pdf)
    if src_paper_png.exists(): shutil.copy(src_paper_png, dst_app_png)
    print(f"\nCopied to {dst_app_pdf} and {dst_app_png}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
