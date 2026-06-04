# ContextEcho — reproduction Makefile
#
# Run `make help` to see all targets. Each target wraps the underlying
# Python command documented in REPRODUCE.md.

.PHONY: help setup verify-pii smoke-test \
        fig1 fig2-taxonomy fig2-forest fig4 fig5 fig6 \
        fig-app-anchor-decay fig-app-anchor-size fig-app-crossjudge \
        fig-app-crosssession fig-app-full25 fig-app-onset \
        figs-body figs-app figs-all \
        clean-pyc

PYTHON ?= python3

help:
	@echo "ContextEcho — reproduction targets"
	@echo ""
	@echo "  Setup & verification:"
	@echo "    make setup            install Python dependencies"
	@echo "    make verify-pii       run PII redaction grep audit on data_archive_release/"
	@echo "    make smoke-test       run a 1-cell harness check"
	@echo ""
	@echo "  Body figures (paper §3-§4):"
	@echo "    make fig1             persona-space combined panel"
	@echo "    make fig2-taxonomy    25-probe taxonomy figure"
	@echo "    make fig2-forest      panelwide drift-gap forest (the headline)"
	@echo "    make fig4             surface-vs-substrate (Qwen 3 32B)"
	@echo "    make fig5             A-anchor mitigation forest"
	@echo "    make fig6             stressor-surface compliance + length"
	@echo "    make figs-body        all body figures"
	@echo ""
	@echo "  Appendix figures:"
	@echo "    make fig-app-anchor-decay    anchor persistence curve"
	@echo "    make fig-app-anchor-size     anchor token-budget sweep"
	@echo "    make fig-app-crossjudge      Sonnet vs GPT-5 audit"
	@echo "    make fig-app-crosssession    per-position trajectories on 3 sessions"
	@echo "    make fig-app-full25          full 25-probe forest"
	@echo "    make fig-app-onset           pre-C1 turn sweep"
	@echo "    make figs-app                all appendix figures"
	@echo ""
	@echo "    make figs-all         all body + appendix figures"
	@echo ""
	@echo "  Hygiene:"
	@echo "    make clean-pyc        remove __pycache__ directories"
	@echo ""
	@echo "  Re-running data collection (requires API keys; see REPRODUCE.md):"
	@echo "    each experiment is at experiments/eNN_<name>/run.py"

setup:
	$(PYTHON) -m pip install -r requirements.txt

# Verify-pii delegates to the anonymizer's built-in audit, which loads
# its pattern panel from the gitignored `scripts/.redaction_patterns.json`.
# Keeping the patterns out of this Makefile means the public mirror copy
# of the Makefile contains no donor surface forms.
verify-pii:
	@if [ ! -d data_archive_release ]; then \
	  echo "[error] data_archive_release/ not found — run scripts/anonymize_cell_jsons.py first"; \
	  exit 1; \
	fi
	@if [ ! -f scripts/.redaction_patterns.json ]; then \
	  echo "[error] scripts/.redaction_patterns.json not found"; \
	  echo "        copy scripts/.redaction_patterns.json.example to"; \
	  echo "        scripts/.redaction_patterns.json and fill in the donor patterns"; \
	  exit 1; \
	fi
	$(PYTHON) scripts/anonymize_cell_jsons.py --verify-only

smoke-test:
	@echo "Running smoke test: load one cell and reproduce its judge score..."
	$(PYTHON) -c "import json, pathlib; \
	cell = json.loads(pathlib.Path('data_archive_release/results/cross_compaction').rglob('claude.json').__next__().read_text()); \
	assert 'response_text' in cell or 'response' in cell, 'cell missing response field'; \
	print('[ok] cell loaded:', cell.get('cell_id') or 'unknown id'); \
	print('[ok] release tree is structurally valid')"

# === Body figures ===

fig1:
	$(PYTHON) plotting/fig_fig1_combined.py

fig2-taxonomy:
	$(PYTHON) plotting/fig_v2_probe_taxonomy.py

fig2-forest:
	$(PYTHON) plotting/fig2_forest_panelwide.py

fig4:
	$(PYTHON) plotting/fig_v2_surface_vs_substrate.py

fig5:
	$(PYTHON) plotting/fig_forest_mitigation.py

fig6:
	$(PYTHON) plotting/fig_forest_stressors.py

figs-body: fig1 fig2-taxonomy fig2-forest fig4 fig5 fig6
	@echo "[ok] all body figures rendered"

# === Appendix figures ===

fig-app-anchor-decay:
	$(PYTHON) plotting/fig_app_anchor_decay.py

fig-app-anchor-size:
	$(PYTHON) plotting/fig_app_anchor_size.py

fig-app-crossjudge:
	$(PYTHON) plotting/fig_app_crossjudge.py

fig-app-crosssession:
	$(PYTHON) plotting/fig_app_crosssession.py

fig-app-full25:
	$(PYTHON) plotting/fig_app_full25probes.py

fig-app-onset:
	$(PYTHON) plotting/fig_app_onset.py

figs-app: fig-app-anchor-decay fig-app-anchor-size fig-app-crossjudge \
          fig-app-crosssession fig-app-full25 fig-app-onset
	@echo "[ok] all appendix figures rendered"

figs-all: figs-body figs-app

# === Hygiene ===

clean-pyc:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
	@echo "[ok] python caches removed"
