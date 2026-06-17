# ContextEcho — reproduction Makefile
#
# Run `make help` to see all targets. Each target wraps the underlying
# Python command documented in REPRODUCE.md.

.PHONY: help setup setup-donate setup-maintainer setup-relay run-relay verify-pii smoke-test \
        fig1 fig2-taxonomy fig2-forest fig4 fig5 fig6 \
        fig-app-anchor-decay fig-app-anchor-size fig-app-crossjudge \
        fig-app-crosssession fig-app-full25 fig-app-onset \
        download-donations intake-donations promote-donation reset-donation-test-state \
        maintainer-intake \
        update-contributors check-contributors update-dataset-card check-dataset-card \
        update-release-metadata check-release-metadata \
        review-donation review-donation-quick \
        fig-session-validation session-validation-dry-run session-validation-quick \
        session-validation-quick-summary session-validation-summary \
        figs-body figs-app figs-all \
        clean-pyc

PYTHON ?= python3
DONATE_VENV ?= .venv
DONATE_PYTHON ?= $(DONATE_VENV)/bin/python

help:
	@echo "ContextEcho — reproduction targets"
	@echo ""
	@echo "  Setup & verification:"
	@echo "    make setup            install Python dependencies"
	@echo "    make setup-donate     install minimal local donation wizard dependencies"
	@echo "    make setup-maintainer install donation intake + quick-validation dependencies"
	@echo "    make setup-relay      install optional server-side donation relay dependencies"
	@echo "    make run-relay        run relay; requires HF_STAGING_TOKEN"
	@echo "    make verify-pii       run PII redaction grep audit on data_archive_release/"
	@echo "    make smoke-test       run a 1-cell harness check"
	@echo "    make download-donations                  download private HF staging donations"
	@echo "    make maintainer-intake                   one-command full intake + promotion + metadata"
	@echo "    make intake-donations                    download + technical-review all pending donations"
	@echo "    make intake-donations RUN_QUICK=1        include quick validation gate"
	@echo "    make intake-donations PROMOTE=1          promote accepted donations into data_archive_release_v2/"
	@echo "    make intake-donations INCLUDE_PROMOTED=1 re-review already promoted submissions"
	@echo "    make intake-donations INCLUDE_REVIEWED=1 re-review unchanged processed submissions"
	@echo "    make intake-donations INCLUDE_DUPLICATES=1 review duplicate session artifacts"
	@echo "    make review-donation SUBMISSION=...       one-command maintainer technical review"
	@echo "    make review-donation-quick SUBMISSION=... run review + 30-cell quick validation"
	@echo "    make promote-donation SUBMISSION=...      promote one accepted donation"
	@echo "    make reset-donation-test-state            archive + clear local test intake state"
	@echo "    make update-contributors                  regenerate CONTRIBUTORS.md from accepted ledger"
	@echo "    make check-contributors                   verify CONTRIBUTORS.md is up to date"
	@echo "    make update-dataset-card                  regenerate DATASET_CARD.md from release metadata"
	@echo "    make check-dataset-card                   verify DATASET_CARD.md is up to date"
	@echo "    make update-release-metadata              regenerate CONTRIBUTORS.md + DATASET_CARD.md"
	@echo "    make check-release-metadata               verify public release metadata is up to date"
	@echo "    make session-validation-dry-run SESSION=... LABEL=...  plan a new donor validation"
	@echo "    make session-validation-quick SESSION=... LABEL=...    run 30-cell donor intake check"
	@echo "    make session-validation-quick-summary LABEL=...        summarize quick validation"
	@echo "    make session-validation-summary LABEL=...              summarize validation outputs"
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
	@echo "    make fig-session-validation  existing + v2 candidate session trajectories"
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

setup-donate:
	python3 -m venv $(DONATE_VENV)
	$(DONATE_PYTHON) -m pip install -r requirements-donate.txt

setup-maintainer:
	$(PYTHON) -m pip install -r requirements-maintainer.txt

setup-relay:
	$(PYTHON) -m pip install -r requirements-relay.txt

run-relay:
	$(PYTHON) -m uvicorn donate.relay_server:app --host $${HOST:-0.0.0.0} --port $${PORT:-8088}

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

download-donations:
	$(PYTHON) scripts/download_donations.py

maintainer-intake:
	bash scripts/run-maintainer-intake.sh

intake-donations:
	$(PYTHON) scripts/intake_donations.py \
	  $(if $(RUN_QUICK),--run-quick,) \
	  $(if $(PROMOTE),--promote,) \
	  $(if $(INCLUDE_PROMOTED),--include-promoted,) \
	  $(if $(INCLUDE_REVIEWED),--include-reviewed,) \
	  $(if $(INCLUDE_DUPLICATES),--include-duplicates,)

review-donation:
	@if [ -z "$(SUBMISSION)" ]; then \
	  echo "Usage: make review-donation SUBMISSION=hf_staging_download/pending/submission-xxxx"; \
	  exit 2; \
	fi
	$(PYTHON) scripts/review_donation.py "$(SUBMISSION)"

review-donation-quick:
	@if [ -z "$(SUBMISSION)" ]; then \
	  echo "Usage: make review-donation-quick SUBMISSION=hf_staging_download/pending/submission-xxxx"; \
	  exit 2; \
	fi
	$(PYTHON) scripts/review_donation.py "$(SUBMISSION)" --run-quick

promote-donation:
	@if [ -z "$(SUBMISSION)" ]; then \
	  echo "Usage: make promote-donation SUBMISSION=hf_staging_download/pending/submission-xxxx"; \
	  exit 2; \
	fi
	$(PYTHON) scripts/promote_donation.py "$(SUBMISSION)" \
	  $(if $(RUN_QUICK),--run-quick,)

reset-donation-test-state:
	$(PYTHON) scripts/reset_donation_test_state.py \
	  $(if $(YES),--yes,)

update-contributors:
	$(PYTHON) scripts/update_contributors.py

check-contributors:
	$(PYTHON) scripts/update_contributors.py --check

update-dataset-card:
	$(PYTHON) scripts/update_dataset_card.py

check-dataset-card:
	$(PYTHON) scripts/update_dataset_card.py --check

update-release-metadata: update-contributors update-dataset-card

check-release-metadata: check-contributors check-dataset-card

session-validation-dry-run:
	@if [ -z "$(SESSION)" ] || [ -z "$(LABEL)" ]; then \
	  echo "Usage: make session-validation-dry-run SESSION=/path/session.redacted.jsonl LABEL=donor04"; \
	  exit 2; \
	fi
	$(PYTHON) experiments/e18_session_validation/run.py --session "$(SESSION)" --label "$(LABEL)" --quick --dry-run

session-validation-quick:
	@if [ -z "$(SESSION)" ] || [ -z "$(LABEL)" ]; then \
	  echo "Usage: make session-validation-quick SESSION=/path/session.redacted.jsonl LABEL=donor04"; \
	  exit 2; \
	fi
	$(PYTHON) experiments/e18_session_validation/run.py --session "$(SESSION)" --label "$(LABEL)" --quick

session-validation-quick-summary:
	@if [ -z "$(LABEL)" ]; then \
	  echo "Usage: make session-validation-quick-summary LABEL=donor04"; \
	  exit 2; \
	fi
	$(PYTHON) analysis/analyze_session_validation.py \
	  --root results_v2_candidate/session_validation_quick/$(LABEL)/claude-sonnet-4-5 \
	  --positions 3 --probes 5

session-validation-summary:
	@if [ -z "$(LABEL)" ]; then \
	  echo "Usage: make session-validation-summary LABEL=donor04"; \
	  exit 2; \
	fi
	$(PYTHON) analysis/analyze_session_validation.py \
	  --root results_v2_candidate/session_validation/$(LABEL)/claude-sonnet-4-5

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

fig-session-validation:
	$(PYTHON) plotting/fig_session_validation.py

figs-app: fig-app-anchor-decay fig-app-anchor-size fig-app-crossjudge \
          fig-app-crosssession fig-app-full25 fig-app-onset
	@echo "[ok] all appendix figures rendered"

figs-all: figs-body figs-app

# === Hygiene ===

clean-pyc:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
	@echo "[ok] python caches removed"
