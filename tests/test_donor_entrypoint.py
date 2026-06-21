from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_DONATE = (ROOT / "scripts" / "run-donate.sh").read_text(encoding="utf-8")
LANDING = (ROOT / "docs" / "donate" / "index.html").read_text(encoding="utf-8")
README = (ROOT / "README.md").read_text(encoding="utf-8")
CONTRIBUTING = (ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
DONATE_URL = "https://accenture.github.io/ContextEcho/donate/"


def test_launcher_starts_local_browser_wizard_with_managed_python() -> None:
    assert "contextecho-donate" in RUN_DONATE
    assert "--python 3.14" in RUN_DONATE
    assert "--managed-python" in RUN_DONATE
    assert "sys.version_info >= (3, 8)" in RUN_DONATE
    assert "Python 3.10+ to bootstrap" not in RUN_DONATE
    assert "raw sessions stay on this machine" in RUN_DONATE
    assert "The local donation wizard did not start" in RUN_DONATE
    assert "Apple Silicon" in RUN_DONATE


def test_hosted_landing_page_points_to_local_scanner() -> None:
    assert "Donate a coding-agent session to ContextEcho" in LANDING
    assert "curl -Ls https://github.com/Accenture/ContextEcho/raw/main/scripts/run-donate.sh | bash" in LANDING
    assert "Raw session history stays local" in LANDING
    assert "Continue in Local Wizard" in LANDING
    assert 'aria-disabled="true"' in LANDING
    assert "Run the command first" in LANDING
    assert "setAttribute('aria-disabled', 'false')" in LANDING
    assert "/api/health" in LANDING
    assert "localPorts" in LANDING
    assert "Step-by-step donation guide" in LANDING
    assert "Run the command in Terminal" in LANDING
    assert "Leave this Terminal window open while donating" in LANDING
    assert "Discover and select a session" in LANDING
    assert "Redact locally and verify" in LANDING
    assert "Submit for maintainer review" in LANDING
    assert "Pending maintainer review" in LANDING
    assert "Local wizard preview" not in LANDING
    assert "Discover sessions" in LANDING
    assert "file picker" not in LANDING
    assert "do not need to know where session history files live" in LANDING


def test_public_docs_point_to_hosted_donor_page() -> None:
    assert DONATE_URL in README
    assert DONATE_URL in CONTRIBUTING
    assert "Donate a session" in README
