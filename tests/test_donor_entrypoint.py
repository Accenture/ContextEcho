from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_DONATE = (ROOT / "scripts" / "run-donate.sh").read_text(encoding="utf-8")
LANDING = (ROOT / "docs" / "donate" / "index.html").read_text(encoding="utf-8")


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
    assert "Local wizard preview" in LANDING
    assert "Discover sessions" in LANDING
    assert "Discover Sessions" in LANDING
    assert "manual file picker fallback" in LANDING
