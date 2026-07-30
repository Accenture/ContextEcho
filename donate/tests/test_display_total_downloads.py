"""Tests for the display-only Total Downloads overlay in donate.web.

SYNTHETIC stats only — numbers below are fabricated fixtures, not real
project_stats.json snapshots. The overlay must mirror the month-bucket
accounting in scripts/update_project_stats.py without ever writing it.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from donate.web import display_total_downloads

# historical(1000) + 2026-06(100) + 2026-07(200) == 1300
TRACKED = {
    "dataset_total_downloads": 1300,
    "dataset_total_downloads_updated": "2026-07-15",
    "dataset_historical_downloads": 1000,
    "dataset_hf_downloads_last_month": 200,
    "dataset_hf_downloads_last_month_period": "2026-07",
    "dataset_hf_monthly_downloads": {"2026-06": 100, "2026-07": 200},
}

# Legacy shape without the monthly-bucket dict (pre-bucket snapshots).
TRACKED_LEGACY = {
    "dataset_total_downloads": 1300,
    "dataset_total_downloads_updated": "2026-07-15",
    "dataset_historical_downloads": 1000,
    "dataset_hf_downloads_last_month": 300,
}


def test_same_month_overlay_uses_max_of_recorded_and_live():
    # live 250 > recorded 200 for 2026-07 -> bucket becomes 250
    assert display_total_downloads(TRACKED, 250, today="2026-07-30") == 1350


def test_same_month_live_below_recorded_is_noop():
    # live 150 < recorded 200 -> tracked total unchanged
    assert display_total_downloads(TRACKED, 150, today="2026-07-30") == 1300


def test_month_rollover_adds_live_as_new_bucket():
    # month rolled over since the last maintainer run; prior month keeps 200
    assert display_total_downloads(TRACKED, 40, today="2026-08-05") == 1340


def test_month_rollover_with_legacy_bucketless_snapshot():
    # buckets inferred as {2026-07: 300}; live 50 lands in 2026-08
    assert display_total_downloads(TRACKED_LEGACY, 50, today="2026-08-05") == 1350


def test_live_fetch_failure_falls_back_to_tracked_total():
    assert display_total_downloads(TRACKED, None, today="2026-07-30") == 1300
    assert display_total_downloads(TRACKED, "not-a-number", today="2026-07-30") == 1300
    assert display_total_downloads(TRACKED, -5, today="2026-07-30") == 1300


def test_glitch_cap_ignores_live_values_over_10x_recorded_month():
    # 2001 > 10 * 200 -> treated as an API glitch, tracked total stands
    assert display_total_downloads(TRACKED, 2001, today="2026-07-30") == 1300
    # exactly 10x is still accepted
    assert display_total_downloads(TRACKED, 2000, today="2026-07-30") == 1000 + 100 + 2000


def test_glitch_cap_after_rollover_references_last_recorded_month():
    # after rollover the reference is the 2026-07 recorded bucket (200)
    assert display_total_downloads(TRACKED, 2001, today="2026-08-05") == 1300


def test_never_displays_less_than_tracked_total():
    # tracked total higher than what the buckets reconstruct -> keep tracked
    inflated = dict(TRACKED, dataset_total_downloads=5000)
    assert display_total_downloads(inflated, 250, today="2026-07-30") == 5000


def test_parity_with_update_script_roll_download_total():
    # The overlay must reproduce scripts/update_project_stats.py accounting
    # bit-for-bit for non-glitch live values (no double counting).
    import importlib.util

    script = Path(__file__).resolve().parents[2] / "scripts" / "update_project_stats.py"
    spec = importlib.util.spec_from_file_location("update_project_stats_for_test", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for tracked in (TRACKED, TRACKED_LEGACY, {}):
        for live in (0, 40, 250, 1999):
            for today in ("2026-07-30", "2026-08-05"):
                expected, _ = mod.roll_download_total(dict(tracked), live, today[:7])
                tracked_total = tracked.get("dataset_total_downloads") or 0
                assert display_total_downloads(tracked, live, today=today) == max(expected, tracked_total)


def test_empty_tracked_mirrors_update_script_default_historical():
    # update_project_stats.py seeds an empty snapshot with the default 39,000
    # historical baseline; the read-only overlay mirrors that exactly.
    assert display_total_downloads({}, 250, today="2026-07-30") == 39_000 + 250
    assert display_total_downloads(None, 250, today="2026-07-30") is None  # type: ignore[arg-type]
