"""Tests for the display-only Total Downloads overlay in donate.web.

SYNTHETIC stats only — numbers below are fabricated fixtures, not real
project_stats.json snapshots. The displayed value is the tracked cumulative
total plus the growth of the live HF rolling last-month count since the
snapshot recorded alongside that total; scripts/update_project_stats.py
folds the identical delta into the stored total on refresh, so the two
accountings must stay in lockstep (parity test below).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from donate.web import display_total_downloads

TRACKED = {
    "dataset_total_downloads": 1300,
    "dataset_total_downloads_updated": "2026-07-15",
    "dataset_historical_downloads": 1000,
    "dataset_hf_downloads_last_month": 200,
    "dataset_hf_downloads_last_month_period": "2026-07",
    "dataset_hf_monthly_downloads": {"2026-06": 100, "2026-07": 200},
}


def test_live_growth_since_snapshot_adds_delta():
    # live 250 vs snapshot 200 -> +50 shown immediately
    assert display_total_downloads(TRACKED, 250) == 1350


def test_live_equal_to_snapshot_is_noop():
    assert display_total_downloads(TRACKED, 200) == 1300


def test_rolling_window_shrinking_never_subtracts():
    # old days falling out of HF's 30-day window -> total holds
    assert display_total_downloads(TRACKED, 150) == 1300


def test_live_fetch_failure_falls_back_to_tracked_total():
    assert display_total_downloads(TRACKED, None) == 1300
    assert display_total_downloads(TRACKED, "not-a-number") == 1300
    assert display_total_downloads(TRACKED, -5) == 1300


def test_glitch_cap_ignores_live_values_over_10x_snapshot():
    # 2001 > 10 * 200 -> treated as an API glitch, tracked total stands
    assert display_total_downloads(TRACKED, 2001) == 1300
    # exactly 10x is still accepted: +1800 over the snapshot
    assert display_total_downloads(TRACKED, 2000) == 1300 + 1800


def test_missing_snapshot_falls_back_to_tracked_total():
    assert display_total_downloads({"dataset_total_downloads": 1300}, 250) == 1300


def test_missing_total_returns_none():
    assert display_total_downloads({}, 250) is None
    assert display_total_downloads(None, 250) is None  # type: ignore[arg-type]


def test_parity_with_update_script_delta_accumulation():
    # A maintainer refresh at the same live value must land on exactly the
    # number donors were already seeing (monotonic across refreshes).
    import importlib.util

    script = Path(__file__).resolve().parents[2] / "scripts" / "update_project_stats.py"
    spec = importlib.util.spec_from_file_location("update_project_stats_for_test", script)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for live in (0, 150, 200, 250, 1999, 2001):
        displayed = display_total_downloads(TRACKED, live)
        refreshed = mod.update_stats(dict(TRACKED), {"downloads": live}, "2026-07-30")
        assert refreshed["dataset_total_downloads"] == displayed


def test_refresh_then_further_growth_keeps_accumulating():
    import importlib.util

    script = Path(__file__).resolve().parents[2] / "scripts" / "update_project_stats.py"
    spec = importlib.util.spec_from_file_location("update_project_stats_for_test2", script)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    refreshed = mod.update_stats(dict(TRACKED), {"downloads": 250}, "2026-07-30")
    assert refreshed["dataset_total_downloads"] == 1350
    assert refreshed["dataset_hf_downloads_last_month"] == 250
    # growth after the refresh stacks on the new baseline
    assert display_total_downloads(refreshed, 300) == 1400
