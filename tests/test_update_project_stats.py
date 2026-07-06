from scripts.update_project_stats import roll_download_total, update_stats


def test_download_rollup_adds_positive_hf_increase():
    current = {
        "dataset_total_downloads": 47_350,
        "dataset_historical_downloads": 39_000,
        "dataset_hf_downloads_last_month": 8_350,
    }

    rollup = roll_download_total(current, 8_400)

    assert rollup.delta_applied == 50
    assert rollup.total == 47_400


def test_download_rollup_does_not_subtract_when_hf_window_drops():
    current = {
        "dataset_total_downloads": 47_350,
        "dataset_historical_downloads": 39_000,
        "dataset_hf_downloads_last_month": 8_350,
    }

    rollup = roll_download_total(current, 5_029)

    assert rollup.delta_applied == 0
    assert rollup.total == 47_350
    assert rollup.hf_last_month == 5_029


def test_update_stats_infers_previous_hf_snapshot_from_total():
    current = {
        "dataset_total_downloads": 47_350,
        "dataset_historical_downloads": 39_000,
    }

    updated = update_stats(current, {"downloads": 8_360, "likes": 5}, "2026-07-06")

    assert updated["dataset_total_downloads"] == 47_360
    assert updated["dataset_hf_downloads_last_month_previous"] == 8_350
    assert updated["dataset_hf_downloads_last_month_delta_applied"] == 10
    assert updated["dataset_hf_downloads_last_month"] == 8_360
